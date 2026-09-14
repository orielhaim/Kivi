//! First replicated Kivi node on `OpenRaft` 0.10.
//!
//! [`ReplicatedNode`] binds one consensus group to one process: durable
//! log ([`DurableRaftStore`]), Kivi state
//! machine ([`ReplicatedStateMachine`]),
//! shared peer mesh ([`PeerTransport`]),
//! and the `OpenRaft` instance gluing them through the router boundary.
//!
//! ## Owner thread + channel front
//!
//! 0.10's `single-threaded` mode makes the `Raft` handle `!Send`: it lives
//! on the Compio owner thread spawned here (`kivi-consensus`), alongside
//! the transport tasks and the snapshot reassembly state. The public
//! [`ReplicatedNode`] front stays `Send + Sync` — one bounded
//! [`async_channel`] request queue plus `Send` oneshot replies — so
//! callers on any runtime (including the Tokio admin plane) keep ordinary
//! `Send` futures. No Tokio exists on either side of the boundary.
//!
//! ```text
//! caller (any thread/runtime)
//!   │ propose / barrier / status / serve / seal
//!   ▼
//! bounded owner queue (256)
//!   ▼
//! owner loop (Compio reactor thread)
//!   ├── per-request tasks (never awaited inline: client_write, barriers,
//!   │   and snapshot installs can all pend indefinitely while status,
//!   │   diagnostics, and peer RPCs keep flowing)
//!   ├── Raft handle (!Send, pinned here)
//!   ├── snapshot reassembly table
//!   └── store / machine / transport clones
//! ```
//!
//! ## Leader mutation path
//!
//! ```text
//! client mutation
//!     ↓ leader dedup check (read-only; retries answer from any replica)
//!     ↓ sidecar gate (chunked roots verified durable; chunked-base
//!       range patches restaged reusing untouched chunk ids)
//!     ↓ prepare deterministic MutationIR + expected outcome
//! OpenRaft client_write (quorum durable commit; followers gate the
//! flush on sidecar durability)
//!     ↓ state-machine apply + verification (every replica)
//! dedup outcome available
//!     ↓ reply
//! ```
//!
//! The Raft log is authoritative in replicated mode: no single-node
//! mutation WAL entry is appended alongside it (no double-logging). Large
//! values replicate as tiny `ReplaceChunkedRoot` mutations plus immutable
//! bulk sidecars (see [`crate::sidecar`] and [`crate::gate`]).
//!
//! ## Supported operations
//!
//! The small/state-local semantic set replicates: inline `SET`, `DELETE`,
//! `EXISTS`/reads, conditional `SET` (materialized at prepare),
//! `COUNTER_ADD`, expiry mutations, `PERSIST`, and `SETRANGE` against
//! inline bases. Large values replicate as [`Operation::SetChunked`] and
//! friends once staged durably locally; `SETRANGE` against a chunked base
//! restages here (untouched chunks reuse via content addressing) and
//! followers fetch only missing sidecars before acknowledging.
//!
//! ## Leader-only writes
//!
//! Only the leader accepts strong mutations. Followers answer
//! Kivi-owned [`ConsensusError::NotLeader`] with a leader hint (or
//! [`ConsensusError::LeaderUnknown`]); `OpenRaft` error types never cross
//! the native protocol.
//!
//! ## Linearizable reads and follower reads
//!
//! `Latest` runs a `ReadIndex` barrier on the leader — 0.10's
//! `ensure_linearizable` waits for local applied coverage of the
//! leadership-bound [`ReadBarrier`] itself —
//! and then reads local applied state. No leader leases. `Latest` on a
//! follower is `NotLeader`; `Any` (and `BoundedStale`, with no staleness
//! bound claimed yet) reads local applied state; `AtLeast` waits for
//! local applied coverage of its token on any replica.
//!
//! ## Storage health
//!
//! Consensus persistence failure fails closed: the log store's barriers
//! already refuse to acknowledge on I/O faults, and a state machine that
//! latched unhealthy (divergence or snapshot failure) fails reads and is
//! reported through [`NodeStatus`]. Fault injection lives at the storage
//! adapter level (state-machine snapshot barrier, log-store barriers).

use std::cell::RefCell;
use std::collections::HashMap;
use std::future::Future;
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use kivi_state::{DurableOutcome, OpError, Operation, OperationResult};
use kivi_types::{
    ClusterId, CommitPosition, IdempotencyKey, MutationIdentity, NamespaceId, NodeId, ReadContract,
    TabletAuthority, TabletId, UnixMicros,
};
use openraft::storage::{RaftLogReader as _, RaftLogStorage as _, RaftStateMachine as _};
use openraft::type_config::async_runtime::watch::WatchReceiver as _;

use crate::cluster::{Bootstrap, ClusterTopology, DurableClusterView, classify_bootstrap};
use crate::config::{KiviTypeConfig, cluster_config};
use crate::gate::SidecarGate;
use crate::mutation::ReplicatedMutation;
use crate::peer::{
    PeerChunkResponse, PeerManifestResponse, PeerRequest, PeerResponse, PeerRpcError,
    PeerSnapshotRequest,
};
use crate::router::{GroupNetworkFactory, PeerRouter, decode_snapshot_meta, serve_peer_request};
use crate::sidecar::SidecarStore;
use crate::state_machine::{ProposalGate, ReplicatedStateMachine};
use crate::store::DurableRaftStore;
use crate::transport::{PeerHandler, PeerStats, PeerTransport, TransportConfig};
use crate::types::{
    ConsensusError, ConsensusGroupId, ConsensusLogIndex, ConsensusTerm, LeaderHint, ReadBarrier,
    ReplicaId, ReplicaRole,
};

/// How long `AtLeast` waits for local applied coverage before failing.
const AT_LEAST_TIMEOUT: Duration = Duration::from_secs(10);

/// Depth of the owner request queue. Every request carries its own
/// reply channel, so depth only bounds burst memory; a full queue fails
/// the submitter with `ShuttingDown`-class backpressure instead of
/// growing.
const OWNER_QUEUE: usize = 256;

/// Replicated-node construction parameters. Everything is validated
/// before serving: topology incoherence, identity conflicts, and corrupt
/// history all fail `open`, never mid-operation.
#[derive(Debug)]
pub struct NodeConfig {
    /// Data-directory root (consensus WAL lane + state-machine images
    /// live beneath it; the process lock is held for the node's life).
    pub data_dir: PathBuf,
    /// Namespace served.
    pub namespace: NamespaceId,
    /// Tablet replicated (exactly one in this stage).
    pub tablet: TabletId,
    /// Replica authority (epoch/guard fencing, stage-1 static).
    pub authority: TabletAuthority,
    /// This process's node identity (operator-declared; verified against
    /// the data directory, adopted when fresh).
    pub local: NodeId,
    /// Static cluster definition (identity, members, assignment).
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
    /// Never set in production: without verification any network peer
    /// could impersonate a member.
    pub insecure_peer_tls: bool,
}

/// Why a replicated node could not open.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum NodeOpenError {
    /// Static topology incoherent.
    #[error("topology invalid: {reason}")]
    Topology {
        /// Human-readable cause.
        reason: String,
    },
    /// Data-directory identity failure.
    #[error("data directory failed: {reason}")]
    DataDir {
        /// Human-readable cause.
        reason: String,
    },
    /// Consensus log store failed to open.
    #[error("consensus log store failed: {reason}")]
    LogStore {
        /// Human-readable cause.
        reason: String,
    },
    /// State machine failed to open.
    #[error("state machine failed: {reason}")]
    StateMachine {
        /// Human-readable cause.
        reason: String,
    },
    /// Peer mesh failed to open.
    #[error("peer mesh failed: {reason}")]
    Transport {
        /// Human-readable cause.
        reason: String,
    },
    /// Startup config conflicts with durable identity/membership.
    #[error("bootstrap refused: {reason}")]
    Bootstrap {
        /// Human-readable cause.
        reason: String,
    },
    /// `OpenRaft` membership installation failed on first formation.
    #[error("group initialization failed: {reason}")]
    Initialize {
        /// Human-readable cause.
        reason: String,
    },
    /// Immutable sidecar store failed to open.
    #[error("sidecar store failed: {reason}")]
    Sidecar {
        /// Human-readable cause.
        reason: String,
    },
    /// Peer TLS identity failed to load or generate.
    #[error("peer TLS failed: {reason}")]
    Tls {
        /// Human-readable cause.
        reason: String,
    },
    /// The owner thread failed to start or died during startup.
    #[error("consensus owner failed: {reason}")]
    Owner {
        /// Human-readable cause.
        reason: String,
    },
}

/// Outcome of one proposed mutation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProposeOutcome {
    /// Quorum-committed, leader-applied, dedup-installed.
    Applied {
        /// Committed log index.
        index: ConsensusLogIndex,
        /// Deterministic outcome.
        outcome: OperationResult,
    },
    /// Same-identity retry answered from retained state (no proposal).
    Duplicate {
        /// Original outcome.
        outcome: OperationResult,
    },
    /// Read-only operation answered locally (no proposal).
    Read {
        /// Read outcome.
        outcome: OperationResult,
    },
    /// Terminally rejected without mutating (deterministic rejections
    /// never replicate; a retry re-evaluates against converged state).
    Rejected {
        /// Terminal outcome.
        outcome: DurableOutcome,
    },
}

/// Why a proposal failed. Leader routing failures are Kivi-owned
/// ([`ConsensusError`]); validation and capability failures are explicit.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum ProposeError {
    /// Consensus routing failure (not leader / unknown / unavailable).
    #[error("{0}")]
    Consensus(#[from] ConsensusError),
    /// A retried identity fell below the session floor.
    #[error("mutation identity expired below the session floor")]
    Expired,
    /// The session holds too many unacknowledged outcomes.
    #[error("session outcome window exhausted")]
    Overloaded,
    /// The operation needs unreplicated chunk sidecars (stage-1
    /// capability boundary).
    #[error("unsupported in replicated mode: {reason}")]
    Unsupported {
        /// What is missing.
        reason: String,
    },
    /// Read-only opcode validation failure.
    #[error("operation rejected: {0}")]
    Op(#[from] OpError),
}

/// Why a strong read failed.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum ReadError {
    /// Consensus routing failure (not leader / unknown / unavailable).
    #[error("{0}")]
    Consensus(#[from] ConsensusError),
    /// Read validation failure.
    #[error("read rejected: {0}")]
    Op(#[from] OpError),
    /// The local-read path was asked to serve a mutating operation.
    #[error("not a read operation")]
    NotARead,
    /// Applied coverage never arrived before the deadline.
    #[error("applied coverage timed out")]
    CoverageTimeout,
}

/// Per-replicated-tablet diagnostics for the admin plane.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NodeStatus {
    /// Group served.
    pub group: ConsensusGroupId,
    /// This replica.
    pub replica: ReplicaId,
    /// Observed role.
    pub role: ReplicaRole,
    /// Observed term.
    pub term: ConsensusTerm,
    /// Best-known leader.
    pub leader: Option<ReplicaId>,
    /// Last log entry.
    pub last_log: ConsensusLogIndex,
    /// Last committed entry.
    pub committed: ConsensusLogIndex,
    /// Last applied entry.
    pub applied: ConsensusLogIndex,
    /// Applied commit position through the index mapping.
    pub applied_commit: CommitPosition,
    /// Installed snapshot base.
    pub snapshot: Option<ConsensusLogIndex>,
    /// Last purged log index (`None` before the first purge).
    pub purged: Option<ConsensusLogIndex>,
    /// Whether the replica serves reads and applies.
    pub healthy: bool,
    /// Latched failure detail.
    pub detail: Option<String>,
    /// Per-peer connection diagnostics.
    pub peers: HashMap<NodeId, PeerStats>,
    /// Immutable sidecar replication diagnostics.
    pub sidecar: crate::sidecar::SidecarMetricsSnapshot,
}

type Reply<T> = futures::channel::oneshot::Sender<T>;

/// One unit of work for the owner thread. Every variant carries its reply
/// channel; dropping the reply (owner shutdown race) surfaces as
/// `ShuttingDown` on the caller, never a hang.
enum OwnerRequest {
    /// Quorum-commit one deterministic command.
    Propose {
        /// Deterministic command plus expectation.
        command: ReplicatedMutation,
        /// Mapped proposal outcome.
        reply: Reply<Result<ProposeOutcome, ProposeError>>,
    },
    /// Establish a `ReadIndex` linearizable barrier.
    Barrier {
        /// Leadership-bound barrier or routing failure.
        reply: Reply<Result<ReadBarrier, ConsensusError>>,
    },
    /// Wait for local applied coverage of `index` (bounded).
    WaitApplied {
        /// Index to cover.
        index: u64,
        /// Covered (`Ok`) or timed out (`Err(())`).
        reply: Reply<Result<(), ()>>,
    },
    /// Gather full diagnostics.
    Status {
        /// Diagnostics including per-peer stats (the owner holds the
        /// transport handle, so no second hop is needed).
        reply: Reply<NodeStatus>,
    },
    /// Serve one incoming peer RPC for `group`.
    Serve {
        /// Sending node.
        from: NodeId,
        /// Addressed group.
        group: ConsensusGroupId,
        /// Request body.
        request: PeerRequest,
        /// Family-matched answer or refusal.
        reply: Reply<Result<PeerResponse, PeerRpcError>>,
    },
    /// Trigger a snapshot seal and purge the log through its base.
    SnapshotPurge {
        /// Sealed base or human-readable reason.
        reply: Reply<Result<ConsensusLogIndex, String>>,
    },
    /// Suspend one peer link (partition test hook, drain tooling).
    SuspendPeer {
        /// Suspended peer.
        peer: NodeId,
        /// Completion (no payload; the suspend itself is synchronous on
        /// the owner).
        reply: Reply<()>,
    },
    /// Resume a suspended peer link.
    ResumePeer {
        /// Resumed peer.
        peer: NodeId,
        /// Completion.
        reply: Reply<()>,
    },
    /// Stop Raft and the mesh, then exit the owner loop.
    Shutdown,
}

/// One replicated Kivi node: a `Send + Sync` front over the Compio owner
/// thread driving a single tablet group. The `Raft` handle itself never
/// leaves that thread; every method below is an ordinary `Send` future on
/// any runtime.
pub struct ReplicatedNode {
    machine: ReplicatedStateMachine,
    /// Durable log handle for test/diagnostic introspection (production
    /// reads go through the owner; this never touches the lane).
    #[allow(dead_code)]
    store: DurableRaftStore,
    sidecar: SidecarStore,
    owner_tx: async_channel::Sender<OwnerRequest>,
    owner_thread: Mutex<Option<std::thread::JoinHandle<()>>>,
    peer_addr: std::net::SocketAddr,
    namespace: NamespaceId,
    tablet: TabletId,
    authority: TabletAuthority,
    local: NodeId,
    cluster: ClusterId,
    incarnation: kivi_types::NodeIncarnation,
    group: ConsensusGroupId,
    data_dir: PathBuf,
    _dir_guard: kivi_durability::OpenDir,
}

impl std::fmt::Debug for ReplicatedNode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ReplicatedNode")
            .field("node", &self.local)
            .field("cluster", &self.cluster)
            .field("tablet", &self.tablet)
            .field("group", &self.group)
            .field("peer", &self.peer_addr)
            .finish_non_exhaustive()
    }
}

impl ReplicatedNode {
    /// Returns the bound peer endpoint.
    #[must_use]
    pub const fn peer_addr(&self) -> std::net::SocketAddr {
        self.peer_addr
    }
}

/// Everything recovery establishes before the owner thread starts:
/// validated topology, directory identity, durable log and state,
/// immutable sidecar store, peer TLS identity, and the dial/membership
/// maps.
struct RecoveredParts {
    opened: kivi_durability::OpenDir,
    store: DurableRaftStore,
    machine: ReplicatedStateMachine,
    sidecar: SidecarStore,
    tls: crate::transport::TlsMaterial,
    voters: std::collections::BTreeSet<u64>,
    peer_addrs: std::collections::BTreeMap<u64, String>,
    peers: HashMap<NodeId, std::net::SocketAddr>,
    group: ConsensusGroupId,
}

/// Parameters crossing into the owner thread at spawn. Everything is
/// owned (`Send`); the `!Send` Raft handle is built inside.
struct OwnerParams {
    store: DurableRaftStore,
    machine: ReplicatedStateMachine,
    sidecar: SidecarStore,
    tls: crate::transport::TlsMaterial,
    topology: ClusterTopology,
    transport_config: TransportConfig,
    peer_identity: crate::peer::PeerIdentity,
    peers: HashMap<NodeId, std::net::SocketAddr>,
    voters: std::collections::BTreeSet<u64>,
    peer_addrs: std::collections::BTreeMap<u64, String>,
    local: NodeId,
    group: ConsensusGroupId,
    tablet: TabletId,
    namespace: NamespaceId,
    serve_tx: async_channel::Sender<OwnerRequest>,
    requests: async_channel::Receiver<OwnerRequest>,
    ready: futures::channel::oneshot::Sender<Result<std::net::SocketAddr, NodeOpenError>>,
}

/// Validates topology, opens the data directory (adopting operator
/// identity when fresh, advancing the incarnation on restart), and
/// recovers the log and state machine. Synchronous: nothing here needs
/// a runtime; the owner thread (mesh and `Raft`) starts afterwards.
///
/// # Errors
///
/// Returns [`NodeOpenError`] on topology, identity, or storage failures.
#[allow(clippy::too_many_lines)]
fn open_recovered_parts(config: &NodeConfig) -> Result<RecoveredParts, NodeOpenError> {
    use NodeOpenError as Fault;
    config
        .topology
        .validate()
        .map_err(|error| Fault::Topology {
            reason: error.to_string(),
        })?;
    let assignment = config
        .topology
        .assignment(config.tablet)
        .ok_or_else(|| Fault::Topology {
            reason: format!("no tablet assignment for tablet {}", config.tablet),
        })?;
    if !assignment.replicas.contains(&config.local) {
        return Err(Fault::Topology {
            reason: format!("node {} replicates no tablet", config.local),
        });
    }
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
    let lane_identity = kivi_durability::LaneIdentity {
        cluster: opened.meta.cluster,
        node: opened.meta.node,
        incarnation: opened.meta.incarnation,
    };
    let group = ConsensusGroupId::of_tablet(config.tablet);
    let store = DurableRaftStore::open(
        &config.data_dir,
        config.namespace,
        group,
        lane_identity,
        config.segment_target_bytes,
    )
    .map_err(|error| Fault::LogStore {
        reason: error.to_string(),
    })?;
    let machine = ReplicatedStateMachine::open(
        &config.data_dir,
        config.namespace,
        config.tablet,
        config.authority,
    )
    .map_err(|error| Fault::StateMachine {
        reason: error.to_string(),
    })?;
    // Immutable sidecar store: content-addressed chunks + manifests under
    // `<data_dir>/sidecar`, one lane, namespace-derived domain. Recovery
    // replays packs; startup verification (applied roots resolvable)
    // happens after the owner starts serving.
    let domain = kivi_types::SecurityDomainId::from_u64(config.namespace.as_u64());
    let (sidecar, _sidecar_thread) =
        SidecarStore::open(&config.data_dir, domain, 256 * 1024 * 1024).map_err(|error| {
            Fault::Sidecar {
                reason: error.to_string(),
            }
        })?;
    let (voters, peer_addrs) = config
        .topology
        .membership_for(config.tablet)
        .map_err(|error| Fault::Topology {
            reason: error.to_string(),
        })?;
    let peers: HashMap<NodeId, std::net::SocketAddr> = config
        .topology
        .nodes
        .iter()
        .map(|descriptor| (descriptor.node, descriptor.peer))
        .collect();
    // Peer TLS identity: this node's certificate (generated on first
    // open) plus the pinned peer certificates from static config.
    let peer_tls_cert = crate::tls::NodeCert::load_or_generate(&config.data_dir, opened.meta.node)
        .map_err(|error| Fault::Tls {
            reason: error.to_string(),
        })?;
    let tls = crate::transport::TlsMaterial {
        cert: peer_tls_cert,
        peer_certs: config.peer_certs.clone(),
        insecure_skip_verify: config.insecure_peer_tls,
    };
    Ok(RecoveredParts {
        opened,
        store,
        machine,
        sidecar,
        tls,
        voters,
        peer_addrs,
        peers,
        group,
    })
}

impl ReplicatedNode {
    /// Opens the node: validates topology, opens the data directory
    /// (adopting operator identity when fresh, advancing the
    /// incarnation), recovers log and state, classifies bootstrap,
    /// verifies-or-installs membership, and starts the peer mesh on the
    /// Compio owner thread. Listeners start only after recovery and
    /// bootstrap complete — a corrupt database never serves.
    ///
    /// # Errors
    ///
    /// Returns [`NodeOpenError`] on topology, identity, storage, mesh, or
    /// bootstrap failures. Restarting with conflicting durable
    /// identity/membership fails loudly; a fresh directory forms the
    /// group exactly once.
    pub async fn open(config: NodeConfig) -> Result<Self, NodeOpenError> {
        use NodeOpenError as Fault;
        let RecoveredParts {
            opened,
            store,
            machine,
            sidecar,
            tls,
            voters,
            peer_addrs,
            peers,
            group,
        } = open_recovered_parts(&config)?;
        // Paused mesh: no H3 request serves before every member of a
        // fresh formation installed membership (static-bootstrap race
        // discipline: a premature granted vote would refuse our own
        // `initialize` as `NotAllowed`).
        let mut transport_config = config.transport.clone();
        transport_config.start_paused = true;
        let peer_identity = crate::peer::PeerIdentity {
            cluster: opened.meta.cluster,
            node: opened.meta.node,
            incarnation: opened.meta.incarnation,
        };
        let (owner_tx, owner_rx) = async_channel::bounded::<OwnerRequest>(OWNER_QUEUE);
        let (ready_tx, ready_rx) =
            futures::channel::oneshot::channel::<Result<std::net::SocketAddr, NodeOpenError>>();
        let router_tx = owner_tx.clone();
        let owner_params = OwnerParams {
            store: store.clone(),
            machine: machine.clone(),
            sidecar: sidecar.clone(),
            tls,
            topology: config.topology.clone(),
            transport_config,
            peer_identity,
            peers,
            voters,
            peer_addrs,
            local: opened.meta.node,
            group,
            tablet: config.tablet,
            namespace: config.namespace,
            serve_tx: router_tx,
            requests: owner_rx,
            ready: ready_tx,
        };
        let thread = spawn_owner_thread(owner_params)?;
        let peer_addr = ready_rx.await.map_err(|_| Fault::Owner {
            reason: "consensus owner died during startup".to_owned(),
        })??;
        // Startup verification: every applied chunked root must resolve
        // locally before serving reads. Missing sidecars poison the replica
        // (reads fail closed, readiness reflects unhealthy) rather than
        // serving incomplete state; repair arrives via sidecar fetch or a
        // snapshot install carrying the missing bulk (which heals health).
        // Never blocks startup waiting for peers.
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
        Ok(Self {
            machine,
            store: store.clone(),
            sidecar: sidecar.clone(),
            owner_tx,
            owner_thread: Mutex::new(Some(thread)),
            peer_addr,
            namespace: config.namespace,
            tablet: config.tablet,
            authority: config.authority,
            local: opened.meta.node,
            cluster: opened.meta.cluster,
            incarnation: opened.meta.incarnation,
            group,
            data_dir: config.data_dir,
            _dir_guard: opened,
        })
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

    /// Ensures proposal sidecars (see [`ensure_proposal_sidecars_for`]).
    async fn ensure_proposal_sidecars(
        &self,
        op: &Operation,
        now: UnixMicros,
    ) -> Result<Option<Operation>, ProposeError> {
        ensure_proposal_sidecars_for(&self.sidecar, &self.machine, op, now).await
    }

    /// Sends one owner request and awaits its reply. A closed queue or a
    /// dropped reply means the owner is gone: `ShuttingDown`, never a
    /// hang.
    async fn owner_call<T>(
        &self,
        build: impl FnOnce(Reply<T>) -> OwnerRequest,
    ) -> Result<T, ConsensusError> {
        let (reply, rx) = futures::channel::oneshot::channel();
        self.owner_tx
            .send(build(reply))
            .await
            .map_err(|_| ConsensusError::ShuttingDown)?;
        rx.await.map_err(|_| ConsensusError::ShuttingDown)
    }

    /// Proposes one client mutation through the replicated path:
    /// capability gate, leader dedup check, deterministic prepare,
    /// quorum commit, verified apply. Follows Kivi-owned routing errors;
    /// `OpenRaft` errors never escape.
    ///
    /// The gate/dedup/prepare stages run on the caller (shared machine
    /// state); only the quorum commit hops to the owner thread.
    ///
    /// # Errors
    ///
    /// Returns [`ProposeError`] for routing, dedup, capability, or
    /// validation failures.
    pub async fn propose(
        &self,
        op: &Operation,
        identity: Option<MutationIdentity>,
        idempotency: Option<IdempotencyKey>,
        now: UnixMicros,
    ) -> Result<ProposeOutcome, ProposeError> {
        use kivi_state::StorePrepared;
        // Leader dedup check first: retries answer from any replica's
        // retained state without proposing (lost-response failover). This
        // precedes sidecar work so a retry never stages a second root.
        if let Some(marker) = identity {
            match self
                .machine
                .check_proposal(&marker)
                .await
                .map_err(|error| {
                    ProposeError::Consensus(ConsensusError::Unavailable {
                        reason: error.to_string(),
                    })
                })? {
                ProposalGate::Admit => {}
                ProposalGate::Hit { outcome } => {
                    return Ok(ProposeOutcome::Duplicate { outcome });
                }
                ProposalGate::Expired => return Err(ProposeError::Expired),
                ProposalGate::Overloaded => return Err(ProposeError::Overloaded),
            }
        }
        // Sidecar gate: chunked roots must already be durable locally
        // (cluster server stages before proposing); chunked-base range
        // patches restage here reusing untouched chunk ids via content
        // addressing. Returns a restaged operation when the input cannot
        // prepare directly.
        let restaged = self.ensure_proposal_sidecars(op, now).await?;
        let effective = restaged.as_ref().unwrap_or(op);
        // Deterministic prepare against current state (read-only).
        let prepared = self.machine.prepare_operation(effective, now).await?;
        let (mutation, expected) = match prepared {
            StorePrepared::Read(outcome) => return Ok(ProposeOutcome::Read { outcome }),
            StorePrepared::Terminal(outcome) => {
                return Ok(ProposeOutcome::Rejected { outcome });
            }
            StorePrepared::Write { mutation, expected } => (mutation, expected),
        };
        // No separate leader pre-check (`is_leader` is deprecated in
        // favor of the barrier): `client_write` fails fast with
        // `ForwardToLeader` on followers, which maps to `NotLeader` with
        // a hint below. A stale pre-check would only add a
        // lost-leadership race without removing this mapping.
        // Identity-less requests envelop the reserved anonymous identity
        // (no retry contract): the state machine applies them fresh
        // without dedup install, so unrelated anonymous writes never
        // alias. Client adapters must never issue session zero.
        let client = identity.map_or(
            kivi_types::RequestIdentity::new(
                kivi_types::SessionId::from_u128(crate::state_machine::ANONYMOUS_SESSION),
                kivi_types::RequestSeq::from_u64(0),
            ),
            |marker| marker.client,
        );
        let envelope = kivi_state::MutationEnvelope::new(
            self.namespace,
            self.authority,
            client,
            idempotency,
            mutation,
        );
        let command = ReplicatedMutation::new(
            envelope,
            expected,
            now,
            identity.map_or(kivi_types::RequestSeq::from_u64(0), |marker| {
                marker.ack_floor
            }),
        );
        self.owner_call(|reply| OwnerRequest::Propose { command, reply })
            .await?
    }

    /// Serves one read under its contract.
    ///
    /// # Errors
    ///
    /// Returns [`ReadError`] for routing, validation, or coverage
    /// failures.
    pub async fn read(
        &self,
        op: &Operation,
        contract: ReadContract,
        now: UnixMicros,
    ) -> Result<OperationResult, ReadError> {
        match contract {
            ReadContract::Latest => {
                // Leader-only strong read: the `ReadIndex` barrier proves
                // leadership with a quorum AND waits for local applied
                // coverage of its boundary (0.10 `ensure_linearizable`
                // awaits readiness itself), and fails with
                // `ForwardToLeader` on followers.
                self.owner_call(|reply| OwnerRequest::Barrier { reply })
                    .await
                    .map_err(ReadError::Consensus)?
                    .map_err(ReadError::Consensus)?;
                Ok(self.read_applied(op, now).await?)
            }
            ReadContract::Any | ReadContract::BoundedStale { .. } => {
                // Weak read over local applied state; no staleness bound
                // is claimed in this stage.
                Ok(self.read_applied(op, now).await?)
            }
            ReadContract::AtLeast(token) => {
                // Any replica whose applied coverage reaches the token
                // serves it (committed by definition of applied). The
                // wait runs on the owner thread (reactor sleep); this
                // future stays runtime-agnostic.
                let index =
                    crate::state_machine::index_of_commit(token.position()).ok_or_else(|| {
                        ReadError::Consensus(ConsensusError::Unavailable {
                            reason: "at-least token names no log slot".to_owned(),
                        })
                    })?;
                self.owner_call(|reply| OwnerRequest::WaitApplied { index, reply })
                    .await
                    .map_err(ReadError::Consensus)?
                    .map_err(|()| ReadError::CoverageTimeout)?;
                Ok(self.read_applied(op, now).await?)
            }
        }
    }

    /// Reads local applied state, mapping machine faults honestly.
    async fn read_applied(
        &self,
        op: &Operation,
        now: UnixMicros,
    ) -> Result<OperationResult, ReadError> {
        self.machine
            .read_local(op, now)
            .await
            .map_err(|error| match error {
                crate::state_machine::StateMachineFault::NotARead => ReadError::NotARead,
                other => ReadError::Consensus(ConsensusError::Unavailable {
                    reason: other.to_string(),
                }),
            })
    }

    /// Reads full diagnostics: group, role, term, leader, log pointers,
    /// snapshot base, health, and per-peer stats.
    pub async fn status(&self) -> NodeStatus {
        // The owner is gone only after shutdown; a dead owner reports a
        // closed status rather than hanging the admin plane.
        self.owner_call(|reply| OwnerRequest::Status { reply })
            .await
            .unwrap_or_else(|_| NodeStatus {
                group: self.group,
                replica: ReplicaId::of_node(self.local),
                role: ReplicaRole::Follower,
                term: ConsensusTerm::INITIAL,
                leader: None,
                last_log: ConsensusLogIndex::NONE,
                committed: ConsensusLogIndex::NONE,
                applied: ConsensusLogIndex::NONE,
                applied_commit: CommitPosition::UNASSIGNED,
                snapshot: None,
                purged: None,
                healthy: false,
                detail: Some("consensus owner shut down".to_owned()),
                peers: HashMap::new(),
                sidecar: self.sidecar.metrics().snapshot(),
            })
    }

    /// Seals a snapshot and purges the log through its base through
    /// `OpenRaft`'s own trigger path: `RaftCore` builds the image via
    /// the state machine (durable before streaming) and purges through
    /// the store, recording the logical purge and updating its pointers.
    /// A restart replays only the retained tail. Purge never passes the
    /// installed snapshot base, so history stays recoverable; an
    /// unhealthy machine blocks the snapshot (and hence the purge).
    ///
    /// # Errors
    ///
    /// Returns a human-readable reason when unhealthy, when no entries
    /// are applied yet, or when the snapshot/purge never completes
    /// before the deadline.
    pub async fn snapshot_and_purge(&self) -> Result<ConsensusLogIndex, String> {
        let (reply, rx) = futures::channel::oneshot::channel();
        self.owner_tx
            .send(OwnerRequest::SnapshotPurge { reply })
            .await
            .map_err(|_| "consensus owner shut down".to_owned())?;
        rx.await
            .map_err(|_| "consensus owner shut down".to_owned())?
    }

    /// Suspends one peer link (partition test hook, drain tooling).
    pub async fn suspend_peer(&self, peer: NodeId) {
        let (reply, rx) = futures::channel::oneshot::channel();
        if self
            .owner_tx
            .send(OwnerRequest::SuspendPeer { peer, reply })
            .await
            .is_ok()
        {
            let _ = rx.await;
        }
    }

    /// Resumes a suspended peer link.
    pub async fn resume_peer(&self, peer: NodeId) {
        let (reply, rx) = futures::channel::oneshot::channel();
        if self
            .owner_tx
            .send(OwnerRequest::ResumePeer { peer, reply })
            .await
            .is_ok()
        {
            let _ = rx.await;
        }
    }

    /// Stops the Raft instance, the peer mesh, and the owner thread.
    ///
    /// # Panics
    ///
    /// Panics when the owner-handle lock is poisoned (a previous holder
    /// panicked while taking the handle — a lifecycle bug, never a
    /// runtime condition).
    pub async fn shutdown(&self) {
        let _ = self.owner_tx.send(OwnerRequest::Shutdown).await;
        // The owner exits promptly (Raft + mesh shutdown are local);
        // joining here reclaims the thread before the caller proceeds.
        // Shutdown is rare and never latency-sensitive, so blocking the
        // caller beats leaking the thread.
        if let Some(thread) = self.owner_thread.lock().expect("owner handle holds").take() {
            let _ = thread.join();
        }
    }
}

/// Spawns the Compio owner thread driving one node's consensus work.
/// Returns the thread handle; readiness (bound peer address or startup
/// failure) arrives separately over the params' `ready` channel so a
/// reactor-construction failure reports instead of hanging the opener.
///
/// # Errors
///
/// Returns [`NodeOpenError::Owner`] when the OS refuses the thread.
fn spawn_owner_thread(
    owner_params: OwnerParams,
) -> Result<std::thread::JoinHandle<()>, NodeOpenError> {
    std::thread::Builder::new()
        .name("kivi-consensus".to_owned())
        .spawn(move || {
            // `ready` travels separately so a reactor-construction
            // failure still reaches the opener instead of hanging it.
            let OwnerParams {
                store,
                machine,
                sidecar,
                tls,
                topology,
                transport_config,
                peer_identity,
                peers,
                voters,
                peer_addrs,
                local,
                group,
                tablet,
                namespace,
                serve_tx,
                requests,
                ready,
            } = owner_params;
            let runtime = match compio::runtime::Runtime::new() {
                Ok(runtime) => runtime,
                Err(error) => {
                    let _ = ready.send(Err(NodeOpenError::Owner {
                        reason: format!("compio reactor failed to start: {error}"),
                    }));
                    return;
                }
            };
            runtime.block_on(owner_main(OwnerMain {
                store,
                machine,
                sidecar,
                tls,
                topology,
                transport_config,
                peer_identity,
                peers,
                voters,
                peer_addrs,
                local,
                group,
                tablet,
                namespace,
                serve_tx,
                requests,
                ready,
            }));
        })
        .map_err(|error| NodeOpenError::Owner {
            reason: format!("consensus owner thread failed to spawn: {error}"),
        })
}

/// Converts a raw voter id into its replica identity.
fn replica_of(id: u64) -> ReplicaId {
    ReplicaId::of_node(NodeId::from_u64(id))
}

type OwnerRaft = openraft::Raft<KiviTypeConfig, ReplicatedStateMachine>;

/// Owner thread main: opens the mesh, spawns Raft, bootstraps
/// membership, starts the mesh, signals readiness, then serves the
/// request loop until `Shutdown`. Runs inside the node's Compio reactor;
/// every `.await` here may drive `!Send` Raft futures.
#[allow(clippy::too_many_lines)]
async fn owner_main(params: OwnerMain) {
    let OwnerMain {
        mut store,
        machine,
        sidecar,
        tls,
        topology,
        transport_config,
        peer_identity,
        peers,
        voters,
        peer_addrs,
        local,
        group,
        tablet,
        namespace,
        serve_tx,
        requests,
        ready,
    } = params;
    // Mesh: one UDP endpoint bound to the topology's peer address; QUIC
    // handshake plus H3 streams replace the old TCP listener/accept pump.
    // A bind conflict fails loudly so the harness retries formation.
    let bulk_timeout = transport_config.bulk_timeout;
    let started = open_mesh(
        transport_config,
        peer_identity,
        peer_identity.cluster,
        &peers,
        tls,
        serve_tx,
    )
    .await;
    let (transport, peer_addr) = match started {
        Ok(started) => started,
        Err(error) => {
            let _ = ready.send(Err(error));
            return;
        }
    };
    // Sidecar durability gate: other replicas are candidate sources
    // (leader-first at fetch time via the entry's leader hint). The gate
    // is installed on the log store before Raft spawns so no append can
    // report durable persistence without its sidecars. A clone serves
    // snapshot installs on the owner thread (same deduplicating fetch).
    let gate = {
        let domain = kivi_types::SecurityDomainId::from_u64(namespace.as_u64());
        let mut sources: Vec<NodeId> = peers
            .keys()
            .copied()
            .filter(|peer| *peer != local)
            .collect();
        sources.sort_by_key(|peer| peer.as_u64());
        let gate = SidecarGate::new(
            sidecar.clone(),
            transport.clone(),
            sources,
            group,
            domain,
            bulk_timeout,
        );
        store.set_gate(gate.clone());
        gate
    };
    let router = PeerRouter::new(transport.clone());
    let raft = match spawn_raft(local, group, router, &store, &machine).await {
        Ok(raft) => raft,
        Err(error) => {
            let _ = ready.send(Err(error));
            return;
        }
    };
    if let Err(error) = bootstrap_group(
        &BootstrapInputs {
            topology: &topology,
            tablet,
            local,
            cluster: peer_identity.cluster,
            node: local,
            voters: &voters,
            peer_addrs: &peer_addrs,
        },
        &mut store,
        &machine,
        &raft,
    )
    .await
    {
        let _ = ready.send(Err(error));
        return;
    }
    transport.start();
    if ready.send(Ok(peer_addr)).is_err() {
        // Opener gone: shut down cleanly rather than serving nobody.
        let _ = raft.shutdown().await;
        transport.shutdown().await;
        return;
    }
    owner_serve_loop(
        OwnerCtx {
            raft: raft.clone(),
            store,
            machine,
            sidecar,
            gate,
            transport: transport.clone(),
            group,
            namespace,
            tablet,
            transfers: Rc::new(RefCell::new(HashMap::new())),
        },
        raft,
        transport,
        requests,
    )
    .await;
}

/// Spawns the Raft instance on the shared router (the fallible Raft
/// half of [`owner_main`], extracted so the main flow stays readable).
///
/// # Errors
///
/// Returns [`NodeOpenError`] when the Raft config is incoherent or the
/// spawn fails.
async fn spawn_raft(
    local: NodeId,
    group: ConsensusGroupId,
    router: PeerRouter,
    store: &DurableRaftStore,
    machine: &ReplicatedStateMachine,
) -> Result<OwnerRaft, NodeOpenError> {
    use NodeOpenError as Fault;
    let raft_config = cluster_config().map_err(|error| Fault::Topology {
        reason: format!("invalid raft config: {error}"),
    })?;
    OwnerRaft::new(
        local.as_u64(),
        raft_config,
        GroupNetworkFactory::new(router, group),
        store.clone(),
        machine.clone(),
    )
    .await
    .map_err(|error| Fault::Transport {
        reason: format!("raft spawn failed: {error}"),
    })
}

/// Opens the QUIC/H3 peer mesh (the fallible mesh half of
/// [`owner_main`], extracted so the main flow stays readable). Binds one
/// UDP endpoint and spawns dial/serve tasks; callers must already run on
/// a Compio reactor.
///
/// # Errors
///
/// Returns [`NodeOpenError`] when the mesh refuses to open.
async fn open_mesh(
    transport_config: TransportConfig,
    peer_identity: crate::peer::PeerIdentity,
    cluster: ClusterId,
    peers: &HashMap<NodeId, std::net::SocketAddr>,
    tls: crate::transport::TlsMaterial,
    serve_tx: async_channel::Sender<OwnerRequest>,
) -> Result<(PeerTransport, std::net::SocketAddr), NodeOpenError> {
    use NodeOpenError as Fault;
    let handler: Arc<dyn PeerHandler> = Arc::new(ServeRouter { tx: serve_tx });
    PeerTransport::open(
        transport_config,
        peer_identity,
        cluster,
        peers,
        tls,
        handler,
    )
    .await
    .map_err(|error| Fault::Transport {
        reason: error.to_string(),
    })
}

/// Serves the owner request loop until `Shutdown`, then stops Raft and
/// the mesh. The loop never awaits Raft work inline: every request
/// spawns its own task (`client_write`, barriers, snapshot installs can
/// all pend indefinitely, and the loop must keep serving status,
/// diagnostics, and peer RPCs meanwhile — awaiting inline deadlocks
/// any client that overlaps a pending proposal with another request).
/// All tasks stay on this reactor (`!Send` handles never leave it).
async fn owner_serve_loop(
    ctx: OwnerCtx,
    raft: OwnerRaft,
    transport: PeerTransport,
    requests: async_channel::Receiver<OwnerRequest>,
) {
    while let Ok(request) = requests.recv().await {
        match request {
            OwnerRequest::Shutdown => break,
            other => {
                let ctx = ctx.clone();
                // Detach: compio cancels on handle drop, and the loop
                // must not serialize on long-lived Raft work.
                compio::runtime::spawn(async move {
                    ctx.serve_one(other).await;
                })
                .detach();
            }
        }
    }
    let _ = raft.shutdown().await;
    transport.shutdown().await;
}

/// Owned state on the owner thread: the `!Send` Raft handle, its
/// storage, sidecar store, peer TLS material, shared mesh, and snapshot
/// reassembly state.
struct OwnerMain {
    store: DurableRaftStore,
    machine: ReplicatedStateMachine,
    sidecar: SidecarStore,
    tls: crate::transport::TlsMaterial,
    topology: ClusterTopology,
    transport_config: TransportConfig,
    peer_identity: crate::peer::PeerIdentity,
    peers: HashMap<NodeId, std::net::SocketAddr>,
    voters: std::collections::BTreeSet<u64>,
    peer_addrs: std::collections::BTreeMap<u64, String>,
    local: NodeId,
    group: ConsensusGroupId,
    tablet: TabletId,
    namespace: NamespaceId,
    serve_tx: async_channel::Sender<OwnerRequest>,
    requests: async_channel::Receiver<OwnerRequest>,
    ready: futures::channel::oneshot::Sender<Result<std::net::SocketAddr, NodeOpenError>>,
}

/// Shared owner state behind the request loop. Cloneable so each
/// request runs on its own task; interior mutability is single-threaded
/// (`Rc<RefCell<..>>`) because nothing here ever leaves the owner
/// reactor.
#[derive(Clone)]
struct OwnerCtx {
    raft: OwnerRaft,
    store: DurableRaftStore,
    machine: ReplicatedStateMachine,
    sidecar: SidecarStore,
    gate: SidecarGate,
    transport: PeerTransport,
    group: ConsensusGroupId,
    namespace: NamespaceId,
    tablet: TabletId,
    transfers: Rc<RefCell<HashMap<u64, PartialTransfer>>>,
}

/// One in-flight snapshot transfer being reassembled from fragments.
struct PartialTransfer {
    vote: openraft::type_config::alias::VoteOf<KiviTypeConfig>,
    meta: openraft::type_config::alias::SnapshotMetaOf<KiviTypeConfig>,
    bytes: Vec<u8>,
}

/// `PeerHandler` over the owner queue: transport tasks forward every
/// incoming RPC (including snapshot fragments) to the owner loop, which
/// owns the Raft handles and the reassembly table. The future stays
/// `Send` (channel ends only).
#[derive(Clone)]
struct ServeRouter {
    tx: async_channel::Sender<OwnerRequest>,
}

impl PeerHandler for ServeRouter {
    fn handle(
        &self,
        from: NodeId,
        group: ConsensusGroupId,
        request: PeerRequest,
    ) -> std::pin::Pin<Box<dyn Future<Output = Result<PeerResponse, PeerRpcError>> + Send + '_>>
    {
        let tx = self.tx.clone();
        Box::pin(async move {
            let (reply, rx) = futures::channel::oneshot::channel();
            tx.send(OwnerRequest::Serve {
                from,
                group,
                request,
                reply,
            })
            .await
            .map_err(|_| PeerRpcError {
                detail: "consensus owner shut down".to_owned(),
            })?;
            rx.await.map_err(|_| PeerRpcError {
                detail: "consensus owner shut down".to_owned(),
            })?
        })
    }
}

impl OwnerCtx {
    /// Serves one request on its own task (see the loop above for why
    /// nothing awaits inline there).
    async fn serve_one(&self, request: OwnerRequest) {
        match request {
            OwnerRequest::Propose { command, reply } => {
                let outcome = self.propose(command).await;
                let _ = reply.send(outcome);
            }
            OwnerRequest::Barrier { reply } => {
                let barrier = self.barrier().await;
                let _ = reply.send(barrier);
            }
            OwnerRequest::WaitApplied { index, reply } => {
                let covered = self.wait_applied(index).await;
                let _ = reply.send(covered);
            }
            OwnerRequest::Status { reply } => {
                let status = self.status().await;
                let _ = reply.send(status);
            }
            OwnerRequest::Serve {
                from,
                group,
                request,
                reply,
            } => {
                let answer = self.serve_rpc(from, group, request).await;
                let _ = reply.send(answer);
            }
            OwnerRequest::SnapshotPurge { reply } => {
                let sealed = self.snapshot_purge().await;
                let _ = reply.send(sealed);
            }
            OwnerRequest::SuspendPeer { peer, reply } => {
                self.transport.suspend_peer(peer).await;
                let _ = reply.send(());
            }
            OwnerRequest::ResumePeer { peer, reply } => {
                self.transport.resume_peer(peer).await;
                let _ = reply.send(());
            }
            // Handled by the loop (breaks it); unreachable here.
            OwnerRequest::Shutdown => {}
        }
    }

    /// Quorum-commits one deterministic command and maps the result onto
    /// the Kivi-owned proposal contract. Runs on the owner thread.
    async fn propose(&self, command: ReplicatedMutation) -> Result<ProposeOutcome, ProposeError> {
        let response = self
            .raft
            .client_write(command)
            .await
            .map_err(|error| client_write_error(&error))?;
        let outcome = response.data.outcome().cloned().ok_or_else(|| {
            ProposeError::Consensus(ConsensusError::Unavailable {
                reason: "committed entry carried no outcome".to_owned(),
            })
        })?;
        Ok(ProposeOutcome::Applied {
            index: ConsensusLogIndex::new(response.log_id.index),
            outcome,
        })
    }

    /// Establishes a `ReadIndex` linearizable barrier: the leader proves
    /// freshness with a quorum, waits for local applied coverage of the
    /// leadership-bound boundary, and returns the owned barrier. The
    /// leadership identity and boundary are preserved (never collapsed
    /// to a bare index) for future `AtLeast`/follower-read work.
    async fn barrier(&self) -> Result<ReadBarrier, ConsensusError> {
        let fallback = self.leader_fallback();
        let read = self
            .raft
            .ensure_linearizable(openraft::ReadPolicy::ReadIndex)
            .await
            .map_err(|error| map_raft_error(&error, fallback))?;
        Ok(ReadBarrier {
            term: ConsensusTerm::new(read.committed_leader_id().term),
            leader: replica_of(read.committed_leader_id().node_id),
            boundary: ConsensusLogIndex::new(read.index()),
        })
    }

    /// Waits for local applied coverage of `index` (bounded reactor
    /// sleeps; the caller stays runtime-agnostic). Timeout is `Err(())`
    /// (the caller maps it to `CoverageTimeout`); owner death surfaces
    /// through the dropped reply instead.
    async fn wait_applied(&self, index: u64) -> Result<(), ()> {
        let deadline = std::time::Instant::now() + AT_LEAST_TIMEOUT;
        loop {
            if self.machine.status().await.applied_index.unwrap_or(0) >= index {
                return Ok(());
            }
            if std::time::Instant::now() >= deadline {
                return Err(());
            }
            compio::time::sleep(Duration::from_millis(10)).await;
        }
    }

    /// Best-effort leader hint from metrics (fills gaps when the error
    /// itself names no leader).
    fn leader_fallback(&self) -> Option<ReplicaId> {
        self.raft
            .metrics()
            .borrow_watched()
            .current_leader
            .map(replica_of)
    }

    /// Gathers full diagnostics on the owner thread (metrics handles are
    /// `!Send`, so this cannot run on the caller).
    async fn status(&self) -> NodeStatus {
        let metrics = self.raft.metrics().borrow_watched().clone();
        let role = match metrics.state {
            openraft::ServerState::Leader => ReplicaRole::Leader,
            openraft::ServerState::Follower => ReplicaRole::Follower,
            openraft::ServerState::Candidate => ReplicaRole::Candidate,
            // Learners do not exist yet; shutdown is reported through
            // `healthy`, so both map to the closest serving role.
            openraft::ServerState::Learner | openraft::ServerState::Shutdown => {
                ReplicaRole::Follower
            }
        };
        let mut store = self.store.clone();
        let committed = store
            .read_committed()
            .await
            .ok()
            .flatten()
            .map_or(0, |id| id.index);
        let mut machine = self.machine.clone();
        let (applied_id, _) = machine.applied_state().await.unwrap_or_default();
        let applied = applied_id.map_or(0, |id| id.index);
        let sm = self.machine.status().await;
        let running_ok = metrics.running_state.is_ok();
        let running_error = match &metrics.running_state {
            Ok(()) => None,
            Err(error) => Some(format!("raft halted: {error}")),
        };
        let detail = match (sm.last_error, running_error) {
            (Some(sm_error), Some(raft_error)) => Some(format!("{sm_error}; {raft_error}")),
            (Some(sm_error), None) => Some(sm_error),
            (None, Some(raft_error)) => Some(raft_error),
            (None, None) => None,
        };
        // Sidecar diagnostics are best-effort (never fail status).
        let sidecar_metrics = self.sidecar.metrics().snapshot();
        NodeStatus {
            group: self.group,
            replica: ReplicaId::of_node(self.transport.local().node),
            role,
            term: ConsensusTerm::new(metrics.current_term),
            leader: metrics.current_leader.map(replica_of),
            last_log: ConsensusLogIndex::new(metrics.last_log_index.unwrap_or(0)),
            committed: ConsensusLogIndex::new(committed),
            applied: ConsensusLogIndex::new(applied),
            applied_commit: sm.applied_commit,
            snapshot: sm.snapshot_index.map(ConsensusLogIndex::new),
            purged: metrics.purged.map(|id| ConsensusLogIndex::new(id.index)),
            healthy: sm.healthy && running_ok,
            detail,
            peers: self.transport.stats().await,
            sidecar: sidecar_metrics,
        }
    }

    /// Serves one peer RPC for the owned group. Election/append/pre-vote
    /// go straight to Raft; snapshot fragments reassemble here and
    /// install whole via `install_full_snapshot` once complete; manifest
    /// and chunk sidecars serve from the local sidecar store on the bulk
    /// lane (content identity only, never pack offsets).
    async fn serve_rpc(
        &self,
        _from: NodeId,
        group: ConsensusGroupId,
        request: PeerRequest,
    ) -> Result<PeerResponse, PeerRpcError> {
        // Sidecars are group-independent (content-addressed, domain-scoped)
        // and decode with a zero tablet: serve before the group check.
        match request {
            PeerRequest::Manifest(request) => return self.serve_manifest(request).await,
            PeerRequest::Chunk(request) => return self.serve_chunk(request).await,
            _ => {}
        }
        if group != self.group {
            return Err(PeerRpcError {
                detail: format!("group {group} not served here"),
            });
        }
        match request {
            PeerRequest::Snapshot(fragment) => self.serve_fragment(fragment).await,
            other => serve_peer_request(&self.raft, other).await,
        }
    }

    /// Serves one manifest fetch from the local sidecar store.
    async fn serve_manifest(
        &self,
        request: crate::peer::PeerManifestRequest,
    ) -> Result<PeerResponse, PeerRpcError> {
        let refused = |detail: String| PeerRpcError { detail };
        let id = kivi_types::ManifestId::from_bytes(request.manifest);
        if !id.is_valid() {
            return Err(refused("zero manifest id".to_owned()));
        }
        let canonical = self
            .sidecar
            .read_manifest(id)
            .await
            .map_err(|error| refused(format!("manifest {id} unavailable: {error}")))?;
        self.sidecar
            .metrics()
            .bulk_bytes_sent
            .fetch_add(canonical.len() as u64, std::sync::atomic::Ordering::Relaxed);
        Ok(PeerResponse::Manifest(PeerManifestResponse {
            manifest: request.manifest,
            canonical,
        }))
    }

    /// Serves one chunk fetch from the local sidecar store.
    async fn serve_chunk(
        &self,
        request: crate::peer::PeerChunkRequest,
    ) -> Result<PeerResponse, PeerRpcError> {
        let refused = |detail: String| PeerRpcError { detail };
        let id = kivi_types::ChunkId::from_bytes(request.chunk);
        if !id.is_valid() {
            return Err(refused("zero chunk id".to_owned()));
        }
        let bytes = self
            .sidecar
            .read_chunk(id)
            .await
            .map_err(|error| refused(format!("chunk {id} unavailable: {error}")))?;
        self.sidecar
            .metrics()
            .bulk_bytes_sent
            .fetch_add(bytes.len() as u64, std::sync::atomic::Ordering::Relaxed);
        Ok(PeerResponse::Chunk(PeerChunkResponse {
            chunk: request.chunk,
            bytes,
        }))
    }

    /// Buffers one snapshot fragment; on `done`, validates and installs
    /// the whole image. Gaps, meta conflicts, and mid-transfer restarts
    /// (fresh transfer id) reset or refuse loudly — the sender retries
    /// the transfer from scratch with a new id, never resumes a
    /// corrupted reassembly.
    async fn serve_fragment(
        &self,
        fragment: PeerSnapshotRequest,
    ) -> Result<PeerResponse, PeerRpcError> {
        let refused = |detail: String| PeerRpcError { detail };
        // Decode-first, buffer-second: metadata faults refuse before any
        // bytes are retained.
        let meta = decode_snapshot_meta(&fragment.meta)
            .map_err(|error| refused(format!("snapshot metadata invalid: {error}")))?;
        let vote = crate::router::decode_wire_vote(&fragment.vote);
        // A fresh transfer id always starts a new reassembly (a retried
        // transfer never continues an aborted one's bytes). The map
        // borrow never crosses an await: all branches below drop it
        // before any async work.
        let step = {
            let mut transfers = self.transfers.borrow_mut();
            let partial = transfers
                .entry(fragment.transfer)
                .or_insert(PartialTransfer {
                    vote,
                    meta: meta.clone(),
                    bytes: Vec::new(),
                });
            if partial.meta != meta {
                transfers.remove(&fragment.transfer);
                Err(refused(
                    "snapshot transfer metadata changed mid-stream".to_owned(),
                ))
            } else if fragment.offset != partial.bytes.len() as u64 {
                // Gap or overlap: drop the partial attempt; the sender's
                // next transfer (fresh id) starts clean.
                let buffered = partial.bytes.len();
                transfers.remove(&fragment.transfer);
                Err(refused(format!(
                    "snapshot fragment gap: offset {} != buffered {buffered}",
                    fragment.offset,
                )))
            } else {
                partial.bytes.extend_from_slice(&fragment.data);
                Ok((partial.vote, fragment.done))
            }
        };
        let (ack_vote, done) = step?;
        if !done {
            return Ok(PeerResponse::Snapshot(crate::peer::PeerSnapshotResponse {
                vote: crate::router::encode_wire_vote(&ack_vote),
            }));
        }
        let partial = self
            .transfers
            .borrow_mut()
            .remove(&fragment.transfer)
            .expect("transfer buffered above");
        self.install_transfer(partial).await
    }

    /// Pre-validates and installs one reassembled snapshot. The
    /// prechecks keep a forged or skewed transfer out of
    /// `install_full_snapshot`, which panics (by upstream contract) on
    /// vote/committed inconsistency instead of returning an error.
    async fn install_transfer(
        &self,
        partial: PartialTransfer,
    ) -> Result<PeerResponse, PeerRpcError> {
        use openraft::type_config::alias::SnapshotOf;
        let refused = |detail: String| PeerRpcError { detail };
        // Stale vote: the sender lost leadership mid-transfer. Refuse
        // without touching Raft (a smaller vote would trip the
        // install-time leadership assert).
        let local_vote = {
            let mut store = self.store.clone();
            store.read_vote().await.map_err(|error| {
                refused(format!("snapshot precheck failed to read vote: {error}"))
            })?
        };
        if let Some(local) = local_vote
            && vote_is_stale(&local, &partial.vote)
        {
            return Err(refused("snapshot transfer vote is stale".to_owned()));
        }
        // Committed-history conflict: a snapshot at or behind committed
        // history contradicts it unless it is exactly the committed
        // prefix (which `install_full_snapshot` accepts as a no-op).
        // Anything else would trip the install-time committed assert.
        if let Some(last) = partial.meta.last_log_id {
            let mut store = self.store.clone();
            let committed = store.read_committed().await.map_err(|error| {
                refused(format!("snapshot precheck failed to read commit: {error}"))
            })?;
            if let Some(committed) = committed
                && (last.index < committed.index
                    || (last.index == committed.index && last != committed))
            {
                return Err(refused(
                    "snapshot contradicts locally committed history".to_owned(),
                ));
            }
        }
        // Differential sidecar catch-up: the snapshot body carries only
        // manifest references; bulk bytes stay in chunk packs. Fetch only
        // missing immutable objects before installing, so the install never
        // lands a root it cannot reconstruct. Transfer is content-addressed:
        // a follower holding 95% of chunks downloads only the missing 5%.
        if let Ok(decoded) = crate::state_machine::ReplicatedTablet::decode_snapshot(
            &partial.bytes,
            self.namespace,
            self.tablet,
        ) {
            let leader_hint = Some(NodeId::from_u64(partial.vote.leader_id.node_id));
            for (_key, object) in &decoded.objects {
                if let kivi_state::LogicalValue::Chunked(chunked) = object.value() {
                    self.gate
                        .ensure_root(chunked.manifest, chunked.logical_len, leader_hint)
                        .await
                        .map_err(|error| {
                            refused(format!("snapshot sidecar unavailable: {error}"))
                        })?;
                }
            }
        }
        let snapshot = SnapshotOf::<KiviTypeConfig, std::io::Cursor<Vec<u8>>> {
            meta: partial.meta,
            snapshot: std::io::Cursor::new(partial.bytes),
        };
        let response = self
            .raft
            .install_full_snapshot(partial.vote, snapshot)
            .await
            .map_err(|error| refused(format!("snapshot install refused: {error}")))?;
        Ok(PeerResponse::Snapshot(crate::peer::PeerSnapshotResponse {
            vote: crate::router::encode_wire_vote(&response.vote),
        }))
    }

    /// Triggers a snapshot seal, waits for the durable base, then purges
    /// the log through it — all on the owner thread (reactor sleeps; the
    /// caller stays runtime-agnostic).
    async fn snapshot_purge(&self) -> Result<ConsensusLogIndex, String> {
        if !self.machine.status().await.healthy {
            return Err("state machine unhealthy: snapshot blocked".to_owned());
        }
        self.raft
            .trigger()
            .snapshot()
            .await
            .map_err(|error| format!("snapshot trigger refused: {error}"))?;
        // Triggers return at once; wait for the durable base to land.
        let deadline = std::time::Instant::now() + Duration::from_secs(30);
        let base = loop {
            if let Some(base) = self.machine.snapshotted_index().await {
                break base;
            }
            if std::time::Instant::now() >= deadline {
                return Err("snapshot never sealed before the deadline".to_owned());
            }
            compio::time::sleep(Duration::from_millis(50)).await;
        };
        self.raft
            .trigger()
            .purge_log(base)
            .await
            .map_err(|error| format!("purge trigger refused: {error}"))?;
        let deadline = std::time::Instant::now() + Duration::from_secs(30);
        loop {
            let purged = self
                .raft
                .metrics()
                .borrow_watched()
                .purged
                .map_or(0, |id| id.index);
            if purged >= base {
                break;
            }
            if std::time::Instant::now() >= deadline {
                return Err(format!(
                    "log never purged through {base} before the deadline (purged {purged})"
                ));
            }
            compio::time::sleep(Duration::from_millis(50)).await;
        }
        Ok(ConsensusLogIndex::new(base))
    }
}

/// Whether `incoming` is strictly older than `local` under Raft vote
/// order (term, then leader, then committed-beats-uncommitted). Used to
/// refuse stale snapshot transfers before they reach the install-time
/// leadership assert.
fn vote_is_stale(
    local: &openraft::type_config::alias::VoteOf<KiviTypeConfig>,
    incoming: &openraft::type_config::alias::VoteOf<KiviTypeConfig>,
) -> bool {
    (
        local.leader_id.term,
        local.leader_id.node_id,
        local.committed,
    ) > (
        incoming.leader_id.term,
        incoming.leader_id.node_id,
        incoming.committed,
    )
}

/// Inputs to [`bootstrap_group`], bundled so the arg count stays readable.
struct BootstrapInputs<'a> {
    topology: &'a ClusterTopology,
    tablet: TabletId,
    local: NodeId,
    cluster: ClusterId,
    node: NodeId,
    voters: &'a std::collections::BTreeSet<u64>,
    peer_addrs: &'a std::collections::BTreeMap<u64, String>,
}

/// Installs or verifies group membership: first formation installs the
/// joint membership exactly once; restarts recover existing Raft state
/// and fail loudly on identity/membership conflicts instead of
/// initializing a new group.
///
/// Runs on the owner thread (Raft futures are `!Send`).
///
/// # Errors
///
/// Returns [`NodeOpenError::Initialize`] when first formation fails, or
/// [`NodeOpenError::Bootstrap`] when restart config conflicts with
/// durable state.
async fn bootstrap_group(
    inputs: &BootstrapInputs<'_>,
    store: &mut DurableRaftStore,
    machine: &ReplicatedStateMachine,
    raft: &OwnerRaft,
) -> Result<(), NodeOpenError> {
    use NodeOpenError as Fault;
    let BootstrapInputs {
        topology,
        tablet,
        local,
        cluster,
        node,
        voters,
        peer_addrs,
    } = *inputs;
    // Any trace of vote, log, or snapshot base means existing — recover
    // it, never re-initialize.
    match classify_from_store(store, machine).await {
        Bootstrap::Fresh => {
            let members: std::collections::BTreeMap<u64, openraft::BasicNode> = voters
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
        Bootstrap::Existing => {
            // Durable voters come from storage, not from fresh metrics
            // (which start default until `RaftCore` loads): newest
            // retained membership entry first, snapshot membership when
            // the log was purged past it (every purge grounds in a
            // snapshot). An empty set means no membership evidence
            // survives (e.g. a bare granted vote): membership adopts via
            // replication — refusing here would brick a legitimate
            // rejoin with no conflicting evidence.
            let mut durable_voters = store.membership_voters().await;
            if durable_voters.is_empty() {
                durable_voters = machine.member_voters().await;
            }
            if !durable_voters.is_empty() {
                crate::cluster::verify_against_durable(
                    topology,
                    tablet,
                    local,
                    &DurableClusterView {
                        cluster,
                        node,
                        voters: durable_voters,
                    },
                )
                .map_err(|error| Fault::Bootstrap {
                    reason: error.to_string(),
                })?;
            }
            Ok(())
        }
    }
}

/// Decides fresh vs. existing from durable signals: any vote, any log
/// entry, or any snapshot base means existing.
async fn classify_from_store(
    store: &mut DurableRaftStore,
    machine: &ReplicatedStateMachine,
) -> Bootstrap {
    let has_vote = matches!(store.read_vote().await, Ok(Some(_)));
    let has_log = store
        .get_log_state()
        .await
        .is_ok_and(|state| state.last_log_id.is_some());
    let has_snapshot = machine.snapshotted_index().await.is_some();
    classify_bootstrap(has_vote, has_log, has_snapshot)
}

/// Maps a linearizable-read failure onto Kivi-owned routing errors: a
/// named leader becomes a hint (falling back to the metrics view when
/// the error itself is leaderless), quorum loss without a leader is
/// unknown, and fatal storage faults are unavailable. Never a phantom
/// success: losing leadership between check and commit surfaces here as
/// `NotLeader`.
fn map_raft_error(
    error: &openraft::errors::RaftError<
        KiviTypeConfig,
        openraft::errors::LinearizableReadError<KiviTypeConfig>,
    >,
    fallback: Option<ReplicaId>,
) -> ConsensusError {
    use openraft::errors::{LinearizableReadError, RaftError};
    match error {
        RaftError::Fatal(fatal) => ConsensusError::Unavailable {
            reason: format!("consensus fatal: {fatal}"),
        },
        RaftError::APIError(LinearizableReadError::ForwardToLeader(forward)) => {
            ConsensusError::NotLeader {
                hint: LeaderHint {
                    leader: forward.leader_id.map(replica_of).or(fallback),
                },
            }
        }
        RaftError::APIError(LinearizableReadError::QuorumNotEnough(_)) => {
            ConsensusError::LeaderUnknown
        }
    }
}

/// Maps a `client_write` failure onto proposal errors. Membership-change
/// failures are unreachable for data writes and fail closed as
/// unavailable.
fn client_write_error(
    error: &openraft::errors::RaftError<
        KiviTypeConfig,
        openraft::errors::ClientWriteError<KiviTypeConfig>,
    >,
) -> ProposeError {
    use openraft::errors::{ClientWriteError, RaftError};
    match &error {
        RaftError::APIError(ClientWriteError::ForwardToLeader(forward)) => {
            ProposeError::Consensus(ConsensusError::NotLeader {
                hint: LeaderHint {
                    leader: forward.leader_id.map(replica_of),
                },
            })
        }
        RaftError::APIError(ClientWriteError::ChangeMembershipError(install)) => {
            ProposeError::Consensus(ConsensusError::Unavailable {
                reason: format!("unexpected membership verdict on a data write: {install}"),
            })
        }
        RaftError::Fatal(fatal) => ProposeError::Consensus(ConsensusError::Unavailable {
            reason: format!("consensus fatal: {fatal}"),
        }),
    }
}

/// Ensures proposal sidecars are durable before replication.
///
/// * `SetChunked` / `SetConditionalChunked`: the referenced root must
///   already be durable locally (cluster server stages before proposing).
///   Missing sidecars fail closed as unavailable — the leader never
///   proposes a root referencing volatile buffers.
/// * `SetRange` against a chunked base: restages by reading the old value
///   via the sidecar store, applying the patch (zero-pad gaps, Redis
///   `SETRANGE` semantics), staging the new value (untouched chunk ids
///   reuse via content addressing), and returning a restaged
///   `SetConditionalChunked` preserving expiry (`Keep`). Followers fetch
///   only missing chunks via the durability gate.
/// * All other operations need no sidecars.
///
/// Returns `Ok(None)` to proceed with `op`, `Ok(Some(restaged))` to
/// proceed with a restaged operation instead.
#[allow(clippy::too_many_lines)]
async fn ensure_proposal_sidecars_for(
    sidecar: &SidecarStore,
    machine: &ReplicatedStateMachine,
    op: &Operation,
    now: UnixMicros,
) -> Result<Option<Operation>, ProposeError> {
    match op {
        Operation::SetChunked {
            manifest,
            logical_len,
            ..
        } => {
            let durable = sidecar
                .check_root(*manifest, *logical_len)
                .await
                .map_err(|error| {
                    ProposeError::Consensus(ConsensusError::Unavailable {
                        reason: format!("sidecar check failed: {error}"),
                    })
                })?;
            if !durable {
                return Err(ProposeError::Consensus(ConsensusError::Unavailable {
                    reason: format!(
                        "chunked root {manifest} not durable locally; stage before proposing"
                    ),
                }));
            }
            Ok(None)
        }
        Operation::SetConditionalChunked {
            manifest,
            logical_len,
            ..
        } => {
            let durable = sidecar
                .check_root(*manifest, *logical_len)
                .await
                .map_err(|error| {
                    ProposeError::Consensus(ConsensusError::Unavailable {
                        reason: format!("sidecar check failed: {error}"),
                    })
                })?;
            if !durable {
                return Err(ProposeError::Consensus(ConsensusError::Unavailable {
                    reason: format!(
                        "chunked root {manifest} not durable locally; stage before proposing"
                    ),
                }));
            }
            Ok(None)
        }
        Operation::SetRange { key, offset, patch } => {
            let base = machine.peek_base(key, now).await.map_err(|error| {
                ProposeError::Consensus(ConsensusError::Unavailable { reason: error })
            })?;
            if base != crate::state_machine::RangeBase::Chunked {
                return Ok(None);
            }
            // Chunked base: restage. Read the reference via a local Get,
            // resolve bytes via the sidecar store, patch, stage anew.
            let reference = machine
                .read_local(&Operation::Get { key: key.clone() }, now)
                .await
                .map_err(|error| {
                    ProposeError::Consensus(ConsensusError::Unavailable {
                        reason: format!("chunked base unreadable: {error}"),
                    })
                })?;
            let (manifest, logical_len) = match reference {
                kivi_state::OperationResult::ChunkedValue {
                    manifest,
                    logical_len,
                } => (manifest, logical_len),
                kivi_state::OperationResult::Value(_) | kivi_state::OperationResult::Length(_) => {
                    // Raced to inline between peek and read; proceed inline.
                    return Ok(None);
                }
                other => {
                    return Err(ProposeError::Consensus(ConsensusError::Unavailable {
                        reason: format!("unexpected chunked base read: {other:?}"),
                    }));
                }
            };
            let old = sidecar
                .read_value(manifest, logical_len)
                .await
                .map_err(|error| {
                    ProposeError::Consensus(ConsensusError::Unavailable {
                        reason: format!("chunked base bytes unreadable: {error}"),
                    })
                })?;
            let patched = apply_range_patch(&old, *offset, patch).map_err(ProposeError::Op)?;
            if patched.len() as u64 > kivi_chunk::policy::MAX_LEGACY_VALUE_BYTES {
                return Err(ProposeError::Unsupported {
                    reason: "range result exceeds 64 MiB; rewrite through streaming upload"
                        .to_owned(),
                });
            }
            let staged = sidecar.stage_value(&patched).await.map_err(|error| {
                ProposeError::Consensus(ConsensusError::Unavailable {
                    reason: format!("restage failed: {error}"),
                })
            })?;
            Ok(Some(Operation::SetConditionalChunked {
                key: key.clone(),
                manifest: staged.manifest,
                logical_len: staged.logical_len,
                condition: kivi_state::SetCondition::Always,
                expiry: kivi_state::ExpiryPolicy::Keep,
            }))
        }
        _ => Ok(None),
    }
}

/// Applies a Redis-`SETRANGE` patch: overwrites `[offset, offset+len)`
/// zero-padding past-the-end gaps. Pure; mirrors `SpliceBytes` apply.
fn apply_range_patch(
    base: &[u8],
    offset: u64,
    patch: &bytes::Bytes,
) -> Result<Vec<u8>, kivi_state::OpError> {
    use kivi_state::OpError;
    let offset = usize::try_from(offset).map_err(|_| OpError::StaleRangeBase)?;
    let patch_len = patch.len();
    let new_len = offset.saturating_add(patch_len).max(base.len());
    if new_len as u64 > kivi_chunk::policy::MAX_LEGACY_VALUE_BYTES {
        return Err(OpError::StaleRangeBase);
    }
    let mut out = vec![0u8; new_len];
    out[..base.len()].copy_from_slice(base);
    out[offset..offset + patch_len].copy_from_slice(patch);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use std::future::Future;
    use std::time::Duration;

    use kivi_state::{Key, Operation, OperationResult};
    use kivi_types::{
        ClusterId, MutationIdentity, NamespaceId, NodeId, ReadContract, RequestIdentity,
        RequestSeq, SessionId, TabletAuthority, TabletEpoch, TabletId, UnixMicros,
        WriteGuardGeneration,
    };

    use super::{NodeConfig, ProposeError, ProposeOutcome, ReadError, ReplicatedNode};
    use crate::cluster::{ClusterTopology, NodeDescriptor, TabletAssignment};
    use crate::types::{ConsensusError, ReplicaRole};

    const CLUSTER: u128 = 0x0C10_57E2;
    const NS: NamespaceId = NamespaceId::from_u64(1);
    const TABLET: TabletId = TabletId::from_u64(9);
    const NOW: UnixMicros = UnixMicros::from_micros(1_000_000);

    fn authority() -> TabletAuthority {
        TabletAuthority::new(TABLET, TabletEpoch::INITIAL, WriteGuardGeneration::INITIAL)
    }

    fn identity(session: u128, seq: u64) -> MutationIdentity {
        MutationIdentity::new(
            RequestIdentity::new(SessionId::from_u128(session), RequestSeq::from_u64(seq)),
            RequestSeq::from_u64(0),
        )
    }

    fn block_on<F, T>(future: F) -> T
    where
        F: Future<Output = T>,
    {
        compio::runtime::Runtime::new()
            .expect("compio test runtime")
            .block_on(future)
    }

    /// Static 1×3 topology over three loopback peer addresses.
    fn test_topology(addrs: &[std::net::SocketAddr]) -> ClusterTopology {
        ClusterTopology {
            cluster: ClusterId::from_u128(CLUSTER),
            nodes: vec![
                NodeDescriptor {
                    node: NodeId::from_u64(1),
                    peer: addrs[0],
                    native: "127.0.0.1:9001".parse().expect("addr"),
                },
                NodeDescriptor {
                    node: NodeId::from_u64(2),
                    peer: addrs[1],
                    native: "127.0.0.1:9002".parse().expect("addr"),
                },
                NodeDescriptor {
                    node: NodeId::from_u64(3),
                    peer: addrs[2],
                    native: "127.0.0.1:9003".parse().expect("addr"),
                },
            ],
            tablets: vec![TabletAssignment {
                tablet: TABLET,
                replicas: vec![
                    NodeId::from_u64(1),
                    NodeId::from_u64(2),
                    NodeId::from_u64(3),
                ],
            }],
        }
    }

    /// Opens one test node: scratch directory and static identity. QUIC
    /// binds its UDP socket inside the owner thread, so no pre-bound
    /// listener crosses the boundary (insecure test TLS: loopback only).
    async fn open_test_node(
        dir: &std::path::Path,
        id: u64,
        topology: &ClusterTopology,
    ) -> ReplicatedNode {
        ReplicatedNode::open(NodeConfig {
            data_dir: dir.to_owned(),
            namespace: NS,
            tablet: TABLET,
            authority: authority(),
            local: NodeId::from_u64(id),
            topology: topology.clone(),
            segment_target_bytes: 1024 * 1024,
            transport: crate::transport::TransportConfig::default(),
            peer_certs: std::collections::HashMap::new(),
            insecure_peer_tls: true,
        })
        .await
        .expect("node opens")
    }

    /// Probes one free loopback UDP port for the QUIC endpoint.
    fn probe_udp() -> std::net::SocketAddr {
        let socket = std::net::UdpSocket::bind("127.0.0.1:0").expect("probe binds");
        socket.local_addr().expect("probe addr")
    }

    /// One test cluster: three nodes on loopback with probed ports and
    /// scratch data directories. The topology (with the probed ports) is
    /// returned for restarts reusing the same addresses.
    async fn cluster() -> (Vec<ReplicatedNode>, Vec<tempfile::TempDir>, ClusterTopology) {
        let addrs = vec![probe_udp(), probe_udp(), probe_udp()];
        let topology = test_topology(&addrs);
        let mut nodes = Vec::new();
        let mut dirs = Vec::new();
        for id in [1u64, 2, 3] {
            let dir = tempfile::tempdir().expect("scratch");
            nodes.push(open_test_node(dir.path(), id, &topology).await);
            dirs.push(dir);
        }
        (nodes, dirs, topology)
    }

    /// Restarts the node at `index` from its existing directory with the
    /// same topology (the bootstrap-existing path: no re-initialization).
    /// UDP has no `TIME_WAIT`, so the restarted node rebinds its topology
    /// port verbatim and the mesh heals through outbound dials.
    async fn restart(
        dirs: &[tempfile::TempDir],
        topology: &ClusterTopology,
        index: usize,
    ) -> ReplicatedNode {
        open_test_node(dirs[index].path(), index as u64 + 1, topology).await
    }

    /// Waits for exactly one agreed leader, then confirms it stays the
    /// leader across consecutive polls (a single agreeing poll can catch
    /// a transition mid-handover).
    async fn wait_leader(nodes: &[ReplicatedNode]) -> usize {
        let deadline = std::time::Instant::now() + Duration::from_secs(30);
        let mut stable = 0usize;
        let mut last = usize::MAX;
        loop {
            let mut leaders = Vec::new();
            for (index, node) in nodes.iter().enumerate() {
                if node.status().await.role == ReplicaRole::Leader {
                    leaders.push(index);
                }
            }
            if leaders.len() == 1 && leaders[0] == last {
                stable += 1;
                if stable >= 3 {
                    return leaders[0];
                }
            } else {
                stable = 0;
                last = leaders.first().copied().unwrap_or(usize::MAX);
            }
            assert!(
                std::time::Instant::now() < deadline,
                "cluster never elected exactly one stable leader"
            );
            compio::time::sleep(Duration::from_millis(100)).await;
        }
    }

    /// Waits until a leader other than `exclude` is stable.
    async fn wait_leader_except(nodes: &[ReplicatedNode], exclude: usize) -> usize {
        let deadline = std::time::Instant::now() + Duration::from_secs(30);
        let mut stable = 0usize;
        let mut last = usize::MAX;
        loop {
            let mut leaders = Vec::new();
            for (index, node) in nodes.iter().enumerate() {
                if index != exclude && node.status().await.role == ReplicaRole::Leader {
                    leaders.push(index);
                }
            }
            if leaders.len() == 1 && leaders[0] == last {
                stable += 1;
                if stable >= 3 {
                    return leaders[0];
                }
            } else {
                stable = 0;
                last = leaders.first().copied().unwrap_or(usize::MAX);
            }
            assert!(
                std::time::Instant::now() < deadline,
                "majority never elected without node {exclude}"
            );
            compio::time::sleep(Duration::from_millis(100)).await;
        }
    }

    /// Symmetrically partitions the leader from every follower (both
    /// directions suspended) and waits until every suspended link reads
    /// disconnected on both ends: proposals after this point cannot have
    /// flown before the partition closed (no phantom quorum from
    /// in-flight RPCs).
    async fn partition_leader(nodes: &[ReplicatedNode], leader: usize) {
        let leader_id = NodeId::from_u64(leader as u64 + 1);
        for (index, node) in nodes.iter().enumerate() {
            if index == leader {
                continue;
            }
            let peer = NodeId::from_u64(index as u64 + 1);
            nodes[leader].suspend_peer(peer).await;
            node.suspend_peer(leader_id).await;
        }
        let deadline = std::time::Instant::now() + Duration::from_secs(10);
        loop {
            let mut down = true;
            for (index, node) in nodes.iter().enumerate() {
                let status = node.status().await;
                for (peer, stats) in &status.peers {
                    // A leader link has exactly one end at the leader.
                    let mine = index == leader;
                    let theirs = peer.as_u64() == leader as u64 + 1;
                    if mine != theirs && (stats.connected_control || stats.connected_bulk) {
                        down = false;
                    }
                }
            }
            if down {
                break;
            }
            assert!(
                std::time::Instant::now() < deadline,
                "partition never fully closed"
            );
            compio::time::sleep(Duration::from_millis(50)).await;
        }
    }

    /// Heals every partition in the cluster.
    async fn heal_all(nodes: &[ReplicatedNode]) {
        for node in nodes {
            for peer in [1u64, 2, 3] {
                node.resume_peer(NodeId::from_u64(peer)).await;
            }
        }
    }

    /// Waits until every node applied at least `index`.
    async fn wait_applied(nodes: &[ReplicatedNode], index: u64) {
        let deadline = std::time::Instant::now() + Duration::from_secs(20);
        loop {
            let mut ready = true;
            for node in nodes {
                if node.status().await.applied.get() < index {
                    ready = false;
                }
            }
            if ready {
                break;
            }
            assert!(
                std::time::Instant::now() < deadline,
                "cluster never applied index {index}"
            );
            compio::time::sleep(Duration::from_millis(50)).await;
        }
    }

    /// Opens a cluster, elects, writes `x=1` through the leader, and
    /// waits for full application. Returns nodes, directories, topology,
    /// leader index, and the write index.
    async fn elected_cluster() -> (
        Vec<ReplicatedNode>,
        Vec<tempfile::TempDir>,
        ClusterTopology,
        usize,
        u64,
    ) {
        let (nodes, dirs, topology) = cluster().await;
        let leader = wait_leader(&nodes).await;
        let outcome = nodes[leader]
            .propose(
                &Operation::Set {
                    key: Key::from("x"),
                    value: bytes::Bytes::from_static(b"1"),
                },
                None,
                None,
                NOW,
            )
            .await
            .expect("leader proposes");
        let index = match outcome {
            ProposeOutcome::Applied { index, .. } => index.get(),
            other => panic!("expected Applied, got {other:?}"),
        };
        wait_applied(&nodes, index).await;
        (nodes, dirs, topology, leader, index)
    }

    /// Leader writes and linearizable reads serve the committed value.
    #[test]
    fn leader_writes_and_reads_latest() {
        block_on(async {
            let (nodes, _dirs, _topology, leader, _index) = elected_cluster().await;
            let read = nodes[leader]
                .read(
                    &Operation::Get {
                        key: Key::from("x"),
                    },
                    ReadContract::Latest,
                    NOW,
                )
                .await
                .expect("latest reads");
            assert_eq!(
                read,
                OperationResult::Value(Some(bytes::Bytes::from_static(b"1")))
            );
            for node in &nodes {
                node.shutdown().await;
            }
        });
    }

    /// Same-identity retry returns the original outcome without
    /// re-executing (exactly-once across the replicated path).
    #[test]
    fn same_identity_retry_is_duplicate() {
        block_on(async {
            let (nodes, _dirs, _topology, leader, _index) = elected_cluster().await;
            let add = Operation::CounterAdd {
                key: Key::from("n"),
                delta: 1,
            };
            let first = nodes[leader]
                .propose(&add, Some(identity(0x5E55, 1)), None, NOW)
                .await
                .expect("counter applies");
            assert!(matches!(first, ProposeOutcome::Applied { .. }));
            let retry = nodes[leader]
                .propose(&add, Some(identity(0x5E55, 1)), None, NOW)
                .await
                .expect("retry answers");
            assert!(
                matches!(retry, ProposeOutcome::Duplicate { .. }),
                "same identity retries without re-executing, got {retry:?}"
            );
            let value = nodes[leader]
                .read(
                    &Operation::CounterGet {
                        key: Key::from("n"),
                    },
                    ReadContract::Latest,
                    NOW,
                )
                .await
                .expect("reads counter");
            assert_eq!(value, OperationResult::Counter(Some(1)));
            for node in &nodes {
                node.shutdown().await;
            }
        });
    }

    /// Followers hint the leader (never a phantom commit), refuse
    /// `Latest`, serve `Any`, and chunk sidecars fail explicitly.
    #[test]
    fn follower_hints_and_weak_reads() {
        block_on(async {
            let (nodes, _dirs, _topology, leader, _index) = elected_cluster().await;
            let follower = (leader + 1) % 3;
            // Follower mutations fail with a leader hint (never OpenRaft
            // types, never a phantom commit).
            let error = nodes[follower]
                .propose(
                    &Operation::Set {
                        key: Key::from("y"),
                        value: bytes::Bytes::from_static(b"2"),
                    },
                    None,
                    None,
                    NOW,
                )
                .await
                .expect_err("follower refuses writes");
            match error {
                ProposeError::Consensus(ConsensusError::NotLeader { hint }) => {
                    assert_eq!(
                        hint.leader.map(|leader| leader.node().as_u64()),
                        Some(leader as u64 + 1)
                    );
                }
                other => panic!("expected NotLeader with hint, got {other:?}"),
            }
            // Follower `Latest` refuses; `Any` serves applied state.
            let error = nodes[follower]
                .read(
                    &Operation::Get {
                        key: Key::from("x"),
                    },
                    ReadContract::Latest,
                    NOW,
                )
                .await
                .expect_err("follower refuses Latest");
            assert!(matches!(
                error,
                ReadError::Consensus(ConsensusError::NotLeader { .. })
            ));
            let weak = nodes[follower]
                .read(
                    &Operation::Get {
                        key: Key::from("x"),
                    },
                    ReadContract::Any,
                    NOW,
                )
                .await
                .expect("follower serves Any");
            assert_eq!(
                weak,
                OperationResult::Value(Some(bytes::Bytes::from_static(b"1")))
            );
            // Undurable chunked roots fail closed (never proposed): the
            // leader stages sidecars before proposing, so a fake manifest
            // is unavailable, never silently replicated.
            let error = nodes[leader]
                .propose(
                    &Operation::SetChunked {
                        key: Key::from("big"),
                        manifest: kivi_types::ManifestId::from_bytes([0x11; 32]),
                        logical_len: 1_000_000,
                    },
                    None,
                    None,
                    NOW,
                )
                .await
                .expect_err("undurable chunked root must not propose");
            assert!(
                matches!(
                    error,
                    ProposeError::Consensus(ConsensusError::Unavailable { .. })
                ),
                "fake manifest fails as unavailable, got {error:?}"
            );
            for node in &nodes {
                node.shutdown().await;
            }
        });
    }

    /// Restart reuses durable state (new incarnation, no re-bootstrap)
    /// and the restarted replica catches up and converges.
    #[test]
    fn restarted_replica_catches_up() {
        block_on(async {
            let (mut nodes, dirs, topology) = cluster().await;
            let leader = wait_leader(&nodes).await;
            for (session, value) in [0xA1u128, 0xA2, 0xA3].into_iter().zip([1i64, 2, 3]) {
                nodes[leader]
                    .propose(
                        &Operation::CounterAdd {
                            key: Key::from("n"),
                            delta: value,
                        },
                        Some(identity(session, 1)),
                        None,
                        NOW,
                    )
                    .await
                    .expect("writes apply");
            }
            let status = nodes[leader].status().await;
            wait_applied(&nodes, status.applied.get()).await;
            // Restart a follower from its data directory.
            let follower = (leader + 1) % 3;
            let incarnation = nodes[follower].incarnation();
            nodes[follower].shutdown().await;
            drop(nodes.remove(follower));
            let restarted = restart(&dirs, &topology, follower).await;
            assert!(
                restarted.incarnation().as_u64() > incarnation.as_u64(),
                "incarnation advances across restarts"
            );
            // Convergence without repair: the restarted replica applies the
            // retained tail up to the pre-restart applied watermark.
            let watermark = status.applied;
            let deadline = std::time::Instant::now() + Duration::from_secs(20);
            loop {
                if restarted.status().await.applied >= watermark {
                    break;
                }
                assert!(
                    std::time::Instant::now() < deadline,
                    "restarted replica never converged"
                );
                compio::time::sleep(Duration::from_millis(50)).await;
            }
            let value = restarted
                .read(
                    &Operation::CounterGet {
                        key: Key::from("n"),
                    },
                    ReadContract::Any,
                    NOW,
                )
                .await
                .expect("converged replica reads");
            assert_eq!(value, OperationResult::Counter(Some(6)));
            restarted.shutdown().await;
            for node in &nodes {
                node.shutdown().await;
            }
        });
    }

    /// Minority fencing: an isolated node must never acknowledge a new
    /// strong mutation, while the majority elects and proceeds. Healing
    /// converges with no dual acknowledged history.
    ///
    /// The isolated proposal resolves in one of two correct ways: it
    /// stays pending while the node still considers itself leader without
    /// quorum, or it fails fast with a routing error once the node steps
    /// down or campaigns (0.10 elections move promptly). `Applied` is the
    /// only forbidden outcome — acknowledging without quorum.
    #[test]
    fn isolated_leader_cannot_acknowledge() {
        block_on(async {
            let (nodes, _dirs, _topology) = cluster().await;
            let leader = wait_leader(&nodes).await;
            nodes[leader]
                .propose(
                    &Operation::CounterAdd {
                        key: Key::from("n"),
                        delta: 1,
                    },
                    Some(identity(0xD0, 1)),
                    None,
                    NOW,
                )
                .await
                .expect("baseline applies");
            let base = nodes[leader].status().await.applied.get();
            wait_applied(&nodes, base).await;
            // `suspend_peer` returns only after the evicted tasks exited
            // and their sockets closed, so the partition is airtight for
            // the fencing verdict below (no dying-writer flush can still
            // complete quorum).
            partition_leader(&nodes, leader).await;
            // The majority elects without the isolated node.
            let majority = wait_leader_except(&nodes, leader).await;
            // The isolated node must not acknowledge a new strong
            // mutation: without quorum its commit can never complete.
            let verdict = compio::time::timeout(
                Duration::from_secs(5),
                nodes[leader].propose(
                    &Operation::CounterAdd {
                        key: Key::from("n"),
                        delta: 100,
                    },
                    Some(identity(0xD1, 1)),
                    None,
                    NOW,
                ),
            )
            .await;
            match verdict {
                // Still leader, still pending, or stepped down with a
                // prompt routing failure: no ack without quorum either way.
                Err(_) | Ok(Err(_)) => {}
                Ok(Ok(outcome)) => {
                    panic!("isolated node acknowledged without quorum: {outcome:?}");
                }
            }
            // The majority side proceeds.
            nodes[majority]
                .propose(
                    &Operation::CounterAdd {
                        key: Key::from("n"),
                        delta: 10,
                    },
                    Some(identity(0xD2, 1)),
                    None,
                    NOW,
                )
                .await
                .expect("majority writes");
            let read = nodes[majority]
                .read(
                    &Operation::CounterGet {
                        key: Key::from("n"),
                    },
                    ReadContract::Latest,
                    NOW,
                )
                .await
                .expect("majority reads");
            assert_eq!(read, OperationResult::Counter(Some(11)));
            // Heal: the isolated entry (never quorum-committed) vanishes and
            // every replica converges on the majority history — no dual
            // acknowledged history exists anywhere.
            heal_all(&nodes).await;
            wait_applied(&nodes, nodes[majority].status().await.applied.get()).await;
            for node in &nodes {
                let value = node
                    .read(
                        &Operation::CounterGet {
                            key: Key::from("n"),
                        },
                        ReadContract::Any,
                        NOW,
                    )
                    .await
                    .expect("converged read");
                assert_eq!(value, OperationResult::Counter(Some(11)));
            }
            for node in &nodes {
                node.shutdown().await;
            }
        });
    }

    /// Log conflict repair: a real divergent uncommitted suffix on the
    /// isolated leader is replaced by the authoritative log on heal,
    /// through the persisted `RaftTruncate` record — and the record
    /// survives a restart.
    #[test]
    fn conflicting_suffix_truncates_through_durable_record() {
        block_on(async {
            let (nodes, dirs, topology) = cluster().await;
            let leader = wait_leader(&nodes).await;
            nodes[leader]
                .propose(
                    &Operation::CounterAdd {
                        key: Key::from("n"),
                        delta: 1,
                    },
                    Some(identity(0xE0, 1)),
                    None,
                    NOW,
                )
                .await
                .expect("baseline applies");
            let base = nodes[leader].status().await.applied.get();
            wait_applied(&nodes, base).await;
            partition_leader(&nodes, leader).await;
            // The isolated leader appends an uncommitted suffix entry: it
            // lands in its local log (last_log advances) but can never
            // commit without quorum. The proposal goes out immediately —
            // well inside the election timeout — while the node still
            // considers itself leader; if it already stepped down or
            // campaigns, the scenario cannot start and the test reports
            // that instead of hanging.
            assert_eq!(
                nodes[leader].status().await.role,
                crate::types::ReplicaRole::Leader,
                "isolated node must still lead to seed divergence"
            );
            let divergent = Operation::CounterAdd {
                key: Key::from("n"),
                delta: 100,
            };
            let pending = nodes[leader].propose(&divergent, Some(identity(0xE1, 1)), None, NOW);
            let mut appended = false;
            let mut early: Option<String> = None;
            {
                futures::pin_mut!(pending);
                let deadline = std::time::Instant::now() + Duration::from_secs(10);
                loop {
                    if nodes[leader].status().await.last_log.get() > base {
                        appended = true;
                        break;
                    }
                    if let Some(settled) =
                        futures::future::FutureExt::now_or_never(pending.as_mut())
                    {
                        // Settled without appending (stepped down or
                        // campaigning mid-scenario): report instead of hanging.
                        early = Some(format!("{settled:?}"));
                        break;
                    }
                    if std::time::Instant::now() >= deadline {
                        break;
                    }
                    compio::time::sleep(Duration::from_millis(50)).await;
                }
            }
            assert!(
                appended,
                "suffix entry must append locally, never commit (early: {early:?})"
            );
            // The majority elects and commits a divergent entry.
            let majority = wait_leader_except(&nodes, leader).await;
            nodes[majority]
                .propose(
                    &Operation::CounterAdd {
                        key: Key::from("n"),
                        delta: 10,
                    },
                    Some(identity(0xE2, 1)),
                    None,
                    NOW,
                )
                .await
                .expect("majority commits divergently");
            // Heal: the authoritative log replaces the stale suffix and
            // every replica converges on the majority history.
            heal_all(&nodes).await;
            wait_applied(&nodes, nodes[majority].status().await.applied.get()).await;
            for node in &nodes {
                let value = node
                    .read(
                        &Operation::CounterGet {
                            key: Key::from("n"),
                        },
                        ReadContract::Any,
                        NOW,
                    )
                    .await
                    .expect("converged read");
                assert_eq!(
                    value,
                    OperationResult::Counter(Some(11)),
                    "divergent +100 must not survive anywhere"
                );
            }
            // The replacement went through the persisted Kivi truncate
            // record — not a memory-only path.
            assert!(
                nodes[leader].store.cached_truncate_count().await >= 1,
                "stale suffix replacement must record a durable truncate"
            );
            // The record survives a restart: reopening replays the marker
            // and converges identically.
            nodes[leader].shutdown().await;
            drop(nodes);
            let reopened = restart(&dirs, &topology, leader).await;
            assert!(
                reopened.store.cached_truncate_count().await >= 1,
                "truncate record must survive restart"
            );
            reopened.shutdown().await;
        });
    }

    /// Snapshot seals, purge advances the logical base through the
    /// durable record, and the base survives a restart.
    #[test]
    fn snapshot_purge_and_restart() {
        block_on(async {
            let (nodes, dirs, topology) = cluster().await;
            let leader = wait_leader(&nodes).await;
            for (session, value) in [0xB1u128, 0xB2, 0xB3, 0xB4, 0xB5]
                .into_iter()
                .zip([1i64, 2, 3, 4, 5])
            {
                nodes[leader]
                    .propose(
                        &Operation::CounterAdd {
                            key: Key::from("n"),
                            delta: value,
                        },
                        Some(identity(session, 1)),
                        None,
                        NOW,
                    )
                    .await
                    .expect("writes apply");
            }
            // Purge trails replication: wait for every replica to apply the
            // writes first.
            let last = nodes[leader].status().await.last_log.get();
            wait_applied(&nodes, last).await;
            let base = nodes[leader]
                .snapshot_and_purge()
                .await
                .expect("snapshot and purge");
            assert!(base.get() >= 5, "purge covers the writes");
            // The purge base is visible in diagnostics and the tail still
            // serves: one more write commits past it.
            let status = nodes[leader].status().await;
            assert_eq!(status.purged, Some(base));
            nodes[leader]
                .propose(
                    &Operation::CounterAdd {
                        key: Key::from("n"),
                        delta: 100,
                    },
                    Some(identity(0xC0, 1)),
                    None,
                    NOW,
                )
                .await
                .expect("post-purge write applies");
            // Restart the cluster from its directories: the purge record
            // replays, the snapshot base holds, and new writes succeed.
            // Every node shuts down first (dropping a live node would leak
            // its owner thread and its bound peer port).
            let incarnation = nodes[leader].incarnation();
            for node in &nodes {
                node.shutdown().await;
            }
            drop(nodes);
            let mut fresh = Vec::new();
            for (index, _dir) in dirs.iter().enumerate() {
                fresh.push(restart(&dirs, &topology, index).await);
            }
            assert!(
                fresh[leader].incarnation().as_u64() > incarnation.as_u64(),
                "leader incarnation advances"
            );
            let elected = wait_leader(&fresh).await;
            fresh[elected]
                .propose(
                    &Operation::CounterAdd {
                        key: Key::from("n"),
                        delta: 1000,
                    },
                    Some(identity(0xC1, 1)),
                    None,
                    NOW,
                )
                .await
                .expect("post-restart write applies");
            let value = fresh[elected]
                .read(
                    &Operation::CounterGet {
                        key: Key::from("n"),
                    },
                    ReadContract::Latest,
                    NOW,
                )
                .await
                .expect("post-restart latest reads");
            // 1+2+3+4+5+100+1000, exactly once each.
            assert_eq!(value, OperationResult::Counter(Some(1115)));
            for node in &fresh {
                node.shutdown().await;
            }
        });
    }
}
