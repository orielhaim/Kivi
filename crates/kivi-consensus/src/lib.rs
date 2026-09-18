//! Kivi consensus boundary: Kivi-owned replicated-tablet concepts over an
//! `OpenRaft` implementation detail.
//!
//! ## Isolation rule
//!
//! No `OpenRaft` type may leak into `kivi-state`, `kivi-tablet`,
//! `kivi-protocol`, `kivi-resp`, `kivi-client`, `kivi-chunk`, or
//! `MutationIR`. `OpenRaft` lives in [`config`], [`router`], [`spike`],
//! [`state_machine`], [`store`], [`shared`], [`transport`], and the owner
//! halves of [`node`] and [`multi`] (the private containment zone, and
//! only in private fields there — every public signature outside
//! [`config`] is Kivi-owned); everything else in the workspace talks to
//! the node fronts ([`ReplicatedNode`], [`ConsensusNode`]) and the owned
//! types in [`types`]. `OpenRaft`'s API is explicitly pre-1.0 and
//! unstable, so this crate absorbs its churn behind those fronts.
//!
//! ## `OpenRaft` selection record
//!
//! * Version: `openraft =0.10.0-alpha.34`, exact-pinned — verified live on
//!   crates.io as the newest published `0.10.0-alpha.x` (the `0.10` line is
//!   still alpha; see [`config`] for the full record including known alpha
//!   instability).
//! * Runtime: the official `openraft-rt-compio` `AsyncRuntime`. There is no
//!   isolated Tokio runtime in this crate: `cargo tree -p kivi-consensus`
//!   must show no Tokio. `OpenRaft` tasks park on Kivi's Compio reactors.
//!
//! ## Runtime architecture: owner thread + channel front
//!
//! 0.10's `single-threaded` mode makes every `Raft` handle `!Send`: the
//! handle is pinned to the Compio reactor thread that owns it, which is
//! the type-level enforcement of Kivi's single-owner invariant. Day one,
//! [`ReplicatedNode`] spawns one Compio owner thread
//! per node driving its groups; the public [`ReplicatedNode`] front stays
//! `Send + Sync` (bounded [`async_channel`] requests, `Send` oneshot
//! replies), so callers on any runtime — including the Tokio admin plane —
//! keep ordinary `Send` futures. Moving a group onto a `DataWorker`
//! reactor later changes no `RaftTypeConfig`: it is the same
//! `AsyncRuntime` everywhere.
//!
//! ## Density question
//!
//! Whether `OpenRaft`'s "one `RaftCore` task per group, one replication
//! task per target on the leader" model scales to thousands of mostly-idle
//! tablet groups is measured, not assumed: see `examples/density.rs`.
//! Shared physical routing ([`router`]) means the per-group cost is Raft
//! tasks plus membership in the shared mesh — never a connection per
//! group.

pub mod alr;
pub mod cluster;
pub mod command;
pub mod config;
pub mod consistency;
pub mod control;
pub mod gate;
pub mod multi;
pub mod mutation;
pub mod node;
pub mod peer;
pub mod preflight;
pub mod router;
pub mod shared;
pub mod sidecar;
pub mod spike;
pub mod state_machine;
pub mod store;
pub mod tls;
pub mod transport;
pub mod types;
pub mod worker;

pub use alr::AlrCoordinator;
pub use cluster::{
    Bootstrap, BootstrapError, ClusterTopology, DurableClusterView, NodeDescriptor,
    TabletAssignment, TopologyError, classify_bootstrap, verify_against_durable,
};
pub use command::{
    ALR_FENCE_SYNC_VERSION, AlrFenceSync, CONSENSUS_COMMAND_VERSION, ConsensusCommand,
    ConsensusCommandError,
};
pub use consistency::{
    CachedServe, ConsistencyHub, LeaseDriveOut, PlannedRead, ReadPlan, ServedRead, ServedScan,
};
pub use control::{
    CATCH_UP_LAG_THRESHOLD, ObservationAuthority, ObservedMembership, ReconcileStep, caught_up,
    next_step, observation_authority, select_authoritative,
};
pub use gate::{SidecarGate, sidecar_io_error};
pub use multi::{ConsensusNode, ControlGroupConfig, MultiNodeConfig};
pub use mutation::{
    REPLICATED_MUTATION_VERSION, ReplicatedMutation, ReplicatedMutationError, ReplicatedOutcome,
};
pub use node::{
    NodeConfig, NodeOpenError, NodeStatus, ProposeError, ProposeOutcome, ReadError, ReplicatedNode,
};
pub use peer::{
    AuthReject, CAP_CONSENSUS_V2, CAP_SIDECAR_V1, DecodedH3Request, IncarnationTable,
    LEASE_BATCH_CAP, LEASE_BATCH_VERSION, LEASE_ROSTER_CAP, PeerAlrSyncRequest,
    PeerAlrSyncResponse, PeerAppendRequest, PeerAppendResponse, PeerChunkRequest,
    PeerChunkResponse, PeerCoveragePoll, PeerCoverageReport, PeerEntry, PeerEntryPayload,
    PeerIdentity, PeerLeaseBatch, PeerLogId, PeerManifestRequest, PeerManifestResponse,
    PeerPrepareRequest, PeerPreparedResponse, PeerRequest, PeerResponse, PeerRpcError,
    PeerSnapshotMeta, PeerSnapshotRequest, PeerSnapshotResponse, PeerVote, PeerVoteRequest,
    PeerVoteResponse, REQUIRED_CAPABILITIES, ValidatedPeer, decode_h3_request, decode_h3_response,
    encode_h3_request, encode_h3_response, encode_lease_message, h3_media_type, h3_method,
    h3_request_path, id32_hex, parse_id32_hex, route, validate_peer_headers,
};
pub use preflight::{
    PreflightMetrics, PreflightMetricsSnapshot, PreflightOutcome, followers_needed,
    preflight_to_quorum, quorum_needed,
};
pub use router::{
    GroupNetworkFactory, PeerRouter, RpcCodecError, decode_append_request, decode_snapshot_meta,
    decode_vote_request, decode_wire_entry, decode_wire_log, decode_wire_vote,
    encode_append_request, encode_snapshot_meta, encode_vote_request, encode_wire_entry,
    encode_wire_log, encode_wire_vote, serve_peer_request,
};
pub use shared::{
    GroupLogReader, GroupRaftStore, SharedDurabilityConfig, SharedDurabilityMetrics,
    SharedDurabilityStats, SharedOpenError, SharedRaftDurability, dispatch_by_group,
};
pub use sidecar::{
    ImmutableDependencies, PinStage, ProposalPin, SidecarError, SidecarMetrics,
    SidecarMetricsSnapshot, SidecarPins, SidecarStore, StagedRoot, build_manifest_for,
    chunk_id_for, split_value,
};
pub use state_machine::{
    ANONYMOUS_SESSION, AppliedPointer, DecodedSnapshot, ProposalGate, RangeBase,
    ReplicatedStateMachine, ReplicatedTablet, RetainedOutcome, SESSION_OUTCOME_CAP,
    SealedSnapshotBuilder, SessionState, StateMachineFault, StateMachineOpenError,
    StateMachineStatus, commit_of_index, index_of_commit,
};
pub use store::{
    CONSENSUS_LANE, ConsensusLogStore, DurableLogReader, DurableRaftStore, StoreOpenError,
};
pub use tls::{CertFingerprint, NodeCert, TlsError, TrustRegistry, empty_trust};
pub use transport::{PeerHandler, PeerStats, PeerTransport, TransportConfig, TransportError};
pub use types::{
    CONTROL_TABLET_RAW, ConsensusError, ConsensusGroupId, ConsensusLogIndex, ConsensusTerm,
    LeaderHint, ReadBarrier, ReplicaId, ReplicaRole,
};
pub use worker::{tablets_for_worker, worker_for_tablet};
