//! Kivi consensus boundary: Kivi-owned replicated-tablet concepts over an
//! `OpenRaft` implementation detail.
//!
//! ## Isolation rule
//!
//! No `OpenRaft` type may leak into `kivi-state`, `kivi-tablet`,
//! `kivi-protocol`, `kivi-resp`, `kivi-client`, `kivi-chunk`, or
//! `MutationIR`. `OpenRaft` reaches the workspace only through the four
//! modules declared `pub` below - [`config`], [`router`], [`spike`], and
//! [`distcoord`] - which exist for the density experiment and the
//! redundancy coordinator, not for ordinary node wiring. Every other
//! module is private, so the rest of the workspace talks to the node
//! fronts ([`ReplicatedNode`], [`ConsensusNode`]) and the owned types
//! re-exported here. `OpenRaft`'s API is explicitly pre-1.0 and unstable,
//! so this crate absorbs its churn behind those fronts.
//!
//! ## `OpenRaft` selection record
//!
//! * Version: an exact `=` pin to a published `0.10.0-alpha.x`. Pre-1.0
//!   alpha APIs shift between releases, so caret drift is unacceptable.
//! * Runtime: the official Compio `AsyncRuntime`. There is no isolated Tokio
//!   runtime in this crate: `cargo tree -p kivi-consensus` must show no
//!   Tokio runtime crate. `OpenRaft` tasks park on Kivi's Compio reactors.
//! * `single-threaded` empties `OptionalSend`/`OptionalSync`, so every
//!   `Raft` handle is `!Send`: that is the type-level enforcement of the
//!   single-owner invariant, not a limitation to work around.
//!
//! ## Runtime architecture: owner thread + channel front
//!
//! [`ReplicatedNode`] spawns one Compio owner thread per node driving its
//! groups; the public front stays `Send + Sync` (bounded [`async_channel`]
//! requests, `Send` oneshot replies), so callers on any runtime -
//! including the Tokio admin plane - keep ordinary `Send` futures.
//!
//! ## Density question
//!
//! Whether `OpenRaft`'s "one `RaftCore` task per group, one replication
//! task per target on the leader" model scales to thousands of mostly-idle
//! tablet groups is measured, not assumed: see `examples/density.rs`.
//! Shared physical routing ([`router`]) means the per-group cost is Raft
//! tasks plus membership in the shared mesh - never a connection per
//! group.

mod alr;
mod cluster;
mod command;
mod consistency;
mod control;
mod fragments;
mod gate;
mod multi;
mod mutation;
mod node;
mod peer;
mod preflight;
mod resolve;
mod shared;
mod sidecar;
mod state_machine;
mod store;
mod transport;
mod types;
mod worker;

/// QUIC/H3 peer TLS identity (self-signed node certificates, fingerprint
/// pinning). Public so the operator tooling can mint a node's certificate
/// without reaching into this crate's internals.
pub mod tls;

/// `OpenRaft`-typed surfaces, reachable only for the density experiment
/// and the redundancy coordinator. Everything else in this crate is
/// Kivi-owned; see the isolation rule above.
pub mod config;
pub mod distcoord;
pub mod router;
pub mod spike;

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
pub use fragments::FragmentClient;
pub use gate::{SidecarGate, sidecar_io_error};
pub use multi::{ConsensusNode, ControlGroupConfig, MultiNodeConfig};
pub use mutation::{
    REPLICATED_MUTATION_VERSION, ReplicatedMutation, ReplicatedMutationError, ReplicatedOutcome,
};
pub use node::{
    NodeConfig, NodeOpenError, NodeStatus, ProposeError, ProposeOutcome, ReadError, ReplicatedNode,
};
pub use resolve::{
    ProtectedRoot, ResolutionSlots, ResolveError, ResolveFuture, SidecarProtector, SidecarResolver,
    protect_staged_root,
};
pub use shared::{SharedDurabilityConfig, SharedDurabilityMetrics, SharedOpenError};
pub use sidecar::{SidecarError, SidecarMetricsSnapshot, SidecarStore, StagedRoot};
pub use state_machine::{
    ANONYMOUS_SESSION, AppliedPointer, ProposalGate, RangeBase, SESSION_OUTCOME_CAP,
    StateMachineFault, commit_of_index, index_of_commit,
};
pub use tls::{CertFingerprint, NodeCert, TrustRegistry};
pub use transport::{PeerStats, TransportConfig};
pub use types::{
    CONTROL_TABLET_RAW, ConsensusError, ConsensusGroupId, ConsensusLogIndex, ConsensusTerm,
    ControlProposeError, LeaderHint, ReadBarrier, ReplicaId, ReplicaRole,
};
pub use worker::worker_for_tablet;
