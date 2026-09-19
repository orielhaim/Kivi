//! Deterministic simulation kernel for Kivi (RFC §176–§178).
//!
//! This crate implements the interfaces from `kivi-core` (`Clock`,
//! [`RandomSource`](kivi_core::RandomSource),
//! [`FaultPolicy`](kivi_core::FaultPolicy), [`StateMachine`](kivi_core::StateMachine))
//! with virtual components: a frozen-algorithm RNG ([`SimRng`]), a totally
//! ordered event [`Scheduler`], built-in fault policies, and a replayable
//! [`Trace`].
//!
//! Determinism contract: identical initial state, seed, scheduled events,
//! and fault policy produce identical event order, virtual timestamps,
//! random decisions, fault decisions, trace, and final state — across
//! repeated runs in one binary and across dependency upgrades (the kernel
//! owns its algorithms; no behavior flows from external crate versions).
//!
//! Banned as simulation inputs: system time, thread-local RNG, hash-map
//! randomized iteration order (only ordered maps), OS scheduling, and thread
//! races. The kernel is synchronous and single-threaded; there is no async
//! executor here and none is planned at this layer.
//!
//! Future virtual network/storage/node simulators plug in as *drivers*: they
//! own their event payloads and state machines, schedule work on
//! [`Scheduler`], answer fate through [`FaultPolicy`](kivi_core::FaultPolicy),
//! and record to [`Trace`]. The kernel never names their domains.

pub mod cluster;
pub mod faults;
pub mod net;
pub mod node;
pub mod rng;
pub mod scheduler;
pub mod storage;
pub mod trace;
pub mod txn;

pub use cluster::{
    AppHandler, ClusterAction, ClusterError, ClusterEvent, ClusterTrace, DropKind,
    InvariantFailure, NodeStep, RunStats, SIM_FORMAT_VERSION, ScenarioResult, SimCluster,
    StepOutcome,
};
pub use faults::{AllowAll, DropEveryNth, DropWithProbability, MAX_PPM};
pub use net::{
    Endpoint, LinkConfig, LinkStatus, MsgId, MsgIdExhausted, NetDelivery, NetDeliveryContext,
    NetDropReason, NetError, NetSendOutcome, NetSim,
};
pub use node::{EndpointError, NodeError, NodeInfo, NodeState, NodeTable};
pub use rng::SimRng;
pub use scheduler::{ScheduleError, Scheduled, Scheduler};
pub use storage::{StorageError, StorageFault, StorageOp, StorageResult, VirtualDisk};
pub use trace::{RecordedDecision, Trace, TraceEntry};
pub use txn::{
    AppliedWrite, TxnDecision, TxnDigest, TxnDriverEv, TxnIntent, TxnKey, TxnMsg, TxnSim, TxnSimId,
    TxnWrite, assert_quiesced_atomic, decode_msg, encode_msg, invariant_no_guess,
    invariant_no_partial_commit, invariant_no_split_decision,
};
