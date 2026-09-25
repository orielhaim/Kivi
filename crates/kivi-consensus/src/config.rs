//! `OpenRaft` containment zone: the project-owned 0.10 type configuration.
//!
//! Everything in this module (and in [`crate::store`], [`crate::shared`],
//! [`crate::state_machine`], [`crate::transport`], [`crate::router`], and
//! the owner halves of [`crate::node`] and [`crate::multi`]) names
//! `OpenRaft` types. Nothing outside this crate may do so: the rest of
//! the workspace sees only [`crate::types`] and the node fronts.
//! `OpenRaft`'s API is explicitly pre-1.0 and unstable, so this crate
//! absorbs its churn behind those fronts.
//!
//! ## `OpenRaft` selection record
//!
//! * Version: `openraft =0.10.0-alpha.35` — the latest published `0.10.0-alpha.x`
//!   release at the time of this audit. Pinned with an exact `=` requirement:
//!   pre-1.0 alpha APIs shift between releases, so caret drift is unacceptable.
//!   The release adds the dedicated pre-vote RPC and keeps the pre-1.0 warning.
//! * Features: `default-features = false` (no Tokio runtime, no clap) plus
//!   `single-threaded`. The latter empties `OptionalSend`/`OptionalSync`, so
//!   every `Raft` handle and every consensus future is `!Send`: Raft tasks
//!   are pinned to the Compio reactor thread that owns them. That is not a
//!   limitation to work around — it is the type-level enforcement of Kivi's
//!   single-owner invariant (RFC §16).
//!
//! ## Configurable-point decisions
//!
//! * `AsyncRuntime = CompioRuntime` (`openraft-rt-compio`, official): Raft
//!   tasks park on Kivi's Compio reactors — no isolated Tokio island, no
//!   cross-runtime bridge channels. Evidence: `examples/compio_smoke.rs`
//!   prototype (single voter elects, writes, serves a `ReadIndex` barrier on
//!   a pure-Compio tree with zero Tokio in `cargo tree`).
//! * `Batch<T>`: official default `InlineBatch` (inline single element,
//!   heap on spill). No custom implementation: Kivi's batching story is the
//!   physical Multi-Raft durability batching on the WAL lane, which lives
//!   below this API; nothing measured justifies a custom in-memory batch.
//! * `Responder<T>`: official default `ProgressResponder` (commit + complete
//!   channels). No custom implementation: the owner-thread front answers
//!   through its own `Send` oneshots, and the state machine uses the
//!   streaming `ApplyResponder::send` path directly.
//! * `ErrorSource`: official default `BoxedErrorSource`. Kivi's typed domain
//!   errors cross the crate boundary as [`crate::types::ConsensusError`],
//!   never as storage sources; inside storage, `io::Error` (0.10's storage
//!   error type) carries enough detail, and a custom source would only save
//!   one heap allocation per fault path.
//! * `Term`/`LeaderId`/`Vote`/`Entry`: official defaults (`u64`,
//!   `leader_id_adv::LeaderId`, `Vote`, `Entry`). Kivi's wire and durable
//!   records keep their own `(term, leader)` vocabulary (see [`crate::peer`]
//!   and `kivi-durability`'s `RaftRecord`); adopting `OpenRaft`'s generic vote
//!   machinery would leak its layout into Kivi-owned formats for no gain.
//! * `SnapshotData`: no longer a `RaftTypeConfig` member in 0.10. It lives on
//!   [`openraft::storage::RaftStateMachine`] (Kivi checkpoint bytes) and on
//!   [`openraft::network::RaftNetworkV2`] (the same bytes on the wire).

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use openraft::BasicNode;

openraft::declare_raft_types!(
    /// Project-owned OpenRaft type configuration.
    ///
    /// `D` is the Kivi-owned [`ConsensusCommand`](crate::command::ConsensusCommand)
    /// envelope (deterministic tablet commands plus typed control-plane
    /// mutations over one shared log implementation); `R` is its installed
    /// [`ReplicatedOutcome`](crate::mutation::ReplicatedOutcome).
    /// `AsyncRuntime` is the official Compio runtime: consensus tasks run on
    /// Kivi's Compio reactors. Every other associated type takes the official
    /// default (see the module docs for why each default stands).
    pub KiviTypeConfig:
        D = crate::command::ConsensusCommand,
        R = crate::mutation::ReplicatedOutcome,
        NodeId = u64,
        Node = BasicNode,
        AsyncRuntime = openraft_rt_compio::CompioRuntime,
);

/// Production Raft config for one replicated tablet group: election and
/// heartbeat timeouts shaped for commodity networks and loaded CI boxes
/// (multi-second failure detection without flapping under parallel test
/// load). Snapshots stay enabled (the state machine seals them; purge
/// follows the snapshot base).
///
/// # Errors
///
/// Returns the config violation when the bounds are incoherent (cannot
/// happen for these literals; the `Result` is `OpenRaft`'s API, not ours).
#[allow(clippy::result_large_err)]
pub fn cluster_config() -> Result<Arc<openraft::Config>, openraft::ConfigError> {
    openraft::Config {
        heartbeat_interval: 300,
        election_timeout_min: 1500,
        election_timeout_max: 3000,
        max_in_snapshot_log_to_keep: 100,
        enable_pre_vote: Some(true),
        ..openraft::Config::default()
    }
    .validate()
    .map(Arc::new)
}

/// Validated spike Raft config: production-shaped timeouts, snapshots
/// disabled (the spike measures group overhead, not compaction). Shared
/// because every group on the node runs the same config.
///
/// # Errors
///
/// Returns the config violation when the bounds are incoherent (cannot
/// happen for these literals; the `Result` is `OpenRaft`'s API, not ours).
#[allow(clippy::result_large_err)]
pub fn spike_config() -> Result<Arc<openraft::Config>, openraft::ConfigError> {
    openraft::Config {
        heartbeat_interval: 500,
        election_timeout_min: 1000,
        election_timeout_max: 2000,
        enable_pre_vote: Some(true),
        ..openraft::Config::default()
    }
    .validate()
    .map(Arc::new)
}

/// Builds the `OpenRaft` membership value for a static voter set: one voter
/// group plus every voter's dialable peer address (the address rides as the
/// node's `BasicNode::addr`, which is also what the durable membership
/// record persists).
///
/// # Errors
///
/// Returns the membership violation when a voter has no dialable node (a
/// static-topology construction bug, never a runtime condition).
#[allow(clippy::result_large_err)]
pub fn static_membership(
    voters: &BTreeSet<u64>,
    nodes: &BTreeMap<u64, String>,
) -> Result<openraft::Membership<u64, BasicNode>, openraft::error::MembershipError<u64>> {
    let indexed: BTreeMap<u64, BasicNode> = nodes
        .iter()
        .map(|(id, addr)| (*id, BasicNode::new(addr.clone())))
        .collect();
    openraft::Membership::new(vec![voters.clone()], indexed)
}

#[cfg(test)]
mod tests {
    use super::{cluster_config, spike_config};

    #[test]
    fn pre_vote_is_enabled_for_production_and_spike_configs() {
        assert_eq!(
            cluster_config().expect("cluster config").enable_pre_vote,
            Some(true)
        );
        assert_eq!(
            spike_config().expect("spike config").enable_pre_vote,
            Some(true)
        );
    }
}
