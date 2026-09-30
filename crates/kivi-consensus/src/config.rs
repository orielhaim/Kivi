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
//! * Version: an exact `=` pin to a published `0.10.0-alpha.x`. Pre-1.0
//!   alpha APIs shift between releases, so caret drift is unacceptable.
//! * Features: `default-features = false` (no Tokio runtime, no clap) plus
//!   `single-threaded`. The latter empties `OptionalSend`/`OptionalSync`, so
//!   every `Raft` handle and every consensus future is `!Send`: Raft tasks
//!   are pinned to the Compio reactor thread that owns them. That is not a
//!   limitation to work around - it is the type-level enforcement of Kivi's
//!   single-owner invariant.
//!
//! ## Configurable-point decisions
//!
//! * `AsyncRuntime = CompioRuntime` (official): Raft tasks park on Kivi's
//!   Compio reactors - no isolated Tokio runtime, no cross-runtime bridge
//!   channels. Evidence: `examples/density.rs`, whose three-voter mode
//!   elects, replicates, and reports idle CPU on a pure-Compio tree.
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
//! * `SnapshotData` lives on [`openraft::storage::RaftStateMachine`] (Kivi
//!   checkpoint bytes) and on [`openraft::network::RaftNetworkV2`] (the same
//!   bytes on the wire).
//! * `reset_backoff_on_transfer_leader = Some(true)`: Kivi's only leadership
//!   moves are graceful (migration handoff, rebalance), and their targets are
//!   nodes that just joined or restarted - exactly the nodes the leader is
//!   backing off from. A transfer is one broadcast that a target which has not
//!   caught up refuses, so the intent is to clear that backoff first rather
//!   than inherit a delay chosen while the target was unreachable. `None`
//!   would behave identically; the explicit `Some(true)` records the reset as
//!   the intent rather than an accident of the default.

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
        reset_backoff_on_transfer_leader: Some(true),
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
        reset_backoff_on_transfer_leader: Some(true),
        ..openraft::Config::default()
    }
    .validate()
    .map(Arc::new)
}

#[cfg(test)]
mod tests {
    use super::{cluster_config, spike_config};

    /// Pre-vote is on for both shapes: a partitioned node must not
    /// disrupt the incumbent term.
    #[test]
    fn pre_vote_is_enabled_for_every_config() {
        for config in [
            cluster_config().expect("cluster config"),
            spike_config().expect("spike config"),
        ] {
            assert_eq!(config.enable_pre_vote, Some(true));
        }
    }

    /// Both shapes reset a transfer target's replication backoff. A
    /// transfer is one broadcast, so a target still inside a backoff has
    /// not caught up and refuses it - the handoff silently never lands.
    #[test]
    fn leadership_transfer_resets_the_targets_backoff() {
        for config in [
            cluster_config().expect("cluster config"),
            spike_config().expect("spike config"),
        ] {
            assert_eq!(config.reset_backoff_on_transfer_leader, Some(true));
        }
    }
}
