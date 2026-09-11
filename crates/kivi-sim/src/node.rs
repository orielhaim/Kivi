//! Deterministic simulated node/process lifecycle.
//!
//! Each node keeps a persistent [`NodeId`] across restarts while its
//! [`NodeIncarnation`] advances on every restart (checked, never wrapping).
//! Lifecycle states are exactly [`Running`](NodeState::Running),
//! [`Crashed`](NodeState::Crashed), and [`Restarting`](NodeState::Restarting):
//!
//! ```text
//! Running → Crashed → Restarting → Running
//! ```
//!
//! A crash discards process-local volatile state (owned by the driver: app
//! state, scheduler entries addressed to the old instance, open network
//! endpoints); durable storage and the node identity survive. Restarting
//! marks a process that exists but serves nothing yet (replaying durable
//! state, rejoining); only `Running` instances authorize work.
//!
//! Stale-incarnation fencing lives here too: [`check_endpoint`](NodeTable::check_endpoint)
//! rejects traffic for obsolete incarnations, so a delayed message sent by
//! incarnation 7 can never become valid after the node restarts as 8. The
//! simulator is generic on purpose — tablets, Raft groups, control-plane
//! actors, and durability providers will all run inside it later.

use core::fmt;
use std::collections::BTreeMap;

use kivi_types::{NodeId, NodeIncarnation};

use crate::net::Endpoint;

/// Lifecycle state of one simulated process instance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NodeState {
    /// Serving work; the only state authorizing sends, receives, and app steps.
    Running,
    /// Dead: volatile state discarded, endpoints closed, nothing valid.
    Crashed,
    /// Restart begun (new incarnation minted) but not serving yet.
    Restarting,
}

impl fmt::Display for NodeState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Running => write!(f, "running"),
            Self::Crashed => write!(f, "crashed"),
            Self::Restarting => write!(f, "restarting"),
        }
    }
}

/// Identity, current incarnation, and lifecycle state of one node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeInfo {
    id: NodeId,
    incarnation: NodeIncarnation,
    state: NodeState,
}

impl NodeInfo {
    /// Returns the persistent node identity.
    #[must_use]
    pub const fn id(self) -> NodeId {
        self.id
    }

    /// Returns the current incarnation.
    #[must_use]
    pub const fn incarnation(self) -> NodeIncarnation {
        self.incarnation
    }

    /// Returns the lifecycle state.
    #[must_use]
    pub const fn state(self) -> NodeState {
        self.state
    }

    /// Returns the current addressable endpoint.
    #[must_use]
    pub const fn endpoint(self) -> Endpoint {
        Endpoint::new(self.id, self.incarnation)
    }
}

/// Node lifecycle failures: construction misuse and exhausted incarnations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, thiserror::Error)]
#[non_exhaustive]
pub enum NodeError {
    /// The node is already registered.
    #[error("duplicate node {node}")]
    DuplicateNode {
        /// Offending node.
        node: NodeId,
    },
    /// No such node is registered.
    #[error("unknown node {node}")]
    UnknownNode {
        /// Missing node.
        node: NodeId,
    },
    /// Transition requested from a state it cannot start in.
    #[error("cannot {action} node {node} while {from}")]
    IllegalTransition {
        /// Affected node.
        node: NodeId,
        /// Observed state.
        from: NodeState,
        /// Requested transition.
        action: &'static str,
    },
    /// The incarnation counter reached `u64::MAX` on restart: the node can
    /// never restart again (explicit, never a wrap to a stale identity).
    #[error("node {node} incarnation space exhausted")]
    IncarnationExhausted {
        /// Affected node.
        node: NodeId,
    },
}

/// Endpoint validation failure: why one process instance may not act.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, thiserror::Error)]
#[non_exhaustive]
pub enum EndpointError {
    /// No such node is registered.
    #[error("unknown node {node}")]
    UnknownNode {
        /// Missing node.
        node: NodeId,
    },
    /// The instance exists but is not serving.
    #[error("node {node} is {state}, not running")]
    NotRunning {
        /// Affected node.
        node: NodeId,
        /// Observed state.
        state: NodeState,
    },
    /// The incarnation is obsolete: the node has restarted since.
    /// A stale endpoint never becomes valid again.
    #[error("stale incarnation {presented} for node {node} (current {current})")]
    StaleIncarnation {
        /// Affected node.
        node: NodeId,
        /// Presented (obsolete) incarnation.
        presented: NodeIncarnation,
        /// Current incarnation.
        current: NodeIncarnation,
    },
}

/// Registry of node identities, incarnations, and lifecycle states.
#[derive(Debug, Clone, Default)]
pub struct NodeTable {
    nodes: BTreeMap<NodeId, NodeInfo>,
}

impl NodeTable {
    /// Creates an empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self {
            nodes: BTreeMap::new(),
        }
    }

    /// Registers a node at [`INITIAL`](NodeIncarnation::INITIAL) incarnation,
    /// already running.
    ///
    /// # Errors
    ///
    /// Returns [`NodeError::DuplicateNode`] if already registered.
    pub fn add_node(&mut self, node: NodeId) -> Result<(), NodeError> {
        if self.nodes.contains_key(&node) {
            return Err(NodeError::DuplicateNode { node });
        }
        self.nodes.insert(
            node,
            NodeInfo {
                id: node,
                incarnation: NodeIncarnation::INITIAL,
                state: NodeState::Running,
            },
        );
        Ok(())
    }

    /// Returns the info for one node, if registered.
    #[must_use]
    pub fn get(&self, node: NodeId) -> Option<NodeInfo> {
        self.nodes.get(&node).copied()
    }

    /// Crashes a running node, returning its obsolete endpoint so the driver
    /// can isolate it from the network and cancel its process-owned events.
    ///
    /// # Errors
    ///
    /// Returns [`NodeError`] for unknown nodes or non-running states.
    pub fn crash(&mut self, node: NodeId) -> Result<Endpoint, NodeError> {
        let info = self.info_mut(node)?;
        if info.state != NodeState::Running {
            return Err(NodeError::IllegalTransition {
                node,
                from: info.state,
                action: "crash",
            });
        }
        info.state = NodeState::Crashed;
        Ok(info.endpoint())
    }

    /// Begins a restart: `Crashed → Restarting` with a checked newer
    /// incarnation. Returns the new (not yet serving) endpoint.
    ///
    /// # Errors
    ///
    /// Returns [`NodeError`] for unknown nodes, non-crashed states, or an
    /// exhausted incarnation counter.
    pub fn begin_restart(&mut self, node: NodeId) -> Result<Endpoint, NodeError> {
        let info = self.info_mut(node)?;
        if info.state != NodeState::Crashed {
            return Err(NodeError::IllegalTransition {
                node,
                from: info.state,
                action: "begin_restart",
            });
        }
        info.incarnation = info
            .incarnation
            .next()
            .map_err(|_| NodeError::IncarnationExhausted { node })?;
        info.state = NodeState::Restarting;
        Ok(info.endpoint())
    }

    /// Completes a restart: `Restarting → Running`. The node serves again
    /// under its new incarnation.
    ///
    /// # Errors
    ///
    /// Returns [`NodeError`] for unknown nodes or non-restarting states.
    pub fn finish_restart(&mut self, node: NodeId) -> Result<Endpoint, NodeError> {
        let info = self.info_mut(node)?;
        if info.state != NodeState::Restarting {
            return Err(NodeError::IllegalTransition {
                node,
                from: info.state,
                action: "finish_restart",
            });
        }
        info.state = NodeState::Running;
        Ok(info.endpoint())
    }

    /// Validates that `endpoint` names the currently running instance:
    /// known node, `Running` state, and current incarnation.
    ///
    /// # Errors
    ///
    /// Returns [`EndpointError`] for unknown nodes, non-running instances,
    /// or obsolete incarnations (which never become valid again).
    pub fn check_endpoint(&self, endpoint: &Endpoint) -> Result<(), EndpointError> {
        let info = self
            .nodes
            .get(&endpoint.node())
            .ok_or(EndpointError::UnknownNode {
                node: endpoint.node(),
            })?;
        if info.state != NodeState::Running {
            return Err(EndpointError::NotRunning {
                node: info.id,
                state: info.state,
            });
        }
        if info.incarnation != endpoint.incarnation() {
            return Err(EndpointError::StaleIncarnation {
                node: info.id,
                presented: endpoint.incarnation(),
                current: info.incarnation,
            });
        }
        Ok(())
    }

    /// Returns the current endpoint of a registered node, if any.
    #[must_use]
    pub fn current_endpoint(&self, node: NodeId) -> Option<Endpoint> {
        self.nodes.get(&node).map(|info| info.endpoint())
    }

    /// Iterates over all registered infos in node order.
    pub fn iter(&self) -> impl Iterator<Item = NodeInfo> + '_ {
        self.nodes.values().copied()
    }

    fn info_mut(&mut self, node: NodeId) -> Result<&mut NodeInfo, NodeError> {
        self.nodes
            .get_mut(&node)
            .ok_or(NodeError::UnknownNode { node })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    fn node(n: u64) -> NodeId {
        NodeId::from_u64(n)
    }

    #[test]
    fn full_lifecycle_advances_state_and_incarnation() {
        let mut table = NodeTable::new();
        table.add_node(node(1)).expect("add");
        let first = table.current_endpoint(node(1)).expect("endpoint");
        assert_eq!(first.incarnation(), NodeIncarnation::INITIAL);

        let crashed = table.crash(node(1)).expect("crash");
        assert_eq!(crashed, first);
        assert_eq!(
            table.get(node(1)).expect("info").state(),
            NodeState::Crashed
        );

        let restarting = table.begin_restart(node(1)).expect("begin");
        assert_eq!(
            restarting.incarnation().as_u64(),
            first.incarnation().as_u64() + 1
        );
        let running = table.finish_restart(node(1)).expect("finish");
        assert_eq!(running, restarting);
        assert_eq!(
            table.get(node(1)).expect("info").state(),
            NodeState::Running
        );
    }

    #[test]
    fn stale_incarnation_never_becomes_valid_again() {
        let mut table = NodeTable::new();
        table.add_node(node(1)).expect("add");
        let old = table.current_endpoint(node(1)).expect("endpoint");
        table.crash(node(1)).expect("crash");
        // Crashed instance is not running.
        assert!(matches!(
            table.check_endpoint(&old),
            Err(EndpointError::NotRunning { .. })
        ));
        let next = table.begin_restart(node(1)).expect("begin");
        // Restarting instance is not running either.
        assert!(matches!(
            table.check_endpoint(&next),
            Err(EndpointError::NotRunning { .. })
        ));
        table.finish_restart(node(1)).expect("finish");
        // The old endpoint is now stale — permanently rejected.
        assert_eq!(
            table.check_endpoint(&old),
            Err(EndpointError::StaleIncarnation {
                node: node(1),
                presented: old.incarnation(),
                current: next.incarnation(),
            })
        );
        // The new endpoint works.
        assert_eq!(table.check_endpoint(&next), Ok(()));
        assert_eq!(
            table.check_endpoint(&old),
            Err(EndpointError::StaleIncarnation {
                node: node(1),
                presented: old.incarnation(),
                current: next.incarnation(),
            })
        );
    }

    /// Histories as (action, expect-ok) scripts over one node.
    #[derive(Debug, Clone, Copy)]
    enum Step {
        Crash,
        Begin,
        Finish,
    }

    #[rstest]
    #[case(&[Step::Crash], true, NodeState::Crashed, 1)]
    #[case(&[Step::Crash, Step::Begin, Step::Finish], true, NodeState::Running, 2)]
    #[case(&[Step::Crash, Step::Crash], false, NodeState::Crashed, 1)]
    #[case(&[Step::Begin], false, NodeState::Running, 1)]
    #[case(&[Step::Crash, Step::Finish], false, NodeState::Crashed, 1)]
    #[case(&[Step::Crash, Step::Begin, Step::Begin], false, NodeState::Restarting, 2)]
    fn lifecycle_histories(
        #[case] steps: &[Step],
        #[case] all_ok: bool,
        #[case] state: NodeState,
        #[case] incarnation: u64,
    ) {
        let mut table = NodeTable::new();
        table.add_node(node(1)).expect("add");
        let mut ok = true;
        for step in steps {
            let result = match step {
                Step::Crash => table.crash(node(1)).map(|_| ()),
                Step::Begin => table.begin_restart(node(1)).map(|_| ()),
                Step::Finish => table.finish_restart(node(1)).map(|_| ()),
            };
            ok &= result.is_ok();
        }
        assert_eq!(ok, all_ok);
        let info = table.get(node(1)).expect("info");
        assert_eq!(info.state(), state);
        assert_eq!(info.incarnation().as_u64(), incarnation);
    }

    #[test]
    fn incarnation_exhaustion_blocks_restart_explicitly() {
        let mut table = NodeTable::new();
        table.add_node(node(1)).expect("add");
        table.crash(node(1)).expect("crash");
        // Force the counter to the top through the private field (same crate).
        table.nodes.get_mut(&node(1)).expect("info").incarnation =
            NodeIncarnation::from_u64(u64::MAX);
        assert_eq!(
            table.begin_restart(node(1)),
            Err(NodeError::IncarnationExhausted { node: node(1) })
        );
        // Still crashed: no silent wrap into a stale identity.
        assert_eq!(
            table.get(node(1)).expect("info").state(),
            NodeState::Crashed
        );
    }

    #[test]
    fn duplicate_and_unknown_nodes_fail() {
        let mut table = NodeTable::new();
        table.add_node(node(1)).expect("add");
        assert_eq!(
            table.add_node(node(1)),
            Err(NodeError::DuplicateNode { node: node(1) })
        );
        assert_eq!(
            table.crash(node(9)),
            Err(NodeError::UnknownNode { node: node(9) })
        );
        let ghost = Endpoint::new(node(9), NodeIncarnation::INITIAL);
        assert_eq!(
            table.check_endpoint(&ghost),
            Err(EndpointError::UnknownNode { node: node(9) })
        );
    }
}
