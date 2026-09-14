//! Static distributed topology and durable bootstrap semantics
//! (tasks F, G).
//!
//! The first replicated deployment is one statically configured cluster:
//!
//! ```text
//! ClusterId
//!   Node { NodeId, peer endpoint, native endpoint } × N
//!   Tablet { TabletId, replicas: [NodeId] }
//! ```
//!
//! Supported initial topology: 1 replicated tablet × 3 voting replicas,
//! one replica per node. The durable format never hardcodes the replica
//! count: voters are an ordered set, so larger or smaller static groups
//! (and later multi-tablet groups) reuse the same types.
//!
//! Out of scope: learners, add/remove voter, placement, migration,
//! split/merge, control-plane Raft (task AL).
//!
//! ## Bootstrap semantics (task G)
//!
//! Membership installs exactly once. First formation persists the
//! `OpenRaft` membership entry through the normal log path and the group
//! forms; restarts recover existing Raft state and must NOT initialize a
//! new group. Startup config conflicting with durable identity or
//! membership fails loudly.
//!
//! [`classify_bootstrap`] decides fresh vs. existing from durable state
//! (vote, log, snapshot base — all three empty means fresh). Every node in
//! a fresh static cluster calls `initialize` on its own `Raft` instance
//! with the joint membership; `OpenRaft`'s contract makes a repeated local
//! `initialize` with identical membership benign (already-initialized),
//! which is the race the density spike met: concurrent local
//! initialization attempts converge instead of forking groups.
//!
//! [`verify_against_durable`] enforces restart discipline: cluster and
//! node identity must match the data directory, and the configured voter
//! set must equal the durable membership. Endpoint addresses may change
//! across restarts (operator port remapping); identities may not.

use std::collections::{BTreeMap, BTreeSet};
use std::net::SocketAddr;

use kivi_types::{ClusterId, NodeId, TabletId};

/// One cluster member: its stable identity plus its dialable endpoints.
/// `NodeId` survives restarts; endpoints are runtime dial info.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NodeDescriptor {
    /// Stable process identity (also the Raft voter id).
    pub node: NodeId,
    /// Dedicated peer endpoint for consensus traffic (never the native
    /// client endpoint, never RESP, never admin HTTP).
    pub peer: SocketAddr,
    /// Native client endpoint served by this node.
    pub native: SocketAddr,
}

/// One replicated tablet and its ordered voter set.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TabletAssignment {
    /// Tablet replicated.
    pub tablet: TabletId,
    /// Ordered voters (replication and quorum order follow this set).
    pub replicas: Vec<NodeId>,
}

/// Static cluster definition: identity, members, and tablet placement.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClusterTopology {
    /// Cluster identity (peers reject foreign clusters at handshake).
    pub cluster: ClusterId,
    /// All members (voters and dial info).
    pub nodes: Vec<NodeDescriptor>,
    /// Replicated tablets (one entry in this stage).
    pub tablets: Vec<TabletAssignment>,
}

/// Why a static topology was rejected.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum TopologyError {
    /// No members configured.
    #[error("cluster topology has no nodes")]
    NoNodes,
    /// No tablet assigned.
    #[error("cluster topology has no tablet assignment")]
    NoTablets,
    /// Two nodes share an identity.
    #[error("duplicate node identity {node}")]
    DuplicateNode {
        /// The repeated identity.
        node: u64,
    },
    /// Two nodes share a peer endpoint.
    #[error("duplicate peer endpoint {endpoint}")]
    DuplicatePeer {
        /// The repeated endpoint.
        endpoint: SocketAddr,
    },
    /// A tablet names no replicas.
    #[error("tablet {tablet} names no replicas")]
    EmptyReplicas {
        /// The affected tablet.
        tablet: u64,
    },
    /// A tablet repeats a voter.
    #[error("tablet {tablet} repeats voter {node}")]
    DuplicateVoter {
        /// The affected tablet.
        tablet: u64,
        /// The repeated voter.
        node: u64,
    },
    /// A tablet names an unknown node.
    #[error("tablet {tablet} names unknown node {node}")]
    UnknownNode {
        /// The affected tablet.
        tablet: u64,
        /// The unknown voter.
        node: u64,
    },
}

impl ClusterTopology {
    /// Validates the static definition: identities unique, endpoints
    /// unique, every replica known. The replica count itself is NOT fixed
    /// here (durable formats stay count-agnostic); callers enforcing the
    /// first deployment's 1×3 shape check [`Self::tablets`] explicitly.
    ///
    /// # Errors
    ///
    /// Returns [`TopologyError`] on any structural incoherence.
    pub fn validate(&self) -> Result<(), TopologyError> {
        use TopologyError as Fault;
        if self.nodes.is_empty() {
            return Err(Fault::NoNodes);
        }
        if self.tablets.is_empty() {
            return Err(Fault::NoTablets);
        }
        let mut seen_nodes = BTreeSet::new();
        let mut seen_peers = BTreeSet::new();
        for node in &self.nodes {
            if !seen_nodes.insert(node.node.as_u64()) {
                return Err(Fault::DuplicateNode {
                    node: node.node.as_u64(),
                });
            }
            if !seen_peers.insert(node.peer) {
                return Err(Fault::DuplicatePeer {
                    endpoint: node.peer,
                });
            }
        }
        for tablet in &self.tablets {
            if tablet.replicas.is_empty() {
                return Err(Fault::EmptyReplicas {
                    tablet: tablet.tablet.as_u64(),
                });
            }
            let mut seen_voters = BTreeSet::new();
            for replica in &tablet.replicas {
                if !seen_nodes.contains(&replica.as_u64()) {
                    return Err(Fault::UnknownNode {
                        tablet: tablet.tablet.as_u64(),
                        node: replica.as_u64(),
                    });
                }
                if !seen_voters.insert(replica.as_u64()) {
                    return Err(Fault::DuplicateVoter {
                        tablet: tablet.tablet.as_u64(),
                        node: replica.as_u64(),
                    });
                }
            }
        }
        Ok(())
    }

    /// Finds the descriptor for one node.
    #[must_use]
    pub fn node(&self, node: NodeId) -> Option<&NodeDescriptor> {
        self.nodes.iter().find(|descriptor| descriptor.node == node)
    }

    /// Finds the assignment for one tablet.
    #[must_use]
    pub fn assignment(&self, tablet: TabletId) -> Option<&TabletAssignment> {
        self.tablets
            .iter()
            .find(|assignment| assignment.tablet == tablet)
    }

    /// Builds the `OpenRaft`-shaped membership for one tablet: the voter
    /// set plus every voter's dialable peer address. The adapter owns the
    /// final conversion into `OpenRaft` types; this stays Kivi-owned.
    ///
    /// # Errors
    ///
    /// Returns [`TopologyError::UnknownNode`] when the tablet is
    /// unassigned (unknown tablets never mint membership).
    pub fn membership_for(
        &self,
        tablet: TabletId,
    ) -> Result<(BTreeSet<u64>, BTreeMap<u64, String>), TopologyError> {
        let assignment = self.assignment(tablet).ok_or(TopologyError::UnknownNode {
            tablet: tablet.as_u64(),
            node: 0,
        })?;
        let mut voters = BTreeSet::new();
        let mut nodes = BTreeMap::new();
        for replica in &assignment.replicas {
            voters.insert(replica.as_u64());
            let descriptor = self.node(*replica).ok_or(TopologyError::UnknownNode {
                tablet: tablet.as_u64(),
                node: replica.as_u64(),
            })?;
            nodes.insert(replica.as_u64(), descriptor.peer.to_string());
        }
        Ok((voters, nodes))
    }
}

/// Fresh vs. existing group, decided from durable state (task G).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Bootstrap {
    /// No vote, no log entries, no snapshot base: first formation may
    /// install membership exactly once.
    Fresh,
    /// Durable Raft state exists: recover it, never re-initialize.
    Existing,
}

/// Decides fresh vs. existing from the three durable signals. All three
/// empty means fresh; ANY durable trace means existing (a vote without
/// entries is still a promise to a leader and must survive).
#[must_use]
pub const fn classify_bootstrap(
    has_vote: bool,
    has_log_entries: bool,
    has_snapshot_base: bool,
) -> Bootstrap {
    if has_vote || has_log_entries || has_snapshot_base {
        Bootstrap::Existing
    } else {
        Bootstrap::Fresh
    }
}

/// Durable identity and membership observed at startup, for restart
/// verification.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DurableClusterView {
    /// Cluster identity from the data directory.
    pub cluster: ClusterId,
    /// Node identity from the data directory.
    pub node: NodeId,
    /// Voter set from the durable log/snapshot membership (empty when the
    /// group never committed membership — only possible on a fresh
    /// directory, which never reaches verification).
    pub voters: BTreeSet<u64>,
}

/// Why startup refused to join the configured cluster.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum BootstrapError {
    /// The data directory belongs to a different cluster.
    #[error("cluster mismatch: directory holds {found}, config declares {expected}")]
    ClusterMismatch {
        /// Configured cluster.
        expected: u128,
        /// Directory cluster.
        found: u128,
    },
    /// The data directory belongs to a different node.
    #[error("node mismatch: directory holds {found}, config declares {expected}")]
    NodeMismatch {
        /// Configured node.
        expected: u64,
        /// Directory node.
        found: u64,
    },
    /// This node is not a member of the configured cluster.
    #[error("node {node} is not a member of the configured cluster")]
    NotAMember {
        /// The local node.
        node: u64,
    },
    /// The configured voter set differs from durable membership:
    /// re-bootstrapping over history would fork it.
    #[error("membership mismatch: durable voters {durable:?}, configured voters {configured:?}")]
    MembershipMismatch {
        /// Configured voters.
        configured: BTreeSet<u64>,
        /// Durable voters.
        durable: BTreeSet<u64>,
    },
}

/// Verifies startup config against durable state (restart discipline):
/// identities must match exactly, and the configured voter set for the
/// tablet must equal the durable voter set. Endpoint addresses are
/// deliberately NOT compared (operators may remap ports across restarts).
///
/// # Errors
///
/// Returns [`BootstrapError`] on any identity or membership conflict —
/// startup fails loudly instead of forking history.
pub fn verify_against_durable(
    topology: &ClusterTopology,
    tablet: TabletId,
    local: NodeId,
    durable: &DurableClusterView,
) -> Result<(), BootstrapError> {
    use BootstrapError as Fault;
    if durable.cluster != topology.cluster {
        return Err(Fault::ClusterMismatch {
            expected: topology.cluster.as_u128(),
            found: durable.cluster.as_u128(),
        });
    }
    if durable.node != local {
        return Err(Fault::NodeMismatch {
            expected: local.as_u64(),
            found: durable.node.as_u64(),
        });
    }
    if topology.node(local).is_none() {
        return Err(Fault::NotAMember {
            node: local.as_u64(),
        });
    }
    let (voters, _) = topology
        .membership_for(tablet)
        .map_err(|_| Fault::NotAMember {
            node: local.as_u64(),
        })?;
    if voters != durable.voters {
        return Err(Fault::MembershipMismatch {
            configured: voters,
            durable: durable.voters.clone(),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn topology() -> ClusterTopology {
        ClusterTopology {
            cluster: ClusterId::from_u128(0x0C10_57E2),
            nodes: vec![
                NodeDescriptor {
                    node: NodeId::from_u64(1),
                    peer: "127.0.0.1:9101".parse().expect("peer addr"),
                    native: "127.0.0.1:9001".parse().expect("native addr"),
                },
                NodeDescriptor {
                    node: NodeId::from_u64(2),
                    peer: "127.0.0.1:9102".parse().expect("peer addr"),
                    native: "127.0.0.1:9002".parse().expect("native addr"),
                },
                NodeDescriptor {
                    node: NodeId::from_u64(3),
                    peer: "127.0.0.1:9103".parse().expect("peer addr"),
                    native: "127.0.0.1:9003".parse().expect("native addr"),
                },
            ],
            tablets: vec![TabletAssignment {
                tablet: TabletId::from_u64(9),
                replicas: vec![
                    NodeId::from_u64(1),
                    NodeId::from_u64(2),
                    NodeId::from_u64(3),
                ],
            }],
        }
    }

    #[test]
    fn first_deployment_shape_validates() {
        let topology = topology();
        topology.validate().expect("1x3 validates");
        let (voters, nodes) = topology
            .membership_for(TabletId::from_u64(9))
            .expect("membership builds");
        assert_eq!(voters, BTreeSet::from([1, 2, 3]));
        assert_eq!(nodes.len(), 3);
        assert_eq!(nodes[&1], "127.0.0.1:9101");
    }

    #[test]
    fn incoherent_topologies_fail_loudly() {
        let mut bad = topology();
        bad.nodes[1].peer = bad.nodes[0].peer;
        assert!(matches!(
            bad.validate(),
            Err(TopologyError::DuplicatePeer { .. })
        ));
        let mut bad = topology();
        bad.tablets[0].replicas = vec![NodeId::from_u64(1), NodeId::from_u64(1)];
        assert!(matches!(
            bad.validate(),
            Err(TopologyError::DuplicateVoter { .. })
        ));
        let mut bad = topology();
        bad.tablets[0].replicas = vec![NodeId::from_u64(99)];
        assert!(matches!(
            bad.validate(),
            Err(TopologyError::UnknownNode { .. })
        ));
        let mut bad = topology();
        bad.tablets.clear();
        assert!(matches!(bad.validate(), Err(TopologyError::NoTablets)));
    }

    #[test]
    fn bootstrap_classification_needs_no_signals_for_fresh() {
        assert_eq!(classify_bootstrap(false, false, false), Bootstrap::Fresh);
        // Any single durable trace means existing — including a lone vote.
        assert_eq!(classify_bootstrap(true, false, false), Bootstrap::Existing);
        assert_eq!(classify_bootstrap(false, true, false), Bootstrap::Existing);
        assert_eq!(classify_bootstrap(false, false, true), Bootstrap::Existing);
    }

    #[test]
    fn restart_verification_fails_loudly_on_conflicts() {
        let topology = topology();
        let durable = DurableClusterView {
            cluster: topology.cluster,
            node: NodeId::from_u64(1),
            voters: BTreeSet::from([1, 2, 3]),
        };
        verify_against_durable(
            &topology,
            TabletId::from_u64(9),
            NodeId::from_u64(1),
            &durable,
        )
        .expect("matching restart verifies");
        // Foreign cluster fails.
        let mut foreign = durable.clone();
        foreign.cluster = ClusterId::from_u128(0xDEAD);
        assert!(matches!(
            verify_against_durable(
                &topology,
                TabletId::from_u64(9),
                NodeId::from_u64(1),
                &foreign
            ),
            Err(BootstrapError::ClusterMismatch { .. })
        ));
        // Reconfigured voters fail (would fork history).
        let mut forked = durable.clone();
        forked.voters = BTreeSet::from([1, 2, 4]);
        assert!(matches!(
            verify_against_durable(
                &topology,
                TabletId::from_u64(9),
                NodeId::from_u64(1),
                &forked
            ),
            Err(BootstrapError::MembershipMismatch { .. })
        ));
        // A node outside the configured set fails.
        assert!(matches!(
            verify_against_durable(
                &topology,
                TabletId::from_u64(9),
                NodeId::from_u64(7),
                &durable
            ),
            Err(BootstrapError::NodeMismatch { .. })
        ));
    }
}
