//! Strongly typed observation scopes.

use kivi_types::{ClusterId, NamespaceId, NodeId, TabletId, WorkerId};

/// Domain boundary at which an observation was produced.
///
/// Scope variants deliberately use distinct Kivi identifier types. A worker,
/// tablet, node, namespace, or cluster observation cannot be misattributed to
/// another hierarchy level by accidental cast.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ObservationScope {
    /// Execution owned by one engine worker.
    Worker(WorkerId),
    /// Authority and routing unit for one tablet.
    Tablet(TabletId),
    /// One cluster member process or host.
    Node(NodeId),
    /// Logical policy and ownership boundary.
    Namespace(NamespaceId),
    /// One complete Kivi cluster.
    Cluster(ClusterId),
}
