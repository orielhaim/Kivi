//! Local routing: immutable directory plus tablet-to-worker placement,
//! published coherently.
//!
//! A [`RoutingSnapshot`] bundles one [`DirectorySnapshot`] with the
//! [`Placement`] mapping its active tablets to workers. The engine publishes
//! snapshots as a single `Arc` through `arc-swap`: a caller either sees the
//! old pair or the new pair, never a new directory with old placement (or
//! the reverse). Routing reads load the `Arc` with no mutex on the path.

use std::collections::BTreeMap;
use std::sync::Arc;

use kivi_tablet::DirectorySnapshot;
use kivi_types::{NamespaceId, PartitionHash, TabletId, WorkerId};

/// Tablet-to-worker assignment for the active tablets of one directory.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Placement {
    tablet_to_worker: BTreeMap<TabletId, WorkerId>,
}

impl Placement {
    /// Builds a placement from `(tablet, worker)` pairs. Later pairs win on
    /// duplicate tablets (validated for exactness at snapshot build time).
    #[must_use]
    pub fn new(entries: impl IntoIterator<Item = (TabletId, WorkerId)>) -> Self {
        Self {
            tablet_to_worker: entries.into_iter().collect(),
        }
    }

    /// Returns the owning worker of one tablet, if placed.
    #[must_use]
    pub fn worker_of(&self, tablet: TabletId) -> Option<WorkerId> {
        self.tablet_to_worker.get(&tablet).copied()
    }

    /// Returns the number of placed tablets.
    #[must_use]
    pub fn len(&self) -> usize {
        self.tablet_to_worker.len()
    }

    /// Whether no tablet is placed.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.tablet_to_worker.is_empty()
    }

    /// Iterates placements in tablet order (deterministic for bootstrap).
    pub fn iter(&self) -> impl Iterator<Item = (TabletId, WorkerId)> + '_ {
        self.tablet_to_worker
            .iter()
            .map(|(tablet, worker)| (*tablet, *worker))
    }
}

/// One coherent routing publication: directory plus placement, validated
/// together and shared immutably (`Arc`) across threads.
#[derive(Debug, Clone)]
pub struct RoutingSnapshot {
    namespace: NamespaceId,
    directory: DirectorySnapshot,
    placement: Placement,
    worker_count: usize,
}

impl RoutingSnapshot {
    /// Builds and validates a coherent snapshot.
    ///
    /// Validation: the directory itself validates; its namespace matches;
    /// placements cover exactly the active tablets (no missing owners, no
    /// placements for non-directory tablets); every worker id is below
    /// `worker_count`.
    ///
    /// # Errors
    ///
    /// Returns [`RoutingError`] for any inconsistency. The engine refuses to
    /// serve rather than route against a half-valid publication.
    pub fn build(
        namespace: NamespaceId,
        directory: DirectorySnapshot,
        placement: Placement,
        worker_count: usize,
    ) -> Result<Self, RoutingError> {
        directory.validate().map_err(RoutingError::Directory)?;
        if directory.namespace() != namespace {
            return Err(RoutingError::NamespaceMismatch {
                expected: namespace,
                found: directory.namespace(),
            });
        }
        for tablet in directory.tablets() {
            if tablet.state().is_writable() && placement.worker_of(tablet.id()).is_none() {
                return Err(RoutingError::MissingPlacement {
                    tablet: tablet.id(),
                });
            }
        }
        for (tablet, worker) in placement.iter() {
            if directory.get(tablet).is_none() {
                return Err(RoutingError::UnknownPlacement { tablet });
            }
            let index = usize::try_from(worker.as_u64()).unwrap_or(usize::MAX);
            if index >= worker_count {
                return Err(RoutingError::UnknownWorker { worker });
            }
        }
        Ok(Self {
            namespace,
            directory,
            placement,
            worker_count,
        })
    }

    /// Returns the routed namespace.
    #[must_use]
    pub const fn namespace(&self) -> NamespaceId {
        self.namespace
    }

    /// Returns the directory half of this publication.
    #[must_use]
    pub fn directory(&self) -> &DirectorySnapshot {
        &self.directory
    }

    /// Returns the placement half of this publication.
    #[must_use]
    pub fn placement(&self) -> &Placement {
        &self.placement
    }

    /// Returns the worker count this publication was validated against.
    #[must_use]
    pub const fn worker_count(&self) -> usize {
        self.worker_count
    }

    /// Returns the directory version routed here.
    #[must_use]
    pub fn version(&self) -> kivi_tablet::DirectoryVersion {
        self.directory.version()
    }

    /// Routes one partition hash to its `(tablet, owner worker)`.
    ///
    /// # Errors
    ///
    /// Returns [`RoutingError::NoRoute`] when no active tablet covers the
    /// hash (a coverage gap, e.g. mid-topology-change). Placement was
    /// validated at build, so a covered hash always resolves a worker.
    pub fn route(&self, hash: PartitionHash) -> Result<(TabletId, WorkerId), RoutingError> {
        let tablet = self
            .directory
            .lookup_by_hash(hash)
            .ok_or(RoutingError::NoRoute)?;
        let worker = self
            .placement
            .worker_of(tablet)
            .ok_or(RoutingError::NoRoute)?;
        Ok((tablet, worker))
    }

    /// Wraps this snapshot for atomic publication.
    #[must_use]
    pub fn shared(self) -> Arc<Self> {
        Arc::new(self)
    }
}

/// Routing construction or lookup failure.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum RoutingError {
    /// The directory itself is invalid.
    #[error("invalid directory: {0}")]
    Directory(#[source] kivi_tablet::DirectoryError),
    /// Snapshot namespace differs from the directory's.
    #[error("namespace mismatch: expected {expected}, directory has {found}")]
    NamespaceMismatch {
        /// Expected namespace.
        expected: NamespaceId,
        /// Directory's namespace.
        found: NamespaceId,
    },
    /// An active tablet has no owning worker.
    #[error("active tablet {tablet} has no owning worker")]
    MissingPlacement {
        /// Unplaced tablet.
        tablet: TabletId,
    },
    /// A placement names a tablet absent from the directory.
    #[error("placement names unknown tablet {tablet}")]
    UnknownPlacement {
        /// Unknown tablet.
        tablet: TabletId,
    },
    /// A placement names a worker outside `0..worker_count`.
    #[error("placement names unknown worker {worker}")]
    UnknownWorker {
        /// Out-of-range worker.
        worker: WorkerId,
    },
    /// No active tablet covers the routed hash.
    #[error("no active tablet covers the routed hash")]
    NoRoute,
}

#[cfg(test)]
mod tests {
    use super::*;
    use kivi_tablet::{HashPrefix, PartitionRange};
    use kivi_types::{TabletEpoch, WriteGuardGeneration};

    const NS: NamespaceId = NamespaceId::from_u64(1);
    const EPOCH: TabletEpoch = TabletEpoch::INITIAL;
    const GUARD: WriteGuardGeneration = WriteGuardGeneration::INITIAL;

    fn hash_range(bits: u128, len: u8) -> PartitionRange {
        PartitionRange::Hash(HashPrefix::new(bits, len).expect("range"))
    }

    fn active_snapshot() -> DirectorySnapshot {
        let genesis =
            DirectorySnapshot::bootstrap(NS, TabletId::from_u64(1), hash_range(0, 0), EPOCH, GUARD)
                .expect("genesis");
        genesis
            .stage(TabletId::from_u64(1))
            .and_then(|s| s.activate(TabletId::from_u64(1)))
            .expect("active")
    }

    fn worker(n: u64) -> WorkerId {
        WorkerId::from_u64(n)
    }

    #[test]
    fn coherent_snapshot_routes() {
        let snapshot = RoutingSnapshot::build(
            NS,
            active_snapshot(),
            Placement::new([(TabletId::from_u64(1), worker(0))]),
            1,
        )
        .expect("valid");
        let hash = kivi_state::PartitionHasher::V1
            .hash(NS, b"anything")
            .expect("v1 supported");
        assert_eq!(
            snapshot.route(hash).expect("root covers all"),
            (TabletId::from_u64(1), worker(0))
        );
    }

    #[test]
    fn validation_rejects_incoherent_publications() {
        let directory = active_snapshot();
        // Missing owner for the active tablet.
        assert!(matches!(
            RoutingSnapshot::build(NS, directory.clone(), Placement::new([]), 1),
            Err(RoutingError::MissingPlacement { .. })
        ));
        // Placement for an unknown tablet.
        assert!(matches!(
            RoutingSnapshot::build(
                NS,
                directory.clone(),
                Placement::new([
                    (TabletId::from_u64(1), worker(0)),
                    (TabletId::from_u64(9), worker(0))
                ]),
                1,
            ),
            Err(RoutingError::UnknownPlacement { .. })
        ));
        // Worker out of range.
        assert!(matches!(
            RoutingSnapshot::build(
                NS,
                directory.clone(),
                Placement::new([(TabletId::from_u64(1), worker(7))]),
                1,
            ),
            Err(RoutingError::UnknownWorker { .. })
        ));
        // Namespace mismatch.
        assert!(matches!(
            RoutingSnapshot::build(
                NamespaceId::from_u64(2),
                directory,
                Placement::new([(TabletId::from_u64(1), worker(0))]),
                1,
            ),
            Err(RoutingError::NamespaceMismatch { .. })
        ));
    }

    #[test]
    fn inactive_tablets_need_no_placement_but_never_route() {
        let directory = active_snapshot()
            .allocate(TabletId::from_u64(2), hash_range(0, 0), EPOCH, GUARD)
            .expect("allocate overlapping inactive");
        let snapshot = RoutingSnapshot::build(
            NS,
            directory,
            Placement::new([(TabletId::from_u64(1), worker(0))]),
            1,
        )
        .expect("inactive needs no owner");
        let hash = kivi_state::PartitionHasher::V1.hash(NS, b"k").expect("v1");
        assert_eq!(
            snapshot.route(hash).expect("route").0,
            TabletId::from_u64(1)
        );
    }
}
