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

use kivi_state::NamespaceRouteHasher;
use kivi_tablet::DirectorySnapshot;
use kivi_types::{NamespaceId, PartitionHash, TabletId, TabletRoute, WorkerId};

use crate::slots::LocalTabletSet;

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
    compiled: CompiledRoute,
    /// The namespace's route hasher, precomputed.
    ///
    /// Held here rather than passed per call because a snapshot is *per namespace*:
    /// this is the natural home for it, and it means `route_key` needs no second
    /// argument and cannot be called with the wrong namespace's seed.
    hasher: NamespaceRouteHasher,
}

/// A routing decision with the work already done, or the work still to do.
///
/// This is the whole of the "do not repeat work" rule for routing, and it exists
/// because a hash is the *last* thing that should happen on a read, not the first.
///
/// # The single-tablet case
///
/// When a namespace's directory has exactly one active tablet, **every key routes
/// to it**. That is a constant function, so the correct implementation of "route a
/// key" is "return the tablet" - no key bytes hashed, no prefix extracted, no
/// descriptor scanned, no placement looked up. The hash is not merely cheap here; it
/// is *unnecessary*, and a design that computes it anyway is doing work whose result
/// cannot change the answer.
///
/// This is not a small cluster optimisation. It is the shape every namespace has at
/// creation, and it is the shape the Redis-war single-tablet benchmark measures, so it
/// is the shape most of the reported numbers are taken in.
///
/// # Staleness
///
/// [`CompiledRoute::Single`] is a property of *this* immutable snapshot, and the
/// snapshot is replaced atomically on a split. A caller that caches a
/// [`SingleTabletRoute`] across a split would keep serving the old tablet, so the
/// route carries the directory version it was compiled from and
/// [`SingleTabletRoute::is_current_for`] is the check. The reference-counted
/// publication is what makes the replacement safe; the version is what makes a
/// *deliberately* cached route checkable rather than silently wrong.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompiledRoute {
    /// One active tablet: the route is a constant.
    Single(SingleTabletRoute),
    /// More than one active tablet, or a layout that routes by key: hash or scan.
    Variable,
}

/// A routing decision that was computed once because it is the same for every key.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SingleTabletRoute {
    tablet: TabletId,
    worker: WorkerId,
    version: kivi_tablet::DirectoryVersion,
}

impl SingleTabletRoute {
    /// The one tablet that serves this namespace.
    #[must_use]
    pub const fn tablet(&self) -> TabletId {
        self.tablet
    }

    /// The worker that owns it.
    #[must_use]
    pub const fn worker(&self) -> WorkerId {
        self.worker
    }

    /// The directory version this was compiled from.
    #[must_use]
    pub const fn version(&self) -> kivi_tablet::DirectoryVersion {
        self.version
    }

    /// Whether this route is still the one a given snapshot would produce.
    ///
    /// The check a caller that caches a route across a topology change needs. A
    /// snapshot whose version differs is a different routing world, and a route from
    /// the old one must not be used against it.
    #[must_use]
    pub fn is_current_for(&self, snapshot: &RoutingSnapshot) -> bool {
        matches!(snapshot.compiled, CompiledRoute::Single(current) if current == *self)
    }
}

impl CompiledRoute {
    /// Decides, once per publication, which shape the route can take.
    ///
    /// A `const fn` would be wrong here: it needs the directory's tablet states, and
    /// a decision that cannot be folded means the *check* runs on every request
    /// instead of the hash, which is a different kind of repeat-work bug.
    fn compile(directory: &DirectorySnapshot, placement: &Placement) -> Self {
        let mut active = directory
            .tablets()
            .iter()
            .filter(|tablet| tablet.state().is_writable());
        let Some(only) = active.next() else {
            // No active tablet is a coverage gap, not a constant route. Handing back
            // `Variable` means the normal path runs and reports `NoRoute`, which is
            // the honest answer; inventing a route here would serve from a tablet
            // that is mid-retire.
            return Self::Variable;
        };
        if active.next().is_some() {
            return Self::Variable;
        }
        let tablet = only.id();
        // `build` already proved every writable tablet has a placement, so this
        // cannot be `None` for a valid snapshot. If it somehow is, `Variable` is the
        // safe answer: the slow path will report the gap rather than assert a route.
        match placement.worker_of(tablet) {
            Some(worker) => Self::Single(SingleTabletRoute {
                tablet,
                worker,
                version: directory.version(),
            }),
            None => Self::Variable,
        }
    }
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
        let compiled = CompiledRoute::compile(&directory, &placement);
        Ok(Self {
            hasher: NamespaceRouteHasher::new(namespace),
            namespace,
            directory,
            placement,
            worker_count,
            compiled,
        })
    }

    /// The routing decision shape this snapshot allows.
    #[must_use]
    pub const fn compiled(&self) -> CompiledRoute {
        self.compiled
    }

    /// Routes an ordinary key's bytes, doing only the work the answer depends on.
    ///
    /// The order is deliberate, and it is the whole point of the compiled route:
    ///
    /// 1. **One active tablet** ⇒ return it. Nothing is hashed. This is the common
    ///    case and it is a constant function.
    /// 2. **Hash layout** ⇒ hash the key once and resolve the range. The hash is
    ///    computed here and nowhere else on the path.
    /// 3. **Ordered layout** ⇒ scan by key bytes, never hashing. A generic key type
    ///    that knows how to hash must not hash an ordered namespace, because the
    ///    hash has no consumer there.
    ///
    /// Step 3 is why the layout is consulted before the hash rather than after: a
    /// point lookup in an ordered namespace is a byte comparison, and producing a
    /// route hash nobody reads is exactly the "repeat work to reuse work" mistake in
    /// its purest form.
    ///
    /// # Errors
    ///
    /// Returns [`RoutingError::NoRoute`] when no active tablet covers the key (a
    /// coverage gap, e.g. mid-split).
    pub fn route_key(&self, key: &[u8]) -> Result<(TabletId, WorkerId), RoutingError> {
        if let CompiledRoute::Single(route) = self.compiled {
            return Ok((route.tablet(), route.worker()));
        }
        // Ordered ranges route by bytes and hash ranges by hash; a snapshot is
        // single-layout, so at most one of these can hit. Ordered is tried first
        // because a byte comparison is cheaper than a hash, and a hash layout
        // answers `contains_key` with `false` for every tablet.
        if let Some(tablet) = self.directory.lookup_by_key(key) {
            return self.worker_for(tablet);
        }
        let tablet = self
            .directory
            .lookup_by_hash(self.hasher.hash_key(key).as_partition_hash())
            .ok_or(RoutingError::NoRoute)?;
        self.worker_for(tablet)
    }

    /// Resolves a tablet to its owning worker, or a coverage gap.
    fn worker_for(&self, tablet: TabletId) -> Result<(TabletId, WorkerId), RoutingError> {
        let worker = self
            .placement
            .worker_of(tablet)
            .ok_or(RoutingError::NoRoute)?;
        Ok((tablet, worker))
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

/// One worker's routing, resolved down to slots.
///
/// [`RoutingSnapshot`] is the control plane's answer: *which tablet* serves this
/// key. That is not an executable instruction, because a tablet is named by a
/// [`TabletId`] and finding the tablet means a lookup. So a worker compiles the
/// snapshot once, against the tablets it actually owns, into the data plane's
/// answer: *which slot*.
///
/// The compile is trustworthy because its inputs cannot move underneath it: the
/// snapshot is an `Arc` behind an `ArcSwap`, and the slot assignment is fixed
/// for the worker's life (tablets are installed when the worker starts, and a
/// split produces a new worker).
///
/// # Fencing
///
/// A route compiled against version *V* must not be used once version *V+1* is
/// published: a split may have moved the key. [`Self::is_current_for`] carries
/// the version so a caller can check, and `single` is [`Option`] precisely so a
/// namespace that is no longer a single tablet cannot be served from a stale
/// constant. The check belongs to the *executor*, not the table: the version is
/// there to be compared, and the drain compares it once per turn.
#[derive(Debug)]
pub struct LocalRoutes {
    version: kivi_tablet::DirectoryVersion,
    /// The constant route, when the publication says the namespace has one active
    /// tablet *and* this worker owns it. `None` otherwise, including for a
    /// namespace this worker does not own at all, and for a namespace with
    /// several tablets that happen to all be ours.
    single: Option<TabletRoute>,
}

impl LocalRoutes {
    /// Compiles a publication against the tablets one worker owns.
    ///
    /// Yields no constant route when this worker owns nothing in the namespace:
    /// that is not a routing failure, it is a statement about this worker, and
    /// the executor answers it by declining the command so the forwarder can
    /// send it where it belongs.
    #[must_use]
    pub fn compile(snapshot: &RoutingSnapshot, worker: WorkerId, tablets: &LocalTabletSet) -> Self {
        // The constant route is only the constant route if the whole namespace is
        // one tablet, so this checks the compiled route and not the local count:
        // a worker owning two of five tablets must not answer `GET` from
        // whichever of the two it happens to hold.
        let single = match snapshot.compiled() {
            CompiledRoute::Single(route) if route.worker() == worker => tablets
                .slot_of(route.tablet())
                .map(|slot| TabletRoute::new(route.tablet(), slot)),
            _ => None,
        };
        Self {
            version: snapshot.version(),
            single,
        }
    }

    /// The constant route, if this worker has one.
    #[must_use]
    pub const fn single(&self) -> Option<TabletRoute> {
        self.single
    }

    /// The directory version this was compiled from.
    #[must_use]
    pub const fn version(&self) -> kivi_tablet::DirectoryVersion {
        self.version
    }

    /// Whether this compilation still describes a given publication.
    #[must_use]
    pub fn is_current_for(&self, snapshot: &RoutingSnapshot) -> bool {
        self.version == snapshot.version()
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
        let hash = kivi_state::route_hash(NS, b"anything");
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
        let hash = kivi_state::route_hash(NS, b"k");
        assert_eq!(
            snapshot.route(hash).expect("route").0,
            TabletId::from_u64(1)
        );
    }

    // ---- the compiled route: one active tablet means no hash at all ----

    /// A canonical hash prefix: `value` occupies the top `len` bits.
    ///
    /// `HashPrefix` requires its insignificant low bits to be zero, so a width-1
    /// prefix of `1` is `1 << 127` and not `1`. Getting that wrong is
    /// `PrefixUncanonical`, which is a useful error: the alternative - a prefix tree
    /// that silently disagreed with itself about which half a key was in - is not.
    fn hash_prefix(value: u128, len: u8) -> kivi_tablet::PartitionRange {
        hash_range(
            if len == 0 {
                0
            } else {
                value << (128 - u32::from(len))
            },
            len,
        )
    }

    /// Splits a single-tablet directory in two and returns it with both halves live.
    ///
    /// The full lifecycle - allocate, stage, seal, activate, retire - because the
    /// point of the tests that use it is that a *published* snapshot changes shape
    /// correctly, and a shortcut that skipped staging would be testing a state the
    /// control plane can never publish.
    fn split_root(
        root: &DirectorySnapshot,
        left: TabletId,
        right: TabletId,
        left_range: kivi_tablet::PartitionRange,
        right_range: kivi_tablet::PartitionRange,
    ) -> DirectorySnapshot {
        root.allocate(left, left_range, EPOCH, GUARD)
            .expect("allocate left")
            .allocate(right, right_range, EPOCH, GUARD)
            .expect("allocate right")
            .stage(left)
            .and_then(|s| s.stage(right))
            .expect("stage both")
            .seal(TabletId::from_u64(1))
            .expect("seal root")
            .activate(left)
            .and_then(|s| s.activate(right))
            .expect("activate both")
            .retire(
                TabletId::from_u64(1),
                kivi_tablet::Redirect::new(vec![left, right]),
            )
            .expect("retire root")
    }

    #[test]
    fn one_active_tablet_compiles_to_a_constant_route() {
        let snapshot = RoutingSnapshot::build(
            NS,
            active_snapshot(),
            Placement::new([(TabletId::from_u64(1), worker(0))]),
            1,
        )
        .expect("valid");
        let CompiledRoute::Single(route) = snapshot.compiled() else {
            panic!("a single active tablet must compile to a constant route");
        };
        assert_eq!(route.tablet(), TabletId::from_u64(1));
        assert_eq!(route.worker(), worker(0));
        // Every key, of every shape, resolves to the same answer - which is the
        // property that makes skipping the hash correct rather than merely fast.
        for key in [
            b"".as_slice(),
            b"a",
            b"a long key that would sit in any band at all",
            &[0xffu8; 512],
        ] {
            assert_eq!(
                snapshot.route_key(key).expect("routes"),
                (TabletId::from_u64(1), worker(0)),
                "the constant route disagreed for {key:?}"
            );
        }
    }

    /// The transition the specification calls out: one tablet, then a split, then
    /// many. A route compiled against the single-tablet snapshot must stop being
    /// current at the split, and the new snapshot must route by hash.
    #[test]
    fn a_single_tablet_route_stops_being_current_after_a_split() {
        let single = RoutingSnapshot::build(
            NS,
            active_snapshot(),
            Placement::new([(TabletId::from_u64(1), worker(0))]),
            1,
        )
        .expect("valid");
        let stale = match single.compiled() {
            CompiledRoute::Single(route) => route,
            CompiledRoute::Variable => panic!("expected a constant route"),
        };
        assert!(stale.is_current_for(&single));

        let split = RoutingSnapshot::build(
            NS,
            split_root(
                single.directory(),
                TabletId::from_u64(2),
                TabletId::from_u64(3),
                hash_prefix(0, 1),
                hash_prefix(1, 1),
            ),
            Placement::new([
                (TabletId::from_u64(2), worker(0)),
                (TabletId::from_u64(3), worker(1)),
            ]),
            2,
        )
        .expect("split snapshot is coherent");

        // The whole point: the new snapshot is *not* a constant route, and the old
        // one is not current for it.
        assert_eq!(split.compiled(), CompiledRoute::Variable);
        assert!(!stale.is_current_for(&split));
        assert_ne!(stale.version(), split.version());

        // And every key still routes somewhere real, by hash.
        for index in 0..64u32 {
            let key = format!("key:{index}");
            let (tablet, worker_id) = split.route_key(key.as_bytes()).expect("routes");
            assert!(
                tablet == TabletId::from_u64(2) || tablet == TabletId::from_u64(3),
                "a key routed to a retired tablet"
            );
            assert_eq!(
                worker_id,
                if tablet == TabletId::from_u64(2) {
                    worker(0)
                } else {
                    worker(1)
                },
                "the tablet and its worker disagree"
            );
        }
    }

    /// Two tablets means the hash is load-bearing again, so the answer has to
    /// actually depend on it - a route that ignored the key would pass the
    /// single-tablet tests and fail here.
    #[test]
    fn two_tablets_make_the_hash_load_bearing_again() {
        let snapshot = RoutingSnapshot::build(
            NS,
            split_root(
                &active_snapshot(),
                TabletId::from_u64(2),
                TabletId::from_u64(3),
                hash_prefix(0, 1),
                hash_prefix(1, 1),
            ),
            Placement::new([
                (TabletId::from_u64(2), worker(0)),
                (TabletId::from_u64(3), worker(1)),
            ]),
            2,
        )
        .expect("coherent");

        let mut seen = std::collections::HashSet::new();
        for index in 0..512u32 {
            let key = format!("key:{index}");
            let (tablet, _) = snapshot.route_key(key.as_bytes()).expect("routes");
            seen.insert(tablet);
        }
        assert_eq!(
            seen.len(),
            2,
            "512 keys across a 50/50 split should reach both tablets"
        );
    }

    /// An ordered namespace routes by bytes, so a route hash is not merely
    /// unnecessary there - it is unobservable. This asserts the layout is consulted
    /// *instead of* the hash: a key outside every ordered range reports `NoRoute`
    /// rather than being silently routed by a hash nobody asked for.
    ///
    /// Two tablets, deliberately: with one, the constant route is correct and there
    /// is nothing to fall through from. The first version of this test used a single
    /// tablet and asserted `NoRoute`, which failed for the right reason - a single
    /// tablet owns the whole namespace whatever its layout, and the test was asking
    /// it to behave as if it did not.
    #[test]
    fn an_ordered_layout_never_falls_through_to_the_hash() {
        use kivi_tablet::OrderedRange;
        let root = DirectorySnapshot::bootstrap(
            NS,
            TabletId::from_u64(1),
            kivi_tablet::PartitionRange::Ordered(
                OrderedRange::new(vec![b'a'], None).expect("unbounded range"),
            ),
            EPOCH,
            GUARD,
        )
        .expect("genesis")
        .stage(TabletId::from_u64(1))
        .and_then(|s| s.activate(TabletId::from_u64(1)))
        .expect("active");
        let snapshot = RoutingSnapshot::build(
            NS,
            split_root(
                &root,
                TabletId::from_u64(2),
                TabletId::from_u64(3),
                kivi_tablet::PartitionRange::Ordered(
                    OrderedRange::new(vec![b'a'], Some(vec![b'm'])).expect("lower half"),
                ),
                kivi_tablet::PartitionRange::Ordered(
                    OrderedRange::new(vec![b'm'], None).expect("upper half"),
                ),
            ),
            Placement::new([
                (TabletId::from_u64(2), worker(0)),
                (TabletId::from_u64(3), worker(1)),
            ]),
            2,
        )
        .expect("coherent");
        assert_eq!(snapshot.compiled(), CompiledRoute::Variable);

        assert_eq!(
            snapshot.route_key(b"zebra").expect("upper half").0,
            TabletId::from_u64(3)
        );
        assert_eq!(
            snapshot.route_key(b"apple").expect("lower half").0,
            TabletId::from_u64(2)
        );
        // Below every range: a coverage gap, and *not* a silent hash-based route.
        assert!(matches!(
            snapshot.route_key(b"AAA"),
            Err(RoutingError::NoRoute)
        ));
    }
}
#[cfg(test)]
mod local_routes {
    use kivi_tablet::{DirectorySnapshot, TabletDescriptor};
    use kivi_types::{NamespaceId, TabletAuthority, TabletEpoch, WorkerId, WriteGuardGeneration};

    use super::{CompiledRoute, LocalRoutes, Placement, RoutingSnapshot};
    use crate::slots::LocalTabletSet;
    use crate::tablet::LiveTablet;

    const NS: NamespaceId = NamespaceId::from_u64(1);
    const EPOCH: TabletEpoch = TabletEpoch::INITIAL;
    const GUARD: WriteGuardGeneration = WriteGuardGeneration::INITIAL;

    /// A hash-tiled directory of `count` writable tablets, ids 1..=count.
    fn tiled(count: usize) -> DirectorySnapshot {
        DirectorySnapshot::static_tiles(NS, count, EPOCH, GUARD).expect("tiling builds")
    }

    fn live_from(descriptor: &TabletDescriptor) -> LiveTablet {
        LiveTablet::from_descriptor(
            descriptor,
            TabletAuthority::new(descriptor.id(), EPOCH, GUARD),
        )
        .expect("authority matches descriptor")
    }

    fn set_of(directory: &DirectorySnapshot) -> LocalTabletSet {
        LocalTabletSet::from_tablets(
            directory
                .tablets()
                .iter()
                .filter(|t| t.state().is_writable())
                .map(live_from)
                .collect(),
        )
    }

    fn routed(directory: DirectorySnapshot) -> RoutingSnapshot {
        let placement = Placement::new(
            directory
                .tablets()
                .iter()
                .filter(|t| t.state().is_writable())
                .map(|t| (t.id(), WorkerId::from_u64(0))),
        );
        RoutingSnapshot::build(NS, directory, placement, 1).expect("routing builds")
    }

    fn single() -> RoutingSnapshot {
        routed(tiled(1))
    }

    #[test]
    fn a_single_local_tablet_compiles_to_a_constant_route() {
        let routing = single();
        let set = set_of(routing.directory());
        let routes = LocalRoutes::compile(&routing, WorkerId::from_u64(0), &set);
        let route = routes.single().expect("constant route");
        assert_eq!(route.tablet(), routing.directory().tablets()[0].id());
        // The whole point: the route names a slot, so resolving it is an index.
        assert_eq!(route.slot(), set.slot_of(route.tablet()).expect("owned"));
    }

    /// The trap this type exists to make impossible: a worker can own *every* tablet
    /// and still not have a constant route, because "every tablet I own" is not "the
    /// only tablet there is". Reading that as a constant would answer `GET` from
    /// whichever tablet the worker happened to hold.
    #[test]
    fn owning_every_tablet_is_not_a_constant_route() {
        let routing = routed(tiled(3));
        assert!(matches!(routing.compiled(), CompiledRoute::Variable));
        let set = set_of(routing.directory());
        assert!(
            LocalRoutes::compile(&routing, WorkerId::from_u64(0), &set)
                .single()
                .is_none(),
            "a three-tablet namespace has no constant, whoever owns it"
        );
    }

    #[test]
    fn a_worker_owning_nothing_has_no_route() {
        let routing = single();
        let set = set_of(routing.directory());
        assert!(
            LocalRoutes::compile(&routing, WorkerId::from_u64(7), &set)
                .single()
                .is_none(),
            "a worker that owns nothing declines rather than guessing"
        );
    }

    /// A route compiled against one publication must be recognisably stale once a
    /// newer one is published: a split may have moved the key, and a stale
    /// constant must never answer for it.
    #[test]
    fn a_stale_compilation_is_detectable_by_version() {
        let routing = single();
        let set = set_of(routing.directory());
        let routes = LocalRoutes::compile(&routing, WorkerId::from_u64(0), &set);
        assert!(routes.is_current_for(&routing));
        assert_eq!(routes.version(), routing.version());

        let moved = routed(tiled(2));
        assert_ne!(moved.version(), routing.version());
        assert!(
            !routes.is_current_for(&moved),
            "a new publication is a new world"
        );
        assert!(
            LocalRoutes::compile(&moved, WorkerId::from_u64(0), &set_of(moved.directory()))
                .single()
                .is_none(),
            "two tablets means no constant route, whoever owns them"
        );
    }

    /// The route a compiled publication names must be the tablet that publication
    /// says, not merely one the worker happens to hold. With a single tablet these
    /// coincide; this is the case that tells them apart.
    #[test]
    fn a_constant_route_names_the_published_tablet() {
        let routing = single();
        let set = set_of(routing.directory());
        let CompiledRoute::Single(constant) = routing.compiled() else {
            panic!("one tablet is a constant route");
        };
        let routes = LocalRoutes::compile(&routing, WorkerId::from_u64(0), &set);
        assert_eq!(
            routes.single().expect("compiled").tablet(),
            constant.tablet()
        );
        assert!(routes.is_current_for(&routing));
    }
}
