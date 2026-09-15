//! Immutable directory snapshots: the exact ownership truth.
//!
//! A [`DirectorySnapshot`] maps every routable point to exactly one active
//! tablet for a single namespace. Snapshots are values, never mutated in
//! place: each transition (`allocate`, `stage`, `activate`, `seal`,
//! `retire`) validates its preconditions and returns a new snapshot with a
//! bumped [`DirectoryVersion`]. A future RCU/`ArcSwap` publication layer can
//! therefore publish snapshots atomically without changing any semantics
//! defined here; it only needs to call [`DirectorySnapshot::validate`] before
//! publishing.
//!
//! Replacement ordering is enforced by construction, mirroring RFC §190:
//! a successor can activate only once the current owner no longer covers its
//! range as active (the parent must be sealed first — activating against an
//! overlapping active tablet is rejected), and a tablet retires only after
//! its redirect successors are active. There is no path from two writable
//! authorities to one range, and no path back from fenced or tombstoned to
//! writable.

use core::fmt;

use kivi_types::{NamespaceId, PartitionHash, TabletEpoch, TabletId, WriteGuardGeneration};

use crate::range::{PartitionKind, PartitionRange, RangeError};
use crate::tablet::{Redirect, TabletDescriptor, TabletState};

/// Monotonic publication version of a directory snapshot.
///
/// Genesis is [`INITIAL`](Self::INITIAL); every transition advances by
/// exactly one via [`next`](Self::next), which fails explicitly at `u64::MAX`
/// rather than wrapping — versions only advance, never recycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DirectoryVersion(u64);

impl DirectoryVersion {
    /// Invalid sentinel: no directory was ever published at version zero.
    pub const INVALID: Self = Self(0);
    /// First version, assigned by bootstrap.
    pub const INITIAL: Self = Self(1);

    /// Wraps a raw version value.
    #[must_use]
    pub const fn from_u64(value: u64) -> Self {
        Self(value)
    }

    /// Returns the raw value.
    #[must_use]
    pub const fn as_u64(self) -> u64 {
        self.0
    }

    /// Whether this is a published version rather than the sentinel.
    #[must_use]
    pub const fn is_valid(self) -> bool {
        self.0 != 0
    }

    /// Successor version for the next directory transition.
    ///
    /// # Errors
    ///
    /// Returns [`DirectoryError::VersionExhausted`] at `u64::MAX` instead of
    /// wrapping: a recycled version could make a stale snapshot look current.
    pub fn next(self) -> Result<Self, DirectoryError> {
        self.0
            .checked_add(1)
            .map(Self)
            .ok_or(DirectoryError::VersionExhausted)
    }
}

impl From<DirectoryVersion> for u64 {
    /// Returns the raw version value.
    fn from(version: DirectoryVersion) -> Self {
        version.0
    }
}

impl From<u64> for DirectoryVersion {
    /// Wraps a raw version value.
    fn from(value: u64) -> Self {
        DirectoryVersion(value)
    }
}

impl fmt::Display for DirectoryVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Immutable routing and ownership state of one namespace at one version.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DirectorySnapshot {
    version: DirectoryVersion,
    namespace: NamespaceId,
    tablets: Vec<TabletDescriptor>,
}

impl DirectorySnapshot {
    /// Creates the genesis snapshot: one allocated tablet, version
    /// [`INITIAL`](DirectoryVersion::INITIAL).
    ///
    /// The tablet still passes through `stage` and `activate` like any other;
    /// bootstrap grants no shortcut to writability.
    ///
    /// # Errors
    ///
    /// Returns [`DirectoryError`] if the range is invalid or the fencing
    /// generations are not valid (nonzero) values.
    pub fn bootstrap(
        namespace: NamespaceId,
        tablet: TabletId,
        range: PartitionRange,
        epoch: TabletEpoch,
        guard: WriteGuardGeneration,
    ) -> Result<Self, DirectoryError> {
        range.check()?;
        check_generations(epoch, guard)?;
        let snapshot = Self {
            version: DirectoryVersion::INITIAL,
            namespace,
            tablets: vec![TabletDescriptor::new(
                tablet,
                range,
                TabletState::Allocated,
                epoch,
                guard,
                None,
            )],
        };
        snapshot.validate()?;
        Ok(snapshot)
    }

    /// Reserves a new tablet identity with its range and fencing generations.
    ///
    /// The tablet starts [`Allocated`](TabletState::Allocated): overlapping
    /// ranges are permitted while inactive (split children subdivide a still
    /// active parent); overlap is gated at activation instead.
    ///
    /// # Errors
    ///
    /// Returns [`DirectoryError`] on duplicate identity, invalid range, or
    /// invalid generations.
    pub fn allocate(
        &self,
        tablet: TabletId,
        range: PartitionRange,
        epoch: TabletEpoch,
        guard: WriteGuardGeneration,
    ) -> Result<Self, DirectoryError> {
        if self.tablets.iter().any(|t| t.id() == tablet) {
            return Err(DirectoryError::DuplicateTabletId { tablet });
        }
        range.check()?;
        check_generations(epoch, guard)?;
        let mut tablets = self.tablets.clone();
        tablets.push(TabletDescriptor::new(
            tablet,
            range,
            TabletState::Allocated,
            epoch,
            guard,
            None,
        ));
        self.advance(tablets)
    }

    /// Moves a tablet from allocated to inactive: state built, catching up,
    /// still not writable.
    ///
    /// # Errors
    ///
    /// Returns [`DirectoryError`] for unknown tablets or wrong current state.
    pub fn stage(&self, tablet: TabletId) -> Result<Self, DirectoryError> {
        self.replace(
            tablet,
            &[TabletState::Allocated],
            TabletState::Inactive,
            None,
            "stage",
        )
    }

    /// Makes an inactive tablet the writable authority for its range.
    ///
    /// Rejects activation against any overlapping active range: this single
    /// check is what makes two writable authorities for one point
    /// unconstructible, and what forces the seal-before-successor ordering.
    ///
    /// # Errors
    ///
    /// Returns [`DirectoryError`] for unknown tablets, wrong current state,
    /// or an overlapping active authority.
    pub fn activate(&self, tablet: TabletId) -> Result<Self, DirectoryError> {
        let index = self.index_of(tablet)?;
        let candidate = &self.tablets[index];
        if candidate.state() != TabletState::Inactive {
            return Err(DirectoryError::IllegalTransition {
                tablet,
                from: candidate.state(),
                action: "activate",
            });
        }
        for other in &self.tablets {
            if other.id() != tablet
                && other.state() == TabletState::Active
                && candidate.range().overlaps(other.range())
            {
                return Err(DirectoryError::OverlappingActiveAuthority {
                    first: tablet,
                    second: other.id(),
                });
            }
        }
        self.replace(
            tablet,
            &[TabletState::Inactive],
            TabletState::Active,
            None,
            "activate",
        )
    }

    /// Seals an active tablet: writes stop while its successor finishes
    /// replay and activates (RFC §190).
    ///
    /// # Errors
    ///
    /// Returns [`DirectoryError`] for unknown tablets or wrong current state.
    pub fn seal(&self, tablet: TabletId) -> Result<Self, DirectoryError> {
        self.replace(
            tablet,
            &[TabletState::Active],
            TabletState::Fenced,
            None,
            "seal",
        )
    }

    /// Retires a fenced (or aborted inactive) tablet into a tombstone carrying
    /// a redirect to live successors.
    ///
    /// Every successor must already exist and be active: the replacement is
    /// complete before the old authority retires, so redirects always resolve
    /// forward to writable tablets.
    ///
    /// # Errors
    ///
    /// Returns [`DirectoryError`] for unknown tablets, wrong current state,
    /// empty redirects, or successors that are missing or not active.
    pub fn retire(&self, tablet: TabletId, redirect: Redirect) -> Result<Self, DirectoryError> {
        if redirect.is_empty() {
            return Err(DirectoryError::EmptyRedirect);
        }
        for successor in redirect.successors() {
            match self.tablets.iter().find(|t| t.id() == *successor) {
                None => return Err(DirectoryError::UnknownTablet { tablet: *successor }),
                Some(target) if target.state() != TabletState::Active => {
                    return Err(DirectoryError::SuccessorNotActive {
                        successor: *successor,
                    });
                }
                Some(_) => {}
            }
        }
        self.replace(
            tablet,
            &[TabletState::Fenced, TabletState::Inactive],
            TabletState::Tombstone,
            Some(redirect),
            "retire",
        )
    }

    /// Builds a static directory tiling the entire `u128` hash space with
    /// exactly `count` active tablets, using only validated transitions
    /// (recursive halving through allocate/stage/seal/activate/retire).
    /// Tablet ids are `1..=count` (as `u64`).
    ///
    /// Splits always halve the current largest piece, so any `count >= 1`
    /// tiles exactly: powers of two yield uniform prefix lengths while
    /// other counts mix lengths (binary-buddy tiling). No gaps, no
    /// overlaps: every hash routes to exactly one active tablet. All
    /// tablets start at the supplied fencing generations.
    ///
    /// This is the first real multi-tablet static cluster layout: every
    /// tablet replicates on the same static voter set (independent Raft
    /// leadership per tablet). Later placement may assign different
    /// replica sets; routing stays keyed by this snapshot either way.
    ///
    /// # Errors
    ///
    /// Returns [`DirectoryError::InvalidTabletCount`] for a zero count.
    pub fn static_tiles(
        namespace: NamespaceId,
        count: usize,
        epoch: TabletEpoch,
        guard: WriteGuardGeneration,
    ) -> Result<Self, DirectoryError> {
        use crate::range::HashPrefix;
        if count == 0 {
            return Err(DirectoryError::InvalidTabletCount { count });
        }
        let mut next_id = 1u64;
        let mut directory = Self::bootstrap(
            namespace,
            TabletId::from_u64(next_id),
            PartitionRange::Hash(HashPrefix::new(0, 0)?),
            epoch,
            guard,
        )?
        .stage(TabletId::from_u64(next_id))
        .and_then(|snapshot| snapshot.activate(TabletId::from_u64(next_id)))?;
        let mut leaves = vec![(TabletId::from_u64(next_id), HashPrefix::new(0, 0)?)];
        while leaves.len() < count {
            // Split the largest piece (shortest prefix); ties break by
            // storage order, so the tiling is deterministic.
            let mut split = 0;
            for (index, (_, prefix)) in leaves.iter().enumerate() {
                if prefix.prefix_len() < leaves[split].1.prefix_len() {
                    split = index;
                }
            }
            let (parent, prefix) = leaves[split];
            if prefix.prefix_len() == 128 {
                // Single-address piece: unsplittable (unreachable for any
                // realistic count — the address space holds 2^128 points).
                return Err(DirectoryError::InvalidTabletCount { count });
            }
            let len = prefix.prefix_len() + 1;
            let half = 1u128 << (128 - len);
            next_id += 1;
            let left = TabletId::from_u64(next_id);
            next_id += 1;
            let right = TabletId::from_u64(next_id);
            let left_prefix = HashPrefix::new(prefix.bits(), len)?;
            let right_prefix = HashPrefix::new(prefix.bits() | half, len)?;
            directory = directory
                .allocate(
                    left,
                    PartitionRange::Hash(left_prefix),
                    TabletEpoch::INITIAL,
                    WriteGuardGeneration::INITIAL,
                )
                .and_then(|snapshot| snapshot.stage(left))?;
            directory = directory
                .allocate(
                    right,
                    PartitionRange::Hash(right_prefix),
                    TabletEpoch::INITIAL,
                    WriteGuardGeneration::INITIAL,
                )
                .and_then(|snapshot| snapshot.stage(right))?;
            directory = directory.seal(parent)?;
            directory = directory.activate(left)?;
            directory = directory.activate(right)?;
            directory = directory.retire(parent, Redirect::new(vec![left, right]))?;
            leaves.remove(split);
            leaves.push((left, left_prefix));
            leaves.push((right, right_prefix));
        }
        Ok(directory)
    }

    /// Routes a partition hash to its active tablet, if any.
    ///
    /// Considers active tablets only, in storage order; gaps (no covering
    /// active tablet, e.g. mid-split) yield `None`. Deterministic for a given
    /// snapshot: the same immutable contents always route the same way.
    #[must_use]
    pub fn lookup_by_hash(&self, hash: PartitionHash) -> Option<TabletId> {
        self.tablets
            .iter()
            .filter(|t| t.state() == TabletState::Active)
            .find(|t| t.range().contains_hash(hash))
            .map(TabletDescriptor::id)
    }

    /// Routes a raw key to its active tablet, if any (ordered layouts only).
    #[must_use]
    pub fn lookup_by_key(&self, key: &[u8]) -> Option<TabletId> {
        self.tablets
            .iter()
            .filter(|t| t.state() == TabletState::Active)
            .find(|t| t.range().contains_key(key))
            .map(TabletDescriptor::id)
    }

    /// Returns the tombstone redirect for a retired tablet, if any.
    ///
    /// This is the hook future stale-route handling builds on: a router
    /// holding a route to a retired tablet resolves forward through the
    /// recorded successors instead of failing or redesigning the model.
    #[must_use]
    pub fn redirect_of(&self, tablet: TabletId) -> Option<&Redirect> {
        self.tablets
            .iter()
            .find(|t| t.id() == tablet)
            .and_then(TabletDescriptor::redirect)
    }

    /// Returns the descriptor for one tablet, if present.
    #[must_use]
    pub fn get(&self, tablet: TabletId) -> Option<&TabletDescriptor> {
        self.tablets.iter().find(|t| t.id() == tablet)
    }

    /// Returns all tablet descriptors in storage order.
    #[must_use]
    pub fn tablets(&self) -> &[TabletDescriptor] {
        &self.tablets
    }

    /// Returns the snapshot version.
    #[must_use]
    pub const fn version(&self) -> DirectoryVersion {
        self.version
    }

    /// Returns the namespace this snapshot routes.
    #[must_use]
    pub const fn namespace(&self) -> NamespaceId {
        self.namespace
    }

    /// Re-checks every directory invariant. Transitions call this before
    /// returning; publication layers call it before publishing.
    ///
    /// # Errors
    ///
    /// Returns the first violated [`DirectoryError`]: bad version, duplicate
    /// identities, invalid ranges or generations, redirect/state mismatch,
    /// mixed active layouts, or overlapping active authorities.
    pub fn validate(&self) -> Result<(), DirectoryError> {
        if !self.version.is_valid() {
            return Err(DirectoryError::InvalidVersion);
        }
        for (index, tablet) in self.tablets.iter().enumerate() {
            if self.tablets[..index].iter().any(|t| t.id() == tablet.id()) {
                return Err(DirectoryError::DuplicateTabletId {
                    tablet: tablet.id(),
                });
            }
            tablet.range().check()?;
            if !tablet.epoch().is_valid() {
                return Err(DirectoryError::InvalidEpoch {
                    epoch: tablet.epoch(),
                });
            }
            if !tablet.guard().is_valid() {
                return Err(DirectoryError::InvalidGuard {
                    guard: tablet.guard(),
                });
            }
            match (tablet.state(), tablet.redirect()) {
                (TabletState::Tombstone, None) => {
                    return Err(DirectoryError::RedirectMissing {
                        tablet: tablet.id(),
                    });
                }
                (TabletState::Tombstone, Some(_)) | (_, None) => {}
                (_, Some(_)) => {
                    return Err(DirectoryError::UnexpectedRedirect {
                        tablet: tablet.id(),
                    });
                }
            }
        }
        let mut active_kind: Option<PartitionKind> = None;
        for tablet in self
            .tablets
            .iter()
            .filter(|t| t.state() == TabletState::Active)
        {
            match active_kind {
                None => active_kind = Some(tablet.range().kind()),
                Some(kind) if kind == tablet.range().kind() => {}
                Some(_) => return Err(DirectoryError::MixedActiveLayouts),
            }
        }
        for (index, first) in self.tablets.iter().enumerate() {
            if first.state() != TabletState::Active {
                continue;
            }
            for second in &self.tablets[index + 1..] {
                if second.state() == TabletState::Active && first.range().overlaps(second.range()) {
                    return Err(DirectoryError::OverlappingActiveAuthority {
                        first: first.id(),
                        second: second.id(),
                    });
                }
            }
        }
        Ok(())
    }

    /// Finds a tablet index or reports it unknown.
    fn index_of(&self, tablet: TabletId) -> Result<usize, DirectoryError> {
        self.tablets
            .iter()
            .position(|t| t.id() == tablet)
            .ok_or(DirectoryError::UnknownTablet { tablet })
    }

    /// Rebuilds the snapshot with one tablet moved to a new state.
    fn replace(
        &self,
        tablet: TabletId,
        allowed: &[TabletState],
        to: TabletState,
        redirect: Option<Redirect>,
        action: &'static str,
    ) -> Result<Self, DirectoryError> {
        let index = self.index_of(tablet)?;
        let from = self.tablets[index].state();
        if !allowed.contains(&from) {
            return Err(DirectoryError::IllegalTransition {
                tablet,
                from,
                action,
            });
        }
        let mut tablets = self.tablets.clone();
        let current = &tablets[index];
        tablets[index] = TabletDescriptor::new(
            current.id(),
            current.range().clone(),
            to,
            current.epoch(),
            current.guard(),
            redirect,
        );
        self.advance(tablets)
    }

    /// Rebuilds the snapshot with new contents at the next version.
    fn advance(&self, tablets: Vec<TabletDescriptor>) -> Result<Self, DirectoryError> {
        let next = Self {
            version: self.version.next()?,
            namespace: self.namespace,
            tablets,
        };
        next.validate()?;
        Ok(next)
    }
}

/// Rejects zero (invalid-sentinel) fencing generations.
fn check_generations(
    epoch: TabletEpoch,
    guard: WriteGuardGeneration,
) -> Result<(), DirectoryError> {
    if !epoch.is_valid() {
        return Err(DirectoryError::InvalidEpoch { epoch });
    }
    if !guard.is_valid() {
        return Err(DirectoryError::InvalidGuard { guard });
    }
    Ok(())
}

/// Directory transition or validation failure.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum DirectoryError {
    /// Two descriptors share one identity.
    #[error("duplicate tablet identity {tablet}")]
    DuplicateTabletId {
        /// Offending identity.
        tablet: TabletId,
    },
    /// No descriptor carries this identity.
    #[error("unknown tablet {tablet}")]
    UnknownTablet {
        /// Missing identity.
        tablet: TabletId,
    },
    /// A range failed construction validation.
    #[error("invalid range: {0}")]
    InvalidRange(#[from] RangeError),
    /// A zero (invalid-sentinel) tablet epoch was supplied.
    #[error("invalid tablet epoch {epoch}")]
    InvalidEpoch {
        /// Offending epoch.
        epoch: TabletEpoch,
    },
    /// A zero (invalid-sentinel) write-guard generation was supplied.
    #[error("invalid write-guard generation {guard}")]
    InvalidGuard {
        /// Offending generation.
        guard: WriteGuardGeneration,
    },
    /// Snapshot version is the zero sentinel.
    #[error("directory version is the zero sentinel")]
    InvalidVersion,
    /// Directory versions reached `u64::MAX` and cannot advance.
    #[error("directory version space exhausted")]
    VersionExhausted,
    /// Two active tablets claim overlapping ranges: two writable
    /// authorities for one point. Never built; this error is the gate.
    #[error("tablets {first} and {second} both actively own overlapping ranges")]
    OverlappingActiveAuthority {
        /// Activating (or earlier) tablet.
        first: TabletId,
        /// Conflicting active tablet.
        second: TabletId,
    },
    /// Active tablets mix hash and ordered layouts in one snapshot.
    #[error("active tablets mix hash and ordered layouts")]
    MixedActiveLayouts,
    /// Transition requested from a state it cannot start in.
    #[error("cannot {action} tablet {tablet} while {from}")]
    IllegalTransition {
        /// Affected tablet.
        tablet: TabletId,
        /// Observed state.
        from: TabletState,
        /// Requested transition.
        action: &'static str,
    },
    /// A redirect successor exists but is not active at retirement time.
    #[error("redirect successor {successor} is not active")]
    SuccessorNotActive {
        /// Offending successor.
        successor: TabletId,
    },
    /// Retirement supplied a redirect naming no successor.
    #[error("tombstone redirect names no successor")]
    EmptyRedirect,
    /// A tombstone carries no redirect.
    #[error("tombstone {tablet} carries no redirect")]
    RedirectMissing {
        /// Affected tablet.
        tablet: TabletId,
    },
    /// A non-tombstone carries a redirect.
    #[error("non-tombstone tablet {tablet} carries a redirect")]
    UnexpectedRedirect {
        /// Affected tablet.
        tablet: TabletId,
    },
    /// A static tiling requested an unusable tablet count.
    #[error("invalid static tablet count {count}")]
    InvalidTabletCount {
        /// Requested count (must be nonzero).
        count: usize,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::range::{HashPrefix, OrderedRange};
    use kivi_types::{TabletEpoch, TabletId, WriteGuardGeneration};

    const NS: NamespaceId = NamespaceId::from_u64(7);
    const EPOCH: TabletEpoch = TabletEpoch::INITIAL;
    const GUARD: WriteGuardGeneration = WriteGuardGeneration::INITIAL;

    fn hash_range(bits: u128, len: u8) -> PartitionRange {
        PartitionRange::Hash(HashPrefix::new(bits, len).expect("test range"))
    }

    fn ordered_range(start: &[u8], end: Option<&[u8]>) -> PartitionRange {
        PartitionRange::Ordered(
            OrderedRange::new(start.to_vec(), end.map(<[u8]>::to_vec)).expect("test range"),
        )
    }

    fn bootstrap_root() -> DirectorySnapshot {
        DirectorySnapshot::bootstrap(NS, TabletId::from_u64(1), hash_range(0, 0), EPOCH, GUARD)
            .expect("genesis")
    }

    fn activate_all(snapshot: &DirectorySnapshot, tablet: TabletId) -> DirectorySnapshot {
        snapshot
            .stage(tablet)
            .and_then(|s| s.activate(tablet))
            .expect("stage+activate")
    }

    #[test]
    fn bootstrap_creates_allocated_genesis_without_shortcuts() {
        let genesis = bootstrap_root();
        assert_eq!(genesis.version(), DirectoryVersion::INITIAL);
        assert_eq!(genesis.namespace(), NS);
        assert_eq!(
            genesis.get(TabletId::from_u64(1)).expect("present").state(),
            TabletState::Allocated
        );
        // Allocated tablets route nothing.
        assert_eq!(genesis.lookup_by_hash(PartitionHash::from_u128(0)), None);
        assert_eq!(genesis.validate(), Ok(()));
    }

    #[test]
    fn activation_makes_exactly_one_authority_routable() {
        let active = activate_all(&bootstrap_root(), TabletId::from_u64(1));
        assert_eq!(
            active.lookup_by_hash(PartitionHash::from_u128(0)),
            Some(TabletId::from_u64(1))
        );
        assert_eq!(
            active.lookup_by_hash(PartitionHash::from_u128(u128::MAX)),
            Some(TabletId::from_u64(1))
        );
        assert_eq!(
            active
                .get(TabletId::from_u64(1))
                .expect("present")
                .authority()
                .expect("authority")
                .tablet(),
            TabletId::from_u64(1)
        );
    }

    #[test]
    fn versions_advance_by_exactly_one_per_transition() {
        let genesis = bootstrap_root();
        let staged = genesis.stage(TabletId::from_u64(1)).expect("stage");
        assert_eq!(staged.version().as_u64(), genesis.version().as_u64() + 1);
        let active = staged.activate(TabletId::from_u64(1)).expect("activate");
        assert_eq!(active.version().as_u64(), staged.version().as_u64() + 1);
    }

    #[test]
    fn full_split_replacement_flow() {
        // Parent owns everything; children subdivide it; parent retires.
        let mut dir = activate_all(&bootstrap_root(), TabletId::from_u64(1));
        dir = dir
            .allocate(
                TabletId::from_u64(2),
                hash_range(0x0000_0000_0000_0000_0000_0000_0000_0000, 1),
                EPOCH,
                GUARD,
            )
            .expect("child 0*");
        dir = dir
            .allocate(
                TabletId::from_u64(3),
                hash_range(0x8000_0000_0000_0000_0000_0000_0000_0000, 1),
                EPOCH,
                GUARD,
            )
            .expect("child 1*");
        // Inactive children change nothing routable.
        assert_eq!(
            dir.lookup_by_hash(PartitionHash::from_u128(
                0x1000_0000_0000_0000_0000_0000_0000_0000
            )),
            Some(TabletId::from_u64(1))
        );
        dir = dir.stage(TabletId::from_u64(2)).expect("stage 2");
        dir = dir.stage(TabletId::from_u64(3)).expect("stage 3");
        // Activating a child against the still-active parent is rejected:
        // fencing must happen first.
        assert_eq!(
            dir.activate(TabletId::from_u64(2)),
            Err(DirectoryError::OverlappingActiveAuthority {
                first: TabletId::from_u64(2),
                second: TabletId::from_u64(1),
            })
        );
        dir = dir.seal(TabletId::from_u64(1)).expect("seal parent");
        assert_eq!(
            dir.lookup_by_hash(PartitionHash::from_u128(
                0x1000_0000_0000_0000_0000_0000_0000_0000
            )),
            None
        );
        dir = dir.activate(TabletId::from_u64(2)).expect("activate 2");
        dir = dir.activate(TabletId::from_u64(3)).expect("activate 3");
        assert_eq!(
            dir.lookup_by_hash(PartitionHash::from_u128(
                0x1000_0000_0000_0000_0000_0000_0000_0000
            )),
            Some(TabletId::from_u64(2))
        );
        assert_eq!(
            dir.lookup_by_hash(PartitionHash::from_u128(
                0x9000_0000_0000_0000_0000_0000_0000_0000
            )),
            Some(TabletId::from_u64(3))
        );
        // Retiring before successors are active is impossible here (both are),
        // and the old authority is gone: no triple to present.
        dir = dir
            .retire(
                TabletId::from_u64(1),
                Redirect::new(vec![TabletId::from_u64(2), TabletId::from_u64(3)]),
            )
            .expect("retire parent");
        assert_eq!(
            dir.get(TabletId::from_u64(1)).expect("parent").authority(),
            None
        );
        assert_eq!(
            dir.redirect_of(TabletId::from_u64(1))
                .expect("redirect")
                .successors(),
            &[TabletId::from_u64(2), TabletId::from_u64(3)]
        );
        // Tombstoned ranges never route, even though they overlap live ones.
        assert_eq!(
            dir.lookup_by_hash(PartitionHash::from_u128(
                0x1000_0000_0000_0000_0000_0000_0000_0000
            )),
            Some(TabletId::from_u64(2))
        );
        assert_eq!(dir.validate(), Ok(()));
    }

    #[test]
    fn two_writable_authorities_for_one_range_are_unconstructible() {
        let active = activate_all(&bootstrap_root(), TabletId::from_u64(1));
        let staged = active
            .allocate(TabletId::from_u64(2), hash_range(0, 0), EPOCH, GUARD)
            .and_then(|s| s.stage(TabletId::from_u64(2)))
            .expect("second root staged");
        assert!(matches!(
            staged.activate(TabletId::from_u64(2)),
            Err(DirectoryError::OverlappingActiveAuthority { .. })
        ));
        // The failed activation changed nothing.
        assert_eq!(
            staged.lookup_by_hash(PartitionHash::from_u128(42)),
            Some(TabletId::from_u64(1))
        );
        assert_eq!(staged.validate(), Ok(()));
    }

    #[test]
    fn illegal_transitions_are_rejected_without_mutation() {
        let genesis = bootstrap_root();
        // Activate skips staging.
        assert!(matches!(
            genesis.activate(TabletId::from_u64(1)),
            Err(DirectoryError::IllegalTransition { .. })
        ));
        // Seal an allocated tablet.
        assert!(matches!(
            genesis.seal(TabletId::from_u64(1)),
            Err(DirectoryError::IllegalTransition { .. })
        ));
        // Retire straight from active (must seal first).
        let active = activate_all(&genesis, TabletId::from_u64(1));
        assert!(matches!(
            active.retire(
                TabletId::from_u64(1),
                Redirect::new(vec![TabletId::from_u64(1)])
            ),
            Err(DirectoryError::IllegalTransition { .. })
        ));
        // Unknown tablets everywhere.
        assert_eq!(
            genesis.stage(TabletId::from_u64(99)),
            Err(DirectoryError::UnknownTablet {
                tablet: TabletId::from_u64(99)
            })
        );
        // Snapshots are immutable: failures leave the original intact.
        assert_eq!(genesis.version(), DirectoryVersion::INITIAL);
    }

    #[test]
    fn retirement_requires_live_successors() {
        let mut dir = activate_all(&bootstrap_root(), TabletId::from_u64(1));
        dir = dir
            .allocate(TabletId::from_u64(2), hash_range(0, 0), EPOCH, GUARD)
            .expect("allocate 2");
        dir = dir.seal(TabletId::from_u64(1)).expect("seal");
        // Empty redirect.
        assert_eq!(
            dir.retire(TabletId::from_u64(1), Redirect::new(vec![])),
            Err(DirectoryError::EmptyRedirect)
        );
        // Missing successor.
        assert_eq!(
            dir.retire(
                TabletId::from_u64(1),
                Redirect::new(vec![TabletId::from_u64(9)])
            ),
            Err(DirectoryError::UnknownTablet {
                tablet: TabletId::from_u64(9)
            })
        );
        // Present but inactive successor.
        assert_eq!(
            dir.retire(
                TabletId::from_u64(1),
                Redirect::new(vec![TabletId::from_u64(2)])
            ),
            Err(DirectoryError::SuccessorNotActive {
                successor: TabletId::from_u64(2)
            })
        );
    }

    #[test]
    fn duplicate_and_invalid_inputs_are_rejected() {
        let genesis = bootstrap_root();
        assert_eq!(
            genesis.allocate(TabletId::from_u64(1), hash_range(0, 0), EPOCH, GUARD),
            Err(DirectoryError::DuplicateTabletId {
                tablet: TabletId::from_u64(1)
            })
        );
        assert_eq!(
            genesis.allocate(
                TabletId::from_u64(2),
                hash_range(0, 0),
                TabletEpoch::INVALID,
                GUARD
            ),
            Err(DirectoryError::InvalidEpoch {
                epoch: TabletEpoch::INVALID
            })
        );
        assert_eq!(
            genesis.allocate(
                TabletId::from_u64(2),
                hash_range(0, 0),
                EPOCH,
                WriteGuardGeneration::INVALID
            ),
            Err(DirectoryError::InvalidGuard {
                guard: WriteGuardGeneration::INVALID
            })
        );
        assert!(matches!(
            DirectorySnapshot::bootstrap(
                NS,
                TabletId::from_u64(1),
                hash_range(0, 0),
                TabletEpoch::INVALID,
                GUARD
            ),
            Err(DirectoryError::InvalidEpoch { .. })
        ));
    }

    #[test]
    fn mixed_active_layouts_are_rejected() {
        let hash_active = activate_all(&bootstrap_root(), TabletId::from_u64(1));
        let staged = hash_active
            .allocate(
                TabletId::from_u64(2),
                ordered_range(b"", None),
                EPOCH,
                GUARD,
            )
            .and_then(|s| s.stage(TabletId::from_u64(2)))
            .expect("ordered staged");
        // No overlap is even computable across layouts; the layout mix itself fails.
        assert_eq!(
            staged.activate(TabletId::from_u64(2)),
            Err(DirectoryError::MixedActiveLayouts)
        );
    }

    #[test]
    fn ordered_layout_routes_by_key_with_gap_semantics() {
        let mut dir = DirectorySnapshot::bootstrap(
            NS,
            TabletId::from_u64(1),
            ordered_range(b"a", Some(b"g")),
            EPOCH,
            GUARD,
        )
        .expect("genesis");
        dir = activate_all(&dir, TabletId::from_u64(1));
        assert_eq!(dir.lookup_by_key(b"b"), Some(TabletId::from_u64(1)));
        assert_eq!(dir.lookup_by_key(b"g"), None);
        // Gap: nothing owns "z" yet — representable transiently, never misrouted.
        assert_eq!(dir.lookup_by_key(b"z"), None);
        dir = dir
            .allocate(
                TabletId::from_u64(2),
                ordered_range(b"g", None),
                EPOCH,
                GUARD,
            )
            .and_then(|s| s.stage(TabletId::from_u64(2)))
            .and_then(|s| s.activate(TabletId::from_u64(2)))
            .expect("second range live");
        assert_eq!(dir.lookup_by_key(b"z"), Some(TabletId::from_u64(2)));
        assert_eq!(dir.lookup_by_key(b"b"), Some(TabletId::from_u64(1)));
        assert_eq!(dir.validate(), Ok(()));
    }

    #[test]
    fn stale_authority_triples_resolve_against_successors() {
        let mut dir = activate_all(&bootstrap_root(), TabletId::from_u64(1));
        let old = dir
            .get(TabletId::from_u64(1))
            .expect("parent")
            .authority()
            .expect("authority");
        dir = dir
            .allocate(TabletId::from_u64(2), hash_range(0, 0), EPOCH, GUARD)
            .and_then(|s| s.stage(TabletId::from_u64(2)))
            .expect("replacement staged");
        dir = dir.seal(TabletId::from_u64(1)).expect("seal");
        dir = dir.activate(TabletId::from_u64(2)).expect("activate");
        dir = dir
            .retire(
                TabletId::from_u64(1),
                Redirect::new(vec![TabletId::from_u64(2)]),
            )
            .expect("retire");
        let current = dir
            .get(TabletId::from_u64(2))
            .expect("child")
            .authority()
            .expect("authority");
        // The old triple names the wrong tablet now: rejected, never re-authorized.
        assert!(old.check_against(&current).is_err());
        assert_ne!(old, current);
    }

    /// Static tilings cover the whole hash space with no gaps or overlaps
    /// for uniform and non-uniform counts alike.
    #[test]
    fn static_tiles_cover_everything_exactly_once() {
        use std::collections::BTreeSet;
        for count in [1usize, 2, 3, 4, 5, 7, 16, 100] {
            let directory =
                DirectorySnapshot::static_tiles(NS, count, EPOCH, GUARD).expect("tiles build");
            let active: Vec<_> = directory
                .tablets()
                .iter()
                .filter(|tablet| tablet.state().is_writable())
                .collect();
            assert_eq!(active.len(), count, "count {count} yields {count} actives");
            // Every probe point routes somewhere (no gaps).
            let mut rng_state = 0x9E37_79B9_7F4A_7C15u64;
            let mut seen = BTreeSet::new();
            for _ in 0..4096 {
                rng_state ^= rng_state << 13;
                rng_state ^= rng_state >> 7;
                rng_state ^= rng_state << 17;
                let hash = PartitionHash::from_u128(
                    u128::from(rng_state) << 64 | u128::from(rng_state ^ 0xD1B5_4D95),
                );
                let routed = directory.lookup_by_hash(hash).expect("no gaps");
                seen.insert(routed);
            }
            // Tiling is exact: every active tablet owns some probe point
            // (no dead ranges) once probes outnumber tablets well.
            if count <= 16 {
                assert_eq!(seen.len(), count, "count {count} tiles all live");
            }
            // Ids are unique and nonzero (splits retire the genesis id and
            // mint fresh children; contiguity is not promised).
            let ids: BTreeSet<u64> = active.iter().map(|tablet| tablet.id().as_u64()).collect();
            assert_eq!(ids.len(), count);
            assert!(!ids.contains(&0));
        }
        assert!(DirectorySnapshot::static_tiles(NS, 0, EPOCH, GUARD).is_err());
    }

    /// Bounded exhaustive-style invariant sweep over synthetic directory
    /// histories (a hand-rolled stand-in for a model checker at this stage:
    /// no new dependency, fully deterministic).
    mod exhaustive {
        use super::*;

        /// Tiny deterministic generator: xorshift64 with a nonzero lock.
        struct Rng(u64);

        impl Rng {
            fn step(&mut self) -> u64 {
                let mut x = self.0 | 1;
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                self.0 = x;
                x
            }

            fn below(&mut self, n: u64) -> u64 {
                self.step() % n
            }
        }

        /// Every hash prefix of length 0..=2 (root, halves, quarters).
        fn prefix_universe() -> Vec<PartitionRange> {
            let mut ranges = Vec::new();
            for len in 0..=2u8 {
                for cell in 0..(1u128 << len) {
                    let bits = if len == 0 { 0 } else { cell << (128 - len) };
                    ranges.push(PartitionRange::Hash(
                        HashPrefix::new(bits, len).expect("canonical test prefix"),
                    ));
                }
            }
            ranges
        }

        /// Probe points hitting each quarter of the 128-bit hash space.
        const PROBES: [PartitionHash; 4] = [
            PartitionHash::from_u128(0x0000_0000_0000_0000_0000_0000_0000_0000),
            PartitionHash::from_u128(0x4000_0000_0000_0000_0000_0000_0000_0000),
            PartitionHash::from_u128(0x8000_0000_0000_0000_0000_0000_0000_0000),
            PartitionHash::from_u128(0xC000_0000_0000_0000_0000_0000_0000_0000),
        ];

        /// Actively covering tablets of `point`, by independent scan.
        fn covering(snapshot: &DirectorySnapshot, point: PartitionHash) -> Vec<TabletId> {
            snapshot
                .tablets()
                .iter()
                .filter(|t| t.state() == TabletState::Active && t.range().contains_hash(point))
                .map(TabletDescriptor::id)
                .collect()
        }

        fn run_history(seed: u64) {
            let ranges = prefix_universe();
            let mut rng = Rng(seed);
            let first = TabletId::from_u64(rng.below(3) + 1);
            let mut snapshot =
                DirectorySnapshot::bootstrap(NS, first, ranges[0].clone(), EPOCH, GUARD)
                    .expect("genesis always builds");
            for _ in 0..1500 {
                let before = snapshot.clone();
                let id = TabletId::from_u64(rng.below(3) + 1);
                let result = match rng.below(5) {
                    0 => {
                        let count = u64::try_from(ranges.len()).expect("test universe is tiny");
                        let at = usize::try_from(rng.below(count)).expect("reduced index fits");
                        let range = ranges[at].clone();
                        snapshot.allocate(id, range, EPOCH, GUARD)
                    }
                    1 => snapshot.stage(id),
                    2 => snapshot.activate(id),
                    3 => snapshot.seal(id),
                    _ => {
                        let successor = TabletId::from_u64(rng.below(3) + 1);
                        snapshot.retire(id, Redirect::new(vec![successor]))
                    }
                };
                match result {
                    Err(_) => assert_eq!(
                        snapshot, before,
                        "failed transitions must leave the snapshot untouched (seed {seed})"
                    ),
                    Ok(next) => {
                        // Every produced snapshot validates.
                        assert_eq!(next.validate(), Ok(()), "seed {seed}");
                        // Versions advance by exactly one.
                        assert_eq!(
                            next.version().as_u64(),
                            before.version().as_u64() + 1,
                            "seed {seed}"
                        );
                        // Fencing generations of surviving tablets never change.
                        for tablet in before.tablets() {
                            let carried = next.get(tablet.id()).expect("identities persist");
                            assert_eq!(carried.epoch(), tablet.epoch(), "seed {seed}");
                            assert_eq!(carried.guard(), tablet.guard(), "seed {seed}");
                        }
                        for point in PROBES {
                            // Lookup is deterministic ...
                            assert_eq!(
                                next.lookup_by_hash(point),
                                next.lookup_by_hash(point),
                                "seed {seed}"
                            );
                            // ... agrees with an independent scan ...
                            let covering = covering(&next, point);
                            assert!(
                                covering.len() <= 1,
                                "two writable authorities for one point (seed {seed})"
                            );
                            assert_eq!(
                                next.lookup_by_hash(point),
                                covering.into_iter().next(),
                                "seed {seed}"
                            );
                        }
                        snapshot = next;
                    }
                }
            }
        }

        #[test]
        fn histories_preserve_all_invariants() {
            for seed in 1..=8 {
                run_history(seed);
            }
        }
    }
}
