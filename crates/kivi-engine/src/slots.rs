//! Dense, slot-addressed storage for the tablets a worker owns.
//!
//! # The two identities of a tablet
//!
//! A tablet has two names.
//!
//! * [`TabletId`] is the **control-plane** name. The directory assigns it, the
//!   placement maps it to a worker, and a routing publication carries it. It is
//!   stable across the worker's life and is what a snapshot, a log record, or a
//!   client-visible error must name.
//! * [`LocalTabletSlot`] is the **data-plane** name. The worker assigns it when it
//!   builds its set, and nothing outside the owning thread ever sees it. It is a
//!   dense index, so resolving it is a bounds check.
//!
//! A compiled route ([`LocalRoutes`](crate::routing::LocalRoutes)) carries the
//! data-plane name, so the request path never hashes. The id index answers the
//! control plane instead - a split, a merge, a checkpoint walk, a
//! commit-coordinator sweep - at topology-change rates rather than request
//! rates. That hash is a `HashMap` keyed by `u64` under `RandomState`: a seeded
//! hash, which is only defensible for bytes an attacker chooses. A tablet id is
//! an integer this process was handed by its own control plane, so the index
//! pays for seeding it does not need.
//!
//! A compiled route is produced once per routing publication and then executed,
//! and a slot that is stale against a newer publication is detected by the
//! version it was compiled from, not by being silently wrong.

use core::fmt;

use std::collections::HashMap;

use kivi_types::{LocalTabletSlot, TabletId};

use crate::tablet::LiveTablet;

/// Every tablet one worker owns, addressable both ways.
///
/// The dense vector is the data plane: [`Self::by_slot`] is a bounds check and a
/// null check. The id index is the control plane: [`Self::get`] is a hash, and
/// pays for that knowingly.
pub struct LocalTabletSet {
    /// Dense storage. A hole is a slot in the free list (or never issued).
    slots: Vec<Option<LiveTablet>>,
    /// Control-plane name to data-plane name.
    index: HashMap<TabletId, LocalTabletSlot>,
    /// Reclaimed slots, reused before the vector grows.
    free: Vec<LocalTabletSlot>,
}

impl LocalTabletSet {
    /// Takes ownership of a worker's tablets and assigns their slots.
    ///
    /// Slot assignment is the vector position, so a worker with one tablet - the
    /// shape most single-node deployments have - gets slot 0 and the whole lookup
    /// is `slots[0]`.
    #[must_use]
    pub(crate) fn from_tablets(tablets: Vec<LiveTablet>) -> Self {
        let mut set = Self {
            slots: Vec::with_capacity(tablets.len()),
            index: HashMap::with_capacity(tablets.len()),
            free: Vec::new(),
        };
        for tablet in tablets {
            set.insert(tablet);
        }
        set
    }

    /// An empty set. For tests and for a worker that is not yet assigned tablets.
    #[must_use]
    pub fn new() -> Self {
        Self {
            slots: Vec::new(),
            index: HashMap::new(),
            free: Vec::new(),
        }
    }

    /// Adds a tablet, returning the slot that now addresses it.
    ///
    /// # Panics
    ///
    /// Panics past [`u32::MAX`] live tablets. A slot is a `u32` because a worker
    /// cannot own four billion tablets; the assertion states that bound rather
    /// than letting a truncating cast alias two tablets onto one slot.
    pub fn insert(&mut self, tablet: LiveTablet) -> LocalTabletSlot {
        let id = tablet.id();
        if let Some(existing) = self.index.remove(&id) {
            // Re-inserting an id would leave the old slot holding a tablet the
            // index no longer names, and `by_slot` would hand it out. Drop the
            // stale occupant first so the set never has two live slots per id.
            self.slots[existing.index()] = None;
            self.free.push(existing);
        }
        let slot = if let Some(slot) = self.free.pop() {
            self.slots[slot.index()] = Some(tablet);
            slot
        } else {
            let slot = LocalTabletSlot::new(
                u32::try_from(self.slots.len()).expect("a worker cannot own four billion tablets"),
            );
            self.slots.push(Some(tablet));
            slot
        };
        self.index.insert(id, slot);
        slot
    }

    /// Removes a tablet, returning it and the slot it vacated.
    pub fn remove(&mut self, id: TabletId) -> Option<(LiveTablet, LocalTabletSlot)> {
        let slot = self.index.remove(&id)?;
        let tablet = self.slots[slot.index()].take()?;
        self.free.push(slot);
        Some((tablet, slot))
    }

    /// The tablet at a slot, with no hashing and no id comparison.
    ///
    /// The only way the data plane should reach a tablet. A slot vacated by a
    /// split reads as absent: a stale read is a stale answer, not corruption.
    #[must_use]
    pub(crate) fn by_slot(&self, slot: LocalTabletSlot) -> Option<&LiveTablet> {
        self.slots.get(slot.index())?.as_ref()
    }

    /// The slot a tablet id occupies. Control plane.
    #[must_use]
    pub(crate) fn slot_of(&self, id: TabletId) -> Option<LocalTabletSlot> {
        self.index.get(&id).copied()
    }

    /// Looks a tablet up by its control-plane name.
    ///
    /// This hashes. A request that reaches the data plane through a
    /// [`TabletId`] instead of a compiled route is paying for routing twice.
    #[must_use]
    pub fn get(&self, id: TabletId) -> Option<&LiveTablet> {
        self.by_slot(*self.index.get(&id)?)
    }

    /// Looks a tablet up by its control-plane name, mutably. This hashes.
    ///
    /// # Panics
    ///
    /// Panics if the index names a slot with no occupant. The index only ever
    /// holds slots of live tablets - `remove` deletes the index entry before
    /// vacating the slot - so this is an invariant break, not a reachable
    /// condition, and the alternative (mutating whichever tablet took the slot
    /// next) is corruption.
    pub fn get_mut(&mut self, id: TabletId) -> Option<&mut LiveTablet> {
        let slot = *self.index.get(&id)?;
        Some(
            self.slots[slot.index()]
                .as_mut()
                .expect("the id index names only live slots"),
        )
    }

    /// Whether a tablet id is present. This hashes.
    #[must_use]
    pub fn contains_key(&self, id: TabletId) -> bool {
        self.index.contains_key(&id)
    }

    /// Every tablet, in slot order.
    pub fn values(&self) -> impl Iterator<Item = &LiveTablet> {
        self.slots.iter().flatten()
    }

    /// Every tablet mutably, in slot order.
    pub fn values_mut(&mut self) -> impl Iterator<Item = &mut LiveTablet> {
        self.slots.iter_mut().flatten()
    }
}

impl Default for LocalTabletSet {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for LocalTabletSet {
    /// Reports the dense shape - the slot each tablet occupies - because that is
    /// the thing a slot bug looks like from the outside: two tablets on one slot,
    /// or a hole where a live route points.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_map()
            .entries(
                self.slots
                    .iter()
                    .enumerate()
                    .filter_map(|(i, slot)| slot.as_ref().map(|t| (i, t.id()))),
            )
            .finish()
    }
}

/// Slot assignment is the data plane's whole contract: a route that names a
/// slot must reach the tablet that slot holds, and a vacated slot must read as
/// absent rather than as whatever took it next.
#[cfg(test)]
mod tests {
    use super::*;
    use kivi_tablet::{DirectorySnapshot, HashPrefix, PartitionRange, TabletDescriptor};
    use kivi_types::{NamespaceId, TabletAuthority, TabletEpoch, WriteGuardGeneration};

    fn live(id: u64) -> LiveTablet {
        let tablet = TabletId::from_u64(id);
        let snapshot = DirectorySnapshot::bootstrap(
            NamespaceId::from_u64(1),
            tablet,
            PartitionRange::Hash(HashPrefix::new(0, 0).expect("root")),
            TabletEpoch::INITIAL,
            WriteGuardGeneration::INITIAL,
        )
        .expect("genesis");
        let descriptor: TabletDescriptor = snapshot.get(tablet).expect("descriptor").clone();
        LiveTablet::from_descriptor(
            &descriptor,
            TabletAuthority::new(tablet, TabletEpoch::INITIAL, WriteGuardGeneration::INITIAL),
        )
        .expect("authority matches descriptor")
    }

    #[test]
    fn a_single_tablet_is_slot_zero() {
        let mut set = LocalTabletSet::new();
        assert_eq!(set.insert(live(7)), LocalTabletSlot::new(0));
        assert_eq!(
            set.by_slot(LocalTabletSlot::new(0)).map(LiveTablet::id),
            Some(TabletId::from_u64(7))
        );
    }

    #[test]
    fn slots_survive_a_recycled_removal() {
        let mut set = LocalTabletSet::new();
        set.insert(live(1));
        set.insert(live(2));
        let (removed, slot) = set.remove(TabletId::from_u64(1)).expect("present");
        assert_eq!(removed.id(), TabletId::from_u64(1));
        assert!(set.by_slot(slot).is_none());
        // The freed slot is reused, so the dense vector does not grow per split.
        assert_eq!(set.insert(live(3)), slot);
        assert_eq!(
            set.by_slot(slot).map(LiveTablet::id),
            Some(TabletId::from_u64(3))
        );
    }

    #[test]
    fn both_names_address_the_same_tablet() {
        let set = LocalTabletSet::from_tablets(vec![live(1), live(2), live(3)]);
        for (index, tablet) in [1_u64, 2, 3].into_iter().enumerate() {
            let id = TabletId::from_u64(tablet);
            let slot = set.slot_of(id).expect("present");
            assert_eq!(slot.index(), index);
            assert_eq!(set.by_slot(slot).map(LiveTablet::id), Some(id));
            assert_eq!(set.get(id).map(LiveTablet::id), Some(id));
        }
    }

    #[test]
    fn reinserting_an_id_does_not_leave_two_live_slots() {
        let mut set = LocalTabletSet::new();
        set.insert(live(1));
        set.insert(live(2));
        set.insert(live(1));
        let id = TabletId::from_u64(1);
        let slot = set.slot_of(id).expect("present");
        assert_eq!(set.by_slot(slot).map(LiveTablet::id), Some(id));
    }

    #[test]
    fn a_vacated_slot_reads_as_absent() {
        let mut set = LocalTabletSet::new();
        set.insert(live(1));
        set.remove(TabletId::from_u64(1));
        assert!(set.by_slot(LocalTabletSlot::new(0)).is_none());
        assert!(set.get(TabletId::from_u64(1)).is_none());
        assert!(!set.contains_key(TabletId::from_u64(1)));
    }
}
