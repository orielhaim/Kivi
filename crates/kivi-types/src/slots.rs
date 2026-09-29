//! The data-plane name of a tablet: a dense index into its worker's storage.
//!
//! A tablet has two names and conflating them is what cost a lookup per request.
//! [`TabletId`] is the control-plane name - assigned by the directory, mapped to a
//! worker by the placement, carried in a routing publication, quoted in a log
//! record and in a client-visible error. `LocalTabletSlot` is the data-plane name:
//! issued by the worker that owns the tablet, meaningful only inside that worker,
//! and resolved by indexing rather than searching.
//!
//! It lives here, beside [`TabletId`], rather than in the worker that issues it,
//! because a compiled route has to *cross* the RESP crate boundary: the connection
//! resolves the route once per drain and hands it to the engine per command. A
//! route is a pair of identities, and this is where identities are defined. The
//! alternative is a `u32` with a comment, which is the encoding this newtype exists
//! to prevent - the whole point of the type is that a worker id or a commit
//! position cannot be passed where a slot belongs.

use core::fmt;

use crate::ids::TabletId;

/// A dense index into one worker's tablet storage.
///
/// `u32` because a worker cannot own four billion tablets; the widest realistic
/// deployment is three orders of magnitude below that. The bound is asserted at
/// the point of issue rather than assumed, so a truncating cast cannot quietly
/// alias two tablets onto one slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LocalTabletSlot {
    dense: u32,
}

impl LocalTabletSlot {
    /// Wraps a dense index.
    #[must_use]
    pub const fn new(index: u32) -> Self {
        Self { dense: index }
    }

    /// The dense index, for indexing a slot vector.
    #[must_use]
    pub const fn index(self) -> usize {
        self.dense as usize
    }
}

impl fmt::Display for LocalTabletSlot {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "slot#{}", self.dense)
    }
}

/// A tablet route with the lookup already done.
///
/// This is the whole of "compile routing completely": a request does not ask
/// *which tablet* and then find it, it is handed the answer. `std`'s
/// `RandomState` costs 142 instructions per request to hash a `TabletId` back out
/// of a map, and that was replacing a bounds check.
///
/// `Copy` because it is passed per command and resolved per turn. Making it a
/// reference would tie the reply to a borrow of the publication; making it an
/// engine-defined associated type would stop [`Executor`](crate) being
/// object-safe and cost `Box<dyn Executor>` callers the direct path entirely.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TabletRoute {
    tablet: TabletId,
    slot: LocalTabletSlot,
}

impl TabletRoute {
    /// Pairs a tablet with the slot that addresses it locally.
    #[must_use]
    pub const fn new(tablet: TabletId, slot: LocalTabletSlot) -> Self {
        Self { tablet, slot }
    }

    /// The control-plane name, for a log record, an error, or a handoff.
    #[must_use]
    pub const fn tablet(self) -> TabletId {
        self.tablet
    }

    /// The data-plane name. Resolving a route is now this.
    #[must_use]
    pub const fn slot(self) -> LocalTabletSlot {
        self.slot
    }
}
