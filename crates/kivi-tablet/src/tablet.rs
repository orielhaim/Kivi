//! Tablet lifecycle, descriptors, and redirect metadata.
//!
//! A tablet identity moves through exactly one chain (RFC §190–§194):
//!
//! ```text
//! Allocated → Inactive → Active → Fenced → Tombstone
//!                  ↘________________↗
//!                  abort an unactivated build
//! ```
//!
//! * [`TabletState::Allocated`]: identity reserved by the control plane; no
//!   state built yet. Never writable.
//! * [`TabletState::Inactive`]: state constructed and catching up (split-child
//!   tail mirroring, migration-learner catch-up, bootstrap load) but not yet
//!   the authority. Never writable.
//! * [`TabletState::Active`]: the single writable authority for its range.
//! * [`TabletState::Fenced`]: sealed during replacement; writes stopped while
//!   the successor finishes replay and activates (RFC §190). Never writable,
//!   and never writable again.
//! * [`TabletState::Tombstone`]: retired; carries a [`Redirect`] naming live
//!   successors so stale routes resolve forward instead of failing.
//!
//! There are no backward edges and no boolean state flags: the enum *is* the
//! state, so illegal combinations such as "active but fenced" are
//! unrepresentable. Only [`TabletDescriptor::authority`] exposes a
//! [`TabletAuthority`], and only while active —
//! inactive, fenced, and tombstoned tablets cannot authorize mutations.

use core::fmt;

use kivi_types::{TabletAuthority, TabletEpoch, TabletId, WriteGuardGeneration};

use crate::range::PartitionRange;

/// Lifecycle state of one tablet identity.
///
/// See the module docs for the transition chain. All transitions are
/// performed by [`DirectorySnapshot`](crate::DirectorySnapshot) methods,
/// which additionally enforce range, overlap, successor, and version rules.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TabletState {
    /// Identity reserved; nothing built yet.
    Allocated,
    /// Built and catching up; not yet the authority.
    Inactive,
    /// Single writable authority for its range.
    Active,
    /// Sealed during replacement; writes stopped.
    Fenced,
    /// Retired; carries a redirect to live successors.
    Tombstone,
}

impl TabletState {
    /// Whether this state may authorize mutations (only [`Active`](Self::Active)).
    #[must_use]
    pub const fn is_writable(self) -> bool {
        matches!(self, Self::Active)
    }
}

impl fmt::Display for TabletState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Allocated => write!(f, "allocated"),
            Self::Inactive => write!(f, "inactive"),
            Self::Active => write!(f, "active"),
            Self::Fenced => write!(f, "fenced"),
            Self::Tombstone => write!(f, "tombstone"),
        }
    }
}

/// Forwarding record left on a tombstone (RFC §190 parent redirect, §191
/// merged-source redirect).
///
/// Names the live successor tablets that now own the retired range — a set,
/// not a single identity, because a split retires one parent into several
/// children. Directory transitions require every successor to exist and be
/// active at retirement time, so redirects always resolve forward.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Redirect {
    successors: Vec<TabletId>,
}

impl Redirect {
    /// Builds a redirect naming `successors`.
    ///
    /// Non-emptiness is enforced by directory retirement, not here, so that
    /// construction stays total and the directory remains the single
    /// validation point.
    #[must_use]
    pub fn new(successors: Vec<TabletId>) -> Self {
        Self { successors }
    }

    /// Returns the live successor tablets in deterministic order.
    #[must_use]
    pub fn successors(&self) -> &[TabletId] {
        &self.successors
    }

    /// Whether any successor is named.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.successors.is_empty()
    }
}

impl fmt::Display for Redirect {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "redirect[")?;
        for (index, successor) in self.successors.iter().enumerate() {
            if index > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{successor}")?;
        }
        write!(f, "]")
    }
}

/// Immutable metadata of one tablet identity: exact authority information
/// plus lifecycle state.
///
/// Descriptors are constructed only by
/// [`DirectorySnapshot`](crate::DirectorySnapshot) transitions, which uphold
/// the consistency contract: `redirect` is `Some` exactly for tombstones,
/// and epochs/generations are valid (nonzero) fencing values.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TabletDescriptor {
    id: TabletId,
    range: PartitionRange,
    state: TabletState,
    epoch: TabletEpoch,
    guard: WriteGuardGeneration,
    redirect: Option<Redirect>,
}

impl TabletDescriptor {
    /// Builds a descriptor. Only directory transitions call this; the
    /// redirect/tombstone consistency contract is enforced by
    /// [`DirectorySnapshot::validate`](crate::DirectorySnapshot::validate).
    pub(crate) fn new(
        id: TabletId,
        range: PartitionRange,
        state: TabletState,
        epoch: TabletEpoch,
        guard: WriteGuardGeneration,
        redirect: Option<Redirect>,
    ) -> Self {
        Self {
            id,
            range,
            state,
            epoch,
            guard,
            redirect,
        }
    }

    /// Returns the tablet identity.
    #[must_use]
    pub const fn id(&self) -> TabletId {
        self.id
    }

    /// Returns the owned partition range.
    #[must_use]
    pub fn range(&self) -> &PartitionRange {
        &self.range
    }

    /// Returns the lifecycle state.
    #[must_use]
    pub const fn state(&self) -> TabletState {
        self.state
    }

    /// Returns the fencing epoch.
    #[must_use]
    pub const fn epoch(&self) -> TabletEpoch {
        self.epoch
    }

    /// Returns the write-guard generation.
    #[must_use]
    pub const fn guard(&self) -> WriteGuardGeneration {
        self.guard
    }

    /// Returns the tombstone redirect, if retired.
    #[must_use]
    pub fn redirect(&self) -> Option<&Redirect> {
        self.redirect.as_ref()
    }

    /// Whether this tablet may authorize mutations (active only).
    #[must_use]
    pub const fn is_writable(&self) -> bool {
        self.state.is_writable()
    }

    /// The authority triple, available only while active.
    ///
    /// Returning `None` outside [`TabletState::Active`] is what makes
    /// inactive, fenced, and tombstoned tablets incapable of authorizing
    /// mutations: there is no triple to present.
    #[must_use]
    pub const fn authority(&self) -> Option<TabletAuthority> {
        match self.state {
            TabletState::Active => Some(TabletAuthority::new(self.id, self.epoch, self.guard)),
            TabletState::Allocated
            | TabletState::Inactive
            | TabletState::Fenced
            | TabletState::Tombstone => None,
        }
    }
}

impl fmt::Display for TabletDescriptor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "tablet {} {} epoch {} guard {}",
            self.id, self.state, self.epoch, self.guard
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kivi_types::TabletEpoch;

    fn descriptor(state: TabletState) -> TabletDescriptor {
        TabletDescriptor::new(
            TabletId::from_u64(918),
            PartitionRange::Hash(crate::range::HashPrefix::new(0, 0).expect("root")),
            state,
            TabletEpoch::INITIAL,
            WriteGuardGeneration::INITIAL,
            None,
        )
    }

    #[test]
    fn only_active_tablets_expose_authority() {
        let active = descriptor(TabletState::Active);
        let authority = active.authority().expect("active authorizes");
        assert!(active.is_writable());
        assert_eq!(authority.tablet(), TabletId::from_u64(918));

        for state in [
            TabletState::Allocated,
            TabletState::Inactive,
            TabletState::Fenced,
            TabletState::Tombstone,
        ] {
            let tablet = descriptor(state);
            assert!(!tablet.is_writable(), "{state} must not be writable");
            assert_eq!(tablet.authority(), None, "{state} must not authorize");
        }
    }

    #[test]
    fn redirect_names_successors_in_order() {
        let redirect = Redirect::new(vec![TabletId::from_u64(2), TabletId::from_u64(3)]);
        assert!(!redirect.is_empty());
        assert_eq!(
            redirect.successors(),
            &[TabletId::from_u64(2), TabletId::from_u64(3)]
        );
        assert!(Redirect::new(Vec::new()).is_empty());
    }

    #[test]
    fn states_display_distinctly() {
        assert_eq!(TabletState::Active.to_string(), "active");
        assert_eq!(TabletState::Tombstone.to_string(), "tombstone");
    }
}
