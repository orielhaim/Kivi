//! Commit positions and tokens (RFC §66, §167).
//!
//! Per-tablet ordering is identified by (`TabletId`, [`TabletEpoch`],
//! [`CommitPosition`]). A [`CommitToken`] names one committed mutation and is
//! what `AtLeast` reads wait for; no global total order is fabricated.

use core::fmt;

use crate::ids::{CommitPosition, TabletEpoch, TabletId};

/// Names one committed mutation within one tablet's ordered log.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CommitToken {
    tablet: TabletId,
    epoch: TabletEpoch,
    position: CommitPosition,
}

impl CommitToken {
    /// Builds a token from its three parts.
    #[must_use]
    pub const fn new(tablet: TabletId, epoch: TabletEpoch, position: CommitPosition) -> Self {
        Self {
            tablet,
            epoch,
            position,
        }
    }

    /// Returns the tablet half.
    #[must_use]
    pub const fn tablet(self) -> TabletId {
        self.tablet
    }

    /// Returns the epoch half.
    #[must_use]
    pub const fn epoch(self) -> TabletEpoch {
        self.epoch
    }

    /// Returns the log-position half.
    #[must_use]
    pub const fn position(self) -> CommitPosition {
        self.position
    }

    /// Whether the token names a real committed entry: the position must be
    /// assigned and the epoch valid. The write-guard generation is execution
    /// fencing and deliberately not part of a commit token.
    #[must_use]
    pub const fn is_assigned(self) -> bool {
        self.position.is_assigned() && self.epoch.is_valid()
    }
}

impl fmt::Display for CommitToken {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "tablet {} epoch {} position {}",
            self.tablet, self.epoch, self.position
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn assigned_token_holds_all_parts() {
        let token = CommitToken::new(
            TabletId::from_u64(5),
            TabletEpoch::from_u64(2),
            CommitPosition::from_u64(100),
        );
        assert!(token.is_assigned());
        assert_eq!(token.tablet(), TabletId::from_u64(5));
        assert_eq!(token.position(), CommitPosition::from_u64(100));
    }

    #[test]
    fn unassigned_position_or_invalid_epoch_is_not_assigned() {
        let no_slot = CommitToken::new(
            TabletId::from_u64(5),
            TabletEpoch::INITIAL,
            CommitPosition::UNASSIGNED,
        );
        assert!(!no_slot.is_assigned());
        let no_epoch = CommitToken::new(
            TabletId::from_u64(5),
            TabletEpoch::INVALID,
            CommitPosition::FIRST,
        );
        assert!(!no_epoch.is_assigned());
    }

    #[test]
    fn tokens_order_by_tablet_epoch_then_position() {
        let a = CommitToken::new(
            TabletId::from_u64(1),
            TabletEpoch::INITIAL,
            CommitPosition::from_u64(2),
        );
        let b = CommitToken::new(
            TabletId::from_u64(1),
            TabletEpoch::INITIAL,
            CommitPosition::from_u64(3),
        );
        assert!(a < b);
    }
}
