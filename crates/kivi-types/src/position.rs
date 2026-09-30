//! Commit positions and tokens.
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

    /// A commit token is assigned only when both its epoch and its position
    /// are real. An `AtLeast` read against a token that was never assigned
    /// would otherwise compare against the sentinel as if it were a real
    /// position, admitting reads the token cannot justify.
    #[test]
    fn a_token_is_assigned_only_with_a_real_epoch_and_position() {
        let assigned = CommitToken::new(
            TabletId::from_u64(5),
            TabletEpoch::from_u64(2),
            CommitPosition::from_u64(100),
        );
        assert!(assigned.is_assigned());
        assert_eq!(assigned.tablet(), TabletId::from_u64(5));
        assert_eq!(assigned.position(), CommitPosition::from_u64(100));

        for unassigned in [
            CommitToken::new(
                TabletId::from_u64(5),
                TabletEpoch::INITIAL,
                CommitPosition::UNASSIGNED,
            ),
            CommitToken::new(
                TabletId::from_u64(5),
                TabletEpoch::INVALID,
                CommitPosition::FIRST,
            ),
        ] {
            assert!(
                !unassigned.is_assigned(),
                "{unassigned} claims to be assigned"
            );
        }
    }
}
