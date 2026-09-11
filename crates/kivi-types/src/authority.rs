//! Tablet authority and fencing checks (RFC §70, §194, §251).
//!
//! Every authoritative tablet range carries a ([`TabletId`], [`TabletEpoch`],
//! [`WriteGuardGeneration`]) triple. Writes must present the current triple;
//! anything older — or for the wrong tablet — is rejected. The check is a
//! pure function so it behaves identically on every replica and in simulation.

use core::fmt;

use crate::ids::{TabletEpoch, TabletId, WriteGuardGeneration};

/// The full fencing identity of one authoritative tablet range.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TabletAuthority {
    tablet: TabletId,
    epoch: TabletEpoch,
    guard: WriteGuardGeneration,
}

impl TabletAuthority {
    /// Builds an authority triple.
    #[must_use]
    pub const fn new(tablet: TabletId, epoch: TabletEpoch, guard: WriteGuardGeneration) -> Self {
        Self {
            tablet,
            epoch,
            guard,
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

    /// Returns the write-guard generation half.
    #[must_use]
    pub const fn guard(self) -> WriteGuardGeneration {
        self.guard
    }

    /// Validates that `self` (presented by a writer) is authorized against the
    /// `current` authority.
    ///
    /// Checks run in fencing hierarchy order — tablet, then epoch, then
    /// generation — so the first reported mismatch is the most significant.
    /// An old owner may still execute code, but a stale triple can never
    /// authorize a mutation (RFC §194).
    ///
    /// # Errors
    ///
    /// Returns [`AuthorityMismatch`] describing the first failed check.
    pub fn check_against(&self, current: &Self) -> Result<(), AuthorityMismatch> {
        if self.tablet != current.tablet {
            return Err(AuthorityMismatch::WrongTablet {
                presented: self.tablet,
                current: current.tablet,
            });
        }
        if self.epoch != current.epoch {
            return Err(AuthorityMismatch::StaleEpoch {
                presented: self.epoch,
                current: current.epoch,
            });
        }
        if self.guard != current.guard {
            return Err(AuthorityMismatch::StaleGuard {
                presented: self.guard,
                current: current.guard,
            });
        }
        Ok(())
    }
}

impl fmt::Display for TabletAuthority {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "tablet {} epoch {} guard {}",
            self.tablet, self.epoch, self.guard
        )
    }
}

/// Reason a presented [`TabletAuthority`] was rejected.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, thiserror::Error)]
#[non_exhaustive]
pub enum AuthorityMismatch {
    /// The authority belongs to a different tablet (misrouted write).
    #[error("wrong tablet: presented {presented}, current authority is {current}")]
    WrongTablet {
        /// Tablet the writer presented.
        presented: TabletId,
        /// Tablet that actually holds authority here.
        current: TabletId,
    },
    /// The epoch is not the current one (stale owner after a topology change).
    #[error("stale tablet epoch: presented {presented}, current is {current}")]
    StaleEpoch {
        /// Epoch the writer presented.
        presented: TabletEpoch,
        /// Current epoch.
        current: TabletEpoch,
    },
    /// The write-guard generation is not the current one.
    #[error("stale write-guard generation: presented {presented}, current is {current}")]
    StaleGuard {
        /// Generation the writer presented.
        presented: WriteGuardGeneration,
        /// Current generation.
        current: WriteGuardGeneration,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    fn current() -> TabletAuthority {
        TabletAuthority::new(
            TabletId::from_u64(918),
            TabletEpoch::from_u64(381),
            WriteGuardGeneration::from_u64(12),
        )
    }

    #[test]
    fn current_authority_passes() {
        assert_eq!(current().check_against(&current()), Ok(()));
    }

    #[test]
    fn stale_epoch_is_rejected() {
        let stale = TabletAuthority::new(
            TabletId::from_u64(918),
            TabletEpoch::from_u64(380),
            WriteGuardGeneration::from_u64(12),
        );
        assert_eq!(
            stale.check_against(&current()),
            Err(AuthorityMismatch::StaleEpoch {
                presented: TabletEpoch::from_u64(380),
                current: TabletEpoch::from_u64(381),
            })
        );
    }

    #[test]
    fn stale_guard_is_rejected() {
        let stale = TabletAuthority::new(
            TabletId::from_u64(918),
            TabletEpoch::from_u64(381),
            WriteGuardGeneration::from_u64(11),
        );
        assert!(matches!(
            stale.check_against(&current()),
            Err(AuthorityMismatch::StaleGuard { .. })
        ));
    }

    #[test]
    fn wrong_tablet_is_rejected_before_epoch() {
        let misrouted = TabletAuthority::new(
            TabletId::from_u64(919),
            TabletEpoch::from_u64(1),
            WriteGuardGeneration::from_u64(1),
        );
        assert!(matches!(
            misrouted.check_against(&current()),
            Err(AuthorityMismatch::WrongTablet { .. })
        ));
    }

    #[test]
    fn zeroed_authority_never_authorizes() {
        let zeroed = TabletAuthority::new(
            TabletId::from_u64(918),
            TabletEpoch::INVALID,
            WriteGuardGeneration::INVALID,
        );
        assert!(zeroed.check_against(&current()).is_err());
    }

    #[test]
    fn display_mentions_all_three_halves() {
        let text = current().to_string();
        assert!(text.contains("918"));
        assert!(text.contains("381"));
    }
}
