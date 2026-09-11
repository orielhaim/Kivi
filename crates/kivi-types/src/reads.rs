//! Explicit read contracts (RFC §66, §67).
//!
//! There is no ambiguous "read replica" mode: every native read names the
//! freshness it requires, and the engine picks the cheapest mechanism that
//! still satisfies it.

use core::fmt;
use core::time::Duration;

use crate::position::CommitToken;

/// Freshness contract of one read.
///
/// The four variants are the closed contract set from RFC §66. Adding a new
/// contract is a consensus-layer semantic change, so this enum is
/// deliberately exhaustive: new variants break downstream matches (including
/// the canonical codec) at compile time, forcing an explicit wire tag
/// assignment instead of silent misinterpretation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReadContract {
    /// Linearizable read: establish a consensus read barrier, wait until local
    /// applied state reaches it, then read (RFC §67, conservative baseline).
    Latest,
    /// Wait until local applied state reaches at least `0`, then read.
    AtLeast(CommitToken),
    /// Any state whose staleness is bounded by `max_staleness` is acceptable.
    BoundedStale {
        /// Maximum acceptable staleness as a pure span (no clock is read).
        max_staleness: Duration,
    },
    /// Any available state, including potentially stale caches.
    Any,
}

impl ReadContract {
    /// Whether serving this contract needs a consensus read barrier
    /// (strong path) rather than a directly servable replica or cache.
    #[must_use]
    pub const fn requires_barrier(self) -> bool {
        match self {
            Self::Latest | Self::AtLeast(_) => true,
            Self::BoundedStale { .. } | Self::Any => false,
        }
    }
}

impl fmt::Display for ReadContract {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Latest => write!(f, "latest"),
            Self::AtLeast(token) => write!(f, "at-least({token})"),
            Self::BoundedStale { max_staleness } => {
                write!(f, "bounded-stale({}ms)", max_staleness.as_millis())
            }
            Self::Any => write!(f, "any"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ids::{CommitPosition, TabletEpoch, TabletId};

    #[test]
    fn strong_contracts_need_a_barrier_weak_ones_do_not() {
        assert!(ReadContract::Latest.requires_barrier());
        let token = CommitToken::new(
            TabletId::from_u64(1),
            TabletEpoch::INITIAL,
            CommitPosition::FIRST,
        );
        assert!(ReadContract::AtLeast(token).requires_barrier());
        assert!(
            !ReadContract::BoundedStale {
                max_staleness: Duration::from_millis(100)
            }
            .requires_barrier()
        );
        assert!(!ReadContract::Any.requires_barrier());
    }

    #[test]
    fn display_names_each_contract() {
        assert_eq!(ReadContract::Latest.to_string(), "latest");
        assert_eq!(ReadContract::Any.to_string(), "any");
        assert_eq!(
            ReadContract::BoundedStale {
                max_staleness: Duration::from_millis(250)
            }
            .to_string(),
            "bounded-stale(250ms)"
        );
    }
}
