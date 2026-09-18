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
    /// Whether this contract promises linearizable (strong) semantics:
    /// `Latest` always does; `AtLeast(token)` does for the token's lineage
    /// (it never returns state before the token). `BoundedStale` and `Any`
    /// are explicitly weak and must never be served as if strong.
    #[must_use]
    pub const fn is_strong(self) -> bool {
        match self {
            Self::Latest | Self::AtLeast(_) => true,
            Self::BoundedStale { .. } | Self::Any => false,
        }
    }

    /// Stable metric discriminant for [`ConsistencyMetrics`](crate::ConsistencyMetrics):
    /// `Latest = 0`, `AtLeast = 1`, `BoundedStale = 2`, `Any = 3`.
    /// Discriminants only, never wire tags (wire tags live in `kivi-codec`
    /// and `kivi-protocol`, which own their own versioning).
    #[must_use]
    pub const fn metric_discriminant(self) -> u8 {
        match self {
            Self::Latest => 0,
            Self::AtLeast(_) => 1,
            Self::BoundedStale { .. } => 2,
            Self::Any => 3,
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
    fn strong_contracts_are_latest_and_at_least() {
        assert!(ReadContract::Latest.is_strong());
        let token = CommitToken::new(
            TabletId::from_u64(1),
            TabletEpoch::INITIAL,
            CommitPosition::FIRST,
        );
        assert!(ReadContract::AtLeast(token).is_strong());
        assert!(
            !ReadContract::BoundedStale {
                max_staleness: Duration::from_millis(100)
            }
            .is_strong()
        );
        assert!(!ReadContract::Any.is_strong());
    }

    #[test]
    fn metric_discriminants_are_stable() {
        assert_eq!(ReadContract::Latest.metric_discriminant(), 0);
        let token = CommitToken::new(
            TabletId::from_u64(1),
            TabletEpoch::INITIAL,
            CommitPosition::FIRST,
        );
        assert_eq!(ReadContract::AtLeast(token).metric_discriminant(), 1);
        assert_eq!(
            ReadContract::BoundedStale {
                max_staleness: Duration::from_millis(100)
            }
            .metric_discriminant(),
            2
        );
        assert_eq!(ReadContract::Any.metric_discriminant(), 3);
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
