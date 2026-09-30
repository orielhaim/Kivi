//! Explicit read contracts.
//!
//! There is no ambiguous "read replica" mode: every native read names the
//! freshness it requires, and the engine picks the cheapest mechanism that
//! still satisfies it.

use core::fmt;
use core::time::Duration;

use crate::position::CommitToken;

/// Freshness contract of one read.
///
/// The four variants are the closed contract set from. Adding a new
/// contract is a consensus-layer semantic change, so this enum is
/// deliberately exhaustive: new variants break downstream matches (including
/// the canonical codec) at compile time, forcing an explicit wire tag
/// assignment instead of silent misinterpretation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReadContract {
    /// Linearizable read: establish a consensus read barrier, wait until local
    /// applied state reaches it, then read (conservative baseline).
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

    fn token() -> CommitToken {
        CommitToken::new(
            TabletId::from_u64(1),
            TabletEpoch::INITIAL,
            CommitPosition::FIRST,
        )
    }

    /// `is_strong` is the gate that decides whether a weak read may be served
    /// as if it were linearizable, and `metric_discriminant` is a stable
    /// per-contract counter that a dashboard groups by. Both are pinned
    /// together because a variant silently reclassified as strong would let a
    /// stale read pass as a fresh one.
    #[test]
    fn strength_and_metric_discriminant_are_pinned_per_variant() {
        let cases = [
            (ReadContract::Latest, true, 0u8),
            (ReadContract::AtLeast(token()), true, 1),
            (
                ReadContract::BoundedStale {
                    max_staleness: Duration::from_millis(100),
                },
                false,
                2,
            ),
            (ReadContract::Any, false, 3),
        ];
        for (contract, strong, discriminant) in cases {
            assert_eq!(contract.is_strong(), strong, "{contract} strength changed");
            assert_eq!(
                contract.metric_discriminant(),
                discriminant,
                "{contract} metric discriminant changed"
            );
        }
    }
}
