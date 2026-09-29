//! Transition targets and the in-flight marker.
//!
//! The fabric plans a representation change by writing a
//! [`MoveRequest`](crate::movement::MoveRequest) into its bounded queue and
//! marking the object as moving. Execution is the move drain's job; this
//! module names the destinations and holds the marker the planner,
//! reclaimer, and mutator consult to skip an object whose representation is
//! in flight.

/// Where a representation change is heading.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TransitionTarget {
    /// Decoded DRAM arena residence.
    Dram,
    /// Compressed DRAM representation.
    Compressed,
    /// `NVMe` demotion.
    Nvme,
    /// Full eviction (requires a reconstruction source).
    Evicted,
}

impl TransitionTarget {
    /// Residence tag the transition publishes (for planner bookkeeping).
    #[must_use]
    pub const fn publishes(self) -> &'static str {
        match self {
            Self::Dram => "arena",
            Self::Compressed => "compressed",
            Self::Nvme => "nvme",
            Self::Evicted => "evicted",
        }
    }
}

/// An object whose representation is in flight.
///
/// Carries the version the move started from and the target it is heading
/// for, so a completion for a stale version is fenced rather than applied
/// over newer bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Transition {
    /// Which object is moving.
    pub object: u64,
    /// Logical version the move started from.
    pub build_version: u64,
    /// Where it is heading.
    pub target: TransitionTarget,
}

impl Transition {
    /// Begins a transition for an object at a logical version.
    #[must_use]
    pub const fn begin(object: u64, version: u64, target: TransitionTarget) -> Self {
        Self {
            object,
            build_version: version,
            target,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_transition_records_its_object_version_and_target() {
        let transition = Transition::begin(9, 7, TransitionTarget::Nvme);
        assert_eq!(transition.object, 9);
        assert_eq!(transition.build_version, 7);
        assert_eq!(transition.target.publishes(), "nvme");
    }
}
