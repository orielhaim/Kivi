//! Desired vs actual placement.
//!
//! For every tablet the control plane owns a [`DesiredReplicaSet`];
//! the tablet's own Raft membership remains authoritative for its actual
//! voter state. Migration reconciles actual toward desired — placement is
//! never claimed changed merely because the control plane requested it.
//!
//! `PlacementVersion` counts placement transitions. It is distinct from
//! the key-range [`DirectoryVersion`](kivi_tablet_docs): changing replica
//! placement and changing key-range topology are different operations and
//! must not share one counter. (This crate owns the placement counter;
//! the directory crate keeps owning the range counter.)

/// Placeholder link target for the directory-version docs above.
#[doc(hidden)]
pub mod kivi_tablet_docs {}

use std::collections::BTreeSet;

use kivi_types::{NodeId, TabletId};

/// Monotonic placement version: every committed desired-placement change
/// advances it by exactly one. Stale control actions carrying an older
/// version must not mutate newer placement.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PlacementVersion(pub u64);

impl PlacementVersion {
    /// Invalid sentinel: no placement was ever published at version zero.
    pub const INVALID: Self = Self(0);
    /// First version, assigned at cluster bootstrap.
    pub const INITIAL: Self = Self(1);

    /// Wraps a raw version value.
    #[must_use]
    pub const fn from_u64(value: u64) -> Self {
        Self(value)
    }

    /// Returns the raw value.
    #[must_use]
    pub const fn as_u64(self) -> u64 {
        self.0
    }

    /// Successor version.
    ///
    /// # Errors
    ///
    /// Returns [`PlacementError::VersionExhausted`] at `u64::MAX`.
    pub const fn next(self) -> Result<Self, PlacementError> {
        match self.0.checked_add(1) {
            Some(next) => Ok(Self(next)),
            None => Err(PlacementError::VersionExhausted),
        }
    }
}

/// Why a desired placement was rejected.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum PlacementError {
    /// Placement names no replicas.
    #[error("tablet {tablet} names no replicas")]
    EmptyReplicas {
        /// The affected tablet.
        tablet: u64,
    },
    /// Placement repeats a voter.
    #[error("tablet {tablet} repeats voter {node}")]
    DuplicateVoter {
        /// The affected tablet.
        tablet: u64,
        /// The repeated voter.
        node: u64,
    },
    /// Version counter exhausted.
    #[error("placement version space exhausted")]
    VersionExhausted,
    /// Truncated canonical input.
    #[error("truncated placement")]
    Truncated,
    /// Trailing bytes after the framed placement.
    #[error("trailing bytes in placement")]
    TrailingBytes,
}

/// Desired voter set for one tablet, at one placement version.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DesiredReplicaSet {
    /// Tablet placed.
    pub tablet: TabletId,
    /// Ordered desired voters (replication and quorum order follow this).
    pub replicas: Vec<NodeId>,
    /// Placement version that published this set.
    pub version: PlacementVersion,
}

impl DesiredReplicaSet {
    /// Builds a desired set, validating uniqueness and nonemptiness.
    ///
    /// # Errors
    ///
    /// Returns [`PlacementError`] on empty or duplicate voter sets.
    pub fn new(
        tablet: TabletId,
        mut replicas: Vec<NodeId>,
        version: PlacementVersion,
    ) -> Result<Self, PlacementError> {
        if replicas.is_empty() {
            return Err(PlacementError::EmptyReplicas {
                tablet: tablet.as_u64(),
            });
        }
        let mut sorted: Vec<u64> = replicas.iter().map(|node| node.as_u64()).collect();
        sorted.sort_unstable();
        for pair in sorted.windows(2) {
            if pair[0] == pair[1] {
                return Err(PlacementError::DuplicateVoter {
                    tablet: tablet.as_u64(),
                    node: pair[0],
                });
            }
        }
        replicas.sort_by_key(|node: &NodeId| node.as_u64());
        Ok(Self {
            tablet,
            replicas,
            version,
        })
    }

    /// Returns the voter set.
    #[must_use]
    pub fn voter_set(&self) -> BTreeSet<u64> {
        self.replicas.iter().map(|node| node.as_u64()).collect()
    }

    /// Whether `node` is a desired voter.
    #[must_use]
    pub fn contains(&self, node: NodeId) -> bool {
        self.replicas.contains(&node)
    }

    /// Encodes the canonical bytes: `tablet u64, version u64,
    /// count u32, replicas[count] u64`.
    #[must_use]
    pub fn encode_to_vec(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(8 + 8 + 4 + 8 * self.replicas.len());
        out.extend_from_slice(&self.tablet.as_u64().to_le_bytes());
        out.extend_from_slice(&self.version.as_u64().to_le_bytes());
        let count = u32::try_from(self.replicas.len()).unwrap_or(u32::MAX);
        out.extend_from_slice(&count.to_le_bytes());
        for replica in &self.replicas {
            out.extend_from_slice(&replica.as_u64().to_le_bytes());
        }
        out
    }

    /// Decodes exactly one set.
    ///
    /// # Errors
    ///
    /// Returns [`PlacementError`] on truncation, trailing bytes, or
    /// incoherent voter sets.
    pub fn decode_exact(input: &[u8]) -> Result<Self, PlacementError> {
        use PlacementError as Fault;
        if input.len() < 8 + 8 + 4 {
            return Err(Fault::Truncated);
        }
        let tablet =
            TabletId::from_u64(u64::from_le_bytes(input[..8].try_into().unwrap_or([0; 8])));
        let version = PlacementVersion::from_u64(u64::from_le_bytes(
            input[8..16].try_into().unwrap_or([0; 8]),
        ));
        let count = u32::from_le_bytes(input[16..20].try_into().unwrap_or([0; 4])) as usize;
        if input.len() != 20 + 8 * count {
            if input.len() < 20 + 8 * count {
                return Err(Fault::Truncated);
            }
            return Err(Fault::TrailingBytes);
        }
        let mut replicas = Vec::with_capacity(count);
        for index in 0..count {
            let at = 20 + 8 * index;
            replicas.push(NodeId::from_u64(u64::from_le_bytes(
                input[at..at + 8].try_into().unwrap_or([0; 8]),
            )));
        }
        Self::new(tablet, replicas, version)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn desired_set_round_trips_sorted() {
        let set = DesiredReplicaSet::new(
            TabletId::from_u64(17),
            vec![
                NodeId::from_u64(3),
                NodeId::from_u64(1),
                NodeId::from_u64(2),
            ],
            PlacementVersion::INITIAL,
        )
        .expect("valid");
        assert_eq!(
            set.replicas,
            vec![
                NodeId::from_u64(1),
                NodeId::from_u64(2),
                NodeId::from_u64(3)
            ]
        );
        let back = DesiredReplicaSet::decode_exact(&set.encode_to_vec()).expect("decodes");
        assert_eq!(back, set);
    }

    #[test]
    fn empty_set_fails() {
        assert!(matches!(
            DesiredReplicaSet::new(TabletId::from_u64(1), vec![], PlacementVersion::INITIAL),
            Err(PlacementError::EmptyReplicas { .. })
        ));
    }

    #[test]
    fn versions_advance() {
        assert_eq!(
            PlacementVersion::INITIAL.next().expect("next"),
            PlacementVersion::from_u64(2)
        );
        assert!(matches!(
            PlacementVersion::from_u64(u64::MAX).next(),
            Err(PlacementError::VersionExhausted)
        ));
    }
}
