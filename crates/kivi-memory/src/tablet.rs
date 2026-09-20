//! Tablet migration / split / merge interaction.
//!
//! Object materializations are fenced by tablet epoch (WriteGuard-style):
//! every envelope carries the epoch the bytes were captured under, and a
//! stale epoch is rejected on import instead of resurrecting dead state.
//! Migration exports logical bytes (never handles: handles are
//! worker-local and never cross the wire); the destination re-materializes
//! them under the new epoch. Split partitions the exported set by key
//! order; merge concatenates sibling sets. All three preserve logical
//! state exactly while physical placement restarts from DRAM.

use bytes::Bytes;

/// One object's migration envelope: logical bytes plus fencing metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TabletEnvelope {
    /// Fabric-scoped object id on the source (informational; the
    /// destination assigns its own ids).
    pub object: u64,
    /// Logical version.
    pub version: u64,
    /// Tablet epoch captured under.
    pub epoch: u64,
    /// Logical bytes.
    pub bytes: Vec<u8>,
}

/// A tablet's materializations prepared for movement.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TabletMaterializations {
    /// Source tablet identifier (opaque to the fabric).
    pub tablet: u64,
    /// Epoch captured under.
    pub epoch: u64,
    /// Envelopes in key order.
    pub envelopes: Vec<TabletEnvelope>,
}

impl TabletMaterializations {
    /// Creates an empty set for a tablet at an epoch.
    #[must_use]
    pub const fn new(tablet: u64, epoch: u64) -> Self {
        Self {
            tablet,
            epoch,
            envelopes: Vec::new(),
        }
    }

    /// Adds one envelope. Panics in tests only when epochs mix; production
    /// callers use [`try_add`](Self::try_add).
    ///
    /// # Panics
    ///
    /// Panics when the envelope's epoch differs from the set's epoch.
    /// Production callers use [`try_add`](Self::try_add) instead.
    pub fn add(&mut self, envelope: TabletEnvelope) {
        if let Err(detail) = self.try_add(envelope) {
            panic!("{detail}");
        }
    }

    /// Adds one envelope, rejecting epoch mismatches.
    ///
    /// # Errors
    ///
    /// Returns [`crate::error::MemoryError::Invalid`] when the envelope's
    /// epoch differs from the set's epoch.
    pub fn try_add(&mut self, envelope: TabletEnvelope) -> Result<(), crate::error::MemoryError> {
        if envelope.epoch != self.epoch {
            return Err(crate::error::MemoryError::Invalid {
                detail: "tablet envelope epoch mismatch".to_owned(),
            });
        }
        self.envelopes.push(envelope);
        Ok(())
    }

    /// Splits the set at `at` (by position in key order) into two sets
    /// for child tablets. Both children inherit the epoch; the directory
    /// assigns new tablet ids and epochs on activation.
    #[must_use]
    pub fn split_at(&self, at: usize, left_tablet: u64, right_tablet: u64) -> (Self, Self) {
        let at = at.min(self.envelopes.len());
        let mut left = Self::new(left_tablet, self.epoch);
        let mut right = Self::new(right_tablet, self.epoch);
        left.envelopes.extend_from_slice(&self.envelopes[..at]);
        right.envelopes.extend_from_slice(&self.envelopes[at..]);
        (left, right)
    }

    /// Merges sibling sets captured under the same epoch.
    ///
    /// # Errors
    ///
    /// Returns [`crate::error::MemoryError::Invalid`] when epochs differ.
    pub fn merge(
        left: &Self,
        right: &Self,
        merged_tablet: u64,
    ) -> Result<Self, crate::error::MemoryError> {
        if left.epoch != right.epoch {
            return Err(crate::error::MemoryError::Invalid {
                detail: "merge requires equal epochs".to_owned(),
            });
        }
        let mut merged = Self::new(merged_tablet, left.epoch);
        merged.envelopes.extend_from_slice(&left.envelopes);
        merged.envelopes.extend_from_slice(&right.envelopes);
        Ok(merged)
    }

    /// Total logical bytes in the set.
    #[must_use]
    pub fn logical_bytes(&self) -> u64 {
        self.envelopes
            .iter()
            .map(|env| env.bytes.len() as u64)
            .sum()
    }

    /// Imports envelopes into a fabric, rejecting stale epochs.
    ///
    /// `current_epoch` is the destination's active epoch; envelopes from
    /// an older epoch are skipped (stale owner) and counted.
    pub fn import_into(
        &self,
        fabric: &mut crate::fabric::MemoryFabric,
        current_epoch: u64,
        intent: &crate::intent::MaterializationIntent,
        class: crate::handle::BehaviorClass,
    ) -> (usize, usize) {
        if self.epoch != current_epoch {
            return (0, self.envelopes.len());
        }
        let mut imported = 0;
        for envelope in &self.envelopes {
            if fabric
                .insert(
                    Bytes::from(envelope.bytes.clone()),
                    intent.clone(),
                    class,
                    crate::criticality::Mutability::Mutable,
                )
                .is_ok()
            {
                imported += 1;
            }
        }
        (imported, self.envelopes.len() - imported)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fabric::{MemoryFabric, MemoryFabricConfig};
    use crate::handle::BehaviorClass;
    use crate::intent::MaterializationIntent;

    #[test]
    fn split_merge_roundtrip_preserves_bytes() {
        let set = TabletMaterializations {
            tablet: 1,
            epoch: 9,
            envelopes: (0u8..10)
                .map(|index| TabletEnvelope {
                    object: u64::from(index),
                    version: 1,
                    epoch: 9,
                    bytes: vec![index],
                })
                .collect(),
        };
        let (left, right) = set.split_at(4, 2, 3);
        assert_eq!(left.envelopes.len(), 4);
        assert_eq!(right.envelopes.len(), 6);
        let merged = TabletMaterializations::merge(&left, &right, 4).expect("merge");
        assert_eq!(merged.envelopes, set.envelopes);
    }

    #[test]
    fn stale_epoch_import_is_rejected() {
        let fabric_config = MemoryFabricConfig::default();
        let mut fabric = MemoryFabric::new(fabric_config).expect("fabric");
        let set = TabletMaterializations {
            tablet: 1,
            epoch: 8,
            envelopes: vec![TabletEnvelope {
                object: 1,
                version: 1,
                epoch: 8,
                bytes: b"hello".to_vec(),
            }],
        };
        let (imported, skipped) = set.import_into(
            &mut fabric,
            9,
            &MaterializationIntent::hot(),
            BehaviorClass::HotMutable,
        );
        assert_eq!((imported, skipped), (0, 1));
        let (imported, _) = set.import_into(
            &mut fabric,
            8,
            &MaterializationIntent::hot(),
            BehaviorClass::HotMutable,
        );
        assert_eq!(imported, 1);
    }
}
