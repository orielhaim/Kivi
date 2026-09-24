//! Layout catalog: control-plane authority over current layouts.
//!
//! The catalog answers one question: *which layout generation is current
//! for this asset?* The control plane owns that answer (propose + state
//! read); the redundancy plane only consumes it through [`LayoutCatalog`]
//! and advances it through [`CatalogPublish`]. [`AssetId`] stays independent
//! of layouts — no layout field ever enters the identity — so assets keep
//! their names across generations, schemes, and repairs.
//!
//! [`MemCatalog`] is the in-process authority for tests (max-generation
//! fencing, stale rejection); the server implements both traits over the
//! control-plane propose path with a blocking wait. [`merge_discovery`]
//! classifies holder probes against authority for observability only.

use std::collections::HashMap;
use std::sync::Mutex;

use kivi_codec::integrity::blake3_256;

use crate::{AssetId, RedundancyError};

/// One published layout: the control generation that published it plus the
/// exact layout that generation installed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PublishedLayout {
    /// Control-plane generation that published this layout (fencing).
    pub control_generation: u64,
    /// The published layout (asset, scheme, fragments, generation).
    pub layout: crate::RedundancyLayout,
}

/// Read side of the catalog: which layout is current for an asset.
pub trait LayoutCatalog: Send + Sync + std::fmt::Debug {
    /// Returns the current published layout, if any.
    fn get(&self, asset: &AssetId) -> Option<PublishedLayout>;

    /// Returns the current published layouts known to this catalog.
    ///
    /// Incremental maintenance callers use this as a bounded refresh source;
    /// the default keeps small custom catalogs source-compatible.
    fn list(&self) -> Vec<PublishedLayout> {
        Vec::new()
    }
}

/// Publish side of the catalog: advance or drop the current layout.
///
/// The server implements this over control propose + state read with a
/// blocking wait; fencing is max-generation-wins (stale control
/// generations are rejected, never merged).
pub trait CatalogPublish: Send + Sync + std::fmt::Debug {
    /// Publishes a layout, returning the accepted current record.
    ///
    /// # Errors
    ///
    /// Returns [`RedundancyError::StaleGeneration`] when the record's
    /// control generation is fenced by a newer publish (or ties one with
    /// conflicting bytes).
    fn publish(&self, record: PublishedLayout) -> Result<PublishedLayout, RedundancyError>;

    /// Drops the whole asset's catalog entry (idempotent).
    ///
    /// # Errors
    ///
    /// Returns [`RedundancyError`] only on internal failure; absent assets
    /// report success.
    fn drop_asset(&self, asset: &AssetId) -> Result<(), RedundancyError>;
}

/// In-process catalog authority for tests: max-control-generation wins,
/// stale publishes rejected, idempotent retries accepted.
#[derive(Debug, Default)]
pub struct MemCatalog {
    /// Current records by asset.
    current: Mutex<HashMap<AssetId, PublishedLayout>>,
}

impl MemCatalog {
    /// Creates an empty catalog.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

impl LayoutCatalog for MemCatalog {
    fn get(&self, asset: &AssetId) -> Option<PublishedLayout> {
        let Ok(guard) = self.current.lock() else {
            return None;
        };
        guard.get(asset).cloned()
    }

    fn list(&self) -> Vec<PublishedLayout> {
        let Ok(guard) = self.current.lock() else {
            return Vec::new();
        };
        let mut layouts: Vec<PublishedLayout> = guard.values().cloned().collect();
        layouts.sort_by_key(|record| record.layout.asset);
        layouts
    }
}

impl CatalogPublish for MemCatalog {
    fn publish(&self, record: PublishedLayout) -> Result<PublishedLayout, RedundancyError> {
        let mut guard = self
            .current
            .lock()
            .map_err(|_| RedundancyError::Unreadable {
                detail: "catalog lock poisoned".to_owned(),
            })?;
        if let Some(current) = guard.get(&record.layout.asset) {
            if record.control_generation < current.control_generation {
                return Err(RedundancyError::StaleGeneration {
                    generation: record.control_generation,
                    current: current.control_generation,
                });
            }
            if record.control_generation == current.control_generation {
                if record.layout == current.layout {
                    return Ok(current.clone());
                }
                return Err(RedundancyError::StaleGeneration {
                    generation: record.control_generation,
                    current: current.control_generation,
                });
            }
            if record.layout.generation < current.layout.generation {
                return Err(RedundancyError::StaleGeneration {
                    generation: record.layout.generation,
                    current: current.layout.generation,
                });
            }
        }
        guard.insert(record.layout.asset, record.clone());
        Ok(record)
    }

    fn drop_asset(&self, asset: &AssetId) -> Result<(), RedundancyError> {
        let mut guard = self
            .current
            .lock()
            .map_err(|_| RedundancyError::Unreadable {
                detail: "catalog lock poisoned".to_owned(),
            })?;
        guard.remove(asset);
        Ok(())
    }
}

/// What holder probes say about control authority (observability only:
/// discovery never overrides the catalog).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Discovery {
    /// Probed holders agree with the control-plane layout (or nothing
    /// contradicts it).
    Current {
        /// Agreed generation.
        generation: u64,
    },
    /// No control record and no observations.
    Unknown,
    /// Observations contradict authority — or data exists with no
    /// authority at all. Holds a human-readable cause.
    Divergent {
        /// Human-readable cause.
        detail: String,
    },
}

/// Classifies holder probes (`generation`, layout-bytes hash) against the
/// control-plane record: agreement is [`Discovery::Current`], silence on
/// both sides is [`Discovery::Unknown`], and any contradiction — including
/// data with no authority — is [`Discovery::Divergent`].
#[must_use]
pub fn merge_discovery(control: Option<&PublishedLayout>, probed: &[(u64, [u8; 32])]) -> Discovery {
    let Some(current) = control else {
        if probed.is_empty() {
            return Discovery::Unknown;
        }
        return Discovery::Divergent {
            detail: format!("{} holder reports without control authority", probed.len()),
        };
    };
    let expect_hash = blake3_256(&current.layout.encode());
    let expect = (current.layout.generation, expect_hash);
    for (generation, hash) in probed {
        if (*generation, *hash) != expect {
            return Discovery::Divergent {
                detail: format!(
                    "holder reports generation {generation} against control {}",
                    current.layout.generation
                ),
            };
        }
    }
    Discovery::Current {
        generation: current.layout.generation,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn layout(generation: u64) -> crate::RedundancyLayout {
        let asset = AssetId::new(crate::AssetKind::Chunk, 7, [11; 32]).expect("asset");
        let params = crate::SchemeParams::Replication(crate::ReplicationParams { copies: 2 });
        let fragments = (0..2u32)
            .map(|index| crate::FragmentRecord {
                id: crate::FragmentId::for_bytes(asset, generation, index, params, b"bytes"),
                node: kivi_types::NodeId::from_u64(u64::from(index) + 1),
                stored_len: 5,
            })
            .collect();
        crate::RedundancyLayout::new(asset, 5, generation, params, fragments).expect("builds")
    }

    fn asset() -> AssetId {
        AssetId::new(crate::AssetKind::Chunk, 7, [11; 32]).expect("asset")
    }

    #[test]
    fn max_generation_wins_and_stale_rejected() {
        let catalog = MemCatalog::new();
        assert!(catalog.get(&asset()).is_none());
        let first = PublishedLayout {
            control_generation: 5,
            layout: layout(1),
        };
        let accepted = catalog.publish(first.clone()).expect("publishes");
        assert_eq!(accepted, first);
        // Idempotent retry of identical bytes succeeds.
        assert_eq!(catalog.publish(first).expect("idempotent"), accepted);
        // Stale control generations fail closed.
        let stale = PublishedLayout {
            control_generation: 3,
            layout: layout(1),
        };
        let error = catalog.publish(stale).expect_err("stale rejected");
        assert!(matches!(error, RedundancyError::StaleGeneration { .. }));
        // Fresh control wins, even superseding generations.
        let second = PublishedLayout {
            control_generation: 9,
            layout: layout(2),
        };
        catalog.publish(second.clone()).expect("wins");
        assert_eq!(catalog.get(&asset()), Some(second));
        // Whole-asset drops are idempotent.
        catalog.drop_asset(&asset()).expect("drops");
        assert!(catalog.get(&asset()).is_none());
        catalog.drop_asset(&asset()).expect("idempotent");
    }

    #[test]
    fn discovery_classifies_current_unknown_divergent() {
        assert_eq!(merge_discovery(None, &[]), Discovery::Unknown);
        assert!(matches!(
            merge_discovery(None, &[(1, [2; 32])]),
            Discovery::Divergent { .. }
        ));
        let current = PublishedLayout {
            control_generation: 4,
            layout: layout(2),
        };
        let hash = blake3_256(&current.layout.encode());
        assert_eq!(
            merge_discovery(Some(&current), &[(2, hash)]),
            Discovery::Current { generation: 2 }
        );
        assert_eq!(
            merge_discovery(Some(&current), &[]),
            Discovery::Current { generation: 2 }
        );
        assert!(matches!(
            merge_discovery(Some(&current), &[(2, [9; 32])]),
            Discovery::Divergent { .. }
        ));
    }
}
