//! Sparse route cache: learn ranges as they are used, never download the
//! full directory.
//!
//! Lookup is insertion-ordered first match over cached ranges; updates evict
//! same-kind overlaps before inserting, so the cache always reflects the
//! newest authority information per range. Replacement is copy-on-write
//! behind `ArcSwap`: readers never lock, writers clone a small vector.

use std::sync::Arc;

use arc_swap::ArcSwap;
use kivi_protocol::{RedirectInfo, RouteHint};
use kivi_tablet::PartitionRange;
use kivi_types::{PartitionHash, TabletEpoch, TabletId, WorkerId};

/// One learned authority route: the range plus how to dial it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RouteEntry {
    /// Owned range (a redirect's range, cached whole).
    pub range: PartitionRange,
    /// Authoritative tablet.
    pub tablet: TabletId,
    /// Its epoch.
    pub epoch: TabletEpoch,
    /// Its owner worker.
    pub worker: WorkerId,
    /// Dialable worker endpoint.
    pub endpoint: String,
    /// Directory version that published it (rejects obvious staleness).
    pub dir_version: u64,
}

impl RouteEntry {
    /// Builds a cache entry from redirect info.
    #[must_use]
    pub fn from_redirect(info: &RedirectInfo) -> Self {
        Self {
            range: info.range.clone(),
            tablet: info.tablet,
            epoch: info.epoch,
            worker: info.worker,
            endpoint: info.endpoint.clone(),
            dir_version: info.dir_version.as_u64(),
        }
    }

    /// Builds the request hint carried to the worker.
    #[must_use]
    pub fn hint(&self) -> RouteHint {
        RouteHint {
            tablet: self.tablet,
            epoch: self.epoch,
            worker: self.worker,
        }
    }
}

/// Copy-on-write sparse route cache shared across cloned clients.
#[derive(Debug, Clone, Default)]
pub struct RouteCache {
    entries: Arc<ArcSwap<Vec<RouteEntry>>>,
}

impl RouteCache {
    /// Creates an empty cache.
    #[must_use]
    pub fn new() -> Self {
        Self {
            entries: Arc::new(ArcSwap::from_pointee(Vec::new())),
        }
    }

    /// Finds the first cached route covering `hash`, if any.
    #[must_use]
    pub fn lookup(&self, hash: PartitionHash) -> Option<RouteEntry> {
        self.entries.load().iter().find_map(|entry| {
            let covers = match (&entry.range, hash) {
                (PartitionRange::Hash(prefix), hash) => prefix.contains(hash),
                (PartitionRange::Ordered(_), _) => false,
            };
            covers.then(|| entry.clone())
        })
    }

    /// Inserts a learned route, evicting same-kind overlaps first so newer
    /// authority information always wins per range.
    pub fn insert(&self, entry: RouteEntry) {
        let current = self.entries.load();
        let mut next: Vec<RouteEntry> = current
            .iter()
            .filter(|existing| {
                existing.range.kind() != entry.range.kind()
                    || !existing.range.overlaps(&entry.range)
            })
            .cloned()
            .collect();
        next.push(entry);
        self.entries.store(Arc::new(next));
    }

    /// Returns the number of cached routes (introspection for tests).
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.load().len()
    }

    /// Whether no routes are cached yet.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.load().is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kivi_tablet::HashPrefix;

    fn entry(bits: u128, len: u8, worker: u64, version: u64) -> RouteEntry {
        RouteEntry {
            range: PartitionRange::Hash(HashPrefix::new(bits, len).expect("prefix")),
            tablet: TabletId::from_u64(1),
            epoch: TabletEpoch::from_u64(2),
            worker: WorkerId::from_u64(worker),
            endpoint: format!("127.0.0.1:{}", 9100 + worker),
            dir_version: version,
        }
    }

    fn hash(value: u128) -> PartitionHash {
        PartitionHash::from_u128(value)
    }

    #[test]
    fn miss_then_learn_then_hit() {
        let cache = RouteCache::new();
        assert!(cache.is_empty());
        assert_eq!(cache.lookup(hash(0)), None);
        cache.insert(entry(0, 1, 0, 7));
        let hit = cache.lookup(hash(0)).expect("covers low half");
        assert_eq!(hit.worker, WorkerId::from_u64(0));
        assert_eq!(hit.dir_version, 7);
        assert_eq!(cache.lookup(hash(u128::MAX)), None);
    }

    #[test]
    fn newer_overlaps_evict_older_same_kind() {
        let cache = RouteCache::new();
        cache.insert(entry(0, 1, 0, 7));
        cache.insert(entry(0, 2, 1, 8));
        assert_eq!(cache.len(), 1, "overlapping prefix evicted");
        assert_eq!(
            cache.lookup(hash(0)).expect("hit").worker,
            WorkerId::from_u64(1)
        );
    }
}
