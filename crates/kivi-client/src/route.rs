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

/// Tablet-keyed leader hint cache for replicated clusters.
///
/// The sparse [`RouteCache`] below routes single-node ranges; a replicated
/// deployment has one leader per tablet, potentially different per tablet
/// in the future. This cache is conceptually keyed by [`TabletId`] even
/// though the current product replicates a single tablet: hints learned
/// from `StaleRoute` redirects update here, and the next request prefers
/// the cached leader endpoint before falling back to seeds. Bounded retry
/// and same-identity preservation live in the client execute loop; this
/// type only remembers where the leader was last seen.
#[derive(Debug, Clone, Default)]
pub struct TabletLeaderCache {
    inner: Arc<ArcSwap<Vec<(TabletId, String)>>>,
}

impl TabletLeaderCache {
    /// Creates an empty leader cache.
    #[must_use]
    pub fn new() -> Self {
        Self {
            inner: Arc::new(ArcSwap::from_pointee(Vec::new())),
        }
    }

    /// Remembers the leader endpoint for `tablet`.
    pub fn insert(&self, tablet: TabletId, endpoint: String) {
        let current = self.inner.load();
        let mut next: Vec<(TabletId, String)> = current
            .iter()
            .filter(|(id, _)| *id != tablet)
            .cloned()
            .collect();
        next.push((tablet, endpoint));
        self.inner.store(Arc::new(next));
    }

    /// Returns the last-known leader endpoint for `tablet`, if any.
    #[must_use]
    pub fn lookup(&self, tablet: TabletId) -> Option<String> {
        self.inner
            .load()
            .iter()
            .find_map(|(id, endpoint)| (*id == tablet).then(|| endpoint.clone()))
    }

    /// Drops every hint dialing a dead endpoint (connection refused or
    /// reset): the next request re-discovers through seeds and fresh
    /// redirects instead of hammering a grave.
    pub fn evict_endpoint(&self, endpoint: &str) {
        let current = self.inner.load();
        if !current.iter().any(|(_, known)| known == endpoint) {
            return;
        }
        let next: Vec<(TabletId, String)> = current
            .iter()
            .filter(|(_, known)| *known != endpoint)
            .cloned()
            .collect();
        self.inner.store(Arc::new(next));
    }

    /// Returns any cached leader endpoint (single-tablet fast path: the
    /// only entry when exactly one tablet is replicated).
    #[must_use]
    pub fn any(&self) -> Option<String> {
        self.inner
            .load()
            .first()
            .map(|(_, endpoint)| endpoint.clone())
    }

    /// Number of cached tablet leaders (introspection for tests).
    #[must_use]
    pub fn len(&self) -> usize {
        self.inner.load().len()
    }

    /// Whether no leader hints are cached yet.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.inner.load().is_empty()
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

    /// Finds the first cached route covering `key`, if any (ordered
    /// namespaces). Hash ranges never match a key lookup: those route by
    /// partition hash, never by raw bytes.
    #[must_use]
    pub fn lookup_key(&self, key: &[u8]) -> Option<RouteEntry> {
        self.entries.load().iter().find_map(|entry| {
            let covers = match &entry.range {
                PartitionRange::Ordered(range) => range.contains(key),
                PartitionRange::Hash(_) => false,
            };
            covers.then(|| entry.clone())
        })
    }

    /// Inserts a learned tablet range directly (scan pages and prepare
    /// responses teach full ranges, not just points).
    pub fn insert_range(
        &self,
        range: PartitionRange,
        tablet: TabletId,
        epoch: TabletEpoch,
        worker: WorkerId,
        endpoint: String,
        dir_version: u64,
    ) {
        self.insert(RouteEntry {
            range,
            tablet,
            epoch,
            worker,
            endpoint,
            dir_version,
        });
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

    /// Drops every route dialing a dead endpoint (connection refused or
    /// reset): placement moves and killed members converge through
    /// seeds and fresh redirects instead of a cached grave. Only the
    /// affected tablets re-discover (§34); the rest of the cache stays.
    pub fn evict_endpoint(&self, endpoint: &str) {
        let current = self.entries.load();
        if !current.iter().any(|entry| entry.endpoint == endpoint) {
            return;
        }
        let next: Vec<RouteEntry> = current
            .iter()
            .filter(|entry| entry.endpoint != endpoint)
            .cloned()
            .collect();
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
    fn tablet_leaders_are_keyed_by_tablet() {
        let cache = TabletLeaderCache::new();
        assert!(cache.is_empty());
        let a = TabletId::from_u64(9);
        let b = TabletId::from_u64(10);
        cache.insert(a, "127.0.0.1:9201".to_owned());
        assert_eq!(cache.lookup(a).as_deref(), Some("127.0.0.1:9201"));
        assert_eq!(cache.lookup(b), None);
        assert_eq!(cache.any().as_deref(), Some("127.0.0.1:9201"));
        cache.insert(a, "127.0.0.1:9202".to_owned());
        assert_eq!(cache.len(), 1, "same tablet replaces its hint");
        assert_eq!(cache.lookup(a).as_deref(), Some("127.0.0.1:9202"));
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
