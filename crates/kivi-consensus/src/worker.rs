//! Consensus worker ownership: deterministic tablet-to-worker mapping.
//!
//! The multi-tablet engine shards tablet groups across a fixed number of
//! Compio reactor threads (consensus workers):
//!
//! ```text
//! Node
//!  ├─ ConsensusWorker 0 ── Tablet group 1, 1+W, 1+2W, …
//!  ├─ ConsensusWorker 1 ── Tablet group 2, 2+W, 2+2W, …
//!  └─ …
//! ```
//!
//! ## Mapping
//!
//! [`worker_for_tablet`] is the single deterministic local mapping from
//! tablet to worker: `(tablet_id - 1) % worker_count` (tablet ids start at
//! 1, so group 1 lands on worker 0). It is a stable arithmetic function -
//! never randomized `HashMap` hashing - documented here and shared by open
//! and routing. The assignment only determines local physical ownership
//! (which reactor drives the group's `Raft`); it is not distributed tablet
//! placement, and different nodes may use different worker counts without
//! affecting correctness.
//!
//! ## Ownership
//!
//! A `Raft` handle never migrates between threads: each worker owns its
//! groups' handles pinned to its own Compio reactor, and cross-thread
//! callers communicate through bounded worker ingress channels. No locks
//! wrap `Raft` internals.

use kivi_types::TabletId;

/// Returns the owning worker index for `tablet` under `worker_count`
/// workers: `(tablet_id - 1) % worker_count`.
///
/// # Panics
///
/// Panics when `worker_count` is zero (a configuration error the caller
/// validates loudly at startup).
#[must_use]
pub fn worker_for_tablet(tablet: TabletId, worker_count: usize) -> usize {
    assert!(worker_count > 0, "worker count must be nonzero");
    // Tablet ids start at 1 (genesis); subtract one so group 1 lands on
    // worker 0 and consecutive tablets stripe round-robin.
    usize::try_from(tablet.as_u64().saturating_sub(1)).unwrap_or(usize::MAX) % worker_count
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 4 workers, 8 tablets: worker `i` owns tablets `i+1` and `i+5`.
    #[test]
    fn mapping_is_stable_round_robin() {
        let owned = |worker: usize| -> Vec<TabletId> {
            (1..=8u64)
                .map(TabletId::from_u64)
                .filter(|tablet| worker_for_tablet(*tablet, 4) == worker)
                .collect()
        };
        assert_eq!(owned(0), vec![TabletId::from_u64(1), TabletId::from_u64(5)]);
        assert_eq!(owned(3), vec![TabletId::from_u64(4), TabletId::from_u64(8)]);
    }

    #[test]
    #[should_panic(expected = "worker count must be nonzero")]
    fn zero_workers_panics() {
        let _ = worker_for_tablet(TabletId::from_u64(1), 0);
    }
}
