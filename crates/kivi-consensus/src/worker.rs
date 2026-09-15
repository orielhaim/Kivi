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
//! 1, so group 1 lands on worker 0). It is a stable arithmetic function —
//! never randomized `HashMap` hashing — documented here and shared by open
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

/// Returns the tablets owned by `worker` out of `tablets` (tablet ids
/// `1..=tablet_count`), in tablet-id order. Empty when the worker owns
/// nothing (more workers than tablets).
///
/// # Panics
///
/// Panics when `worker_count` is zero.
#[must_use]
pub fn tablets_for_worker(
    worker: usize,
    worker_count: usize,
    tablet_count: usize,
) -> Vec<TabletId> {
    assert!(worker_count > 0, "worker count must be nonzero");
    (1..=tablet_count as u64)
        .map(TabletId::from_u64)
        .filter(|tablet| worker_for_tablet(*tablet, worker_count) == worker)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mapping_is_stable_round_robin() {
        // 4 workers, 8 tablets: worker i owns tablets i+1, i+5.
        for tablet in 1..=8u64 {
            let worker = worker_for_tablet(TabletId::from_u64(tablet), 4);
            assert_eq!(
                worker,
                (usize::try_from(tablet).unwrap_or(usize::MAX) - 1) % 4
            );
        }
        assert_eq!(
            tablets_for_worker(0, 4, 8),
            vec![TabletId::from_u64(1), TabletId::from_u64(5)]
        );
        assert_eq!(
            tablets_for_worker(3, 4, 8),
            vec![TabletId::from_u64(4), TabletId::from_u64(8)]
        );
    }

    #[test]
    fn single_worker_owns_everything() {
        assert_eq!(tablets_for_worker(0, 1, 16).len(), 16);
    }

    #[test]
    fn extra_workers_own_nothing() {
        assert!(tablets_for_worker(7, 8, 4).is_empty());
    }

    #[test]
    #[should_panic(expected = "worker count must be nonzero")]
    fn zero_workers_panics() {
        let _ = worker_for_tablet(TabletId::from_u64(1), 0);
    }
}
