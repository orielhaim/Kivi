//! Live tablets: runtime mutable tablet replicas owned by exactly one worker.
//!
//! A [`LiveTablet`] pairs immutable directory metadata (identity, authority)
//! with its mutable [`ObjectStore`]. It is deliberately distinct from
//! [`TabletDescriptor`]: descriptors are
//! shared routing facts, live tablets are single-owner execution state.
//! There is no lock anywhere in this path — the owning worker thread is the
//! only mutator, and execution is plain synchronous Rust once a request
//! arrives.
//!
//! Execution mirrors the future replicated path exactly: [`prepare`](LiveTablet::prepare)
//! validates, [`apply_mutation`](LiveTablet::apply_mutation) commits, and
//! [`execute`](LiveTablet::execute) runs both atomically for the single-node
//! engine. Expiry reclamation also flows through explicit [`Delete`](kivi_state::Mutation::Delete)
//! mutations via [`sweep_expired`](LiveTablet::sweep_expired), so a future
//! consensus layer can propose the same mutations without changing tablet logic.

use core::fmt;

use kivi_state::{
    ApplyError, ApplyOutcome, Key, Mutation, ObjectStore, OpError, Operation, OperationResult,
    Prepared,
};
use kivi_tablet::TabletDescriptor;
use kivi_types::{TabletAuthority, TabletId, UnixMicros};

/// Local per-tablet counters. Plain integers behind the owner thread —
/// aggregation across workers happens outside the hot path.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TabletMetrics {
    /// Operations executed (reads and writes).
    pub ops_total: u64,
    /// Read operations executed.
    pub reads: u64,
    /// Mutations applied.
    pub writes: u64,
    /// Expired objects reclaimed through explicit delete mutations.
    pub expired_reclaimed: u64,
}

/// Live tablet failure modes.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum TabletError {
    /// The authority triple names a different tablet than constructed.
    #[error("authority {authority} does not name tablet {tablet}")]
    AuthorityMismatch {
        /// Tablet the live instance was built for.
        tablet: TabletId,
        /// Authority triple supplied.
        authority: TabletAuthority,
    },
    /// Operation validation failed; nothing was mutated.
    #[error("operation rejected: {0}")]
    Op(#[from] OpError),
    /// Mutation application failed; see the variant for what held.
    #[error("mutation failed: {0}")]
    Apply(#[from] ApplyError),
}

/// One mutable tablet replica. `Send` (it moves between threads only at
/// construction/migration boundaries) but used from exactly one thread at a
/// time — the owner enforces that, not the type system.
pub struct LiveTablet {
    id: TabletId,
    authority: TabletAuthority,
    store: ObjectStore,
    metrics: TabletMetrics,
}

impl LiveTablet {
    /// Builds a live tablet from directory metadata and a matching authority.
    /// Starts empty; state arrives through normal mutation application
    /// (single-node: direct execution; future: consensus replay / catch-up).
    ///
    /// # Errors
    ///
    /// Returns [`TabletError::AuthorityMismatch`] if `authority` names a
    /// different tablet than `descriptor` describes.
    pub fn from_descriptor(
        descriptor: &TabletDescriptor,
        authority: TabletAuthority,
    ) -> Result<Self, TabletError> {
        if authority.tablet() != descriptor.id() {
            return Err(TabletError::AuthorityMismatch {
                tablet: descriptor.id(),
                authority,
            });
        }
        Ok(Self {
            id: descriptor.id(),
            authority,
            store: ObjectStore::new(),
            metrics: TabletMetrics::default(),
        })
    }

    /// Returns the tablet identity.
    #[must_use]
    pub const fn id(&self) -> TabletId {
        self.id
    }

    /// Returns the current authority triple (for fencing checks upstream).
    #[must_use]
    pub const fn authority(&self) -> TabletAuthority {
        self.authority
    }

    /// Returns the underlying object store (read-only inspection).
    #[must_use]
    pub fn store(&self) -> &ObjectStore {
        &self.store
    }

    /// Returns a copy of the local metrics.
    #[must_use]
    pub const fn metrics(&self) -> TabletMetrics {
        self.metrics
    }

    /// Validates an operation without mutating (replica-proposal shape).
    ///
    /// # Errors
    ///
    /// Returns [`TabletError::Op`] for validation failures.
    pub fn prepare(&self, op: &Operation, now: UnixMicros) -> Result<Prepared, TabletError> {
        Ok(self.store.prepare(op, now)?)
    }

    /// Applies one validated mutation (replica-apply shape).
    ///
    /// # Errors
    ///
    /// Returns [`TabletError::Apply`] when versions cannot advance, counters
    /// overflow, or types diverge.
    pub fn apply_mutation(
        &mut self,
        mutation: &Mutation,
        now: UnixMicros,
    ) -> Result<ApplyOutcome, TabletError> {
        let outcome = self.store.apply(mutation, now)?;
        self.metrics.writes += 1;
        self.metrics.ops_total += 1;
        Ok(outcome)
    }

    /// Executes one native operation atomically: prepare, then apply when the
    /// operation is a write. This is the single-node fast path; consensus
    /// will split the two phases across proposer and replicas later.
    ///
    /// # Errors
    ///
    /// Returns [`TabletError`] for validation or application failures;
    /// failures never partially mutate.
    ///
    /// # Panics
    ///
    /// Panics if preparation and application disagree on the outcome shape
    /// (e.g. a `Set` yielding a counter outcome). The two phases run
    /// back-to-back on the same state with no interleaving, so disagreement
    /// would indicate an internal logic bug rather than a runtime condition;
    /// the panic makes that unmistakable instead of returning a wrong result.
    pub fn execute(
        &mut self,
        op: &Operation,
        now: UnixMicros,
    ) -> Result<OperationResult, TabletError> {
        match self.store.prepare(op, now)? {
            Prepared::Read(result) => {
                self.metrics.reads += 1;
                self.metrics.ops_total += 1;
                Ok(result)
            }
            Prepared::Write(mutation) => {
                let outcome = self.store.apply(&mutation, now)?;
                self.metrics.writes += 1;
                self.metrics.ops_total += 1;
                Ok(match (op, outcome) {
                    (Operation::Set { .. }, ApplyOutcome::Put { version }) => {
                        OperationResult::Stored { version }
                    }
                    (Operation::Delete { .. }, ApplyOutcome::Deleted { existed }) => {
                        OperationResult::Deleted { existed }
                    }
                    (Operation::CounterAdd { .. }, ApplyOutcome::Counter { value, version }) => {
                        OperationResult::CounterUpdated { value, version }
                    }
                    (Operation::ExpireAt { .. }, ApplyOutcome::Expiry { applied, .. }) => {
                        OperationResult::ExpirySet { applied }
                    }
                    (Operation::PersistExpiry { .. }, ApplyOutcome::Expiry { applied, .. }) => {
                        OperationResult::ExpiryPersisted { removed: applied }
                    }
                    (op, outcome) => {
                        panic!("prepare/apply contract violated: {op:?} -> {outcome:?}")
                    }
                })
            }
        }
    }

    /// Reclaims up to `limit` expired objects through explicit delete
    /// mutations, returning each reclaimed key with its outcome. Bounded by
    /// `limit` so reclamation never starves foreground work; the driver
    /// repeats sweeps until `collect_expired` comes back empty.
    pub fn sweep_expired(&mut self, now: UnixMicros, limit: usize) -> Vec<(Key, ApplyOutcome)> {
        let mut reclaimed = Vec::new();
        for key in self.store.collect_expired(now, limit) {
            // Delete of a swept key cannot fail (no error paths exist for it);
            // a hypothetical failure would simply leave the key for the next
            // sweep, so skipping is the safe direction.
            if let Ok(outcome) = self
                .store
                .apply(&Mutation::Delete { key: key.clone() }, now)
            {
                self.metrics.writes += 1;
                self.metrics.ops_total += 1;
                self.metrics.expired_reclaimed += 1;
                reclaimed.push((key, outcome));
            }
        }
        reclaimed
    }
}

impl fmt::Debug for LiveTablet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LiveTablet")
            .field("id", &self.id)
            .field("authority", &self.authority)
            .field("store", &self.store)
            .field("metrics", &self.metrics)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kivi_state::{Key, LogicalValue, ObjectVersion};
    use kivi_tablet::{DirectorySnapshot, PartitionRange};
    use kivi_types::{NamespaceId, TabletEpoch, WriteGuardGeneration};

    const NS: NamespaceId = NamespaceId::from_u64(1);

    fn live() -> LiveTablet {
        let tablet = TabletId::from_u64(1);
        let authority =
            TabletAuthority::new(tablet, TabletEpoch::INITIAL, WriteGuardGeneration::INITIAL);
        let snapshot = DirectorySnapshot::bootstrap(
            NS,
            tablet,
            PartitionRange::Hash(kivi_tablet::HashPrefix::new(0, 0).expect("root")),
            TabletEpoch::INITIAL,
            WriteGuardGeneration::INITIAL,
        )
        .expect("genesis");
        let descriptor = snapshot.get(tablet).expect("descriptor").clone();
        LiveTablet::from_descriptor(&descriptor, authority).expect("live tablet")
    }

    const NOW: UnixMicros = UnixMicros::from_micros(1_000_000);

    #[test]
    fn authority_must_name_the_tablet() {
        let tablet = live();
        let wrong = TabletAuthority::new(
            TabletId::from_u64(999),
            TabletEpoch::INITIAL,
            WriteGuardGeneration::INITIAL,
        );
        let snapshot = DirectorySnapshot::bootstrap(
            NS,
            TabletId::from_u64(1),
            PartitionRange::Hash(kivi_tablet::HashPrefix::new(0, 0).expect("root")),
            TabletEpoch::INITIAL,
            WriteGuardGeneration::INITIAL,
        )
        .expect("genesis");
        let descriptor = snapshot.get(TabletId::from_u64(1)).expect("descriptor");
        assert!(matches!(
            LiveTablet::from_descriptor(descriptor, wrong),
            Err(TabletError::AuthorityMismatch { .. })
        ));
        assert_eq!(tablet.id(), TabletId::from_u64(1));
    }

    #[test]
    fn execute_runs_typed_operations_and_counts_metrics() {
        let mut tablet = live();
        tablet
            .execute(
                &Operation::Set {
                    key: Key::from("k"),
                    value: bytes::Bytes::from_static(b"v"),
                },
                NOW,
            )
            .expect("set");
        tablet
            .execute(
                &Operation::CounterAdd {
                    key: Key::from("n"),
                    delta: 3,
                },
                NOW,
            )
            .expect("add");
        tablet
            .execute(
                &Operation::Get {
                    key: Key::from("k"),
                },
                NOW,
            )
            .expect("get");
        let metrics = tablet.metrics();
        assert_eq!(metrics.ops_total, 3);
        assert_eq!(metrics.reads, 1);
        assert_eq!(metrics.writes, 2);
        assert_eq!(metrics.expired_reclaimed, 0);
    }

    #[test]
    fn prepare_apply_split_matches_direct_execute() {
        let mut direct = live();
        let mut split = live();
        let op = Operation::CounterAdd {
            key: Key::from("n"),
            delta: 7,
        };
        let expected = direct.execute(&op, NOW).expect("direct");
        let prepared = split.prepare(&op, NOW).expect("prepare");
        let Prepared::Write(mutation) = prepared else {
            panic!("counter add must prepare a mutation");
        };
        split.apply_mutation(&mutation, NOW).expect("apply");
        assert_eq!(
            split.store().snapshot_sorted(),
            direct.store().snapshot_sorted()
        );
        assert_eq!(
            expected,
            OperationResult::CounterUpdated {
                value: 7,
                version: ObjectVersion::FIRST
            }
        );
    }

    #[test]
    fn sweep_reclaims_through_explicit_mutations() {
        let mut tablet = live();
        for name in ["a", "b", "keep"] {
            tablet
                .execute(
                    &Operation::Set {
                        key: Key::from(name),
                        value: bytes::Bytes::from_static(b"v"),
                    },
                    NOW,
                )
                .expect("set");
            tablet
                .execute(
                    &Operation::ExpireAt {
                        key: Key::from(name),
                        expires_at: UnixMicros::from_micros(10_000_000),
                    },
                    NOW,
                )
                .expect("expire");
        }
        // Keep one alive by persisting before the deadline.
        tablet
            .execute(
                &Operation::PersistExpiry {
                    key: Key::from("keep"),
                },
                NOW,
            )
            .expect("persist");
        let later = UnixMicros::from_micros(20_000_000);
        let reclaimed = tablet.sweep_expired(later, 1);
        assert_eq!(reclaimed.len(), 1, "sweep honors its bound");
        let reclaimed = tablet.sweep_expired(later, 100);
        assert_eq!(reclaimed.len(), 1);
        assert_eq!(tablet.metrics().expired_reclaimed, 2);
        assert_eq!(tablet.store().live_count(later), 1);
        assert!(matches!(
            tablet.store().get(&Key::from("keep"), later),
            Some(object) if matches!(object.value(), LogicalValue::Bytes(_))
        ));
    }
}
