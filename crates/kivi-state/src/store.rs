//! Worker-local logical object store with deterministic prepare/apply.
//!
//! [`ObjectStore`] is a plain hash map from [`Key`] to [`StoredObject`],
//! built on `hashbrown` with a keyed `RandomState` (`SipHash`) hasher: hostile
//! user keys cannot degrade the table, and full byte equality always decides
//! membership. The hasher is an internal detail — iteration order is
//! unspecified, and every enumeration leaving the store is either aggregated
//! (counts) or sorted (sweeps, snapshots), so simulation replay and tests
//! never depend on it.
//!
//! State changes flow through two explicit phases (RFC Mutation IR direction):
//!
//! ```text
//! Operation → prepare() → Read(result) | Write(mutation) → apply() → outcome
//! ```
//!
//! `prepare` validates against current state (types, overflow, presence) and
//! `apply` executes a mutation deterministically from explicit inputs only
//! (`&Mutation`, `now`). Future replicas will run `apply` alone over the
//! replicated log; the single-node engine runs both phases atomically.

use core::fmt;
use std::collections::hash_map::RandomState;

use kivi_types::{Expiry, UnixMicros};

use crate::mutation::{ApplyError, ApplyOutcome, Mutation};
use crate::object::{Key, LogicalValue, ObjectType, ObjectVersion, StoredObject};
use crate::ops::{OpError, Operation, OperationResult};

/// A validated operation: either an immediately answered read or a mutation
/// awaiting deterministic application.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Prepared {
    /// Read-only outcome, computed during preparation.
    Read(OperationResult),
    /// Mutation to apply (locally now, or by replicas on replay).
    Write(Mutation),
}

/// Worker-local logical object map. Single-threaded by construction: the
/// owning worker is the only mutator, so no locks appear anywhere here.
#[derive(Debug, Clone, Default)]
pub struct ObjectStore {
    objects: hashbrown::HashMap<Key, StoredObject, RandomState>,
}

impl ObjectStore {
    /// Creates an empty store.
    #[must_use]
    pub fn new() -> Self {
        Self {
            objects: hashbrown::HashMap::default(),
        }
    }

    /// Returns the number of physically stored entries, including
    /// logically expired but not yet reclaimed objects.
    #[must_use]
    pub fn len(&self) -> usize {
        self.objects.len()
    }

    /// Whether no entries are physically stored.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.objects.is_empty()
    }

    /// Counts logically live objects at `now` (present and unexpired).
    #[must_use]
    pub fn live_count(&self, now: UnixMicros) -> usize {
        self.objects
            .values()
            .filter(|object| !object.is_expired(now))
            .count()
    }

    /// Returns the live object for `key` at `now` (`None` when missing or
    /// expired — reads never mutate, not even to clean up).
    #[must_use]
    pub fn get(&self, key: &Key, now: UnixMicros) -> Option<&StoredObject> {
        self.objects
            .get(key)
            .filter(|object| !object.is_expired(now))
    }

    /// Validates `op` against current state, answering reads immediately and
    /// translating writes into typed mutations. Pure: borrows state, reads
    /// the supplied `now`, touches nothing else.
    ///
    /// # Errors
    ///
    /// Returns [`OpError::WrongType`] for type mismatches and
    /// [`OpError::CounterOverflow`] for overflowing additions. Neither
    /// mutates anything.
    pub fn prepare(&self, op: &Operation, now: UnixMicros) -> Result<Prepared, OpError> {
        match op {
            Operation::Get { key } => match self.get(key, now) {
                None => Ok(Prepared::Read(OperationResult::Value(None))),
                Some(object) => match object.value() {
                    LogicalValue::Bytes(value) => {
                        Ok(Prepared::Read(OperationResult::Value(Some(value.clone()))))
                    }
                    LogicalValue::StrictCounter(_) => Err(OpError::WrongType {
                        expected: ObjectType::Bytes,
                        found: ObjectType::StrictCounter,
                    }),
                },
            },
            Operation::Set { key, value } => Ok(Prepared::Write(Mutation::PutBytes {
                key: key.clone(),
                value: value.clone(),
            })),
            Operation::Delete { key } => Ok(Prepared::Write(Mutation::Delete { key: key.clone() })),
            Operation::Exists { key } => Ok(Prepared::Read(OperationResult::Exists(
                self.get(key, now).is_some(),
            ))),
            Operation::CounterGet { key } => match self.get(key, now) {
                None => Ok(Prepared::Read(OperationResult::Counter(None))),
                Some(object) => match object.value() {
                    LogicalValue::StrictCounter(value) => {
                        Ok(Prepared::Read(OperationResult::Counter(Some(*value))))
                    }
                    LogicalValue::Bytes(_) => Err(OpError::WrongType {
                        expected: ObjectType::StrictCounter,
                        found: ObjectType::Bytes,
                    }),
                },
            },
            Operation::CounterAdd { key, delta } => match self.get(key, now) {
                None => Ok(Prepared::Write(Mutation::CounterAdd {
                    key: key.clone(),
                    delta: *delta,
                })),
                Some(object) => match object.value() {
                    LogicalValue::StrictCounter(value) => {
                        value.checked_add(*delta).ok_or(OpError::CounterOverflow)?;
                        Ok(Prepared::Write(Mutation::CounterAdd {
                            key: key.clone(),
                            delta: *delta,
                        }))
                    }
                    LogicalValue::Bytes(_) => Err(OpError::WrongType {
                        expected: ObjectType::StrictCounter,
                        found: ObjectType::Bytes,
                    }),
                },
            },
            Operation::ExpireAt { key, expires_at } => match self.get(key, now) {
                None => Ok(Prepared::Read(OperationResult::ExpirySet {
                    applied: false,
                })),
                Some(_) => Ok(Prepared::Write(Mutation::SetExpiry {
                    key: key.clone(),
                    expiry: Expiry::at(*expires_at),
                })),
            },
            Operation::PersistExpiry { key } => match self.get(key, now) {
                Some(object) if object.expiry() != Expiry::NEVER => {
                    Ok(Prepared::Write(Mutation::SetExpiry {
                        key: key.clone(),
                        expiry: Expiry::NEVER,
                    }))
                }
                _ => Ok(Prepared::Read(OperationResult::ExpiryPersisted {
                    removed: false,
                })),
            },
            Operation::GetExpiry { key } => Ok(Prepared::Read(OperationResult::Expiry(
                self.get(key, now).map(StoredObject::expiry),
            ))),
        }
    }

    /// Deterministically applies one mutation. The same mutation on the same
    /// state always yields the same outcome: no clock reads (caller supplies
    /// `now`), no randomness, no I/O.
    ///
    /// # Errors
    ///
    /// Returns [`ApplyError`] when versions cannot advance, a counter
    /// addition overflows, or a counter mutation meets a byte string (only
    /// reachable by applying mutations against divergent state — same log
    /// on same state never produces it).
    pub fn apply(
        &mut self,
        mutation: &Mutation,
        now: UnixMicros,
    ) -> Result<ApplyOutcome, ApplyError> {
        match mutation {
            Mutation::PutBytes { key, value } => {
                let version = self
                    .next_version(key, now)
                    .map_err(|_| ApplyError::VersionExhausted)?;
                self.objects.insert(
                    key.clone(),
                    StoredObject::new(LogicalValue::Bytes(value.clone()), version, Expiry::NEVER),
                );
                Ok(ApplyOutcome::Put { version })
            }
            Mutation::Delete { key } => {
                let existed = self.live(key, now).is_some();
                self.objects.remove(key);
                Ok(ApplyOutcome::Deleted { existed })
            }
            Mutation::CounterAdd { key, delta } => match self.live(key, now) {
                None => {
                    self.objects.insert(
                        key.clone(),
                        StoredObject::new(
                            LogicalValue::StrictCounter(*delta),
                            ObjectVersion::FIRST,
                            Expiry::NEVER,
                        ),
                    );
                    Ok(ApplyOutcome::Counter {
                        version: ObjectVersion::FIRST,
                        value: *delta,
                    })
                }
                Some(current) => match current.value() {
                    LogicalValue::StrictCounter(value) => {
                        let next = value
                            .checked_add(*delta)
                            .ok_or(ApplyError::CounterOverflow)?;
                        let version = self
                            .version_of(key)
                            .next()
                            .map_err(|_| ApplyError::VersionExhausted)?;
                        self.objects.insert(
                            key.clone(),
                            StoredObject::new(
                                LogicalValue::StrictCounter(next),
                                version,
                                self.expiry_of(key),
                            ),
                        );
                        Ok(ApplyOutcome::Counter {
                            version,
                            value: next,
                        })
                    }
                    LogicalValue::Bytes(_) => Err(ApplyError::TypeMismatch {
                        expected: ObjectType::StrictCounter,
                        found: ObjectType::Bytes,
                    }),
                },
            },
            Mutation::SetExpiry { key, expiry } => match self.live(key, now) {
                None => Ok(ApplyOutcome::Expiry {
                    applied: false,
                    version: None,
                }),
                Some(current) => {
                    let version = self
                        .version_of(key)
                        .next()
                        .map_err(|_| ApplyError::VersionExhausted)?;
                    self.objects.insert(
                        key.clone(),
                        StoredObject::new(current.value().clone(), version, *expiry),
                    );
                    Ok(ApplyOutcome::Expiry {
                        applied: true,
                        version: Some(version),
                    })
                }
            },
        }
    }

    /// Collects up to `limit` expired keys at `now`, sorted by key for
    /// deterministic cleanup order. The caller turns each into an explicit
    /// [`Delete`](Mutation::Delete) mutation — expiration is logical here,
    /// reclamation happens through the same mutation path as everything else.
    #[must_use]
    pub fn collect_expired(&self, now: UnixMicros, limit: usize) -> Vec<Key> {
        let mut expired: Vec<Key> = self
            .objects
            .iter()
            .filter(|(_, object)| object.is_expired(now))
            .map(|(key, _)| key.clone())
            .collect();
        expired.sort();
        expired.truncate(limit);
        expired
    }

    /// Snapshots all entries sorted by key. Deterministic regardless of hash
    /// order; used by tests, checkpoints, and future state transfer — never
    /// by the hot path.
    #[must_use]
    pub fn snapshot_sorted(&self) -> Vec<(Key, StoredObject)> {
        let mut entries: Vec<(Key, StoredObject)> = self
            .objects
            .iter()
            .map(|(key, object)| (key.clone(), object.clone()))
            .collect();
        entries.sort_by(|left, right| left.0.cmp(&right.0));
        entries
    }

    /// Live object ignoring nothing: present and unexpired at `now`.
    fn live(&self, key: &Key, now: UnixMicros) -> Option<&StoredObject> {
        self.get(key, now)
    }

    /// Current version of a stored entry (`FIRST` when absent — callers use
    /// this only after establishing presence, or for fresh creation).
    fn version_of(&self, key: &Key) -> ObjectVersion {
        self.objects
            .get(key)
            .map_or(ObjectVersion::FIRST, StoredObject::version)
    }

    /// Current expiry of a stored entry (`NEVER` when absent).
    fn expiry_of(&self, key: &Key) -> Expiry {
        self.objects
            .get(key)
            .map_or(Expiry::NEVER, StoredObject::expiry)
    }

    /// Next version for a key gaining new state: `FIRST` for fresh or
    /// expired-overwritten objects, checked successor for live ones.
    fn next_version(
        &self,
        key: &Key,
        now: UnixMicros,
    ) -> Result<ObjectVersion, crate::object::VersionExhausted> {
        match self.live(key, now) {
            None => Ok(ObjectVersion::FIRST),
            Some(object) => object.version().next(),
        }
    }
}

impl fmt::Display for ObjectStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ObjectStore({} objects)", self.objects.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::object::Key;
    use kivi_types::UnixMicros;

    const NOW: UnixMicros = UnixMicros::from_micros(1_000_000);

    fn key(name: &str) -> Key {
        Key::from(name)
    }

    /// Runs one operation end to end (prepare, then apply when required),
    /// propagating any failure as a debug string for test assertions.
    fn execute(
        store: &mut ObjectStore,
        op: &Operation,
        now: UnixMicros,
    ) -> Result<OperationResult, String> {
        match store
            .prepare(op, now)
            .map_err(|error| format!("{error:?}"))?
        {
            Prepared::Read(result) => Ok(result),
            Prepared::Write(mutation) => {
                let outcome = store.apply(&mutation, now).expect("apply must not fail");
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
                    (op, outcome) => panic!("unexpected {op:?} -> {outcome:?}"),
                })
            }
        }
    }

    #[test]
    fn bytes_set_get_overwrite_delete_exists() {
        let mut store = ObjectStore::new();
        assert!(store.is_empty());
        let stored = execute(
            &mut store,
            &Operation::Set {
                key: key("k"),
                value: bytes::Bytes::from_static(b"v1"),
            },
            NOW,
        )
        .expect("set");
        assert_eq!(
            stored,
            OperationResult::Stored {
                version: ObjectVersion::FIRST
            }
        );
        assert_eq!(store.len(), 1);
        assert_eq!(store.live_count(NOW), 1);
        assert_eq!(
            execute(&mut store, &Operation::Get { key: key("k") }, NOW).expect("get"),
            OperationResult::Value(Some(bytes::Bytes::from_static(b"v1")))
        );
        assert_eq!(
            execute(&mut store, &Operation::Exists { key: key("k") }, NOW).expect("exists"),
            OperationResult::Exists(true)
        );
        // Overwrite bumps the version and clears nothing else (no expiry set).
        assert_eq!(
            execute(
                &mut store,
                &Operation::Set {
                    key: key("k"),
                    value: bytes::Bytes::from_static(b"v2"),
                },
                NOW,
            )
            .expect("overwrite"),
            OperationResult::Stored {
                version: ObjectVersion::from_u64(2)
            }
        );
        assert_eq!(
            execute(&mut store, &Operation::Delete { key: key("k") }, NOW).expect("delete"),
            OperationResult::Deleted { existed: true }
        );
        assert_eq!(
            execute(&mut store, &Operation::Delete { key: key("k") }, NOW).expect("re-delete"),
            OperationResult::Deleted { existed: false }
        );
        assert_eq!(
            execute(&mut store, &Operation::Exists { key: key("k") }, NOW).expect("gone"),
            OperationResult::Exists(false)
        );
        assert!(store.is_empty());
    }

    #[test]
    fn counters_create_add_read_and_reject_wrong_types() {
        let mut store = ObjectStore::new();
        assert_eq!(
            execute(&mut store, &Operation::CounterGet { key: key("c") }, NOW).expect("get"),
            OperationResult::Counter(None)
        );
        assert_eq!(
            execute(
                &mut store,
                &Operation::CounterAdd {
                    key: key("c"),
                    delta: 41,
                },
                NOW,
            )
            .expect("create"),
            OperationResult::CounterUpdated {
                value: 41,
                version: ObjectVersion::FIRST
            }
        );
        assert_eq!(
            execute(
                &mut store,
                &Operation::CounterAdd {
                    key: key("c"),
                    delta: -50,
                },
                NOW,
            )
            .expect("decrement"),
            OperationResult::CounterUpdated {
                value: -9,
                version: ObjectVersion::from_u64(2)
            }
        );
        // Native Get on a counter fails; CounterGet on bytes fails. Neither mutates.
        assert_eq!(
            store.prepare(&Operation::Get { key: key("c") }, NOW),
            Err(OpError::WrongType {
                expected: ObjectType::Bytes,
                found: ObjectType::StrictCounter,
            })
        );
        execute(
            &mut store,
            &Operation::Set {
                key: key("b"),
                value: bytes::Bytes::from_static(b"str"),
            },
            NOW,
        )
        .expect("set bytes");
        assert_eq!(
            store.prepare(&Operation::CounterGet { key: key("b") }, NOW),
            Err(OpError::WrongType {
                expected: ObjectType::StrictCounter,
                found: ObjectType::Bytes,
            })
        );
        assert_eq!(
            store.prepare(
                &Operation::CounterAdd {
                    key: key("b"),
                    delta: 1,
                },
                NOW,
            ),
            Err(OpError::WrongType {
                expected: ObjectType::StrictCounter,
                found: ObjectType::Bytes,
            })
        );
        // Failed validations leave versions untouched.
        assert_eq!(
            store.get(&key("c"), NOW).expect("counter").version(),
            ObjectVersion::from_u64(2)
        );
    }

    #[test]
    fn counter_overflow_and_underflow_fail_without_mutation() {
        let mut store = ObjectStore::new();
        execute(
            &mut store,
            &Operation::CounterAdd {
                key: key("max"),
                delta: i64::MAX,
            },
            NOW,
        )
        .expect("seed max");
        assert_eq!(
            store.prepare(
                &Operation::CounterAdd {
                    key: key("max"),
                    delta: 1,
                },
                NOW,
            ),
            Err(OpError::CounterOverflow)
        );
        execute(
            &mut store,
            &Operation::CounterAdd {
                key: key("min"),
                delta: i64::MIN,
            },
            NOW,
        )
        .expect("seed min");
        assert_eq!(
            store.prepare(
                &Operation::CounterAdd {
                    key: key("min"),
                    delta: -1,
                },
                NOW,
            ),
            Err(OpError::CounterOverflow)
        );
        assert_eq!(
            store.get(&key("max"), NOW).expect("max intact").version(),
            ObjectVersion::FIRST
        );
    }

    #[test]
    fn version_exhaustion_fails_apply_closed() {
        let mut store = ObjectStore::new();
        // Forge a max-version object directly (unreachable by counting).
        store.objects.insert(
            key("old"),
            StoredObject::new(
                LogicalValue::Bytes(bytes::Bytes::from_static(b"x")),
                ObjectVersion::from_u64(u64::MAX),
                Expiry::NEVER,
            ),
        );
        assert_eq!(
            store.apply(
                &Mutation::PutBytes {
                    key: key("old"),
                    value: bytes::Bytes::from_static(b"y"),
                },
                NOW,
            ),
            Err(ApplyError::VersionExhausted)
        );
        // Untouched by the failed apply.
        assert_eq!(
            store.get(&key("old"), NOW).expect("intact").version(),
            ObjectVersion::from_u64(u64::MAX)
        );
    }

    #[test]
    fn ttl_boundaries_and_persist() {
        let mut store = ObjectStore::new();
        execute(
            &mut store,
            &Operation::Set {
                key: key("t"),
                value: bytes::Bytes::from_static(b"v"),
            },
            NOW,
        )
        .expect("set");
        let deadline = UnixMicros::from_micros(2_000_000);
        assert_eq!(
            execute(
                &mut store,
                &Operation::ExpireAt {
                    key: key("t"),
                    expires_at: deadline,
                },
                NOW,
            )
            .expect("expire"),
            OperationResult::ExpirySet { applied: true }
        );
        // Before: live. Exactly at: absent (>= boundary). After: absent.
        assert_eq!(
            execute(
                &mut store,
                &Operation::Get { key: key("t") },
                UnixMicros::from_micros(1_999_999)
            )
            .expect("before"),
            OperationResult::Value(Some(bytes::Bytes::from_static(b"v")))
        );
        assert_eq!(
            execute(&mut store, &Operation::Get { key: key("t") }, deadline).expect("exact"),
            OperationResult::Value(None)
        );
        assert_eq!(
            execute(
                &mut store,
                &Operation::Get { key: key("t") },
                UnixMicros::from_micros(2_000_001)
            )
            .expect("after"),
            OperationResult::Value(None)
        );
        // Expiry reports the stamp while live.
        assert_eq!(
            execute(
                &mut store,
                &Operation::GetExpiry { key: key("t") },
                UnixMicros::from_micros(1_000_000)
            )
            .expect("getexpiry"),
            OperationResult::Expiry(Some(Expiry::at(deadline)))
        );
        // Persist before expiry restores immortality.
        assert_eq!(
            execute(
                &mut store,
                &Operation::PersistExpiry { key: key("t") },
                UnixMicros::from_micros(1_500_000)
            )
            .expect("persist"),
            OperationResult::ExpiryPersisted { removed: true }
        );
        assert_eq!(
            execute(
                &mut store,
                &Operation::Get { key: key("t") },
                UnixMicros::from_micros(9_999_999)
            )
            .expect("immortal"),
            OperationResult::Value(Some(bytes::Bytes::from_static(b"v")))
        );
        // Persisting again reports no removal.
        assert_eq!(
            execute(&mut store, &Operation::PersistExpiry { key: key("t") }, NOW)
                .expect("persist again"),
            OperationResult::ExpiryPersisted { removed: false }
        );
        // Expiry on a missing key applies nothing.
        assert_eq!(
            execute(
                &mut store,
                &Operation::ExpireAt {
                    key: key("missing"),
                    expires_at: deadline,
                },
                NOW,
            )
            .expect("missing expiry"),
            OperationResult::ExpirySet { applied: false }
        );
    }

    #[test]
    fn overwrite_expired_starts_fresh_and_reads_never_delete() {
        let mut store = ObjectStore::new();
        execute(
            &mut store,
            &Operation::Set {
                key: key("t"),
                value: bytes::Bytes::from_static(b"old"),
            },
            NOW,
        )
        .expect("set");
        execute(
            &mut store,
            &Operation::ExpireAt {
                key: key("t"),
                expires_at: UnixMicros::from_micros(100),
            },
            NOW,
        )
        .expect("expire");
        let after = UnixMicros::from_micros(200);
        // Reads see absence but perform no deletion (physical entry remains).
        assert_eq!(
            execute(&mut store, &Operation::Get { key: key("t") }, after).expect("absent"),
            OperationResult::Value(None)
        );
        assert_eq!(store.len(), 1, "reads must not secretly delete");
        // Overwriting the expired object restarts versioning at FIRST.
        assert_eq!(
            execute(
                &mut store,
                &Operation::Set {
                    key: key("t"),
                    value: bytes::Bytes::from_static(b"new"),
                },
                after,
            )
            .expect("overwrite"),
            OperationResult::Stored {
                version: ObjectVersion::FIRST
            }
        );
    }

    #[test]
    fn expiry_sweep_uses_explicit_delete_mutations() {
        let mut store = ObjectStore::new();
        for name in ["a", "b", "c", "live"] {
            execute(
                &mut store,
                &Operation::Set {
                    key: key(name),
                    value: bytes::Bytes::from_static(b"v"),
                },
                NOW,
            )
            .expect("set");
        }
        for name in ["a", "b", "c"] {
            execute(
                &mut store,
                &Operation::ExpireAt {
                    key: key(name),
                    expires_at: UnixMicros::from_micros(50),
                },
                NOW,
            )
            .expect("expire");
        }
        let now = UnixMicros::from_micros(100);
        // Bounded, sorted collection of the expired.
        assert_eq!(store.collect_expired(now, 2).len(), 2);
        let swept = store.collect_expired(now, 1_000);
        assert_eq!(swept, vec![key("a"), key("b"), key("c")]);
        // Reclamation flows through the normal mutation path. The deletes
        // report `existed: false`: the objects were already logically
        // absent, and only their physical shells are reclaimed now.
        for expired in swept {
            let outcome = store
                .apply(&Mutation::Delete { key: expired }, now)
                .expect("sweep delete");
            assert_eq!(outcome, ApplyOutcome::Deleted { existed: false });
        }
        assert_eq!(store.len(), 1);
        assert_eq!(store.live_count(now), 1);
        assert!(store.collect_expired(now, 1_000).is_empty());
    }

    #[test]
    fn same_mutations_on_same_state_converge() {
        let script = vec![
            Mutation::PutBytes {
                key: key("a"),
                value: bytes::Bytes::from_static(b"1"),
            },
            Mutation::CounterAdd {
                key: key("n"),
                delta: 5,
            },
            Mutation::CounterAdd {
                key: key("n"),
                delta: -2,
            },
            Mutation::SetExpiry {
                key: key("a"),
                expiry: Expiry::at(UnixMicros::from_micros(9_999_999)),
            },
            Mutation::Delete { key: key("n") },
        ];
        let mut left = ObjectStore::new();
        let mut right = ObjectStore::new();
        for mutation in &script {
            let first = left.apply(mutation, NOW).expect("left applies");
            let second = right.apply(mutation, NOW).expect("right applies");
            assert_eq!(first, second);
        }
        assert_eq!(left.snapshot_sorted(), right.snapshot_sorted());
    }
}
