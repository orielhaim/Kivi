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
use crate::object::{ChunkedRef, Key, LogicalValue, ObjectType, ObjectVersion, StoredObject};
use crate::ops::{DurableOutcome, OpError, Operation, OperationResult};

/// Splices `patch` over `base` at `offset`, zero-padding past-the-end
/// gaps (Redis `SETRANGE` semantics). Pure: the single implementation
/// shared by apply and the engine's admission planner, so the two can
/// never disagree. Returns `None` when `offset` does not fit the address
/// space or `offset + patch.len()` overflows `u64`; every other input is
/// total (including offsets far past the end — the gap zero-pads,
/// bounded by the caller's admission cap, never here).
#[must_use]
pub fn splice_inline(base: &[u8], offset: u64, patch: &[u8]) -> Option<bytes::Bytes> {
    let offset = usize::try_from(offset).ok()?;
    let end = offset.checked_add(patch.len())?;
    let mut out = Vec::new();
    out.extend_from_slice(base.get(..offset.min(base.len())).unwrap_or(base));
    out.resize(offset, 0);
    out.extend_from_slice(patch);
    if base.len() > end {
        out.extend_from_slice(&base[end..]);
    }
    Some(bytes::Bytes::from(out))
}

/// A validated operation: either an immediately answered read or a mutation
/// awaiting deterministic application.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Prepared {
    /// Read-only outcome, computed during preparation.
    Read(OperationResult),
    /// Mutation to apply (locally now, or by replicas on replay).
    Write(Mutation),
}

/// A durability-aware preparation: everything `Prepared` carries, plus the
/// terminal outcomes of mutating opcodes that produce no mutation
/// (counter-type/overflow rejections, expiry no-ops). Read-only opcodes
/// still answer inline and never touch the WAL; every other completion the
/// caller persists exactly once before replying.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StorePrepared {
    /// Read-only outcome: reply directly, no WAL, no dedup.
    Read(OperationResult),
    /// Terminal outcome of a mutating opcode with no mutation: persist an
    /// outcome-only record, install dedup, then reply.
    Terminal(DurableOutcome),
    /// Mutating opcode with a mutation: persist a mutation record carrying
    /// the predicted outcome, then apply and verify.
    Write {
        /// Mutation to persist and apply.
        mutation: Mutation,
        /// Outcome `apply` must produce (predicted from current state).
        expected: OperationResult,
    },
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

    /// Returns the physically stored object for `key`, including
    /// logically expired entries. This bypasses the liveness filter for
    /// batch-overlay materialization and checkpoint capture, which must
    /// mirror committed state exactly (an expired-but-present object and
    /// an absent key predict identically today, but the overlay stores
    /// the truth so future semantics cannot silently diverge).
    #[must_use]
    pub fn get_stored(&self, key: &Key) -> Option<&StoredObject> {
        self.objects.get(key)
    }

    /// Stores `object` under `key` verbatim, bypassing prepare/apply.
    /// Batch-overlay materialization and tests use this to stage exact
    /// state; production mutation still flows through [`apply`](Self::apply)
    /// only.
    pub fn put_stored(&mut self, key: Key, object: StoredObject) {
        self.objects.insert(key, object);
    }

    /// Removes whatever is physically stored under `key`, if anything.
    /// Batch-overlay counterpart to [`put_stored`](Self::put_stored).
    pub fn remove_stored(&mut self, key: &Key) {
        self.objects.remove(key);
    }

    /// Snapshots all entries without sorting. The hot worker captures
    /// checkpoints with this (no `O(n log n)` pause on the data path);
    /// deterministic ordering is the serializer's job, never the capture's.
    /// See [`snapshot_sorted`](Self::snapshot_sorted) for the sorted form
    /// used by tests and verification.
    #[must_use]
    pub fn snapshot_entries(&self) -> Vec<(Key, StoredObject)> {
        self.objects
            .iter()
            .map(|(key, object)| (key.clone(), object.clone()))
            .collect()
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
                    // Chunked bytes read identically; the engine resolves
                    // the reference through the chunk lane — this layer
                    // never touches the filesystem.
                    LogicalValue::Chunked(chunked) => {
                        Ok(Prepared::Read(OperationResult::ChunkedValue {
                            manifest: chunked.manifest,
                            logical_len: chunked.logical_len,
                        }))
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
            Operation::SetChunked {
                key,
                manifest,
                logical_len,
            } => Ok(Prepared::Write(Mutation::ReplaceChunkedRoot {
                key: key.clone(),
                manifest: *manifest,
                logical_len: *logical_len,
            })),
            Operation::SetRange { key, offset, patch } => {
                self.prepare_set_range(key, *offset, patch, now)
            }
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
                    // Chunked values are bytes: same rejection as inline.
                    LogicalValue::Bytes(_) | LogicalValue::Chunked(_) => Err(OpError::WrongType {
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
                    LogicalValue::Bytes(_) | LogicalValue::Chunked(_) => Err(OpError::WrongType {
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

    /// Validates a range patch against current state: absent, expired, and
    /// inline bases become [`SpliceBytes`](Mutation::SpliceBytes) writes;
    /// counters reject without mutation; chunked bases report
    /// [`StaleRangeBase`](OpError::StaleRangeBase) (the engine restages
    /// those through the lane before preparing, so meeting one means the
    /// root changed representation under admission).
    fn prepare_set_range(
        &self,
        key: &Key,
        offset: u64,
        patch: &bytes::Bytes,
        now: UnixMicros,
    ) -> Result<Prepared, OpError> {
        let write = || {
            Prepared::Write(Mutation::SpliceBytes {
                key: key.clone(),
                offset,
                patch: patch.clone(),
            })
        };
        match self.get(key, now) {
            None => Ok(write()),
            Some(object) => match object.value() {
                LogicalValue::Bytes(_) => Ok(write()),
                LogicalValue::StrictCounter(_) => Err(OpError::WrongType {
                    expected: ObjectType::Bytes,
                    found: ObjectType::StrictCounter,
                }),
                LogicalValue::Chunked(_) => Err(OpError::StaleRangeBase),
            },
        }
    }

    /// Validates `op` for the durable pipeline, predicting the exact outcome
    /// a subsequent [`apply`](Self::apply) must produce. Pure like
    /// [`prepare`](Self::prepare): borrows state, touches nothing else.
    ///
    /// Read-only opcodes answer inline and never reach the WAL. Every
    /// mutating opcode completion — a mutation with its predicted outcome,
    /// or a terminal outcome with no mutation (counter rejections, expiry
    /// no-ops, version exhaustion) — is returned for exactly-once
    /// persistence before any reply.
    ///
    /// # Errors
    ///
    /// Returns [`OpError`] only for read-only opcode failures (wrong-type
    /// `Get`/`CounterGet`), which the caller answers directly. Mutating
    /// opcode failures arrive as [`StorePrepared::Terminal`] instead, so a
    /// lost reply to them stays retry-safe.
    ///
    /// # Panics
    ///
    /// Panics if a read-only opcode ever prepares a mutation (impossible by
    /// construction of [`prepare`](Self::prepare); loud by design, mirroring
    /// the prepare/apply contract panic in the live tablet path).
    pub fn prepare_durable(
        &self,
        op: &Operation,
        now: UnixMicros,
    ) -> Result<StorePrepared, OpError> {
        match op {
            Operation::Get { .. }
            | Operation::Exists { .. }
            | Operation::CounterGet { .. }
            | Operation::GetExpiry { .. } => match self.prepare(op, now)? {
                Prepared::Read(result) => Ok(StorePrepared::Read(result)),
                Prepared::Write(_) => {
                    panic!("read-only opcode prepared a mutation: {op:?}")
                }
            },
            Operation::Set { key, value } => Ok(match self.predict_version(key, now) {
                Ok(version) => StorePrepared::Write {
                    mutation: Mutation::PutBytes {
                        key: key.clone(),
                        value: value.clone(),
                    },
                    expected: OperationResult::Stored { version },
                },
                Err(outcome) => StorePrepared::Terminal(outcome),
            }),
            Operation::SetChunked {
                key,
                manifest,
                logical_len,
            } => Ok(match self.predict_version(key, now) {
                Ok(version) => StorePrepared::Write {
                    mutation: Mutation::ReplaceChunkedRoot {
                        key: key.clone(),
                        manifest: *manifest,
                        logical_len: *logical_len,
                    },
                    expected: OperationResult::Stored { version },
                },
                Err(outcome) => StorePrepared::Terminal(outcome),
            }),
            Operation::SetRange { key, offset, patch } => {
                // Type gate first: counters reject without mutation (a
                // terminal client error, retry-safe to persist), chunked
                // bases answer directly (transient: never persisted, the
                // retry re-plans against the new root).
                match self.get(key, now) {
                    Some(object) if matches!(object.value(), LogicalValue::StrictCounter(_)) => {
                        return Ok(StorePrepared::Terminal(DurableOutcome::Rejected(
                            OpError::WrongType {
                                expected: ObjectType::Bytes,
                                found: ObjectType::StrictCounter,
                            },
                        )));
                    }
                    Some(object) if matches!(object.value(), LogicalValue::Chunked(_)) => {
                        return Err(OpError::StaleRangeBase);
                    }
                    _ => {}
                }
                Ok(match self.predict_version(key, now) {
                    Ok(version) => StorePrepared::Write {
                        mutation: Mutation::SpliceBytes {
                            key: key.clone(),
                            offset: *offset,
                            patch: patch.clone(),
                        },
                        expected: OperationResult::Stored { version },
                    },
                    Err(outcome) => StorePrepared::Terminal(outcome),
                })
            }
            Operation::Delete { key } => Ok(StorePrepared::Write {
                mutation: Mutation::Delete { key: key.clone() },
                expected: OperationResult::Deleted {
                    existed: self.live(key, now).is_some(),
                },
            }),
            Operation::CounterAdd { key, delta } => Ok(self.predict_counter_add(key, *delta, now)),
            Operation::ExpireAt { key, expires_at } => {
                Ok(self.predict_expiry(key, Some(*expires_at), now))
            }
            Operation::PersistExpiry { key } => Ok(self.predict_expiry(key, None, now)),
        }
    }

    /// Predicts the next version for a key gaining state, mapping
    /// exhaustion to the terminal outcome the ephemeral apply path would
    /// have produced (plus retry-safety).
    fn predict_version(&self, key: &Key, now: UnixMicros) -> Result<ObjectVersion, DurableOutcome> {
        self.next_version(key, now)
            .map_err(|_| DurableOutcome::VersionExhausted)
    }

    /// Predicts a counter addition: fresh creation, validated increment,
    /// or a terminal rejection (overflow, wrong type).
    fn predict_counter_add(&self, key: &Key, delta: i64, now: UnixMicros) -> StorePrepared {
        match self.get(key, now) {
            None => StorePrepared::Write {
                mutation: Mutation::CounterAdd {
                    key: key.clone(),
                    delta,
                },
                expected: OperationResult::CounterUpdated {
                    value: delta,
                    version: ObjectVersion::FIRST,
                },
            },
            Some(object) => match object.value() {
                LogicalValue::StrictCounter(value) => match value.checked_add(delta) {
                    None => {
                        StorePrepared::Terminal(DurableOutcome::Rejected(OpError::CounterOverflow))
                    }
                    Some(next) => match self.predict_version(key, now) {
                        Ok(version) => StorePrepared::Write {
                            mutation: Mutation::CounterAdd {
                                key: key.clone(),
                                delta,
                            },
                            expected: OperationResult::CounterUpdated {
                                value: next,
                                version,
                            },
                        },
                        Err(outcome) => StorePrepared::Terminal(outcome),
                    },
                },
                LogicalValue::Bytes(_) | LogicalValue::Chunked(_) => {
                    StorePrepared::Terminal(DurableOutcome::Rejected(OpError::WrongType {
                        expected: ObjectType::StrictCounter,
                        found: ObjectType::Bytes,
                    }))
                }
            },
        }
    }

    /// Predicts an expiry change: `Some` timestamp is `ExpireAt` semantics
    /// (any live key takes it), `None` is `PersistExpiry` semantics (only a
    /// live dated key takes `NEVER`); everything else is a terminal no-op.
    fn predict_expiry(
        &self,
        key: &Key,
        expires_at: Option<UnixMicros>,
        now: UnixMicros,
    ) -> StorePrepared {
        let live_dated = match self.get(key, now) {
            None => false,
            Some(object) => expires_at.is_some() || object.expiry() != Expiry::NEVER,
        };
        if !live_dated {
            let expected = match expires_at {
                Some(_) => OperationResult::ExpirySet { applied: false },
                None => OperationResult::ExpiryPersisted { removed: false },
            };
            return StorePrepared::Terminal(DurableOutcome::Completed(expected));
        }
        let expiry = expires_at.map_or(Expiry::NEVER, Expiry::at);
        let expected = match expires_at {
            Some(_) => OperationResult::ExpirySet { applied: true },
            None => OperationResult::ExpiryPersisted { removed: true },
        };
        match self.predict_version(key, now) {
            Ok(_) => StorePrepared::Write {
                mutation: Mutation::SetExpiry {
                    key: key.clone(),
                    expiry,
                },
                expected,
            },
            Err(outcome) => StorePrepared::Terminal(outcome),
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
    // One arm per mutation variant (five today); splitting would scatter
    // the single total function the model test verifies.
    #[allow(clippy::too_many_lines)]
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
            Mutation::ReplaceChunkedRoot {
                key,
                manifest,
                logical_len,
            } => {
                let version = self
                    .next_version(key, now)
                    .map_err(|_| ApplyError::VersionExhausted)?;
                self.objects.insert(
                    key.clone(),
                    StoredObject::new(
                        LogicalValue::Chunked(ChunkedRef {
                            manifest: *manifest,
                            logical_len: *logical_len,
                        }),
                        version,
                        Expiry::NEVER,
                    ),
                );
                Ok(ApplyOutcome::Put { version })
            }
            Mutation::SpliceBytes { key, offset, patch } => {
                // Admission guarantees an absent, expired, or inline base
                // with validated arithmetic; anything else is divergent
                // state (or a corrupt record) and fails loudly, never
                // wrong bytes. Expiry clears exactly like `PutBytes`.
                let base: &[u8] = match self.live(key, now) {
                    None => &[],
                    Some(current) => match current.value() {
                        LogicalValue::Bytes(value) => value,
                        LogicalValue::StrictCounter(_) => {
                            return Err(ApplyError::TypeMismatch {
                                expected: ObjectType::Bytes,
                                found: ObjectType::StrictCounter,
                            });
                        }
                        LogicalValue::Chunked(_) => return Err(ApplyError::UnresolvableSplice),
                    },
                };
                let spliced =
                    splice_inline(base, *offset, patch).ok_or(ApplyError::UnresolvableSplice)?;
                let version = self
                    .next_version(key, now)
                    .map_err(|_| ApplyError::VersionExhausted)?;
                self.objects.insert(
                    key.clone(),
                    StoredObject::new(LogicalValue::Bytes(spliced), version, Expiry::NEVER),
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
                    LogicalValue::Bytes(_) | LogicalValue::Chunked(_) => {
                        Err(ApplyError::TypeMismatch {
                            expected: ObjectType::StrictCounter,
                            found: ObjectType::Bytes,
                        })
                    }
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

    /// Durable preparation must predict exactly what the live path produces:
    /// for every op, `prepare_durable`'s expectation equals `execute`'s
    /// result (or its rejection), so replay verification can trust it.
    #[test]
    fn prepare_durable_predicts_execute_exactly() {
        use bytes::Bytes;
        let battery = [
            Operation::Set {
                key: key("a"),
                value: Bytes::from_static(b"1"),
            },
            Operation::Set {
                key: key("a"),
                value: Bytes::from_static(b"2"),
            },
            Operation::Get { key: key("a") },
            Operation::CounterAdd {
                key: key("a"),
                delta: 5,
            },
            Operation::CounterAdd {
                key: key("c"),
                delta: 5,
            },
            Operation::CounterAdd {
                key: key("c"),
                delta: -2,
            },
            Operation::CounterGet { key: key("c") },
            Operation::Delete { key: key("a") },
            Operation::Delete {
                key: key("missing"),
            },
            Operation::ExpireAt {
                key: key("c"),
                expires_at: UnixMicros::from_micros(9_000_000),
            },
            Operation::PersistExpiry { key: key("c") },
            Operation::PersistExpiry {
                key: key("missing"),
            },
            Operation::ExpireAt {
                key: key("missing"),
                expires_at: UnixMicros::from_micros(9_000_000),
            },
            Operation::GetExpiry { key: key("c") },
            Operation::Exists { key: key("c") },
            // Chunked roots predict exactly like inline bytes: stores
            // carry versions, reads name the reference, counters reject.
            Operation::SetChunked {
                key: key("big"),
                manifest: kivi_types::ManifestId::from_bytes([0xC4; 32]),
                logical_len: 3_000_000,
            },
            Operation::Get { key: key("big") },
            Operation::CounterAdd {
                key: key("big"),
                delta: 1,
            },
            Operation::Set {
                key: key("big"),
                value: bytes::Bytes::from_static(b"small again"),
            },
            Operation::SetChunked {
                key: key("big"),
                manifest: kivi_types::ManifestId::from_bytes([0xC5; 32]),
                logical_len: 4_000_000,
            },
            Operation::Delete { key: key("big") },
        ];
        let mut store = ObjectStore::new();
        for op in &battery {
            let predicted = store.prepare_durable(op, NOW);
            let actual = execute(&mut store, op, NOW);
            match predicted {
                Ok(StorePrepared::Read(expected)) => {
                    assert_eq!(actual.expect("read succeeds"), expected);
                }
                Ok(StorePrepared::Terminal(outcome)) => match outcome {
                    DurableOutcome::Completed(expected) => {
                        assert_eq!(actual.expect("terminal ok succeeds"), expected);
                    }
                    DurableOutcome::Rejected(expected_error) => {
                        assert_eq!(
                            actual.expect_err("terminal rejection fails"),
                            format!("{expected_error:?}")
                        );
                    }
                    DurableOutcome::VersionExhausted => {
                        panic!("version space must not exhaust in this battery");
                    }
                },
                Ok(StorePrepared::Write { expected, .. }) => {
                    assert_eq!(actual.expect("predicted write succeeds"), expected);
                }
                Err(error) => {
                    assert_eq!(
                        actual.expect_err("read failure matches"),
                        format!("{error:?}")
                    );
                }
            }
        }
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
                Ok(crate::ops::outcome_for(
                    &mutation,
                    &outcome,
                    matches!(op, Operation::PersistExpiry { .. }),
                ))
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
    fn set_range_splices_inline_bases() {
        use bytes::Bytes;
        let mut store = ObjectStore::new();
        let patch = |offset: u64, patch: &[u8]| Operation::SetRange {
            key: key("k"),
            offset,
            patch: Bytes::copy_from_slice(patch),
        };
        // Absent state reads as empty: a patch at zero creates the value.
        let stored = execute(&mut store, &patch(0, b"hello"), NOW).expect("create");
        assert!(matches!(stored, OperationResult::Stored { .. }));
        // In-place overwrite keeps prefix and suffix.
        execute(&mut store, &patch(1, b"ELL"), NOW).expect("overwrite");
        assert_eq!(read(&store, "k"), b"hELLo");
        // Patch past the end truncates nothing; the gap zero-pads.
        execute(&mut store, &patch(8, b"!"), NOW).expect("gap");
        assert_eq!(read(&store, "k"), b"hELLo\0\0\0!");
        // Empty patch at the end is a version-bumping no-op write.
        let before = read(&store, "k");
        execute(&mut store, &patch(9, b""), NOW).expect("empty patch");
        assert_eq!(read(&store, "k"), before);
        // Expiry clears exactly like `Set`: expire into the past so the
        // old bytes are logically gone, and the patch starts from empty.
        execute(
            &mut store,
            &Operation::ExpireAt {
                key: key("k"),
                expires_at: UnixMicros::from_micros(500),
            },
            NOW,
        )
        .expect("expire");
        execute(&mut store, &patch(0, b"new"), NOW).expect("overwrite clears expiry");
        assert_eq!(read(&store, "k"), b"new");
        assert_eq!(
            store.get(&key("k"), NOW).expect("live").expiry(),
            Expiry::NEVER
        );
    }

    #[test]
    fn set_range_rejects_counters() {
        use bytes::Bytes;
        let mut store = ObjectStore::new();
        execute(
            &mut store,
            &Operation::CounterAdd {
                key: key("n"),
                delta: 4,
            },
            NOW,
        )
        .expect("counter");
        let error = store
            .prepare(
                &Operation::SetRange {
                    key: key("n"),
                    offset: 0,
                    patch: Bytes::from_static(b"x"),
                },
                NOW,
            )
            .expect_err("counter rejects range patches");
        assert_eq!(
            error,
            OpError::WrongType {
                expected: ObjectType::Bytes,
                found: ObjectType::StrictCounter,
            }
        );
        // The durable pipeline reports the same rejection as terminal
        // (retry-safe to persist: the type will still mismatch).
        assert!(matches!(
            store.prepare_durable(
                &Operation::SetRange {
                    key: key("n"),
                    offset: 0,
                    patch: Bytes::from_static(b"x"),
                },
                NOW,
            ),
            Ok(StorePrepared::Terminal(DurableOutcome::Rejected(
                OpError::WrongType { .. }
            )))
        ));
    }

    #[test]
    fn set_range_reports_stale_chunked_bases() {
        use bytes::Bytes;
        let mut store = ObjectStore::new();
        // A chunked base answers directly (transient: never persisted,
        // the retry re-plans against the new root).
        store
            .apply(
                &Mutation::ReplaceChunkedRoot {
                    key: key("c"),
                    manifest: kivi_types::ManifestId::from_bytes([0x22; 32]),
                    logical_len: 10,
                },
                NOW,
            )
            .expect("chunked root");
        assert_eq!(
            store
                .prepare(
                    &Operation::SetRange {
                        key: key("c"),
                        offset: 0,
                        patch: Bytes::from_static(b"x"),
                    },
                    NOW,
                )
                .expect_err("chunked base is stale"),
            OpError::StaleRangeBase
        );
        assert_eq!(
            store
                .prepare_durable(
                    &Operation::SetRange {
                        key: key("c"),
                        offset: 0,
                        patch: Bytes::from_static(b"x"),
                    },
                    NOW,
                )
                .expect_err("durable answers stale directly"),
            OpError::StaleRangeBase
        );
        // ... and a splice reaching apply against one fails loudly.
        assert_eq!(
            store
                .apply(
                    &Mutation::SpliceBytes {
                        key: key("c"),
                        offset: 0,
                        patch: Bytes::from_static(b"x"),
                    },
                    NOW,
                )
                .expect_err("chunked apply fails"),
            ApplyError::UnresolvableSplice
        );
        // Overflowing arithmetic fails loudly too, never wrong bytes.
        assert_eq!(
            store
                .apply(
                    &Mutation::SpliceBytes {
                        key: key("k"),
                        offset: u64::MAX,
                        patch: Bytes::from_static(b"x"),
                    },
                    NOW,
                )
                .expect_err("overflow fails"),
            ApplyError::UnresolvableSplice
        );
    }

    #[test]
    fn set_range_durable_prediction_matches_live_apply() {
        use bytes::Bytes;
        let mut store = ObjectStore::new();
        let op = Operation::SetRange {
            key: key("k"),
            offset: 3,
            patch: Bytes::from_static(b"XYZ"),
        };
        let StorePrepared::Write { mutation, expected } =
            store.prepare_durable(&op, NOW).expect("prepares")
        else {
            panic!("set-range prepares a write");
        };
        assert!(matches!(mutation, Mutation::SpliceBytes { .. }));
        let outcome = store.apply(&mutation, NOW).expect("applies");
        assert_eq!(
            crate::ops::outcome_for(&mutation, &outcome, false),
            expected
        );
        assert_eq!(read(&store, "k"), b"\0\0\0XYZ");
    }

    /// Reads one inline value for assertions (test values stay inline).
    fn read(store: &ObjectStore, name: &str) -> Vec<u8> {
        match store.get(&key(name), NOW).expect("present").value() {
            LogicalValue::Bytes(value) => value.to_vec(),
            LogicalValue::StrictCounter(_) | LogicalValue::Chunked(_) => {
                panic!("test value stays inline")
            }
        }
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
