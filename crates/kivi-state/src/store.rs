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
use std::collections::{BTreeMap, BTreeSet};

use kivi_types::{Expiry, TabletId, UnixMicros};

use crate::mutation::{ApplyError, ApplyOutcome, Mutation};
use crate::object::{ChunkedRef, Key, LogicalValue, ObjectType, ObjectVersion, StoredObject};
use crate::ops::{DurableOutcome, OpError, Operation, OperationResult};
use crate::scan::{
    ScanDirection, ScanEntry, ScanError, ScanPage, ScanProjection, ScanSpec, ScannedValue,
};
use crate::txn::{TxnExpect, TxnId, TxnIntent, TxnWrite, TxnWriteKind};

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

/// Slices `[offset, offset + len)` clamped to the logical length (empty
/// when `offset` is past the end). Pure: the single implementation shared
/// by inline reads, so admission and tests cannot disagree. `offset`/`len`
/// are `u64` logical indexes; values larger than memory address space clamp
/// to the end rather than failing (reads never reject on bounds).
#[must_use]
pub fn slice_range(base: &[u8], offset: u64, len: u64) -> bytes::Bytes {
    let total = base.len() as u64;
    let start = offset.min(total);
    let available = total.saturating_sub(start);
    let take = len.min(available);
    let start_usize = usize::try_from(start).unwrap_or(usize::MAX).min(base.len());
    let take_usize = usize::try_from(take)
        .unwrap_or(usize::MAX)
        .min(base.len() - start_usize);
    bytes::Bytes::copy_from_slice(&base[start_usize..start_usize + take_usize])
}

/// Checks a transaction OCC expectation against live state. Pure helper
/// shared by prepare and apply so both compute identical verdicts.
fn check_txn_expectation(expect: &TxnExpect, live: Option<&StoredObject>) -> Result<(), OpError> {
    let holds = match expect {
        TxnExpect::Any => true,
        TxnExpect::Absent => live.is_none(),
        TxnExpect::Version(version) => live.map(StoredObject::version) == Some(*version),
    };
    if holds {
        Ok(())
    } else {
        Err(OpError::TxnConflict)
    }
}

/// Checks a transactional write is deterministically applicable to live
/// state (counter overflow and type gates mirror the non-transactional
/// paths exactly). Pure helper shared by prepare and apply.
fn check_txn_writable(kind: &TxnWriteKind, live: Option<&StoredObject>) -> Result<(), OpError> {
    match kind {
        TxnWriteKind::Put(_) | TxnWriteKind::Delete => Ok(()),
        TxnWriteKind::CounterAdd(delta) => match live.map(StoredObject::value) {
            None => Ok(()),
            Some(LogicalValue::StrictCounter(value)) => {
                value.checked_add(*delta).ok_or(OpError::CounterOverflow)?;
                Ok(())
            }
            Some(_) => Err(OpError::WrongType {
                expected: ObjectType::StrictCounter,
                found: ObjectType::Bytes,
            }),
        },
    }
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
///
/// Besides the primary hash map, the store optionally maintains an ordered
/// key index (`BTreeSet`) for ordered-layout tablets: `seek`/`range` over
/// logical key bytes while values stay stored once. Hash-layout tablets
/// leave it disabled and pay nothing. The index is deterministic state —
/// rebuilt from checkpoint + Raft log, verified by [`verify_ordered_index`](Self::verify_ordered_index)
/// — never a best-effort cache.
///
/// The store also owns the participant intent table for distributed
/// transactions: at most one prepared intent per key, consulted by every
/// mutating prepare and persisted through checkpoints/snapshots.
#[derive(Debug, Clone, Default)]
pub struct ObjectStore {
    objects: hashbrown::HashMap<Key, StoredObject, RandomState>,
    ordered: Option<BTreeSet<Key>>,
    intents: BTreeMap<Key, TxnIntent>,
}

/// Per-tablet logical telemetry feeding split/merge policy.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct StoreStats {
    /// Physically stored entries (including expired-but-unreclaimed).
    pub object_count: usize,
    /// Logically live objects at the query instant.
    pub live_count: usize,
    /// Total key bytes physically stored.
    pub key_bytes: u64,
    /// Total logical value bytes (inline lengths + chunked logical
    /// lengths + 8 per counter).
    pub logical_value_bytes: u64,
    /// Prepared transaction intents currently reserving keys.
    pub intent_count: usize,
    /// Whether the ordered index is enabled.
    pub ordered_indexed: bool,
}

impl ObjectStore {
    /// Creates an empty store.
    #[must_use]
    pub fn new() -> Self {
        Self {
            objects: hashbrown::HashMap::default(),
            ordered: None,
            intents: BTreeMap::new(),
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
        self.note_stored(&key);
        self.objects.insert(key, object);
    }

    /// Removes whatever is physically stored under `key`, if anything.
    /// Batch-overlay counterpart to [`put_stored`](Self::put_stored).
    pub fn remove_stored(&mut self, key: &Key) {
        self.objects.remove(key);
        self.note_removed(key);
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
    /// Normal mutating operations fail with [`TxnConflict`](OpError::TxnConflict)
    /// while a prepared transaction intent reserves the key; transaction
    /// operations (`TxnPrepare`/`TxnFinalize`) follow the 2PC path instead.
    ///
    /// # Errors
    ///
    /// Returns [`OpError::WrongType`] for type mismatches and
    /// [`OpError::CounterOverflow`] for overflowing additions. Neither
    /// mutates anything.
    #[allow(clippy::too_many_lines)]
    pub fn prepare(&self, op: &Operation, now: UnixMicros) -> Result<Prepared, OpError> {
        match op {
            Operation::TxnPrepare {
                txn,
                coordinator,
                write,
            } => {
                return self
                    .prepare_txn_reserve(*txn, *coordinator, write, now)
                    .map(Prepared::Write);
            }
            Operation::TxnFinalize { txn, key, commit } => {
                return Ok(Prepared::Write(Mutation::TxnFinalize {
                    txn: *txn,
                    key: key.clone(),
                    commit: *commit,
                }));
            }
            _ => {}
        }
        if op.is_mutating() && self.intents.contains_key(op.key()) {
            return Err(OpError::TxnConflict);
        }
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
            Operation::GetRange { key, offset, len } => {
                self.prepare_get_range(key, *offset, *len, now)
            }
            Operation::BytesLength { key } => match self.get(key, now) {
                None => Ok(Prepared::Read(OperationResult::Length(None))),
                Some(object) => match object.value() {
                    LogicalValue::Bytes(value) => Ok(Prepared::Read(OperationResult::Length(
                        Some(value.len() as u64),
                    ))),
                    LogicalValue::Chunked(chunked) => Ok(Prepared::Read(OperationResult::Length(
                        Some(chunked.logical_len),
                    ))),
                    LogicalValue::StrictCounter(_) => Err(OpError::WrongType {
                        expected: ObjectType::Bytes,
                        found: ObjectType::StrictCounter,
                    }),
                },
            },
            Operation::GetVersion { key } => Ok(Prepared::Read(OperationResult::Version(
                self.get(key, now).map(StoredObject::version),
            ))),
            Operation::SetConditional {
                key,
                value,
                condition,
                expiry,
            } => Ok(self.prepare_set_conditional(key, value, *condition, *expiry, now)),
            Operation::SetConditionalChunked {
                key,
                manifest,
                logical_len,
                condition,
                expiry,
            } => Ok(self.prepare_set_conditional_chunked(
                key,
                *manifest,
                *logical_len,
                *condition,
                *expiry,
                now,
            )),
            // Transaction operations dispatch before this match; the arm
            // exists only for exhaustiveness.
            Operation::TxnPrepare { .. } | Operation::TxnFinalize { .. } => {
                unreachable!("transaction operations return before the prepare match")
            }
        }
    }

    /// Validates a range read against current state: absent answers `None`,
    /// inline slices `[offset, offset + len)` clamped to the logical length
    /// (empty when past the end), counters reject, and chunked roots answer
    /// as [`ChunkedValue`](OperationResult::ChunkedValue) so the engine can
    /// resolve and slice without this layer touching the filesystem.
    fn prepare_get_range(
        &self,
        key: &Key,
        offset: u64,
        len: u64,
        now: UnixMicros,
    ) -> Result<Prepared, OpError> {
        match self.get(key, now) {
            None => Ok(Prepared::Read(OperationResult::Value(None))),
            Some(object) => match object.value() {
                LogicalValue::Bytes(value) => Ok(Prepared::Read(OperationResult::Value(Some(
                    slice_range(value, offset, len),
                )))),
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
        }
    }

    /// Validates a transaction prepare: a foreign intent blocks the key,
    /// the OCC expectation must hold against live state, and the write
    /// itself must be deterministically applicable (overflow/type gates).
    /// Re-validating an identical same-transaction intent is idempotent and
    /// succeeds whenever state still satisfies the expectation.
    fn prepare_txn_reserve(
        &self,
        txn: TxnId,
        coordinator: TabletId,
        write: &TxnWrite,
        now: UnixMicros,
    ) -> Result<Mutation, OpError> {
        if let Some(intent) = self.intents.get(&write.key)
            && intent.id != txn
        {
            return Err(OpError::TxnConflict);
        }
        check_txn_expectation(&write.expect, self.get(&write.key, now))?;
        check_txn_writable(&write.kind, self.get(&write.key, now))?;
        Ok(Mutation::TxnPrepare {
            txn,
            coordinator: coordinator.as_u64(),
            key: write.key.clone(),
            expect: write.expect,
            write: write.kind.clone(),
        })
    }

    /// Validates a conditional store: evaluates the presence condition
    /// against live state and, when it holds, resolves the expiry policy to
    /// a concrete stamp carried in the mutation (so replay applies the same
    /// stamp without re-reading state). Total: conditions never reject with
    /// an error, they answer not-applied.
    fn prepare_set_conditional(
        &self,
        key: &Key,
        value: &bytes::Bytes,
        condition: crate::ops::SetCondition,
        policy: crate::ops::ExpiryPolicy,
        now: UnixMicros,
    ) -> Prepared {
        let live = self.get(key, now);
        let holds = match condition {
            crate::ops::SetCondition::Always => true,
            crate::ops::SetCondition::IfAbsent => live.is_none(),
            crate::ops::SetCondition::IfPresent => live.is_some(),
        };
        if !holds {
            return Prepared::Read(OperationResult::ConditionalSet {
                applied: false,
                version: None,
            });
        }
        let expiry = match policy {
            crate::ops::ExpiryPolicy::Clear => Expiry::NEVER,
            crate::ops::ExpiryPolicy::Keep => live.map_or(Expiry::NEVER, StoredObject::expiry),
            crate::ops::ExpiryPolicy::ExpireAt(stamp) => Expiry::at(stamp),
        };
        Prepared::Write(Mutation::PutBytesWithExpiry {
            key: key.clone(),
            value: value.clone(),
            expiry,
        })
    }

    /// Validates a staged conditional chunked store: same condition/expiry
    /// resolution as inline, but produces a chunked root mutation. Total
    /// like its inline twin.
    fn prepare_set_conditional_chunked(
        &self,
        key: &Key,
        manifest: kivi_types::ManifestId,
        logical_len: u64,
        condition: crate::ops::SetCondition,
        policy: crate::ops::ExpiryPolicy,
        now: UnixMicros,
    ) -> Prepared {
        let live = self.get(key, now);
        let holds = match condition {
            crate::ops::SetCondition::Always => true,
            crate::ops::SetCondition::IfAbsent => live.is_none(),
            crate::ops::SetCondition::IfPresent => live.is_some(),
        };
        if !holds {
            return Prepared::Read(OperationResult::ConditionalSet {
                applied: false,
                version: None,
            });
        }
        let expiry = match policy {
            crate::ops::ExpiryPolicy::Clear => Expiry::NEVER,
            crate::ops::ExpiryPolicy::Keep => live.map_or(Expiry::NEVER, StoredObject::expiry),
            crate::ops::ExpiryPolicy::ExpireAt(stamp) => Expiry::at(stamp),
        };
        Prepared::Write(Mutation::ReplaceChunkedRootWithExpiry {
            key: key.clone(),
            manifest,
            logical_len,
            expiry,
        })
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
    #[allow(clippy::too_many_lines)]
    pub fn prepare_durable(
        &self,
        op: &Operation,
        now: UnixMicros,
    ) -> Result<StorePrepared, OpError> {
        if let Operation::TxnPrepare {
            txn,
            coordinator,
            write,
        } = op
        {
            return Ok(
                match self.prepare_txn_reserve(*txn, *coordinator, write, now) {
                    Ok(mutation) => StorePrepared::Write {
                        mutation,
                        expected: OperationResult::TxnPrepared,
                    },
                    Err(error) => StorePrepared::Terminal(DurableOutcome::Rejected(error)),
                },
            );
        }
        if let Operation::TxnFinalize { txn, key, commit } = op {
            return Ok(self.predict_finalize(*txn, key, *commit, now));
        }
        if op.is_mutating() && self.intents.contains_key(op.key()) {
            return Ok(StorePrepared::Terminal(DurableOutcome::Rejected(
                OpError::TxnConflict,
            )));
        }
        match op {
            Operation::Get { .. }
            | Operation::Exists { .. }
            | Operation::CounterGet { .. }
            | Operation::GetExpiry { .. }
            | Operation::GetRange { .. }
            | Operation::GetVersion { .. }
            | Operation::BytesLength { .. } => match self.prepare(op, now)? {
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
            Operation::SetConditional {
                key,
                value,
                condition,
                expiry,
            } => Ok(self.predict_set_conditional(key, value, *condition, *expiry, now)),
            Operation::SetConditionalChunked {
                key,
                manifest,
                logical_len,
                condition,
                expiry,
            } => Ok(self.predict_set_conditional_chunked(
                key,
                *manifest,
                *logical_len,
                *condition,
                *expiry,
                now,
            )),
            // Transaction operations dispatch before this match (intent
            // reservation and finalize prediction need no overlay staging
            // beyond the key view); the arm exists only for exhaustiveness.
            Operation::TxnPrepare { .. } | Operation::TxnFinalize { .. } => {
                unreachable!("transaction operations return before the durable match")
            }
        }
    }

    /// Predicts a conditional store: unmet conditions become terminal
    /// no-op completions (retry-safe, nothing to replay), met conditions
    /// become version-predicted writes carrying the resolved expiry.
    fn predict_set_conditional(
        &self,
        key: &Key,
        value: &bytes::Bytes,
        condition: crate::ops::SetCondition,
        policy: crate::ops::ExpiryPolicy,
        now: UnixMicros,
    ) -> StorePrepared {
        let live = self.get(key, now);
        let holds = match condition {
            crate::ops::SetCondition::Always => true,
            crate::ops::SetCondition::IfAbsent => live.is_none(),
            crate::ops::SetCondition::IfPresent => live.is_some(),
        };
        if !holds {
            return StorePrepared::Terminal(DurableOutcome::Completed(
                OperationResult::ConditionalSet {
                    applied: false,
                    version: None,
                },
            ));
        }
        let expiry = match policy {
            crate::ops::ExpiryPolicy::Clear => Expiry::NEVER,
            crate::ops::ExpiryPolicy::Keep => live.map_or(Expiry::NEVER, StoredObject::expiry),
            crate::ops::ExpiryPolicy::ExpireAt(stamp) => Expiry::at(stamp),
        };
        match self.predict_version(key, now) {
            Ok(version) => StorePrepared::Write {
                mutation: Mutation::PutBytesWithExpiry {
                    key: key.clone(),
                    value: value.clone(),
                    expiry,
                },
                expected: OperationResult::ConditionalSet {
                    applied: true,
                    version: Some(version),
                },
            },
            Err(outcome) => StorePrepared::Terminal(outcome),
        }
    }

    /// Predicts a staged conditional chunked store (see
    /// [`predict_set_conditional`](Self::predict_set_conditional)).
    fn predict_set_conditional_chunked(
        &self,
        key: &Key,
        manifest: kivi_types::ManifestId,
        logical_len: u64,
        condition: crate::ops::SetCondition,
        policy: crate::ops::ExpiryPolicy,
        now: UnixMicros,
    ) -> StorePrepared {
        let live = self.get(key, now);
        let holds = match condition {
            crate::ops::SetCondition::Always => true,
            crate::ops::SetCondition::IfAbsent => live.is_none(),
            crate::ops::SetCondition::IfPresent => live.is_some(),
        };
        if !holds {
            return StorePrepared::Terminal(DurableOutcome::Completed(
                OperationResult::ConditionalSet {
                    applied: false,
                    version: None,
                },
            ));
        }
        let expiry = match policy {
            crate::ops::ExpiryPolicy::Clear => Expiry::NEVER,
            crate::ops::ExpiryPolicy::Keep => live.map_or(Expiry::NEVER, StoredObject::expiry),
            crate::ops::ExpiryPolicy::ExpireAt(stamp) => Expiry::at(stamp),
        };
        match self.predict_version(key, now) {
            Ok(version) => StorePrepared::Write {
                mutation: Mutation::ReplaceChunkedRootWithExpiry {
                    key: key.clone(),
                    manifest,
                    logical_len,
                    expiry,
                },
                expected: OperationResult::ConditionalSet {
                    applied: true,
                    version: Some(version),
                },
            },
            Err(outcome) => StorePrepared::Terminal(outcome),
        }
    }

    /// Predicts the next version for a key gaining state, mapping
    /// exhaustion to the terminal outcome the ephemeral apply path would
    /// have produced (plus retry-safety).
    /// Predicts a finalize: missing or foreign intents answer without
    /// touching state, aborts discard, and commits predict the inner write
    /// with the same version arithmetic [`apply`](Self::apply) will use —
    /// prediction and application agree by construction on identical state.
    fn predict_finalize(
        &self,
        txn: TxnId,
        key: &Key,
        commit: bool,
        now: UnixMicros,
    ) -> StorePrepared {
        let mutation = Mutation::TxnFinalize {
            txn,
            key: key.clone(),
            commit,
        };
        let not_applied = || StorePrepared::Write {
            mutation: mutation.clone(),
            expected: OperationResult::TxnFinalized {
                applied: false,
                version: None,
            },
        };
        let Some(intent) = self.intents.get(key) else {
            return not_applied();
        };
        if intent.id != txn {
            return StorePrepared::Write {
                mutation,
                expected: OperationResult::TxnConflict,
            };
        }
        if !commit {
            return not_applied();
        }
        match &intent.write.kind {
            TxnWriteKind::Delete => StorePrepared::Write {
                mutation,
                expected: OperationResult::TxnFinalized {
                    applied: true,
                    version: None,
                },
            },
            TxnWriteKind::Put(_) => match self.predict_version(key, now) {
                Ok(version) => StorePrepared::Write {
                    mutation,
                    expected: OperationResult::TxnFinalized {
                        applied: true,
                        version: Some(version),
                    },
                },
                Err(outcome) => StorePrepared::Terminal(outcome),
            },
            TxnWriteKind::CounterAdd(delta) => match self.get(key, now) {
                None => StorePrepared::Write {
                    mutation,
                    expected: OperationResult::TxnFinalized {
                        applied: true,
                        version: Some(ObjectVersion::FIRST),
                    },
                },
                Some(object) => match object.value() {
                    LogicalValue::StrictCounter(value) => match value.checked_add(*delta) {
                        None => StorePrepared::Terminal(DurableOutcome::Rejected(
                            OpError::CounterOverflow,
                        )),
                        Some(_) => match self.predict_version(key, now) {
                            Ok(version) => StorePrepared::Write {
                                mutation,
                                expected: OperationResult::TxnFinalized {
                                    applied: true,
                                    version: Some(version),
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
            },
        }
    }

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
    /// Transaction reservations resolve through [`TxnPrepare`](Mutation::TxnPrepare)
    /// / [`TxnFinalize`](Mutation::TxnFinalize) arms below; ordinary
    /// mutations meeting a prepared intent fail closed with
    /// [`TxnDiverged`](ApplyError::TxnDiverged) (prepare blocks this first —
    /// reaching apply means divergent state). The ordered index is
    /// maintained centrally here, so no arm can forget it.
    ///
    /// # Errors
    ///
    /// Returns [`ApplyError`] when versions cannot advance, a counter
    /// addition overflows, or a counter mutation meets a byte string (only
    /// reachable by applying mutations against divergent state — same log
    /// on same state never produces it).
    // One arm per mutation variant; splitting would scatter the single
    // total function the model test verifies.
    #[allow(clippy::too_many_lines)]
    pub fn apply(
        &mut self,
        mutation: &Mutation,
        now: UnixMicros,
    ) -> Result<ApplyOutcome, ApplyError> {
        match mutation {
            Mutation::TxnPrepare {
                txn,
                coordinator,
                key,
                expect,
                write,
            } => {
                return self.apply_txn_prepare(*txn, *coordinator, key, expect, write, now);
            }
            Mutation::TxnFinalize { txn, key, commit } => {
                return self.apply_txn_finalize(*txn, key, *commit, now);
            }
            _ => {}
        }
        if self.intents.contains_key(mutation.key()) {
            return Err(ApplyError::TxnDiverged);
        }
        let outcome = self.apply_inner(mutation, now)?;
        match &outcome {
            ApplyOutcome::Put { .. }
            | ApplyOutcome::Counter { .. }
            | ApplyOutcome::Expiry { applied: true, .. } => {
                self.note_stored(mutation.key());
            }
            ApplyOutcome::Deleted { .. } => {
                if matches!(mutation, Mutation::Delete { .. }) {
                    self.note_removed(mutation.key());
                }
            }
            _ => {}
        }
        Ok(outcome)
    }

    /// Reserves one key for a transaction: re-validates the OCC expectation
    /// (prepare already did; same state → same verdict) and records the
    /// durable intent. Conflicts are normal outcomes, never divergence.
    fn apply_txn_prepare(
        &mut self,
        txn: TxnId,
        coordinator: u64,
        key: &Key,
        expect: &TxnExpect,
        write: &TxnWriteKind,
        now: UnixMicros,
    ) -> Result<ApplyOutcome, ApplyError> {
        use ApplyError as Fault;
        if let Some(intent) = self.intents.get(key)
            && intent.id != txn
        {
            return Ok(ApplyOutcome::TxnConflict);
        }
        let live = self.get(key, now);
        if check_txn_expectation(expect, live).is_err() {
            return Ok(ApplyOutcome::TxnConflict);
        }
        match write {
            TxnWriteKind::Put(_) | TxnWriteKind::Delete => {}
            TxnWriteKind::CounterAdd(delta) => match live.map(StoredObject::value) {
                None => {}
                Some(LogicalValue::StrictCounter(value)) => {
                    value.checked_add(*delta).ok_or(Fault::CounterOverflow)?;
                }
                Some(_) => {
                    return Err(Fault::TypeMismatch {
                        expected: ObjectType::StrictCounter,
                        found: ObjectType::Bytes,
                    });
                }
            },
        }
        let observed = live.map(StoredObject::version);
        self.intents.insert(
            key.clone(),
            TxnIntent {
                id: txn,
                coordinator: TabletId::from_u64(coordinator),
                key: key.clone(),
                write: TxnWrite {
                    key: key.clone(),
                    kind: write.clone(),
                    expect: *expect,
                },
                observed,
                prepared_at: now.as_micros(),
            },
        );
        Ok(ApplyOutcome::TxnPrepared)
    }

    /// Resolves one key's intent: commit applies the prepared write through
    /// the normal path (ordered index included), abort discards it. Missing
    /// intents (already finalized) and foreign intents answer
    /// deterministically without touching state.
    fn apply_txn_finalize(
        &mut self,
        txn: TxnId,
        key: &Key,
        commit: bool,
        now: UnixMicros,
    ) -> Result<ApplyOutcome, ApplyError> {
        use ApplyError as Fault;
        let Some(intent) = self.intents.get(key) else {
            return Ok(ApplyOutcome::TxnFinalized {
                applied: false,
                version: None,
            });
        };
        if intent.id != txn {
            return Ok(ApplyOutcome::TxnConflict);
        }
        if !commit {
            self.intents.remove(key);
            return Ok(ApplyOutcome::TxnFinalized {
                applied: false,
                version: None,
            });
        }
        let intent = self.intents.remove(key).expect("intent checked above");
        let inner = match &intent.write.kind {
            TxnWriteKind::Put(value) => Mutation::PutBytes {
                key: key.clone(),
                value: value.clone(),
            },
            TxnWriteKind::Delete => Mutation::Delete { key: key.clone() },
            TxnWriteKind::CounterAdd(delta) => Mutation::CounterAdd {
                key: key.clone(),
                delta: *delta,
            },
        };
        // The intent is gone, so the divergence guard passes; the recursive
        // call maintains the ordered index like any ordinary mutation.
        match self.apply(&inner, now)? {
            ApplyOutcome::Put { version } | ApplyOutcome::Counter { version, .. } => {
                Ok(ApplyOutcome::TxnFinalized {
                    applied: true,
                    version: Some(version),
                })
            }
            ApplyOutcome::Deleted { .. } => Ok(ApplyOutcome::TxnFinalized {
                applied: true,
                version: None,
            }),
            _ => Err(Fault::TxnDiverged),
        }
    }

    /// Inner deterministic apply for ordinary mutations (single-key,
    /// intent-free by the time it runs). Split out so the ordered index is
    /// maintained in exactly one place ([`apply`](Self::apply)).
    #[allow(clippy::too_many_lines)]
    fn apply_inner(
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
            Mutation::PutBytesWithExpiry { key, value, expiry } => {
                let version = self
                    .next_version(key, now)
                    .map_err(|_| ApplyError::VersionExhausted)?;
                self.objects.insert(
                    key.clone(),
                    StoredObject::new(LogicalValue::Bytes(value.clone()), version, *expiry),
                );
                Ok(ApplyOutcome::Put { version })
            }
            Mutation::ReplaceChunkedRootWithExpiry {
                key,
                manifest,
                logical_len,
                expiry,
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
                        *expiry,
                    ),
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
                // wrong bytes. A range patch touches bytes, never the TTL:
                // the live expiry survives (Redis SETRANGE parity,
                // differential-tested); absent bases stay immortal.
                let (base, expiry): (&[u8], Expiry) = match self.live(key, now) {
                    None => (&[], Expiry::NEVER),
                    Some(current) => match current.value() {
                        LogicalValue::Bytes(value) => (value, current.expiry()),
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
                    StoredObject::new(LogicalValue::Bytes(spliced), version, expiry),
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
            // Transaction mutations dispatch in `apply` before reaching the
            // inner path; the arm exists only for exhaustiveness.
            Mutation::TxnPrepare { .. } | Mutation::TxnFinalize { .. } => {
                unreachable!("transaction mutations dispatch in apply")
            }
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

    /// Enables or disables the ordered key index. Enabling rebuilds it
    /// deterministically from current objects; disabling drops it (hash
    /// fast path pays nothing afterwards). Tablets adopt this at
    /// creation/split according to namespace layout.
    pub fn set_ordered_indexing(&mut self, enabled: bool) {
        if enabled {
            self.rebuild_ordered_index();
        } else {
            self.ordered = None;
        }
    }

    /// Whether the ordered index is currently maintained.
    #[must_use]
    pub fn ordered_index_enabled(&self) -> bool {
        self.ordered.is_some()
    }

    /// Rebuilds the ordered index deterministically from current objects
    /// (checkpoint restore + restart path): every physically stored key is
    /// indexed, including expired-but-present ones — expiry filters at
    /// scan time, never at index time, so the index always mirrors the
    /// object map exactly.
    pub fn rebuild_ordered_index(&mut self) {
        self.ordered = Some(self.objects.keys().cloned().collect());
    }

    /// Verifies the ordered index mirrors the object map exactly (tests,
    /// debug assertions, restart gates). Unindexed stores verify vacuously.
    #[must_use]
    pub fn verify_ordered_index(&self) -> bool {
        match &self.ordered {
            None => true,
            Some(index) => {
                index.len() == self.objects.len()
                    && index.iter().all(|key| self.objects.contains_key(key))
            }
        }
    }

    /// Tracks a stored key in the ordered index, if enabled.
    fn note_stored(&mut self, key: &Key) {
        if let Some(ordered) = self.ordered.as_mut() {
            ordered.insert(key.clone());
        }
    }

    /// Drops a removed key from the ordered index, if enabled.
    fn note_removed(&mut self, key: &Key) {
        if let Some(ordered) = self.ordered.as_mut() {
            ordered.remove(key);
        }
    }

    /// Executes one bounded local scan page (the canonical scan primitive).
    /// See the [module docs](crate::scan) for budget and projection rules.
    ///
    /// # Errors
    ///
    /// Returns [`ScanError::OrderedIndexDisabled`] when the tablet keeps no
    /// ordered index.
    pub fn scan_range(&self, spec: &ScanSpec, now: UnixMicros) -> Result<ScanPage, ScanError> {
        use core::ops::Bound;
        let ordered = self
            .ordered
            .as_ref()
            .ok_or(ScanError::OrderedIndexDisabled)?;
        // Candidate keys in scan order, one past the page budget (the extra
        // slot tells a full page from an exhausted range).
        let mut candidates: Vec<&Key> = Vec::new();
        match spec.direction {
            ScanDirection::Forward => {
                let lower = spec.start.clone().map_or(Key::from(Vec::new()), Key::from);
                for key in ordered.range(lower..) {
                    if let Some(upper) = &spec.end
                        && key.as_bytes() >= upper.as_slice()
                    {
                        break;
                    }
                    candidates.push(key);
                    if candidates.len() >= spec.max_items.saturating_add(1) {
                        break;
                    }
                }
            }
            ScanDirection::Reverse => {
                let lower = spec.start.as_deref().unwrap_or(&[]);
                let upper: Bound<Key> = match &spec.end {
                    None => Bound::Unbounded,
                    Some(end) => Bound::Excluded(Key::from(end.clone())),
                };
                for key in ordered.range((Bound::Unbounded, upper)).rev() {
                    if key.as_bytes() < lower {
                        break;
                    }
                    candidates.push(key);
                    if candidates.len() >= spec.max_items.saturating_add(1) {
                        break;
                    }
                }
            }
        }
        let mut entries = Vec::new();
        let mut bytes_used: usize = 0;
        let mut last_key: Option<Key> = None;
        let mut stopped_on_budget = false;
        for key in candidates {
            if entries.len() >= spec.max_items {
                stopped_on_budget = true;
                break;
            }
            if key.is_system() {
                continue;
            }
            if !spec.contains(key.as_bytes()) {
                continue;
            }
            let Some(object) = self.get(key, now) else {
                continue;
            };
            let value = match spec.projection {
                ScanProjection::KeysOnly => None,
                ScanProjection::KeysAndValues => Some(match object.value() {
                    LogicalValue::Bytes(value) => {
                        if bytes_used.saturating_add(value.len()) <= spec.max_bytes {
                            bytes_used = bytes_used.saturating_add(value.len());
                            ScannedValue::Inline(value.clone())
                        } else if entries.is_empty() {
                            ScannedValue::Oversize {
                                logical_len: value.len() as u64,
                            }
                        } else {
                            stopped_on_budget = true;
                            break;
                        }
                    }
                    LogicalValue::StrictCounter(counter) => {
                        bytes_used = bytes_used.saturating_add(8);
                        ScannedValue::Counter(*counter)
                    }
                    LogicalValue::Chunked(chunked) => ScannedValue::Chunked {
                        manifest: chunked.manifest,
                        logical_len: chunked.logical_len,
                    },
                }),
            };
            last_key = Some(key.clone());
            entries.push(ScanEntry {
                key: key.clone(),
                value,
            });
        }
        // `exhausted` is exact in every case except a page that fills to
        // exactly `max_items` at the range end (one extra empty page may
        // follow); the cursor protocol tolerates that single roundtrip and
        // neither duplicates nor skips.
        Ok(ScanPage {
            entries,
            exhausted: !stopped_on_budget,
            last_key,
        })
    }

    /// Computes logical telemetry at `now` for split/merge policy.
    #[must_use]
    pub fn stats(&self, now: UnixMicros) -> StoreStats {
        let mut key_bytes: u64 = 0;
        let mut logical_value_bytes: u64 = 0;
        let mut live_count: usize = 0;
        for (key, object) in &self.objects {
            key_bytes = key_bytes.saturating_add(key.len() as u64);
            logical_value_bytes = logical_value_bytes.saturating_add(match object.value() {
                LogicalValue::Bytes(value) => value.len() as u64,
                LogicalValue::Chunked(chunked) => chunked.logical_len,
                LogicalValue::StrictCounter(_) => 8,
            });
            if !object.is_expired(now) {
                live_count += 1;
            }
        }
        StoreStats {
            object_count: self.objects.len(),
            live_count,
            key_bytes,
            logical_value_bytes,
            intent_count: self.intents.len(),
            ordered_indexed: self.ordered.is_some(),
        }
    }

    /// Approximate median split key: the first user key past half of
    /// stored user logical bytes in index order (`None` with fewer than
    /// two user keys — nothing interior to split at). System (`\xff`)
    /// keys are excluded from candidacy: they route by fiat (stripped
    /// remainder / coordinator), not by range, so they may sort outside
    /// this tablet's range and must never anchor a range split. Leaders
    /// compute this locally from a scan; no distributed histogram needed
    /// at this stage.
    #[must_use]
    pub fn median_split_key(&self) -> Option<Key> {
        let ordered = self.ordered.as_ref()?;
        // User keys only (index order preserved by the filter).
        let candidates: Vec<&Key> = ordered.iter().filter(|key| !key.is_system()).collect();
        if candidates.len() < 2 {
            return None;
        }
        let total: u64 = candidates
            .iter()
            .filter_map(|key| self.objects.get(*key))
            .map(|object| match object.value() {
                LogicalValue::Bytes(value) => value.len() as u64,
                LogicalValue::Chunked(chunked) => chunked.logical_len,
                LogicalValue::StrictCounter(_) => 8,
            })
            .sum();
        let mut accumulated: u64 = 0;
        // The first key can never be a split point (nothing lies below it
        // in this tablet), so candidates start at the second key.
        for key in candidates.iter().skip(1) {
            if let Some(object) = self.objects.get(*key) {
                let size = match object.value() {
                    LogicalValue::Bytes(value) => value.len() as u64,
                    LogicalValue::Chunked(chunked) => chunked.logical_len,
                    LogicalValue::StrictCounter(_) => 8,
                };
                accumulated = accumulated.saturating_add(size);
                if accumulated.saturating_mul(2) >= total {
                    return Some((*key).clone());
                }
            }
        }
        None
    }

    /// Returns the prepared intent reserving `key`, if any.
    #[must_use]
    pub fn intent_for_key(&self, key: &Key) -> Option<&TxnIntent> {
        self.intents.get(key)
    }

    /// Clones the prepared intent for `key` (overlay staging copies exactly
    /// this key's intents into scratch so predictions match committed apply).
    #[must_use]
    pub fn cloned_intent_for_key(&self, key: &Key) -> Option<TxnIntent> {
        self.intents.get(key).cloned()
    }

    /// Replaces this key's staged intents (overlay use: the scratch store's
    /// post-apply intent for the key becomes the overlay's staged truth).
    pub fn stage_intent_for_key(&mut self, key: Key, intent: Option<TxnIntent>) {
        match intent {
            Some(staged) => {
                self.intents.insert(key, staged);
            }
            None => {
                self.intents.remove(&key);
            }
        }
    }

    /// Number of prepared intents (split/merge fencing consults this:
    /// tablets with unresolved intents do not cut over).
    #[must_use]
    pub fn pending_intent_count(&self) -> usize {
        self.intents.len()
    }

    /// All prepared intents in key order (snapshots, checkpoints, resolver).
    #[must_use]
    pub fn snapshot_intents(&self) -> Vec<TxnIntent> {
        self.intents.values().cloned().collect()
    }

    /// Installs a recovered intent verbatim (checkpoint/snapshot restore and
    /// tests). Unresolved intents are never discarded on restart.
    pub fn restore_intent(&mut self, intent: TxnIntent) {
        self.intents.insert(intent.key.clone(), intent);
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
    fn set_range_preserves_live_expiry() {
        let mut store = ObjectStore::new();
        let deadline = UnixMicros::from_micros(9_000_000);
        execute(
            &mut store,
            &Operation::Set {
                key: key("k"),
                value: bytes::Bytes::from_static(b"hello"),
            },
            NOW,
        )
        .expect("set");
        execute(
            &mut store,
            &Operation::ExpireAt {
                key: key("k"),
                expires_at: deadline,
            },
            NOW,
        )
        .expect("expire");
        // A patch changes bytes but never the TTL (Redis SETRANGE parity).
        execute(
            &mut store,
            &Operation::SetRange {
                key: key("k"),
                offset: 0,
                patch: bytes::Bytes::from_static(b"X"),
            },
            NOW,
        )
        .expect("splice");
        assert_eq!(
            execute(&mut store, &Operation::Get { key: key("k") }, NOW).expect("get"),
            OperationResult::Value(Some(bytes::Bytes::from_static(b"Xello")))
        );
        assert_eq!(
            execute(&mut store, &Operation::GetExpiry { key: key("k") }, NOW).expect("expiry"),
            OperationResult::Expiry(Some(Expiry::at(deadline)))
        );
        // Durable preparation predicts the same preserved-expiry outcome.
        assert!(matches!(
            store.prepare_durable(
                &Operation::SetRange {
                    key: key("k"),
                    offset: 1,
                    patch: bytes::Bytes::from_static(b"Y"),
                },
                NOW,
            ),
            Ok(StorePrepared::Write { .. })
        ));
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

    fn ordered_store(keys: &[&str]) -> ObjectStore {
        let mut store = ObjectStore::new();
        store.set_ordered_indexing(true);
        for name in keys {
            store
                .apply(
                    &Mutation::PutBytes {
                        key: key(name),
                        value: bytes::Bytes::from(name.as_bytes().to_vec()),
                    },
                    NOW,
                )
                .expect("put");
        }
        assert!(store.verify_ordered_index());
        store
    }

    fn scan_all(store: &ObjectStore, direction: ScanDirection) -> Vec<String> {
        let spec = ScanSpec::new(
            None,
            None,
            direction,
            10_000,
            1 << 20,
            ScanProjection::KeysOnly,
        )
        .expect("spec");
        let page = store.scan_range(&spec, NOW).expect("scan");
        assert!(page.exhausted);
        page.entries
            .iter()
            .map(|entry| entry.key.to_string())
            .collect()
    }

    #[test]
    fn ordered_index_mirrors_state_and_scans_in_order() {
        let store = ordered_store(&["c", "a", "b"]);
        assert_eq!(
            scan_all(&store, ScanDirection::Forward),
            vec!["a", "b", "c"]
        );
        assert_eq!(
            scan_all(&store, ScanDirection::Reverse),
            vec!["c", "b", "a"]
        );
        // Hash stores pay nothing and refuse scans.
        let plain = ObjectStore::new();
        assert!(
            plain
                .scan_range(
                    &ScanSpec::new(
                        None,
                        None,
                        ScanDirection::Forward,
                        10,
                        1024,
                        ScanProjection::KeysOnly
                    )
                    .expect("spec"),
                    NOW
                )
                .is_err()
        );
    }

    #[test]
    fn scan_paginates_with_cursor_and_skips_system_keys() {
        let mut store = ordered_store(&["a", "b", "c", "d"]);
        store.put_stored(
            crate::txn_record_key(
                kivi_types::TabletId::from_u64(1),
                crate::txn::TxnId::derive(1, 1, 0),
            ),
            StoredObject::restore(
                LogicalValue::Bytes(bytes::Bytes::from_static(b"record")),
                ObjectVersion::FIRST,
                Expiry::NEVER,
            ),
        );
        assert!(store.verify_ordered_index());
        // Pages of 2 walk the whole range with no duplicates or gaps.
        let mut seen = Vec::new();
        let mut start: Option<Vec<u8>> = None;
        loop {
            let spec = ScanSpec::new(
                start.clone(),
                None,
                ScanDirection::Forward,
                2,
                1 << 20,
                ScanProjection::KeysAndValues,
            )
            .expect("spec");
            let page = store.scan_range(&spec, NOW).expect("page");
            for entry in &page.entries {
                assert!(!entry.key.is_system(), "system keys never surface");
                seen.push(entry.key.to_string());
                assert!(entry.value.is_some());
            }
            match page.last_key {
                None => break,
                Some(last) => {
                    if page.exhausted {
                        break;
                    }
                    // Cursor resumes strictly after the last emitted key.
                    let mut next = last.as_bytes().to_vec();
                    next.push(0x00);
                    start = Some(next);
                }
            }
            assert!(seen.len() <= 8, "scan did not terminate");
        }
        assert_eq!(seen, vec!["a", "b", "c", "d"]);
    }

    #[test]
    fn scan_respects_values_budget_with_oversize_progress() {
        let mut store = ObjectStore::new();
        store.set_ordered_indexing(true);
        store
            .apply(
                &Mutation::PutBytes {
                    key: key("a-small"),
                    value: bytes::Bytes::from_static(b"v"),
                },
                NOW,
            )
            .expect("put");
        store
            .apply(
                &Mutation::PutBytes {
                    key: key("b-big"),
                    value: bytes::Bytes::from(vec![9u8; 100]),
                },
                NOW,
            )
            .expect("put");
        // Budget fits `a-small` but not `b-big`: first page stops before it.
        let spec = ScanSpec::new(
            None,
            None,
            ScanDirection::Forward,
            100,
            50,
            ScanProjection::KeysAndValues,
        )
        .expect("spec");
        let page = store.scan_range(&spec, NOW).expect("page");
        assert!(!page.exhausted);
        assert_eq!(page.entries.len(), 1);
        assert_eq!(page.entries[0].key, key("a-small"));
        // Resuming at `b-big` with a tiny budget still makes progress via
        // Oversize rather than stalling forever.
        let tiny = ScanSpec::new(
            Some(b"b-big".to_vec()),
            None,
            ScanDirection::Forward,
            100,
            1,
            ScanProjection::KeysAndValues,
        )
        .expect("spec");
        let page = store.scan_range(&tiny, NOW).expect("page");
        assert_eq!(page.entries.len(), 1);
        assert!(matches!(
            page.entries[0].value,
            Some(crate::scan::ScannedValue::Oversize { logical_len: 100 })
        ));
    }

    #[test]
    fn scan_skips_expired_and_chunked_names_references() {
        use kivi_types::ManifestId;
        let mut store = ObjectStore::new();
        store.set_ordered_indexing(true);
        store
            .apply(
                &Mutation::PutBytesWithExpiry {
                    key: key("gone"),
                    value: bytes::Bytes::from_static(b"v"),
                    expiry: Expiry::at(UnixMicros::from_micros(2_000_000)),
                },
                NOW,
            )
            .expect("put");
        store
            .apply(
                &Mutation::ReplaceChunkedRoot {
                    key: key("big"),
                    manifest: ManifestId::from_bytes([0xC4; 32]),
                    logical_len: 9_000_000,
                },
                NOW,
            )
            .expect("put");
        let later = UnixMicros::from_micros(3_000_000);
        let spec = ScanSpec::new(
            None,
            None,
            ScanDirection::Forward,
            100,
            1 << 20,
            ScanProjection::KeysAndValues,
        )
        .expect("spec");
        let page = store.scan_range(&spec, later).expect("page");
        assert_eq!(page.entries.len(), 1);
        assert_eq!(page.entries[0].key, key("big"));
        assert!(matches!(
            page.entries[0].value,
            Some(crate::scan::ScannedValue::Chunked {
                logical_len: 9_000_000,
                ..
            })
        ));
    }

    #[test]
    #[allow(clippy::too_many_lines)]
    fn transaction_prepare_finalize_round_trip() {
        use crate::txn::{TxnExpect, TxnId, TxnWrite, TxnWriteKind};
        use kivi_types::TabletId;
        let mut store = ObjectStore::new();
        store.set_ordered_indexing(true);
        let txn = TxnId::derive(7, 1, 0);
        let coordinator = TabletId::from_u64(1);
        // Seed a key to version-check against.
        store
            .apply(
                &Mutation::PutBytes {
                    key: key("a"),
                    value: bytes::Bytes::from_static(b"1"),
                },
                NOW,
            )
            .expect("seed");
        // Blind prepare reserves without touching user state.
        let prepare = Operation::TxnPrepare {
            txn,
            coordinator,
            write: TxnWrite {
                key: key("b"),
                kind: TxnWriteKind::Put(bytes::Bytes::from_static(b"2")),
                expect: TxnExpect::Absent,
            },
        };
        let prepared = store.prepare_durable(&prepare, NOW).expect("prepare");
        match prepared {
            StorePrepared::Write { expected, .. } => {
                assert_eq!(expected, OperationResult::TxnPrepared);
            }
            StorePrepared::Read(_) | StorePrepared::Terminal(_) => panic!("must prepare"),
        }
        // Materialize through apply like the tablet layer would.
        let Prepared::Write(mutation) = store.prepare(&prepare, NOW).expect("prepare") else {
            panic!("must prepare")
        };
        assert_eq!(store.apply(&mutation, NOW), Ok(ApplyOutcome::TxnPrepared));
        assert_eq!(store.pending_intent_count(), 1);
        // User state untouched while prepared.
        assert_eq!(
            store.get(&key("b"), NOW),
            None,
            "intents hide until finalize"
        );
        // Conflicting normal write fails.
        assert_eq!(
            store.prepare(
                &Operation::Set {
                    key: key("b"),
                    value: bytes::Bytes::from_static(b"race"),
                },
                NOW
            ),
            Err(OpError::TxnConflict)
        );
        // Versioned prepare against the seeded key.
        let versioned = Operation::TxnPrepare {
            txn,
            coordinator,
            write: TxnWrite {
                key: key("a"),
                kind: TxnWriteKind::Put(bytes::Bytes::from_static(b"3")),
                expect: TxnExpect::Version(ObjectVersion::FIRST),
            },
        };
        let Prepared::Write(mutation) = store.prepare(&versioned, NOW).expect("prepare") else {
            panic!("must prepare")
        };
        assert_eq!(store.apply(&mutation, NOW), Ok(ApplyOutcome::TxnPrepared));
        // Stale version rejected.
        let other = TxnId::derive(8, 1, 0);
        assert_eq!(
            store.prepare(
                &Operation::TxnPrepare {
                    txn: other,
                    coordinator,
                    write: TxnWrite {
                        key: key("a"),
                        kind: TxnWriteKind::Put(bytes::Bytes::from_static(b"4")),
                        expect: TxnExpect::Version(ObjectVersion::FIRST),
                    },
                },
                NOW
            ),
            Err(OpError::TxnConflict),
            "live intent blocks foreign prepares"
        );
        // Finalize commit applies both writes atomically per key.
        for name in ["a", "b"] {
            let finalize = Operation::TxnFinalize {
                txn,
                key: key(name),
                commit: true,
            };
            let StorePrepared::Write { mutation, expected } = store
                .prepare_durable(&finalize, NOW)
                .expect("finalize predicts")
            else {
                panic!("finalize must predict a write")
            };
            let outcome = store.apply(&mutation, NOW).expect("finalize applies");
            assert_eq!(
                crate::ops::outcome_for(&mutation, &outcome, false),
                expected
            );
        }
        assert_eq!(store.pending_intent_count(), 0);
        assert!(store.verify_ordered_index());
        let page = store
            .scan_range(
                &ScanSpec::new(
                    None,
                    None,
                    ScanDirection::Forward,
                    10,
                    1024,
                    ScanProjection::KeysOnly,
                )
                .expect("spec"),
                NOW,
            )
            .expect("scan");
        assert_eq!(page.entries.len(), 2);
    }

    #[test]
    fn transaction_abort_discards_without_touching_state() {
        use crate::txn::{TxnExpect, TxnId, TxnWrite, TxnWriteKind};
        use kivi_types::TabletId;
        let mut store = ObjectStore::new();
        let txn = TxnId::derive(9, 1, 0);
        let prepare = Operation::TxnPrepare {
            txn,
            coordinator: TabletId::from_u64(1),
            write: TxnWrite {
                key: key("x"),
                kind: TxnWriteKind::Put(bytes::Bytes::from_static(b"v")),
                expect: TxnExpect::Any,
            },
        };
        let Prepared::Write(mutation) = store.prepare(&prepare, NOW).expect("prepare") else {
            panic!("must prepare")
        };
        store.apply(&mutation, NOW).expect("apply");
        let abort = Operation::TxnFinalize {
            txn,
            key: key("x"),
            commit: false,
        };
        let Prepared::Write(mutation) = store.prepare(&abort, NOW).expect("abort") else {
            panic!("must prepare")
        };
        assert_eq!(
            store.apply(&mutation, NOW),
            Ok(ApplyOutcome::TxnFinalized {
                applied: false,
                version: None
            })
        );
        assert_eq!(store.get(&key("x"), NOW), None);
        assert_eq!(store.pending_intent_count(), 0);
    }

    #[test]
    fn store_stats_and_median_split_key() {
        let store = ordered_store(&["a", "bb", "ccc"]);
        let stats = store.stats(NOW);
        assert_eq!(stats.object_count, 3);
        assert_eq!(stats.live_count, 3);
        assert_eq!(stats.key_bytes, 6);
        assert_eq!(stats.logical_value_bytes, 6);
        assert!(stats.ordered_indexed);
        // Median balances logical bytes: total 6, half 3 → key "ccc"
        // (a=1, bb=2 cumulative... first candidate "bb" acc=2 <3, "ccc" acc=5 ≥3).
        assert_eq!(store.median_split_key(), Some(key("ccc")));
        let single = ordered_store(&["only"]);
        assert_eq!(single.median_split_key(), None);
    }

    #[test]
    fn median_split_key_ignores_system_keys() {
        // System (\xff) keys route by fiat and may sort outside the
        // tablet's range: they must never anchor a range split, no matter
        // how many bytes they hold.
        let mut store = ordered_store(&["a", "bb"]);
        for sys in [
            b"\xffkivi/txn/record".as_slice(),
            b"\xffkivi/proj/a".as_slice(),
        ] {
            store
                .apply(
                    &Mutation::PutBytes {
                        key: Key::from(sys.to_vec()),
                        value: bytes::Bytes::from_static(b"0123456789abcdef"),
                    },
                    NOW,
                )
                .expect("put");
        }
        // Candidates are just a/bb (total 3 bytes, half rounded): the
        // median stays inside user keyspace despite 32 system bytes.
        assert_eq!(store.median_split_key(), Some(key("bb")));
        // A tablet holding only system keys has nothing to split.
        let mut systems_only = ObjectStore::new();
        systems_only.set_ordered_indexing(true);
        systems_only
            .apply(
                &Mutation::PutBytes {
                    key: Key::from(vec![0xFF, 0x01]),
                    value: bytes::Bytes::from_static(b"v"),
                },
                NOW,
            )
            .expect("put");
        assert_eq!(systems_only.median_split_key(), None);
    }

    #[test]
    fn intents_survive_ordered_rebuild() {
        use crate::txn::{TxnExpect, TxnId, TxnWrite, TxnWriteKind};
        use kivi_types::TabletId;
        let mut store = ordered_store(&["a"]);
        let txn = TxnId::derive(1, 2, 0);
        let Prepared::Write(mutation) = store
            .prepare(
                &Operation::TxnPrepare {
                    txn,
                    coordinator: TabletId::from_u64(1),
                    write: TxnWrite {
                        key: key("b"),
                        kind: TxnWriteKind::Put(bytes::Bytes::from_static(b"v")),
                        expect: TxnExpect::Any,
                    },
                },
                NOW,
            )
            .expect("prepare")
        else {
            panic!("must prepare")
        };
        store.apply(&mutation, NOW).expect("apply");
        let intents = store.snapshot_intents();
        let mut rebuilt = ObjectStore::new();
        for (k, v) in store.snapshot_sorted() {
            rebuilt.put_stored(k, v);
        }
        for intent in intents {
            rebuilt.restore_intent(intent);
        }
        rebuilt.set_ordered_indexing(true);
        assert!(rebuilt.verify_ordered_index());
        assert_eq!(rebuilt.pending_intent_count(), 1);
    }
}
