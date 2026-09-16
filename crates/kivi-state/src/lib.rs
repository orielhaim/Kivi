//! Production logical state layer for Kivi: objects, typed operations,
//! Mutation IR, and partition hashing.
//!
//! This crate owns what an object *means*. [`ObjectStore`] holds worker-local
//! logical objects ([`Bytes`](LogicalValue::Bytes),
//! [`StrictCounter`](LogicalValue::StrictCounter)) with explicit versions and
//! expiry. Native [`Operation`]s are validated by [`prepare`](ObjectStore::prepare)
//! into reads or typed [`Mutation`]s, which [`apply`](ObjectStore::apply)
//! deterministically — the same mutation on the same state always yields the
//! same outcome, with no clock reads or randomness inside.
//!
//! Physical representation is deliberately private: small values live inline
//! while large byte strings live as chunk references (manifest id plus
//! logical length, resolved by the engine), without changing any of these
//! semantics.

pub mod envelope;
pub mod hash;
pub mod index;
pub mod mutation;
pub mod object;
pub mod ops;
pub mod scan;
pub mod store;
pub mod txn;
pub use envelope::{EnvelopeError, MUTATION_ENVELOPE_VERSION, MutationEnvelope};
pub use hash::{PARTITION_DOMAIN_TAG, PartitionHashAlgorithmId, PartitionHasher};
pub use index::{
    INDEX_ENCODING_VERSION, INDEX_FORMAT_TAG, IndexCodecError, IndexDefinition, IndexId, IndexKind,
    IndexProjections, IndexState, PROJECTION_PREFIX, decode_entry_value, decode_non_unique_key,
    decode_unique_key, encode_non_unique_key, encode_non_unique_value, encode_unique_key,
    encode_unique_value, is_index_key, parse_projection_primary, prefix_successor, projection_key,
    term_prefix,
};
use kivi_types::TabletId;
pub use mutation::{ApplyError, ApplyOutcome, Mutation};
pub use object::{
    ChunkedRef, Key, LogicalValue, ObjectType, ObjectVersion, StoredObject, VersionExhausted,
};
pub use ops::{
    DurableOutcome, ExpiryPolicy, OpError, Operation, OperationResult, SetCondition, outcome_for,
};
pub use scan::{
    ScanDirection, ScanEntry, ScanError, ScanPage, ScanProjection, ScanSpec, ScannedValue,
};
pub use store::{ObjectStore, Prepared, StorePrepared, StoreStats, slice_range, splice_inline};
pub use txn::{
    MAX_TXN_BYTES, MAX_TXN_KEYS, MAX_TXN_LIFETIME_MICROS, MAX_TXN_PARTICIPANTS,
    RESOLVE_GRACE_MICROS, TXN_RECORD_KEY_LEN, TXN_RECORD_PREFIX, TXN_RECORD_VERSION,
    TxnDecodeError, TxnDriverPlan, TxnError, TxnExpect, TxnId, TxnIntent, TxnRecord, TxnState,
    TxnWrite, TxnWriteKind, check_bounds, group_by_tablet, parse_txn_record_key, participant_set,
    plan_transaction, select_coordinator, txn_write_from_wire, write_set_digest,
};

/// System key holding the coordinator decision record for `txn` on
/// coordinator tablet `coordinator`:
/// `\xff kivi/txn/ <coordinator BE64> <16 id bytes>`.
///
/// Routing is by fiat, not by hash: the embedded tablet owns the record
/// while active (so single-tablet fast paths stay local); after a
/// split/merge retires it, routers fall back to normal routing and the
/// seeded record object is found wherever the topology copy placed it.
/// See [`parse_txn_record_key`].
#[must_use]
pub fn txn_record_key(coordinator: TabletId, txn: TxnId) -> Key {
    let mut out = Vec::with_capacity(11 + 8 + 16);
    out.push(Key::SYSTEM_PREFIX);
    out.extend_from_slice(b"kivi/txn/");
    out.extend_from_slice(&coordinator.as_u64().to_be_bytes());
    out.extend_from_slice(&txn.as_bytes());
    Key::from(out)
}
