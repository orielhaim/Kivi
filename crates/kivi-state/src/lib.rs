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
pub mod mutation;
pub mod object;
pub mod ops;
pub mod store;
pub use envelope::{EnvelopeError, MUTATION_ENVELOPE_VERSION, MutationEnvelope};
pub use hash::{PARTITION_DOMAIN_TAG, PartitionHashAlgorithmId, PartitionHasher};
pub use mutation::{ApplyError, ApplyOutcome, Mutation};
pub use object::{
    ChunkedRef, Key, LogicalValue, ObjectType, ObjectVersion, StoredObject, VersionExhausted,
};
pub use ops::{
    DurableOutcome, ExpiryPolicy, OpError, Operation, OperationResult, SetCondition, outcome_for,
};
pub use store::{ObjectStore, Prepared, StorePrepared, slice_range, splice_inline};
