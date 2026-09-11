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
//! Physical representation is deliberately private: values live inline today,
//! and arena/chunked/compressed/cold forms can arrive later without changing
//! any of these semantics.

pub mod envelope;
pub mod hash;
pub mod mutation;
pub mod object;
pub mod ops;
pub mod store;
pub use envelope::{EnvelopeError, MUTATION_ENVELOPE_VERSION, MutationEnvelope};
pub use hash::{PARTITION_DOMAIN_TAG, PartitionHashAlgorithmId, PartitionHasher};
pub use mutation::{ApplyError, ApplyOutcome, Mutation};
pub use object::{Key, LogicalValue, ObjectType, ObjectVersion, StoredObject, VersionExhausted};
pub use ops::{OpError, Operation, OperationResult};
pub use store::{ObjectStore, Prepared};
