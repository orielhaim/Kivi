//! Strongly typed identifiers and shared domain primitives for Kivi.
//!
//! This crate is the bottom of the dependency graph: it depends on nothing but
//! `std`, introduces no runtime, I/O, clock, randomness, or framework
//! dependencies, and defines no wire framing. It owns _value semantics_
//! (identity, ordering, fencing comparisons); the canonical little-endian
//! binary encoding of every type here is owned by `kivi-codec`.
//!
//! Design rules applied throughout (per `docs/rfc.md`):
//!
//! * Distinct concepts are distinct types: a [`TabletId`] can never be passed
//!   where a [`WorkerId`] is expected, even though both wrap a `u64`.
//! * Fencing-related generations treat `0` as an invalid sentinel. The default
//!   (zeroed) value therefore always fails closed instead of authorizing
//!   anything. None of these types implement `Default`, so call sites must
//!   choose [`TabletEpoch::INITIAL`] (or equivalent) explicitly.
//! * Time is always passed in, never read: helpers such as
//!   [`Expiry::is_expired`] take an explicit timestamp so deterministic
//!   simulation can supply virtual time.

pub mod authority;
pub mod hash;
pub mod identity;
pub mod ids;
pub mod names;
pub mod position;
pub mod reads;
pub mod time;

pub use authority::{AuthorityMismatch, TabletAuthority};
pub use hash::{ChunkId, ManifestId, PartitionHash};
pub use identity::{IdempotencyKey, MutationIdentity, RequestIdentity};
pub use ids::{
    ClusterId, CommitPosition, CpuId, GenerationExhausted, NamespaceId, NodeId, NodeIncarnation,
    NumaId, RequestSeq, SecurityDomainId, SequenceExhausted, SessionId, TabletEpoch, TabletId,
    WorkerId, WriteGuardGeneration,
};
pub use names::{MAX_NAMESPACE_NAME_LEN, NamespaceName, NamespaceNameError};
pub use position::CommitToken;
pub use reads::ReadContract;
pub use time::{Expiry, Ticks, UnixMicros};
