//! `RouteHash128`: the routing identity, under a contract that says what it is
//! *not*.
//!
//! # The contract, stated negatively because that is the part that matters
//!
//! `RouteHash128` is a **deterministic, stable, public** function of
//! `(namespace, key bytes)`. It is:
//!
//! * a partition hash - the value the directory's prefix tree maps to a tablet,
//! * a range/tablet selection input,
//! * a physical band input for checkpoint layout,
//! * a cross-node and cross-client routing identity, so a directory snapshot
//!   written on one architecture is readable on another and a client can compute
//!   where a key goes.
//!
//! It is **not**, and must never be used as:
//!
//! * a cryptographic digest,
//! * a HashDoS boundary,
//! * authentication, integrity, or a MAC,
//! * content addressing - that is BLAKE3's job, and BLAKE3 stays wherever its own
//!   contract requires it (manifest digests, WAL records, dedup identities).
//!
//! The construction is XXH3-128 with a namespace-derived seed. XXH3 upstream
//! deliberately stabilises its output across platforms and releases, which is
//! precisely what a persisted and distributed routing identity needs and precisely
//! what a cryptographic hash does not promise to give you cheaply.
//!
//! # Why the non-cryptographic property is acceptable *here*
//!
//! Because the exposure is availability, not correctness, and because it is not
//! actually an exposure:
//!
//! * The directory resolves by **range containment** ([`kivi_tablet::DirectorySnapshot`]
//!   finds the tablet whose range contains the hash), so two keys sharing a
//!   `RouteHash128` resolve to the *same* tablet. That is co-location. Full key
//!   equality decides membership, so a collision cannot misroute a write, lose one,
//!   or read the wrong value.
//! * An attacker holding a public route hash does not need a collision to aim -
//!   they compute. Reaching a chosen 8-bit band costs 2⁸ evaluations and flooding
//!   it costs 2¹⁶, against *any* 128-bit hash. That is a load-balancing problem and
//!   it belongs to admission control, per-client quotas and hot-tablet detection -
//!   not to the hash, and putting a cryptographic hash on the request path does not
//!   fix it, it only makes the attack costlier to *the attacker who does not need
//!   it*.
//! * The one place a collision is genuinely expensive is the **local table's**
//!   bucket index, which an adaptive client can walk key by key. That table is
//!   keyed and ephemeral (see [`crate::store`]), so it hashes the key bytes itself
//!   under its own secret rather than trusting this value.
//!
//! # Byte order, frozen
//!
//! XXH3-128 produces two 64-bit halves. `RouteHash128` stores them as
//! `(high64 << 64) | low64` - that is, the `u128` whose little-endian byte image is
//! `low64`'s eight bytes followed by `high64`'s eight, matching XXH3's own
//! `XXH128_hash_t { low64, high64 }` memory layout. It is a *numeric* value, not a
//! byte string: nothing in Kivi encodes it to bytes by casting a struct, so
//! endianness of the host cannot change it.
//! [`V2_FROZEN_VECTORS`](crate::hash::V2_FROZEN_VECTORS) pins it.
//!
//! # The namespace seed is domain separation, not secrecy
//!
//! [`seed_for`] is a published, frozen `splitmix64` finaliser. It exists so two
//! namespaces produce unrelated hashes. It is not a secret, it is not
//! attacker-resistant, and it is not derived from anything confidential. The value
//! is a `const fn` - three multiplies - and it is computed per namespace, not per
//! request, by [`NamespaceRouteHasher`].

use kivi_types::{NamespaceId, PartitionHash};

use crate::hash::{seed_for, xxh3_128};

/// A 128-bit routing identity: deterministic, stable, public, and **not** a
/// cryptographic digest.
///
/// See this module's documentation for the contract. The type exists to make the
/// separation structural rather than conventional:
///
/// * **No `From` conversions.** There is no `From<u128>`, no `From<PartitionHash>`,
///   and no `Into<PartitionHash>`. A route hash cannot silently become a persisted
///   partition hash, or a raw integer, without a call that names the conversion.
/// * **No `Default`, no `Hash`, no arithmetic.** It is a value, not a number you
///   compute with.
/// * The only way out is [`RouteHash128::as_partition_hash`], explicitly named
///   after the boundary it crosses. A `RouteHash128` is never decoded from a
///   persisted value: durable routing state stores a `PartitionHash`, and a
///   directory snapshot read back is handed to the directory as one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct RouteHash128(u128);

impl RouteHash128 {
    /// Wraps a raw 128-bit routing value.
    ///
    /// Named `from_raw` rather than implementing `From<u128>` so that producing one
    /// from an integer is a visible act at every call site.
    #[must_use]
    const fn from_raw(value: u128) -> Self {
        Self(value)
    }

    /// Converts to the persisted/routed type.
    ///
    /// Explicit and named, because this is the boundary where a routing value
    /// becomes durable state - a directory prefix, a checkpoint band, a snapshot.
    /// Nothing in Kivi performs it implicitly.
    #[must_use]
    pub const fn as_partition_hash(self) -> PartitionHash {
        PartitionHash::from_u128(self.0)
    }
}

/// The routing identity of `key` within `namespace`.
///
/// # This is the only way to make one
///
/// Every site that routes - the engine's snapshot, the cluster edge, the
/// control plane, the client's route cache, the split planner - reaches its
/// routing identity through this function. It exists because a hash with two
/// spellings is a hash whose two callers can disagree, and this tree had three:
/// `PartitionHasher::V1` (BLAKE3-128), `PartitionHasher::V2` (XXH3-128) and
/// `NamespaceRouteHasher` (XXH3-128 again, with the namespace seed hoisted).
///
/// The consequence was not theoretical. The client keyed its route cache with
/// V1 while the engine routed with V2, so every cache lookup missed and
/// `route_miss_redirects_then_goes_direct` measured 21 redirects where 1 is
/// correct. Nothing in the type system noticed, because both spellings returned
/// a `PartitionHash`.
///
/// # Infallible
///
/// There is no `Option` and no error variant. The previous signature returned
/// `Option<PartitionHash>` only so a caller could report that a *build* lacked a
/// hash algorithm; with one construction there is nothing to lack. A router that
/// cannot route has a coverage gap, which is `RoutingError::NoRoute` and is a
/// different question with a different answer.
///
/// The namespace enters as a **derived seed**, not as prepended bytes.
/// Prepending would need a contiguous `le64(ns) || key` slice, so the key would
/// have to be copied into a stack buffer on every request; deriving the seed
/// removes the buffer, the branch, and the allocation fallback. It is also the
/// better construction on its own terms:
///
/// * No concatenation means no framing hazard at all. The empty key, a
///   zero-padded key and a key with trailing zeros are three different inputs
///   to XXH3 rather than one padded buffer, so the whole class of bug a domain
///   tag exists to prevent is designed out rather than tested for.
/// * Two namespaces get two decorrelated seeds, so their hashes are unrelated
///   by construction rather than by a shared prefix.
#[must_use]
pub fn route_hash(namespace: NamespaceId, key: &[u8]) -> PartitionHash {
    NamespaceRouteHasher::new(namespace)
        .hash_key(key)
        .as_partition_hash()
}

/// A namespace's route hasher, with everything per-namespace work already done.
///
/// `twox_hash`'s `oneshot_with_seed` copies a 192-byte default secret onto the
/// stack on **every** call, even on the short-input path where the secret is never
/// read (the crate skips only the *derivation*, not the copy). That is pure
/// per-request setup the namespace knows once, so it is hoisted here: the request
/// path becomes `hasher.hash_key(key_bytes)` with no initialisation inside it.
#[derive(Debug, Clone)]
pub struct NamespaceRouteHasher {
    namespace: NamespaceId,
    seed: u64,
}

impl NamespaceRouteHasher {
    /// Precomputes everything a namespace needs. Called once per namespace, when
    /// the namespace is opened - not per request.
    #[must_use]
    pub const fn new(namespace: NamespaceId) -> Self {
        Self {
            namespace,
            seed: seed_for(namespace),
        }
    }

    /// The namespace this hasher is scoped to.
    #[must_use]
    pub const fn namespace(&self) -> NamespaceId {
        self.namespace
    }

    /// Hashes ordinary key bytes to a routing identity.
    #[must_use]
    pub fn hash_key(&self, key: &[u8]) -> RouteHash128 {
        RouteHash128::from_raw(xxh3_128(self.seed, key))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const NS: NamespaceId = NamespaceId::from_u64(7);

    /// The hoisted hasher and the free function are two spellings of one thing,
    /// and a router that used each would place a key differently if they ever
    /// diverged. The free function is defined in terms of the hasher, so this
    /// pins that it stays that way rather than becoming a second derivation.
    #[test]
    fn the_hoisted_hasher_and_the_free_function_agree() {
        let hasher = NamespaceRouteHasher::new(NS);
        for length in [0usize, 1, 16, 31, 64, 128, 241, 400] {
            let key: Vec<u8> = (0..length)
                .map(|i| u8::try_from(i % 251).expect("under 251"))
                .collect();
            assert_eq!(
                hasher.hash_key(&key).as_partition_hash(),
                crate::route_hash(NS, &key),
                "the hoisted hasher disagreed at {length} bytes"
            );
        }
    }
}
