//! Project-owned partition hashing boundary (RFC §9).
//!
//! Routing needs a stable `key bytes → 128-bit hash` function that never changes
//! under a declared algorithm version. The directory consumes
//! [`PartitionHash`] values without knowing their producer; this module owns
//! the producers behind a single route hash.
//!
//! # Two jobs, not one
//!
//! The edge hashes a key twice - once to route, once inside the route - and the two
//! have opposite requirements:
//!
//! | | the route hash | the in-memory table's hash |
//! | --- | --- | --- |
//! | who computes it | every client, every node | the owning node, in process |
//! | must agree across nodes | yes | no |
//! | persisted | yes: directory ranges, checkpoint bands, route caches | no |
//! | needs a secret | **no** | **yes** |
//! | stable across versions and architectures | yes | no |
//!
//! A secret makes a route hash impossible, because a client that cannot compute the
//! route hash cannot address a tablet. The route hash is public by construction, and
//! this module's job is to make it stable and portable.
//!
//! # Why the route hash needs no cryptographic strength
//!
//! * `Directory::lookup_by_hash` resolves by **range containment**: the tablet whose
//!   range contains the hash. Two keys with the same 128-bit route hash therefore
//!   resolve to the **same tablet** - co-location, never misrouting, never a lost
//!   write. Full key equality, not the hash, decides membership.
//! * `BandLayout::band_of` takes the **top 8 bits**. An attacker reaches a chosen
//!   band in 2⁸ work and floods it with 2¹⁶ keys, against *any* 128-bit hash, keyed
//!   or not, secret or not. Tablet- and band-level hotspotting is not a property of
//!   the hash, and no choice of hash changes it. It belongs to admission control.
//! * So an attacker holding a public route hash does not need a collision to aim -
//!   they compute. A collision buys nothing they could not do by choosing keys.
//!
//! The one place a collision is genuinely expensive is the in-memory table's bucket
//! index, where an adaptive client can turn a collision into an O(n) probe. That hash
//! is secret, per-process, and never leaves the process, so it needs no cluster
//! lifecycle and no persisted secret. See [`crate::store`].
//!
//! # The construction
//!
//! ```text
//! route_hash = XXH3_128(key, seed = seed_for(namespace))
//! ```
//!
//! There is no second construction. This module once carried a `PartitionHasher`
//! enum selecting between BLAKE3-128 and XXH3-128, and while it did, nine
//! production sites computed a routing identity with BLAKE3 while the engine
//! routed with XXH3 - including the client, whose route cache was therefore keyed
//! on a value the server never produced and missed on every lookup. The enum is
//! gone; [`crate::route_hash`] is the only way to make a routing identity, and
//! nothing here selects an algorithm.
//!
//! BLAKE3 still appears elsewhere in Kivi, for the job it is good at:
//! content-addressing checkpoint artifacts. A content hash and a route hash answer
//! different questions, and conflating them is what let a route hash change under
//! a client that had no idea a choice had been made.
//!
//! # The table hash is not this hash
//!
//! The in-memory table uses its own keyed hash, held per store and never
//! persisted. This one is stable, public, and derived from a namespace, because a
//! client has to be able to compute a route without being trusted and a
//! checkpoint written on one node has to be readable on another.

use kivi_types::NamespaceId;

/// The construction's name, for operator-facing output.
///
/// A startup log should say how keys are partitioned. With one construction there
/// is no version to report, so this is a name and not an identifier: a future
/// change of construction changes it, and nothing in the format depends on it.
pub const ROUTE_HASH_NAME: &str = "xxh3-128";

/// The fixed seed every route hash uses.
///
/// Public, and deliberately so: a client has to compute this hash to pick a route,
/// and a directory snapshot written on one node has to be readable on another. See
/// this module's documentation for why a public route hash costs nothing.
pub const ROUTE_SEED: u64 = 0x243f_6a88_85a3_08d3;

/// The XXH3 seed a namespace's hashes are computed under.
///
/// `splitmix64`'s finaliser over the namespace mixed into [`ROUTE_SEED`]. Chosen
/// because it is a published, standard bit mixer with a full avalanche, so two
/// namespaces one apart get seeds with no visible relationship.
///
/// **This is domain separation, not secrecy.** The value is a `const fn` derived
/// from public inputs, it is frozen, and it is reproducible by anyone who has the
/// binary - which is required, because a client has to compute a route hash. See
/// [`crate::route`] for what the route hash is and is not allowed to be.
///
/// A `const fn` because the derivation is on the read path: three multiplies, once
/// per namespace, and [`crate::NamespaceRouteHasher`] hoists even that.
#[must_use]
pub const fn seed_for(namespace: NamespaceId) -> u64 {
    let mut z = ROUTE_SEED ^ namespace.as_u64().wrapping_mul(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// XXH3-128 of `key` under `seed`, as a `u128`.
///
/// One definition of the construction, used by both [`crate::route_hash`] and
/// [`crate::NamespaceRouteHasher`], because a hash with two spellings is a hash
/// whose two callers can disagree.
///
/// # The byte order is frozen
///
/// `twox_hash` returns a `u128` built as `(high64 << 64) | low64`, matching XXH3's
/// own `XXH128_hash_t { low64, high64 }` field order and therefore its in-memory
/// layout. Kivi stores it as that **number** and never reinterprets it through a
/// struct cast or a byte slice, so host endianness cannot change the value.
/// [`V2_FROZEN_VECTORS`] pins it, and the vectors are cross-checked against XXH3's
/// published `XXH3_128bits` output for the default secret and seed.
#[must_use]
pub fn xxh3_128(seed: u64, key: &[u8]) -> u128 {
    twox_hash::XxHash3_128::oneshot_with_seed(seed, key)
}

/// The namespaces the frozen vectors are stated under.
///
/// Three, because one namespace cannot show that namespace scoping works: a
/// construction that ignored the namespace would pass a single-namespace table
/// completely. `0` and `u64::MAX` are the two a derivation is most likely to
/// mishandle - one multiplies the golden-ratio constant to zero, the other relies
/// on wrapping arithmetic actually wrapping.
pub const V2_FROZEN_NAMESPACES: [u64; 3] = [0, 7, u64::MAX];

/// The keys the frozen vectors are stated over.
///
/// A separate constant from [`V2_FROZEN_VECTORS`] so the *matrix* is readable
/// without reading 60 hex numbers, and so adding a key is one line in one place
/// rather than a coordinated edit to two. The two are index-aligned and a test
/// asserts it, so a key added here without a vector fails rather than being
/// silently unchecked.
///
/// Every length in the specification's list is present, plus the families a framing
/// bug hides in:
///
/// * **Lengths.** 0, 1, 2, 3, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128,
///   129, 240, 241. The brackets straddle XXH3's `len <= 3`, 8, 16, 32, 64 and
///   128-byte special cases, and 240 is the secret-derivation cutoff, so 241 is the
///   first length that takes a different internal path from 240.
/// * **Binary zeros and all-ones**, at several lengths. A construction that
///   confuses a value with a length, or that pads rather than slices, collides here.
/// * **Prefix-related keys**: `k`, `kk`, `key`, `key\0`. A key that is also a
///   prefix of another, or of itself padded, is the family that a framing bug eats.
/// * **Embedded NUL**: `a\0b`. RESP carries binary-safe keys, and a C-string reading
///   would truncate `a\0b` to `a` while every other vector still passed.
pub const V2_FROZEN_KEYS: &[&[u8]] = &[
    b"",
    b"a",
    b"ab",
    b"abc",
    b"k",
    b"kk",
    b"kkkkkkk",
    b"kkkkkkkk",
    b"user:123",
    b"key",
    b"key\0",
    b"a\0b",
    b"\0",
    b"\0\0\0\0",
    &[b'k'; 15],
    &[b'k'; 16],
    &[b'k'; 17],
    &[b'k'; 31],
    &[b'k'; 32],
    &[b'k'; 33],
    &[b'k'; 63],
    &[b'k'; 64],
    &[b'k'; 65],
    &[b'k'; 127],
    &[b'k'; 128],
    &[b'k'; 129],
    &[b'k'; 240],
    &[b'k'; 241],
    &[0u8; 15],
    &[0u8; 16],
    &[0u8; 17],
    &[0u8; 64],
    &[0u8; 129],
    &[0xffu8; 15],
    &[0xffu8; 16],
    &[0xffu8; 17],
    &[0xffu8; 64],
    &[0xffu8; 128],
    &[0xffu8; 129],
];

/// The frozen test vectors for the route hash, index-aligned with
/// [`V2_FROZEN_KEYS`], under namespace [`V2_FROZEN_NAMESPACE`].
///
/// The deliverable that makes the algorithm reviewable: an identifier plus
/// published vectors can be implemented by a third party and checked, and a
/// construction that moves mints a new identifier rather than editing these,
/// because these are assertions and not comments.
///
/// **Kivi's vectors, not XXH3's.** `tests/routehash_upstream.rs` carries 60 verbatim
/// vectors from xxHash's own table, which pin the *construction underneath* - the
/// `twox-hash` port, the 128-bit mode, and the byte order of the two halves. This
/// table pins the *Kivi function on top of it*: the namespace seed derivation and
/// the framing. Both are needed and neither substitutes for the other.
pub const V2_FROZEN_VECTORS: &[u128] = &[
    0x4fbd_2e15_02a1_41c1_208a_7965_fe33_e53e,
    0x9afa_10b2_29cd_7e31_81b3_8ea1_9b0a_3e8f,
    0xd671_4a51_b457_8618_eb0e_3985_7e12_1a40,
    0x8ac6_2abe_3fdc_aeaf_639c_1d5f_72ee_2a89,
    0x8227_4337_9610_8022_2dbb_b1d2_cdd0_2f5e,
    0x59f5_8602_e770_0252_3491_3bbe_ab12_0f9c,
    0x9873_f3c2_463b_fe4a_6588_6aab_f92a_6dee,
    0x2da9_f92f_86bb_4b51_a9a4_98be_eca7_ff8d,
    0xf1f0_ee71_08f2_0582_fbd7_869a_105a_2773,
    0x5819_38d2_dedb_b9a5_f098_d9c7_0b3e_bd8d,
    0x8abc_b407_516d_8400_ab32_e9c2_f62e_2dc2,
    0x980e_eba6_480e_b734_6241_0552_f176_0ed6,
    0x7053_3be4_9f02_3c8f_0fe0_2a7b_94cd_e0ad,
    0xd43e_9999_2ed5_85b7_296b_af3f_3421_8392,
    0x1c2a_4cd5_c979_5faf_b44a_486b_3142_c32f,
    0xf0c7_f720_48db_2213_6292_300c_d908_bb48,
    0x6260_8b6d_376e_a946_232a_5843_121a_1580,
    0xf36d_29e4_23d4_ccff_70be_724c_0cc7_a99e,
    0x3755_84ce_a424_180b_4dad_c6df_d39e_4f33,
    0x02c7_cd41_8736_2207_3c48_655b_b17c_11d0,
    0xc69b_c74f_f449_113d_45c0_b105_0bb2_62c9,
    0xe670_7cd9_6b46_636f_13fd_9a8e_18b3_61fb,
    0xbe4f_ae2a_ae0f_2fcd_0719_0d3f_810e_5133,
    0xd391_53c8_6e60_125e_4e09_a5cb_1345_d88d,
    0x52c4_81f1_e2a4_0d67_3609_f1a2_446e_2813,
    0xee96_0ae3_8385_b1a0_fc43_3d17_126c_f3f9,
    0xa0fc_343c_ef15_b83c_e29d_5817_c8e3_ddac,
    0x22f2_d125_1f67_09e4_b93d_7e6d_9e72_1c68,
    0xb3b7_7c4b_9a10_0494_34ec_8893_e18c_6d09,
    0x9993_452f_1db1_85ab_aee5_5478_a197_b1e2,
    0xfd49_b5c9_70a1_7830_ae64_d637_2d86_92bf,
    0xd8d9_31de_8828_0973_2061_e3c3_2028_b756,
    0xe182_6361_da31_a021_92e4_29bc_1655_d654,
    0xb3ee_b4e4_9243_4ee3_825c_a57a_3eb9_f0f2,
    0x66c3_c662_5557_da05_2900_9049_aa37_c5c1,
    0x56c0_7d19_8cc8_ac9d_e407_dadd_1ba7_433c,
    0x165a_94bf_82b3_6b85_65fe_f763_4777_bd2f,
    0x8e6f_8715_9f41_7d3b_bf49_b10c_ea85_5a1d,
    0x9cba_9b2e_2853_1c16_d681_193f_04aa_e1bb,
];

/// The namespace every [`V2_FROZEN_VECTORS`] entry is stated under.
pub const V2_FROZEN_NAMESPACE: u64 = 7;

/// Identifies the partition-hash algorithm behind a routing decision.
/// Stored wherever routing provenance is recorded; unknown ids remain
#[cfg(test)]
mod tests {
    use super::*;
    use crate::route_hash;

    const NS: NamespaceId = NamespaceId::from_u64(7);

    /// A key's routing identity is a function of the key and the namespace, and of
    /// nothing else - no clock, no address, no per-process state. A client has to
    /// be able to compute the same value a server does, and that is only true if
    /// the derivation is pure.
    #[test]
    fn the_route_hash_is_deterministic_and_namespace_scoped() {
        let first = route_hash(NS, b"user:123");
        assert_eq!(route_hash(NS, b"user:123"), first);
        assert_ne!(route_hash(NS, b"user:124"), first);
        assert_ne!(
            route_hash(NamespaceId::from_u64(8), b"user:123"),
            first,
            "the namespace participates in the hash"
        );
    }

    /// A key and a key plus trailing zeros are distinct keys at every length, so
    /// "empty" can never be spelled two ways. The namespace enters as a seed rather
    /// than as prepended bytes, so there is no buffer in which a short key could be
    /// padded to a longer one's width.
    #[test]
    fn a_key_differs_from_its_zero_padded_form() {
        for length in 0..=64usize {
            let base: Vec<u8> = (0..length)
                .map(|i| u8::try_from(i % 251).expect("under 251").wrapping_mul(31) ^ 0x5a)
                .collect();
            let expected = route_hash(NS, &base);
            for zeros in 1..=17usize {
                let mut extended = base.clone();
                extended.resize(length + zeros, 0);
                assert_ne!(
                    expected,
                    route_hash(NS, &extended),
                    "a {length}-byte key collided with its {zeros}-zero-padded form"
                );
            }
        }
    }

    /// The seed derivation is the whole of namespace scoping, so it is pinned
    /// rather than left to the mixer's reputation: these are the seeds a third
    /// party needs in order to reproduce a route hash.
    #[test]
    fn the_seed_derivation_is_pinned() {
        // Namespace 0, the frozen vector namespace, and `u64::MAX` - the two ends
        // a derivation is most likely to mishandle.
        assert_eq!(seed_for(NamespaceId::from_u64(0)), 0xe9e0_033e_3bad_af36);
        assert_eq!(seed_for(NamespaceId::from_u64(7)), 0x1c6b_d5cc_e354_0403);
        assert_eq!(
            seed_for(NamespaceId::from_u64(u64::MAX)),
            0x2386_bcd9_343e_3c68
        );
    }

    /// The frozen vectors are the contract. A change here is not a refactor: it
    /// moves every key in every namespace, so a directory written by one build
    /// stops matching another.
    #[test]
    fn the_frozen_vectors_are_the_contract() {
        assert_eq!(
            V2_FROZEN_VECTORS.len(),
            V2_FROZEN_KEYS.len(),
            "every key needs a vector and every vector a key"
        );
        let namespace = NamespaceId::from_u64(V2_FROZEN_NAMESPACE);
        for (key, expected) in V2_FROZEN_KEYS.iter().zip(V2_FROZEN_VECTORS) {
            assert_eq!(
                route_hash(namespace, key).as_u128(),
                *expected,
                "the vector for a {}-byte key moved",
                key.len()
            );
        }
    }

    /// Scoping has to be visible in the vectors themselves, not only in the
    /// property test: a construction that ignored the namespace would pass a
    /// single-namespace table completely.
    #[test]
    fn the_frozen_vectors_span_several_namespaces() {
        assert!(
            V2_FROZEN_NAMESPACES.len() >= 2,
            "one namespace cannot show that scoping works"
        );
        let key = b"user:1";
        let mut seen = std::collections::HashSet::new();
        for namespace in V2_FROZEN_NAMESPACES {
            seen.insert(route_hash(NamespaceId::from_u64(namespace), key).as_u128());
        }
        assert_eq!(
            seen.len(),
            V2_FROZEN_NAMESPACES.len(),
            "two namespaces produced the same routing identity"
        );
    }
}
