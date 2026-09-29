//! Cross-implementation check of the construction behind `RouteHash128`.
//!
//! # Why this file exists separately from the frozen Kivi vectors
//!
//! `kivi_state::hash::V2_FROZEN_VECTORS` pins *Kivi's* routing function: namespace
//! seed derivation, then XXH3-128. That table can only prove Kivi has not changed.
//!
//! This table proves something stronger and more external: that the **construction
//! underneath it** is the published XXH3 and not a lookalike. If `twox-hash` ever
//! diverged from the reference - a mis-transcribed constant, a different byte order
//! for the two 64-bit halves, a "128-bit" mode that quietly returns 64 - these
//! would fail while every Kivi-owned vector still passed.
//!
//! # Provenance
//!
//! Verbatim from xxHash's own `tests/sanity_test_vectors.h`
//! (`Cyan4973/xxHash`, branch `dev`), table `XSUM_XXH128_testdata`, which is the
//! output of `XXH3_128bits_withSeed`. Each row is upstream's
//! `{ len, seed, { low64, high64 } }` recombined as the `u128` Kivi stores:
//! `(high64 << 64) | low64`.
//!
//! The test inputs are upstream's, not ours: `fillTestBuffer` is reproduced below
//! exactly, so the bytes hashed are the bytes upstream hashed.
//!
//! Three seeds, chosen from upstream's own table: zero and two non-zero. The
//! non-zero rows matter because a seed that is ignored, or applied to the wrong
//! field, passes every zero-seed vector.
//!
//! Lengths bracket every boundary in XXH3's dispatch: 0/1/2/3 and 7/8 (the
//! `len <= 3`, `4..8`, `9..16` special cases), 15/16/17 (the 16-byte lane pair),
//! 31/32/33 and 63/64/65 (32-byte stripes), 127/128/129 (two stripes), and
//! 240/241 - 240 is the secret-derivation cutoff, so 241 is the first length that
//! derives its secret from the seed rather than using the default.

use kivi_types::NamespaceId;

use kivi_state::{route_hash, seed_for, xxh3_128};

/// Upstream's `PRIME32`, the initial value of the test buffer's generator.
const PRIME32: u64 = 2_654_435_761;
/// Upstream's `PRIME64`, the generator's multiplier.
const PRIME64: u64 = 11_400_714_785_074_694_797;

/// Upstream's `fillTestBuffer`, byte for byte.
///
/// ```c
/// XSUM_U64 byteGen = PRIME32;
/// for (i = 0; i < len; ++i) { buffer[i] = (XSUM_U8)(byteGen>>56); byteGen *= PRIME64; }
/// ```
#[must_use]
fn upstream_test_buffer(len: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(len);
    let mut byte_gen = PRIME32;
    for _ in 0..len {
        out.push((byte_gen >> 56) as u8);
        byte_gen = byte_gen.wrapping_mul(PRIME64);
    }
    out
}

/// Upstream's `XSUM_XXH128_testdata`, for the lengths that bracket XXH3's dispatch
/// boundaries. `(input length, seed, expected u128)`.
const UPSTREAM_XXH3_128: &[(u32, u64, u128)] = &[
    (
        0,
        0x0000_0000_0000_0000,
        0x99aa_06d3_0147_98d8_6001_c324_468d_497f,
    ),
    (
        0,
        0x0000_0000_9e37_79b1,
        0x9222_0ae5_5e14_ab50_5444_f786_9c67_1ab0,
    ),
    (
        0,
        0x9e37_79b1_85eb_ca8d,
        0x00fe_aa73_2a3c_e25e_a986_dfc5_d760_5bfe,
    ),
    (
        1,
        0x0000_0000_0000_0000,
        0xa6cd_5e93_9200_0f6a_c44b_dff4_074e_ecdb,
    ),
    (
        1,
        0x0000_0000_9e37_79b1,
        0x89b9_9554_ba22_467c_b53d_5557_e7f7_6f8d,
    ),
    (
        1,
        0x9e37_79b1_85eb_ca8d,
        0x20e4_9abc_c53b_3842_032b_e332_dd76_6ef8,
    ),
    (
        2,
        0x0000_0000_0000_0000,
        0x7675_0c3c_7bf9_5668_7a99_7804_4cb8_a8bb,
    ),
    (
        2,
        0x0000_0000_9e37_79b1,
        0x8b75_a791_ec03_4873_8295_910c_7638_b180,
    ),
    (
        2,
        0x9e37_79b1_85eb_ca8d,
        0x7b96_e6a6_00da_e67d_764b_35c9_0519_ad88,
    ),
    (
        3,
        0x0000_0000_0000_0000,
        0x20ef_c49f_f024_22ea_5424_7382_a8d6_b94d,
    ),
    (
        3,
        0x0000_0000_9e37_79b1,
        0x48f8_2c2f_e0ab_d468_f173_d14d_ad53_a5dc,
    ),
    (
        3,
        0x9e37_79b1_85eb_ca8d,
        0x1c7e_cf6a_308c_f00e_634b_8990_b497_6373,
    ),
    (
        7,
        0x0000_0000_0000_0000,
        0xdd9b_6039_f79e_c416_081c_22dd_284a_2f0a,
    ),
    (
        7,
        0x0000_0000_9e37_79b1,
        0x0bf1_be77_9aac_3093_8bb5_45dd_2d2c_9c7a,
    ),
    (
        7,
        0x9e37_79b1_85eb_ca8d,
        0x833c_ba02_82dd_6619_d5da_1979_2393_8e53,
    ),
    (
        8,
        0x0000_0000_0000_0000,
        0x47a7_f080_d82b_b456_64c6_9cab_4bb2_1dc5,
    ),
    (
        8,
        0x0000_0000_9e37_79b1,
        0xf959_0132_3265_5ff1_5f46_2f3d_e2e8_b940,
    ),
    (
        8,
        0x9e37_79b1_85eb_ca8d,
        0xf50c_ec14_5bcd_5c5a_7b29_471d_c729_b5ff,
    ),
    (
        15,
        0x0000_0000_0000_0000,
        0xc402_609e_57ee_5772_9589_55df_1889_e6bc,
    ),
    (
        15,
        0x0000_0000_9e37_79b1,
        0xcbf9_7d05_3781_b18e_69b2_91e8_b041_c577,
    ),
    (
        15,
        0x9e37_79b1_85eb_ca8d,
        0x4886_4ef5_80a9_5f6b_6a0a_9ca5_fd33_cb9d,
    ),
    (
        16,
        0x0000_0000_0000_0000,
        0xc68c_368e_cf8a_9c05_5629_8025_8a99_8629,
    ),
    (
        16,
        0x0000_0000_9e37_79b1,
        0x3767_c90d_0cdb_b93d_b07e_eeab_4c56_392b,
    ),
    (
        16,
        0x9e37_79b1_85eb_ca8d,
        0x6ffc_b80c_d330_85c8_0346_d13a_7a54_98c7,
    ),
    (
        17,
        0x0000_0000_0000_0000,
        0x955f_a786_43ed_3669_abbc_12d1_1973_d7db,
    ),
    (
        17,
        0x0000_0000_9e37_79b1,
        0x99e7_c628_e75d_6431_3cc9_ff6c_ae79_accb,
    ),
    (
        17,
        0x9e37_79b1_85eb_ca8d,
        0xd776_8121_9e46_4828_980a_1411_9985_a7df,
    ),
    (
        31,
        0x0000_0000_0000_0000,
        0x3010_48a7_ab47_6d21_ec83_65e7_4dc0_0653,
    ),
    (
        31,
        0x0000_0000_9e37_79b1,
        0x767f_caf8_af69_6da5_4375_0505_aa76_94e2,
    ),
    (
        31,
        0x9e37_79b1_85eb_ca8d,
        0x4639_cf7b_77ba_9096_d747_50f8_9523_60c3,
    ),
    (
        32,
        0x0000_0000_0000_0000,
        0x98fc_6458_710d_c2e8_2784_10a1_7595_e3f9,
    ),
    (
        32,
        0x0000_0000_9e37_79b1,
        0x326d_21e5_bcd3_95de_3589_c5cd_99cd_6267,
    ),
    (
        32,
        0x9e37_79b1_85eb_ca8d,
        0xcc58_7e4f_cdb8_6bc5_0054_e826_31ce_f166,
    ),
    (
        33,
        0x0000_0000_0000_0000,
        0x3103_c192_ceaa_2ded_e593_bc4e_5914_c9d1,
    ),
    (
        33,
        0x0000_0000_9e37_79b1,
        0x1b97_38af_be6c_bb38_5a78_5fec_2ae2_b28f,
    ),
    (
        33,
        0x9e37_79b1_85eb_ca8d,
        0x2127_3c81_90c6_45cd_c361_d36c_ea59_7c31,
    ),
    (
        63,
        0x0000_0000_0000_0000,
        0x7fc0_ae04_a6d2_bb0b_5e4c_b79e_9a81_21d9,
    ),
    (
        63,
        0x0000_0000_9e37_79b1,
        0xe4db_72a4_c9f1_78cf_32c0_f908_da71_d4fd,
    ),
    (
        63,
        0x9e37_79b1_85eb_ca8d,
        0xaec9_2695_d77f_ea89_c6ee_4817_4165_4b12,
    ),
    (
        64,
        0x0000_0000_0000_0000,
        0x6d90_e81a_9b0f_d622_efdb_6a44_6907_21a9,
    ),
    (
        64,
        0x0000_0000_9e37_79b1,
        0x5f29_e4ed_ba49_a7ae_592b_9762_bbeb_cebb,
    ),
    (
        64,
        0x9e37_79b1_85eb_ca8d,
        0x37b7_3896_8d40_bda5_9405_ba2a_ffa9_5ceb,
    ),
    (
        65,
        0x0000_0000_0000_0000,
        0x6c07_4d65_e54d_b85a_fe2f_650f_a500_ec6e,
    ),
    (
        65,
        0x0000_0000_9e37_79b1,
        0xfd0e_bd6e_8127_5e11_db4d_44a4_34cf_6830,
    ),
    (
        65,
        0x9e37_79b1_85eb_ca8d,
        0x7250_3a6f_a8d0_7adb_9d60_c345_e5c2_97cd,
    ),
    (
        127,
        0x0000_0000_0000_0000,
        0xdcfa_e800_2712_db1c_802a_565a_8a79_a999,
    ),
    (
        127,
        0x0000_0000_9e37_79b1,
        0xfd50_7af3_ede6_2925_716a_7d18_e901_9e45,
    ),
    (
        127,
        0x9e37_79b1_85eb_ca8d,
        0x3745_2e19_67d3_445b_54c9_d67f_8b29_dc74,
    ),
    (
        128,
        0x0000_0000_0000_0000,
        0x3999_2220_e045_260a_ebb1_5e34_a7fb_5ab1,
    ),
    (
        128,
        0x0000_0000_9e37_79b1,
        0x9880_1187_df8d_614d_1453_8199_41d9_3c1d,
    ),
    (
        128,
        0x9e37_79b1_85eb_ca8d,
        0xa0f7_ccb6_8ee0_2add_8394_f5c5_1f1d_8246,
    ),
    (
        129,
        0x0000_0000_0000_0000,
        0x0381_5fc9_1f1b_30b6_86c9_e3bc_8f0a_3b5c,
    ),
    (
        129,
        0x0000_0000_9e37_79b1,
        0xb7f7_349a_47b3_9e56_b37b_716f_66b4_0f02,
    ),
    (
        129,
        0x9e37_79b1_85eb_ca8d,
        0xad55_9266_067c_0bf3_d4aa_e26f_cec7_dc03,
    ),
    (
        240,
        0x0000_0000_0000_0000,
        0xaa42_02da_a276_9dc8_5c9a_ae94_c8eb_e5a0,
    ),
    (
        240,
        0x0000_0000_9e37_79b1,
        0xda88_8104_beae_5ae0_ca19_087f_1d33_5dae,
    ),
    (
        240,
        0x9e37_79b1_85eb_ca8d,
        0x29d2_133d_6ea5_8c5b_604e_98db_085c_1864,
    ),
    (
        241,
        0x0000_0000_0000_0000,
        0x99a8_0ecf_0ecf_c647_c5a6_39ec_d203_0e5e,
    ),
    (
        241,
        0x0000_0000_9e37_79b1,
        0x4bf2_229c_3a8f_c3c3_5927_e363_7bac_8149,
    ),
    (
        241,
        0x9e37_79b1_85eb_ca8d,
        0xec64_afae_6a13_7582_dda9_b0a1_61d4_829a,
    ),
];

#[test]
fn matches_upstream_xxh3_128_vectors() {
    for (length, seed, expected) in UPSTREAM_XXH3_128 {
        let input = upstream_test_buffer(*length as usize);
        assert_eq!(
            xxh3_128(*seed, &input),
            *expected,
            "XXH3-128 disagreed with the reference implementation at len {length} seed {seed:#018x}"
        );
    }
}

#[test]
fn the_upstream_buffer_generator_is_reproduced_correctly() {
    // The first byte of upstream's buffer, so a transcription error in the
    // generator fails here rather than as 60 mysterious hash mismatches.
    //
    // It is **zero**, which is worth asserting deliberately: `PRIME32` is
    // `0x9E3779B1`, only 32 bits wide, so the *first* `byteGen >> 56` is zero - and
    // only the first, because one multiplication by `PRIME64` carries it past 32
    // bits. Reading it as `0x9e` is the obvious mistake, it was this test's first
    // version, and the 60-vector test passed anyway; a check that is wrong and does
    // not fail is worse than no check. Hence all eight bytes, pinned.
    let buffer = upstream_test_buffer(8);
    assert_eq!(buffer, [0x00, 0x52, 0x92, 0x9b, 0xb7, 0x32, 0xa3, 0x24]);
}

#[test]
fn kivis_own_vectors_are_also_upstream_vectors() {
    // Kivi's namespace seeds are not upstream's, so this asserts the *shape* of the
    // relationship rather than sharing a table: the same construction, reached
    // through a different seed. It catches a `seed_for` that stopped feeding the
    // seed into XXH3, which would leave every upstream vector passing while every
    // Kivi vector went wrong.
    let namespace = NamespaceId::from_u64(1);
    let key = b"user:123";
    let via_hasher = route_hash(namespace, key).as_u128();
    let via_seed = xxh3_128(seed_for(namespace), key);
    assert_eq!(via_hasher, via_seed);
    assert_ne!(
        via_seed,
        xxh3_128(0, key),
        "a namespace must not hash as if it were namespace 0"
    );
}
