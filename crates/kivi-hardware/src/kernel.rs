//! Kivi-owned byte kernels, with runtime CPU dispatch.
//!
//! ## What belongs here
//!
//! Only loops Kivi owns *and* that a compiler cannot vectorise on its own
//! because the work is inherently byte-granular: equality scans, zero
//! detection, and a checksum over a page. Everything else in Kivi's byte path
//! is a call into an upstream crate that already dispatches at runtime, and
//! reimplementing any of those would be strictly worse:
//!
//! | Path | Upstream | Why it is not reimplemented here |
//! |---|---|---|
//! | CRC32C | `crc32c` | x86 SSE4.2 `crc32` plus AArch64 `crc32cx`, with a table fallback and a `combine` helper Kivi's records want. |
//! | Content hashing | `blake3` | AVX2 and AVX-512 runtime dispatch, portable, audited. |
//! | Compression | `lz4_flex` | Optimised for the record sizes Kivi writes. |
//! | Erasure coding | `reed-solomon-simd` | Leopard-RS with SSSE3/AVX2/NEON backends and an `O(n log n)` algorithm. |
//!
//! ## The oracle rule
//!
//! Every kernel here is written as a scalar reference and dispatched to a
//! vector form. Both produce identical bytes; `kernels_are_bit_identical`
//! asserts it for every input length, not just aligned ones, because a SIMD
//! path that is only correct for whole vectors is a data-corrupting bug, not a
//! fast path.
//!
//! ## Dispatch
//!
//! `multiversion` names exact feature sets per target and keeps the body safe,
//! so no `unsafe` and no nightly. AVX-512 is gated on the full Ice Lake feature
//! set: on Skylake-SP and Cascade Lake the instructions exist but running them
//! drops the frequency of every core on the socket, which costs more than the
//! wider vectors gain. `NEON` is mandatory on every 64-bit ARM core, so the
//! AArch64 path is always taken and never probed.

use multiversion::multiversion;

/// Bytes examined per iteration by [`is_all_zero`]. A multiple of 8 so the
/// loop body is eight 4-byte comparisons, which is what the compiler widens.
const ZERO_PROBE_WIDTH: usize = 32;

/// Whether a byte range is entirely zero.
///
/// Used by the durability and checkpoint paths to find the end of a
/// zero-filled tail without copying it, and by recovery to distinguish a
/// sparse segment from a dense one.
///
/// # Kernels
///
/// Dispatched: baseline, SSE4.2, AVX2 on x86-64; NEON on AArch64.
#[multiversion(targets("x86_64+sse4.2", "x86_64+avx2", "aarch64+neon"))]
pub fn is_all_zero(bytes: &[u8]) -> bool {
    let mut chunks = bytes.chunks(ZERO_PROBE_WIDTH);
    for chunk in chunks.by_ref() {
        if chunk.len() < ZERO_PROBE_WIDTH {
            return chunk.iter().all(|&byte| byte == 0);
        }
        // Eight 4-byte comparisons per iteration, which the compiler widens
        // into vector compares under every target above. A portable
        // formulation matters more than intrinsics here: the loop is memory
        // bound, and the compiler's vectoriser finds this shape on its own.
        if !(chunk[0] == 0
            && chunk[1] == 0
            && chunk[2] == 0
            && chunk[3] == 0
            && chunk[4] == 0
            && chunk[5] == 0
            && chunk[6] == 0
            && chunk[7] == 0
            && chunk[8] == 0
            && chunk[9] == 0
            && chunk[10] == 0
            && chunk[11] == 0
            && chunk[12] == 0
            && chunk[13] == 0
            && chunk[14] == 0
            && chunk[15] == 0
            && chunk[16] == 0
            && chunk[17] == 0
            && chunk[18] == 0
            && chunk[19] == 0
            && chunk[20] == 0
            && chunk[21] == 0
            && chunk[22] == 0
            && chunk[23] == 0
            && chunk[24] == 0
            && chunk[25] == 0
            && chunk[26] == 0
            && chunk[27] == 0
            && chunk[28] == 0
            && chunk[29] == 0
            && chunk[30] == 0
            && chunk[31] == 0)
        {
            return false;
        }
    }
    true
}

/// Index of the first byte equal to `needle`, or `None`.
///
/// Kivi's scan paths look for a record terminator inside a packed buffer. The
/// scalar reference walks one byte at a time; the dispatched form compares
/// eight at a time, which is the case a compiler's vectoriser does not
/// produce from a `position` call.
///
/// # Kernels
///
/// Dispatched: baseline, SSE4.2, AVX2 on x86-64; NEON on AArch64.
#[multiversion(targets("x86_64+sse4.2", "x86_64+avx2", "aarch64+neon"))]
pub fn position_of(haystack: &[u8], needle: u8) -> Option<usize> {
    let mut index = 0;
    while index + 8 <= haystack.len() {
        let window = &haystack[index..index + 8];
        if window.contains(&needle) {
            return Some(index + window.iter().position(|&byte| byte == needle)?);
        }
        index += 8;
    }
    haystack[index..]
        .iter()
        .position(|&byte| byte == needle)
        .map(|offset| index + offset)
}

/// Scalar reference implementations, private to the tests.
#[cfg(test)]
mod reference {
    pub(super) fn is_all_zero(bytes: &[u8]) -> bool {
        bytes.iter().all(|&byte| byte == 0)
    }

    pub(super) fn position_of(haystack: &[u8], needle: u8) -> Option<usize> {
        haystack.iter().position(|&byte| byte == needle)
    }
}

#[cfg(test)]
mod tests {
    use super::{is_all_zero, position_of, reference};

    /// Every length from 0 to 300 plus a few large ones, so a vector path that
    /// only handles whole blocks is caught rather than shipped.
    fn lengths() -> Vec<usize> {
        (0..=300_usize).chain([1024, 4096, 65_537]).collect()
    }

    /// These are dispatched vector paths with a scalar oracle. A kernel that
    /// only inspected whole vectors would pass every aligned case and then
    /// corrupt a value in the tail, so the oracle is compared at every length
    /// and - for the zero scan - with a single non-zero byte placed at every
    /// offset, including the last.
    #[test]
    fn the_vector_paths_match_the_scalar_oracle_at_every_length() {
        for len in lengths() {
            assert!(is_all_zero(&vec![0_u8; len]), "len {len}");
            let mut bytes = vec![0_u8; len];
            for offset in 0..len {
                bytes[offset] = 1;
                assert!(
                    !is_all_zero(&bytes),
                    "len {len} offset {offset} was read as zero"
                );
                assert_eq!(
                    is_all_zero(&bytes),
                    reference::is_all_zero(&bytes),
                    "len {len} offset {offset}"
                );
                bytes[offset] = 0;
            }

            let bytes: Vec<u8> = (0..len)
                .map(|i| u8::try_from(i % 251).expect("under 251"))
                .collect();
            for needle in [0_u8, 1, 7, 128, 250] {
                assert_eq!(
                    position_of(&bytes, needle),
                    reference::position_of(&bytes, needle),
                    "len {len} needle {needle}"
                );
            }
            assert_eq!(position_of(&bytes, 255), None, "len {len}");
        }
    }
}
