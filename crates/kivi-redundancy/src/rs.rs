//! Reed-Solomon provider: production coding mechanism, Kivi-owned semantics.
//!
//! # Ecosystem selection and benchmarks
//!
//! Inspected September 2026:
//!
//! * `reed-solomon-erasure` v6.0.0 — classic Klaus-Post port, 5.9M downloads,
//!   MIT, `galois_8`/`galois_16` backends, optional `simd-accel` via bundled
//!   C kernels tuned for Haswell+ (`RUST_REED_SOLOMON_ERASURE_ARCH`). Last
//!   release 2022 (stale), C toolchain dependency, published ~1.1 GiB/s
//!   encode on Sia's 10+20/4 MiB bench vs 22–28 GiB/s for modern SIMD ports.
//! * `reed-solomon-simd` v3.1.0 — Leopard-RS port, pure Rust, runtime SIMD
//!   selection (SSSE3/AVX2/NEON + scalar fallback), `O(n log n)`, 2M
//!   downloads, maintained (2025, MSRV 1.82), MIT+BSD. Published 4.8–10 GiB/s
//!   encode and fastest decode in its `quick-comparison` vs
//!   `reed-solomon-16`, `reed-solomon-erasure`, `novelpoly`, `leopard-codec`.
//! * `sia_reed_solomon` (Aug 2026) — new Klaus-Post SIMD port, WASM-SIMD,
//!   wire-compatible, 6–25x faster than `reed-solomon-erasure`, but weeks old
//!   and browser-focused: too new for a durability baseline.
//!
//! Selected: `reed-solomon-simd`. Pure Rust (no C), actively maintained,
//! fastest maintained option, huge width support (we bound it down), even
//! shard sizes with 64-byte-multiple cross-version compatibility — which is
//! exactly what [`crate::RsParams`] (`V1`, 64-byte-aligned shards) encodes.
//! Local `benches/redundancy.rs` re-measures encode/decode on this machine
//! for representative Kivi asset sizes; see the Phase 10 report for numbers.
//!
//! Kivi owns everything around the kernels: [`RsParams`] validation, shard
//! padding/truncation, fragment identity/hashes, durable metadata, placement,
//! verification (a successful decode is never sufficient — bytes are
//! content-hash verified against [`crate::InformationAsset`] before
//! publication), and repair scheduling. No `reed-solomon-simd` type leaks
//! outside this module.

use crate::{RedundancyError, RsParams};

/// Maximum bytes one `encode` call buffers per shard (bounded by
/// [`RsParams::MAX_FRAGMENT_LEN`]; validation enforces it).
fn check_params(params: RsParams, logical_len: u64) -> Result<u32, RedundancyError> {
    params.validate()?;
    if logical_len > RsParams::MAX_ASSET_BYTES {
        return Err(RedundancyError::invalid_params(format!(
            "asset {logical_len} exceeds RS cap {}",
            RsParams::MAX_ASSET_BYTES
        )));
    }
    if logical_len == 0 {
        return Err(RedundancyError::invalid_params(
            "RS needs a nonempty asset (empty values stay inline)",
        ));
    }
    let expect = RsParams::shard_for_len(logical_len, params.data);
    if expect != params.fragment_len {
        return Err(RedundancyError::invalid_params(format!(
            "rs fragment_len {} disagrees with derived {expect} for len {logical_len}",
            params.fragment_len
        )));
    }
    Ok(params.fragment_len)
}

/// Splits padded asset bytes into `data` equal shards.
fn split_shards(padded: &[u8], data: u8, shard: usize) -> Vec<Vec<u8>> {
    let mut out = Vec::with_capacity(data as usize);
    for index in 0..data as usize {
        out.push(padded[index * shard..(index + 1) * shard].to_vec());
    }
    out
}

/// Encodes asset bytes into `data + parity` shards.
///
/// Returns shards in index order (`0..data` systematic data, `data..`
/// parity), each exactly `params.fragment_len` bytes. The caller assigns
/// fragment identities and hashes; this function never invents them.
///
/// # Errors
///
/// Returns [`RedundancyError`] on invalid params, oversize assets, or codec
/// failures.
pub fn encode(bytes: &[u8], params: RsParams) -> Result<Vec<Vec<u8>>, RedundancyError> {
    let shard = check_params(params, bytes.len() as u64)? as usize;
    // Pad to data * shard with zeros (explicit, deterministic; truncated on
    // decode and excluded by content-hash verification).
    let total = shard * params.data as usize;
    let mut padded = vec![0u8; total];
    padded[..bytes.len()].copy_from_slice(bytes);
    let original: Vec<&[u8]> = (0..params.data as usize)
        .map(|i| &padded[i * shard..(i + 1) * shard])
        .collect();
    let recovery = reed_solomon_simd::encode(
        usize::from(params.data),
        usize::from(params.parity),
        &original,
    )
    .map_err(|error| RedundancyError::InvalidParams {
        detail: format!("rs encode rejected params: {error:?}"),
    })?;
    let mut shards = split_shards(&padded, params.data, shard);
    for parity in recovery {
        debug_assert_eq!(parity.len(), shard);
        shards.push(parity);
    }
    debug_assert_eq!(shards.len(), params.total() as usize);
    Ok(shards)
}

/// Decodes the asset from any `data` shards of `data + parity`.
///
/// `present` maps shard index → shard bytes (each `fragment_len`); at least
/// `data` entries must be present. Returns the truncated logical bytes
/// (`logical_len`); the caller must content-verify them against the asset id
/// before publication — a successful decode alone proves nothing.
///
/// # Errors
///
/// Returns [`RedundancyError`] on invalid params, too few shards, ragged
/// shards, codec failures, or length disagreement.
pub fn decode(
    present: &[(u32, Vec<u8>)],
    params: RsParams,
    logical_len: u64,
) -> Result<Vec<u8>, RedundancyError> {
    params.validate()?;
    if present.len() < params.data as usize {
        return Err(RedundancyError::Unrecoverable {
            detail: format!(
                "rs needs {} shards, only {} present",
                params.data,
                present.len()
            ),
        });
    }
    let shard = params.fragment_len as usize;
    for (index, bytes) in present {
        if *index >= u32::from(params.total()) {
            return Err(RedundancyError::CorruptFragment {
                detail: format!("rs shard index {index} out of range"),
            });
        }
        if bytes.len() != shard {
            return Err(RedundancyError::CorruptFragment {
                detail: format!(
                    "rs shard {index} length {} disagrees with fragment_len {shard}",
                    bytes.len()
                ),
            });
        }
    }
    // Partition into original vs recovery by index.
    let mut original: Vec<(usize, &[u8])> = Vec::new();
    let mut recovery: Vec<(usize, &[u8])> = Vec::new();
    for (index, bytes) in present.iter().take(usize::from(params.data)) {
        // `take(data)` is enough: the decoder needs any `data` shards, and
        // the first `data` present suffice (MDS threshold, exact).
        let _ = (index, bytes);
    }
    for (index, bytes) in present {
        let idx = *index as usize;
        if idx < params.data as usize {
            original.push((idx, bytes.as_slice()));
        } else {
            recovery.push((idx - params.data as usize, bytes.as_slice()));
        }
    }
    // Fast path: all data shards present — concatenate, no decode.
    if original.len() == params.data as usize {
        let mut by_index = original;
        by_index.sort_by_key(|(i, _)| *i);
        // `logical_len` <=64MiB, fits `usize` even on 32-bit.
        #[allow(clippy::cast_possible_truncation)]
        let image_len = logical_len as usize;
        let mut out = Vec::with_capacity(image_len);
        for (_, bytes) in by_index {
            out.extend_from_slice(bytes);
        }
        out.truncate(image_len);
        if out.len() as u64 != logical_len {
            return Err(RedundancyError::Unrecoverable {
                detail: "rs truncation disagrees with logical length".to_owned(),
            });
        }
        return Ok(out);
    }
    let restored = reed_solomon_simd::decode(
        usize::from(params.data),
        usize::from(params.parity),
        original,
        recovery,
    )
    .map_err(|error| RedundancyError::Unrecoverable {
        detail: format!("rs decode failed: {error:?}"),
    })?;
    // `restored` maps missing original index → bytes; merge with present
    // originals to rebuild the full data image in order.
    let mut data_image: Vec<Option<Vec<u8>>> = (0..params.data as usize).map(|_| None).collect();
    for (index, bytes) in present {
        let idx = *index as usize;
        if idx < params.data as usize {
            data_image[idx] = Some(bytes.clone());
        }
    }
    for (index, bytes) in restored {
        if index < data_image.len() && data_image[index].is_none() {
            data_image[index] = Some(bytes);
        }
    }
    // `logical_len` <=64MiB, fits `usize` even on 32-bit.
    #[allow(clippy::cast_possible_truncation)]
    let image_len = logical_len as usize;
    let mut out = Vec::with_capacity(image_len);
    for slot in data_image {
        let Some(shard_bytes) = slot else {
            return Err(RedundancyError::Unrecoverable {
                detail: "rs decode left a data shard unrestored".to_owned(),
            });
        };
        out.extend_from_slice(&shard_bytes);
    }
    out.truncate(image_len);
    if out.len() as u64 != logical_len {
        return Err(RedundancyError::Unrecoverable {
            detail: "rs truncation disagrees with logical length".to_owned(),
        });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params_for(len: u64, data: u8, parity: u8) -> RsParams {
        RsParams {
            data,
            parity,
            fragment_len: RsParams::shard_for_len(len, data),
        }
    }

    fn bytes(len: usize) -> Vec<u8> {
        // Modulo 251 bounds the value to `0..251`, fits `u8`.
        #[allow(clippy::cast_possible_truncation)]
        fn byte(i: usize) -> u8 {
            ((i * 13 + 5) % 251) as u8
        }
        (0..len).map(byte).collect()
    }

    #[test]
    fn round_trips_all_missing_combinations() {
        let original = bytes(10_000);
        let params = params_for(original.len() as u64, 4, 2);
        let shards = encode(&original, params).expect("encodes");
        assert_eq!(shards.len(), 6);
        for shard in &shards {
            assert_eq!(shard.len(), params.fragment_len as usize);
        }
        // Drop every 2-combination; each must still decode exactly.
        for a in 0..6u32 {
            for b in (a + 1)..6u32 {
                let mut present = Vec::new();
                for (index, shard) in shards.iter().enumerate() {
                    // Index <6, far below `u32::MAX`.
                    #[allow(clippy::cast_possible_truncation)]
                    let index = index as u32;
                    if index != a && index != b {
                        present.push((index, shard.clone()));
                    }
                }
                let back = decode(&present, params, original.len() as u64).expect("decodes");
                assert_eq!(back, original, "missing {a},{b}");
            }
        }
        // Three missing exceeds parity: fails closed.
        let present: Vec<(u32, Vec<u8>)> = shards
            .iter()
            .enumerate()
            .take(3)
            .map(|(i, s)| {
                // Index <6, far below `u32::MAX`.
                #[allow(clippy::cast_possible_truncation)]
                let index = i as u32;
                (index, s.clone())
            })
            .collect();
        assert!(decode(&present, params, original.len() as u64).is_err());
    }

    #[test]
    fn wrong_shard_sizes_fail_loudly() {
        let original = bytes(1000);
        let params = params_for(original.len() as u64, 4, 2);
        let mut shards = encode(&original, params).expect("encodes");
        shards[0].push(0);
        let present: Vec<(u32, Vec<u8>)> = shards
            .into_iter()
            .enumerate()
            .map(|(i, s)| {
                // Index <6, far below `u32::MAX`.
                #[allow(clippy::cast_possible_truncation)]
                let index = i as u32;
                (index, s)
            })
            .collect();
        assert!(decode(&present, params, original.len() as u64).is_err());
    }

    #[test]
    fn fragment_len_must_be_derived() {
        let original = bytes(1000);
        let bad = RsParams {
            data: 4,
            parity: 2,
            fragment_len: 64,
        };
        // 1000/4 = 250 -> padded 256, not 64.
        assert!(encode(&original, bad).is_err());
    }
}
