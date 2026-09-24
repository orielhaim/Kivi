//! Redundancy benchmarks: encode/decode throughput, degraded reads, repair.
//!
//! Measures encode/decode throughput and CPU/byte for replication vs
//! Reed-Solomon across representative Kivi asset sizes (4 KiB manifests,
//! 64 KiB bands, 1 MiB chunks, 4 MiB large bands), plus degraded-read
//! latency, storage amplification, and repair bandwidth. Run with
//! `cargo bench -p kivi-redundancy`.

use divan::Bencher;
use kivi_redundancy::{RedundancyIntent, RsParams, SchemeParams};

fn asset_bytes(len: usize) -> Vec<u8> {
    // Modulo 251 bounds the value to `0..251`, fits `u8`.
    #[allow(clippy::cast_possible_truncation)]
    fn byte(i: usize) -> u8 {
        ((i * 13 + 5) % 251) as u8
    }
    (0..len).map(byte).collect()
}

#[divan::bench(args = [4096, 65536, 1_048_576, 4_194_304])]
fn encode_replication(bencher: Bencher, len: usize) {
    let bytes = asset_bytes(len);
    bencher.bench(|| {
        kivi_redundancy::replication::encode(
            &bytes,
            kivi_redundancy::ReplicationParams { copies: 3 },
        )
        .expect("encodes")
    });
}

#[divan::bench(args = [65536, 1_048_576, 4_194_304])]
fn encode_rs_8_3(bencher: Bencher, len: usize) {
    let bytes = asset_bytes(len);
    let params = RsParams {
        data: 8,
        parity: 3,
        fragment_len: RsParams::shard_for_len(len as u64, 8),
    };
    bencher.bench(|| kivi_redundancy::rs::encode(&bytes, params).expect("encodes"));
}

#[divan::bench(args = [65536, 1_048_576])]
fn decode_rs_8_3_full(bencher: Bencher, len: usize) {
    let bytes = asset_bytes(len);
    let params = RsParams {
        data: 8,
        parity: 3,
        fragment_len: RsParams::shard_for_len(len as u64, 8),
    };
    let shards = kivi_redundancy::rs::encode(&bytes, params).expect("encodes");
    let present: Vec<(u32, Vec<u8>)> = shards
        .into_iter()
        .enumerate()
        .map(|(i, s)| {
            // Index <11, far below `u32::MAX`.
            #[allow(clippy::cast_possible_truncation)]
            let index = i as u32;
            (index, s)
        })
        .collect();
    bencher.bench(|| kivi_redundancy::rs::decode(&present, params, len as u64).expect("decodes"));
}

#[divan::bench(args = [1_048_576])]
fn decode_rs_8_3_degraded(bencher: Bencher, len: usize) {
    let bytes = asset_bytes(len);
    let params = RsParams {
        data: 8,
        parity: 3,
        fragment_len: RsParams::shard_for_len(len as u64, 8),
    };
    let shards = kivi_redundancy::rs::encode(&bytes, params).expect("encodes");
    // Drop 3 data shards: worst-case degraded read within tolerance.
    let present: Vec<(u32, Vec<u8>)> = shards
        .into_iter()
        .enumerate()
        .skip(3)
        .map(|(i, s)| {
            // Index <11, far below `u32::MAX`.
            #[allow(clippy::cast_possible_truncation)]
            let index = i as u32;
            (index, s)
        })
        .collect();
    bencher.bench(|| kivi_redundancy::rs::decode(&present, params, len as u64).expect("decodes"));
}

#[divan::bench]
fn plan_baseline_small(bencher: Bencher) {
    let intent = RedundancyIntent::survive_two();
    bencher.bench(|| {
        let _: SchemeParams = kivi_redundancy::plan_baseline(&intent, 4096).expect("plans");
    });
}

fn main() {
    divan::main();
}
