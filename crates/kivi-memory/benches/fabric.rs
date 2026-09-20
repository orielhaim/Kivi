//! Phase-9 Memory Fabric costs against the all-DRAM baseline: inline
//! and arena point reads, mixed insert/read throughput, the background
//! compression/demotion path, and the cold promotion path. Repeatable
//! locally with `cargo bench -p kivi-memory`.
//!
//! Reporting follows one style everywhere: an `ItemsCount` throughput
//! counter plus divan 0.1's built-in timing distribution
//! (mean/median/min/max). Divan 0.1.21 reports no p99-style latency
//! percentiles, so p50/p99/p99.9 stay in `tests/measure.rs`, which
//! computes them by hand.

use divan::{Bencher, black_box, counter::ItemsCount};

use bytes::Bytes;
use kivi_memory::{
    BehaviorClass, GetOutcome, MaterializationIntent, MemoryFabric, MemoryFabricConfig, Mutability,
    NvmeOptions,
};

fn main() {
    divan::main();
}

fn fabric() -> MemoryFabric {
    MemoryFabric::new(MemoryFabricConfig {
        arena_capacity_bytes: 256 << 20,
        nvme_capacity_bytes: 1 << 30,
        ..MemoryFabricConfig::default()
    })
    .expect("fabric opens")
}

fn medium_bytes() -> Bytes {
    Bytes::from(vec![0xABu8; 2048])
}

/// Highly compressible bytes: a repeated phrase (same generator as the
/// integration tests, so benches replay identical payloads).
fn compressible_bytes(len: usize) -> Bytes {
    const PHRASE: &[u8] = b"the quick brown fox jumps over the lazy dog. pack my box! ";
    let mut out = Vec::with_capacity(len);
    while out.len() < len {
        let take = (len - out.len()).min(PHRASE.len());
        out.extend_from_slice(&PHRASE[..take]);
    }
    Bytes::from(out)
}

/// Deterministic incompressible bytes from a fixed-seed xorshift stream
/// (same generator as the integration tests).
fn incompressible_bytes(len: usize, seed: u64) -> Bytes {
    let mut state = if seed == 0 { 1 } else { seed };
    let mut out = vec![0u8; len];
    for byte in &mut out {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        *byte = u8::try_from((state >> 11) & 0xFF).expect("masked to one byte");
    }
    Bytes::from(out)
}

#[divan::bench]
fn get_inline_baseline(bencher: Bencher) {
    let mut fabric = fabric();
    let object = fabric
        .insert(
            Bytes::from_static(b"tiny baseline value"),
            MaterializationIntent::hot(),
            BehaviorClass::HotMutable,
            Mutability::Mutable,
        )
        .expect("insert");
    bencher
        .counter(ItemsCount::new(1u64))
        .bench_local(|| black_box(fabric.get(object).expect("get")));
}

#[divan::bench]
fn get_arena_resident(bencher: Bencher) {
    let mut fabric = fabric();
    let object = fabric
        .insert(
            medium_bytes(),
            MaterializationIntent::hot(),
            BehaviorClass::HotMutable,
            Mutability::Mutable,
        )
        .expect("insert");
    bencher
        .counter(ItemsCount::new(1u64))
        .bench_local(|| black_box(fabric.get(object).expect("get")));
}

#[divan::bench]
fn insert_medium(bencher: Bencher) {
    let mut fabric = fabric();
    bencher.counter(ItemsCount::new(1u64)).bench_local(|| {
        black_box(
            fabric
                .insert(
                    medium_bytes(),
                    MaterializationIntent::hot(),
                    BehaviorClass::HotMutable,
                    Mutability::Mutable,
                )
                .expect("insert"),
        )
    });
}

#[divan::bench]
fn tick_with_cold_set(bencher: Bencher) {
    let mut fabric = fabric();
    for index in 0u64..200 {
        let mut bytes = vec![0u8; 1024];
        for (offset, byte) in bytes.iter_mut().enumerate() {
            let value = u64::try_from(offset).unwrap_or(0) ^ index;
            *byte = u8::try_from(value & 0xFF).unwrap_or(0);
        }
        let _ = fabric.insert(
            Bytes::from(bytes),
            MaterializationIntent::cold(),
            BehaviorClass::ColdCandidate,
            Mutability::ReadMostly,
        );
    }
    bencher.bench_local(|| {
        fabric.tick();
        black_box(());
    });
}

/// Background tick with the arena forced into compression: a 32 KiB
/// arena overflows, and the file backend runs with a zero endurance
/// budget, so demotion admission always rejects and pressure relief
/// must compress in-tick (lz4 build/verify/publish). Pins the
/// compression-path cost against `tick_with_cold_set`.
#[divan::bench]
fn tick_constrained_compressed(bencher: Bencher) {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut fabric = MemoryFabric::new(MemoryFabricConfig {
        arena_capacity_bytes: 32 << 10,
        nvme_capacity_bytes: 16 << 20,
        nvme_dir: Some(dir.path().to_owned()),
        nvme_options: NvmeOptions {
            daily_write_budget_bytes: 0,
            ..NvmeOptions::default()
        },
        ..MemoryFabricConfig::default()
    })
    .expect("fabric opens");
    let _ = fabric.insert(
        compressible_bytes(4096),
        MaterializationIntent::cold(),
        BehaviorClass::ColdCandidate,
        Mutability::ReadMostly,
    );
    // Incompressible neighbors fill the arena past the pressure floor
    // so reclamation must act on every tick.
    for index in 0..6u64 {
        let _ = fabric.insert(
            incompressible_bytes(4096, 100 + index),
            MaterializationIntent::cold(),
            BehaviorClass::HotMutable,
            Mutability::Mutable,
        );
    }
    bencher.counter(ItemsCount::new(1u64)).bench_local(|| {
        fabric.tick();
        black_box(());
    });
}

/// Background tick plus synchronous file-backend demotion: a 64 KiB
/// arena overflows with incompressible objects (which can never
/// compress), so the planner demotes to the `TempDir` file backend and
/// `pump_offcore` performs real file I/O per record. Pins the
/// demotion-path cost (planning plus device write plus shadow publish).
#[divan::bench]
fn tick_constrained_nvme(bencher: Bencher) {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut fabric = MemoryFabric::new(MemoryFabricConfig {
        arena_capacity_bytes: 64 << 10,
        nvme_capacity_bytes: 16 << 20,
        nvme_dir: Some(dir.path().to_owned()),
        ..MemoryFabricConfig::default()
    })
    .expect("fabric opens");
    for index in 0..12u64 {
        let _ = fabric.insert(
            incompressible_bytes(4096, 200 + index),
            MaterializationIntent::cold(),
            BehaviorClass::HotMutable,
            Mutability::Mutable,
        );
    }
    bencher.counter(ItemsCount::new(1u64)).bench_local(|| {
        fabric.tick();
        while fabric.pump_offcore().unwrap_or(false) {}
        black_box(());
    });
}

/// Cold read including promotion: the rotation set is demoted to
/// exclusively off-core residence (no DRAM shadow, proven with
/// `try_resolve_sync`), so each first touch enqueues a promote, the
/// pump serves the device read, and the re-issue hits the new shadow.
/// Compare against `get_arena_resident` for the all-DRAM floor. The
/// tail (everything promoted) degrades to shadow hits, so the median
/// mixes miss and hit cost.
#[divan::bench]
fn get_cold_promote(bencher: Bencher) {
    let mut fabric = MemoryFabric::new(MemoryFabricConfig {
        arena_capacity_bytes: 64 << 10,
        nvme_capacity_bytes: 16 << 20,
        ..MemoryFabricConfig::default()
    })
    .expect("fabric opens");
    // Fill past the pressure floor; `HotMutable` objects never compress,
    // so surplus bytes must demote. Failed inserts (arena full) just
    // stop the fill.
    let mut objects = Vec::new();
    for index in 0..40u64 {
        match fabric.insert(
            incompressible_bytes(2048, 300 + index),
            MaterializationIntent::cold(),
            BehaviorClass::HotMutable,
            Mutability::Mutable,
        ) {
            Ok(object) => objects.push(object),
            Err(_) => break,
        }
    }
    for _ in 0..8 {
        fabric.tick();
        while fabric.pump_offcore().unwrap_or(false) {}
    }
    let demoted: Vec<u64> = objects
        .into_iter()
        .filter(|object| fabric.try_resolve_sync(*object).is_err())
        .collect();
    assert!(
        !demoted.is_empty(),
        "setup must demote at least one object off-core"
    );
    let mut next = 0usize;
    bencher.counter(ItemsCount::new(1u64)).bench_local(|| {
        let object = demoted[next % demoted.len()];
        next += 1;
        match fabric.get(object).expect("get serves") {
            GetOutcome::Ready(bytes) => {
                black_box(bytes);
            }
            GetOutcome::Pending { .. } => {
                while fabric.pump_offcore().unwrap_or(false) {}
                match fabric.get(object).expect("get after promote") {
                    GetOutcome::Ready(bytes) => {
                        black_box(bytes);
                    }
                    // No arena headroom for the shadow (stay Pending):
                    // still a served read, just without the fast path.
                    GetOutcome::Pending { .. } => {
                        black_box(());
                    }
                }
            }
        }
    });
}
