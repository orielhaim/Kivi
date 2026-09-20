//! Shared deterministic helpers for Memory Fabric integration tests.
//!
//! All byte generators use fixed seeds: the same test replays the same
//! bytes, placements, and transitions on every run.
//!
//! Helpers are shared across test targets, so not every target uses
//! every helper. The module-level dead-code allow keeps shared code
//! warning-free without splitting the module.
#![allow(dead_code)]

use bytes::Bytes;
use kivi_memory::{BehaviorClass, GetOutcome, MemoryFabric, MemoryFabricConfig, Mutability};

/// Creates a fabric with the given arena and `NVMe` capacities (sim
/// backend, deterministic seed).
#[must_use]
pub fn make_fabric(arena_bytes: u64, nvme_bytes: u64) -> MemoryFabric {
    MemoryFabric::new(MemoryFabricConfig {
        arena_capacity_bytes: arena_bytes,
        nvme_capacity_bytes: nvme_bytes,
        ..MemoryFabricConfig::default()
    })
    .expect("fabric opens")
}

/// Highly compressible bytes: a repeated 64-byte phrase.
#[must_use]
pub fn compressible_bytes(len: usize) -> Bytes {
    const PHRASE: &[u8] = b"the quick brown fox jumps over the lazy dog. pack my box! ";
    let mut out = Vec::with_capacity(len);
    while out.len() < len {
        let take = (len - out.len()).min(PHRASE.len());
        out.extend_from_slice(&PHRASE[..take]);
    }
    Bytes::from(out)
}

/// Deterministic incompressible bytes from a fixed-seed xorshift stream.
#[must_use]
pub fn incompressible_bytes(len: usize, seed: u64) -> Bytes {
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

/// Pumps off-core operations until idle, returning completed count.
pub fn pump_all(fabric: &mut MemoryFabric) -> usize {
    let mut count = 0;
    while fabric.pump_offcore().unwrap_or(false) {
        count += 1;
    }
    count
}

/// Runs `ticks` fabric ticks, pumping the off-core queue after each.
pub fn tick_and_pump(fabric: &mut MemoryFabric, ticks: usize) {
    for _ in 0..ticks {
        fabric.tick();
        pump_all(fabric);
    }
}

/// Reads an object to completion, driving at most one promotion round.
/// Panics when the object needs more than one promotion (which would
/// indicate a lost shadow).
#[must_use]
pub fn read_to_end(fabric: &mut MemoryFabric, object: u64) -> Bytes {
    match fabric.get(object).expect("get serves") {
        GetOutcome::Ready(bytes) => bytes,
        GetOutcome::Pending { .. } => {
            pump_all(fabric);
            match fabric.get(object).expect("get after promote") {
                GetOutcome::Ready(bytes) => bytes,
                GetOutcome::Pending { .. } => panic!("promotion did not resolve"),
            }
        }
    }
}

/// Inserts a medium object with explicit class and mutability.
pub fn insert_medium(
    fabric: &mut MemoryFabric,
    bytes: Bytes,
    class: BehaviorClass,
    mutability: Mutability,
) -> u64 {
    fabric
        .insert(
            bytes,
            kivi_memory::MaterializationIntent::cold(),
            class,
            mutability,
        )
        .expect("insert fits")
}
