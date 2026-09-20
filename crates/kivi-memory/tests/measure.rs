//! Measurement harness: footprint, compression, movement, hit rates,
//! and p50/p99/p99.9 read latency in steady state vs active
//! compression/demotion/promotion. Prints `measure:` lines (visible with
//! `-- --nocapture`); asserts the invariants behind the numbers.

mod common;

use std::time::Instant;

use kivi_memory::{BehaviorClass, GetOutcome, Mutability};

/// Percentiles of a sorted nanosecond sample vector (integer math only).
fn percentiles(sorted_ns: &[u64]) -> (u64, u64, u64) {
    if sorted_ns.is_empty() {
        return (0, 0, 0);
    }
    let last = sorted_ns.len() - 1;
    let at = |num: usize, den: usize| sorted_ns[(sorted_ns.len() * num / den).min(last)];
    (at(1, 2), at(99, 100), at(999, 1000))
}

fn time_gets(fabric: &mut kivi_memory::MemoryFabric, objects: &[u64], rounds: usize) -> Vec<u64> {
    let mut samples = Vec::with_capacity(objects.len() * rounds);
    for _ in 0..rounds {
        for object in objects {
            let start = Instant::now();
            let outcome = fabric.get(*object).expect("get serves");
            if matches!(outcome, GetOutcome::Ready(_)) {
                samples.push(u64::try_from(start.elapsed().as_nanos()).unwrap_or(u64::MAX));
            }
        }
    }
    samples.sort_unstable();
    samples
}

/// Steady state: all-DRAM resident set, no background work pending.
#[test]
fn measure_steady_state() {
    let mut fabric = common::make_fabric(8 << 20, 64 << 20);
    let mut objects = Vec::new();
    for index in 0u64..200 {
        let object = common::insert_medium(
            &mut fabric,
            common::incompressible_bytes(1024, 10_000 + index),
            BehaviorClass::HotMutable,
            Mutability::Mutable,
        );
        objects.push(object);
    }
    // Quiet ticks refresh statistics without moving anything
    // (low pressure, hot intent, cooldown-clean planner stays put).
    // Statistics refresh every 8th tick (amortized cadence).
    for _ in 0..8 {
        fabric.tick();
    }
    let mut samples = time_gets(&mut fabric, &objects, 5);
    samples.sort_unstable();
    let (p50, p99, p999) = percentiles(&samples);
    let stats = fabric.stats();
    eprintln!(
        "measure: phase=steady objects={} dram_bytes={} hits={} get_p50_ns={p50} get_p99_ns={p99} get_p999_ns={p999}",
        stats.objects, stats.dram_bytes, stats.hits,
    );
    assert_eq!(stats.objects, 200);
    assert_eq!(stats.promotion_misses, 0);
}

/// Active movement: pressure drives compression, demotion, and
/// promotion while reads continue; logical state stays exact.
#[test]
fn measure_active_movement() {
    let mut fabric = common::make_fabric(128 << 10, 64 << 20);
    let mut objects = Vec::new();
    for index in 0u64..60 {
        let bytes = if index % 2 == 0 {
            common::compressible_bytes(2048)
        } else {
            common::incompressible_bytes(2048, 20_000 + index)
        };
        let object = common::insert_medium(
            &mut fabric,
            bytes,
            BehaviorClass::Warm,
            Mutability::ReadMostly,
        );
        objects.push(object);
    }
    // Drive eight active ticks (a full statistics cadence), timing
    // reads interleaved with movement.
    let mut active_samples = Vec::new();
    for _ in 0..8 {
        fabric.tick();
        for object in &objects {
            let start = Instant::now();
            match fabric.get(*object).expect("get") {
                GetOutcome::Ready(_) => {
                    active_samples
                        .push(u64::try_from(start.elapsed().as_nanos()).unwrap_or(u64::MAX));
                }
                GetOutcome::Pending { .. } => {}
            }
        }
        common::pump_all(&mut fabric);
    }
    active_samples.sort_unstable();
    let (p50, p99, p999) = percentiles(&active_samples);
    let stats = fabric.stats();
    eprintln!(
        "measure: phase=active objects={} dram_bytes={} compressed_bytes={} ratio_bps={} nvme_bytes={} demotions={} promotions={} migration_bytes={} hits={} misses={} read_p50_ns={p50} read_p99_ns={p99} read_p999_ns={p999}",
        stats.objects,
        stats.dram_bytes,
        stats.compressed_bytes,
        stats.compression_ratio_bps,
        stats.nvme_bytes,
        stats.demotions,
        stats.promotions,
        stats.migration_bytes,
        stats.hits,
        stats.promotion_misses,
    );
    assert!(stats.migration_bytes > 0);
    assert!(stats.demotions > 0 || stats.compressed_bytes > 0);
}
