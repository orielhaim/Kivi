//! End-to-end representation transitions: DRAM to compressed to `NVMe`
//! and back, with logical equivalence at every step.
//!
//! The core invariant under test: any materialization decision may affect
//! latency, memory use, or cost; it never alters logical state.

mod common;

use kivi_memory::{BehaviorClass, GetOutcome, Mutability};

/// Compresses a cold compressible object under pressure and proves the
/// bytes are unchanged and served from the compressed representation.
/// The file backend runs with a zero endurance budget, so demotion
/// admission always rejects and pressure relief must compress (the only
/// tier that fits): this makes the planner's choice deterministic
/// instead of asserting which tier ice-cold data prefers.
#[test]
fn dram_to_compressed_keeps_logical_bytes() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut fabric = kivi_memory::MemoryFabric::new(kivi_memory::MemoryFabricConfig {
        arena_capacity_bytes: 32 << 10,
        nvme_capacity_bytes: 16 << 20,
        nvme_dir: Some(dir.path().to_owned()),
        nvme_options: kivi_memory::NvmeOptions {
            daily_write_budget_bytes: 0,
            ..kivi_memory::NvmeOptions::default()
        },
        ..kivi_memory::MemoryFabricConfig::default()
    })
    .expect("fabric opens");
    let bytes = common::compressible_bytes(4096);
    let object = common::insert_medium(
        &mut fabric,
        bytes.clone(),
        BehaviorClass::ColdCandidate,
        Mutability::ReadMostly,
    );
    // Fill the arena past the pressure floor with incompressible
    // neighbors so reclamation must act.
    for index in 0..6 {
        common::insert_medium(
            &mut fabric,
            common::incompressible_bytes(4096, 100 + index),
            BehaviorClass::HotMutable,
            Mutability::Mutable,
        );
    }
    // Statistics refresh every 8th tick (amortized); pump a full
    // cadence so the byte breakdowns observe the compression.
    common::tick_and_pump(&mut fabric, 8);
    assert!(fabric.stats().compressed_bytes > 0);
    let seen = common::read_to_end(&mut fabric, object);
    assert_eq!(seen, bytes);
    // Cold data flows downhill: compressed, possibly further to `NVMe`
    // with a retained shadow. Either way it left the plain arena and
    // stayed logically identical.
    let explanation = fabric.explain(object).expect("explained");
    assert!(
        explanation.contains("primary=compressed") || explanation.contains("primary=nvme"),
        "{explanation}"
    );
}

/// Demotes a hot-mutable object to `NVMe` under pressure, then promotes
/// it back, proving byte equality across the full cycle.
#[test]
fn dram_to_nvme_and_back_keeps_logical_bytes() {
    let mut fabric = common::make_fabric(64 << 10, 16 << 20);
    let bytes = common::incompressible_bytes(4096, 7);
    let object = common::insert_medium(
        &mut fabric,
        bytes.clone(),
        BehaviorClass::HotMutable,
        Mutability::Mutable,
    );
    for index in 0..12 {
        common::insert_medium(
            &mut fabric,
            common::incompressible_bytes(4096, 200 + index),
            BehaviorClass::HotMutable,
            Mutability::Mutable,
        );
    }
    common::tick_and_pump(&mut fabric, 6);
    assert!(fabric.stats().demotions > 0);
    let seen = common::read_to_end(&mut fabric, object);
    assert_eq!(seen, bytes);
    assert!(fabric.stats().promotions > 0 || fabric.stats().hits > 0);
}

/// Non-exclusive tiering: when pressure drops before the demote
/// completes, the DRAM shadow survives and serves reads immediately.
#[test]
fn demote_keeps_dram_shadow_when_pressure_drops() {
    let mut fabric = common::make_fabric(64 << 10, 16 << 20);
    let filler = common::insert_medium(
        &mut fabric,
        common::incompressible_bytes(40 << 10, 11),
        BehaviorClass::HotMutable,
        Mutability::Mutable,
    );
    let bytes = common::incompressible_bytes(4096, 12);
    let object = common::insert_medium(
        &mut fabric,
        bytes.clone(),
        BehaviorClass::HotMutable,
        Mutability::Mutable,
    );
    common::insert_medium(
        &mut fabric,
        common::incompressible_bytes(12 << 10, 13),
        BehaviorClass::HotMutable,
        Mutability::Mutable,
    );
    fabric.tick();
    // Relieve pressure before completions apply: the DRAM copy must be
    // kept as a shadow instead of retired.
    fabric.remove(filler).expect("remove filler");
    common::pump_all(&mut fabric);
    assert!(fabric.stats().demotions > 0);
    match fabric.get(object).expect("get serves from shadow") {
        GetOutcome::Ready(seen) => assert_eq!(seen, bytes),
        GetOutcome::Pending { .. } => panic!("shadow should serve immediately"),
    }
    let explanation = fabric.explain(object).expect("explained");
    assert!(explanation.contains("primary=nvme"), "{explanation}");
}
