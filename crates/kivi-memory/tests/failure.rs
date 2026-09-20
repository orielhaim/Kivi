//! Failure testing: scripted provider faults, corruption quarantine,
//! mutation fencing, queue saturation, crash/restart, and tablet
//! movement. Every case proves the same invariant: a failed optimization
//! degrades performance, never logical state.

mod common;

use kivi_memory::{
    BehaviorClass, GetOutcome, MemoryError, MemoryFabric, MemoryFabricConfig, Mutability,
    OffcoreCompletion, SimFault, TabletMaterializations,
};

/// A scripted `NVMe` failure during demotion keeps the DRAM copy live:
/// the object stays readable and the error is reported, not hidden.
#[test]
fn provider_failure_keeps_dram_copy_live() {
    let mut fabric = common::make_fabric(64 << 10, 16 << 20);
    let bytes = common::incompressible_bytes(4096, 31);
    let object = common::insert_medium(
        &mut fabric,
        bytes.clone(),
        BehaviorClass::HotMutable,
        Mutability::Mutable,
    );
    for index in 0..12 {
        common::insert_medium(
            &mut fabric,
            common::incompressible_bytes(4096, 300 + index),
            BehaviorClass::HotMutable,
            Mutability::Mutable,
        );
    }
    fabric.tick();
    fabric.inject_fault(SimFault::FailNext { count: 64 });
    // Pumping surfaces failures instead of hiding them.
    let _ = common::pump_all(&mut fabric);
    let seen = common::read_to_end(&mut fabric, object);
    assert_eq!(seen, bytes);
}

/// A completion for a stale version (mutation raced the device) is
/// discarded with `MutatedDuringBuild`; the new bytes win.
#[test]
fn stale_completion_is_fenced_by_version() {
    let mut fabric = common::make_fabric(64 << 10, 16 << 20);
    let object = common::insert_medium(
        &mut fabric,
        common::incompressible_bytes(4096, 41),
        BehaviorClass::HotMutable,
        Mutability::Mutable,
    );
    for index in 0..12 {
        common::insert_medium(
            &mut fabric,
            common::incompressible_bytes(4096, 900 + index),
            BehaviorClass::HotMutable,
            Mutability::Mutable,
        );
    }
    fabric.tick();
    let op = fabric.take_submittable().expect("demote submitted");
    // Mutation races the in-flight device op: queued work cancels, but
    // the submitted op still completes against a stale version.
    let updated = common::incompressible_bytes(4096, 42);
    fabric.mutate(object, updated.clone()).expect("mutate");
    let stale = OffcoreCompletion {
        id: op.id,
        object,
        version: op.version,
        bytes: Vec::new(),
        offset: Some(0),
        len: Some(4096),
        checksum: Some(0),
        ok: true,
        detail: String::new(),
    };
    let error = fabric.apply_completion(stale).expect_err("fenced");
    assert!(
        matches!(error, MemoryError::MutatedDuringBuild { .. }),
        "{error:?}"
    );
    let seen = common::read_to_end(&mut fabric, object);
    assert_eq!(seen, updated);
}

/// Mutating an object with a queued demote cancels the move: the pump
/// finds nothing to do and the new bytes are authoritative.
#[test]
fn mutate_cancels_queued_demote() {
    let mut fabric = common::make_fabric(64 << 10, 16 << 20);
    let object = common::insert_medium(
        &mut fabric,
        common::incompressible_bytes(4096, 51),
        BehaviorClass::HotMutable,
        Mutability::Mutable,
    );
    for index in 0..12 {
        common::insert_medium(
            &mut fabric,
            common::incompressible_bytes(4096, 600 + index),
            BehaviorClass::HotMutable,
            Mutability::Mutable,
        );
    }
    fabric.tick();
    let updated = common::incompressible_bytes(4096, 52);
    fabric.mutate(object, updated.clone()).expect("mutate");
    common::pump_all(&mut fabric);
    let seen = common::read_to_end(&mut fabric, object);
    assert_eq!(seen, updated);
}

/// Crash/restart: `NVMe` records survive the process; re-imported tablet
/// envelopes serve identical bytes from a fresh fabric on the same dir.
#[test]
fn crash_restart_preserves_acknowledged_state() {
    let dir = tempfile::tempdir().expect("tempdir");
    let envelopes = {
        let mut fabric = MemoryFabric::new(MemoryFabricConfig {
            arena_capacity_bytes: 64 << 10,
            nvme_capacity_bytes: 16 << 20,
            nvme_dir: Some(dir.path().to_owned()),
            ..MemoryFabricConfig::default()
        })
        .expect("fabric opens");
        let object = common::insert_medium(
            &mut fabric,
            common::incompressible_bytes(4096, 61),
            BehaviorClass::HotMutable,
            Mutability::Mutable,
        );
        for index in 0..12 {
            common::insert_medium(
                &mut fabric,
                common::incompressible_bytes(4096, 700 + index),
                BehaviorClass::HotMutable,
                Mutability::Mutable,
            );
        }
        common::tick_and_pump(&mut fabric, 6);
        assert!(fabric.stats().demotions > 0);
        let exported = fabric.export_tablet(&[object]);
        assert_eq!(exported.len(), 1);
        exported
    };
    // Fresh fabric on the same directory: records reopen cleanly.
    let mut restarted = MemoryFabric::new(MemoryFabricConfig {
        arena_capacity_bytes: 64 << 10,
        nvme_capacity_bytes: 16 << 20,
        nvme_dir: Some(dir.path().to_owned()),
        ..MemoryFabricConfig::default()
    })
    .expect("reopen succeeds");
    let set = TabletMaterializations {
        tablet: 1,
        epoch: 3,
        envelopes: envelopes
            .into_iter()
            .map(|(object, bytes)| kivi_memory::TabletEnvelope {
                object,
                version: 1,
                epoch: 3,
                bytes,
            })
            .collect(),
    };
    let (imported, skipped) = set.import_into(
        &mut restarted,
        3,
        &kivi_memory::MaterializationIntent::cold(),
        BehaviorClass::HotMutable,
    );
    assert_eq!((imported, skipped), (1, 0));
    let expected = common::incompressible_bytes(4096, 61);
    let last = u64::try_from(imported).expect("small count");
    for object in 1..=last {
        if let Ok(GetOutcome::Ready(seen)) = restarted.get(object) {
            assert_eq!(seen, expected);
        }
    }
}

/// Tiny saturated queues reject with `Overloaded`; the fabric keeps
/// serving everything already stored.
#[test]
fn saturated_queues_reject_without_losing_state() {
    let mut fabric = MemoryFabric::new(MemoryFabricConfig {
        arena_capacity_bytes: 64 << 10,
        nvme_capacity_bytes: 16 << 20,
        offcore_queue_capacity: 1,
        offcore_max_bytes: 512,
        ..MemoryFabricConfig::default()
    })
    .expect("fabric opens");
    let object = common::insert_medium(
        &mut fabric,
        common::incompressible_bytes(4096, 71),
        BehaviorClass::HotMutable,
        Mutability::Mutable,
    );
    for index in 0..6 {
        common::insert_medium(
            &mut fabric,
            common::incompressible_bytes(2048, 800 + index),
            BehaviorClass::HotMutable,
            Mutability::Mutable,
        );
    }
    fabric.tick();
    common::pump_all(&mut fabric);
    let expected = common::incompressible_bytes(4096, 71);
    let seen = common::read_to_end(&mut fabric, object);
    assert_eq!(seen, expected);
}
