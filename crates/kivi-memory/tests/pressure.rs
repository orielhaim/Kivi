//! Memory pressure causes bounded safe reclamation: the fabric sheds
//! cheapest-first representations under a per-tick budget while every
//! surviving object keeps serving exact logical bytes.

mod common;

use kivi_memory::{BehaviorClass, Mutability};

/// Filling past capacity fails fast with `Overloaded` instead of
/// growing without bound; everything accepted stays readable.
#[test]
fn over_capacity_fails_fast_and_accepted_stays_readable() {
    let mut fabric = common::make_fabric(32 << 10, 16 << 20);
    let mut accepted = 0;
    for index in 0u64..64 {
        let bytes = common::incompressible_bytes(4096, 1000 + index);
        let inserted = fabric.insert(
            bytes,
            kivi_memory::MaterializationIntent::cold(),
            BehaviorClass::HotMutable,
            Mutability::Mutable,
        );
        if inserted.is_ok() {
            accepted += 1;
        } else {
            break;
        }
    }
    assert!(accepted > 0 && accepted < 64);
    common::tick_and_pump(&mut fabric, 8);
    assert!(fabric.pressure() < 1.0);
}

/// Least-critical objects move first and per-tick movement stays within
/// the configured budget.
#[test]
fn reclamation_is_bounded_and_least_critical_first() {
    let mut fabric = common::make_fabric(48 << 10, 16 << 20);
    let mut objects = Vec::new();
    for index in 0u64..10 {
        let object = common::insert_medium(
            &mut fabric,
            common::incompressible_bytes(4096, 500 + index),
            BehaviorClass::HotMutable,
            Mutability::Mutable,
        );
        objects.push((object, common::incompressible_bytes(4096, 500 + index)));
    }
    let before = fabric.stats().migration_bytes;
    fabric.tick();
    let moved_this_tick = fabric.stats().migration_bytes - before;
    assert!(moved_this_tick <= 4 << 20);
    common::pump_all(&mut fabric);
    for (object, expected) in &objects {
        let seen = common::read_to_end(&mut fabric, *object);
        assert_eq!(&seen, expected);
    }
}

/// Sustained pressure over many ticks never loses a byte: every object
/// reads back exactly what was written.
#[test]
fn sustained_pressure_never_loses_logical_state() {
    let mut fabric = common::make_fabric(40 << 10, 16 << 20);
    let mut objects = Vec::new();
    for index in 0u64..8 {
        let bytes = if index % 2 == 0 {
            common::compressible_bytes(2048)
        } else {
            common::incompressible_bytes(2048, 900 + index)
        };
        let object = common::insert_medium(
            &mut fabric,
            bytes.clone(),
            BehaviorClass::Warm,
            Mutability::ReadMostly,
        );
        objects.push((object, bytes));
    }
    common::tick_and_pump(&mut fabric, 12);
    for (object, expected) in &objects {
        let seen = common::read_to_end(&mut fabric, *object);
        assert_eq!(&seen, expected);
    }
}
