//! Determinism: identical inputs replay identical placements,
//! transitions, and statistics with no hardware clock or RNG involved.

mod common;

use kivi_memory::{BehaviorClass, Mutability};

/// Two fabrics fed the same bytes and ticks converge to identical
/// statistics and serve identical reads.
#[test]
fn identical_inputs_replay_identically() {
    let build = || {
        let mut fabric = common::make_fabric(48 << 10, 16 << 20);
        let mut objects = Vec::new();
        for index in 0u64..12 {
            let bytes = if index % 3 == 0 {
                common::compressible_bytes(2048)
            } else {
                common::incompressible_bytes(2048, 100 + index)
            };
            let object = common::insert_medium(
                &mut fabric,
                bytes.clone(),
                if index % 2 == 0 {
                    BehaviorClass::Warm
                } else {
                    BehaviorClass::HotMutable
                },
                Mutability::ReadMostly,
            );
            objects.push((object, bytes));
        }
        common::tick_and_pump(&mut fabric, 10);
        (fabric, objects)
    };
    let (mut first, first_objects) = build();
    let (mut second, second_objects) = build();
    assert_eq!(first.stats(), second.stats());
    for ((left, left_bytes), (right, right_bytes)) in
        first_objects.iter().zip(second_objects.iter())
    {
        assert_eq!(left_bytes, right_bytes);
        let seen_left = common::read_to_end(&mut first, *left);
        let seen_right = common::read_to_end(&mut second, *right);
        assert_eq!(seen_left, seen_right);
        assert_eq!(&seen_left, left_bytes);
    }
}

/// Mutation order is part of the deterministic input: replaying the
/// same mutation at the same tick yields the same state.
#[test]
fn mutation_order_replays_deterministically() {
    let run = || {
        let mut fabric = common::make_fabric(64 << 10, 16 << 20);
        let object = common::insert_medium(
            &mut fabric,
            common::incompressible_bytes(1024, 5),
            BehaviorClass::HotMutable,
            Mutability::Mutable,
        );
        common::tick_and_pump(&mut fabric, 2);
        fabric
            .mutate(object, common::incompressible_bytes(1024, 6))
            .expect("mutate");
        common::tick_and_pump(&mut fabric, 4);
        let seen = common::read_to_end(&mut fabric, object);
        (seen, fabric.stats())
    };
    let (first_bytes, first_stats) = run();
    let (second_bytes, second_stats) = run();
    assert_eq!(first_bytes, second_bytes);
    assert_eq!(first_stats, second_stats);
}
