//! Litmus tests: the observable outcomes of racing operations under every
//! interleaving, enumerated single-threaded. Where the model-based test
//! proves agreement with a reference and the unit tests prove single-shot
//! behavior, these prove the concurrency contract - commutativity under
//! reordering, idempotent retry under exhaustion, and offset monotonicity
//! across trims - by running the orders, not by describing them.
//!
//! Fencing, transactions and releases are covered once, in `store.rs`'s unit
//! tests; repeating them here proved the same properties twice.

use bytes::Bytes;
use kivi_state::{
    Key, ObjectStore, OpError, Operation, OperationResult, PermitId, Prepared, outcome_for,
};
use kivi_types::WallTimestamp;

const NOW: WallTimestamp = WallTimestamp::from_micros(1_000_000);

fn key(name: &str) -> Key {
    Key::from(name)
}

fn execute(
    store: &mut ObjectStore,
    op: &Operation,
    now: WallTimestamp,
) -> Result<OperationResult, OpError> {
    match store.prepare(op, now)? {
        Prepared::Read(result) => Ok(result),
        Prepared::Write(mutation) => {
            let outcome = store.apply(&mutation, now).expect("apply must not fail");
            Ok(outcome_for(&mutation, &outcome, false))
        }
    }
}

fn permit(byte: u8) -> PermitId {
    PermitId::from_bytes([byte; 16])
}

/// Every application order of commutative adds lands on the same sum and
/// exposes no ordinal: the order-free promise, all six ways.
#[test]
fn commutative_every_order_converges() {
    let deltas = [3, -1, 4];
    let orders = [
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ];
    for order in orders {
        let mut store = ObjectStore::new();
        for index in order {
            assert_eq!(
                execute(
                    &mut store,
                    &Operation::CommutativeAdd {
                        key: key("cc"),
                        delta: deltas[index],
                    },
                    NOW,
                )
                .expect("add applies"),
                OperationResult::CommutativeApplied,
                "no ordinal may leak in order {order:?}",
            );
        }
        match store.get(&key("cc"), NOW).expect("present").value() {
            kivi_state::LogicalValue::CommutativeCounter(value) => assert_eq!(*value, 6),
            _ => panic!("must hold a commutative counter"),
        }
    }
}

/// Bounded adds commute inside owned rights: both client orders converge,
/// and both orders hit the bound identically.
#[test]
fn bounded_adds_converge_inside_rights() {
    use kivi_types::TabletId;
    let holder = TabletId::from_u64(7);
    for order in [[3, 4], [4, 3]] {
        let mut store = ObjectStore::new();
        execute(
            &mut store,
            &Operation::BoundedCounterCreate {
                key: key("budget"),
                capacity: 10,
                holder,
            },
            NOW,
        )
        .expect("create");
        for delta in order {
            execute(
                &mut store,
                &Operation::BoundedCounterAdd {
                    key: key("budget"),
                    delta,
                },
                NOW,
            )
            .expect("fits");
        }
        match execute(
            &mut store,
            &Operation::BoundedCounterGet { key: key("budget") },
            NOW,
        )
        .expect("get")
        {
            OperationResult::BoundedValue {
                value: Some(7),
                capacity: Some(10),
                ..
            } => {}
            other => panic!("converged value 7 in order {order:?}, got {other:?}"),
        }
        assert!(
            matches!(
                store.prepare(
                    &Operation::BoundedCounterAdd {
                        key: key("budget"),
                        delta: 4,
                    },
                    NOW,
                ),
                Err(OpError::BoundedExceeded),
            ),
            "bound binds in order {order:?}",
        );
    }
}

/// Exhaustion and identical retry agree in every ask order: anyone else is
/// rejected while full, and the same permit id retry succeeds without
/// acquiring twice.
#[test]
fn semaphore_exhaustion_and_idempotent_retry() {
    let mut store = ObjectStore::new();
    execute(
        &mut store,
        &Operation::SemaphoreCreate {
            key: key("sem"),
            capacity: 2,
        },
        NOW,
    )
    .expect("create");
    execute(
        &mut store,
        &Operation::SemaphoreAcquire {
            key: key("sem"),
            permit: permit(1),
            owner: 11,
            qty: 2,
        },
        NOW,
    )
    .expect("full acquire");
    // Exhausted for anyone else, in any order of asking.
    for other in [2, 3] {
        assert!(matches!(
            store.prepare(
                &Operation::SemaphoreAcquire {
                    key: key("sem"),
                    permit: permit(other),
                    owner: 12,
                    qty: 1,
                },
                NOW,
            ),
            Err(OpError::SemaphoreExhausted),
        ));
    }
    // The identical retry succeeds without acquiring twice.
    assert_eq!(
        execute(
            &mut store,
            &Operation::SemaphoreAcquire {
                key: key("sem"),
                permit: permit(1),
                owner: 11,
                qty: 2,
            },
            NOW,
        )
        .expect("retry"),
        OperationResult::SemaphoreAcquired,
    );
    assert_eq!(
        execute(
            &mut store,
            &Operation::SemaphoreInspect { key: key("sem") },
            NOW,
        )
        .expect("inspect"),
        OperationResult::SemaphoreLoad {
            outstanding: 2,
            capacity: 2,
        },
    );
}

/// Releases never mint capacity in any order: unknown ids report false,
/// re-release reports false, and re-acquire fills exactly to the maximum.
#[test]
fn semaphore_releases_never_mint() {
    let mut store = ObjectStore::new();
    execute(
        &mut store,
        &Operation::SemaphoreCreate {
            key: key("sem"),
            capacity: 2,
        },
        NOW,
    )
    .expect("create");
    execute(
        &mut store,
        &Operation::SemaphoreAcquire {
            key: key("sem"),
            permit: permit(1),
            owner: 11,
            qty: 2,
        },
        NOW,
    )
    .expect("full acquire");
    // Unknown releases report false and mint nothing.
    assert_eq!(
        execute(
            &mut store,
            &Operation::SemaphoreRelease {
                key: key("sem"),
                permit: permit(9),
            },
            NOW,
        )
        .expect("unknown release"),
        OperationResult::SemaphoreReleased { released: false },
    );
    // Release, re-release, re-acquire: capacity never exceeds the maximum.
    assert_eq!(
        execute(
            &mut store,
            &Operation::SemaphoreRelease {
                key: key("sem"),
                permit: permit(1),
            },
            NOW,
        )
        .expect("release"),
        OperationResult::SemaphoreReleased { released: true },
    );
    assert_eq!(
        execute(
            &mut store,
            &Operation::SemaphoreRelease {
                key: key("sem"),
                permit: permit(1),
            },
            NOW,
        )
        .expect("re-release"),
        OperationResult::SemaphoreReleased { released: false },
    );
    execute(
        &mut store,
        &Operation::SemaphoreAcquire {
            key: key("sem"),
            permit: permit(2),
            owner: 12,
            qty: 2,
        },
        NOW,
    )
    .expect("re-acquire fills exactly");
    assert!(matches!(
        store.prepare(
            &Operation::SemaphoreAcquire {
                key: key("sem"),
                permit: permit(3),
                owner: 13,
                qty: 1,
            },
            NOW,
        ),
        Err(OpError::SemaphoreExhausted),
    ));
}

/// Trimmed stream offsets are gone, not reusable: the next append continues
/// the shard clock, never rewinds it.
#[test]
fn stream_trim_never_rewinds_offsets() {
    let mut store = ObjectStore::new();
    execute(
        &mut store,
        &Operation::StreamCreate {
            key: key("shard"),
            stream: [7; 16],
            shard: 0,
        },
        NOW,
    )
    .expect("create");
    for (index, payload) in [b"e0".as_slice(), b"e1".as_slice(), b"e2".as_slice()]
        .iter()
        .enumerate()
    {
        assert_eq!(
            execute(
                &mut store,
                &Operation::StreamAppend {
                    key: key("shard"),
                    partition: b"p".to_vec(),
                    payload: Bytes::copy_from_slice(payload),
                },
                NOW,
            )
            .expect("append"),
            OperationResult::StreamAppended {
                offset: index as u64,
            },
        );
    }
    assert_eq!(
        execute(
            &mut store,
            &Operation::StreamTrim {
                key: key("shard"),
                through_offset: 1,
            },
            NOW,
        )
        .expect("trim"),
        OperationResult::StreamTrimmed { removed: 2 },
    );
    match execute(
        &mut store,
        &Operation::StreamRead {
            key: key("shard"),
            from_offset: 0,
            max_entries: 16,
        },
        NOW,
    )
    .expect("read")
    {
        OperationResult::StreamEntries {
            entries,
            next_offset,
        } => {
            assert_eq!(next_offset, 3);
            assert_eq!(entries.len(), 1);
            assert_eq!(entries[0].offset, 2);
        }
        other => panic!("entries, got {other:?}"),
    }
    assert_eq!(
        execute(
            &mut store,
            &Operation::StreamAppend {
                key: key("shard"),
                partition: b"p".to_vec(),
                payload: Bytes::from_static(b"e3"),
            },
            NOW,
        )
        .expect("append after trim"),
        OperationResult::StreamAppended { offset: 3 },
    );
}
