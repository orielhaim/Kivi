//! Litmus tests: the observable outcomes of racing operations under every
//! interleaving, enumerated single-threaded. Where the model-based test
//! proves agreement with a reference and the unit tests prove single-shot
//! behavior, these prove the concurrency contract — commutativity,
//! idempotent retry under exhaustion, stale-token powerlessness, and
//! all-or-nothing resolution — by running the orders, not by describing
//! them.

use bytes::Bytes;
use kivi_state::{
    FencingToken, Key, LogicalValue, ObjectStore, ObjectVersion, OpError, Operation,
    OperationResult, PermitId, Prepared, outcome_for,
};
use kivi_types::{TabletId, UnixMicros};

const NOW: UnixMicros = UnixMicros::from_micros(1_000_000);

fn key(name: &str) -> Key {
    Key::from(name)
}

fn execute(
    store: &mut ObjectStore,
    op: &Operation,
    now: UnixMicros,
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
            LogicalValue::CommutativeCounter(value) => assert_eq!(*value, 6),
            _ => panic!("must hold a commutative counter"),
        }
    }
}

/// Bounded adds commute inside owned rights: both client orders converge,
/// and both orders hit the bound identically.
#[test]
fn bounded_adds_converge_inside_rights() {
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

/// Stale fencing tokens are powerless against a live grant: wrong-token
/// renew and release fail, the holder is undisturbed, and the correct
/// token renews in place.
#[test]
fn lease_stale_tokens_powerless_against_live_grant() {
    let mut store = ObjectStore::new();
    let fencing = match execute(
        &mut store,
        &Operation::LeaseAcquire {
            key: key("leader"),
            owner: 1,
            ttl_micros: 1000,
        },
        NOW,
    )
    .expect("acquire")
    {
        OperationResult::LeaseAcquired { fencing, .. } => fencing,
        other => panic!("grant, got {other:?}"),
    };
    // A live holder blocks other owners.
    assert!(matches!(
        store.prepare(
            &Operation::LeaseAcquire {
                key: key("leader"),
                owner: 2,
                ttl_micros: 1000,
            },
            NOW,
        ),
        Err(OpError::LeaseConflict),
    ));
    let stale = FencingToken::from_u64(fencing.as_u64() + 100);
    assert!(matches!(
        store.prepare(
            &Operation::LeaseRenew {
                key: key("leader"),
                owner: 1,
                fencing: stale,
                ttl_micros: 1000,
            },
            NOW,
        ),
        Err(OpError::StaleFencing),
    ));
    assert!(matches!(
        store.prepare(
            &Operation::LeaseRelease {
                key: key("leader"),
                owner: 1,
                fencing: stale,
            },
            NOW,
        ),
        Err(OpError::StaleFencing),
    ));
    // The correct token renews in place (same token, extended expiry).
    match execute(
        &mut store,
        &Operation::LeaseRenew {
            key: key("leader"),
            owner: 1,
            fencing,
            ttl_micros: 1000,
        },
        NOW,
    )
    .expect("renew")
    {
        OperationResult::LeaseRenewed {
            fencing: renewed, ..
        } => assert_eq!(renewed, fencing),
        other => panic!("renewed, got {other:?}"),
    }
    match execute(
        &mut store,
        &Operation::LeaseInspect { key: key("leader") },
        NOW,
    )
    .expect("inspect")
    {
        OperationResult::LeaseInfo { holder, .. } => {
            assert_eq!(holder.map(|h| h.owner), Some(1));
        }
        other => panic!("inspect, got {other:?}"),
    }
}

/// Succession fences strictly: release frees, the next grant carries a
/// greater token, and the old token can neither renew nor free the new
/// lease.
#[test]
fn lease_succession_fences_strictly() {
    let mut store = ObjectStore::new();
    let fencing = match execute(
        &mut store,
        &Operation::LeaseAcquire {
            key: key("leader"),
            owner: 1,
            ttl_micros: 1000,
        },
        NOW,
    )
    .expect("acquire")
    {
        OperationResult::LeaseAcquired { fencing, .. } => fencing,
        other => panic!("grant, got {other:?}"),
    };
    assert_eq!(
        execute(
            &mut store,
            &Operation::LeaseRelease {
                key: key("leader"),
                owner: 1,
                fencing,
            },
            NOW,
        )
        .expect("release"),
        OperationResult::LeaseReleased { released: true },
    );
    // The successor fences the predecessor: strictly greater token.
    let fencing2 = match execute(
        &mut store,
        &Operation::LeaseAcquire {
            key: key("leader"),
            owner: 2,
            ttl_micros: 1000,
        },
        NOW,
    )
    .expect("re-acquire")
    {
        OperationResult::LeaseAcquired { fencing, .. } => fencing,
        other => panic!("grant, got {other:?}"),
    };
    assert!(fencing2.as_u64() > fencing.as_u64());
    // The old token can neither renew nor free the new lease.
    assert!(matches!(
        store.prepare(
            &Operation::LeaseRenew {
                key: key("leader"),
                owner: 1,
                fencing,
                ttl_micros: 1000,
            },
            NOW,
        ),
        Err(OpError::StaleFencing),
    ));
    assert!(matches!(
        store.prepare(
            &Operation::LeaseRelease {
                key: key("leader"),
                owner: 1,
                fencing,
            },
            NOW,
        ),
        Err(OpError::StaleFencing),
    ));
    match execute(
        &mut store,
        &Operation::LeaseInspect { key: key("leader") },
        NOW,
    )
    .expect("inspect")
    {
        OperationResult::LeaseInfo { holder, .. } => {
            assert_eq!(holder.map(|h| h.owner), Some(2));
        }
        other => panic!("inspect, got {other:?}"),
    }
}

/// Expiry frees the lease for a successor; the expired holder's token stays
/// dead — fencing retires holders, not the clock alone.
#[test]
fn lease_expiry_hands_over_without_revival() {
    let mut store = ObjectStore::new();
    let fencing = match execute(
        &mut store,
        &Operation::LeaseAcquire {
            key: key("leader"),
            owner: 1,
            ttl_micros: 100,
        },
        NOW,
    )
    .expect("acquire")
    {
        OperationResult::LeaseAcquired { fencing, .. } => fencing,
        other => panic!("grant, got {other:?}"),
    };
    let later = UnixMicros::from_micros(NOW.as_micros() + 101);
    let fencing2 = match execute(
        &mut store,
        &Operation::LeaseAcquire {
            key: key("leader"),
            owner: 2,
            ttl_micros: 100,
        },
        later,
    )
    .expect("successor acquires expired")
    {
        OperationResult::LeaseAcquired { fencing, .. } => fencing,
        other => panic!("grant, got {other:?}"),
    };
    assert!(fencing2.as_u64() > fencing.as_u64());
    // The expired token cannot revive the lease, before or after handover.
    assert!(matches!(
        store.prepare(
            &Operation::LeaseRenew {
                key: key("leader"),
                owner: 1,
                fencing,
                ttl_micros: 100,
            },
            later,
        ),
        Err(OpError::StaleFencing),
    ));
}

/// Abort discards every intent: neither key becomes visible, and the keys
/// are immediately reusable by a clean local commit.
#[test]
fn txn_abort_discards_all_intents() {
    use kivi_state::{TxnExpect, TxnId, TxnWrite, TxnWriteKind};
    let mut store = ObjectStore::new();
    let txn = TxnId::derive(1, 2, 0);
    let digest = [0xD1; 32];
    let coordinator = TabletId::from_u64(1);
    for (name, value) in [("a", b"1".as_slice()), ("b", b"2".as_slice())] {
        let prepared = store
            .prepare(
                &Operation::TxnPrepare {
                    txn,
                    coordinator,
                    write: TxnWrite {
                        key: key(name),
                        kind: TxnWriteKind::Put(Bytes::copy_from_slice(value)),
                        expect: TxnExpect::Absent,
                    },
                    digest,
                },
                NOW,
            )
            .expect("prepare");
        let Prepared::Write(mutation) = prepared else {
            panic!("must prepare");
        };
        store.apply(&mutation, NOW).expect("reserve");
    }
    for name in ["a", "b"] {
        execute(
            &mut store,
            &Operation::TxnFinalize {
                txn,
                key: key(name),
                commit: false,
                digest,
            },
            NOW,
        )
        .expect("abort finalizes");
        assert!(store.get(&key(name), NOW).is_none(), "{name} discarded");
    }
    // The keys are clean: a local commit reuses them atomically.
    let results = store
        .commit_local(
            &[
                TxnWrite {
                    key: key("a"),
                    kind: TxnWriteKind::Put(Bytes::from_static(b"1")),
                    expect: TxnExpect::Absent,
                },
                TxnWrite {
                    key: key("b"),
                    kind: TxnWriteKind::Put(Bytes::from_static(b"2")),
                    expect: TxnExpect::Absent,
                },
            ],
            NOW,
        )
        .expect("clean local commit");
    assert_eq!(results.len(), 2);
    assert!(store.get(&key("a"), NOW).is_some());
    assert!(store.get(&key("b"), NOW).is_some());
}

/// Commit finalizes every intent: both keys appear together with the
/// prepared values.
#[test]
fn txn_commit_applies_all_intents() {
    use kivi_state::{TxnExpect, TxnId, TxnWrite, TxnWriteKind};
    let mut store = ObjectStore::new();
    let txn = TxnId::derive(3, 4, 0);
    let digest = [0xC0; 32];
    let coordinator = TabletId::from_u64(1);
    for (name, value) in [("a", b"1".as_slice()), ("b", b"2".as_slice())] {
        let prepared = store
            .prepare(
                &Operation::TxnPrepare {
                    txn,
                    coordinator,
                    write: TxnWrite {
                        key: key(name),
                        kind: TxnWriteKind::Put(Bytes::copy_from_slice(value)),
                        expect: TxnExpect::Absent,
                    },
                    digest,
                },
                NOW,
            )
            .expect("prepare");
        let Prepared::Write(mutation) = prepared else {
            panic!("must prepare");
        };
        store.apply(&mutation, NOW).expect("reserve");
        // Reserved keys hide from reads until finalized.
        assert!(store.get(&key(name), NOW).is_none());
    }
    for name in ["a", "b"] {
        execute(
            &mut store,
            &Operation::TxnFinalize {
                txn,
                key: key(name),
                commit: true,
                digest,
            },
            NOW,
        )
        .expect("commit finalizes");
    }
    for name in ["a", "b"] {
        let stored = store.get(&key(name), NOW).expect("committed");
        assert!(stored.version() > ObjectVersion::from_u64(0));
    }
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
