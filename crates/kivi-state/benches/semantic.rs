//! Phase-8 semantic operation costs at the state layer: commutative adds
//! against the strict-counter baseline, permitgated semaphore acquisition,
//! steady-state lease renewal, shard-local stream appends, the cheap
//! same-tablet commit path against the 2PC intent path, and write-set
//! digest binding. Repeatable locally with `cargo bench -p kivi-state`.

use divan::{Bencher, black_box, counter::ItemsCount};

use bytes::Bytes;
use kivi_state::{
    Key, ObjectStore, Operation, OperationResult, PermitId, Prepared, TxnExpect, TxnId, TxnWrite,
    TxnWriteKind, outcome_for, write_set_digest,
};
use kivi_types::{NamespaceId, TabletId, UnixMicros};

const NOW: UnixMicros = UnixMicros::from_micros(1_000_000);
const NS: NamespaceId = NamespaceId::from_u64(1);

fn main() {
    divan::main();
}

fn key(name: &str) -> Key {
    Key::from(name)
}

fn run(store: &mut ObjectStore, op: &Operation) -> OperationResult {
    match store.prepare(op, NOW).expect("prepare") {
        Prepared::Read(result) => result,
        Prepared::Write(mutation) => {
            let outcome = store.apply(&mutation, NOW).expect("apply");
            outcome_for(&mutation, &outcome, false)
        }
    }
}

#[divan::bench]
fn counter_add_baseline(bencher: Bencher) {
    let mut store = ObjectStore::new();
    let op = Operation::CounterAdd {
        key: key("strict"),
        delta: 1,
    };
    bencher
        .counter(ItemsCount::new(1u64))
        .bench_local(|| black_box(run(&mut store, &op)));
}

#[divan::bench]
fn commutative_add(bencher: Bencher) {
    let mut store = ObjectStore::new();
    let op = Operation::CommutativeAdd {
        key: key("free"),
        delta: 1,
    };
    bencher
        .counter(ItemsCount::new(1u64))
        .bench_local(|| black_box(run(&mut store, &op)));
}

#[divan::bench]
fn semaphore_permit_cycle(bencher: Bencher) {
    let mut store = ObjectStore::new();
    run(
        &mut store,
        &Operation::SemaphoreCreate {
            key: key("sem"),
            capacity: 8,
        },
    );
    // Acquire + release the same permit every iteration: steady state
    // under the live-permit cap, measuring the guarded path, not growth.
    let mut id = [3u8; 16];
    bencher.counter(ItemsCount::new(2u64)).bench_local(|| {
        id[0] = id[0].wrapping_add(1);
        let permit = PermitId::from_bytes(id);
        black_box(run(
            &mut store,
            &Operation::SemaphoreAcquire {
                key: key("sem"),
                permit,
                owner: 1,
                qty: 1,
            },
        ));
        black_box(run(
            &mut store,
            &Operation::SemaphoreRelease {
                key: key("sem"),
                permit,
            },
        ));
    });
}

#[divan::bench]
fn lease_renew_steady_state(bencher: Bencher) {
    let mut store = ObjectStore::new();
    let fencing = match run(
        &mut store,
        &Operation::LeaseAcquire {
            key: key("leader"),
            owner: 1,
            ttl_micros: u64::MAX / 2,
        },
    ) {
        OperationResult::LeaseAcquired { fencing, .. } => fencing,
        other => panic!("grant, got {other:?}"),
    };
    let op = Operation::LeaseRenew {
        key: key("leader"),
        owner: 1,
        fencing,
        ttl_micros: u64::MAX / 2,
    };
    bencher
        .counter(ItemsCount::new(1u64))
        .bench_local(|| black_box(run(&mut store, &op)));
}

#[divan::bench]
fn stream_append(bencher: Bencher) {
    let mut store = ObjectStore::new();
    run(
        &mut store,
        &Operation::StreamCreate {
            key: key("shard"),
            stream: [7; 16],
            shard: 0,
        },
    );
    let payload = Bytes::from_static(b"0123456789abcdef");
    let mut seq: u64 = 0;
    bencher.counter(ItemsCount::new(1u64)).bench_local(|| {
        // Retain at most 512 entries: trim every 512 appends so the shard
        // stays far under its entry cap (steady state, not growth).
        if seq % 512 == 511 {
            black_box(run(
                &mut store,
                &Operation::StreamTrim {
                    key: key("shard"),
                    through_offset: seq,
                },
            ));
        }
        seq += 1;
        black_box(run(
            &mut store,
            &Operation::StreamAppend {
                key: key("shard"),
                partition: b"p".to_vec(),
                payload: payload.clone(),
            },
        ))
    });
}

#[divan::bench]
fn commit_local_four_keys(bencher: Bencher) {
    let mut store = ObjectStore::new();
    let writes: Vec<TxnWrite> = (0..4)
        .map(|index| TxnWrite {
            key: Key::from(format!("local-{index}")),
            kind: TxnWriteKind::Put(Bytes::from_static(b"0123456789abcdef")),
            expect: TxnExpect::Any,
        })
        .collect();
    bencher
        .counter(ItemsCount::new(4u64))
        .bench_local(|| black_box(store.commit_local(black_box(&writes), NOW).expect("commit")));
}

#[divan::bench]
fn prepare_finalize_two_keys(bencher: Bencher) {
    let mut store = ObjectStore::new();
    let coordinator = TabletId::from_u64(1);
    let digest = [0xD1; 32];
    let mut seq: u64 = 0;
    bencher.counter(ItemsCount::new(2u64)).bench_local(|| {
        seq += 1;
        let txn = TxnId::derive(0, seq, 0);
        for name in ["a", "b"] {
            let Prepared::Write(mutation) = store
                .prepare(
                    &Operation::TxnPrepare {
                        txn,
                        coordinator,
                        write: TxnWrite {
                            key: key(name),
                            kind: TxnWriteKind::Put(Bytes::from_static(b"v")),
                            expect: TxnExpect::Any,
                        },
                        digest,
                    },
                    NOW,
                )
                .expect("prepare")
            else {
                panic!("must prepare")
            };
            store.apply(&mutation, NOW).expect("reserve");
            black_box(run(
                &mut store,
                &Operation::TxnFinalize {
                    txn,
                    key: key(name),
                    commit: true,
                    digest,
                },
            ));
        }
    });
}

#[divan::bench]
fn write_set_digest_eight_keys(bencher: Bencher) {
    let writes: Vec<TxnWrite> = (0..8)
        .map(|index| TxnWrite {
            key: Key::from(format!("digest-{index}")),
            kind: TxnWriteKind::Put(Bytes::from_static(b"0123456789abcdef")),
            expect: TxnExpect::Any,
        })
        .collect();
    bencher
        .counter(ItemsCount::new(8u64))
        .bench_local(|| black_box(write_set_digest(black_box(&writes), NS)));
}
