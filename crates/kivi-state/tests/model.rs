//! Model-based test: long random operation sequences against a deliberately
//! simple `BTreeMap` reference implementation.
//!
//! The reference optimizes for clarity, not performance, and is written
//! directly from the intended semantics (not by copying store internals), so
//! agreement covers results, types, versions, expiry behavior, and final
//! state on every step — far stronger than hand-written examples alone.

use std::collections::BTreeMap;

use bytes::Bytes;
use kivi_state::{
    ApplyOutcome, Key, LogicalValue, ObjectStore, OpError, Operation, OperationResult, Prepared,
};
use kivi_types::UnixMicros;
use proptest::prelude::*;

// ---------------------------------------------------------------------------
// Reference model (clarity first).
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RefKind {
    Bytes,
    Counter,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct RefObj {
    kind: RefKind,
    bytes: Vec<u8>,
    int: i64,
    version: u64,
    expiry: Option<u64>,
}

#[derive(Debug, Default)]
struct RefStore {
    map: BTreeMap<Vec<u8>, RefObj>,
}

impl RefStore {
    fn live(&self, key: &[u8], now: u64) -> Option<&RefObj> {
        self.map
            .get(key)
            .filter(|obj| obj.expiry.is_none_or(|e| now < e))
    }

    fn read(&self, key: &[u8], now: u64) -> Result<Out, RefErr> {
        match self.live(key, now) {
            None => Ok(Out::Val(None)),
            Some(obj) => match obj.kind {
                RefKind::Bytes => Ok(Out::Val(Some(obj.bytes.clone()))),
                RefKind::Counter => Err(RefErr::WrongType),
            },
        }
    }

    fn run(&mut self, op: &GenOp, now: u64) -> Result<Out, RefErr> {
        let key = op.key().to_vec();
        match op {
            GenOp::Get(_) => self.read(&key, now),
            GenOp::Set(_, value) => {
                let version = self.live(&key, now).map_or(1, |obj| obj.version + 1);
                self.map.insert(
                    key,
                    RefObj {
                        kind: RefKind::Bytes,
                        bytes: value.clone(),
                        int: 0,
                        version,
                        expiry: None,
                    },
                );
                Ok(Out::Stored(version))
            }
            GenOp::Delete(_) => {
                let existed = self.live(&key, now).is_some();
                self.map.remove(&key);
                Ok(Out::Deleted(existed))
            }
            GenOp::Exists(_) => Ok(Out::Exists(self.live(&key, now).is_some())),
            GenOp::CounterGet(_) => match self.live(&key, now) {
                None => Ok(Out::Cnt(None)),
                Some(obj) => match obj.kind {
                    RefKind::Counter => Ok(Out::Cnt(Some(obj.int))),
                    RefKind::Bytes => Err(RefErr::WrongType),
                },
            },
            GenOp::CounterAdd(_, delta) => match self.live(&key, now).cloned() {
                None => {
                    self.map.insert(
                        key,
                        RefObj {
                            kind: RefKind::Counter,
                            bytes: Vec::new(),
                            int: *delta,
                            version: 1,
                            expiry: None,
                        },
                    );
                    Ok(Out::CntUpd(*delta, 1))
                }
                Some(obj) => match obj.kind {
                    RefKind::Bytes => Err(RefErr::WrongType),
                    RefKind::Counter => {
                        let next = obj.int.checked_add(*delta).ok_or(RefErr::Overflow)?;
                        let version = obj.version + 1;
                        self.map.insert(
                            key,
                            RefObj {
                                kind: RefKind::Counter,
                                bytes: Vec::new(),
                                int: next,
                                version,
                                expiry: obj.expiry,
                            },
                        );
                        Ok(Out::CntUpd(next, version))
                    }
                },
            },
            GenOp::ExpireAt(_, at) => {
                if self.live(&key, now).is_none() {
                    return Ok(Out::ExpSet(false));
                }
                if let Some(obj) = self.map.get_mut(&key) {
                    obj.expiry = Some(*at);
                    obj.version += 1;
                }
                Ok(Out::ExpSet(true))
            }
            GenOp::Persist(_) => {
                let live = self.live(&key, now);
                if live.is_some_and(|obj| obj.expiry.is_some()) {
                    if let Some(obj) = self.map.get_mut(&key) {
                        obj.expiry = None;
                        obj.version += 1;
                    }
                    Ok(Out::ExpPer(true))
                } else {
                    Ok(Out::ExpPer(false))
                }
            }
            GenOp::GetExpiry(_) => Ok(Out::Exp(match self.live(&key, now) {
                None => ExpState::Absent,
                Some(obj) => match obj.expiry {
                    None => ExpState::Never,
                    Some(stamp) => ExpState::At(stamp),
                },
            })),
        }
    }

    fn dump(&self) -> BTreeMap<Vec<u8>, Row> {
        self.map
            .iter()
            .map(|(key, obj)| {
                let kind = match obj.kind {
                    RefKind::Bytes => 0,
                    RefKind::Counter => 1,
                };
                (
                    key.clone(),
                    (kind, obj.bytes.clone(), obj.int, obj.version, obj.expiry),
                )
            })
            .collect()
    }
}

/// Normalized durable row: (kind tag, bytes, counter, version, expiry).
/// Tuples keep the reference dump brutally simple; the alias silences
/// complexity lints without hiding anything.
type Row = (u8, Vec<u8>, i64, u64, Option<u64>);

/// Normalized expiry answer: absent key, live key without expiry, or a stamp.
#[derive(Debug, Clone, PartialEq, Eq)]
enum ExpState {
    Absent,
    Never,
    At(u64),
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Out {
    Val(Option<Vec<u8>>),
    Stored(u64),
    Deleted(bool),
    Exists(bool),
    Cnt(Option<i64>),
    CntUpd(i64, u64),
    ExpSet(bool),
    ExpPer(bool),
    Exp(ExpState),
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum RefErr {
    WrongType,
    Overflow,
}

// ---------------------------------------------------------------------------
// Generated operations.
// ---------------------------------------------------------------------------

const KEYS: [&[u8]; 6] = [b"k0", b"k1", b"k2", b"k3", b"k4", b"k5"];

#[derive(Debug, Clone)]
enum GenOp {
    Get(usize),
    Set(usize, Vec<u8>),
    Delete(usize),
    Exists(usize),
    CounterGet(usize),
    CounterAdd(usize, i64),
    ExpireAt(usize, u64),
    Persist(usize),
    GetExpiry(usize),
}

impl GenOp {
    fn key(&self) -> &[u8] {
        let index = match self {
            Self::Get(i)
            | Self::Set(i, _)
            | Self::Delete(i)
            | Self::Exists(i)
            | Self::CounterGet(i)
            | Self::CounterAdd(i, _)
            | Self::ExpireAt(i, _)
            | Self::Persist(i)
            | Self::GetExpiry(i) => *i,
        };
        KEYS[index]
    }
}

fn arb_delta() -> impl Strategy<Value = i64> {
    prop_oneof![-10i64..10i64, Just(i64::MAX), Just(i64::MIN), Just(0i64),]
}

fn arb_step() -> impl Strategy<Value = (GenOp, u64)> {
    let op = prop_oneof![
        (0usize..6).prop_map(GenOp::Get),
        (0usize..6, prop::collection::vec(any::<u8>(), 0..8)).prop_map(|(k, v)| GenOp::Set(k, v)),
        (0usize..6).prop_map(GenOp::Delete),
        (0usize..6).prop_map(GenOp::Exists),
        (0usize..6).prop_map(GenOp::CounterGet),
        (0usize..6, arb_delta()).prop_map(|(k, d)| GenOp::CounterAdd(k, d)),
        (0usize..6, 0u64..2000).prop_map(|(k, rel)| GenOp::ExpireAt(k, rel)),
        (0usize..6).prop_map(GenOp::Persist),
        (0usize..6).prop_map(GenOp::GetExpiry),
    ];
    (op, 0u64..500).prop_map(|(op, advance)| (op, advance))
}

// ---------------------------------------------------------------------------
// Kivi-side normalization to the shared comparison vocabulary.
// ---------------------------------------------------------------------------

fn kivi_key(index: usize) -> Key {
    Key::from(KEYS[index])
}

fn norm_result(result: &OperationResult) -> Out {
    match result {
        OperationResult::Value(value) => Out::Val(value.clone().map(|b| b.to_vec())),
        OperationResult::Stored { version } => Out::Stored(version.as_u64()),
        OperationResult::Deleted { existed } => Out::Deleted(*existed),
        OperationResult::Exists(present) => Out::Exists(*present),
        OperationResult::Counter(value) => Out::Cnt(*value),
        OperationResult::CounterUpdated { value, version } => Out::CntUpd(*value, version.as_u64()),
        OperationResult::ExpirySet { applied } => Out::ExpSet(*applied),
        OperationResult::ExpiryPersisted { removed } => Out::ExpPer(*removed),
        OperationResult::Expiry(expiry) => Out::Exp(match expiry {
            None => ExpState::Absent,
            Some(value) => match value.as_stamp() {
                None => ExpState::Never,
                Some(stamp) => ExpState::At(kivi_types::UnixMicros::as_micros(stamp)),
            },
        }),
    }
}

fn norm_error(error: &OpError) -> RefErr {
    match error {
        OpError::WrongType { .. } => RefErr::WrongType,
        OpError::CounterOverflow => RefErr::Overflow,
    }
}

fn kivi_dump(store: &ObjectStore) -> BTreeMap<Vec<u8>, Row> {
    store
        .snapshot_sorted()
        .into_iter()
        .map(|(key, object)| {
            let (kind, bytes, int) = match object.value() {
                LogicalValue::Bytes(value) => (0, value.to_vec(), 0),
                LogicalValue::StrictCounter(value) => (1, Vec::new(), *value),
            };
            (
                key.as_bytes().to_vec(),
                (
                    kind,
                    bytes,
                    int,
                    object.version().as_u64(),
                    object
                        .expiry()
                        .as_stamp()
                        .map(kivi_types::UnixMicros::as_micros),
                ),
            )
        })
        .collect()
}

fn kivi_op(step: &GenOp) -> Operation {
    match step {
        GenOp::Get(i) => Operation::Get { key: kivi_key(*i) },
        GenOp::Set(i, value) => Operation::Set {
            key: kivi_key(*i),
            value: Bytes::from(value.clone()),
        },
        GenOp::Delete(i) => Operation::Delete { key: kivi_key(*i) },
        GenOp::Exists(i) => Operation::Exists { key: kivi_key(*i) },
        GenOp::CounterGet(i) => Operation::CounterGet { key: kivi_key(*i) },
        GenOp::CounterAdd(i, delta) => Operation::CounterAdd {
            key: kivi_key(*i),
            delta: *delta,
        },
        GenOp::ExpireAt(i, abs) => Operation::ExpireAt {
            key: kivi_key(*i),
            // Already absolutized by the driver loop; never re-offset here.
            expires_at: UnixMicros::from_micros(*abs),
        },
        GenOp::Persist(i) => Operation::PersistExpiry { key: kivi_key(*i) },
        GenOp::GetExpiry(i) => Operation::GetExpiry { key: kivi_key(*i) },
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    #[test]
    fn operations_match_reference_model(steps in prop::collection::vec(arb_step(), 1..64)) {
        let mut store = ObjectStore::new();
        let mut reference = RefStore::default();
        let mut now: u64 = 1_000_000;
        for (step, advance) in &steps {
            now = now.checked_add(*advance).expect("test clock fits");
            let stamp = UnixMicros::from_micros(now);
            // Expiry offsets are relative to the step's timestamp on both sides.
            let step = match step {
                GenOp::ExpireAt(i, rel) => GenOp::ExpireAt(*i, now.saturating_add(*rel)),
                other => other.clone(),
            };
            let expected = reference.run(&step, now);
            let actual = match store.prepare(&kivi_op(&step), stamp) {
                Err(error) => Err(norm_error(&error)),
                Ok(Prepared::Read(result)) => Ok(norm_result(&result)),
                Ok(Prepared::Write(mutation)) => {
                    let outcome = store.apply(&mutation, stamp).expect("valid writes apply");
                    Ok(norm_apply(&step, &outcome))
                }
            };
            prop_assert_eq!(actual, expected, "divergence on {:?} at t={}", step, now);
        }
        prop_assert_eq!(kivi_dump(&store), reference.dump());
    }
}

/// Normalizes an apply outcome the way its originating operation observes it.
fn norm_apply(step: &GenOp, outcome: &ApplyOutcome) -> Out {
    match (step, outcome) {
        (_, ApplyOutcome::Put { version }) => Out::Stored(version.as_u64()),
        (_, ApplyOutcome::Deleted { existed }) => Out::Deleted(*existed),
        (_, ApplyOutcome::Counter { version, value }) => Out::CntUpd(*value, version.as_u64()),
        (
            GenOp::ExpireAt(..),
            ApplyOutcome::Expiry {
                applied,
                version: _,
            },
        ) => Out::ExpSet(*applied),
        (
            GenOp::Persist(..),
            ApplyOutcome::Expiry {
                applied,
                version: _,
            },
        ) => Out::ExpPer(*applied),
        (step, outcome) => panic!("unexpected {step:?} -> {outcome:?}"),
    }
}
