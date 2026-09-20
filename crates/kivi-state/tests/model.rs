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
            // Inline, chunked, and fabric stores are logically identical:
            // the reference never learns representations.
            GenOp::Set(_, value) | GenOp::SetChunked(_, value) | GenOp::SetFabric(_, value) => {
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
            GenOp::SetRange(_, offset, patch) => self.splice(&key, *offset, patch, now),
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

    /// Independent spelling of the splice: absent state reads as empty,
    /// counters reject, past-the-end gaps zero-pad, and the live expiry
    /// survives (partial writes never touch the TTL).
    fn splice(&mut self, key: &[u8], offset: u64, patch: &[u8], now: u64) -> Result<Out, RefErr> {
        let (base, expiry) = match self.live(key, now) {
            None => (Vec::new(), None),
            Some(obj) => match obj.kind {
                RefKind::Bytes => (obj.bytes.clone(), obj.expiry),
                RefKind::Counter => return Err(RefErr::WrongType),
            },
        };
        let offset = usize::try_from(offset).expect("model offsets fit the address space");
        let end = offset.checked_add(patch.len()).expect("model splices fit");
        let mut spliced = Vec::new();
        spliced.extend_from_slice(&base[..offset.min(base.len())]);
        spliced.resize(offset, 0);
        spliced.extend_from_slice(patch);
        if base.len() > end {
            spliced.extend_from_slice(&base[end..]);
        }
        let version = self.live(key, now).map_or(1, |obj| obj.version + 1);
        self.map.insert(
            key.to_vec(),
            RefObj {
                kind: RefKind::Bytes,
                bytes: spliced,
                int: 0,
                version,
                expiry,
            },
        );
        Ok(Out::Stored(version))
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
    Len(Option<u64>),
    Cond(bool, Option<u64>),
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
    /// Chunked spelling of `Set`: overwrites any type with a chunk
    /// reference naming `Vec<u8>` bytes. The reference model treats it
    /// exactly like `Set` — representation must never leak into logical
    /// semantics — while the Kivi side stores a chunked root the driver
    /// resolves through the manifest registry below.
    SetChunked(usize, Vec<u8>),
    /// Fabric spelling of `Set`: overwrites any type with a fabric
    /// reference naming `Vec<u8>` bytes. Same contract as `SetChunked`
    /// through the fabric registry below.
    SetFabric(usize, Vec<u8>),
    SetRange(usize, u64, Vec<u8>),
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
            | Self::SetChunked(i, _)
            | Self::SetFabric(i, _)
            | Self::SetRange(i, _, _)
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

/// Deterministic synthetic manifest id for model bytes: stands in for the
/// real chunking the engine performs (which this layer never sees). The
/// registry below maps these back to bytes, mirroring engine resolution.
fn fake_manifest_id(key: &[u8], value: &[u8]) -> kivi_types::ManifestId {
    let mut input = Vec::with_capacity(11 + key.len() + value.len());
    input.extend_from_slice(b"model-chunk");
    input.extend_from_slice(key);
    input.extend_from_slice(value);
    let digest = blake3::hash(&input);
    let id = kivi_types::ManifestId::from_bytes(*digest.as_bytes());
    debug_assert!(id.is_valid(), "test ids must be nonzero");
    id
}

/// Deterministic synthetic fabric id for model bytes: stands in for the
/// engine's staging (which this layer never sees). The registry below
/// maps these back to bytes, mirroring engine resolution.
fn fake_fabric_id(key: &[u8], value: &[u8]) -> u64 {
    let mut input = Vec::with_capacity(12 + key.len() + value.len());
    input.extend_from_slice(b"model-fabric");
    input.extend_from_slice(key);
    input.extend_from_slice(value);
    let digest = blake3::hash(&input);
    let mut raw = [0u8; 8];
    raw.copy_from_slice(&digest.as_bytes()[..8]);
    u64::from_le_bytes(raw).max(1)
}

fn arb_delta() -> impl Strategy<Value = i64> {
    prop_oneof![-10i64..10i64, Just(i64::MAX), Just(i64::MIN), Just(0i64),]
}

fn arb_step() -> impl Strategy<Value = (GenOp, u64)> {
    let op = prop_oneof![
        (0usize..6).prop_map(GenOp::Get),
        (0usize..6, prop::collection::vec(any::<u8>(), 0..8)).prop_map(|(k, v)| GenOp::Set(k, v)),
        (0usize..6, prop::collection::vec(any::<u8>(), 0..8))
            .prop_map(|(k, v)| GenOp::SetChunked(k, v)),
        (0usize..6, prop::collection::vec(any::<u8>(), 0..8))
            .prop_map(|(k, v)| GenOp::SetFabric(k, v)),
        // Small offsets and patches: covers in-place patches, truncation
        // past the end, and zero-padded gaps past the end.
        (
            0usize..6,
            0u64..16,
            prop::collection::vec(any::<u8>(), 0..8),
        )
            .prop_map(|(k, off, p)| GenOp::SetRange(k, off, p)),
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

fn norm_result(
    result: &OperationResult,
    registry: &BTreeMap<[u8; 32], Vec<u8>>,
    fabric: &BTreeMap<u64, Vec<u8>>,
) -> Out {
    match result {
        OperationResult::Value(value) => Out::Val(value.clone().map(|b| b.to_vec())),
        // Engine resolution, mirrored: the test registries stand in for
        // the chunk lane and fabric (which this layer never touches).
        OperationResult::ChunkedValue {
            manifest,
            logical_len,
        } => {
            let bytes = registry
                .get(manifest.as_bytes())
                .expect("chunked reads name planted manifests");
            assert_eq!(
                u64::try_from(bytes.len()).expect("model values fit u64"),
                *logical_len,
                "logical length matches resolved bytes"
            );
            Out::Val(Some(bytes.clone()))
        }
        OperationResult::FabricValue {
            fabric_id,
            logical_len,
            version: _,
        } => {
            let bytes = fabric
                .get(fabric_id)
                .expect("fabric reads name planted ids");
            assert_eq!(
                u64::try_from(bytes.len()).expect("model values fit u64"),
                *logical_len,
                "logical length matches resolved bytes"
            );
            Out::Val(Some(bytes.clone()))
        }
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
        OperationResult::Length(value) => Out::Len(*value),
        // The model generates no version reads; reaching one is a bug.
        OperationResult::Version(_) => panic!("model driver generated no version reads"),
        OperationResult::ConditionalSet { applied, version } => {
            Out::Cond(*applied, version.map(kivi_state::ObjectVersion::as_u64))
        }
        // Transaction outcomes never arise from the model's single-key
        // operation generator; reaching one is a model bug.
        OperationResult::TxnPrepared
        | OperationResult::TxnConflict
        | OperationResult::TxnFinalized { .. }
        | OperationResult::TxnLocalCommitted { .. } => {
            panic!("model driver generated no transaction operations")
        }
        // Semantic outcomes never arise either: the generator covers the
        // original bytes/counter surface, while the semantic types carry
        // their own property suites in the store module.
        OperationResult::CommutativeApplied
        | OperationResult::CommutativeValue(_)
        | OperationResult::BoundedUpdated { .. }
        | OperationResult::BoundedValue { .. }
        | OperationResult::SemaphoreAcquired
        | OperationResult::SemaphoreReleased { .. }
        | OperationResult::SemaphoreLoad { .. }
        | OperationResult::LeaseAcquired { .. }
        | OperationResult::LeaseRenewed { .. }
        | OperationResult::LeaseReleased { .. }
        | OperationResult::LeaseInfo { .. }
        | OperationResult::StreamCreated
        | OperationResult::StreamAppended { .. }
        | OperationResult::StreamEntries { .. }
        | OperationResult::StreamTrimmed { .. } => {
            panic!("model driver generated no semantic operations")
        }
    }
}

fn norm_error(error: &OpError) -> RefErr {
    match error {
        OpError::WrongType { .. } => RefErr::WrongType,
        OpError::CounterOverflow => RefErr::Overflow,
        // Unreachable: the driver resolves chunked bases before a
        // `SetRange` ever reaches `prepare` (mirroring the engine's lane
        // restage), so a stale base here is a model bug, not a case.
        OpError::StaleRangeBase => panic!("model driver leaked a chunked base into SetRange"),
        // Unreachable: the model generates no transaction operations, so
        // no intent can block a key.
        OpError::TxnConflict => panic!("model driver generated no transaction operations"),
        // Unreachable: the generator covers bytes/counters only; semantic
        // failures carry their own suites in the store module.
        OpError::BoundedExceeded
        | OpError::SemaphoreExhausted
        | OpError::LeaseConflict
        | OpError::StaleFencing
        | OpError::StreamFull
        | OpError::NotFound => panic!("model driver generated no semantic operations"),
    }
}

fn kivi_dump(
    store: &ObjectStore,
    registry: &BTreeMap<[u8; 32], Vec<u8>>,
    fabric: &BTreeMap<u64, Vec<u8>>,
) -> BTreeMap<Vec<u8>, Row> {
    store
        .snapshot_sorted()
        .into_iter()
        .map(|(key, object)| {
            let (kind, bytes, int) = match object.value() {
                LogicalValue::Bytes(value) => (0, value.to_vec(), 0),
                LogicalValue::Chunked(chunked) => (
                    0,
                    registry
                        .get(chunked.manifest.as_bytes())
                        .expect("dump resolves planted manifests")
                        .clone(),
                    0,
                ),
                LogicalValue::Fabric(fabric_ref) => (
                    0,
                    fabric
                        .get(&fabric_ref.id)
                        .expect("dump resolves planted fabric ids")
                        .clone(),
                    0,
                ),
                LogicalValue::StrictCounter(value) => (1, Vec::new(), *value),
                // Unreachable: the generator never creates semantic objects.
                LogicalValue::CommutativeCounter(_)
                | LogicalValue::BoundedCounter(_)
                | LogicalValue::Semaphore(_)
                | LogicalValue::Lease(_)
                | LogicalValue::StreamShard(_) => {
                    panic!("model driver generated no semantic objects")
                }
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
        GenOp::SetChunked(i, value) => {
            let key = KEYS[*i];
            let manifest = fake_manifest_id(key, value);
            Operation::SetChunked {
                key: kivi_key(*i),
                manifest,
                logical_len: u64::try_from(value.len()).expect("model values fit u64"),
            }
        }
        GenOp::SetFabric(i, value) => {
            let key = KEYS[*i];
            Operation::SetFabric {
                key: kivi_key(*i),
                fabric_id: fake_fabric_id(key, value),
                logical_len: u64::try_from(value.len()).expect("model values fit u64"),
                // The model pins reads by root version, not by staged
                // pin: any pin applies identically here (apply overwrites
                // it with the authoritative version).
                version: 0,
            }
        }
        GenOp::SetRange(i, offset, patch) => Operation::SetRange {
            key: kivi_key(*i),
            offset: *offset,
            patch: Bytes::from(patch.clone()),
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
        // Manifest/fabric registries: stand in for the chunk lane and
        // fabric when the Kivi side names non-inline bytes. Planted
        // before each referenced store so reads and dumps resolve
        // exactly like engine resolution would.
        let mut registry: BTreeMap<[u8; 32], Vec<u8>> = BTreeMap::new();
        let mut fabric: BTreeMap<u64, Vec<u8>> = BTreeMap::new();
        let mut now: u64 = 1_000_000;
        for (step, advance) in &steps {
            now = now.checked_add(*advance).expect("test clock fits");
            let stamp = UnixMicros::from_micros(now);
            // Expiry offsets are relative to the step's timestamp on both sides.
            let step = match step {
                GenOp::ExpireAt(i, rel) => GenOp::ExpireAt(*i, now.saturating_add(*rel)),
                other => other.clone(),
            };
            if let GenOp::SetChunked(i, value) = &step {
                let key = KEYS[*i];
                registry.insert(*fake_manifest_id(key, value).as_bytes(), value.clone());
            }
            if let GenOp::SetFabric(i, value) = &step {
                let key = KEYS[*i];
                fabric.insert(fake_fabric_id(key, value), value.clone());
            }
            // Engine mirror: a `SetRange` against a chunked or fabric root
            // never reaches `prepare` as a splice — the engine resolves
            // through the registry and stores the patched bytes. Resolve
            // here through the registries and run a plain `Set` on both
            // sides instead.
            let step = match &step {
                GenOp::SetRange(i, offset, patch)
                    if matches!(
                        store.get(&kivi_key(*i), stamp).map(kivi_state::StoredObject::value),
                        Some(LogicalValue::Chunked(_) | LogicalValue::Fabric(_))
                    ) =>
                {
                    let base = match store.get(&kivi_key(*i), stamp) {
                        Some(object) => match object.value() {
                            LogicalValue::Bytes(value) => value.to_vec(),
                            LogicalValue::StrictCounter(_) => Vec::new(),
                            LogicalValue::Chunked(chunked) => registry
                                .get(chunked.manifest.as_bytes())
                                .expect("driver resolves planted manifests")
                                .clone(),
                            LogicalValue::Fabric(fabric_ref) => fabric
                                .get(&fabric_ref.id)
                                .expect("driver resolves planted fabric ids")
                                .clone(),
                            // Unreachable: the generator never creates
                            // semantic objects.
                            LogicalValue::CommutativeCounter(_)
                            | LogicalValue::BoundedCounter(_)
                            | LogicalValue::Semaphore(_)
                            | LogicalValue::Lease(_)
                            | LogicalValue::StreamShard(_) => {
                                panic!("model driver generated no semantic objects")
                            }
                        },
                        None => Vec::new(),
                    };
                    let offset =
                        usize::try_from(*offset).expect("model offsets fit the address space");
                    let end = offset.checked_add(patch.len()).expect("model splices fit");
                    let mut spliced = Vec::new();
                    spliced.extend_from_slice(&base[..offset.min(base.len())]);
                    spliced.resize(offset, 0);
                    spliced.extend_from_slice(patch);
                    if base.len() > end {
                        spliced.extend_from_slice(&base[end..]);
                    }
                    GenOp::Set(*i, spliced)
                }
                _ => step,
            };
            let expected = reference.run(&step, now);
            let actual = match store.prepare(&kivi_op(&step), stamp) {
                Err(error) => Err(norm_error(&error)),
                Ok(Prepared::Read(result)) => Ok(norm_result(&result, &registry, &fabric)),
                Ok(Prepared::Write(mutation)) => {
                    let outcome = store.apply(&mutation, stamp).expect("valid writes apply");
                    Ok(norm_apply(&step, &outcome))
                }
            };
            prop_assert_eq!(actual, expected, "divergence on {:?} at t={}", step, now);
        }
        prop_assert_eq!(kivi_dump(&store, &registry, &fabric), reference.dump());
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
