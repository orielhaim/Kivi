//! New native operations over real loopback TCP: range reads, logical
//! lengths, and conditional stores (all three wire opcodes, both inline and
//! chunked roots, through the reactor path — never the embedded channel).

use std::sync::{Arc, Mutex};
use std::thread;

use bytes::Bytes;
use kivi_client::{ClientConfig, NativeClient};
use kivi_engine::{
    ChunkFabricConfig, ConnLimits, DurabilityMode, EngineConfig, EngineNetwork, FabricConfig,
    LocalEngine, Placement, TurnBudget,
};
use kivi_lab::workload::{Workload, WorkloadOp, workload_op_for};
use kivi_state::{ExpiryPolicy, Key, SetCondition};
use kivi_tablet::{DirectorySnapshot, HashPrefix, PartitionRange};
use kivi_types::{
    ClusterId, NamespaceId, NodeId, NodeIncarnation, TabletId, WallTimestamp, WorkerId,
};

const NS: NamespaceId = NamespaceId::from_u64(1);

fn directory() -> DirectorySnapshot {
    DirectorySnapshot::bootstrap(
        NS,
        TabletId::from_u64(1),
        PartitionRange::Hash(HashPrefix::new(0, 0).expect("root")),
        kivi_types::TabletEpoch::INITIAL,
        kivi_types::WriteGuardGeneration::INITIAL,
    )
    .expect("genesis")
    .stage(TabletId::from_u64(1))
    .and_then(|snapshot| snapshot.activate(TabletId::from_u64(1)))
    .expect("active root")
}

fn start_ephemeral() -> LocalEngine {
    LocalEngine::start(EngineConfig {
        namespace: NS,
        directory: directory(),
        placement: Placement::new([(TabletId::from_u64(1), WorkerId::from_u64(0))]),
        worker_count: 2,
        request_capacity: 128,
        chunks: ChunkFabricConfig::default(),
        fabric: FabricConfig::default(),
        network: Some(EngineNetwork {
            base_port: 0,
            ports: Vec::new(),
            bind_ip: [127, 0, 0, 1].into(),
            max_frame: kivi_protocol::DEFAULT_MAX_FRAME,
            affinity: kivi_engine::AffinityMode::Disabled,
            node_id: NodeId::from_u64(1),
            cluster_id: ClusterId::from_u128(1),
            incarnation: NodeIncarnation::INITIAL,
            conn: ConnLimits::default(),
            turn: TurnBudget::default(),
        }),
        durability: DurabilityMode::Ephemeral,
    })
    .expect("networked engine starts")
}

fn client_for(engine: &LocalEngine) -> NativeClient {
    NativeClient::new(ClientConfig {
        seeds: vec![engine.worker_addrs()[0].to_string()],
        namespace: NS,
        ..ClientConfig::default()
    })
    .expect("client builds")
}

#[test]
fn ranges_slice_and_measure_inline_values() {
    let engine = start_ephemeral();
    let client = client_for(&engine);
    let key = Key::from("rk");
    assert_eq!(client.get_range(&key, 0, 4).expect("range"), None);
    assert_eq!(client.bytes_length(&key).expect("length"), None);
    client.set(&key, Bytes::from_static(b"hello")).expect("set");
    assert_eq!(
        client.get_range(&key, 1, 3).expect("range"),
        Some(Bytes::from_static(b"ell"))
    );
    assert_eq!(
        client.get_range(&key, 99, 4).expect("past end"),
        Some(Bytes::new())
    );
    assert_eq!(client.bytes_length(&key).expect("length"), Some(5));
    engine.shutdown().expect("clean shutdown");
}

#[test]
fn medium_range_patch_can_expire() {
    let engine = start_ephemeral();
    let client = client_for(&engine);
    let key = Key::from("medium-expire");
    let value = kivi_lab::workload::fill_pattern(1024, 9);
    client
        .set(&key, Bytes::copy_from_slice(&value))
        .expect("set");
    client
        .set_range(&key, 0, Bytes::from_static(b"v"))
        .expect("range patch");
    assert_eq!(
        client.get_range(&key, 0, 64).expect("range"),
        Some({
            let mut expected = value[..64].to_vec();
            expected[0] = b'v';
            Bytes::from(expected)
        })
    );
    let deadline = WallTimestamp::from_micros(9_000_000_000_000_000);
    assert!(client.expire_at(&key, deadline).expect("expire"));
    assert_eq!(
        client.get_expiry(&key).expect("expiry"),
        Some(kivi_types::Expiry::at(deadline))
    );
    engine.shutdown().expect("clean shutdown");
}

#[test]
fn conditional_stores_evaluate_atomically_with_policies() {
    let engine = start_ephemeral();
    let client = client_for(&engine);
    let key = Key::from("ck");
    // Absent: NX applies, XX refuses without a version.
    assert_eq!(
        client
            .set_conditional(
                &key,
                Bytes::from_static(b"v1"),
                SetCondition::IfAbsent,
                ExpiryPolicy::Clear
            )
            .expect("nx"),
        (true, Some(1))
    );
    assert_eq!(
        client
            .set_conditional(
                &key,
                Bytes::from_static(b"v2"),
                SetCondition::IfAbsent,
                ExpiryPolicy::Clear
            )
            .expect("nx again"),
        (false, None)
    );
    // Present: XX applies; Keep preserves a dated expiry (far future:
    // wall-clock `now` here is real time, not the unit-test stub).
    let deadline = WallTimestamp::from_micros(9_000_000_000_000_000);
    assert!(client.expire_at(&key, deadline).expect("expire"));
    assert_eq!(
        client
            .set_conditional(
                &key,
                Bytes::from_static(b"v3"),
                SetCondition::IfPresent,
                ExpiryPolicy::Keep
            )
            .expect("xx keep"),
        (true, Some(3))
    );
    assert_eq!(
        client.get_expiry(&key).expect("expiry"),
        Some(kivi_types::Expiry::at(deadline))
    );
    assert_eq!(
        client.get(&key).expect("get"),
        Some(Bytes::from_static(b"v3"))
    );
    engine.shutdown().expect("clean shutdown");
}

#[test]
fn chunked_roots_slice_and_measure_without_full_reads() {
    let engine = start_ephemeral();
    let client = client_for(&engine);
    let key = Key::from("big-range");
    // Multi-chunk value (2.5 MiB): stages through the streaming upload.
    let mut big = Vec::with_capacity(2_500_000);
    let mut state = 0x57EAu64;
    while big.len() < 2_500_000 {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        big.extend_from_slice(&z.to_le_bytes());
    }
    big.truncate(2_500_000);
    let mut source = std::io::Cursor::new(big.clone());
    client
        .put_stream(&key, Some(big.len() as u64), &mut source)
        .expect("upload");
    // Logical length answers from root metadata.
    assert_eq!(client.bytes_length(&key).expect("length"), Some(2_500_000));
    // Slices match the same window of the uploaded bytes.
    let window = client
        .get_range(&key, 1_000_000, 16)
        .expect("range")
        .expect("present");
    assert_eq!(&window[..], &big[1_000_000..1_000_016]);
    engine.shutdown().expect("clean shutdown");
}

#[test]
#[allow(clippy::ignored_unit_patterns)]
fn repeated_fabric_set_range_keeps_root_readable() {
    let data_dir = tempfile::tempdir().expect("data directory");
    let server =
        kivi_lab::process::Server::spawn_auto(data_dir.path(), &[]).expect("server starts");
    let client = NativeClient::new(ClientConfig {
        seeds: vec![server.endpoint()],
        namespace: NS,
        ..ClientConfig::default()
    })
    .expect("client builds");
    let keys = vec!["repeated-fabric-range".to_owned()];
    let counter = "unused-counter";
    let value = Bytes::from(kivi_lab::workload::fill_pattern(1024, 9));
    client
        .set(&Key::from(keys[0].as_str()), value.clone())
        .expect("initial set");
    for index in 0..20 {
        let operation = workload_op_for(Workload::Compat, 1024, index, &keys, counter);
        let result = match &operation {
            WorkloadOp::Get(key) => client.get(&Key::from(key.as_str())).map(|_| ()),
            WorkloadOp::Set { key, .. } => client
                .set(&Key::from(key.as_str()), value.clone())
                .map(|_| ()),
            WorkloadOp::Expire(key) => client
                .expire_at(
                    &Key::from(key.as_str()),
                    WallTimestamp::from_micros(9_000_000_000_000_000),
                )
                .map(|_| ()),
            WorkloadOp::Ttl(key) => client.get_expiry(&Key::from(key.as_str())).map(|_| ()),
            WorkloadOp::SetRange(key) => client
                .set_range(&Key::from(key.as_str()), 0, Bytes::from_static(b"v"))
                .map(|_| ()),
            WorkloadOp::Exists(key) => client.exists(&Key::from(key.as_str())).map(|_| ()),
            WorkloadOp::Delete(key) => client.delete(&Key::from(key.as_str())).map(|_| ()),
            WorkloadOp::GetRange(key) => client
                .get_range(&Key::from(key.as_str()), 0, 64)
                .map(|_| ()),
            WorkloadOp::CounterAdd(_) => Ok(()),
        };
        assert!(
            result.is_ok(),
            "operation {index} {operation:?} failed: {result:?}"
        );
        client
            .get(&Key::from(keys[0].as_str()))
            .unwrap_or_else(|error| panic!("read after {index} {operation:?} failed: {error}"));
    }
    client
        .get(&Key::from(keys[0].as_str()))
        .expect("final get")
        .expect("final value");
    server.kill();
}

#[test]
#[allow(clippy::ignored_unit_patterns)]
fn concurrent_balanced_operations_have_no_terminal_errors() {
    let data_dir = tempfile::tempdir().expect("data directory");
    let server =
        kivi_lab::process::Server::spawn_auto(data_dir.path(), &[]).expect("server starts");
    let keys: Vec<_> = (0..256)
        .map(|index| format!("concurrent-balanced:g:k{index}"))
        .collect();
    let value = Bytes::from(kivi_lab::workload::fill_pattern(1024, 9));
    let seeder = NativeClient::new(ClientConfig {
        seeds: vec![server.endpoint()],
        namespace: NS,
        ..ClientConfig::default()
    })
    .expect("seeder builds");
    for key in &keys {
        seeder
            .set(&Key::from(key.as_str()), value.clone())
            .expect("seed key");
    }
    let errors = Arc::new(Mutex::new(Vec::new()));
    let endpoint = server.endpoint();
    let mut handles = Vec::new();
    for thread in 0..4 {
        let keys = keys.clone();
        let value = value.clone();
        let errors = Arc::clone(&errors);
        let endpoint = endpoint.clone();
        handles.push(thread::spawn(move || {
            let client = NativeClient::new(ClientConfig {
                seeds: vec![endpoint],
                namespace: NS,
                ..ClientConfig::default()
            })
            .expect("client builds");
            for index in 0..100 {
                let operation = workload_op_for(Workload::Balanced, 1024, index, &keys, "unused");
                let result = match &operation {
                    WorkloadOp::Get(key) => client.get(&Key::from(key.as_str())).map(|_| ()),
                    WorkloadOp::Set { key, .. } => client
                        .set(&Key::from(key.as_str()), value.clone())
                        .map(|_| ()),
                    WorkloadOp::Delete(key) => client.delete(&Key::from(key.as_str())).map(|_| ()),
                    WorkloadOp::Exists(key) => client.exists(&Key::from(key.as_str())).map(|_| ()),
                    WorkloadOp::GetRange(key) => client
                        .get_range(&Key::from(key.as_str()), 0, 64)
                        .map(|_| ()),
                    WorkloadOp::SetRange(key) => client
                        .set_range(&Key::from(key.as_str()), 0, Bytes::from_static(b"v"))
                        .map(|_| ()),
                    WorkloadOp::Expire(key) => client
                        .expire_at(
                            &Key::from(key.as_str()),
                            WallTimestamp::from_micros(9_000_000_000_000_000),
                        )
                        .map(|_| ()),
                    WorkloadOp::Ttl(key) => client.get_expiry(&Key::from(key.as_str())).map(|_| ()),
                    WorkloadOp::CounterAdd(_) => Ok(()),
                };
                if let Err(error) = result {
                    errors.lock().expect("errors lock").push(format!(
                        "thread={thread} index={index} operation={operation:?} error={error}"
                    ));
                }
            }
        }));
    }
    for handle in handles {
        handle.join().expect("worker joins");
    }
    let errors = errors.lock().expect("errors lock").clone();
    server.kill();
    assert!(errors.is_empty(), "{}", errors.join("\n"));
}
