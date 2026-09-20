//! Black-box contract for Phase-8 semantics over real loopback TCP: every
//! semantic type and the atomic-batch driver exercised through the public
//! client API only — no engine or store internals. Typed errors must
//! survive the wire; lifecycles must behave end to end.

use bytes::Bytes;
use kivi_client::{
    AtomicBatchResult, BatchExpect, BatchWriteKind, BatchWriteSpec, ClientConfig, ClientError,
    NativeClient,
};
use kivi_engine::{
    ChunkFabricConfig, ConnLimits, DurabilityMode, EngineConfig, EngineNetwork, FabricConfig,
    LocalEngine, Placement, TurnBudget,
};
use kivi_state::Key;
use kivi_tablet::{DirectorySnapshot, HashPrefix, PartitionRange};
use kivi_types::{ClusterId, NamespaceId, NodeId, NodeIncarnation, TabletId, WorkerId};

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

fn key(name: &str) -> Key {
    Key::from(name)
}

#[test]
fn semaphore_lifecycle_over_wire() {
    let engine = start_ephemeral();
    let client = client_for(&engine);
    let sem = key("bb-sem");
    client.semaphore_create(&sem, 2).expect("create");
    let permit = [1; 16];
    client
        .semaphore_acquire(&sem, permit, 11, 2)
        .expect("acquire all");
    assert!(
        matches!(
            client.semaphore_acquire(&sem, [2; 16], 12, 1),
            Err(ClientError::SemaphoreExhausted),
        ),
        "exhaustion is typed over the wire",
    );
    // Identical retry: success without a second acquisition.
    client
        .semaphore_acquire(&sem, permit, 11, 2)
        .expect("retry");
    let load = client.semaphore_inspect(&sem).expect("inspect");
    assert_eq!((load.outstanding, load.capacity), (2, 2));
    assert!(!client.semaphore_release(&sem, [9; 16]).expect("release"));
    assert!(client.semaphore_release(&sem, permit).expect("release"));
    client
        .semaphore_acquire(&sem, [2; 16], 12, 2)
        .expect("re-acquire fills");
}

#[test]
fn lease_lifecycle_over_wire() {
    let engine = start_ephemeral();
    let client = client_for(&engine);
    let lease = key("bb-lease");
    let grant = client
        .lease_acquire(&lease, 1, 60_000_000)
        .expect("acquire");
    assert!(
        matches!(
            client.lease_acquire(&lease, 2, 60_000_000),
            Err(ClientError::LeaseConflict),
        ),
        "live holder blocks over the wire",
    );
    assert!(
        matches!(
            client.lease_renew(&lease, 1, grant.fencing + 100, 60_000_000),
            Err(ClientError::StaleFencing),
        ),
        "stale renew is typed over the wire",
    );
    let renewed = client
        .lease_renew(&lease, 1, grant.fencing, 60_000_000)
        .expect("renew");
    assert_eq!(renewed.fencing, grant.fencing, "renew keeps the token");
    assert!(
        client
            .lease_release(&lease, 1, grant.fencing)
            .expect("release")
    );
    let grant2 = client
        .lease_acquire(&lease, 2, 60_000_000)
        .expect("re-acquire");
    assert!(
        grant2.fencing > grant.fencing,
        "successor fences predecessor",
    );
    let info = client.lease_inspect(&lease).expect("inspect");
    assert_eq!(info.owner, Some(2));
    assert_eq!(info.fencing, grant2.fencing);
}

#[test]
fn counters_over_wire() {
    let engine = start_ephemeral();
    let client = client_for(&engine);
    let free = key("bb-cc");
    client.commutative_add(&free, 5).expect("add");
    client.commutative_add(&free, -2).expect("add");
    assert_eq!(client.commutative_get(&free).expect("get"), Some(3));
    let bounded = key("bb-bounded");
    client.bounded_create(&bounded, 10).expect("create");
    assert_eq!(client.bounded_add(&bounded, 6).expect("add").0, 6,);
    assert!(
        matches!(
            client.bounded_add(&bounded, 5),
            Err(ClientError::BoundedExceeded),
        ),
        "escrow bound is typed over the wire",
    );
    let value = client.bounded_get(&bounded).expect("get").expect("present");
    assert_eq!((value.value, value.capacity), (6, 10));
    assert_eq!((value.share_min, value.share_max), (0, 10));
}

#[test]
fn stream_lifecycle_over_wire() {
    let engine = start_ephemeral();
    let client = client_for(&engine);
    let shard = key("bb-shard");
    client.stream_create(&shard, [7; 16], 0).expect("create");
    assert_eq!(
        client.stream_append(&shard, b"p", b"e0").expect("append"),
        0,
    );
    assert_eq!(
        client.stream_append(&shard, b"p", b"e1").expect("append"),
        1,
    );
    assert_eq!(client.stream_trim(&shard, 0).expect("trim"), 1);
    let page = client.stream_read(&shard, 0, 16).expect("read");
    assert_eq!(page.next_offset, 2);
    assert_eq!(page.entries.len(), 1);
    assert_eq!(page.entries[0].offset, 1);
    assert_eq!(
        client.stream_append(&shard, b"p", b"e2").expect("append"),
        2,
        "trimmed offsets are not reused",
    );
}

#[test]
fn atomic_batch_commits_and_rejects_over_wire() {
    let engine = start_ephemeral();
    let client = client_for(&engine);
    let writes = vec![
        BatchWriteSpec {
            key: b"bb-t1".to_vec(),
            kind: BatchWriteKind::Put(b"v1".to_vec()),
            expect: BatchExpect::Absent,
        },
        BatchWriteSpec {
            key: b"bb-t2".to_vec(),
            kind: BatchWriteKind::Put(b"v2".to_vec()),
            expect: BatchExpect::Absent,
        },
    ];
    let AtomicBatchResult { versions, .. } = client.atomic_batch(NS, &writes).expect("commit");
    assert_eq!(versions.len(), 2);
    assert!(versions.iter().all(Option::is_some));
    assert_eq!(
        client.get(&key("bb-t1")).expect("read").as_deref(),
        Some(b"v1".as_slice()),
    );
    assert!(matches!(
        client.atomic_batch(NS, &[]),
        Err(ClientError::InvalidRequest),
    ));
    let huge: Vec<BatchWriteSpec> = (0..300)
        .map(|index| BatchWriteSpec {
            key: format!("bb-huge-{index}").into_bytes(),
            kind: BatchWriteKind::Put(b"x".to_vec()),
            expect: BatchExpect::Any,
        })
        .collect();
    assert!(matches!(
        client.atomic_batch(NS, &huge),
        Err(ClientError::TxnTooLarge),
    ));
}

#[test]
fn wrong_type_surfaces_typed_over_wire() {
    let engine = start_ephemeral();
    let client = client_for(&engine);
    let key = key("bb-bytes");
    client.set(&key, Bytes::from_static(b"plain")).expect("set");
    assert!(matches!(
        client.commutative_add(&key, 1),
        Err(ClientError::WrongType),
    ));
}
