//! Real loopback TCP integration tests: server engine plus native client.
//!
//! Engines start in-process on ephemeral ports (no binaries involved), and
//! every test shuts its engine down. These prove the wire, routing,
//! redirect, lifecycle, and isolation behavior — not the simulator.

use std::io::{Read, Write};
use std::net::TcpStream;
use std::sync::{Arc, Barrier};
use std::thread;
use std::time::Duration;

use bytes::Bytes;
use kivi_client::{ClientConfig, NativeClient};
use kivi_engine::{ConnLimits, EngineConfig, EngineNetwork, LocalEngine, Placement, TurnBudget};
use kivi_state::Key;
use kivi_tablet::{DirectorySnapshot, HashPrefix, PartitionRange};
use kivi_types::{
    ClusterId, NamespaceId, NodeId, TabletEpoch, TabletId, UnixMicros, WorkerId,
    WriteGuardGeneration,
};

const NS: NamespaceId = NamespaceId::from_u64(1);
const EPOCH: TabletEpoch = TabletEpoch::INITIAL;
const GUARD: WriteGuardGeneration = WriteGuardGeneration::INITIAL;

fn hash_range(bits: u128, len: u8) -> PartitionRange {
    PartitionRange::Hash(HashPrefix::new(bits, len).expect("range"))
}

fn activate(snapshot: &DirectorySnapshot, tablet: TabletId) -> DirectorySnapshot {
    snapshot
        .stage(tablet)
        .and_then(|s| s.activate(tablet))
        .expect("stage+activate")
}

fn split_snapshot() -> DirectorySnapshot {
    let genesis =
        DirectorySnapshot::bootstrap(NS, TabletId::from_u64(1), hash_range(0, 0), EPOCH, GUARD)
            .expect("genesis");
    let mut dir = activate(&genesis, TabletId::from_u64(1));
    for (id, bits) in [(2u64, 0u128), (3u64, 1u128 << 127)] {
        dir = dir
            .allocate(TabletId::from_u64(id), hash_range(bits, 1), EPOCH, GUARD)
            .and_then(|s| s.stage(TabletId::from_u64(id)))
            .expect("child staged");
    }
    dir = dir.seal(TabletId::from_u64(1)).expect("seal");
    dir = dir.activate(TabletId::from_u64(2)).expect("activate 2");
    dir = dir.activate(TabletId::from_u64(3)).expect("activate 3");
    dir.retire(
        TabletId::from_u64(1),
        kivi_tablet::Redirect::new(vec![TabletId::from_u64(2), TabletId::from_u64(3)]),
    )
    .expect("retire")
}

fn network() -> EngineNetwork {
    EngineNetwork {
        base_port: 0,
        ports: Vec::new(),
        bind_ip: [127, 0, 0, 1].into(),
        max_frame: kivi_protocol::DEFAULT_MAX_FRAME,
        affinity: kivi_engine::AffinityMode::Disabled,
        node_id: NodeId::from_u64(1),
        cluster_id: ClusterId::from_u128(1),
        conn: ConnLimits::default(),
        turn: TurnBudget::default(),
    }
}

/// Restarts an engine on the exact ports of a previous one (TIME_WAIT-safe
/// through full shutdown plus `SO_REUSEADDR`).
fn restart_on_ports(ports: &[u16], placement: Placement) -> LocalEngine {
    LocalEngine::start(EngineConfig {
        namespace: NS,
        directory: split_snapshot(),
        placement,
        worker_count: 2,
        request_capacity: 128,
        network: Some(EngineNetwork {
            ports: ports.to_vec(),
            ..network()
        }),
    })
    .expect("restart binds same ports")
}

fn start_split() -> LocalEngine {
    LocalEngine::start(EngineConfig {
        namespace: NS,
        directory: split_snapshot(),
        placement: Placement::new([
            (TabletId::from_u64(2), WorkerId::from_u64(0)),
            (TabletId::from_u64(3), WorkerId::from_u64(1)),
        ]),
        worker_count: 2,
        request_capacity: 128,
        network: Some(network()),
    })
    .expect("networked engine starts")
}

fn client_for(engine: &LocalEngine) -> NativeClient {
    let seed = engine.worker_addrs()[0].to_string();
    NativeClient::new(ClientConfig {
        seeds: vec![seed],
        namespace: NS,
        ..ClientConfig::default()
    })
    .expect("client builds")
}

/// Finds keys routed to each of the two tablets (deterministic probing).
fn probe_keys(engine: &LocalEngine) -> (Key, Key) {
    let mut found: [Option<Key>; 2] = [None, None];
    for i in 0..10_000u64 {
        let key = Key::from(format!("probe:{i}"));
        let (tablet, _) = engine.route_key(&key).expect("covered");
        let slot = usize::from(tablet != TabletId::from_u64(2));
        if found[slot].is_none() {
            found[slot] = Some(key);
        }
        if found.iter().all(Option::is_some) {
            break;
        }
    }
    (
        found[0].clone().expect("tablet 2 key"),
        found[1].clone().expect("tablet 3 key"),
    )
}

#[test]
fn full_crud_over_tcp() {
    let engine = start_split();
    let client = client_for(&engine);
    let key = Key::from("user:1");
    assert_eq!(client.get(&key).expect("get"), None);
    client.set(&key, Bytes::from_static(b"ada")).expect("set");
    assert_eq!(
        client.get(&key).expect("get"),
        Some(Bytes::from_static(b"ada"))
    );
    assert!(client.exists(&key).expect("exists"));
    assert_eq!(client.counter_add(&Key::from("n"), 5).expect("add"), 5);
    assert_eq!(client.counter_add(&Key::from("n"), -2).expect("add"), 3);
    assert_eq!(client.counter_get(&Key::from("n")).expect("read"), Some(3));
    assert!(client.counter_get(&key).is_err(), "bytes are not counters");
    let far = UnixMicros::from_micros(u64::MAX / 2);
    assert!(client.expire_at(&key, far).expect("expire"));
    assert!(client.get(&key).expect("live").is_some());
    assert!(client.persist_expiry(&key).expect("persist"));
    assert!(client.delete(&key).expect("delete"));
    assert!(!client.delete(&key).expect("re-delete"));
    assert_eq!(client.get(&key).expect("gone"), None);
    engine.shutdown().expect("clean shutdown");
}

#[test]
fn route_miss_redirects_then_goes_direct() {
    let engine = start_split();
    let client = client_for(&engine);
    assert_eq!(client.cached_routes(), 0);
    let (key_a, key_b) = probe_keys(&engine);
    // key_a lives on the seed worker: served with no redirect, nothing to
    // learn. key_b lives elsewhere: exactly one seed redirect teaches it.
    client.set(&key_a, Bytes::from_static(b"a")).expect("set a");
    assert_eq!(client.cached_routes(), 0, "seed-owned range needs no route");
    client.set(&key_b, Bytes::from_static(b"b")).expect("set b");
    assert_eq!(client.cached_routes(), 1, "redirected range learned");
    assert_eq!(client.stats().redirects, 1);
    // Steady state: cached routes serve with no further redirects.
    for _ in 0..10 {
        assert_eq!(
            client.get(&key_a).expect("get a"),
            Some(Bytes::from_static(b"a"))
        );
        assert_eq!(
            client.get(&key_b).expect("get b"),
            Some(Bytes::from_static(b"b"))
        );
    }
    assert_eq!(client.stats().redirects, 1, "no new redirects once cached");
    engine.shutdown().expect("clean shutdown");
}

#[test]
fn swapped_placement_relearns_through_redirects() {
    let engine = start_split();
    let ports: Vec<u16> = engine
        .worker_addrs()
        .iter()
        .map(std::net::SocketAddr::port)
        .collect();
    let client = client_for(&engine);
    // Prime with the non-seed tablet's key: the seed redirect teaches a
    // route (seed-owned keys serve with no redirect and teach nothing).
    let (_, key_b) = probe_keys(&engine);
    client
        .set(&key_b, Bytes::from_static(b"v"))
        .expect("prime cache");
    let learned = client.cached_routes();
    assert!(learned >= 1, "route learned from seed redirect");
    let redirects_before = client.stats().redirects;
    engine.shutdown().expect("first engine down");
    // Same ports, swapped tablet→worker placement, fresh empty state. The
    // client's cached endpoint now reaches the wrong worker, which must
    // redirect (not serve, not proxy) to the new owner.
    let swapped = restart_on_ports(
        &ports,
        Placement::new([
            (TabletId::from_u64(2), WorkerId::from_u64(1)),
            (TabletId::from_u64(3), WorkerId::from_u64(0)),
        ]),
    );
    client
        .set(&key_b, Bytes::from_static(b"v2"))
        .expect("relearned write");
    assert_eq!(
        client.get(&key_b).expect("relearned read"),
        Some(Bytes::from_static(b"v2"))
    );
    assert!(
        client.stats().redirects > redirects_before,
        "stale cache caused fresh redirects"
    );
    swapped.shutdown().expect("clean shutdown");
}

#[test]
fn close_reconnect_across_server_restart() {
    let engine = start_split();
    let ports: Vec<u16> = engine
        .worker_addrs()
        .iter()
        .map(std::net::SocketAddr::port)
        .collect();
    let client = client_for(&engine);
    client
        .set(&Key::from("k"), Bytes::from_static(b"v"))
        .expect("set");
    let report = engine.shutdown().expect("first engine down");
    assert_eq!(report.workers_joined, 2);
    // Restart on the same ports with identical layout.
    let engine = restart_on_ports(
        &ports,
        Placement::new([
            (TabletId::from_u64(2), WorkerId::from_u64(0)),
            (TabletId::from_u64(3), WorkerId::from_u64(1)),
        ]),
    );
    // Pooled connections died with the first engine; the client redials and
    // the same logical identity continues on fresh (empty) state.
    assert_eq!(client.get(&Key::from("k")).expect("get"), None);
    client
        .set(&Key::from("k"), Bytes::from_static(b"v2"))
        .expect("set");
    assert_eq!(
        client.get(&Key::from("k")).expect("get"),
        Some(Bytes::from_static(b"v2"))
    );
    engine.shutdown().expect("clean shutdown");
}

#[test]
fn malformed_client_is_isolated() {
    let engine = start_split();
    let addr = engine.worker_addrs()[0].to_string();
    let client = client_for(&engine);
    // Raw socket speaking garbage: the server must close it cleanly.
    let mut socket = TcpStream::connect(&addr).expect("connect");
    socket
        .set_read_timeout(Some(Duration::from_secs(5)))
        .expect("timeout");
    socket
        .write_all(b"\x00\x01garbage-that-is-not-kvnp")
        .expect("write");
    let mut buf = [0u8; 64];
    assert_eq!(
        socket.read(&mut buf).expect("read"),
        0,
        "server closed garbage connection"
    );
    // Healthy clients are unaffected.
    client
        .set(&Key::from("k"), Bytes::from_static(b"v"))
        .expect("set");
    assert_eq!(
        client.get(&Key::from("k")).expect("get"),
        Some(Bytes::from_static(b"v"))
    );
    engine.shutdown().expect("clean shutdown");
}

#[test]
fn pipelined_raw_socket_round_trips_in_order() {
    use kivi_protocol::{FrameKind, Opcode, Request, encode_frame};
    let engine = start_split();
    let addr = engine.worker_addrs()[0].to_string();
    let mut socket = TcpStream::connect(&addr).expect("connect");
    socket
        .set_read_timeout(Some(Duration::from_secs(10)))
        .expect("timeout");
    // Handshake first.
    let hello = kivi_protocol::ClientHello {
        max_frame: u32::try_from(kivi_protocol::DEFAULT_MAX_FRAME).expect("frame bound fits"),
        required_caps: kivi_protocol::Capabilities::BASE_V1,
        optional_caps: kivi_protocol::Capabilities::empty(),
        desired_worker: None,
    };
    socket
        .write_all(&encode_frame(FrameKind::ClientHello, 0, &hello.encode()))
        .expect("hello");
    let mut reader = kivi_protocol::FrameReader::new(kivi_protocol::DEFAULT_MAX_FRAME);
    let reply = read_frame(&mut socket, &mut reader).expect("server hello");
    assert_eq!(reply.kind, FrameKind::ServerHello);
    // Pipeline three requests without waiting between them.
    for id in 10u64..13 {
        let request = Request {
            namespace: NS,
            opcode: Opcode::Set,
            hint: None,
            key: format!("pipe:{id}").into_bytes(),
            value: Some(b"x".to_vec()),
            delta: 0,
            expiry: 0,
        };
        socket
            .write_all(&encode_frame(FrameKind::Request, id, &request.encode()))
            .expect("write");
    }
    // Responses arrive in execution order with matching ids.
    let mut ids = Vec::new();
    for _ in 0..3 {
        let frame = read_frame(&mut socket, &mut reader).expect("response");
        assert_eq!(frame.kind, FrameKind::Response);
        ids.push(frame.request_id);
    }
    assert_eq!(ids, vec![10, 11, 12]);
    engine.shutdown().expect("clean shutdown");
}

fn read_frame(
    socket: &mut TcpStream,
    reader: &mut kivi_protocol::FrameReader,
) -> std::io::Result<kivi_protocol::Frame> {
    let mut chunk = vec![0u8; 4096];
    loop {
        let count = socket.read(&mut chunk)?;
        if count == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "closed",
            ));
        }
        let frames = reader.push(&chunk[..count]).map_err(|error| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, error.to_string())
        })?;
        if let Some(frame) = frames.into_iter().next() {
            return Ok(frame);
        }
    }
}

#[test]
fn direct_path_bypasses_the_channel() {
    let engine = start_split();
    let client = client_for(&engine);
    let (key_a, key_b) = probe_keys(&engine);
    client.set(&key_a, Bytes::from_static(b"a")).expect("set");
    client.set(&key_b, Bytes::from_static(b"b")).expect("set");
    // All native traffic executed inline on owners: zero channel ops.
    let mut direct = 0u64;
    let mut channel = 0u64;
    for (_, metrics) in engine.worker_metrics() {
        direct += metrics.direct_ops;
        channel += metrics.channel_ops;
    }
    assert!(direct >= 2, "native requests executed directly");
    assert_eq!(channel, 0, "native path must not touch the channel queue");
    // The embedded path still works and counts separately.
    let local = engine.client();
    local
        .set(&key_a, Bytes::from_static(b"c"))
        .expect("local set");
    let mut channel_after = 0u64;
    for (_, metrics) in engine.worker_metrics() {
        channel_after += metrics.channel_ops;
    }
    assert_eq!(channel_after, 1);
    engine.shutdown().expect("clean shutdown");
}

#[test]
fn concurrent_native_counter_is_exact() {
    let engine = start_split();
    // Exercise overload admission with a tiny queue plus hammering clients.
    let key = Key::from("hot");
    let barrier = Arc::new(Barrier::new(9));
    let mut threads = Vec::new();
    for _ in 0..8 {
        let client = client_for(&engine);
        let gate = Arc::clone(&barrier);
        let key = key.clone();
        threads.push(thread::spawn(move || {
            gate.wait();
            for _ in 0..200 {
                loop {
                    match client.counter_add(&key, 1) {
                        Ok(_) => break,
                        Err(kivi_client::ClientError::Overloaded) => {
                            thread::yield_now();
                        }
                        Err(error) => panic!("unexpected client error: {error:?}"),
                    }
                }
            }
            client.stats().requests
        }));
    }
    barrier.wait();
    let mut issued = 0u64;
    for thread in threads {
        issued += thread.join().expect("client thread joins");
    }
    assert!(issued >= 1600, "every logical op was issued");
    let client = client_for(&engine);
    assert_eq!(client.counter_get(&key).expect("read"), Some(8 * 200));
    engine.shutdown().expect("clean shutdown");
}
