//! Streaming-upload/download integration over real loopback TCP:
//! engines start in-process on ephemeral ports, and the native client
//! drives `put_stream`/`get_stream` plus `set_range` through the reactor
//! paths (never the embedded channel short-circuit).

use bytes::Bytes;
use kivi_client::{ClientConfig, ClientError, NativeClient};
use kivi_engine::{
    BatchPolicy, CheckpointConfig, ChunkFabricConfig, ConnLimits, DurabilityMode, DurableConfig,
    EngineConfig, EngineNetwork, LocalEngine, Placement, TurnBudget,
};
use kivi_state::Key;
use kivi_tablet::{DirectorySnapshot, HashPrefix, PartitionRange};
use kivi_types::{ClusterId, NamespaceId, NodeId, NodeIncarnation, WorkerId};

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

fn placement() -> Placement {
    Placement::new([(TabletId::from_u64(1), WorkerId::from_u64(0))])
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
        incarnation: NodeIncarnation::INITIAL,
        conn: ConnLimits::default(),
        turn: TurnBudget::default(),
    }
}

fn start_ephemeral(chunks: ChunkFabricConfig) -> LocalEngine {
    LocalEngine::start(EngineConfig {
        namespace: NS,
        directory: directory(),
        placement: placement(),
        worker_count: 2,
        request_capacity: 128,
        chunks,
        network: Some(network()),
        durability: DurabilityMode::Ephemeral,
    })
    .expect("networked engine starts")
}

fn start_durable(dir: &std::path::Path) -> LocalEngine {
    let opened = kivi_durability::open_data_dir(dir).expect("data dir opens");
    LocalEngine::start(EngineConfig {
        namespace: NS,
        directory: directory(),
        placement: placement(),
        worker_count: 2,
        request_capacity: 128,
        chunks: ChunkFabricConfig::default(),
        network: Some(network()),
        durability: DurabilityMode::Durable(DurableConfig {
            data_dir: dir.to_owned(),
            segment_target_bytes: 1024 * 1024,
            node: opened.meta.node,
            cluster: opened.meta.cluster,
            incarnation: opened.meta.incarnation,
            shared_wal: false,
            batch: BatchPolicy::default_policy(),
            checkpoint: CheckpointConfig::default_config(),
        }),
    })
    .expect("durable networked engine starts")
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

/// Deterministic pseudo-random bytes (splitmix64): representative,
/// incompressible-ish content — never zeros-only.
fn pattern_bytes(len: usize, seed: u64) -> Bytes {
    let mut state = seed;
    let mut out = Vec::with_capacity(len);
    while out.len() < len {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        out.extend_from_slice(&z.to_le_bytes());
    }
    out.truncate(len);
    Bytes::from(out)
}

use kivi_types::TabletId;

#[test]
fn upload_download_round_trip_over_the_wire() {
    let engine = start_ephemeral(ChunkFabricConfig::default());
    let client = client_for(&engine);
    // Multi-chunk upload (2.5 MiB at 1 MiB chunks): the client never
    // holds more than one frame, the server stages incrementally.
    let big = pattern_bytes(2_500_000, 0x57EA);
    let mut source = std::io::Cursor::new(big.to_vec());
    client
        .put_stream(&Key::from("big"), Some(big.len() as u64), &mut source)
        .expect("upload");
    // Plain reads see chunked roots transparently.
    assert_eq!(
        client.get(&Key::from("big")).expect("get"),
        Some(big.clone()),
        "uploaded bytes read back through plain get"
    );
    // Streamed reads fan back out frame by frame, byte-exact.
    assert_eq!(
        client.get_stream(&Key::from("big")).expect("download"),
        Some(big.clone()),
        "uploaded bytes stream back exactly"
    );
    // Small values stream too (uniform client path, no special case).
    client
        .set(&Key::from("small"), Bytes::from_static(b"s"))
        .expect("set");
    assert_eq!(
        client.get_stream(&Key::from("small")).expect("download"),
        Some(Bytes::from_static(b"s"))
    );
    // Absent keys stream to `None`, like `get`.
    assert_eq!(
        client.get_stream(&Key::from("missing")).expect("download"),
        None
    );
    engine.shutdown().expect("clean shutdown");
}

#[test]
fn upload_declared_mismatch_aborts_without_committing() {
    let engine = start_ephemeral(ChunkFabricConfig::default());
    let client = client_for(&engine);
    // Declared 10 bytes, delivering 5: the client fails fast locally.
    let mut source = std::io::Cursor::new(b"short".to_vec());
    assert!(matches!(
        client.put_stream(&Key::from("k"), Some(10), &mut source),
        Err(ClientError::InvalidRequest)
    ));
    // Declared 3, delivering 5: fails fast before committing.
    let mut source = std::io::Cursor::new(b"toolong".to_vec());
    assert!(matches!(
        client.put_stream(&Key::from("k"), Some(3), &mut source),
        Err(ClientError::InvalidRequest)
    ));
    assert_eq!(client.get(&Key::from("k")).expect("get"), None);
    // Unknown-length uploads commit whatever the source yields.
    let mut source = std::io::Cursor::new(b"exact".to_vec());
    client
        .put_stream(&Key::from("k"), None, &mut source)
        .expect("upload");
    assert_eq!(
        client.get(&Key::from("k")).expect("get"),
        Some(Bytes::from_static(b"exact"))
    );
    engine.shutdown().expect("clean shutdown");
}

#[test]
fn uploads_multiplex_with_point_ops_on_one_client() {
    let engine = start_ephemeral(ChunkFabricConfig::default());
    let client = client_for(&engine);
    // A large upload alongside ordinary traffic on the same pooled
    // client: stream frames interleave with requests by id, and neither
    // stalls the other past framing.
    let big = pattern_bytes(1_500_000, 0x1A01);
    let mut source = std::io::Cursor::new(big.to_vec());
    client
        .put_stream(&Key::from("big"), Some(big.len() as u64), &mut source)
        .expect("upload");
    for i in 0..50u64 {
        let key = Key::from(format!("point:{i}"));
        client.set(&key, Bytes::from(format!("v{i}"))).expect("set");
    }
    for i in 0..50u64 {
        let key = Key::from(format!("point:{i}"));
        assert_eq!(
            client.get(&key).expect("get"),
            Some(Bytes::from(format!("v{i}")))
        );
    }
    assert_eq!(
        client.get_stream(&Key::from("big")).expect("download"),
        Some(big)
    );
    engine.shutdown().expect("clean shutdown");
}

#[test]
fn set_range_over_the_wire_patches_and_restages() {
    let engine = start_ephemeral(ChunkFabricConfig::default());
    let client = client_for(&engine);
    client
        .set(&Key::from("k"), Bytes::from_static(b"hello world"))
        .expect("set");
    client
        .set_range(&Key::from("k"), 6, Bytes::from_static(b"PLANET"))
        .expect("patch");
    assert_eq!(
        client.get(&Key::from("k")).expect("get"),
        Some(Bytes::from_static(b"hello PLANET"))
    );
    // Chunked bases patch through the lane splice over the wire too.
    let big = pattern_bytes(1_200_000, 0x5E7);
    client.set(&Key::from("big"), big.clone()).expect("set");
    let patch = pattern_bytes(1000, 0x9A7);
    let at = 1_100_000usize;
    client
        .set_range(&Key::from("big"), at as u64, patch.clone())
        .expect("patch");
    let mut expected = big.to_vec();
    expected[at..at + patch.len()].copy_from_slice(&patch);
    assert_eq!(
        client.get(&Key::from("big")).expect("get"),
        Some(Bytes::from(expected))
    );
    engine.shutdown().expect("clean shutdown");
}

fn raw_handshake(addr: &str) -> (std::net::TcpStream, kivi_protocol::FrameReader) {
    use kivi_protocol::{Capabilities, ClientHello, FrameKind, FrameReader, encode_frame};
    use std::io::Write as _;
    let mut socket = std::net::TcpStream::connect(addr).expect("connect");
    socket
        .set_read_timeout(Some(std::time::Duration::from_secs(10)))
        .expect("timeout");
    let hello = ClientHello {
        max_frame: u32::try_from(kivi_protocol::DEFAULT_MAX_FRAME).expect("frame bound fits"),
        required_caps: Capabilities::BASE_V1,
        optional_caps: Capabilities::empty(),
        desired_worker: None,
    };
    socket
        .write_all(&encode_frame(FrameKind::ClientHello, 0, &hello.encode()))
        .expect("hello");
    let mut reader = FrameReader::new(kivi_protocol::DEFAULT_MAX_FRAME);
    let reply = read_raw(&mut socket, &mut reader).expect("server hello");
    assert_eq!(reply.kind, FrameKind::ServerHello);
    let hello = kivi_protocol::ServerHello::decode(&reply.payload).expect("hello decodes");
    assert!(
        hello.caps.contains(Capabilities::STREAMING),
        "servers advertise streaming"
    );
    (socket, reader)
}

fn read_raw(
    socket: &mut std::net::TcpStream,
    reader: &mut kivi_protocol::FrameReader,
) -> std::io::Result<kivi_protocol::Frame> {
    use std::io::Read as _;
    let mut chunk = vec![0u8; 65536];
    loop {
        let count = socket.read(&mut chunk)?;
        if count == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "eof",
            ));
        }
        let frames = reader.push(&chunk[..count]).map_err(|error| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, format!("{error:?}"))
        })?;
        if let Some(frame) = frames.into_iter().next() {
            return Ok(frame);
        }
    }
}

#[test]
fn server_aborts_over_cap_data_and_keeps_serving() {
    use kivi_protocol::{FrameKind, StreamBegin, StreamReady, encode_frame};
    use std::io::Write as _;
    let engine = start_ephemeral(ChunkFabricConfig::default());
    let (mut socket, mut reader) = raw_handshake(&engine.worker_addrs()[0].to_string());
    // Open a stream and learn the negotiated cap.
    let begin = StreamBegin {
        namespace: NS,
        key: b"raw:1".to_vec(),
        total_len: None,
        identity: None,
        ack_floor: kivi_types::RequestSeq::from_u64(0),
    };
    socket
        .write_all(&encode_frame(FrameKind::StreamBegin, 1, &begin.encode()))
        .expect("begin");
    let ready = read_raw(&mut socket, &mut reader).expect("ready");
    assert_eq!(ready.kind, FrameKind::StreamReady);
    assert_eq!(ready.request_id, 1);
    let cap = StreamReady::decode(&ready.payload)
        .expect("ready decodes")
        .max_data;
    // One byte past the cap aborts the stream in-band (framing still
    // allows the frame: the cap is one chunk, far below max_frame).
    socket
        .write_all(&encode_frame(
            FrameKind::StreamData,
            1,
            &vec![7u8; cap as usize + 1],
        ))
        .expect("over-cap data");
    let abort = read_raw(&mut socket, &mut reader).expect("abort");
    assert_eq!(abort.kind, FrameKind::StreamAbort);
    assert_eq!(abort.request_id, 1);
    // The connection survives: a commit for the dead stream echoes an
    // abort (lenient for client/server races), and ordinary traffic
    // flows on the same connection.
    socket
        .write_all(&encode_frame(FrameKind::StreamCommit, 1, &[]))
        .expect("commit");
    let echo = read_raw(&mut socket, &mut reader).expect("abort echo");
    assert_eq!(echo.kind, FrameKind::StreamAbort);
    let client = client_for(&engine);
    assert_eq!(client.get(&Key::from("raw:1")).expect("get"), None);
    client
        .set(&Key::from("raw:1"), Bytes::from_static(b"alive"))
        .expect("set");
    assert_eq!(
        client.get(&Key::from("raw:1")).expect("get"),
        Some(Bytes::from_static(b"alive"))
    );
    engine.shutdown().expect("clean shutdown");
}

#[test]
fn streamed_upload_survives_durable_restart() {
    let scratch = tempfile::tempdir().expect("scratch");
    let engine = start_durable(scratch.path());
    let client = client_for(&engine);
    let big = pattern_bytes(2_200_000, 0xD06E);
    let mut source = std::io::Cursor::new(big.to_vec());
    client
        .put_stream(&Key::from("big"), Some(big.len() as u64), &mut source)
        .expect("upload");
    assert_eq!(
        client.get_stream(&Key::from("big")).expect("download"),
        Some(big.clone())
    );
    engine.shutdown().expect("clean shutdown");
    // The streamed commit is an ordinary small WAL root over durable
    // packs: restart recovers the bytes exactly, then serves streams.
    let engine = start_durable(scratch.path());
    let client = client_for(&engine);
    assert_eq!(
        client.get_stream(&Key::from("big")).expect("download"),
        Some(big.clone())
    );
    assert_eq!(client.get(&Key::from("big")).expect("get"), Some(big));
    engine.shutdown().expect("clean shutdown");
}
