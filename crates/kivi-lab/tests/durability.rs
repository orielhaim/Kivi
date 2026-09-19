//! Crash-recoverable durability through real server processes.
//!
//! These spawn the actual `kivi-server` binary, `kill` it without any
//! graceful shutdown (the honest crash), restart it on the same `--data-dir`
//! with fresh ports, and prove acknowledged writes come back.
//! The lost-response half of exactly-once is covered deterministically:
//! the same `(session, seq)` identity replayed — even after a kill and
//! restart, even on a fresh connection — returns the original outcome
//! without re-executing.

use std::collections::HashSet;
use std::io::{Read, Write};
use std::net::TcpStream;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use std::time::{Duration, Instant};

use kivi_client::{ClientConfig, NativeClient};
use kivi_lab::process::Server;
use kivi_state::Key;
use kivi_types::{NamespaceId, RequestIdentity, RequestSeq, SessionId};

const NS: NamespaceId = NamespaceId::from_u64(1);

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

/// Raw socket with the durable-dedup capability negotiated.
struct RawDurable {
    socket: TcpStream,
    reader: kivi_protocol::FrameReader,
}

impl RawDurable {
    fn connect(endpoint: &str) -> Self {
        use kivi_protocol::{Capabilities, FrameKind};
        let mut socket = TcpStream::connect(endpoint).expect("connect");
        socket
            .set_read_timeout(Some(Duration::from_secs(10)))
            .expect("timeout");
        let hello = kivi_protocol::ClientHello {
            max_frame: u32::try_from(kivi_protocol::DEFAULT_MAX_FRAME).expect("fits"),
            required_caps: Capabilities::BASE_V1,
            optional_caps: Capabilities::DURABLE_MUTATION_DEDUP,
            desired_worker: None,
        };
        socket
            .write_all(&kivi_protocol::encode_frame(
                FrameKind::ClientHello,
                0,
                &hello.encode(),
            ))
            .expect("hello");
        let mut reader = kivi_protocol::FrameReader::new(kivi_protocol::DEFAULT_MAX_FRAME);
        let reply = read_frame(&mut socket, &mut reader).expect("server hello");
        assert_eq!(reply.kind, FrameKind::ServerHello);
        let server = kivi_protocol::ServerHello::decode(&reply.payload).expect("hello decodes");
        assert!(
            server.caps.contains(Capabilities::DURABLE_MUTATION_DEDUP),
            "server must advertise durable dedup, got {:?}",
            server.caps
        );
        Self { socket, reader }
    }

    fn counter_add(
        &mut self,
        request_id: u64,
        session: SessionId,
        seq: u64,
        ack: u64,
        key: &str,
    ) -> kivi_protocol::Response {
        use kivi_protocol::{FrameKind, Opcode, Request};
        let request = Request {
            contract: kivi_types::ReadContract::Latest,
            namespace: NS,
            opcode: Opcode::CounterAdd,
            hint: None,
            key: key.as_bytes().to_vec(),
            value: None,
            delta: 1,
            expiry: 0,
            offset: 0,
            len: 0,
            condition: kivi_protocol::COND_ALWAYS,
            expiry_policy: kivi_protocol::EXPIRY_CLEAR,
            identity: Some(RequestIdentity::new(session, RequestSeq::from_u64(seq))),
            ack_floor: RequestSeq::from_u64(ack),
            scan_start: None,
            scan_end: None,
            scan_direction: kivi_protocol::SCAN_FORWARD,
            scan_max_items: 0,
            scan_max_bytes: 0,
            scan_projection: kivi_protocol::SCAN_KEYS_ONLY,
            scan_consistency: kivi_protocol::SCAN_LATEST_PER_TABLET,
            batch_txn: [0u8; 16],
            batch_writes: Vec::new(),
            txn_coordinator: 0,
            txn_commit: false,
            txn_digest: [0u8; 32],
            capacity: 0,
            holder: 0,
            permit: [0u8; 16],
            owner: 0,
            qty: 0,
            fencing: 0,
            ttl: 0,
            stream: [0u8; 16],
            shard: 0,
            partition: Vec::new(),
            share_min: 0,
            share_max: 0,
        };
        self.socket
            .write_all(&kivi_protocol::encode_frame(
                FrameKind::Request,
                request_id,
                &request.encode(),
            ))
            .expect("write");
        let frame = read_frame(&mut self.socket, &mut self.reader).expect("response");
        assert_eq!(frame.kind, FrameKind::Response);
        assert_eq!(frame.request_id, request_id);
        let (response, _) =
            kivi_protocol::Response::decode(&frame.payload).expect("response decodes");
        response
    }
}

/// Raw streaming-upload driver: begin → data frames → commit. Mirrors
/// exactly what `NativeClient::put_stream` sends, with a caller-chosen
/// identity for replay tests.
fn raw_upload_begin(
    raw: &mut RawDurable,
    stream: u64,
    session: SessionId,
    seq: u64,
    key: &str,
    total: Option<u64>,
) -> u32 {
    use kivi_protocol::{FrameKind, StreamBegin, StreamReady};
    let begin = StreamBegin {
        namespace: NS,
        key: key.as_bytes().to_vec(),
        total_len: total,
        identity: Some(RequestIdentity::new(session, RequestSeq::from_u64(seq))),
        ack_floor: RequestSeq::from_u64(0),
    };
    raw.socket
        .write_all(&kivi_protocol::encode_frame(
            FrameKind::StreamBegin,
            stream,
            &begin.encode(),
        ))
        .expect("begin");
    let frame = read_frame(&mut raw.socket, &mut raw.reader).expect("ready");
    assert_eq!(frame.kind, FrameKind::StreamReady);
    assert_eq!(frame.request_id, stream);
    StreamReady::decode(&frame.payload)
        .expect("ready decodes")
        .max_data
}

fn raw_upload_data(raw: &mut RawDurable, stream: u64, bytes: &[u8]) {
    raw.socket
        .write_all(&kivi_protocol::encode_frame(
            kivi_protocol::FrameKind::StreamData,
            stream,
            bytes,
        ))
        .expect("data");
}

fn raw_upload_commit(raw: &mut RawDurable, stream: u64) -> kivi_protocol::Response {
    raw.socket
        .write_all(&kivi_protocol::encode_frame(
            kivi_protocol::FrameKind::StreamCommit,
            stream,
            &[],
        ))
        .expect("commit");
    let frame = read_frame(&mut raw.socket, &mut raw.reader).expect("commit response");
    assert_eq!(frame.kind, kivi_protocol::FrameKind::Response);
    assert_eq!(frame.request_id, stream);
    kivi_protocol::Response::decode(&frame.payload)
        .expect("response decodes")
        .0
}

fn stored_version(response: &kivi_protocol::Response) -> u64 {
    assert_eq!(
        response.status,
        kivi_protocol::Status::Ok,
        "expected Ok, got {:?}",
        response.status
    );
    match &response.body {
        kivi_protocol::ResponseBody::Stored { version } => *version,
        other => panic!("expected stored outcome, got {other:?}"),
    }
}

fn counter_value(response: &kivi_protocol::Response) -> i64 {
    assert_eq!(
        response.status,
        kivi_protocol::Status::Ok,
        "expected Ok, got {:?}",
        response.status
    );
    match &response.body {
        kivi_protocol::ResponseBody::CounterUpdated { value, .. } => *value,
        other => panic!("expected counter outcome, got {other:?}"),
    }
}

/// Deterministic pseudo-random bytes for stream bodies (small helper;
/// the engine-side pattern generator is not reachable from this crate).
fn stream_bytes(len: usize, seed: u64) -> Vec<u8> {
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
    out
}

#[test]
fn streamed_upload_replay_after_kill_returns_original_without_reexecuting() {
    let scratch = tempfile::tempdir().expect("scratch");
    let server = Server::spawn_auto(scratch.path(), &[]).expect("server spawns");
    let endpoint = server.endpoint();
    // Fixed identity across three commits of the same bytes: the "lost
    // responses". Every replay must dedup-hit the first commit's version,
    // never store twice.
    let session = SessionId::from_u128(0x0057_EA11);
    let body = stream_bytes(1_500_000, 0xCAFE);
    let commit_once = |raw: &mut RawDurable, stream: u64| {
        let max_data = raw_upload_begin(raw, stream, session, 1, "money", Some(body.len() as u64));
        for piece in body.chunks(max_data as usize) {
            raw_upload_data(raw, stream, piece);
        }
        raw_upload_commit(raw, stream)
    };
    let mut raw = RawDurable::connect(&endpoint);
    let first = commit_once(&mut raw, 100);
    let version = stored_version(&first);
    // Same identity on the live server: dedup hit, same version.
    let replay = commit_once(&mut raw, 101);
    assert_eq!(stored_version(&replay), version, "live replay dedups");
    drop(raw);
    server.kill();
    // After the crash the dedup memory is gone — but the WAL remembers
    // the small root, so the replay still dedups instead of storing anew.
    let server = Server::spawn_auto(scratch.path(), &[]).expect("server spawns");
    let mut raw = RawDurable::connect(&server.endpoint());
    let after_crash = commit_once(&mut raw, 102);
    assert_eq!(
        stored_version(&after_crash),
        version,
        "replayed stream identity must not re-execute after recovery"
    );
    drop(raw);
    // And the bytes are the committed ones, exactly once.
    let client = server.client();
    assert_eq!(
        client.get(&Key::from("money")).expect("get"),
        Some(bytes::Bytes::from(body))
    );
}

#[test]
fn kill_mid_upload_leaves_no_root_and_recovers_cleanly() {
    let scratch = tempfile::tempdir().expect("scratch");
    let server = Server::spawn_auto(scratch.path(), &[]).expect("server spawns");
    let endpoint = server.endpoint();
    let session = SessionId::from_u128(0x00DE_C0DE);
    let mut raw = RawDurable::connect(&endpoint);
    // Begin plus two data frames, then the honest crash — no commit, so
    // nothing may become visible, and staged-but-uncommitted packs must
    // not wedge recovery.
    let max_data = raw_upload_begin(&mut raw, 50, session, 1, "half", Some(1_500_000));
    let body = stream_bytes(1_500_000, 0x8A1F);
    for piece in body.chunks(max_data as usize).take(2) {
        raw_upload_data(&mut raw, 50, piece);
    }
    drop(raw);
    server.kill();
    let server = Server::spawn_auto(scratch.path(), &[]).expect("server spawns");
    let client = server.client();
    assert_eq!(client.get(&Key::from("half")).expect("get"), None);
    // The log continues past the orphaned stage: new writes land fine.
    client
        .set(&Key::from("after"), bytes::Bytes::from_static(b"yes"))
        .expect("set");
    assert_eq!(
        client.get(&Key::from("after")).expect("get"),
        Some(bytes::Bytes::from_static(b"yes"))
    );
}

/// Uploads one body, retrying load-induced stalls (timeout/overload)
/// with a fresh source. Retries are commit-safe: a stalled first attempt
/// either never committed (plain retry) or did (same bytes → identical
/// root either way, so the retry converges rather than corrupts). Only
/// terminal errors fail the test — the point here is atomicity under
/// concurrency, not liveness under parallel-suite fsync storms.
fn put_stream_sturdy(
    client: &NativeClient,
    key: &Key,
    body: &[u8],
    attempts: u32,
) -> Result<(), kivi_client::ClientError> {
    use kivi_client::ClientError;
    let mut last = ClientError::Timeout;
    for _ in 0..attempts.max(1) {
        let mut source = std::io::Cursor::new(body.to_vec());
        match client.put_stream(key, Some(body.len() as u64), &mut source) {
            Ok(()) => return Ok(()),
            Err(ClientError::Timeout | ClientError::Overloaded) => {
                last = ClientError::Timeout;
            }
            Err(error) => return Err(error),
        }
    }
    Err(last)
}

#[test]
fn concurrent_same_key_uploads_never_tear() {
    use std::sync::Barrier;
    let scratch = tempfile::tempdir().expect("scratch");
    let server = Server::spawn_auto(scratch.path(), &[]).expect("server spawns");
    let endpoint = server.endpoint();
    let bodies = [
        stream_bytes(1_200_000, 0xA11CE),
        stream_bytes(2_300_000, 0xB0B),
    ];
    let start = Arc::new(Barrier::new(3));
    let errors = Arc::new(std::sync::Mutex::new(Vec::new()));
    let mut handles = Vec::new();
    for body in bodies.clone() {
        let start = Arc::clone(&start);
        let errors = Arc::clone(&errors);
        let endpoint = endpoint.clone();
        handles.push(std::thread::spawn(move || {
            let client = NativeClient::new(ClientConfig {
                seeds: vec![endpoint],
                namespace: NS,
                ..ClientConfig::default()
            })
            .expect("client builds");
            start.wait();
            if let Err(error) = put_stream_sturdy(&client, &Key::from("race"), &body, 3) {
                errors
                    .lock()
                    .expect("errors lock")
                    .push(format!("{error:?}"));
            }
        }));
    }
    start.wait();
    for handle in handles {
        handle.join().expect("uploader joins");
    }
    assert!(
        errors.lock().expect("errors lock").is_empty(),
        "both uploads commit cleanly: {:?}",
        errors.lock().expect("errors lock")
    );
    // Last-writer-wins, never torn: the stored value is exactly one of
    // the two uploads — full length, full bytes, no interleave.
    let client = server.client();
    let stored = client
        .get(&Key::from("race"))
        .expect("get")
        .expect("present");
    let options: Vec<bytes::Bytes> = bodies.into_iter().map(bytes::Bytes::from).collect();
    assert!(
        options.contains(&stored),
        "stored value is atomic: {} bytes, matching neither upload",
        stored.len()
    );
}

#[test]
fn kill_restart_recovers_acknowledged_writes() {
    let scratch = tempfile::tempdir().expect("scratch");
    let server = Server::spawn_auto(scratch.path(), &[]).expect("server spawns");
    let client = server.client();
    client
        .set(&Key::from("hello"), bytes::Bytes::from_static(b"world"))
        .expect("set");
    assert_eq!(
        client
            .counter_add(&Key::from("n"), 41)
            .expect("counter add"),
        41
    );
    drop(client);
    server.kill();
    // Same data directory, fresh ports: recovery keys off the directory,
    // never the socket addresses.
    let server = Server::spawn_auto(scratch.path(), &[]).expect("server spawns");
    let client = server.client();
    assert_eq!(
        client.get(&Key::from("hello")).expect("get"),
        Some(bytes::Bytes::from_static(b"world"))
    );
    assert_eq!(client.counter_get(&Key::from("n")).expect("read"), Some(41));
    // The log continues: new writes land after the replayed ones.
    assert_eq!(client.counter_add(&Key::from("n"), 1).expect("add"), 42);
}

#[test]
fn replay_after_kill_returns_original_without_reexecuting() {
    use kivi_protocol::Status;
    let scratch = tempfile::tempdir().expect("scratch");
    let server = Server::spawn_auto(scratch.path(), &[]).expect("server spawns");
    let endpoint = server.endpoint();
    // Fixed identity: this is the "lost response" — the client sent
    // (session, seq=1), the reply never arrived, and the retry carries
    // the same bytes.
    let session = SessionId::from_u128(0xC0_FFEE);
    let mut raw = RawDurable::connect(&endpoint);
    let first = raw.counter_add(1, session, 1, 0, "exact");
    assert_eq!(counter_value(&first), 1);
    // Same identity on the live server: dedup hit, still 1.
    let replay = raw.counter_add(2, session, 1, 0, "exact");
    assert_eq!(replay.status, Status::Ok);
    assert_eq!(counter_value(&replay), 1);
    drop(raw);
    server.kill();
    // After the crash the dedup memory is gone — but the WAL remembers.
    let server = Server::spawn_auto(scratch.path(), &[]).expect("server spawns");
    let mut raw = RawDurable::connect(&server.endpoint());
    let after_crash = raw.counter_add(3, session, 1, 0, "exact");
    assert_eq!(after_crash.status, Status::Ok);
    assert_eq!(
        counter_value(&after_crash),
        1,
        "replayed identity must not re-execute after recovery"
    );
    // And the window advanced past it: seq 2 executes fresh.
    let next = raw.counter_add(4, session, 2, 1, "exact");
    assert_eq!(counter_value(&next), 2);
}

#[test]
fn second_process_is_refused_while_first_holds_the_lock() {
    let scratch = tempfile::tempdir().expect("scratch");
    let server = Server::spawn_auto(scratch.path(), &[]).expect("server spawns");
    // A second process on the same directory must fail fast, never
    // interleave WAL writes with the live owner. It takes ephemeral ports
    // too: only the data-directory lock is under test, so no address can
    // collide with anything.
    let mut probe =
        std::process::Command::new(kivi_lab::process::server_binary_path().expect("server binary"))
            .arg("--data-dir")
            .arg(scratch.path())
            .arg("--port")
            .arg("0")
            .arg("--admin")
            .arg("127.0.0.1:0")
            .arg("--workers")
            .arg("2")
            .arg("--tablets")
            .arg("1")
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()
            .expect("probe spawns");
    let status = probe.wait().expect("probe exits");
    assert!(
        !status.success(),
        "second owner of a live data directory must fail"
    );
    drop(server);
}

#[test]
fn admin_durability_reports_mode_identity_and_recovery() {
    let scratch = tempfile::tempdir().expect("scratch");
    let server = Server::spawn_auto(scratch.path(), &[]).expect("server spawns");
    let client = server.client();
    for index in 0..3u32 {
        client
            .set(
                &Key::from(format!("admin:{index}")),
                bytes::Bytes::from_static(b"v"),
            )
            .expect("set");
    }
    let (status, first) = server.admin_get("/v1/durability");
    assert_eq!(status, 200, "durability endpoint answers");
    assert!(
        first.contains("\"mode\":\"durable\""),
        "mode reported, got {first}"
    );
    assert!(
        first.contains("\"records_replayed\":0"),
        "fresh boot replays nothing, got {first}"
    );
    drop(client);
    server.kill();
    let server = Server::spawn_auto(scratch.path(), &[]).expect("server spawns");
    let (status, second) = server.admin_get("/v1/durability");
    assert_eq!(status, 200);
    assert!(
        second.contains("\"records_replayed\":3"),
        "all three writes replayed, got {second}"
    );
    assert!(
        second.contains("\"health\":\"healthy\""),
        "storage healthy, got {second}"
    );
    // The data plane agrees with the control plane.
    let client = server.client();
    assert_eq!(
        client.get(&Key::from("admin:2")).expect("get"),
        Some(bytes::Bytes::from_static(b"v"))
    );
}

#[test]
fn segment_rotation_recovers_across_many_segments() {
    let scratch = tempfile::tempdir().expect("scratch");
    // 4 KiB segments: a few hundred small writes must rotate repeatedly.
    let server = Server::spawn_auto(scratch.path(), &["--wal-segment-target", "4096"])
        .expect("server spawns");
    let client = server.client();
    let value = bytes::Bytes::from(vec![b'x'; 97]);
    for index in 0..300u32 {
        client
            .set(&Key::from(format!("rot:{index:04}")), value.clone())
            .expect("set");
    }
    drop(client);
    let segments: Vec<_> = walkdir_wal(scratch.path()).collect();
    assert!(
        segments.len() >= 2,
        "tiny target must rotate, found {}",
        segments.len()
    );
    server.kill();
    let server = Server::spawn_auto(scratch.path(), &[]).expect("server spawns");
    let client = server.client();
    for index in [0u32, 1, 149, 150, 298, 299] {
        assert_eq!(
            client
                .get(&Key::from(format!("rot:{index:04}")))
                .expect("get"),
            Some(value.clone()),
            "key rot:{index:04} survives multi-segment recovery"
        );
    }
}

fn walkdir_wal(dir: &std::path::Path) -> impl Iterator<Item = std::path::PathBuf> {
    let mut found = Vec::new();
    let mut stack = vec![dir.join("wal")];
    while let Some(top) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&top) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else if path.extension().is_some_and(|ext| ext == "wal") {
                found.push(path);
            }
        }
    }
    found.into_iter()
}

/// Waits until the server reports at least `count` completed checkpoints.
fn await_checkpoints(server: &Server, count: u64) {
    let deadline = Instant::now() + Duration::from_secs(90);
    loop {
        let completed = server.admin_number("/v1/checkpoints", "checkpoints_completed");
        if completed >= count {
            return;
        }
        assert!(
            Instant::now() <= deadline,
            "only {completed} checkpoints after 90s, wanted {count}"
        );
        std::thread::sleep(Duration::from_millis(200));
    }
}

/// Lists WAL segment files across all lanes of a data directory.
fn wal_segments(dir: &std::path::Path) -> Vec<std::path::PathBuf> {
    walkdir_wal(dir).collect()
}

#[test]
fn group_commit_batches_hot_writes_with_one_barrier() {
    let scratch = tempfile::tempdir().expect("scratch");
    // No automatic checkpoints: this measures the commit pipeline alone.
    let server = Server::spawn_auto(scratch.path(), &["--no-checkpoint"]).expect("server spawns");
    let endpoint = server.endpoint();
    // Eight clients hammer one hot counter: every add is ordered behind
    // the last on its tablet, so sustained load must batch them.
    let threads: Vec<_> = (0..8u32)
        .map(|_| {
            let endpoint = endpoint.clone();
            std::thread::spawn(move || {
                let client = NativeClient::new(ClientConfig {
                    seeds: vec![endpoint],
                    namespace: NS,
                    ..ClientConfig::default()
                })
                .expect("client builds");
                for _ in 0..200 {
                    client
                        .counter_add(&Key::from("hot"), 1)
                        .expect("counter add");
                }
            })
        })
        .collect();
    for thread in threads {
        thread.join().expect("client thread");
    }
    // Exact under batching: 8 × 200 ordered adds, none lost, none doubled.
    let client = server.client();
    assert_eq!(
        client.counter_get(&Key::from("hot")).expect("read"),
        Some(1600)
    );
    // ...while one barrier acknowledges many mutations.
    let logical = server.admin_number("/v1/durability", "logical_mutations");
    let batches = server.admin_number("/v1/durability", "physical_batches");
    let barriers = server.admin_number("/v1/durability", "barriers");
    let (_, durability_body) = server.admin_get("/v1/durability");
    assert_eq!(logical, 1600, "every add committed");
    assert!(batches < logical, "batches {batches} < mutations {logical}");
    assert_eq!(barriers, batches, "one barrier per batch today");
    // Admin counters are exact u64; the ratio is a human-readable test
    // assertion (values here are in the thousands — nowhere near 2^53).
    #[allow(clippy::cast_precision_loss)]
    let per_fsync = logical as f64 / barriers as f64;
    // Immediate mode pins this at exactly 1.0 (one barrier per mutation
    // by construction); group commit under overlap measures ~4+. The
    // threshold sits between at 1.5: a starved scheduler still overlaps
    // pairs (worst observed: 2.0), while anything at 1.0 proves batching
    // regressed to immediate behavior. Exactness rides on the structural
    // asserts above, not this ratio.
    assert!(
        per_fsync > 1.5,
        "mutations/fsync above immediate's 1.0, got {per_fsync:.1} ({logical}/{barriers}): {durability_body}"
    );
}

#[test]
fn checkpoint_dedup_money_test_retry_after_reclaim_returns_original() {
    use kivi_protocol::Status;
    let scratch = tempfile::tempdir().expect("scratch");
    // Tiny segments (rotation) plus eager checkpoints (mutations): two
    // checkpoints install current + previous, and reclamation runs.
    let server = Server::spawn_auto(
        scratch.path(),
        &[
            "--wal-segment-target",
            "4096",
            "--checkpoint-mutations",
            "5",
        ],
    )
    .expect("server spawns");
    let endpoint = server.endpoint();
    // The lost reply: (session S, seq 1) applies; the client never sees
    // the response and will retry the same bytes after the crash.
    let session = SessionId::from_u128(0xD0_DED);
    let mut raw = RawDurable::connect(&endpoint);
    let first = raw.counter_add(1, session, 1, 0, "exact");
    assert_eq!(counter_value(&first), 1);
    drop(raw);
    // Advance the log in two phases with a checkpoint between them, so
    // two checkpoints install (current + previous) and reclamation has a
    // verified floor. Big pads force several segments so the earliest
    // one must actually go away.
    let client = server.client();
    for index in 0..10u32 {
        client
            .set(
                &Key::from(format!("pad:{index}")),
                bytes::Bytes::from(vec![b'p'; 512]),
            )
            .expect("set");
    }
    await_checkpoints(&server, 1);
    for index in 10..30u32 {
        client
            .set(
                &Key::from(format!("pad:{index}")),
                bytes::Bytes::from(vec![b'p'; 512]),
            )
            .expect("set");
    }
    await_checkpoints(&server, 2);
    let floor_deadline = Instant::now() + Duration::from_secs(60);
    loop {
        if scratch
            .path()
            .join("checkpoints")
            .join("WAL_FLOOR")
            .exists()
        {
            break;
        }
        assert!(
            Instant::now() <= floor_deadline,
            "reclaim never published a WAL floor"
        );
        std::thread::sleep(Duration::from_millis(200));
    }
    // The segment that held seq 1 is sealed history far below the floor:
    // it must be reclaimed (deleted), proving the retry below is answered
    // from checkpointed dedup state rather than the original WAL.
    let first_segment = scratch
        .path()
        .join("wal")
        .join("lane-0000")
        .join("00000000000000000001.wal");
    assert!(
        !first_segment.exists(),
        "seq 1's original segment must be reclaimed before the kill"
    );
    drop(client);
    drop(endpoint);
    server.kill();
    // Restart, then retry the SAME identity on a fresh connection: the
    // checkpointed dedup state must answer without re-executing, even
    // though the original WAL is long reclaimed.
    let server = Server::spawn_auto(scratch.path(), &[]).expect("server spawns");
    let mut raw = RawDurable::connect(&server.endpoint());
    let replay = raw.counter_add(10, session, 1, 0, "exact");
    assert_eq!(replay.status, Status::Ok);
    assert_eq!(
        counter_value(&replay),
        1,
        "replayed identity must not re-execute after WAL reclamation"
    );
    let next = raw.counter_add(11, session, 2, 1, "exact");
    assert_eq!(counter_value(&next), 2, "window continues past the replay");
}

#[test]
fn wal_reclamation_removes_old_segments_and_survives_kill() {
    let scratch = tempfile::tempdir().expect("scratch");
    let server = Server::spawn_auto(
        scratch.path(),
        &[
            "--wal-segment-target",
            "4096",
            "--checkpoint-mutations",
            "60",
        ],
    )
    .expect("server spawns");
    let client = server.client();
    // Phase sizes vs ~190B records and 4KiB segments (~21 records each):
    // phase 1 fills ~3 segments, checkpoint 1 covers the first two fully.
    for index in 0..60u32 {
        client
            .set(
                &Key::from(format!("reclaim:{index:03}")),
                bytes::Bytes::from(vec![b'x'; 64]),
            )
            .expect("set");
    }
    await_checkpoints(&server, 1);
    // Snapshot the pre-reclaim layout now: the second checkpoint installs
    // a previous, which lets reclamation delete fully covered sealed
    // segments (it cannot run before two checkpoints exist).
    let before: Vec<_> = wal_segments(scratch.path());
    assert!(
        before.len() >= 3,
        "phase 1 should span segments, found {before:?}"
    );
    for index in 60..120u32 {
        client
            .set(
                &Key::from(format!("reclaim:{index:03}")),
                bytes::Bytes::from(vec![b'x'; 64]),
            )
            .expect("set");
    }
    await_checkpoints(&server, 2);
    // The floor promise (previous cut) covers the early sealed segments:
    // some pre-checkpoint file must be gone (counts can stay level as
    // deletions and new segments offset), while the active tail remains.
    let deadline = Instant::now() + Duration::from_secs(60);
    loop {
        let now = wal_segments(scratch.path());
        if before.iter().any(|old| !now.contains(old)) {
            break;
        }
        if Instant::now() > deadline {
            let (_, checkpoints) = server.admin_get("/v1/checkpoints");
            let floor = scratch.path().join("checkpoints").join("WAL_FLOOR");
            panic!(
                "reclaim deleted nothing:\nbefore: {before:?}\nafter: {now:?}\ncheckpoints: {checkpoints}\nWAL_FLOOR present: {}",
                floor.exists(),
            );
        }
        std::thread::sleep(Duration::from_millis(200));
    }
    assert!(
        !wal_segments(scratch.path()).is_empty(),
        "the active tail always remains"
    );
    drop(client);
    // Kill mid-steady-state (possibly mid-reclaim: deletes are
    // idempotent) and restart: everything acknowledged comes back.
    server.kill();
    let server = Server::spawn_auto(scratch.path(), &[]).expect("server spawns");
    let client = server.client();
    for index in 0..120u32 {
        assert_eq!(
            client
                .get(&Key::from(format!("reclaim:{index:03}")))
                .expect("get"),
            Some(bytes::Bytes::from(vec![b'x'; 64])),
            "key survives reclaim plus kill"
        );
    }
}

#[test]
fn incremental_checkpoint_reuses_clean_bands() {
    let scratch = tempfile::tempdir().expect("scratch");
    // Phase 1 with checkpoints off: 200 writes, then kill. Restart with
    // an eager threshold so exactly one checkpoint covers all of them.
    let server = Server::spawn_auto(scratch.path(), &["--no-checkpoint"]).expect("server spawns");
    let client = server.client();
    for index in 0..200u32 {
        client
            .set(
                &Key::from(format!("inc:{index:03}")),
                bytes::Bytes::from_static(b"v"),
            )
            .expect("set");
    }
    drop(client);
    server.kill();
    let server = Server::spawn_auto(scratch.path(), &["--checkpoint-mutations", "5"])
        .expect("server spawns");
    await_checkpoints(&server, 1);
    let reused_first = server.admin_number("/v1/checkpoints", "bands_reused");
    assert_eq!(reused_first, 0, "first checkpoint builds everything");
    // Phase 2: overwrite 60 keys that all live in one band.
    let band = band_of_key(&Key::from("reband:000000"));
    let fresh = keys_in_band(band, 60, "reband2");
    let client = server.client();
    for key in &fresh {
        client
            .set(key, bytes::Bytes::from_static(b"v2"))
            .expect("set");
    }
    drop(client);
    await_checkpoints(&server, 2);
    let (status, body) = server.admin_get("/v1/checkpoints");
    assert_eq!(status, 200);
    let reused = server.admin_number("/v1/checkpoints", "bands_reused");
    assert_eq!(reused, 255, "exactly one band changed, got body: {body}");
    // Phase 3: delete every phase-1 key in the smallest-occupied band
    // (plus same-band pad writes to trip the threshold), then prove the
    // deleted keys do not resurrect after restart.
    let occupancy = band_occupancy(&server);
    let (victim_band, victim_keys) = occupancy
        .into_iter()
        .min_by_key(|(_, keys)| keys.len())
        .expect("occupied bands exist");
    let mut phase3 = victim_keys;
    let mut pad = keys_in_band(victim_band, 8, "reband3");
    phase3.append(&mut pad);
    let client = server.client();
    for key in &phase3[..phase3.len().min(8)] {
        if key.as_bytes().starts_with(b"inc:") {
            client.delete(key).expect("delete");
        } else {
            client
                .set(key, bytes::Bytes::from_static(b"v3"))
                .expect("set");
        }
    }
    // Ensure at least 5 mutations tripped the threshold even if the
    // victim band held fewer keys.
    for key in keys_in_band(victim_band, 5, "reband4") {
        client
            .set(&key, bytes::Bytes::from_static(b"v3"))
            .expect("set");
    }
    drop(client);
    await_checkpoints(&server, 3);
    server.kill();
    let server = Server::spawn_auto(scratch.path(), &[]).expect("server spawns");
    let client = server.client();
    for key in &phase3 {
        if key.as_bytes().starts_with(b"inc:") {
            assert_eq!(
                client.get(key).expect("get"),
                None,
                "deleted key {key:?} must not resurrect"
            );
        }
    }
    // A survivor outside the victim band is intact.
    let survivor = (0..200u32)
        .map(|index| Key::from(format!("inc:{index:03}")))
        .find(|key| band_of_key(key) != victim_band)
        .expect("a survivor exists");
    assert_eq!(
        client.get(&survivor).expect("get"),
        Some(bytes::Bytes::from_static(b"v")),
        "untouched keys survive"
    );
}

/// Band of one key under the server's namespace and layout.
fn band_of_key(key: &Key) -> u16 {
    kivi_checkpoint::BandLayout::V1
        .band_of(NS, key.as_bytes())
        .expect("mappable")
}

/// Finds `count` fresh keys that all map to `band`.
fn keys_in_band(band: u16, count: usize, prefix: &str) -> Vec<Key> {
    let mut keys = Vec::new();
    for index in 0..1_000_000u32 {
        let key = Key::from(format!("{prefix}:{index:06}"));
        if band_of_key(&key) == band {
            keys.push(key);
            if keys.len() >= count {
                return keys;
            }
        }
    }
    panic!("could not find {count} keys in band {band}");
}

/// Groups phase-1 keys (`inc:*`) by band (read back through the client).
fn band_occupancy(server: &Server) -> Vec<(u16, Vec<Key>)> {
    use std::collections::HashMap;
    let client = server.client();
    let mut groups: HashMap<u16, Vec<Key>> = HashMap::new();
    for index in 0..200u32 {
        let key = Key::from(format!("inc:{index:03}"));
        // All phase-1 keys exist (written before the kill, checkpointed).
        assert!(
            client.get(&key).expect("get").is_some(),
            "phase-1 key present"
        );
        groups.entry(band_of_key(&key)).or_default().push(key);
    }
    groups.into_iter().collect()
}

/// One pre-kill mutation attempt with its explicit identity: everything
/// needed to retry it verbatim after a crash. Each attempt owns a
/// single-use session (one client per mutation below), so no session
/// floor can advance past it before the kill — a retry is always a
/// dedup hit (applied before the kill) or a first execution (never
/// sent), never an expiry.
enum Ambiguous {
    Set {
        session: SessionId,
        key: Key,
        value: bytes::Bytes,
    },
    Add {
        session: SessionId,
        key: Key,
        delta: i64,
    },
}

/// A client resuming a pre-kill session: retries speak as the original
/// session, so the server answers from the retained outcome instead of
/// executing twice.
fn resume_client(endpoint: &str, session: SessionId) -> NativeClient {
    NativeClient::new(ClientConfig {
        seeds: vec![endpoint.to_owned()],
        namespace: NS,
        session: Some(session),
        ..ClientConfig::default()
    })
    .expect("resume client builds")
}

/// Retries every ambiguous identity verbatim (fresh resume client per
/// attempt: completing an attempt advances that client's floor, so a
/// second attempt on the same object would self-expire). Each identity
/// applies exactly once across kill + resume — hit when the first
/// attempt applied, first execution otherwise — and a repeat retry must
/// return the identical outcome, proving the record is stable rather
/// than re-executed.
fn retry_ambiguous(endpoint: &str, ambiguous: &[Ambiguous]) {
    for op in ambiguous {
        match op {
            Ambiguous::Set {
                session,
                key,
                value,
            } => {
                resume_client(endpoint, *session)
                    .set_with_seq(key, value.clone(), RequestSeq::from_u64(1))
                    .expect("ambiguous set resolves");
                resume_client(endpoint, *session)
                    .set_with_seq(key, value.clone(), RequestSeq::from_u64(1))
                    .expect("ambiguous set stable on re-retry");
            }
            Ambiguous::Add {
                session,
                key,
                delta,
            } => {
                let first = resume_client(endpoint, *session)
                    .counter_add_with_seq(key, *delta, RequestSeq::from_u64(1))
                    .expect("ambiguous add resolves");
                let second = resume_client(endpoint, *session)
                    .counter_add_with_seq(key, *delta, RequestSeq::from_u64(1))
                    .expect("ambiguous add stable on re-retry");
                assert_eq!(
                    second, first,
                    "re-retry is a stable dedup hit ({first}), not a re-execution"
                );
            }
        }
    }
}

/// One hammer thread's outcome: acknowledged sets/adds plus every
/// unacknowledged attempt with its retryable identity.
struct HammerOutcome {
    sets: Vec<(Key, bytes::Bytes)>,
    adds: u64,
    ambiguous: Vec<Ambiguous>,
    acked_sessions: HashSet<SessionId>,
    ambiguous_sessions: HashSet<SessionId>,
}

/// Hammers one key range with per-attempt sessions: a set plus a shared
/// hot-counter add per index, each on its own client (own session) with
/// explicit sequence 1. Stops promptly once `stop` is set.
fn hammer(endpoint: &str, stop: &AtomicBool, thread: u32) -> HammerOutcome {
    let one = || {
        NativeClient::new(ClientConfig {
            seeds: vec![endpoint.to_owned()],
            namespace: NS,
            connect_timeout: Duration::from_millis(300),
            dial_attempts: 1,
            ..ClientConfig::default()
        })
        .expect("client builds")
    };
    let mut outcome = HammerOutcome {
        sets: Vec::new(),
        adds: 0,
        ambiguous: Vec::new(),
        acked_sessions: HashSet::new(),
        ambiguous_sessions: HashSet::new(),
    };
    for index in 0..250u32 {
        if stop.load(Ordering::Relaxed) {
            break;
        }
        let key = Key::from(format!("am:{thread}:{index:03}"));
        let value = bytes::Bytes::from(format!("{thread}-{index}"));
        let set_client = one();
        let set_session = set_client.session();
        if set_client
            .set_with_seq(&key, value.clone(), RequestSeq::from_u64(1))
            .is_ok()
        {
            outcome.sets.push((key, value));
            assert!(
                outcome.acked_sessions.insert(set_session),
                "one session per attempt"
            );
        } else {
            outcome.ambiguous.push(Ambiguous::Set {
                session: set_session,
                key,
                value,
            });
            assert!(
                outcome.ambiguous_sessions.insert(set_session),
                "one session per attempt"
            );
        }
        let add_client = one();
        let add_session = add_client.session();
        if add_client
            .counter_add_with_seq(&Key::from("am-hot"), 1, RequestSeq::from_u64(1))
            .is_ok()
        {
            outcome.adds += 1;
            assert!(
                outcome.acked_sessions.insert(add_session),
                "one session per attempt"
            );
        } else {
            outcome.ambiguous.push(Ambiguous::Add {
                session: add_session,
                key: Key::from("am-hot"),
                delta: 1,
            });
            assert!(
                outcome.ambiguous_sessions.insert(add_session),
                "one session per attempt"
            );
        }
    }
    outcome
}

#[test]
fn concurrent_checkpoint_workload_survives_kill_exactly() {
    let scratch = tempfile::tempdir().expect("scratch");
    let server = Server::spawn_auto(scratch.path(), &["--checkpoint-mutations", "200"])
        .expect("server spawns");
    let endpoint = server.endpoint();
    // Four threads hammering: distinct SET keys plus one shared hot
    // counter. Every mutation carries its own session and explicit
    // sequence 1, so every unacknowledged attempt is retryable verbatim
    // after the kill — no identity is ever inferred or reused across
    // attempts. Workload dials fail fast (short connect budget): once
    // the server is dead there is nothing to learn from redialing it.
    eprintln!("AM: spawning clients");
    let stop = Arc::new(AtomicBool::new(false));
    let threads: Vec<_> = (0..4u32)
        .map(|thread| {
            let endpoint = endpoint.clone();
            let stop = Arc::clone(&stop);
            std::thread::spawn(move || hammer(&endpoint, &stop, thread))
        })
        .collect();
    // Let the workload (and checkpoints) run briefly, then crash mid-run.
    // Threads stop issuing once the server is dead, keeping the
    // ambiguous set to the true kill-window attempts instead of hundreds
    // of never-sent post-kill dials (which would still be correct to
    // retry, only slow).
    std::thread::sleep(Duration::from_secs(2));
    eprintln!("AM: killing server");
    drop(endpoint);
    server.kill();
    stop.store(true, Ordering::Relaxed);
    eprintln!("AM: joining clients");
    let mut acked_sets = Vec::new();
    let mut acked_adds = 0u64;
    let mut ambiguous = Vec::new();
    let mut acked_sessions = HashSet::new();
    let mut ambiguous_sessions = HashSet::new();
    for thread in threads {
        let outcome = thread.join().expect("client thread");
        acked_sets.extend(outcome.sets);
        acked_adds += outcome.adds;
        ambiguous.extend(outcome.ambiguous);
        acked_sessions.extend(outcome.acked_sessions);
        ambiguous_sessions.extend(outcome.ambiguous_sessions);
    }
    for session in &ambiguous_sessions {
        assert!(
            !acked_sessions.contains(session),
            "an identity is acknowledged or ambiguous, never both"
        );
    }
    let ambiguous_adds = ambiguous
        .iter()
        .filter(|op| matches!(op, Ambiguous::Add { .. }))
        .count() as u64;
    eprintln!(
        "AM: acked {} sets, {} adds, {} ambiguous; respawning",
        acked_sets.len(),
        acked_adds,
        ambiguous.len(),
    );
    // Restart: every acknowledged write is present, exactly.
    let server = Server::spawn_auto(scratch.path(), &[]).expect("server spawns");
    eprintln!("AM: respawned; verifying");
    let client = server.client();
    for (key, value) in &acked_sets {
        assert_eq!(
            client.get(key).expect("get"),
            Some(value.clone()),
            "acknowledged set survives"
        );
    }
    // The kill window is honest ambiguity, not loss or duplication: an
    // ambiguous attempt may have applied (durable, reply lost in the
    // kill) or never been sent. The recovered counter must lie within
    // exactly that window — anything outside is loss or double-apply.
    let recovered = u64::try_from(
        client
            .counter_get(&Key::from("am-hot"))
            .expect("read")
            .unwrap_or(0),
    )
    .expect("counter nonnegative");
    assert!(
        recovered >= acked_adds && recovered <= acked_adds + ambiguous_adds,
        "recovered {recovered} within [acked {acked_adds}, acked + ambiguous {}]",
        acked_adds + ambiguous_adds,
    );
    // Retry every ambiguous identity verbatim through the shared
    // helper (fresh resume client per attempt, stability re-retry
    // inside).
    let endpoint = server.endpoint();
    retry_ambiguous(&endpoint, &ambiguous);
    // Exactness, now provable: every ambiguous identity contributed
    // exactly one application in total, and every acknowledged one
    // survived. A loss would fall short; a double-apply would exceed.
    let final_counter = client
        .counter_get(&Key::from("am-hot"))
        .expect("read")
        .unwrap_or(0);
    assert_eq!(
        final_counter,
        acked_adds.cast_signed() + ambiguous_adds.cast_signed(),
        "counter equals acked adds plus one application per ambiguous identity"
    );
    for op in &ambiguous {
        if let Ambiguous::Set { key, value, .. } = op {
            assert_eq!(
                client.get(key).expect("get"),
                Some(value.clone()),
                "retried set is present with its exact value"
            );
        }
    }
}

#[test]
fn deleted_current_recovers_from_wal_replay() {
    // Interrupted publication is invisible: with CURRENT gone, recovery
    // replays the WAL from genesis and restores everything acknowledged.
    let scratch = tempfile::tempdir().expect("scratch");
    let server = Server::spawn_auto(scratch.path(), &["--checkpoint-mutations", "5"])
        .expect("server spawns");
    let client = server.client();
    for index in 0..10u32 {
        client
            .set(
                &Key::from(format!("inv:{index}")),
                bytes::Bytes::from_static(b"v"),
            )
            .expect("set");
    }
    await_checkpoints(&server, 1);
    drop(client);
    server.kill();
    std::fs::remove_file(
        scratch
            .path()
            .join("checkpoints")
            .join("tablet-1")
            .join("CURRENT"),
    )
    .expect("delete CURRENT");
    let server = Server::spawn_auto(scratch.path(), &[]).expect("server spawns");
    let client = server.client();
    for index in 0..10u32 {
        assert_eq!(
            client.get(&Key::from(format!("inv:{index}"))).expect("get"),
            Some(bytes::Bytes::from_static(b"v")),
            "WAL replay restores everything without a catalog"
        );
    }
}

#[test]
fn missing_current_manifest_falls_back_to_previous() {
    // Two checkpoints install current + previous. Destroying the current
    // manifest must fall back loudly to the retained previous (WAL tail
    // covers the gap) — never partial state, never silent loss.
    let scratch = tempfile::tempdir().expect("scratch");
    let server = Server::spawn_auto(scratch.path(), &["--checkpoint-mutations", "5"])
        .expect("server spawns");
    let client = server.client();
    for index in 0..5u32 {
        client
            .set(
                &Key::from(format!("fb:{index}")),
                bytes::Bytes::from_static(b"v"),
            )
            .expect("set");
    }
    await_checkpoints(&server, 1);
    for index in 5..10u32 {
        client
            .set(
                &Key::from(format!("fb:{index}")),
                bytes::Bytes::from_static(b"v"),
            )
            .expect("set");
    }
    await_checkpoints(&server, 2);
    let (status, body) = server.admin_get("/v1/checkpoints");
    assert_eq!(status, 200);
    let current_manifest = extract_manifest(&body, "current");
    drop(client);
    server.kill();
    std::fs::remove_file(
        scratch
            .path()
            .join("checkpoints")
            .join("tablet-1")
            .join("manifests")
            .join(format!("{current_manifest}.manifest")),
    )
    .expect("delete current manifest");
    let server = Server::spawn_auto(scratch.path(), &[]).expect("server spawns");
    let client = server.client();
    for index in 0..10u32 {
        assert_eq!(
            client.get(&Key::from(format!("fb:{index}"))).expect("get"),
            Some(bytes::Bytes::from_static(b"v")),
            "previous checkpoint plus WAL tail restores everything"
        );
    }
    let fell_back = server.admin_number("/v1/durability", "checkpoint_fell_back");
    assert_eq!(fell_back, 1, "fallback is loud in recovery stats");
}

/// Extracts the manifest hash of the first `"which":"current"` entry.
fn extract_manifest(body: &str, which: &str) -> String {
    let marker = format!("\"which\":\"{which}\"");
    let at = body.find(&marker).expect("entry present");
    let hash_key = "\"manifest\":\"";
    let from = body[at..].find(hash_key).expect("manifest present") + at + hash_key.len();
    body[from..from + 64].to_owned()
}

#[test]
fn corrupt_band_is_detected_loudly() {
    // Corrupting a current-only band (rewritten by the second checkpoint,
    // so the previous chain references different files) must fail loudly
    // or fall back — never serve partial state.
    let scratch = tempfile::tempdir().expect("scratch");
    let server = Server::spawn_auto(scratch.path(), &["--checkpoint-mutations", "5"])
        .expect("server spawns");
    let client = server.client();
    for index in 0..5u32 {
        client
            .set(
                &Key::from(format!("cb:{index}")),
                bytes::Bytes::from_static(b"v"),
            )
            .expect("set");
    }
    await_checkpoints(&server, 1);
    for index in 5..10u32 {
        client
            .set(
                &Key::from(format!("cb:{index}")),
                bytes::Bytes::from_static(b"v"),
            )
            .expect("set");
    }
    await_checkpoints(&server, 2);
    let (status, body) = server.admin_get("/v1/checkpoints");
    assert_eq!(status, 200);
    let current_manifest = extract_manifest(&body, "current");
    drop(client);
    server.kill();
    // Find a band file referenced ONLY by the current manifest (a dirty
    // band the second checkpoint rebuilt) and flip one byte.
    let bands_dir = scratch.path().join("checkpoints").join("bands");
    let manifest_path = scratch
        .path()
        .join("checkpoints")
        .join("tablet-1")
        .join("manifests")
        .join(format!("{current_manifest}.manifest"));
    let manifest_bytes = std::fs::read(&manifest_path).expect("manifest readable");
    let manifest = kivi_checkpoint::load_manifest(parse_hash(&current_manifest), &manifest_bytes)
        .expect("manifest loads");
    let band_hash = manifest.bands.first().expect("bands").hash.hex();
    let band_path = bands_dir.join(format!("{band_hash}.band"));
    let mut bytes = std::fs::read(&band_path).expect("band readable");
    let flip = bytes.len() / 2;
    bytes[flip] ^= 0xFF;
    std::fs::write(&band_path, &bytes).expect("band corrupted");
    // Restart: either loud failure or previous-fallback with full state.
    // A corrupt FIRST band (likely an empty reused band — identical file
    // in both chains) breaks both chains and must fail loudly.
    // Fresh ephemeral ports: recovery reads the data directory, so the
    // probe address is irrelevant and cannot collide with anything.
    // Give it a bounded window: success (fallback) or loud exit both end
    // the ambiguity. A hanging or partial server fails the test below.
    let outcome = kivi_lab::process::probe_server(
        kivi_lab::process::ServerMode::DataDir(scratch.path().to_owned()),
        2,
        1,
        &[],
        false,
        Duration::from_secs(20),
    )
    .expect("probe starts");
    match outcome {
        kivi_lab::process::ProbeOutcome::Ready(server) => {
            // Served: fallback must have restored everything.
            let client = server.client();
            for index in 0..10u32 {
                assert_eq!(
                    client.get(&Key::from(format!("cb:{index}"))).expect("get"),
                    Some(bytes::Bytes::from_static(b"v")),
                    "fallback restores everything or the server must not serve"
                );
            }
        }
        kivi_lab::process::ProbeOutcome::Exited { success, .. } => {
            assert!(!success, "corrupt band must fail loudly, not exit 0");
        }
        kivi_lab::process::ProbeOutcome::TimedOut { .. } => {
            panic!("corrupt band: server neither failed loudly nor served");
        }
    }
}

fn parse_hash(hex: &str) -> kivi_checkpoint::ArtifactHash {
    kivi_checkpoint::ArtifactHash::parse_hex(hex).expect("hash parses")
}
