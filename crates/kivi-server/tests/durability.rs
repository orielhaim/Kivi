//! Crash-recoverable durability through real server processes.
//!
//! These spawn the actual `kivi-server` binary, `kill` it without any
//! graceful shutdown (the honest crash), restart it on the same address
//! with the same `--data-dir`, and prove acknowledged writes come back.
//! The lost-response half of exactly-once is covered deterministically:
//! the same `(session, seq)` identity replayed — even after a kill and
//! restart, even on a fresh connection — returns the original outcome
//! without re-executing.

use std::io::{Read, Write};
use std::net::TcpStream;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use kivi_client::{ClientConfig, NativeClient};
use kivi_state::Key;
use kivi_types::{NamespaceId, RequestIdentity, RequestSeq, SessionId};

const NS: NamespaceId = NamespaceId::from_u64(1);
const READY_TIMEOUT: Duration = Duration::from_secs(20);

/// Reserves a worker pair (`base`, `base+1`) plus an admin port, holding
/// every probe until all three are known-free simultaneously. A lone
/// `free_port` is not enough: worker 1 binds `base+1`, which a random
/// free `base` says nothing about.
fn reserve_ports() -> (u16, u16) {
    for _ in 0..50 {
        let base_probe = std::net::TcpListener::bind("127.0.0.1:0").expect("probe bind");
        let base = base_probe.local_addr().expect("probe addr").port();
        let second = std::net::TcpListener::bind(format!("127.0.0.1:{}", base.wrapping_add(1)));
        let admin_probe = std::net::TcpListener::bind("127.0.0.1:0").expect("probe bind");
        let admin = admin_probe.local_addr().expect("probe addr").port();
        if second.is_ok() && admin != base && admin != base.wrapping_add(1) {
            return (base, admin);
        }
    }
    panic!("could not reserve a free worker pair plus admin port");
}

struct Server {
    child: Child,
    base: u16,
    admin: u16,
}

impl Server {
    fn spawn(data_dir: &std::path::Path, base: u16, admin: u16) -> Self {
        Self::spawn_with(data_dir, base, admin, &[])
    }

    fn spawn_with(data_dir: &std::path::Path, base: u16, admin: u16, extra: &[&str]) -> Self {
        let mut child = Command::new(env!("CARGO_BIN_EXE_kivi-server"))
            .arg("--data-dir")
            .arg(data_dir)
            .arg("--port")
            .arg(base.to_string())
            .arg("--admin")
            .arg(format!("127.0.0.1:{admin}"))
            .arg("--workers")
            .arg("2")
            .arg("--tablets")
            .arg("1")
            .args(extra)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .spawn()
            .expect("server binary spawns");
        // Readiness = the worker listener accepts TCP. Recovery runs
        // before any listener binds, so an accepted connection means the
        // WAL was already replayed (or the startup failed loudly).
        let deadline = Instant::now() + READY_TIMEOUT;
        loop {
            if TcpStream::connect(format!("127.0.0.1:{base}")).is_ok() {
                break;
            }
            if let Ok(Some(status)) = child.try_wait() {
                let mut log = String::new();
                if let Some(stderr) = child.stderr.as_mut() {
                    let _ = stderr.read_to_string(&mut log);
                }
                let tail: String = log
                    .lines()
                    .rev()
                    .take(8)
                    .collect::<Vec<_>>()
                    .into_iter()
                    .rev()
                    .collect::<Vec<_>>()
                    .join("\n");
                panic!("server exited during startup: {status}\nstderr tail:\n{tail}");
            }
            if Instant::now() > deadline {
                let _ = child.kill();
                panic!("server not ready after {READY_TIMEOUT:?}");
            }
            std::thread::sleep(Duration::from_millis(25));
        }
        Self { child, base, admin }
    }

    fn endpoint(&self) -> String {
        format!("127.0.0.1:{}", self.base)
    }

    fn admin_endpoint(&self) -> String {
        format!("127.0.0.1:{}", self.admin)
    }

    /// Raw HTTP GET against the admin plane (no HTTP client dependency;
    /// the responses are tiny JSON documents).
    fn admin_get(&self, path: &str) -> (u16, String) {
        let mut socket = TcpStream::connect(self.admin_endpoint()).expect("admin connect");
        socket
            .set_read_timeout(Some(Duration::from_secs(10)))
            .expect("timeout");
        write!(
            socket,
            "GET {path} HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n"
        )
        .expect("write");
        let mut body = String::new();
        socket
            .read_to_string(&mut body)
            .expect("read admin response");
        let status = body
            .lines()
            .next()
            .unwrap_or_default()
            .split_whitespace()
            .nth(1)
            .unwrap_or("0")
            .parse::<u16>()
            .unwrap_or(0);
        let payload = body.split("\r\n\r\n").nth(1).unwrap_or("").to_owned();
        (status, payload)
    }

    fn client(&self) -> NativeClient {
        NativeClient::new(ClientConfig {
            seeds: vec![self.endpoint()],
            namespace: NS,
            ..ClientConfig::default()
        })
        .expect("client builds")
    }

    /// The honest crash: no shutdown handshake, no flush beyond what the
    /// WAL already persisted per acknowledged write.
    fn kill(mut self) {
        self.child.kill().expect("kill succeeds");
        let _ = self.child.wait();
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
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
            namespace: NS,
            opcode: Opcode::CounterAdd,
            hint: None,
            key: key.as_bytes().to_vec(),
            value: None,
            delta: 1,
            expiry: 0,
            identity: Some(RequestIdentity::new(session, RequestSeq::from_u64(seq))),
            ack_floor: RequestSeq::from_u64(ack),
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

#[test]
fn kill_restart_recovers_acknowledged_writes() {
    let scratch = tempfile::tempdir().expect("scratch");
    let (base, admin) = reserve_ports();
    let server = Server::spawn(scratch.path(), base, admin);
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
    // Same address, same data directory, fresh admin port.
    let (_, admin) = reserve_ports();
    let server = Server::spawn(scratch.path(), base, admin);
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
    let (base, admin) = reserve_ports();
    let server = Server::spawn(scratch.path(), base, admin);
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
    let (_, admin) = reserve_ports();
    let server = Server::spawn(scratch.path(), base, admin);
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
    let (live_base, live_admin) = reserve_ports();
    let server = Server::spawn(scratch.path(), live_base, live_admin);
    // A second process on the same directory must fail fast, never
    // interleave WAL writes with the live owner.
    let (probe_base, probe_admin) = reserve_ports();
    let probe = Command::new(env!("CARGO_BIN_EXE_kivi-server"))
        .arg("--data-dir")
        .arg(scratch.path())
        .arg("--port")
        .arg(probe_base.to_string())
        .arg("--admin")
        .arg(format!("127.0.0.1:{probe_admin}"))
        .arg("--workers")
        .arg("2")
        .arg("--tablets")
        .arg("1")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("second process spawns");
    let output = probe.wait_with_output().expect("probe exits");
    assert!(
        !output.status.success(),
        "second owner of a live data directory must fail"
    );
    drop(server);
}

#[test]
fn admin_durability_reports_mode_identity_and_recovery() {
    let scratch = tempfile::tempdir().expect("scratch");
    let (base, admin) = reserve_ports();
    let server = Server::spawn(scratch.path(), base, admin);
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
    let (_, admin) = reserve_ports();
    let server = Server::spawn(scratch.path(), base, admin);
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
    let (base, admin) = reserve_ports();
    // 4 KiB segments: a few hundred small writes must rotate repeatedly.
    let server = Server::spawn_with(
        scratch.path(),
        base,
        admin,
        &["--wal-segment-target", "4096"],
    );
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
    let (_, admin) = reserve_ports();
    let server = Server::spawn(scratch.path(), base, admin);
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
