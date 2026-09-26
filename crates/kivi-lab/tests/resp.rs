//! RESP compatibility integration tests through the real server binary.
//!
//! Spawns `kivi-server` with `--redis-listen 127.0.0.1:0` (ephemeral,
//! race-free via `KIVI_READY`), then drives the RESP edge with the real
//! maintained `redis-rs` client plus raw sockets for error-shape checks.
//! Without the `redis-compat` feature this file compiles to nothing, and
//! native functionality never depends on it.

#![cfg(feature = "redis-compat")]

use std::io::{Read, Write};
use std::net::TcpStream;
use std::time::Duration;

use kivi_lab::process::Server;

const IO_TIMEOUT: Duration = Duration::from_secs(10);

/// Spawns an ephemeral RESP server. A binary without `redis-compat` fails
/// here loudly (never a silent skip): this file only compiles with the lab
/// `redis-compat` feature, and every test below explicitly requires RESP.
fn spawn_resp() -> Server {
    Server::spawn_ephemeral_resp().expect(
        "kivi-server binary has no RESP endpoint; rebuild with `-p kivi-server --features redis-compat`",
    )
}

/// Opens a redis-rs sync connection to the RESP edge.
fn redis_conn(addr: &str) -> redis::Connection {
    redis::Client::open(format!("redis://{addr}"))
        .expect("client opens")
        .get_connection_with_timeout(IO_TIMEOUT)
        .expect("connects fast")
}

/// Common client flows against Kivi RESP: connect, PING, SET/GET/DEL/
/// EXISTS, TTL/expiry, SETRANGE/GETRANGE, pipeline, reconnect.
#[test]
#[allow(clippy::too_many_lines)]
fn redis_rs_client_flows_against_kivi_resp() {
    let server = spawn_resp();
    let resp = server.resp_endpoint().expect("resp endpoint");
    let mut conn = redis_conn(&resp);

    // Connect + PING (redis-rs sends HELLO/CLIENT SETINFO on connect).
    let pong: String = redis::cmd("PING").query(&mut conn).expect("ping");
    assert_eq!(pong, "PONG");

    // SET / GET / EXISTS / DEL.
    let ok: String = redis::cmd("SET")
        .arg("rk:1")
        .arg("hello")
        .query(&mut conn)
        .expect("set");
    assert_eq!(ok, "OK");
    let value: Option<String> = redis::cmd("GET").arg("rk:1").query(&mut conn).expect("get");
    assert_eq!(value.as_deref(), Some("hello"));
    let exists: i64 = redis::cmd("EXISTS")
        .arg("rk:1")
        .query(&mut conn)
        .expect("exists");
    assert_eq!(exists, 1);
    let missing: Option<String> = redis::cmd("GET")
        .arg("rk:missing")
        .query(&mut conn)
        .expect("get missing");
    assert_eq!(missing, None);

    // TTL / expiry.
    let set: i64 = redis::cmd("EXPIRE")
        .arg("rk:1")
        .arg(100)
        .query(&mut conn)
        .expect("expire");
    assert_eq!(set, 1);
    let ttl: i64 = redis::cmd("TTL").arg("rk:1").query(&mut conn).expect("ttl");
    assert!((1..=100).contains(&ttl), "ttl in range, got {ttl}");
    let persisted: i64 = redis::cmd("PERSIST")
        .arg("rk:1")
        .query(&mut conn)
        .expect("persist");
    assert_eq!(persisted, 1);
    let ttl: i64 = redis::cmd("TTL").arg("rk:1").query(&mut conn).expect("ttl");
    assert_eq!(ttl, -1);

    // SETRANGE / GETRANGE / STRLEN.
    let len: i64 = redis::cmd("SETRANGE")
        .arg("rk:2")
        .arg(5)
        .arg("hi")
        .query(&mut conn)
        .expect("setrange");
    assert_eq!(len, 7);
    let slice: String = redis::cmd("GETRANGE")
        .arg("rk:2")
        .arg(5)
        .arg(6)
        .query(&mut conn)
        .expect("getrange");
    assert_eq!(slice, "hi");
    let strlen: i64 = redis::cmd("STRLEN")
        .arg("rk:2")
        .query(&mut conn)
        .expect("strlen");
    assert_eq!(strlen, 7);

    // SET options.
    let nx: Option<String> = redis::cmd("SET")
        .arg("rk:nx")
        .arg("1")
        .arg("NX")
        .query(&mut conn)
        .expect("nx");
    assert_eq!(nx.as_deref(), Some("OK"));
    let nx_again: Option<String> = redis::cmd("SET")
        .arg("rk:nx")
        .arg("2")
        .arg("NX")
        .query(&mut conn)
        .expect("nx again");
    assert_eq!(nx_again, None);
    let xx: Option<String> = redis::cmd("SET")
        .arg("rk:nx")
        .arg("2")
        .arg("XX")
        .query(&mut conn)
        .expect("xx");
    assert_eq!(xx.as_deref(), Some("OK"));

    // DEL.
    let deleted: i64 = redis::cmd("DEL").arg("rk:1").query(&mut conn).expect("del");
    assert_eq!(deleted, 1);
    let deleted: i64 = redis::cmd("DEL")
        .arg("rk:1")
        .query(&mut conn)
        .expect("del again");
    assert_eq!(deleted, 0);

    // Pipeline (redis-rs `pipe` batches N commands per round-trip).
    let (a, b): (Option<String>, Option<String>) = redis::pipe()
        .cmd("SET")
        .arg("rk:p1")
        .arg("v1")
        .ignore()
        .cmd("SET")
        .arg("rk:p2")
        .arg("v2")
        .ignore()
        .cmd("GET")
        .arg("rk:p1")
        .cmd("GET")
        .arg("rk:p2")
        .query(&mut conn)
        .expect("pipeline");
    assert_eq!(a.as_deref(), Some("v1"));
    assert_eq!(b.as_deref(), Some("v2"));

    // Reconnect on a fresh connection.
    drop(conn);
    let mut conn = redis_conn(&resp);
    let value: Option<String> = redis::cmd("GET")
        .arg("rk:p1")
        .query(&mut conn)
        .expect("get after reconnect");
    assert_eq!(value.as_deref(), Some("v1"));

    // Native endpoint still serves the same keys (shared engine).
    let native = kivi_client::NativeClient::new(kivi_client::ClientConfig {
        seeds: vec![server.endpoint()],
        namespace: kivi_types::NamespaceId::from_u64(1),
        ..kivi_client::ClientConfig::default()
    })
    .expect("native client builds");
    let back = native
        .get(&kivi_state::Key::from("rk:p1"))
        .expect("native get");
    assert_eq!(back.as_deref(), Some(&b"v1"[..]));
}

/// RESP3 (HELLO 3) flows: same profile over the RESP3 encoding.
#[test]
fn resp3_client_flows_against_kivi_resp() {
    let server = spawn_resp();
    let resp = server.resp_endpoint().expect("resp endpoint");
    let client = redis::Client::open(format!("redis://{resp}")).expect("opens");
    let mut conn = client
        .get_connection_with_timeout(IO_TIMEOUT)
        .expect("connects");
    // Upgrade this connection to RESP3 explicitly.
    let hello: std::collections::HashMap<String, redis::Value> = redis::cmd("HELLO")
        .arg(3)
        .query(&mut conn)
        .expect("hello 3");
    assert!(
        hello.contains_key("server"),
        "hello map has server: {hello:?}"
    );
    let ok: String = redis::cmd("SET")
        .arg("r3:1")
        .arg("v")
        .query(&mut conn)
        .expect("set");
    assert_eq!(ok, "OK");
    let value: Option<String> = redis::cmd("GET").arg("r3:1").query(&mut conn).expect("get");
    assert_eq!(value.as_deref(), Some("v"));
    let missing: Option<String> = redis::cmd("GET")
        .arg("r3:missing")
        .query(&mut conn)
        .expect("missing");
    assert_eq!(missing, None);
}

/// Sends one RESP array and reads one reply frame (single-threaded,
/// small replies: one read suffices).
fn exchange(stream: &mut TcpStream, parts: &[&[u8]]) -> Vec<u8> {
    let mut out = format!("*{}\r\n", parts.len()).into_bytes();
    for part in parts {
        out.extend_from_slice(format!("${}\r\n", part.len()).as_bytes());
        out.extend_from_slice(part);
        out.extend_from_slice(b"\r\n");
    }
    stream.write_all(&out).expect("writes");
    stream.flush().expect("flush");
    let mut buf = vec![0u8; 64 * 1024];
    let count = stream.read(&mut buf).expect("reads");
    buf[..count].to_vec()
}

/// Raw-socket checks for exact machine-visible error prefixes and shapes.
#[test]
fn raw_error_shapes_and_bootstrap() {
    let server = spawn_resp();
    let resp = server.resp_endpoint().expect("resp endpoint");
    let mut stream = TcpStream::connect(resp.as_str()).expect("connects");
    stream.set_read_timeout(Some(IO_TIMEOUT)).expect("timeout");
    stream.set_write_timeout(Some(IO_TIMEOUT)).expect("timeout");

    // Unknown command.
    let reply = exchange(&mut stream, &[b"FROBNICATE"]);
    assert!(reply.starts_with(b"-ERR unknown command"), "got {reply:?}");
    // Explicitly unsupported counter (never a fake GET-parse-SET).
    let reply = exchange(&mut stream, &[b"INCR", b"k"]);
    assert!(reply.starts_with(b"-ERR unsupported"), "got {reply:?}");
    // Multi-key DEL is rejected, never looped.
    let reply = exchange(&mut stream, &[b"DEL", b"a", b"b"]);
    assert!(reply.starts_with(b"-ERR"), "got {reply:?}");
    // SELECT 1 is rejected; SELECT 0 maps.
    let reply = exchange(&mut stream, &[b"SELECT", b"1"]);
    assert!(reply.starts_with(b"-ERR"), "got {reply:?}");
    let reply = exchange(&mut stream, &[b"SELECT", b"0"]);
    assert_eq!(&reply, b"+OK\r\n");
    // AUTH never succeeds silently.
    let reply = exchange(&mut stream, &[b"AUTH", b"x"]);
    assert!(reply.starts_with(b"-ERR"), "got {reply:?}");
    // Wrong arity shape.
    let reply = exchange(&mut stream, &[b"GET"]);
    assert!(
        reply.starts_with(b"-ERR wrong number of arguments"),
        "got {reply:?}"
    );
    // Bad integers.
    let reply = exchange(&mut stream, &[b"EXPIRE", b"k", b"nope"]);
    assert!(reply.starts_with(b"-ERR"), "got {reply:?}");
    // Conflicting SET options.
    let reply = exchange(&mut stream, &[b"SET", b"k", b"v", b"NX", b"XX"]);
    assert!(reply.starts_with(b"-ERR"), "got {reply:?}");
    // SET ... GET stays unsupported, never materialized.
    let reply = exchange(&mut stream, &[b"SET", b"k", b"v", b"GET"]);
    assert!(reply.starts_with(b"-ERR"), "got {reply:?}");
    // COMMAND COUNT agrees with the registry; COMMAND LIST names SET.
    let reply = exchange(&mut stream, &[b"COMMAND", b"COUNT"]);
    assert!(reply.starts_with(b":"), "got {reply:?}");
    let reply = exchange(&mut stream, &[b"COMMAND", b"LIST"]);
    assert!(reply.windows(3).any(|w| w == b"SET"), "got {reply:?}");
    // Binary-safe key round-trip over the raw socket.
    let key = [0x00, 0xFF, 0x10, 0x00];
    let reply = exchange(&mut stream, &[b"SET", &key, b"bin"]);
    assert_eq!(&reply, b"+OK\r\n");
    let reply = exchange(&mut stream, &[b"GET", &key]);
    assert_eq!(&reply, b"$3\r\nbin\r\n");

    // Admin visibility: /v1/redis reports the frontend and its counters.
    let mut admin = TcpStream::connect(server.admin_endpoint()).expect("admin connects");
    admin.set_read_timeout(Some(IO_TIMEOUT)).expect("timeout");
    admin
        .write_all(b"GET /v1/redis HTTP/1.1\r\nhost: x\r\nconnection: close\r\n\r\n")
        .expect("admin writes");
    let mut body = Vec::new();
    admin.read_to_end(&mut body).expect("admin reads");
    let text = String::from_utf8_lossy(&body);
    assert!(text.contains("200"), "admin /v1/redis: {text:?}");
    assert!(text.contains("\"enabled\":true"), "admin body: {text:?}");
    assert!(text.contains("\"connections\""), "admin body: {text:?}");
}

/// `redis-cli` smoke compatibility when the binary exists in the
/// environment (never required: skipped with a notice when absent).
#[test]
#[ignore = "needs an externally installed redis-cli; run explicitly to smoke it"]
fn redis_cli_smoke_when_available() {
    if std::process::Command::new("redis-cli")
        .arg("--version")
        .output()
        .is_err()
    {
        eprintln!("SKIP redis-cli smoke: redis-cli not installed");
        return;
    }
    let server = spawn_resp();
    let resp = server.resp_endpoint().expect("resp endpoint");
    let port = resp.rsplit(':').next().expect("resp port").to_owned();
    let run = |args: &[&str]| -> String {
        let output = std::process::Command::new("redis-cli")
            .arg("-p")
            .arg(&port)
            .args(args)
            .output()
            .expect("redis-cli runs");
        assert!(output.status.success(), "redis-cli {args:?}: {output:?}");
        String::from_utf8_lossy(&output.stdout).trim().to_owned()
    };
    assert_eq!(run(&["PING"]), "PONG");
    assert_eq!(run(&["SET", "cli:foo", "bar"]), "OK");
    assert_eq!(run(&["GET", "cli:foo"]), "bar");
    assert_eq!(run(&["DEL", "cli:foo"]), "1");
    eprintln!("redis-cli smoke passed against Kivi RESP");
}

/// Normalized observable string for one command on one connection.
/// Compare wire-normalized debug strings (nil vs empty vs zero stay
/// distinct here by construction of the profile).
fn run_reference(conn: &mut redis::Connection, argv: &[&str]) -> String {
    let mut cmd = redis::cmd(argv[0]);
    for arg in &argv[1..] {
        cmd.arg(*arg);
    }
    format!("{:?}", cmd.query::<redis::Value>(conn))
}

/// Reference-conformance mode: when `KIVI_REDIS_REFERENCE_URL` names a real
/// Redis endpoint, runs the exact-profile vectors against both Kivi RESP
/// and real Redis and compares normalized observable results. Only exact
/// commands compare; deviations and unsupported forms have their own tests
/// and never fail this harness. Ignored by default (no Redis required for
/// ordinary `cargo test`).
#[test]
#[ignore = "needs KIVI_REDIS_REFERENCE_URL pointing at a real Redis; run explicitly"]
fn reference_conformance_when_configured() {
    let Ok(url) = std::env::var("KIVI_REDIS_REFERENCE_URL") else {
        eprintln!("SKIP reference conformance: KIVI_REDIS_REFERENCE_URL unset");
        return;
    };
    let server = spawn_resp();
    let resp = server.resp_endpoint().expect("resp endpoint");
    let kivi = redis::Client::open(format!("redis://{resp}")).expect("kivi opens");
    let reference = redis::Client::open(url.as_str()).expect("reference opens");
    let mut kivi = kivi
        .get_connection_with_timeout(IO_TIMEOUT)
        .expect("kivi connects");
    let mut real = reference
        .get_connection_with_timeout(IO_TIMEOUT)
        .expect("real connects");

    // Exact-profile vectors (single-key, bare forms only).
    let vectors: &[&[&str]] = &[
        &["SET", "cf:k", "v"],
        &["GET", "cf:k"],
        &["GET", "cf:missing"],
        &["EXISTS", "cf:k"],
        &["EXISTS", "cf:missing"],
        &["STRLEN", "cf:k"],
        &["STRLEN", "cf:missing"],
        &["SETRANGE", "cf:sr", "0", "hello"],
        &["GETRANGE", "cf:sr", "0", "3"],
        &["GETRANGE", "cf:sr", "-3", "-1"],
        &["EXPIRE", "cf:k", "100"],
        &["TTL", "cf:k"],
        &["PTTL", "cf:k"],
        &["PERSIST", "cf:k"],
        &["TTL", "cf:k"],
        &["DEL", "cf:k"],
        &["DEL", "cf:k"],
    ];
    for argv in vectors {
        // Fresh keys per side where needed: SETRANGE/EXPIRE mutate, so run
        // setup on both sides first (vectors are ordered for this).
        let left = run_reference(&mut kivi, argv);
        let right = run_reference(&mut real, argv);
        // TTL values race wall-clock between the two servers: compare
        // shape (both in 1..=100 or both errors), not exact digits.
        if argv[0] == "TTL" || argv[0] == "PTTL" {
            let left_ok = left.contains("int(") && !left.contains("int(-");
            let right_ok = right.contains("int(") && !right.contains("int(-");
            assert_eq!(
                left_ok, right_ok,
                "TTL shape differs for {argv:?}: {left} vs {right}"
            );
            continue;
        }
        assert_eq!(left, right, "conformance differs for {argv:?}");
    }
    eprintln!("reference conformance passed for {} vectors", vectors.len());
}

#[test]
fn medium_value_reads_and_ranges_remain_available() {
    let server = spawn_resp();
    let resp = server.resp_endpoint().expect("resp endpoint");
    let key = b"medium-range-regression";
    let payload = kivi_lab::workload::fill_pattern(1024, 7);
    let mut immediate = kivi_lab::resp_client::RespClient::connect(&resp).expect("connects");
    let immediate_key = b"medium-range-immediate";
    let immediate_set = immediate
        .round_trip(&[b"SET", immediate_key, payload.as_slice()])
        .expect("immediate set");
    assert_eq!(
        immediate_set,
        kivi_lab::resp_client::Reply::Simple(b"OK".to_vec())
    );
    let immediate_range = immediate
        .round_trip(&[b"SETRANGE", immediate_key, b"0", b"v"])
        .expect("immediate range set");
    assert_eq!(immediate_range, kivi_lab::resp_client::Reply::Integer(1024));
    drop(immediate);
    for _ in 0..24 {
        let mut client = kivi_lab::resp_client::RespClient::connect(&resp).expect("connects");
        let set = client
            .round_trip(&[b"SET", key, payload.as_slice()])
            .expect("set");
        assert_eq!(set, kivi_lab::resp_client::Reply::Simple(b"OK".to_vec()));
        for _ in 0..32 {
            let get = client.round_trip(&[b"GET", key]).expect("get");
            assert_eq!(
                get,
                kivi_lab::resp_client::Reply::Bulk(Some(payload.clone()))
            );
        }
        let range = client
            .round_trip(&[b"GETRANGE", key, b"0", b"63"])
            .expect("range");
        assert_eq!(
            range,
            kivi_lab::resp_client::Reply::Bulk(Some(payload[..64].to_vec()))
        );
        let range_set = client
            .round_trip(&[b"SETRANGE", key, b"0", b"v"])
            .expect("range set");
        assert_eq!(range_set, kivi_lab::resp_client::Reply::Integer(1024));
        let expire = client.round_trip(&[b"EXPIRE", key, b"60"]).expect("expire");
        assert_eq!(expire, kivi_lab::resp_client::Reply::Integer(1));
        let deleted = client.round_trip(&[b"DEL", key]).expect("delete");
        assert_eq!(deleted, kivi_lab::resp_client::Reply::Integer(1));
    }
}
