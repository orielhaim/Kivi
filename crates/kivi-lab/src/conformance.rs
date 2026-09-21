//! Differential-testing helpers: Kivi RESP vs reference Redis.
//!
//! Every run works under a unique key prefix (`kivi-lab:<run-id>:`) and
//! cleans only keys it owns — never `FLUSHALL`/`FLUSHDB` on an externally
//! supplied instance. Comparisons normalize only representation differences
//! with identical semantics (RESP2 vs RESP3 null forms); genuine semantic
//! differences must surface, never hide in the harness.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::resp_client::{ClientError, Reply, RespClient};

/// Environment variable naming the reference Redis URL. Unset means the
/// caller did not opt into external-Redis tests: harness entry points skip
/// with a notice instead of failing.
pub const REDIS_URL_ENV: &str = "KIVI_LAB_REDIS_URL";

/// Default reference endpoint used when the operator passes none
/// explicitly (the task's local instance).
pub const DEFAULT_REDIS_URL: &str = "redis://localhost:6379";

/// Monotonic run-id source for key-prefix isolation.
static RUN_IDS: AtomicU64 = AtomicU64::new(0);

/// Allocates a unique run scope: keys live under `kivi-lab:<id>:` and only
/// that prefix is ever cleaned.
#[must_use]
pub fn run_scope() -> String {
    // pid + counter: unique across parallel test binaries too. No wall
    // clock — an atomic process identity is sufficient for uniqueness.
    let id = RUN_IDS.fetch_add(1, Ordering::Relaxed);
    format!("kivi-lab:{}-{id}:", std::process::id())
}

/// Resolves the reference Redis URL: explicit argument wins, then
/// `KIVI_LAB_REDIS_URL`, then [`DEFAULT_REDIS_URL`].
#[must_use]
pub fn redis_url(explicit: Option<&str>) -> String {
    explicit
        .map(str::to_owned)
        .or_else(|| std::env::var(REDIS_URL_ENV).ok())
        .unwrap_or_else(|| DEFAULT_REDIS_URL.to_owned())
}

/// Whether external-Redis tests opted in (`KIVI_LAB_REDIS_URL` is set).
/// Callers skip with a notice when this is false; a set-but-unreachable
/// URL is a hard failure, never a skip.
#[must_use]
pub fn redis_opted_in() -> bool {
    std::env::var(REDIS_URL_ENV).is_ok()
}

/// Reference Redis configuration relevant to interpreting results.
/// Persistence posture is *reported*, never assumed: memory-only
/// acknowledgements must never be presented as durability-equivalent.
#[derive(Debug, Clone, Default)]
pub struct RedisConfig {
    /// `redis_version` from `INFO server`.
    pub version: String,
    /// `appendonly yes/no`.
    pub appendonly: String,
    /// `appendfsync` mode.
    pub appendfsync: String,
    /// `save` rules (may be empty).
    pub save: String,
    /// `maxmemory` policy summary.
    pub maxmemory_policy: String,
    /// Raw `INFO persistence` section for the report.
    pub persistence_section: String,
}

/// Reads reference configuration over a raw client (`INFO` + `CONFIG GET`).
///
/// # Errors
///
/// Returns [`ClientError`] when the instance is unreachable or answers
/// malformed data (a set-but-unreachable URL fails loudly by design).
pub fn redis_config(client: &mut RespClient) -> Result<RedisConfig, ClientError> {
    let info = command_string(client, &[b"INFO", b"server"])?;
    let persistence = command_string(client, &[b"INFO", b"persistence"])?;
    let mut get = |name: &str| -> String {
        let argv: Vec<&[u8]> = vec![b"CONFIG", b"GET", name.as_bytes()];
        match client.round_trip(&argv) {
            Ok(Reply::Array(Some(items))) if items.len() == 2 => match &items[1] {
                Reply::Bulk(Some(value)) => String::from_utf8_lossy(value).into_owned(),
                other => format!("{other:?}"),
            },
            Ok(other) => format!("{other:?}"),
            Err(error) => format!("unreadable: {error}"),
        }
    };
    // Persistence posture comes from CONFIG (authoritative across versions;
    // INFO field names drift between releases), never assumed.
    let one_line = |value: String| value.replace(['\r', '\n'], " ").trim().to_owned();
    Ok(RedisConfig {
        version: parse_info(&info)
            .get("redis_version")
            .cloned()
            .unwrap_or_default(),
        appendonly: one_line(get("appendonly")),
        appendfsync: one_line(get("appendfsync")),
        save: one_line(get("save")),
        maxmemory_policy: one_line(get("maxmemory-policy")),
        persistence_section: persistence,
    })
}

/// Parses `INFO` `key:value` lines into a map (sections and comments skipped).
#[must_use]
pub fn parse_info(info: &str) -> HashMap<String, String> {
    info.lines()
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .filter_map(|line| {
            line.split_once(':')
                .map(|(key, value)| (key.trim().to_owned(), value.trim().to_owned()))
        })
        .collect()
}

fn command_string(client: &mut RespClient, argv: &[&[u8]]) -> Result<String, ClientError> {
    match client.round_trip(argv)? {
        Reply::Bulk(Some(value)) | Reply::Simple(value) => {
            Ok(String::from_utf8_lossy(&value).into_owned())
        }
        other => Err(ClientError::Malformed(format!(
            "expected string, got {other:?}"
        ))),
    }
}

/// Every key name the matrix and [`setup_edge_keys`] touch (unprefixed).
/// Cleanup deletes exactly these under the run prefix — no `SCAN` (outside
/// the profile), no multi-key `DEL` (rejected, never faked), and especially
/// no `FLUSHALL`/`FLUSHDB` on an externally supplied instance.
#[must_use]
pub fn conformance_keys() -> Vec<&'static str> {
    vec![
        "empty",
        "bin",
        "k",
        "k2",
        "k3",
        "k4",
        "nx",
        "xx-missing",
        "ex",
        "px",
        "exat",
        "pxat",
        "keep",
        "clr",
        "sr",
        "pad",
        "missing",
    ]
}

/// Deletes every matrix key under `prefix` with single-key `DEL`s and
/// returns the number removed (missing keys count zero, never an error).
/// Works on both sides (exact profile only).
///
/// # Errors
///
/// Returns [`ClientError`] on transport failure.
pub fn cleanup_prefix(client: &mut RespClient, prefix: &str) -> Result<u64, ClientError> {
    let mut removed = 0u64;
    for key in conformance_keys() {
        let name = format!("{prefix}{key}");
        let argv: Vec<&[u8]> = vec![b"DEL", name.as_bytes()];
        match client.round_trip(&argv)? {
            Reply::Integer(count) => {
                removed += u64::try_from(count.max(0)).unwrap_or(u64::MAX);
            }
            other => {
                return Err(ClientError::Malformed(format!("DEL failed: {other:?}")));
            }
        }
    }
    Ok(removed)
}

/// Deletes every key under an arbitrary `prefix` via `SCAN` plus
/// single-key `DEL`s. Redis-only (`SCAN` is outside the Kivi profile, so
/// this errors loudly against Kivi): benchmark hygiene for shared
/// reference instances. Never `FLUSHALL`/`FLUSHDB`.
///
/// # Errors
///
/// Returns [`ClientError`] on transport failure or non-Redis replies.
pub fn cleanup_prefix_scan(client: &mut RespClient, prefix: &str) -> Result<u64, ClientError> {
    let mut removed = 0u64;
    let mut cursor = b"0".to_vec();
    loop {
        let pattern = format!("{prefix}*");
        let argv: Vec<&[u8]> = vec![
            b"SCAN",
            &cursor,
            b"MATCH",
            pattern.as_bytes(),
            b"COUNT",
            b"100",
        ];
        let (next, keys) = match client.round_trip(&argv)? {
            Reply::Array(Some(items)) if items.len() == 2 => {
                let next = match &items[0] {
                    Reply::Bulk(Some(value)) => value.clone(),
                    other => {
                        return Err(ClientError::Malformed(format!("bad cursor: {other:?}")));
                    }
                };
                let keys = match &items[1] {
                    Reply::Array(Some(keys)) => keys.clone(),
                    other => {
                        return Err(ClientError::Malformed(format!("bad keys: {other:?}")));
                    }
                };
                (next, keys)
            }
            other => {
                return Err(ClientError::Malformed(format!("SCAN failed: {other:?}")));
            }
        };
        for key in &keys {
            let Reply::Bulk(Some(name)) = key else {
                continue;
            };
            let argv: Vec<&[u8]> = vec![b"DEL", name];
            match client.round_trip(&argv)? {
                Reply::Integer(count) => {
                    removed += u64::try_from(count.max(0)).unwrap_or(u64::MAX);
                }
                other => {
                    return Err(ClientError::Malformed(format!("DEL failed: {other:?}")));
                }
            }
        }
        cursor = next;
        if cursor == b"0" {
            break;
        }
    }
    Ok(removed)
}

/// Normalized observable outcome for differential comparison. Only
/// representation differences with identical semantics collapse here
/// (RESP2 `$-1` vs RESP3 `_` nulls); everything else compares exactly.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Observable {
    /// Status/simple string (uppercased ASCII: `OK`, `PONG`).
    Status(String),
    /// Bulk bytes (empty and nil stay distinct from each other).
    Bulk(Vec<u8>),
    /// Null (nil bulk or RESP3 null — identical semantics).
    Nil,
    /// Empty string (distinct from nil by construction of the profile).
    Empty,
    /// Integer.
    Integer(i64),
    /// Error with normalized prefix (`WRONGTYPE`, `ERR`, ...).
    Error(String),
    /// Anything else, debug-rendered (arrays, maps).
    Other(String),
}

impl Observable {
    /// Normalizes one raw reply. `empty_is_nil` is false: the harness never
    /// conflates empty with nil (GETRANGE-miss vs GET-miss differ).
    #[must_use]
    pub fn of(reply: &Reply) -> Self {
        match reply {
            Reply::Simple(value) => Self::Status(String::from_utf8_lossy(value).to_uppercase()),
            Reply::Error(value) => Self::Error(
                String::from_utf8_lossy(value)
                    .split_whitespace()
                    .next()
                    .unwrap_or("")
                    .to_uppercase(),
            ),
            Reply::Integer(value) => Self::Integer(*value),
            Reply::Bulk(None) => Self::Nil,
            Reply::Bulk(Some(value)) if value.is_empty() => Self::Empty,
            Reply::Bulk(Some(value)) => Self::Bulk(value.clone()),
            Reply::Array(_) => Self::Other(format!("{reply:?}")),
        }
    }
}

/// One exact-profile vector: argv plus how to normalize wall-clock races.
/// TTL-family replies race the clock between two servers, so those compare
/// shape (sign class) rather than digits; everything else compares exactly.
#[derive(Debug, Clone)]
pub struct ExactVector {
    /// Command argv (prefix applied to key positions by the caller via
    /// [`ExactVector::render`]).
    pub argv: Vec<String>,
    /// Indexes into `argv` holding keys (0-based, excluding `argv[0]`).
    pub keys: Vec<usize>,
    /// Whether the reply races wall-clock (TTL/PTTL only).
    pub clock_racy: bool,
}

impl ExactVector {
    /// Renders argv with `prefix` prepended to each key position.
    #[must_use]
    pub fn render(&self, prefix: &str) -> Vec<Vec<u8>> {
        self.argv
            .iter()
            .enumerate()
            .map(|(index, arg)| {
                if index > 0 && self.keys.contains(&index) {
                    format!("{prefix}{arg}").into_bytes()
                } else {
                    arg.as_bytes().to_vec()
                }
            })
            .collect()
    }
}

/// The exact-profile vector matrix: every command Kivi marks `Exact`, plus
/// the supported `SET` options. Key positions are marked for prefixing.
#[must_use]
pub fn exact_vectors() -> Vec<ExactVector> {
    let key = |argv: &[&str]| ExactVector {
        argv: argv.iter().map(ToString::to_string).collect(),
        keys: vec![1],
        clock_racy: false,
    };
    let mut vectors = vec![
        key(&["SET", "k", "v"]),
        key(&["GET", "k"]),
        key(&["GET", "missing"]),
        key(&["EXISTS", "k"]),
        key(&["EXISTS", "missing"]),
        key(&["STRLEN", "k"]),
        key(&["STRLEN", "missing"]),
        key(&["STRLEN", "empty"]),
        key(&["STRLEN", "bin"]),
        key(&["GETRANGE", "bin", "0", "3"]),
        key(&["GETRANGE", "bin", "-4", "-1"]),
        key(&["GETRANGE", "empty", "0", "3"]),
        key(&["GET", "bin"]),
        key(&["GET", "empty"]),
        key(&["SETRANGE", "sr", "0", "hello"]),
        key(&["GETRANGE", "sr", "0", "3"]),
        key(&["GETRANGE", "sr", "-3", "-1"]),
        key(&["GETRANGE", "sr", "0", "-1"]),
        key(&["GETRANGE", "sr", "10", "100"]),
        key(&["GETRANGE", "missing", "0", "3"]),
        key(&["SETRANGE", "pad", "5", "hi"]),
        key(&["GETRANGE", "pad", "0", "1"]),
        key(&["GETRANGE", "pad", "5", "6"]),
        key(&["STRLEN", "pad"]),
        key(&["TTL", "missing"]),
        key(&["PTTL", "missing"]),
        key(&["EXPIRETIME", "missing"]),
        key(&["PEXPIRETIME", "missing"]),
        key(&["PERSIST", "missing"]),
        key(&["EXPIRE", "missing", "100"]),
        key(&["TTL", "empty"]),
        key(&["EXPIRE", "k", "100"]),
        key(&["TTL", "k"]),
        key(&["PTTL", "k"]),
        key(&["EXPIRETIME", "k"]),
        key(&["PEXPIRETIME", "k"]),
        key(&["PERSIST", "k"]),
        key(&["TTL", "k"]),
        key(&["PEXPIRE", "k2", "100000"]),
        key(&["PTTL", "k2"]),
        key(&["EXPIREAT", "k3", "4102444800"]),
        key(&["EXPIRETIME", "k3"]),
        key(&["PEXPIREAT", "k4", "4102444800000"]),
        key(&["PEXPIRETIME", "k4"]),
        key(&["SET", "nx", "1", "NX"]),
        key(&["SET", "nx", "2", "NX"]),
        key(&["SET", "nx", "2", "XX"]),
        key(&["SET", "xx-missing", "1", "XX"]),
        key(&["SET", "ex", "v", "EX", "100"]),
        key(&["TTL", "ex"]),
        key(&["SET", "px", "v", "PX", "100000"]),
        key(&["PTTL", "px"]),
        key(&["SET", "exat", "v", "EXAT", "4102444800"]),
        key(&["EXPIRETIME", "exat"]),
        key(&["SET", "pxat", "v", "PXAT", "4102444800000"]),
        key(&["PEXPIRETIME", "pxat"]),
        key(&["SET", "keep", "v"]),
        key(&["EXPIRE", "keep", "100"]),
        key(&["SET", "keep", "v2", "KEEPTTL"]),
        key(&["TTL", "keep"]),
        key(&["SET", "clr", "v"]),
        key(&["EXPIRE", "clr", "100"]),
        key(&["SET", "clr", "v2"]),
        key(&["TTL", "clr"]),
        key(&["DEL", "k"]),
        key(&["DEL", "k"]),
        key(&["GET", "k"]),
    ];
    for vector in &mut vectors {
        // TTL/PTTL always race the clock. EXPIRETIME/PEXPIRETIME race it too
        // when the key's expiry came from a *relative* option (the two runs
        // execute sequentially, so absolute stamps differ by the inter-run
        // gap); keys with fixed absolute expiries compare exactly.
        let absolute_keys = ["k3", "k4", "exat", "pxat"];
        vector.clock_racy = match vector.argv[0].as_str() {
            "TTL" | "PTTL" => true,
            "EXPIRETIME" | "PEXPIRETIME" => !absolute_keys.contains(&vector.argv[1].as_str()),
            _ => false,
        };
    }
    vectors
}

/// Outcome of one vector on one side.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VectorOutcome {
    /// Rendered argv (for the report).
    pub argv: Vec<String>,
    /// Normalized observable.
    pub observed: Observable,
}

/// Runs every exact vector against one endpoint under `prefix`.
///
/// # Errors
///
/// Returns [`ClientError`] on transport failure or malformed replies.
pub fn run_vectors(
    client: &mut RespClient,
    prefix: &str,
) -> Result<Vec<VectorOutcome>, ClientError> {
    let mut outcomes = Vec::new();
    for vector in exact_vectors() {
        let rendered = vector.render(prefix);
        let argv: Vec<&[u8]> = rendered.iter().map(Vec::as_slice).collect();
        let reply = client.round_trip(&argv)?;
        outcomes.push(VectorOutcome {
            argv: vector.argv.clone(),
            observed: Observable::of(&reply),
        });
    }
    Ok(outcomes)
}

/// Compares two vector runs positionally. Clock-racy TTL/PTTL replies
/// compare sign class (positive vs -1 vs -2), never digits.
///
/// Returns the list of mismatches `(argv, kivi, reference)`; empty means
/// exact agreement across the whole matrix.
#[must_use]
pub fn diff_vectors(
    kivi: &[VectorOutcome],
    reference: &[VectorOutcome],
) -> Vec<(Vec<String>, Observable, Observable)> {
    let clock_shapes = |observed: &Observable| match observed {
        Observable::Integer(value) if *value > 0 => "positive",
        Observable::Integer(value) => {
            if *value == -1 {
                "immortal"
            } else {
                "missing"
            }
        }
        _ => "other",
    };
    kivi.iter()
        .zip(reference.iter())
        .enumerate()
        .filter_map(|(index, (left, right))| {
            let racy = exact_vectors()
                .get(index)
                .is_some_and(|vector| vector.clock_racy);
            let same = if racy {
                clock_shapes(&left.observed) == clock_shapes(&right.observed)
            } else {
                left.observed == right.observed
            };
            (!same).then(|| {
                (
                    left.argv.clone(),
                    left.observed.clone(),
                    right.observed.clone(),
                )
            })
        })
        .collect()
}

/// Seeds edge-case keys both sides need before the matrix runs: an empty
/// value and a binary value (nul bytes, 0xFF, high bytes).
///
/// # Errors
///
/// Returns [`ClientError`] on transport failure or a non-OK setup reply.
pub fn setup_edge_keys(client: &mut RespClient, prefix: &str) -> Result<(), ClientError> {
    let empty = format!("{prefix}empty");
    let bin = format!("{prefix}bin");
    for argv in [
        vec![b"SET".as_slice(), empty.as_bytes(), b""],
        vec![
            b"SET".as_slice(),
            bin.as_bytes(),
            b"\x00\xff\x10\x00binary\xfe",
        ],
    ] {
        match client.round_trip(&argv)? {
            Reply::Simple(_) => {}
            other => {
                return Err(ClientError::Malformed(format!("setup failed: {other:?}")));
            }
        }
    }
    Ok(())
}
