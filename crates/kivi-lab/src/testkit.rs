//! Shared scaffolding for the lab's integration suites.
//!
//! Every suite needs the same handful of things: an in-process engine on
//! ephemeral loopback ports, a native client bound to it, a deterministic
//! filler, key naming, the admin-plane client, and the split/merge round
//! trip. Those live here once instead of being re-typed (and subtly
//! re-tuned) in a dozen test binaries.
//!
//! Every helper here panics rather than returning an error: a suite that
//! cannot write its key has already failed, and threading `Result` through
//! the assertions only hides which key broke.
//!
//! Anything that encodes a *policy decision about what a suite should
//! assert* stays in that suite.

use std::time::Duration;

use kivi_client::{ClientConfig, NativeClient};
use kivi_engine::{
    ChunkFabricConfig, ConnLimits, DurabilityMode, EngineConfig, EngineNetwork, FabricConfig,
    LocalEngine, Placement, TurnBudget,
};
use kivi_state::Key;
use kivi_tablet::{DirectorySnapshot, HashPrefix, PartitionRange};
use kivi_types::{
    ClusterId, Expiry, NamespaceId, NodeId, NodeIncarnation, TabletEpoch, TabletId, WallTimestamp,
    WorkerId, WriteGuardGeneration,
};

use crate::cluster::Cluster;

/// The single namespace every lab harness drives.
pub const NS: NamespaceId = NamespaceId::from_u64(1);

/// A single-tablet directory covering the whole keyspace.
///
/// # Panics
///
/// Panics if the genesis snapshot or its activation is rejected, which only
/// happens if the bootstrap invariants themselves are broken.
#[must_use]
pub fn directory() -> DirectorySnapshot {
    let tablet = TabletId::from_u64(1);
    DirectorySnapshot::bootstrap(
        NS,
        tablet,
        PartitionRange::Hash(HashPrefix::new(0, 0).expect("root")),
        TabletEpoch::INITIAL,
        WriteGuardGeneration::INITIAL,
    )
    .expect("genesis")
    .stage(tablet)
    .and_then(|snapshot| snapshot.activate(tablet))
    .expect("active root")
}

/// One tablet, one worker.
#[must_use]
pub fn placement() -> Placement {
    Placement::new([(TabletId::from_u64(1), WorkerId::from_u64(0))])
}

/// Loopback reactor network on ephemeral ports. `resp: None` - these
/// harnesses exercise the native protocol, never the RESP edge.
#[must_use]
pub fn network() -> EngineNetwork {
    EngineNetwork {
        base_port: 0,
        ports: Vec::new(),
        bind_ip: [127, 0, 0, 1].into(),
        max_frame: kivi_protocol::DEFAULT_MAX_FRAME,
        placement: kivi_engine::ThreadPlacement::unbound(),
        node_id: NodeId::from_u64(1),
        cluster_id: ClusterId::from_u128(1),
        incarnation: NodeIncarnation::INITIAL,
        conn: ConnLimits::default(),
        turn: TurnBudget::default(),
        resp: None,
    }
}

/// The `EngineConfig` every in-process harness starts from.
#[must_use]
fn base_config() -> EngineConfig {
    EngineConfig {
        namespace: NS,
        hardware: kivi_engine::HardwareConfig::default(),
        directory: directory(),
        placement: placement(),
        worker_count: 2,
        request_capacity: 128,
        chunks: ChunkFabricConfig::default(),
        fabric: FabricConfig::default(),
        network: Some(network()),
        durability: DurabilityMode::Ephemeral,
    }
}

/// One in-process engine on ephemeral loopback ports, holding everything in
/// memory.
///
/// # Panics
///
/// Panics if the engine cannot bind its ephemeral ports.
#[must_use]
pub fn start_ephemeral(chunks: ChunkFabricConfig) -> LocalEngine {
    start(EngineConfig {
        chunks,
        ..base_config()
    })
}

/// Starts an in-process engine from a full config.
///
/// # Panics
///
/// Panics if the engine cannot start.
#[must_use]
pub fn start(config: EngineConfig) -> LocalEngine {
    LocalEngine::start(config).expect("networked engine starts")
}

/// An ephemeral engine paired with a client already bound to it. The engine
/// lives for the rest of the process; suites that need an explicit shutdown
/// hold on to it.
#[must_use]
pub fn start_ephemeral_client() -> (LocalEngine, NativeClient) {
    let engine = start_ephemeral(ChunkFabricConfig::default());
    let client = client_for(&engine);
    (engine, client)
}

/// A native client seeded from the engine's first worker.
///
/// # Panics
///
/// Panics if the client config is rejected.
#[must_use]
pub fn client_for(engine: &LocalEngine) -> NativeClient {
    NativeClient::new(ClientConfig {
        seeds: vec![engine.worker_addrs()[0].to_string()],
        namespace: NS,
        ..ClientConfig::default()
    })
    .expect("client builds")
}

/// Deterministic pseudo-random bytes (splitmix64). Stable for a given
/// `(len, seed)`; a regression that corrupts a byte shows up as a
/// byte-exact comparison failure, not as a flaky count.
#[must_use]
pub fn fill(len: usize, seed: u64) -> Vec<u8> {
    let mut state = seed;
    let mut out = vec![0u8; len];
    for byte in &mut out {
        state = state
            .wrapping_add(0x9E37_79B9_7F4A_7C15)
            .wrapping_mul(0xBF58_476D_1CE4_E5B9);
        state ^= state >> 29;
        *byte = (state % 251) as u8;
    }
    out
}

/// The lab's key type from a string name.
#[must_use]
pub fn key(name: &str) -> Key {
    Key::from(name)
}

/// Runs one generated workload op against a client, normalising every op to
/// a unit result so a mixed benchmark can assert on the error only.
///
/// `CounterAdd` is a no-op: a mix that selects it still has to succeed, and
/// the counter path is covered deliberately elsewhere.
///
/// # Errors
///
/// Returns the client's error when the op does not succeed.
pub fn run_workload_op(
    client: &NativeClient,
    op: &crate::workload::WorkloadOp,
    value: &bytes::Bytes,
) -> Result<(), kivi_client::ClientError> {
    use crate::workload::WorkloadOp;
    match op {
        WorkloadOp::Get(key) => client.get(&Key::from(key.as_str())).map(drop),
        WorkloadOp::Set { key, .. } => client
            .set(&Key::from(key.as_str()), value.clone())
            .map(drop),
        WorkloadOp::Delete(key) => client.delete(&Key::from(key.as_str())).map(drop),
        WorkloadOp::Exists(key) => client.exists(&Key::from(key.as_str())).map(drop),
        WorkloadOp::GetRange(key) => client.get_range(&Key::from(key.as_str()), 0, 64).map(drop),
        WorkloadOp::SetRange(key) => client
            .set_range(&Key::from(key.as_str()), 0, bytes::Bytes::from_static(b"v"))
            .map(drop),
        WorkloadOp::Expire(key) => client
            .expire_at(
                &Key::from(key.as_str()),
                WallTimestamp::from_micros(FAR_FUTURE_US),
            )
            .map(drop),
        WorkloadOp::Ttl(key) => client.get_expiry(&Key::from(key.as_str())).map(drop),
        WorkloadOp::CounterAdd(_) => Ok(()),
    }
}

/// The wall-clock stamp behind [`far_future`].
const FAR_FUTURE_US: i64 = 9_000_000_000_000_000;

/// Hex form of a raw key, truncated to its first 8 bytes. Diagnostic text
/// only, so a long key cannot swamp the surrounding message.
///
/// Not a full encoding: anything that must round-trip through a hex
/// *parameter* (a redundancy `hash_hex`, say) needs the whole digest.
#[must_use]
pub fn hex_key(key: &[u8]) -> String {
    use core::fmt::Write as _;
    key.iter().take(8).fold(String::new(), |mut out, byte| {
        let _ = write!(out, "{byte:02x}");
        out
    })
}

/// Sorted-latency report printed to stderr: `name: n=… p50=… p95=… p99=…`.
///
/// Report only, so a latency count in the millions still ranks correctly
/// (52 bits of mantissa) and the index conversion truncates deliberately.
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
pub fn summarize(name: &str, mut latencies: Vec<Duration>) {
    latencies.sort_unstable();
    let at = |q: f64| {
        let rank = (q * latencies.len() as f64)
            .ceil()
            .max(1.0)
            .min(latencies.len() as f64);
        let index = rank as usize - 1;
        latencies[index].as_secs_f64() * 1000.0
    };
    eprintln!(
        "{name}: n={} p50={:.2}ms p95={:.2}ms p99={:.2}ms",
        latencies.len(),
        at(0.50),
        at(0.95),
        at(0.99),
    );
}

/// Far-future expiry stamp. Real wall time drives expiry here, so the
/// wall-clock stubs the unit tests use do not apply.
#[must_use]
pub fn far_future() -> Expiry {
    Expiry::at(WallTimestamp::from_micros(FAR_FUTURE_US))
}

/// A committed `set`.
///
/// # Panics
///
/// Panics with the key name if the write does not commit.
pub fn put(client: &NativeClient, name: &str, value: &[u8]) {
    client
        .set(&key(name), bytes::Bytes::copy_from_slice(value))
        .unwrap_or_else(|error| panic!("set {name} commits: {error:?}"));
}

/// The committed value of `name`.
///
/// # Panics
///
/// Panics on read error or on absence, so a lost key is a failure rather
/// than a silent `None`.
#[must_use]
pub fn get(client: &NativeClient, name: &str) -> Vec<u8> {
    client
        .get(&key(name))
        .unwrap_or_else(|error| panic!("get {name} reads: {error:?}"))
        .unwrap_or_else(|| panic!("{name} present"))
        .to_vec()
}

/// A `set` that retries through leader changes and short unavailability, for
/// traffic a suite fires *while* it breaks the cluster.
///
/// # Panics
///
/// Panics with the key name if ten attempts do not converge.
pub fn traffic_put(client: &NativeClient, name: &str, value: &[u8]) {
    for attempt in 0..10 {
        match client.set(&key(name), bytes::Bytes::copy_from_slice(value)) {
            Ok(()) => return,
            Err(_) if attempt < 9 => std::thread::sleep(Duration::from_millis(200)),
            Err(error) => panic!("traffic set {name} never converges: {error:?}"),
        }
    }
}

/// The committed value of `name`, retrying through leader changes.
///
/// # Panics
///
/// Panics on read error or on absence. An absent key is a lost write, not a
/// retryable condition.
#[must_use]
pub fn traffic_get(client: &NativeClient, name: &str) -> Vec<u8> {
    for attempt in 0..10 {
        match client.get(&key(name)) {
            Ok(Some(value)) => return value.to_vec(),
            Ok(None) => panic!("traffic key {name} lost"),
            Err(_) if attempt < 9 => std::thread::sleep(Duration::from_millis(200)),
            Err(error) => panic!("traffic get {name} never converges: {error:?}"),
        }
    }
    unreachable!()
}

/// A committed `put_stream`.
///
/// # Panics
///
/// Panics with the key name if the upload does not commit.
pub fn put_stream(client: &NativeClient, name: &str, value: &[u8]) {
    let mut source: &[u8] = value;
    client
        .put_stream(&key(name), Some(value.len() as u64), &mut source)
        .unwrap_or_else(|error| panic!("put_stream {name} commits: {error:?}"));
}

/// The streamed value of `name`.
///
/// # Panics
///
/// Panics on read error or on absence.
#[must_use]
pub fn get_stream(client: &NativeClient, name: &str) -> Vec<u8> {
    client
        .get_stream(&key(name))
        .unwrap_or_else(|error| panic!("get_stream {name} reads: {error:?}"))
        .unwrap_or_else(|| panic!("{name} present"))
        .to_vec()
}

/// The suites' reference model: acknowledged byte keys to their values.
pub type Model = std::collections::HashMap<Vec<u8>, Vec<u8>>;

/// Commits `writes` as one atomic batch and mirrors them into `model`.
///
/// # Panics
///
/// Panics with `context` if the batch does not commit. A rejected batch
/// leaves the model untouched, so the two never drift.
pub fn batch_checked(
    client: &NativeClient,
    model: &mut Model,
    writes: Vec<(Vec<u8>, Vec<u8>)>,
    context: &str,
) {
    let specs: Vec<kivi_client::ordered::BatchWriteSpec> = writes
        .iter()
        .map(|(key, value)| kivi_client::ordered::BatchWriteSpec {
            key: key.clone(),
            kind: kivi_client::ordered::BatchWriteKind::Put(value.clone()),
            expect: kivi_client::ordered::BatchExpect::Any,
        })
        .collect();
    client
        .atomic_batch(NS, &specs)
        .unwrap_or_else(|error| panic!("{context}: batch failed: {error:?}"));
    for (key, value) in writes {
        model.insert(key, value);
    }
}

/// A committed `set`, mirrored into `model`.
///
/// # Panics
///
/// Panics with `context` if the write does not commit.
pub fn put_checked(
    client: &NativeClient,
    model: &mut Model,
    key: Vec<u8>,
    value: Vec<u8>,
    context: &str,
) {
    client
        .set(
            &Key::from(key.clone()),
            bytes::Bytes::copy_from_slice(&value),
        )
        .unwrap_or_else(|error| panic!("{context}: set {} commits: {error:?}", hex_key(&key)));
    model.insert(key, value);
}

/// Total sidecar bulk bytes each of the first three members reports.
///
/// # Panics
///
/// Panics if a member's sidecar admin endpoint does not answer 200.
#[must_use]
pub fn bulk_received(cluster: &Cluster) -> [u64; 3] {
    [0, 1, 2].map(|index| {
        let (status, body) = cluster.admin_get(index, "/v1/sidecar");
        assert_eq!(status, 200, "sidecar admin: {body}");
        body["bulk_bytes_received"].as_u64().unwrap_or(0)
    })
}

/// One protocol frame off a raw socket.
///
/// # Errors
///
/// Returns `UnexpectedEof` if the peer closes before a full frame, or
/// `InvalidData` if the bytes do not decode.
pub fn read_frame(
    socket: &mut std::net::TcpStream,
    reader: &mut kivi_protocol::FrameReader,
) -> std::io::Result<kivi_protocol::Frame> {
    use std::io::ErrorKind;
    use std::io::Read as _;
    let mut chunk = vec![0u8; 4096];
    loop {
        // Two ways a read can stop making progress, and they mean different
        // things. `EINTR` is not an event at all: the syscall was interrupted
        // before it moved any bytes, so the stream is exactly where it was and
        // reading again is the entire recovery. `WouldBlock` is what a socket
        // with `SO_RCVTIMEO` reports when its deadline expires - Rust does not
        // translate it to `TimedOut` for a bare `read`, only for the
        // `read_exact`/`read_until` family - so it is translated here rather than
        // surfacing as an errno that looks like a protocol fault.
        let count = loop {
            match socket.read(&mut chunk) {
                Ok(count) => break count,
                // Redundant to a linter, essential here: without it the arm
                // would fall through to `return`, turning a signal that moved no
                // bytes into a protocol failure.
                #[allow(clippy::needless_continue)]
                Err(error) if error.kind() == ErrorKind::Interrupted => continue,
                Err(error) if error.kind() == ErrorKind::WouldBlock => {
                    return Err(std::io::Error::new(
                        ErrorKind::TimedOut,
                        "no frame within the socket's read timeout",
                    ));
                }
                Err(error) => return Err(error),
            }
        };
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

/// A tablet selector the suites already hold. Suites number tablets as
/// `usize` loop counters, `u8` layout ids, or `u64` ids from the directory,
/// and a cast at every one of the ~40 call sites would be noise.
pub trait TabletSlot {
    /// The tablet's ordinal.
    fn slot(self) -> u64;
}

impl TabletSlot for usize {
    #[allow(clippy::cast_precision_loss)]
    fn slot(self) -> u64 {
        self as u64
    }
}

impl TabletSlot for u8 {
    fn slot(self) -> u64 {
        u64::from(self)
    }
}

impl TabletSlot for u64 {
    fn slot(self) -> u64 {
        self
    }
}

/// Untyped integer literals in a `spread_key(..)` argument. Every call site
/// writes a non-negative tablet id, so the fallback is the literal's own
/// value reinterpreted, not a checked conversion.
impl TabletSlot for i32 {
    #[allow(clippy::cast_sign_loss)]
    fn slot(self) -> u64 {
        self as u64
    }
}

/// Deterministic key laid out so four hash-range tablets each get their own
/// leading byte, and index 0 of every tablet sorts adjacent. Ordered suites
/// depend on the ordering, so the layout is fixed rather than seeded.
#[must_use]
pub fn spread_key(tablet: impl TabletSlot, index: usize) -> Vec<u8> {
    let tablet = tablet.slot();

    let prefix = match tablet % 4 {
        0 => 0x10u8,
        1 => b'A',
        2 => 0x90u8,
        _ => 0xF0u8,
    };
    let mut key = vec![prefix];
    key.extend_from_slice(format!("{index:06}").as_bytes());
    key
}

/// Which of the four hash-range tablets owns key `index`.
#[must_use]
pub fn tablet_of(index: usize) -> u64 {
    (index % 4) as u64
}

/// A medium (2–4 KiB) value for key `index`.
///
/// The bytes are pseudo-random over a 251-symbol alphabet, so they are
/// incompressible for any practical purpose. A suite that needs the full
/// 256-symbol alphabet states that itself rather than widening this one.
#[must_use]
pub fn medium_value(index: usize) -> Vec<u8> {
    fill(2048 + (index % 3) * 1024, (index % 251) as u64)
}

/// One admin row per live member for `tablet`, or `m{index}:no-group` where
/// the member does not report it. Included verbatim in failure messages, so
/// a wedged suite says which member holds which term.
#[must_use]
pub fn tablet_state_dump(cluster: &Cluster, tablet: u64) -> String {
    (0..cluster.member_count())
        .map(|index| {
            if !cluster.alive(index) {
                return format!("m{index}:down");
            }
            let row = cluster.tablets_status_opt(index).and_then(|status| {
                status
                    .as_array()
                    .cloned()
                    .unwrap_or_default()
                    .into_iter()
                    .find(|entry| {
                        entry.get("group").and_then(serde_json::Value::as_u64) == Some(tablet)
                    })
            });
            match row {
                Some(entry) => format!(
                    "m{index}:role={} leader={} term={} last_log={} committed={} applied={}",
                    entry["role"].as_str().unwrap_or("?"),
                    entry["leader"]
                        .as_u64()
                        .map_or("none".to_owned(), |leader| leader.to_string()),
                    entry["term"].as_u64().unwrap_or(0),
                    entry["last_log"].as_u64().unwrap_or(0),
                    entry["committed"].as_u64().unwrap_or(0),
                    entry["applied"].as_u64().unwrap_or(0),
                ),
                None => format!("m{index}:no-group"),
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// Splits `parent` and waits for both children to be live, returning
/// `(left, right)`.
///
/// # Panics
///
/// Panics if the split is refused or the children never come up.
#[must_use]
pub fn split_and_wait(cluster: &Cluster, parent: u64) -> (u64, u64) {
    let reply = cluster.split_tablet(parent);
    assert_eq!(
        reply.get("ok").and_then(serde_json::Value::as_bool),
        Some(true),
        "split accepted: {reply}"
    );
    let children = (
        reply
            .get("left")
            .and_then(serde_json::Value::as_u64)
            .expect("left child"),
        reply
            .get("right")
            .and_then(serde_json::Value::as_u64)
            .expect("right child"),
    );
    cluster.wait_splits_done(Duration::from_secs(600));
    children
}

/// Merges `left` and `right` and waits for the merged range to be live.
///
/// # Panics
///
/// Panics if the merge is refused or the merged range never comes up.
pub fn merge_and_wait(cluster: &Cluster, left: u64, right: u64) {
    let reply = cluster.merge_tablets(left, right);
    assert_eq!(
        reply.get("ok").and_then(serde_json::Value::as_bool),
        Some(true),
        "merge accepted: {reply}"
    );
    cluster.wait_merges_done(Duration::from_secs(600));
}

/// Client for the `/v1/redundancy/*` admin plane, over generic immutable
/// bytes (BLAKE3 asset identity).
pub mod redundancy {
    use std::time::{Duration, Instant};

    use serde_json::{Value, json};

    use super::Cluster;

    /// Asset family discriminant for generic immutable bytes.
    pub const KIND_GENERIC: u8 = 5;

    /// Generous deadline: spawning real processes plus elections, fragment
    /// transfer, and Raft commits dominates wall time.
    pub const DEADLINE: Duration = Duration::from_secs(180);

    /// Poll cadence for the admin-plane waits.
    pub const POLL: Duration = Duration::from_millis(250);

    /// Incompressible-ish deterministic bytes (splitmix64 output): enough
    /// entropy that a coded layout cannot cheat on compressibility.
    #[must_use]
    pub fn payload(len: usize, seed: u64) -> Vec<u8> {
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

    /// Standard padded base64 (the admin plane's image encoding).
    fn b64(bytes: &[u8]) -> String {
        use base64::Engine as _;
        base64::engine::general_purpose::STANDARD.encode(bytes)
    }

    /// Decodes standard padded base64, or `None` if the text is not base64.
    #[must_use]
    pub fn unb64(text: &str) -> Option<Vec<u8>> {
        use base64::Engine as _;
        base64::engine::general_purpose::STANDARD.decode(text).ok()
    }

    /// Lowercase hex asset identity: the full BLAKE3 digest, exactly like
    /// the coordinator. The admin plane rejects a `hash_hex` that is not 64
    /// characters, so this cannot share [`hex_key`]'s truncating form.
    #[must_use]
    pub fn asset_hex(bytes: &[u8]) -> String {
        use core::fmt::Write as _;
        kivi_codec::integrity::blake3_256(bytes).iter().fold(
            String::with_capacity(64),
            |mut out, byte| {
                let _ = write!(out, "{byte:02x}");
                out
            },
        )
    }

    /// Fragment store root of one cluster member.
    #[must_use]
    pub fn fragments_root(cluster: &Cluster, index: usize) -> std::path::PathBuf {
        cluster.data_dir(index).join("redundancy").join("fragments")
    }

    /// Protects bytes through the admin plane. The status is returned rather
    /// than asserted: several callers deliberately drive a refusal.
    #[must_use]
    pub fn protect(cluster: &Cluster, index: usize, bytes: &[u8], tolerance: u8) -> (u16, Value) {
        cluster.admin_post(
            index,
            "/v1/redundancy/protect",
            &json!({
                "kind": KIND_GENERIC,
                "domain": 0,
                "bytes_b64": b64(bytes),
                "tolerance": tolerance,
            }),
        )
    }

    /// Reads an asset back through the admin plane.
    #[must_use]
    pub fn read(cluster: &Cluster, index: usize, hash: &str) -> (u16, Value) {
        cluster.admin_get(
            index,
            &format!("/v1/redundancy/read?kind={KIND_GENERIC}&domain=0&hash_hex={hash}"),
        )
    }

    /// Reads one asset's health.
    #[must_use]
    pub fn health(cluster: &Cluster, index: usize, hash: &str) -> (u16, Value) {
        cluster.admin_get(
            index,
            &format!("/v1/redundancy/health?kind={KIND_GENERIC}&domain=0&hash_hex={hash}"),
        )
    }

    /// Repairs one asset.
    #[must_use]
    pub fn repair(cluster: &Cluster, index: usize, hash: &str) -> (u16, Value) {
        cluster.admin_post(
            index,
            "/v1/redundancy/repair",
            &json!({ "kind": KIND_GENERIC, "domain": 0, "hash_hex": hash }),
        )
    }

    /// Transitions one asset to an explicit scheme.
    #[must_use]
    pub fn transition(cluster: &Cluster, index: usize, hash: &str, scheme: &Value) -> (u16, Value) {
        cluster.admin_post(
            index,
            "/v1/redundancy/transition",
            &json!({ "kind": KIND_GENERIC, "domain": 0, "hash_hex": hash, "scheme": scheme }),
        )
    }

    /// Polls until `check` returns a value.
    ///
    /// # Panics
    ///
    /// Panics with `label` if the deadline passes first.
    pub fn await_until<T>(label: &str, mut check: impl FnMut() -> Option<T>) -> T {
        let deadline = Instant::now() + DEADLINE;
        loop {
            if let Some(value) = check() {
                return value;
            }
            assert!(
                Instant::now() < deadline,
                "timed out after {DEADLINE:?} waiting for {label}"
            );
            std::thread::sleep(POLL);
        }
    }

    /// Waits until an asset reports `wanted` health on a node.
    ///
    /// # Panics
    ///
    /// Panics if the health never reaches `wanted`.
    #[must_use]
    pub fn await_health(cluster: &Cluster, index: usize, hash: &str, wanted: &str) -> Value {
        await_until(&format!("health {wanted} on node {index}"), || {
            let (status, body) = health(cluster, index, hash);
            (status == 200 && body["health"].as_str() == Some(wanted)).then_some(body)
        })
    }
}
