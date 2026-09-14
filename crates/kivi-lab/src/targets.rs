//! Benchmark targets: one trait, one implementation per frontend.
//!
//! [`BenchTarget`] is the only thing the [`runner`](crate::runner) module
//! sees. The two implementations execute the *same* [`WorkloadOp`]
//! stream:
//!
//! * [`KiviNativeTarget`] — typed [`NativeClient`]
//!   calls against the native protocol.
//! * [`RespTarget`] — standard RESP commands through the shared raw client
//!   ([`crate::resp_client`]). Deliberately generic: the exact same code
//!   points at Kivi RESP, Redis, Valkey, or Dragonfly, so comparisons stay
//!   apples-to-apples. There is intentionally no separate `RedisTarget`.

use bytes::Bytes;
use kivi_client::{ClientConfig, NativeClient};
use kivi_state::Key;
use kivi_types::NamespaceId;

use crate::resp_client::{Reply, RespClient};
use crate::workload::{RANGE_WINDOW, WorkloadOp, fill_pattern};

/// One workload outcome. Reads of absent keys are `Ok` (seeding makes
/// absence rare; deletes make it legal) — only transport/type errors fail.
pub type OpOutcome = Result<(), String>;

/// Minimal interface the [`runner`](crate::runner) needs. Adapters own
/// every protocol-specific conversion; batching shape (one round-trip per
/// batch when the transport pipelines) lives here, while threading,
/// warmup, timing, and histograms live in the runner — identical for all
/// targets.
pub trait BenchTarget: Send {
    /// Executes a batch of ops in order, returning per-op outcomes.
    /// A batch of one is a single round-trip everywhere.
    fn execute_batch(&mut self, ops: &[WorkloadOp]) -> Vec<OpOutcome>;

    /// Seeds one workload op (byte writes and counter creates). Defaults
    /// to `execute_batch` of one; native overrides counter creation to
    /// avoid counting the seed.
    ///
    /// # Errors
    ///
    /// Returns the seeding failure as a string.
    fn seed_one(&mut self, op: &WorkloadOp) -> OpOutcome {
        self.execute_batch(std::slice::from_ref(op))
            .into_iter()
            .next()
            .unwrap_or(Err("empty batch".to_owned()))
    }

    /// Transfers transport byte counters since the last call, when the
    /// target accounts them (`None` keeps target-specific metrics out of
    /// the shared abstraction).
    fn take_bytes(&mut self) -> Option<(u64, u64)> {
        None
    }
}

/// Far-future expiry stamp for benchmark expirations.
fn bench_expiry() -> kivi_types::UnixMicros {
    use std::time::{SystemTime, UNIX_EPOCH};
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |elapsed| {
            elapsed.as_micros().try_into().unwrap_or(u64::MAX)
        });
    kivi_types::UnixMicros::from_micros(now.saturating_add(60_000_000))
}

/// Kivi native adapter: [`WorkloadOp`] to [`Key`] plus typed
/// [`NativeClient`].
///
/// One instance per benchmark thread (one retry session each, so a
/// straggler never couples acknowledgement floors).
#[derive(Debug, Clone)]
pub struct KiviNativeTarget {
    client: NativeClient,
}

impl KiviNativeTarget {
    /// Connects to the native seed endpoint.
    ///
    /// # Errors
    ///
    /// Returns the client setup error when the seed is unreachable.
    pub fn connect(server: &str, namespace: u64, pending: usize) -> anyhow::Result<Self> {
        use anyhow::Context;
        let client = NativeClient::new(ClientConfig {
            seeds: vec![server.to_owned()],
            namespace: NamespaceId::from_u64(namespace),
            max_pending: pending,
            ..ClientConfig::default()
        })
        .context("native client setup failed")?;
        Ok(Self { client })
    }

    /// Native redirect counter (shared-runner extra metric).
    #[must_use]
    pub fn redirects(&self) -> u64 {
        self.client.stats().redirects
    }
}

impl BenchTarget for KiviNativeTarget {
    fn execute_batch(&mut self, ops: &[WorkloadOp]) -> Vec<OpOutcome> {
        // The native client sends one request per call; batching here is a
        // plain loop so the op stream stays identical across targets.
        ops.iter()
            .map(|op| execute_native(&self.client, op))
            .collect()
    }

    fn seed_one(&mut self, op: &WorkloadOp) -> OpOutcome {
        // Counters seed at 0 (create without counting); the workload adds 1.
        if let WorkloadOp::CounterAdd(key) = op {
            return self
                .client
                .counter_add(&Key::from(key.as_str()), 0)
                .map(|_| ())
                .map_err(|error| error.to_string());
        }
        self.execute_batch(std::slice::from_ref(op))
            .into_iter()
            .next()
            .unwrap_or(Err("empty batch".to_owned()))
    }
}

/// Inline threshold mirroring the server's representation split: values
/// above this stage through chunk lanes either way, so the native target
/// streams them (one giant frame would trip the connection input bound
/// instead of measuring the store).
const NATIVE_STREAM_ABOVE: usize = 256 * 1024;

fn execute_native(client: &NativeClient, op: &WorkloadOp) -> OpOutcome {
    match op {
        WorkloadOp::Get(key) => client
            .get(&Key::from(key.as_str()))
            .map(|_| ())
            .map_err(|error| error.to_string()),
        WorkloadOp::Set { key, len } if *len > NATIVE_STREAM_ABOVE => {
            let bytes = fill_pattern(*len, 0xBE4C4);
            let mut source = std::io::Cursor::new(bytes);
            client
                .put_stream(&Key::from(key.as_str()), Some(*len as u64), &mut source)
                .map_err(|error| error.to_string())
        }
        WorkloadOp::Set { key, len } => client
            .set(
                &Key::from(key.as_str()),
                Bytes::from(fill_pattern(*len, 0xBE4C4)),
            )
            .map_err(|error| error.to_string()),
        WorkloadOp::CounterAdd(key) => client
            .counter_add(&Key::from(key.as_str()), 1)
            .map(|_| ())
            .map_err(|error| error.to_string()),
        WorkloadOp::Delete(key) => client
            .delete(&Key::from(key.as_str()))
            .map(|_| ())
            .map_err(|error| error.to_string()),
        WorkloadOp::Expire(key) => client
            .expire_at(&Key::from(key.as_str()), bench_expiry())
            .map(|_| ())
            .map_err(|error| error.to_string()),
        WorkloadOp::Ttl(key) => client
            .get_expiry(&Key::from(key.as_str()))
            .map(|_| ())
            .map_err(|error| error.to_string()),
        WorkloadOp::SetRange(key) => client
            .set_range(&Key::from(key.as_str()), 0, Bytes::from_static(b"v"))
            .map_err(|error| error.to_string()),
        WorkloadOp::GetRange(key) => client
            .get_range(&Key::from(key.as_str()), 0, RANGE_WINDOW)
            .map(|_| ())
            .map_err(|error| error.to_string()),
    }
}

/// RESP-compatible adapter: [`WorkloadOp`] →
/// standard RESP commands over the shared raw client. One instance (and
/// therefore one connection) per benchmark thread. Points at Kivi RESP,
/// Redis, Valkey, or Dragonfly without code changes.
#[derive(Debug)]
pub struct RespTarget {
    addr: String,
    client: Option<RespClient>,
}

impl RespTarget {
    /// Builds a lazy adapter: connects on first use.
    #[must_use]
    pub fn lazy(addr: &str) -> Self {
        Self {
            addr: addr.to_owned(),
            client: None,
        }
    }

    fn connection(&mut self) -> Result<&mut RespClient, String> {
        if self.client.is_none() {
            self.client = Some(RespClient::connect(&self.addr).map_err(|error| error.to_string())?);
        }
        Ok(self.client.as_mut().expect("just connected"))
    }

    /// Transfers byte counters since the last call (benchmark I/O metrics).
    pub fn take_bytes(&mut self) -> (u64, u64) {
        self.client.as_mut().map_or((0, 0), RespClient::take_bytes)
    }
}

impl BenchTarget for RespTarget {
    fn take_bytes(&mut self) -> Option<(u64, u64)> {
        Some(self.client.as_mut().map_or((0, 0), RespClient::take_bytes))
    }

    fn execute_batch(&mut self, ops: &[WorkloadOp]) -> Vec<OpOutcome> {
        let connection = match self.connection() {
            Ok(connection) => connection,
            Err(error) => return ops.iter().map(|_| Err(error.clone())).collect(),
        };
        if ops.len() == 1 {
            return vec![execute_one(connection, &ops[0])];
        }
        // One pipeline per batch: N commands out, N replies back, in order.
        let cmds: Vec<Vec<Vec<u8>>> = ops.iter().map(encode_op).collect();
        match connection.pipeline(&cmds) {
            Ok(replies) => replies
                .iter()
                .zip(ops.iter())
                .map(|(reply, op)| check_reply(reply, op))
                .collect(),
            Err(error) => ops.iter().map(|_| Err(error.to_string())).collect(),
        }
    }
}

/// Encodes one workload op as a RESP command (argv as owned byte strings).
fn encode_op(op: &WorkloadOp) -> Vec<Vec<u8>> {
    let owned = |parts: &[&[u8]]| parts.iter().map(|part| part.to_vec()).collect::<Vec<_>>();
    match op {
        WorkloadOp::Get(key) => owned(&[b"GET", key.as_bytes()]),
        WorkloadOp::Set { key, len } => {
            let value = fill_pattern(*len, 0xBE4C4);
            vec![b"SET".to_vec(), key.as_bytes().to_vec(), value]
        }
        // Counters are native-only; on RESP this errors loudly per op
        // (callers use `compat`, never `mixed`, for cross-target runs).
        WorkloadOp::CounterAdd(key) => owned(&[b"INCRBY", key.as_bytes(), b"1"]),
        WorkloadOp::Delete(key) => owned(&[b"DEL", key.as_bytes()]),
        WorkloadOp::Expire(key) => owned(&[b"EXPIRE", key.as_bytes(), b"60"]),
        WorkloadOp::Ttl(key) => owned(&[b"TTL", key.as_bytes()]),
        WorkloadOp::SetRange(key) => owned(&[b"SETRANGE", key.as_bytes(), b"0", b"v"]),
        WorkloadOp::GetRange(key) => {
            let end = RANGE_WINDOW.saturating_sub(1).to_string();
            owned(&[b"GETRANGE", key.as_bytes(), b"0", end.as_bytes()])
        }
    }
}

/// Executes one op over a single RESP connection.
fn execute_one(connection: &mut RespClient, op: &WorkloadOp) -> OpOutcome {
    let cmd = encode_op(op);
    let argv: Vec<&[u8]> = cmd.iter().map(Vec::as_slice).collect();
    match connection.round_trip(&argv) {
        Ok(reply) => check_reply(&reply, op),
        Err(error) => Err(error.to_string()),
    }
}

/// Maps one reply onto its op outcome. Absent reads are `Ok` (nil bulk for
/// `GET`, empty string for `GETRANGE`); anything error-shaped fails.
fn check_reply(reply: &Reply, op: &WorkloadOp) -> OpOutcome {
    match (reply, op) {
        (Reply::Error(message), _) => {
            Err(format!("redis error: {}", String::from_utf8_lossy(message)))
        }
        // Missing keys read as nil bulk: legal for `GET`, an anomaly for
        // every other profiled op (GETRANGE misses answer empty, never nil).
        (Reply::Bulk(None), WorkloadOp::Get(_)) => Ok(()),
        (Reply::Bulk(None), _) => Err("unexpected nil".to_owned()),
        (Reply::Bulk(_) | Reply::Simple(_) | Reply::Integer(_) | Reply::Array(_), _) => Ok(()),
    }
}
