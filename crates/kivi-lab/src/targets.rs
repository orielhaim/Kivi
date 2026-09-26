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

use std::thread;
use std::time::Duration;

use bytes::Bytes;
use kivi_client::{ClientConfig, NativeClient};
use kivi_state::Key;
use kivi_types::NamespaceId;

use crate::resp_client::{Reply, RespClient};
use crate::workload::{PAYLOAD_SEED, RANGE_WINDOW, WorkloadOp, fill_pattern};

/// One workload outcome. Reads of absent keys are `Ok` (seeding makes
/// absence rare; deletes make it legal) — only transport/type errors fail.
pub type OpOutcome = Result<(), String>;

const RESP_RETRY_LIMIT: u32 = 12;

/// Optional target counters that do not fit the shared byte schema.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize)]
pub struct TargetStats {
    /// Logical requests issued by the target client.
    pub requests: u64,
    /// Route redirects followed by the target client.
    pub redirects: u64,
    /// Terminal client errors observed by the target client.
    pub errors: u64,
}

/// Minimal interface the [`runner`](crate::runner) needs. Adapters own
/// every protocol-specific conversion; batching shape (one round-trip per
/// batch when the transport pipelines) lives here, while threading,
/// warmup, timing, and histograms live in the runner — identical for all
/// targets.
pub trait BenchTarget: Send {
    /// Executes a batch of ops in order, returning per-op outcomes.
    /// A batch of one is a single round-trip everywhere.
    fn execute_batch(&mut self, ops: &[WorkloadOp]) -> Vec<OpOutcome>;

    /// Executes a batch with a caller-owned immutable payload cache.
    fn execute_batch_with_payload(&mut self, ops: &[WorkloadOp], payload: &[u8]) -> Vec<OpOutcome> {
        let _ = payload;
        self.execute_batch(ops)
    }

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

    /// Seeds one operation with the run's shared payload cache.
    ///
    /// # Errors
    ///
    /// Returns the target's string error when the seed fails.
    fn seed_one_with_payload(&mut self, op: &WorkloadOp, payload: &[u8]) -> OpOutcome {
        let _ = payload;
        self.seed_one(op)
    }

    /// Transfers transport byte counters since the last call, when the
    /// target accounts them (`None` keeps target-specific metrics out of
    /// the shared abstraction).
    fn take_bytes(&mut self) -> Option<(u64, u64)> {
        None
    }

    /// Takes target-specific counters after the measured phase.
    fn take_stats(&mut self) -> Option<TargetStats> {
        None
    }
}

/// Far-future expiry stamp for benchmark expirations: one shared
/// production read plus sixty seconds of Jiff arithmetic.
fn bench_expiry() -> kivi_types::WallTimestamp {
    use kivi_core::{SystemClock, wall_now_or_max};
    wall_now_or_max(&SystemClock)
        .checked_add(jiff::SignedDuration::from_secs(60))
        .unwrap_or(kivi_types::WallTimestamp::MAX)
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
        Self::connect_multi(std::slice::from_ref(&server.to_owned()), namespace, pending)
    }

    /// Connects with several seed endpoints (task AG: the replicated
    /// cluster target). The workload definitions stay untouched: this
    /// target handles leader discovery (`StaleRoute` redirects),
    /// bounded retries, and stable mutation identities internally, so
    /// shared benchmark code remains target-neutral. One instance per
    /// benchmark thread (one retry session each).
    ///
    /// # Errors
    ///
    /// Returns the client setup error when no seed is reachable.
    pub fn connect_multi(seeds: &[String], namespace: u64, pending: usize) -> anyhow::Result<Self> {
        use anyhow::Context;
        let client = NativeClient::new(ClientConfig {
            seeds: seeds.to_owned(),
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
        let payload = fill_pattern(max_payload_len(ops), PAYLOAD_SEED);
        self.execute_batch_with_payload(ops, &payload)
    }

    fn execute_batch_with_payload(&mut self, ops: &[WorkloadOp], payload: &[u8]) -> Vec<OpOutcome> {
        ops.iter()
            .map(|op| execute_native(&self.client, op, payload))
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

    fn seed_one_with_payload(&mut self, op: &WorkloadOp, payload: &[u8]) -> OpOutcome {
        if let WorkloadOp::CounterAdd(key) = op {
            return self
                .client
                .counter_add(&Key::from(key.as_str()), 0)
                .map(|_| ())
                .map_err(|error| error.to_string());
        }
        self.execute_batch_with_payload(std::slice::from_ref(op), payload)
            .into_iter()
            .next()
            .unwrap_or(Err("empty batch".to_owned()))
    }

    fn take_stats(&mut self) -> Option<TargetStats> {
        let stats = self.client.stats();
        Some(TargetStats {
            requests: stats.requests,
            redirects: stats.redirects,
            errors: stats.errors,
        })
    }
}

/// Inline threshold mirroring the server's representation split: values
/// above this stage through chunk lanes either way, so the native target
/// streams them (one giant frame would trip the connection input bound
/// instead of measuring the store).
const NATIVE_STREAM_ABOVE: usize = 256 * 1024;

fn execute_native(client: &NativeClient, op: &WorkloadOp, payload: &[u8]) -> OpOutcome {
    match op {
        WorkloadOp::Get(key) => client
            .get(&Key::from(key.as_str()))
            .map(|_| ())
            .map_err(|error| error.to_string()),
        WorkloadOp::Set { key, len } if *len > NATIVE_STREAM_ABOVE => {
            let bytes = if payload.len() == *len {
                payload.to_vec()
            } else {
                fill_pattern(*len, PAYLOAD_SEED)
            };
            let mut source = std::io::Cursor::new(bytes);
            client
                .put_stream(&Key::from(key.as_str()), Some(*len as u64), &mut source)
                .map_err(|error| error.to_string())
        }
        WorkloadOp::Set { key, len } => {
            let bytes = if payload.len() == *len {
                Bytes::copy_from_slice(payload)
            } else {
                Bytes::from(fill_pattern(*len, PAYLOAD_SEED))
            };
            client
                .set(&Key::from(key.as_str()), bytes)
                .map_err(|error| error.to_string())
        }
        WorkloadOp::CounterAdd(key) => client
            .counter_add(&Key::from(key.as_str()), 1)
            .map(|_| ())
            .map_err(|error| error.to_string()),
        WorkloadOp::Delete(key) => client
            .delete(&Key::from(key.as_str()))
            .map(|_| ())
            .map_err(|error| error.to_string()),
        WorkloadOp::Exists(key) => client
            .exists(&Key::from(key.as_str()))
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

fn max_payload_len(ops: &[WorkloadOp]) -> usize {
    ops.iter()
        .filter_map(|op| match op {
            WorkloadOp::Set { len, .. } => Some(*len),
            _ => None,
        })
        .max()
        .unwrap_or_default()
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
        let payload = fill_pattern(max_payload_len(ops), PAYLOAD_SEED);
        self.execute_batch_with_payload(ops, &payload)
    }

    fn execute_batch_with_payload(&mut self, ops: &[WorkloadOp], payload: &[u8]) -> Vec<OpOutcome> {
        let connection = match self.connection() {
            Ok(connection) => connection,
            Err(error) => return ops.iter().map(|_| Err(error.clone())).collect(),
        };
        if ops.len() == 1 {
            return match execute_one(connection, &ops[0], payload) {
                Ok(result) => vec![result],
                Err(error) => {
                    self.client = None;
                    vec![Err(error)]
                }
            };
        }
        let cmds: Vec<Vec<Vec<u8>>> = ops.iter().map(|op| encode_op(op, payload)).collect();
        match connection.pipeline(&cmds) {
            Ok(replies) => {
                let mut outcomes: Vec<OpOutcome> = replies
                    .iter()
                    .zip(ops.iter())
                    .map(|(reply, op)| check_workload_reply(reply, op))
                    .collect();
                for (index, outcome) in outcomes.iter_mut().enumerate() {
                    if outcome.is_err() && retryable_reply(&replies[index]) {
                        match execute_one(connection, &ops[index], payload) {
                            Ok(retried) => *outcome = retried,
                            Err(error) => {
                                *outcome = Err(error);
                                self.client = None;
                                break;
                            }
                        }
                    }
                }
                outcomes
            }
            Err(error) => {
                self.client = None;
                ops.iter().map(|_| Err(error.to_string())).collect()
            }
        }
    }

    fn seed_one_with_payload(&mut self, op: &WorkloadOp, payload: &[u8]) -> OpOutcome {
        self.execute_batch_with_payload(std::slice::from_ref(op), payload)
            .into_iter()
            .next()
            .unwrap_or(Err("empty batch".to_owned()))
    }
}

/// Encodes one workload op as a RESP command (argv as owned byte strings).
#[must_use]
pub fn encode_workload_command(op: &WorkloadOp, payload: &[u8]) -> Vec<Vec<u8>> {
    encode_op(op, payload)
}

fn encode_op(op: &WorkloadOp, payload: &[u8]) -> Vec<Vec<u8>> {
    let owned = |parts: &[&[u8]]| parts.iter().map(|part| part.to_vec()).collect::<Vec<_>>();
    match op {
        WorkloadOp::Get(key) => owned(&[b"GET", key.as_bytes()]),
        WorkloadOp::Set { key, len } => {
            let value = if payload.len() == *len {
                payload.to_vec()
            } else {
                fill_pattern(*len, PAYLOAD_SEED)
            };
            vec![b"SET".to_vec(), key.as_bytes().to_vec(), value]
        }
        WorkloadOp::CounterAdd(key) => owned(&[b"INCRBY", key.as_bytes(), b"1"]),
        WorkloadOp::Delete(key) => owned(&[b"DEL", key.as_bytes()]),
        WorkloadOp::Exists(key) => owned(&[b"EXISTS", key.as_bytes()]),
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
fn execute_one(
    connection: &mut RespClient,
    op: &WorkloadOp,
    payload: &[u8],
) -> Result<OpOutcome, String> {
    let mut attempt = 0u32;
    loop {
        let cmd = encode_op(op, payload);
        let argv: Vec<&[u8]> = cmd.iter().map(Vec::as_slice).collect();
        let reply = connection
            .round_trip(&argv)
            .map_err(|error| error.to_string())?;
        let outcome = check_workload_reply(&reply, op);
        if outcome.is_ok() || !retryable_reply(&reply) || attempt >= RESP_RETRY_LIMIT {
            return Ok(outcome);
        }
        thread::sleep(Duration::from_millis(1_u64 << attempt.min(6)));
        attempt += 1;
    }
}

fn retryable_reply(reply: &Reply) -> bool {
    matches!(reply, Reply::Error(message) if message.starts_with(b"BUSY ") || message.starts_with(b"TRYAGAIN "))
}

/// Checks the exact RESP reply shape required by one workload operation.
///
/// # Errors
///
/// Returns a string describing an error frame or an unexpected reply shape.
pub fn check_workload_reply(reply: &Reply, op: &WorkloadOp) -> OpOutcome {
    if let Reply::Error(message) = reply {
        return Err(format!("redis error: {}", String::from_utf8_lossy(message)));
    }
    let shape_ok = match op {
        WorkloadOp::Get(_) => matches!(reply, Reply::Bulk(None | Some(_))),
        WorkloadOp::Set { .. } => {
            matches!(reply, Reply::Simple(value) if value.eq_ignore_ascii_case(b"OK"))
        }
        WorkloadOp::CounterAdd(_)
        | WorkloadOp::Delete(_)
        | WorkloadOp::Exists(_)
        | WorkloadOp::Expire(_)
        | WorkloadOp::Ttl(_)
        | WorkloadOp::SetRange(_) => matches!(reply, Reply::Integer(_)),
        WorkloadOp::GetRange(_) => matches!(reply, Reply::Bulk(Some(_))),
    };
    if shape_ok {
        Ok(())
    } else {
        Err(format!("unexpected reply shape for {op:?}: {reply:?}"))
    }
}
