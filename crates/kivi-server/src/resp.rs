//! Optional Redis/RESP compatibility frontend (feature `redis-compat`).
//!
//! One Redis-style `host:port` serving RESP2/RESP3 over the same typed
//! [`Operation`] the native frontend executes — reached here through the
//! embedded [`LocalClient`] ingress path (local routing to the owning
//! tablet), never through a
//! native TCP hop or a native frame encode/decode round-trip inside this
//! process. No Redis Cluster semantics, no worker-port exposure: a normal
//! `host:port`.
//!
//! Even when compiled in, the listener only starts with `--redis-listen`
//! (runtime optional). Metrics are frontend-local atomics (relaxed
//! ordering, no per-request lock); per-connection details stay in the
//! connection task and flush into these totals without contention.

use std::net::SocketAddr;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use kivi_engine::LocalClient;
use kivi_resp::{ConnConfig, ConnMetrics, ExecuteError, Executor, RespConnection, RespVersion};
use kivi_state::{Operation, OperationResult};

/// Shared RESP frontend statistics (frontend-local, lock-free).
#[derive(Debug, Default)]
pub struct RespStats {
    /// Connections currently open.
    pub connections_current: AtomicU64,
    /// Connections opened since startup.
    pub connections_total: AtomicU64,
    /// Requests answered (including immediate replies).
    pub requests: AtomicU64,
    /// Incoming bytes consumed.
    pub bytes_in: AtomicU64,
    /// Outgoing bytes produced.
    pub bytes_out: AtomicU64,
    /// Parse/bound failures answered.
    pub protocol_errors: AtomicU64,
    /// Unsupported commands answered.
    pub unsupported: AtomicU64,
    /// Connections currently in RESP2.
    pub resp2_current: AtomicU64,
    /// Connections currently in RESP3.
    pub resp3_current: AtomicU64,
}

impl RespStats {
    /// Snapshots all counters (admin plane, relaxed loads).
    #[must_use]
    pub fn snapshot(&self) -> RespSnapshot {
        RespSnapshot {
            connections_current: self.connections_current.load(Ordering::Relaxed),
            connections_total: self.connections_total.load(Ordering::Relaxed),
            requests: self.requests.load(Ordering::Relaxed),
            bytes_in: self.bytes_in.load(Ordering::Relaxed),
            bytes_out: self.bytes_out.load(Ordering::Relaxed),
            protocol_errors: self.protocol_errors.load(Ordering::Relaxed),
            unsupported: self.unsupported.load(Ordering::Relaxed),
            resp2_current: self.resp2_current.load(Ordering::Relaxed),
            resp3_current: self.resp3_current.load(Ordering::Relaxed),
        }
    }

    /// Folds one closed connection's local metrics into the totals.
    pub fn fold(&self, metrics: &ConnMetrics) {
        self.requests.fetch_add(metrics.requests, Ordering::Relaxed);
        self.bytes_in.fetch_add(metrics.bytes_in, Ordering::Relaxed);
        self.bytes_out
            .fetch_add(metrics.bytes_out, Ordering::Relaxed);
        self.protocol_errors
            .fetch_add(metrics.protocol_errors, Ordering::Relaxed);
        self.unsupported
            .fetch_add(metrics.unsupported, Ordering::Relaxed);
    }
}

/// Point-in-time RESP statistics for the admin plane.
#[derive(Debug, Clone, Copy)]
pub struct RespSnapshot {
    /// Connections currently open.
    pub connections_current: u64,
    /// Connections opened since startup.
    pub connections_total: u64,
    /// Requests answered.
    pub requests: u64,
    /// Incoming bytes consumed.
    pub bytes_in: u64,
    /// Outgoing bytes produced.
    pub bytes_out: u64,
    /// Parse/bound failures answered.
    pub protocol_errors: u64,
    /// Unsupported commands answered.
    pub unsupported: u64,
    /// Connections currently in RESP2.
    pub resp2_current: u64,
    /// Connections currently in RESP3.
    pub resp3_current: u64,
}

/// Admin view of the RESP frontend (endpoint, namespace, live stats).
#[derive(Debug, Clone)]
pub struct RespAdmin {
    /// Bound endpoint (`host:port`) clients dial.
    pub endpoint: SocketAddr,
    /// Namespace id the frontend serves (Redis DB 0 maps here).
    pub namespace: u64,
    /// Shared statistics.
    pub stats: Arc<RespStats>,
}

/// Engine access for RESP connections: the embedded local client (same
/// ingress path as any embedded caller, never the native wire protocol).
#[derive(Debug, Clone)]
struct LocalExecutor {
    /// Embedded engine handle.
    client: LocalClient,
}

impl Executor for LocalExecutor {
    fn execute(&self, op: &Operation) -> Result<OperationResult, ExecuteError> {
        // `LocalClient`'s typed methods resolve chunked reads to inline
        // bytes on the caller thread; dispatch on the operation here so
        // every current and future operation has one mapping site.
        let out = match op {
            Operation::Get { key } => self
                .client
                .get(key)
                .map(OperationResult::Value)
                .map_err(map_engine)?,
            Operation::Set { key, value } => self
                .client
                .set(key, value.clone())
                .map(|()| OperationResult::Stored {
                    version: version_unknown(),
                })
                .map_err(map_engine)?,
            Operation::SetChunked { .. } | Operation::SetConditionalChunked { .. } => {
                return Err(ExecuteError::Internal);
            }
            Operation::SetRange { key, offset, patch } => self
                .client
                .set_range(key, *offset, patch.clone())
                .map(|()| OperationResult::Stored {
                    version: version_unknown(),
                })
                .map_err(map_engine)?,
            Operation::Delete { key } => self
                .client
                .delete(key)
                .map(|existed| OperationResult::Deleted { existed })
                .map_err(map_engine)?,
            Operation::Exists { key } => self
                .client
                .exists(key)
                .map(OperationResult::Exists)
                .map_err(map_engine)?,
            Operation::CounterGet { key } => self
                .client
                .counter_get(key)
                .map(OperationResult::Counter)
                .map_err(map_engine)?,
            Operation::CounterAdd { key, delta } => self
                .client
                .counter_add(key, *delta)
                .map(|value| OperationResult::CounterUpdated {
                    value,
                    version: version_unknown(),
                })
                .map_err(map_engine)?,
            Operation::ExpireAt { key, expires_at } => self
                .client
                .expire_at(key, *expires_at)
                .map(|applied| OperationResult::ExpirySet { applied })
                .map_err(map_engine)?,
            Operation::PersistExpiry { key } => self
                .client
                .persist_expiry(key)
                .map(|removed| OperationResult::ExpiryPersisted { removed })
                .map_err(map_engine)?,
            Operation::GetExpiry { key } => self
                .client
                .get_expiry(key)
                .map(OperationResult::Expiry)
                .map_err(map_engine)?,
            Operation::GetRange { key, offset, len } => self
                .client
                .get_range(key, *offset, *len)
                .map(OperationResult::Value)
                .map_err(map_engine)?,
            Operation::BytesLength { key } => self
                .client
                .bytes_length(key)
                .map(OperationResult::Length)
                .map_err(map_engine)?,
            Operation::SetConditional {
                key,
                value,
                condition,
                expiry,
            } => self
                .client
                .set_conditional(key, value.clone(), *condition, *expiry)
                .map(|(applied, version)| OperationResult::ConditionalSet { applied, version })
                .map_err(map_engine)?,
        };
        Ok(out)
    }

    fn now_micros(&self) -> u64 {
        unix_micros_now()
    }
}

/// The RESP edge only needs versions for shaping native responses it
/// already computed; the actual version lives in the tablet store, so a
/// placeholder that never crosses a boundary is sound here.
fn version_unknown() -> kivi_state::ObjectVersion {
    kivi_state::ObjectVersion::from_u64(0)
}

/// Maps engine failures onto the adapter's error surface (no Rust
/// internals cross the wire; the reply layer prefixes these).
#[allow(clippy::needless_pass_by_value)]
fn map_engine(error: kivi_engine::EngineError) -> ExecuteError {
    use kivi_engine::{EngineError as E, TabletError as T};
    use kivi_state::{ApplyError as A, OpError as O};
    match error {
        E::Overloaded | E::Tablet(T::Op(O::StaleRangeBase)) => ExecuteError::Overloaded,
        E::InvalidRequest { .. }
        | E::DedupExpired
        | E::SessionOverloaded
        | E::Tablet(
            T::Op(O::CounterOverflow)
            | T::Apply(A::CounterOverflow | A::VersionExhausted)
            | T::CommitExhausted { .. },
        ) => ExecuteError::Rejected,
        E::Tablet(T::Op(O::WrongType { .. }) | T::Apply(A::TypeMismatch { .. })) => {
            ExecuteError::WrongType
        }
        E::Tablet(T::Apply(A::UnresolvableSplice) | T::AuthorityMismatch { .. }) => {
            ExecuteError::Internal
        }
        E::WorkerDown { .. }
        | E::WorkerPanicked { .. }
        | E::NoRoute
        | E::UnknownTablet { .. }
        | E::UnknownWorker { .. }
        | E::HashAlgorithmUnsupported
        | E::NetStart(_)
        | E::Routing(_)
        | E::InvalidConfig { .. }
        | E::Durability(_)
        | E::Recovery(_)
        | E::Checkpoint(_)
        | E::Chunk(_)
        | _ => ExecuteError::Internal,
    }
}

/// Wall-clock Unix microseconds (relative-expiry anchoring only).
fn unix_micros_now() -> u64 {
    let Ok(elapsed) = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH) else {
        return 0;
    };
    elapsed.as_micros().try_into().unwrap_or(u64::MAX)
}

/// Serves one RESP listener until `shutdown` resolves: accepts connections,
/// drives each connection's bounded pipeline (fair per-turn budgets, then
/// yield so one huge pipeline cannot monopolize the frontend), and folds
/// per-connection metrics into `stats` on close.
pub async fn serve(
    listener: tokio::net::TcpListener,
    client: LocalClient,
    stats: Arc<RespStats>,
    shutdown: tokio::sync::watch::Receiver<bool>,
) {
    serve_with(listener, LocalExecutor { client }, stats, shutdown).await;
}

/// Serves one RESP listener over any [`Executor`] (single-node embedded
/// clients and replicated cluster nodes share the entire frontend; only
/// the executor differs).
pub async fn serve_with<E>(
    listener: tokio::net::TcpListener,
    executor: E,
    stats: Arc<RespStats>,
    mut shutdown: tokio::sync::watch::Receiver<bool>,
) where
    E: Executor + Clone + Send + 'static,
{
    let mut next_id: u64 = 1;
    loop {
        tokio::select! {
            _ = shutdown.changed() => break,
            accepted = listener.accept() => {
                let Ok((socket, _)) = accepted else { continue };
                // Request/response traffic is small frames in both
                // directions: disable Nagle so replies never wait for
                // delayed ACKs (same reason the native path avoids
                // batching tiny writes).
                if socket.set_nodelay(true).is_err() {
                    continue;
                }
                let id = next_id;
                next_id = next_id.wrapping_add(1);
                stats.connections_current.fetch_add(1, Ordering::Relaxed);
                stats.connections_total.fetch_add(1, Ordering::Relaxed);
                stats.resp2_current.fetch_add(1, Ordering::Relaxed);
                let executor = executor.clone();
                let stats = Arc::clone(&stats);
                tokio::spawn(async move {
                    serve_conn(socket, id, executor, stats).await;
                });
            }
        }
    }
}

/// Serves one connection: read, decode bounded pipelines in order, execute
/// via blocking offload (the embedded client blocks on its rendezvous),
/// and write replies in the same order.
async fn serve_conn<E>(socket: tokio::net::TcpStream, id: u64, executor: E, stats: Arc<RespStats>)
where
    E: Executor + Send + 'static,
{
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    let (mut reader, mut writer) = socket.into_split();
    let mut slot = Some(RespConnection::new(executor, ConnConfig::default(), id));
    let mut read_buf = vec![0u8; 64 * 1024];
    let mut version = RespVersion::V2;
    loop {
        let count = match reader.read(&mut read_buf).await {
            Ok(0) | Err(_) => break,
            Ok(count) => count,
        };
        let mut connection = slot.take().expect("slot holds the connection");
        if connection.push(&read_buf[..count]).is_err() {
            slot = Some(connection);
            break;
        }
        // Offload the blocking engine rendezvous out of the async
        // worker: move the connection in, drain on the blocking pool,
        // move it back with its replies.
        let drained = tokio::task::spawn_blocking(move || drain_owned(connection)).await;
        let Ok((replies, quit, back)) = drained else {
            break;
        };
        let connection = back;
        // Track version switches for the RESP2/RESP3 gauges.
        if connection.state.version != version {
            match connection.state.version {
                RespVersion::V2 => {
                    stats.resp3_current.fetch_sub(1, Ordering::Relaxed);
                    stats.resp2_current.fetch_add(1, Ordering::Relaxed);
                }
                RespVersion::V3 => {
                    stats.resp2_current.fetch_sub(1, Ordering::Relaxed);
                    stats.resp3_current.fetch_add(1, Ordering::Relaxed);
                }
            }
            version = connection.state.version;
        }
        // Coalesce the batch into one write: fewer packets per
        // round-trip (matters most for pipelined batches).
        let mut failed = false;
        if !replies.is_empty() {
            let bytes: usize = replies.iter().map(Vec::len).sum();
            let mut out = Vec::with_capacity(bytes);
            for reply in &replies {
                out.extend_from_slice(reply);
            }
            if writer.write_all(&out).await.is_err() {
                failed = true;
            }
        }
        if failed || writer.flush().await.is_err() {
            slot = Some(connection);
            break;
        }
        // Fairness: yield after every read batch so a pipelining client
        // cannot starve acceptors or other connections.
        tokio::task::yield_now().await;
        if quit {
            let _ = writer.shutdown().await;
            slot = Some(connection);
            break;
        }
        slot = Some(connection);
    }
    stats.connections_current.fetch_sub(1, Ordering::Relaxed);
    match version {
        RespVersion::V2 => {
            stats.resp2_current.fetch_sub(1, Ordering::Relaxed);
        }
        RespVersion::V3 => {
            stats.resp3_current.fetch_sub(1, Ordering::Relaxed);
        }
    }
    if let Some(connection) = slot {
        stats.fold(&connection.metrics);
    }
}

/// Drains an owned connection on the blocking pool (returns it for reuse).
/// Loops while complete frames remain so one read batch answers fully, but
/// caps the batch so a single read can never produce an unbounded reply
/// vector even if budgets shift.
fn drain_owned<E>(mut connection: RespConnection<E>) -> (Vec<Vec<u8>>, bool, RespConnection<E>)
where
    E: Executor,
{
    const MAX_BATCH_REPLIES: usize = 1024;
    let mut all = Vec::new();
    let mut quit = false;
    while !quit && all.len() < MAX_BATCH_REPLIES {
        let (more, close) = connection.drain();
        if more.is_empty() {
            break;
        }
        all.extend(more);
        quit = close;
    }
    (all, quit, connection)
}
