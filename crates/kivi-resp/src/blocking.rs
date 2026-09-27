//! A RESP serving loop for embedders whose executor is not a worker-local
//! handle.
//!
//! The engine serves RESP from its own reactor, where a connection is answered
//! by the thread that owns the tablet. That is the fast path and the one worth
//! optimising. An embedder that routes operations somewhere else — a replicated
//! node that has to run consensus before it can answer — has no such thread to
//! borrow, and still needs a correct, bounded, one-connection-per-thread
//! server.
//!
//! The turn logic itself is not duplicated here. [`crate::RespConnection`] owns
//! parsing, dispatch, and reply shaping; this module owns only the socket, the
//! accept loop, and the buffers.
//!
//! Request/response traffic is small frames in both directions, so Nagle is
//! disabled: a reply must not wait for a delayed ACK.

use std::io::{Read, Write};
use std::net::TcpListener;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::connection::{ConnConfig, DrainOutcome, RespConnection, RespVersion};
use crate::stats::RespStats;
use crate::translate::Executor;

/// Read buffer per connection. A pipeline shares this; the buffer is reused
/// for the connection's life and never allocated per command.
const READ_CHUNK: usize = 64 * 1024;

/// Reply buffer above which the connection releases it rather than letting one
/// oversized value pin its allocation until the client disconnects.
const REPLY_BUFFER_RECLAIM: usize = 8 * 1024 * 1024;

/// Shared handles every accepted connection needs.
#[derive(Clone)]
pub struct BlockingServe {
    /// Factory for one connection's executor. A connection that outlives this
    /// handle keeps its own executor, so the handle may be dropped.
    executor: Arc<dyn Fn(u64) -> Box<dyn Executor> + Send + Sync>,
    /// Per-connection resource bounds.
    config: ConnConfig,
    /// Frontend statistics.
    stats: Arc<RespStats>,
    /// Set to stop accepting; existing connections finish on their own.
    shutdown: Arc<AtomicU64>,
}

impl std::fmt::Debug for BlockingServe {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BlockingServe")
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

impl BlockingServe {
    /// Builds a server whose connections each get an executor from `executor`.
    #[must_use]
    pub fn new(
        executor: impl Fn(u64) -> Box<dyn Executor> + Send + Sync + 'static,
        config: ConnConfig,
        stats: Arc<RespStats>,
    ) -> Self {
        Self {
            executor: Arc::new(executor),
            config,
            stats,
            shutdown: Arc::new(AtomicU64::new(0)),
        }
    }

    /// A handle that stops this server accepting new connections.
    #[must_use]
    pub fn shutdown_handle(&self) -> Shutdown {
        Shutdown {
            flag: Arc::clone(&self.shutdown),
        }
    }

    /// Accepts and serves until `shutdown` is signalled.
    pub fn serve(&self, listener: &TcpListener) {
        let mut next_id: u64 = 1;
        for accepted in listener.incoming() {
            if self.shutdown.load(Ordering::Relaxed) != 0 {
                return;
            }
            let Ok(socket) = accepted else {
                std::thread::sleep(std::time::Duration::from_millis(10));
                continue;
            };
            if socket.set_nodelay(true).is_err() {
                continue;
            }
            let id = next_id;
            next_id = next_id.wrapping_add(1);
            self.stats
                .connections_current
                .fetch_add(1, Ordering::Relaxed);
            self.stats.connections_total.fetch_add(1, Ordering::Relaxed);
            self.stats.resp2_current.fetch_add(1, Ordering::Relaxed);
            let executor = Arc::clone(&self.executor);
            let config = self.config;
            let stats = Arc::clone(&self.stats);
            // One thread owns its connection end to end. The executor may block
            // (a replicated node must run consensus), so the connection cannot
            // be an async task on a shared runtime without either stalling it
            // or moving the connection to a blocking pool mid-batch.
            if std::thread::Builder::new()
                .name(format!("kivi-resp-{id}"))
                .spawn({
                    let stats = Arc::clone(&stats);
                    move || serve_conn(socket, id, executor(id), config, &stats)
                })
                .is_err()
            {
                // The counters were incremented above for a connection that
                // will now never exist; undo them or the gauges drift upward
                // for the life of the process.
                self.stats
                    .connections_current
                    .fetch_sub(1, Ordering::Relaxed);
                self.stats.connections_total.fetch_sub(1, Ordering::Relaxed);
                self.stats.resp2_current.fetch_sub(1, Ordering::Relaxed);
            }
        }
    }
}

/// Stops a [`BlockingServe`] from accepting new connections.
///
/// Triggered explicitly, never by dropping: the accept loop holds its own
/// handle, so a dropped `Shutdown` would leave it serving.
#[derive(Debug, Clone)]
pub struct Shutdown {
    flag: Arc<AtomicU64>,
}

impl Shutdown {
    /// Signals the server to stop accepting. Connections already open finish on
    /// their own; this is about the listener, not the sessions.
    pub fn trigger(&self) {
        self.flag.store(1, Ordering::Relaxed);
    }
}

/// Serves one connection: read, decode buffered pipelines in order, execute,
/// and write each turn's replies in one syscall.
fn serve_conn<E: Executor>(
    socket: std::net::TcpStream,
    id: u64,
    executor: E,
    config: ConnConfig,
    stats: &RespStats,
) {
    let mut reader = socket;
    let Ok(mut writer) = reader.try_clone() else {
        stats.connections_current.fetch_sub(1, Ordering::Relaxed);
        return;
    };
    let mut connection = RespConnection::new(executor, config, id);
    let mut read_buf = vec![0u8; READ_CHUNK];
    let mut out: Vec<u8> = Vec::with_capacity(READ_CHUNK);
    let mut flushed = 0usize;
    let mut version = RespVersion::V2;
    let mut live = true;
    while live {
        let count = match reader.read(&mut read_buf) {
            Ok(0) | Err(_) => break,
            Ok(count) => count,
        };
        if connection.push(&read_buf[..count]).is_err() {
            break;
        }
        // One write per turn, carrying only the bytes since the last write.
        // The reply buffer is connection-local and reused, so it must not be
        // re-sent from the front: writing the whole buffer per turn would
        // resend every earlier reply and desynchronise a pipelined client.
        while connection.has_unconsumed() {
            let DrainOutcome { replies, close, .. } = connection.drain(&mut out);
            if replies == 0 {
                break;
            }
            if out.len() > flushed && writer.write_all(&out[flushed..]).is_err() {
                live = false;
                break;
            }
            flushed = out.len();
            if close {
                let _ = writer.flush();
                live = false;
                break;
            }
        }
        if out.len() > REPLY_BUFFER_RECLAIM {
            out.clear();
            flushed = 0;
        }
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
    stats.fold(&connection.metrics);
}
