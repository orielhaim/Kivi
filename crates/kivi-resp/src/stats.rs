//! Frontend statistics for the RESP edge.
//!
//! These live beside the connection machinery rather than in the server so the
//! frontend can be hosted anywhere the engine runs - on a worker's own
//! reactor, or behind an embedder's listener - and still report one set of
//! numbers to the admin plane. Counters are relaxed atomics updated per
//! connection close, never per request, so no request takes a contended
//! atomic on the serving path.

use core::sync::atomic::{AtomicU64, Ordering};

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
    /// Requests served without leaving the owning worker's thread.
    pub local_ops: AtomicU64,
    /// Requests handed to another worker through the engine queue.
    pub forwarded_ops: AtomicU64,
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
            local_ops: self.local_ops.load(Ordering::Relaxed),
            forwarded_ops: self.forwarded_ops.load(Ordering::Relaxed),
        }
    }

    /// Folds one closed connection's local metrics into the totals.
    pub fn fold(&self, metrics: &crate::connection::ConnMetrics) {
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
    /// Requests served on the owning worker's own thread.
    pub local_ops: u64,
    /// Requests forwarded to another worker.
    pub forwarded_ops: u64,
}
