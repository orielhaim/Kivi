//! Admin view of the Redis/RESP compatibility edge (feature `redis-compat`).
//!
//! The engine serves the edge from its own workers' reactors, on the same
//! thread that owns the tablets a request addresses. This module only carries
//! the admin DTO: bound endpoint, namespace, and the shared statistics the
//! engine's workers increment.
//!
//! Even when compiled in, the listener only starts with `--redis-listen`.

use std::net::SocketAddr;
use std::sync::Arc;

/// Admin view of the RESP frontend (endpoint, namespace, live stats).
#[derive(Debug, Clone)]
pub struct RespAdmin {
    /// Bound endpoint (`host:port`) clients dial.
    pub endpoint: SocketAddr,
    /// Namespace id the frontend serves (Redis DB 0 maps here).
    pub namespace: u64,
    /// Shared statistics.
    pub stats: Arc<kivi_resp::RespStats>,
}
