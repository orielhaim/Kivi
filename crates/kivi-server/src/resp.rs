//! Admin view of the Redis/RESP compatibility edge (feature `redis-compat`).
//!
//! The edge itself is served by the engine's own workers, on the same reactor
//! that owns the tablets a request addresses:
//!
//! ```text
//! socket → RESP parse → route → LiveTablet::execute → RESP reply → socket
//! ```
//!
//! It used to be served here instead, by a thread per connection reaching
//! storage through the engine's request queue. That cost two thread wakeups per
//! request, and the wakeups — not the parsing, not the store — were the whole
//! cost of a small `GET`. Moving the edge into the engine is also why
//! [`kivi_resp::RespStats`] and its counters live in `kivi-resp`: the thing that
//! increments them is no longer in this crate.
//!
//! Even when compiled in, the listener only starts with `--redis-listen`.

use std::net::SocketAddr;
use std::sync::Arc;

pub use kivi_resp::RespStats;

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
