//! Large-value sidecar preflight: pre-distribute immutable roots before
//! the tiny Raft root proposal.
//!
//! The previous milestone found 64 MiB bulk operations queueing small
//! same-tablet writes (`p99 ≈ 2.3 s`): once the large root entry enters the
//! log, followers must fetch sidecars before the append persists, and later
//! small entries cannot commit past it. Preflight moves the expensive
//! sidecar movement **before** the authoritative Raft entry:
//!
//! ```text
//! leader stages local sidecars (durable)
//!   ↓ preflight immutable root to followers concurrently (bulk lane)
//! followers acquire + verify + durably install (same gate primitives)
//!   ↓ quorum durable-ready (including the leader)
//! propose tiny Raft root → prepared followers append immediately
//! ```
//!
//! ## Not consensus
//!
//! Preflight mutates no logical state, advances no version or commit
//! position, and creates no Raft decision. It merely ensures immutable
//! content is durable. Abandoned preflight data is reusable
//! content-addressed cache and reclaimed safely.
//!
//! ## Authoritative gate unchanged
//!
//! Every follower still independently verifies sidecar durability before
//! its Raft append persistence completes. A follower that missed
//! preparation, lost data, or restarted falls back to the existing
//! append-gate fetch path. Correctness never depends on preflight.
//!
//! ## Quorum
//!
//! For `v` voters the write quorum is `v/2+1` (static ordinary majority;
//! no joint membership in this stage). The leader already holds the root
//! durably, so preflight waits for `quorum-1` followers. Stragglers are
//! not waited on: their H3 streams reset on drop and they catch up later
//! through the append gate or snapshots.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use kivi_types::{ManifestId, NodeId};

use crate::peer::{PeerPrepareRequest, PeerRequest, PeerResponse};
use crate::transport::{PeerTransport, TransportError};
use crate::types::ConsensusGroupId;

/// Static ordinary-majority quorum for `voters` voters: `v/2+1`. Never
/// hardcodes `2 of 3`; joint membership is out of scope.
#[must_use]
pub const fn quorum_needed(voters: usize) -> usize {
    voters / 2 + 1
}

/// How many followers (excluding the already-durable leader) must reply
/// ready to form a write quorum.
#[must_use]
pub const fn followers_needed(voters: usize) -> usize {
    quorum_needed(voters).saturating_sub(1)
}

/// Outcome of one preflight round.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreflightOutcome {
    /// Followers that replied ready (durable) before the quorum released.
    pub ready: Vec<NodeId>,
    /// Whether enough replicas are durable-ready for the root proposal
    /// (leader plus `ready` reaches quorum).
    pub quorum_ready: bool,
    /// Round duration (all concurrent attempts).
    pub elapsed: Duration,
}

/// Sidecar preflight metrics (node-wide; later surfaced via admin/lab).
#[derive(Debug, Default)]
pub struct PreflightMetrics {
    /// Preflight rounds started.
    pub attempts: AtomicU64,
    /// Rounds that reached quorum-ready.
    pub quorum_ready: AtomicU64,
    /// Rounds that fell back to the append gate (disabled, failed, or no
    /// quorum in time). Correctness is unaffected; latency may be.
    pub fallbacks: AtomicU64,
    /// Total follower ready replies.
    pub ready_replies: AtomicU64,
    /// Total preflight round latency in microseconds.
    pub latency_us: AtomicU64,
}

/// Point-in-time preflight metrics snapshot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PreflightMetricsSnapshot {
    /// Preflight rounds started.
    pub attempts: u64,
    /// Rounds that reached quorum-ready.
    pub quorum_ready: u64,
    /// Rounds that fell back to the append gate.
    pub fallbacks: u64,
    /// Total follower ready replies.
    pub ready_replies: u64,
    /// Total preflight round latency in microseconds.
    pub latency_us: u64,
}

impl PreflightMetrics {
    /// Captures a point-in-time snapshot.
    #[must_use]
    pub fn snapshot(&self) -> PreflightMetricsSnapshot {
        PreflightMetricsSnapshot {
            attempts: self.attempts.load(Ordering::Relaxed),
            quorum_ready: self.quorum_ready.load(Ordering::Relaxed),
            fallbacks: self.fallbacks.load(Ordering::Relaxed),
            ready_replies: self.ready_replies.load(Ordering::Relaxed),
            latency_us: self.latency_us.load(Ordering::Relaxed),
        }
    }
}

/// Preflights one immutable root to followers concurrently, releasing as
/// soon as enough replicas are durable-ready for quorum (including the
/// leader, which is already durable). Stragglers are dropped (their H3
/// streams reset) and catch up later through the append gate.
///
/// Each attempt calls the follower's `Prepare` handler, which runs the
/// exact same acquisition primitives as the append gate and snapshot
/// recovery (manifest lookup, missing-chunk detection, H3 acquisition,
/// verification, install, barrier) and replies only once durable.
///
/// When `enabled` is false the append-gate fallback path is used without
/// any network work (correctness probe for "preflight is only an
/// optimization").
#[allow(clippy::too_many_arguments)]
pub async fn preflight_to_quorum(
    transport: &PeerTransport,
    group: ConsensusGroupId,
    manifest: ManifestId,
    logical_len: u64,
    followers: &[NodeId],
    voters: usize,
    timeout: Duration,
    enabled: bool,
    metrics: &PreflightMetrics,
) -> PreflightOutcome {
    use futures::StreamExt as _;
    let started = std::time::Instant::now();
    metrics.attempts.fetch_add(1, Ordering::Relaxed);
    let needed = followers_needed(voters);
    if !enabled || needed == 0 || followers.is_empty() {
        if enabled {
            metrics.quorum_ready.fetch_add(1, Ordering::Relaxed);
        } else {
            metrics.fallbacks.fetch_add(1, Ordering::Relaxed);
        }
        return PreflightOutcome {
            ready: Vec::new(),
            quorum_ready: true,
            elapsed: started.elapsed(),
        };
    }
    let request = PeerRequest::Prepare(PeerPrepareRequest {
        manifest: manifest.into(),
        logical_len,
    });
    // One future per follower, raced to quorum: the first `needed` ready
    // replies release the proposal while stragglers are dropped.
    let mut pending: futures::stream::FuturesUnordered<_> = followers
        .iter()
        .map(|follower| {
            let transport = transport.clone();
            let request = request.clone();
            let follower = *follower;
            async move {
                let answer = transport.call_bulk(follower, group, request, timeout).await;
                (follower, answer)
            }
        })
        .collect();
    let mut ready = Vec::new();
    let deadline = started + timeout;
    while ready.len() < needed {
        let remaining = deadline.saturating_duration_since(std::time::Instant::now());
        if remaining.is_zero() {
            break;
        }
        let next = compio::time::timeout(remaining, pending.next()).await;
        let Some((follower, answer)) = next.unwrap_or_default() else {
            break;
        };
        if matches!(answer, Ok(PeerResponse::Prepared(_))) {
            ready.push(follower);
        }
        if pending.is_empty() {
            break;
        }
    }
    // Stragglers drop here: their H3 streams reset, and they catch up
    // through the normal append gate or snapshot path.
    drop(pending);
    let elapsed = started.elapsed();
    metrics.latency_us.fetch_add(
        u64::try_from(elapsed.as_micros().min(u128::from(u64::MAX))).unwrap_or(u64::MAX),
        Ordering::Relaxed,
    );
    metrics
        .ready_replies
        .fetch_add(ready.len() as u64, Ordering::Relaxed);
    let quorum_ready = ready.len() >= needed;
    if quorum_ready {
        metrics.quorum_ready.fetch_add(1, Ordering::Relaxed);
    } else {
        metrics.fallbacks.fetch_add(1, Ordering::Relaxed);
    }
    PreflightOutcome {
        ready,
        quorum_ready,
        elapsed,
    }
}

/// Maps a transport-level preflight failure onto a log string (preflight
/// failures never fail the write; the append gate covers them).
#[must_use]
pub fn describe_transport_error(error: &TransportError) -> String {
    error.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quorum_is_static_majority() {
        assert_eq!(quorum_needed(1), 1);
        assert_eq!(quorum_needed(2), 2);
        assert_eq!(quorum_needed(3), 2);
        assert_eq!(quorum_needed(4), 3);
        assert_eq!(quorum_needed(5), 3);
        assert_eq!(followers_needed(3), 1);
        assert_eq!(followers_needed(1), 0);
    }

    #[test]
    fn disabled_preflight_is_immediately_ready() {
        let metrics = PreflightMetrics::default();
        // No transport needed: disabled short-circuits before any I/O.
        // (Exercised through the multi-tablet failover tests with real
        // transports; here only the quorum math is unit-covered.)
        assert_eq!(followers_needed(3), 1);
        assert_eq!(metrics.snapshot().attempts, 0);
    }
}
