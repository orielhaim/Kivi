//! Deterministic message transport simulator (not a simulated IP stack).
//!
//! [`NetSim`] models nodes, directed links, and message deliveries with
//! explicit latency, bandwidth, capacity, and connectivity state. It owns no
//! scheduler: [`send`](NetSim::send) returns a [`NetSendOutcome`] describing
//! what must happen (schedule this delivery / record this drop), and the
//! driver — unit-test loops or the cluster runner — performs it. Time always
//! arrives as an explicit `now` parameter; nothing here reads a clock.
//!
//! Capacity discipline (the driver contract): each scheduled delivery holds
//! one in-flight slot until the driver reports its fate via
//! [`complete_delivery`](NetSim::complete_delivery), whether delivered,
//! policy-dropped, or cancelled. Forgetting the call leaks a slot (visible
//! in tests as spurious congestion); calling it twice is caught by a debug
//! assertion.
//!
//! Not modeled here (deliberately): TCP/QUIC, sockets, fragmentation,
//! retransmission, congestion control, or peer protocol semantics. Fault
//! actions `duplicate` / `reorder` / `delay` are driver operations — schedule
//! an extra copy, or reschedule later — taken through the checked scheduler
//! so they stay deterministic.

use core::fmt;
use core::time::Duration;
use std::collections::{BTreeMap, BTreeSet};

use kivi_types::{NodeId, NodeIncarnation, Ticks};

/// One process endpoint: the node plus the incarnation addressing it.
///
/// Incarnations matter because a restarted process is a different endpoint:
/// sends from an isolated (crashed) endpoint are rejected, and receivers
/// validate sender incarnations against the lifecycle table, so traffic from
/// an obsolete incarnation can never become valid after restart.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Endpoint {
    node: NodeId,
    incarnation: NodeIncarnation,
}

impl Endpoint {
    /// Builds an endpoint from its node and incarnation.
    #[must_use]
    pub const fn new(node: NodeId, incarnation: NodeIncarnation) -> Self {
        Self { node, incarnation }
    }

    /// Returns the node half.
    #[must_use]
    pub const fn node(self) -> NodeId {
        self.node
    }

    /// Returns the incarnation half.
    #[must_use]
    pub const fn incarnation(self) -> NodeIncarnation {
        self.incarnation
    }
}

impl fmt::Display for Endpoint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "n{}@{}", self.node.as_u64(), self.incarnation.as_u64())
    }
}

/// Stable message identity, minted in send order with checked exhaustion.
///
/// Duplicated deliveries share one `MsgId` (they are the same message twice);
/// distinct sends never share one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MsgId(u64);

impl MsgId {
    /// First identifier minted by a fresh simulator.
    pub const FIRST: Self = Self(0);

    /// Wraps a raw identifier value.
    #[must_use]
    pub const fn from_u64(value: u64) -> Self {
        Self(value)
    }

    /// Returns the raw value.
    #[must_use]
    pub const fn as_u64(self) -> u64 {
        self.0
    }

    /// Successor identifier for the next sent message.
    ///
    /// # Errors
    ///
    /// Returns [`MsgIdExhausted`] at `u64::MAX` instead of reusing an
    /// identifier (a reused identity would alias two distinct messages).
    pub const fn next(self) -> Result<Self, MsgIdExhausted> {
        match self.0.checked_add(1) {
            Some(value) => Ok(Self(value)),
            None => Err(MsgIdExhausted),
        }
    }
}

impl From<MsgId> for u64 {
    /// Returns the raw identifier value.
    fn from(id: MsgId) -> Self {
        id.0
    }
}

impl From<u64> for MsgId {
    /// Wraps a raw identifier value.
    fn from(value: u64) -> Self {
        MsgId(value)
    }
}

impl fmt::Display for MsgId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "msg{}", self.0)
    }
}

/// Reported when no message identifier remains to mint.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, thiserror::Error)]
#[error("message identifier space exhausted")]
pub struct MsgIdExhausted;

/// Static behavior of one directed link.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LinkConfig {
    base_latency: Duration,
    bandwidth_bytes_per_sec: u64,
    capacity: usize,
}

impl LinkConfig {
    /// Builds a link configuration.
    ///
    /// `bandwidth_bytes_per_sec` must be nonzero (serialization delay divides
    /// by it); `capacity` bounds in-flight messages and may be zero (a link
    /// that admits nothing — every send congests).
    ///
    /// # Errors
    ///
    /// Returns [`NetError::ZeroBandwidth`] for a zero bandwidth.
    pub const fn new(
        base_latency: Duration,
        bandwidth_bytes_per_sec: u64,
        capacity: usize,
    ) -> Result<Self, NetError> {
        if bandwidth_bytes_per_sec == 0 {
            return Err(NetError::ZeroBandwidth);
        }
        Ok(Self {
            base_latency,
            bandwidth_bytes_per_sec,
            capacity,
        })
    }

    /// Returns the propagation latency before serialization delay.
    #[must_use]
    pub const fn base_latency(self) -> Duration {
        self.base_latency
    }

    /// Returns the serialization bandwidth in bytes per second.
    #[must_use]
    pub const fn bandwidth_bytes_per_sec(self) -> u64 {
        self.bandwidth_bytes_per_sec
    }

    /// Returns the maximum in-flight (scheduled, unfate-reported) messages.
    #[must_use]
    pub const fn capacity(self) -> usize {
        self.capacity
    }
}

/// Connectivity state of one directed link.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LinkStatus {
    /// Delivering normally.
    Up,
    /// Fault-partitioned: sends drop until healed (see `heal`).
    Partitioned,
    /// Administratively disconnected: sends drop until reconnected.
    Disconnected,
}

impl fmt::Display for LinkStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Up => write!(f, "up"),
            Self::Partitioned => write!(f, "partitioned"),
            Self::Disconnected => write!(f, "disconnected"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct LinkState {
    config: LinkConfig,
    status: LinkStatus,
    in_flight: usize,
}

/// A message ready for the driver to schedule for delivery.
///
/// Retains everything a delivered message must identify: source
/// node/incarnation, destination node, message identity, send tick, delivery
/// tick, and payload. The destination *incarnation* is intentionally absent:
/// it is resolved at delivery time against the lifecycle table, which is
/// what makes stale-incarnation rejection possible.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NetDelivery {
    id: MsgId,
    from: Endpoint,
    to: NodeId,
    sent_at: Ticks,
    deliver_at: Ticks,
    payload: Vec<u8>,
}

impl NetDelivery {
    /// Returns the message identity (shared by duplicate deliveries).
    #[must_use]
    pub const fn id(&self) -> MsgId {
        self.id
    }

    /// Returns the source endpoint (node + sender incarnation).
    #[must_use]
    pub const fn from(&self) -> Endpoint {
        self.from
    }

    /// Returns the destination node (incarnation resolved at delivery).
    #[must_use]
    pub const fn to(&self) -> NodeId {
        self.to
    }

    /// Returns the tick the message was sent at.
    #[must_use]
    pub const fn sent_at(&self) -> Ticks {
        self.sent_at
    }

    /// Returns the tick the message is due for delivery.
    #[must_use]
    pub const fn deliver_at(&self) -> Ticks {
        self.deliver_at
    }

    /// Returns the payload.
    #[must_use]
    pub fn payload(&self) -> &[u8] {
        &self.payload
    }

    /// Builds the typed fault-policy context for this delivery.
    #[must_use]
    pub fn fault_context(&self) -> NetDeliveryContext {
        NetDeliveryContext {
            from: self.from,
            to: self.to,
            msg: self.id,
            size_bytes: self.payload.len(),
            deliver_at: self.deliver_at,
        }
    }
}

/// Typed fault context for network deliveries: the `Ctx` instantiating
/// `FaultPolicy<Ctx>` for drop/delay/duplicate decisions. No IP, transport,
/// or protocol concepts — just who, what size, and when.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NetDeliveryContext {
    from: Endpoint,
    to: NodeId,
    msg: MsgId,
    size_bytes: usize,
    deliver_at: Ticks,
}

impl NetDeliveryContext {
    /// Returns the source endpoint.
    #[must_use]
    pub const fn from(self) -> Endpoint {
        self.from
    }

    /// Returns the destination node.
    #[must_use]
    pub const fn to(self) -> NodeId {
        self.to
    }

    /// Returns the message identity.
    #[must_use]
    pub const fn msg(self) -> MsgId {
        self.msg
    }

    /// Returns the payload size in bytes.
    #[must_use]
    pub const fn size_bytes(self) -> usize {
        self.size_bytes
    }

    /// Returns the scheduled delivery tick.
    #[must_use]
    pub const fn deliver_at(self) -> Ticks {
        self.deliver_at
    }
}

/// Why a send did not produce a delivery.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NetDropReason {
    /// No directed link exists (legitimate for asymmetric topologies).
    NoRoute,
    /// The directed link is partitioned.
    Partitioned,
    /// The directed link is disconnected.
    Disconnected,
    /// The sending endpoint instance is isolated (e.g. crashed).
    EndpointDown,
    /// The link already holds `capacity` in-flight messages.
    Congested,
}

impl fmt::Display for NetDropReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoRoute => write!(f, "no-route"),
            Self::Partitioned => write!(f, "partitioned"),
            Self::Disconnected => write!(f, "disconnected"),
            Self::EndpointDown => write!(f, "endpoint-down"),
            Self::Congested => write!(f, "congested"),
        }
    }
}

/// Outcome of one send: either a delivery for the driver to schedule, or a
/// deterministic drop with its reason (recorded in the trace, never silent).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NetSendOutcome {
    /// Schedule `delivery` at its `deliver_at` tick.
    Scheduled {
        /// The delivery to schedule.
        delivery: NetDelivery,
    },
    /// The message was dropped for `reason`.
    Dropped {
        /// Why the message was dropped.
        reason: NetDropReason,
    },
}

/// Network simulator failures: topology misuse and checked-arithmetic
/// exhaustion. Reachability outcomes are [`NetSendOutcome::Dropped`], not
/// errors — only misuse and exhaustion fail.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, thiserror::Error)]
#[non_exhaustive]
pub enum NetError {
    /// A referenced node was never added.
    #[error("unknown node {node}")]
    UnknownNode {
        /// The missing node.
        node: NodeId,
    },
    /// A delivery tick computation exceeded the 64-bit microsecond range.
    #[error("delivery tick overflows u64 microseconds")]
    TickOverflow,
    /// No message identifier remains to mint.
    #[error("no message identifier remains to mint")]
    MsgIdExhausted(#[from] MsgIdExhausted),
    /// Link bandwidth must be nonzero.
    #[error("link bandwidth must be nonzero")]
    ZeroBandwidth,
}

/// Deterministic directed-link network simulator.
#[derive(Debug, Clone)]
pub struct NetSim {
    nodes: BTreeSet<NodeId>,
    links: BTreeMap<(NodeId, NodeId), LinkState>,
    isolated: BTreeSet<Endpoint>,
    next_msg: MsgId,
}

impl NetSim {
    /// Creates an empty simulator (no nodes, no links).
    #[must_use]
    pub fn new() -> Self {
        Self {
            nodes: BTreeSet::new(),
            links: BTreeMap::new(),
            isolated: BTreeSet::new(),
            next_msg: MsgId::FIRST,
        }
    }
}

impl Default for NetSim {
    /// Creates an empty simulator (see [`new`](Self::new)).
    fn default() -> Self {
        Self::new()
    }
}

impl NetSim {
    /// Adds a node. Returns `false` if already present.
    pub fn add_node(&mut self, node: NodeId) -> bool {
        self.nodes.insert(node)
    }

    /// Whether the node is known.
    #[must_use]
    pub fn has_node(&self, node: NodeId) -> bool {
        self.nodes.contains(&node)
    }

    /// Creates or replaces the directed link `from -> to`.
    ///
    /// Replacement keeps the current in-flight count (outstanding messages
    /// are unaffected by reconfiguration) and resets a non-`Up` status to
    /// `Up`: reconfiguration is an administrative act, not a fault action.
    ///
    /// # Errors
    ///
    /// Returns [`NetError::UnknownNode`] if either endpoint node is unknown.
    pub fn set_link(
        &mut self,
        from: NodeId,
        to: NodeId,
        config: LinkConfig,
    ) -> Result<(), NetError> {
        self.require_node(from)?;
        self.require_node(to)?;
        let in_flight = self.links.get(&(from, to)).map_or(0, |link| link.in_flight);
        self.links.insert(
            (from, to),
            LinkState {
                config,
                status: LinkStatus::Up,
                in_flight,
            },
        );
        Ok(())
    }

    /// Returns the status of a directed link, or `None` if absent.
    #[must_use]
    pub fn link_status(&self, from: NodeId, to: NodeId) -> Option<LinkStatus> {
        self.links.get(&(from, to)).map(|link| link.status)
    }

    /// Returns the in-flight count of a directed link (0 if absent).
    #[must_use]
    pub fn in_flight(&self, from: NodeId, to: NodeId) -> usize {
        self.links.get(&(from, to)).map_or(0, |link| link.in_flight)
    }

    /// Sends `payload` from `from` to `to` at tick `now`.
    ///
    /// Delivery tick = `now + base_latency + size * 1s / bandwidth`, all
    /// checked. A scheduled delivery holds one capacity slot until the
    /// driver reports its fate via [`complete_delivery`](Self::complete_delivery).
    ///
    /// # Errors
    ///
    /// Returns [`NetError`] on unknown nodes, tick overflow, or identifier
    /// exhaustion. Unreachability yields `Dropped`, never an error.
    pub fn send(
        &mut self,
        from: Endpoint,
        to: NodeId,
        payload: Vec<u8>,
        now: Ticks,
    ) -> Result<NetSendOutcome, NetError> {
        self.require_node(from.node())?;
        self.require_node(to)?;
        if self.isolated.contains(&from) {
            return Ok(NetSendOutcome::Dropped {
                reason: NetDropReason::EndpointDown,
            });
        }
        let Some(link) = self.links.get(&(from.node(), to)) else {
            return Ok(NetSendOutcome::Dropped {
                reason: NetDropReason::NoRoute,
            });
        };
        match link.status {
            LinkStatus::Partitioned => {
                return Ok(NetSendOutcome::Dropped {
                    reason: NetDropReason::Partitioned,
                });
            }
            LinkStatus::Disconnected => {
                return Ok(NetSendOutcome::Dropped {
                    reason: NetDropReason::Disconnected,
                });
            }
            LinkStatus::Up => {}
        }
        if link.in_flight >= link.config.capacity() {
            return Ok(NetSendOutcome::Dropped {
                reason: NetDropReason::Congested,
            });
        }
        let deliver_at = Self::delivery_tick(now, link.config, payload.len())?;
        let id = self.next_msg;
        self.next_msg = id.next()?;
        if let Some(slot) = self.links.get_mut(&(from.node(), to)) {
            slot.in_flight += 1;
        }
        Ok(NetSendOutcome::Scheduled {
            delivery: NetDelivery {
                id,
                from,
                to,
                sent_at: now,
                deliver_at,
                payload,
            },
        })
    }

    /// Reports the fate of one in-flight message (delivered, policy-dropped,
    /// or cancelled), freeing its capacity slot.
    pub fn complete_delivery(&mut self, from: NodeId, to: NodeId) {
        if let Some(link) = self.links.get_mut(&(from, to)) {
            debug_assert!(
                link.in_flight > 0,
                "completed more deliveries than sent on {from}->{to}"
            );
            link.in_flight = link.in_flight.saturating_sub(1);
        }
    }

    /// Admits one fault-injected copy (duplicate or deferral) on a directed
    /// link without minting a new message identity. Returns `false` — and
    /// admits nothing — when the link is missing or already at capacity, in
    /// which case the driver drops the copy as congested.
    pub fn duplicate_slot(&mut self, from: NodeId, to: NodeId) -> bool {
        if let Some(link) = self.links.get_mut(&(from, to))
            && link.in_flight < link.config.capacity()
        {
            link.in_flight += 1;
            return true;
        }
        false
    }

    /// Symmetrically partitions `a <-> b` (both directions). Returns whether
    /// any link state changed.
    ///
    /// # Errors
    ///
    /// Returns [`NetError::UnknownNode`] if either node is unknown. Absent
    /// directed links are left alone (asymmetric topologies stay asymmetric).
    pub fn partition(&mut self, a: NodeId, b: NodeId) -> Result<bool, NetError> {
        self.require_node(a)?;
        self.require_node(b)?;
        let first = self.set_status(a, b, LinkStatus::Partitioned, LinkStatus::Up);
        let second = self.set_status(b, a, LinkStatus::Partitioned, LinkStatus::Up);
        Ok(first || second)
    }

    /// Partitions one direction only: `from -> to` drops while `to -> from`
    /// still delivers.
    ///
    /// # Errors
    ///
    /// Returns [`NetError::UnknownNode`] if either node is unknown.
    pub fn partition_one_way(&mut self, from: NodeId, to: NodeId) -> Result<bool, NetError> {
        self.require_node(from)?;
        self.require_node(to)?;
        Ok(self.set_status(from, to, LinkStatus::Partitioned, LinkStatus::Up))
    }

    /// Heals symmetrically partitioned links back to `Up`. Links in any
    /// other state (including `Disconnected`) are untouched. Returns whether
    /// any link state changed.
    ///
    /// # Errors
    ///
    /// Returns [`NetError::UnknownNode`] if either node is unknown.
    pub fn heal(&mut self, a: NodeId, b: NodeId) -> Result<bool, NetError> {
        self.require_node(a)?;
        self.require_node(b)?;
        let first = self.set_status(a, b, LinkStatus::Up, LinkStatus::Partitioned);
        let second = self.set_status(b, a, LinkStatus::Up, LinkStatus::Partitioned);
        Ok(first || second)
    }

    /// Symmetrically disconnects `a <-> b`. Returns whether any link state
    /// changed. See [`reconnect`](Self::reconnect).
    ///
    /// # Errors
    ///
    /// Returns [`NetError::UnknownNode`] if either node is unknown.
    pub fn disconnect(&mut self, a: NodeId, b: NodeId) -> Result<bool, NetError> {
        self.require_node(a)?;
        self.require_node(b)?;
        let first = self.set_status(a, b, LinkStatus::Disconnected, LinkStatus::Up);
        let second = self.set_status(b, a, LinkStatus::Disconnected, LinkStatus::Up);
        Ok(first || second)
    }

    /// Reconnects symmetrically disconnected links back to `Up`, leaving
    /// partitioned links alone. Returns whether any link state changed.
    ///
    /// # Errors
    ///
    /// Returns [`NetError::UnknownNode`] if either node is unknown.
    pub fn reconnect(&mut self, a: NodeId, b: NodeId) -> Result<bool, NetError> {
        self.require_node(a)?;
        self.require_node(b)?;
        let first = self.set_status(a, b, LinkStatus::Up, LinkStatus::Disconnected);
        let second = self.set_status(b, a, LinkStatus::Up, LinkStatus::Disconnected);
        Ok(first || second)
    }

    /// Closes network connectivity for one process instance (crash path):
    /// sends from `endpoint` drop with `EndpointDown` until released.
    pub fn isolate(&mut self, endpoint: Endpoint) {
        self.isolated.insert(endpoint);
    }

    /// Reopens connectivity for one process instance.
    /// Returns `false` if it was not isolated.
    pub fn release(&mut self, endpoint: Endpoint) -> bool {
        self.isolated.remove(&endpoint)
    }

    /// Whether the endpoint instance is currently isolated.
    #[must_use]
    pub fn is_isolated(&self, endpoint: Endpoint) -> bool {
        self.isolated.contains(&endpoint)
    }

    /// Sets a directed link status only when it currently holds `expected`;
    /// returns whether anything changed. Absent links are never created.
    fn set_status(
        &mut self,
        from: NodeId,
        to: NodeId,
        next: LinkStatus,
        expected: LinkStatus,
    ) -> bool {
        if let Some(link) = self.links.get_mut(&(from, to))
            && link.status == expected
        {
            link.status = next;
            return true;
        }
        false
    }

    fn require_node(&self, node: NodeId) -> Result<(), NetError> {
        if self.nodes.contains(&node) {
            Ok(())
        } else {
            Err(NetError::UnknownNode { node })
        }
    }

    /// `now + base_latency + size * 1s / bandwidth`, fully checked.
    fn delivery_tick(now: Ticks, config: LinkConfig, size_bytes: usize) -> Result<Ticks, NetError> {
        let base = u64::try_from(config.base_latency().as_micros()).unwrap_or(u64::MAX);
        let size = u64::try_from(size_bytes).unwrap_or(u64::MAX);
        let serial = size
            .checked_mul(1_000_000)
            .ok_or(NetError::TickOverflow)?
            .checked_div(config.bandwidth_bytes_per_sec())
            .unwrap_or(0);
        let at = now
            .as_micros()
            .checked_add(base)
            .ok_or(NetError::TickOverflow)?
            .checked_add(serial)
            .ok_or(NetError::TickOverflow)?;
        Ok(Ticks::from_micros(at))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kivi_core::{EventView, FaultDecision, FaultPolicy, RandomSource};
    use kivi_types::NodeIncarnation;
    use rstest::{fixture, rstest};

    use crate::scheduler::Scheduler;

    const INC: NodeIncarnation = NodeIncarnation::INITIAL;

    fn node(n: u64) -> NodeId {
        NodeId::from_u64(n)
    }

    fn ep(n: u64) -> Endpoint {
        Endpoint::new(node(n), INC)
    }

    fn fast() -> LinkConfig {
        LinkConfig::new(Duration::ZERO, 1_000_000, 16).expect("valid config")
    }

    #[fixture]
    fn pair() -> NetSim {
        let mut net = NetSim::new();
        net.add_node(node(1));
        net.add_node(node(2));
        net.set_link(node(1), node(2), fast()).expect("link 1->2");
        net.set_link(node(2), node(1), fast()).expect("link 2->1");
        net
    }

    fn is_partition_drop(outcome: &NetSendOutcome) -> bool {
        matches!(
            outcome,
            NetSendOutcome::Dropped {
                reason: NetDropReason::Partitioned
            }
        )
    }

    #[rstest]
    #[case(0, 1_000_000, 0, 0)]
    #[case(100_000, 1_000_000, 1_000, 1_000)]
    #[case(0, 500_000, 1_000, 2_000)]
    #[case(50_000, 10_000_000, 10_000, 1_000)]
    fn delivery_tick_math(
        #[case] base_us: u64,
        #[case] bandwidth_bps: u64,
        #[case] size: usize,
        #[case] serial_us: u64,
    ) {
        let mut net = NetSim::new();
        net.add_node(node(1));
        net.add_node(node(2));
        net.set_link(
            node(1),
            node(2),
            LinkConfig::new(Duration::from_micros(base_us), bandwidth_bps, 8).expect("config"),
        )
        .expect("link");
        let now = Ticks::from_micros(1_000_000);
        let outcome = net
            .send(ep(1), node(2), vec![9u8; size], now)
            .expect("send");
        let NetSendOutcome::Scheduled { delivery } = outcome else {
            panic!("symmetric up link must schedule");
        };
        assert_eq!(
            delivery.deliver_at(),
            Ticks::from_micros(1_000_000 + base_us + serial_us)
        );
        assert_eq!(delivery.sent_at(), now);
        assert_eq!(delivery.id(), MsgId::FIRST);
        assert_eq!(delivery.from(), ep(1));
        assert_eq!(delivery.to(), node(2));
        assert_eq!(delivery.payload().len(), size);
    }

    enum Cut {
        Symmetric,
        OneWayAB,
        OneWayBA,
    }

    #[rstest]
    #[case(Cut::Symmetric, true, true)]
    #[case(Cut::OneWayAB, true, false)]
    #[case(Cut::OneWayBA, false, true)]
    fn partitions_block_configured_directions(
        mut pair: NetSim,
        #[case] cut: Cut,
        #[case] ab_down: bool,
        #[case] ba_down: bool,
    ) {
        match cut {
            Cut::Symmetric => {
                assert!(pair.partition(node(1), node(2)).expect("partition"));
            }
            Cut::OneWayAB => {
                assert!(pair.partition_one_way(node(1), node(2)).expect("partition"));
            }
            Cut::OneWayBA => {
                assert!(pair.partition_one_way(node(2), node(1)).expect("partition"));
            }
        }
        let now = Ticks::from_micros(0);
        let ab = pair.send(ep(1), node(2), vec![1], now).expect("send");
        let ba = pair.send(ep(2), node(1), vec![1], now).expect("send");
        assert_eq!(is_partition_drop(&ab), ab_down);
        assert_eq!(is_partition_drop(&ba), ba_down);
        // Healing restores both directions regardless of cut shape.
        pair.heal(node(1), node(2)).expect("heal");
        assert!(matches!(
            pair.send(ep(1), node(2), vec![1], now).expect("send"),
            NetSendOutcome::Scheduled { .. }
        ));
        assert!(matches!(
            pair.send(ep(2), node(1), vec![1], now).expect("send"),
            NetSendOutcome::Scheduled { .. }
        ));
    }

    #[test]
    fn asymmetric_topology_has_no_route_back() {
        let mut net = NetSim::new();
        net.add_node(node(1));
        net.add_node(node(2));
        net.set_link(node(1), node(2), fast()).expect("link");
        let now = Ticks::from_micros(0);
        assert!(matches!(
            net.send(ep(1), node(2), vec![1], now).expect("send"),
            NetSendOutcome::Scheduled { .. }
        ));
        assert_eq!(
            net.send(ep(2), node(1), vec![1], now).expect("send"),
            NetSendOutcome::Dropped {
                reason: NetDropReason::NoRoute
            }
        );
    }

    #[test]
    fn disconnect_and_reconnect_are_independent_of_partitions() {
        let mut net = pair();
        let now = Ticks::from_micros(0);
        assert!(net.disconnect(node(1), node(2)).expect("disconnect"));
        assert_eq!(
            net.send(ep(1), node(2), vec![1], now).expect("send"),
            NetSendOutcome::Dropped {
                reason: NetDropReason::Disconnected
            }
        );
        // Healing does not touch disconnected links.
        assert!(!net.heal(node(1), node(2)).expect("heal"));
        assert!(net.reconnect(node(1), node(2)).expect("reconnect"));
        assert!(matches!(
            net.send(ep(1), node(2), vec![1], now).expect("send"),
            NetSendOutcome::Scheduled { .. }
        ));
        // Reconnecting an already-up link changes nothing.
        assert!(!net.reconnect(node(1), node(2)).expect("reconnect"));
    }

    #[test]
    fn capacity_bounds_in_flight_until_completion() {
        let mut net = NetSim::new();
        net.add_node(node(1));
        net.add_node(node(2));
        net.set_link(
            node(1),
            node(2),
            LinkConfig::new(Duration::ZERO, 1_000_000, 1).expect("config"),
        )
        .expect("link");
        let now = Ticks::from_micros(0);
        assert!(matches!(
            net.send(ep(1), node(2), vec![1], now).expect("send"),
            NetSendOutcome::Scheduled { .. }
        ));
        assert_eq!(net.in_flight(node(1), node(2)), 1);
        assert_eq!(
            net.send(ep(1), node(2), vec![1], now).expect("send"),
            NetSendOutcome::Dropped {
                reason: NetDropReason::Congested
            }
        );
        net.complete_delivery(node(1), node(2));
        assert_eq!(net.in_flight(node(1), node(2)), 0);
        assert!(matches!(
            net.send(ep(1), node(2), vec![1], now).expect("send"),
            NetSendOutcome::Scheduled { .. }
        ));
    }

    #[test]
    fn isolated_endpoints_reject_sends_by_incarnation() {
        let mut net = pair();
        let now = Ticks::from_micros(0);
        let old = Endpoint::new(node(1), NodeIncarnation::from_u64(7));
        let renewed = Endpoint::new(node(1), NodeIncarnation::from_u64(8));
        net.isolate(old);
        assert!(net.is_isolated(old));
        assert_eq!(
            net.send(old, node(2), vec![1], now).expect("send"),
            NetSendOutcome::Dropped {
                reason: NetDropReason::EndpointDown
            }
        );
        // A different incarnation of the same node is unaffected.
        assert!(matches!(
            net.send(renewed, node(2), vec![1], now).expect("send"),
            NetSendOutcome::Scheduled { .. }
        ));
        assert!(net.release(old));
        assert!(!net.release(old));
        assert!(matches!(
            net.send(old, node(2), vec![1], now).expect("send"),
            NetSendOutcome::Scheduled { .. }
        ));
    }

    #[test]
    fn unknown_nodes_and_bad_configs_fail_explicitly() {
        let mut net = NetSim::new();
        net.add_node(node(1));
        let now = Ticks::from_micros(0);
        assert_eq!(
            net.send(ep(1), node(9), vec![1], now),
            Err(NetError::UnknownNode { node: node(9) })
        );
        assert_eq!(
            net.partition(node(1), node(9)),
            Err(NetError::UnknownNode { node: node(9) })
        );
        assert_eq!(
            LinkConfig::new(Duration::ZERO, 0, 1),
            Err(NetError::ZeroBandwidth)
        );
    }

    #[test]
    fn message_identifiers_exhaust_explicitly() {
        let mut net = pair();
        net.next_msg = MsgId::from_u64(u64::MAX);
        assert_eq!(
            net.send(ep(1), node(2), vec![1], Ticks::from_micros(0)),
            Err(NetError::MsgIdExhausted(MsgIdExhausted))
        );
        net.next_msg = MsgId::from_u64(u64::MAX - 1);
        let outcome = net
            .send(ep(1), node(2), vec![1], Ticks::from_micros(0))
            .expect("penultimate fits");
        assert!(matches!(
            outcome,
            NetSendOutcome::Scheduled { ref delivery } if delivery.id() == MsgId::from_u64(u64::MAX - 1)
        ));
        assert_eq!(
            net.send(ep(1), node(2), vec![1], Ticks::from_micros(0)),
            Err(NetError::MsgIdExhausted(MsgIdExhausted))
        );
    }

    #[test]
    fn delivery_tick_overflow_is_explicit() {
        let mut net = NetSim::new();
        net.add_node(node(1));
        net.add_node(node(2));
        net.set_link(
            node(1),
            node(2),
            LinkConfig::new(Duration::from_millis(100), 1_000_000, 4).expect("config"),
        )
        .expect("link");
        let outcome = net.send(ep(1), node(2), vec![1], Ticks::from_micros(u64::MAX - 10));
        assert_eq!(outcome, Err(NetError::TickOverflow));
    }

    /// A policy over typed network context: drop oversized messages while
    /// letting small ones through — the shape future loss models take.
    struct DropOversized {
        max_bytes: usize,
    }

    impl FaultPolicy<NetDeliveryContext> for DropOversized {
        fn decide(
            &mut self,
            _: &EventView,
            ctx: &NetDeliveryContext,
            _: &mut dyn RandomSource,
        ) -> FaultDecision {
            if ctx.size_bytes() > self.max_bytes {
                FaultDecision::Drop
            } else {
                FaultDecision::Allow
            }
        }
    }

    struct ConstRng(u64);

    impl RandomSource for ConstRng {
        fn next_u64(&mut self) -> u64 {
            self.0
        }
    }

    #[test]
    fn typed_context_policy_drives_deliveries() {
        let mut net = pair();
        let now = Ticks::from_micros(1_000);
        let mut scheduler = Scheduler::new(now);
        for size in [10usize, 500] {
            let NetSendOutcome::Scheduled { delivery } = net
                .send(ep(1), node(2), vec![7u8; size], now)
                .expect("send")
            else {
                panic!("up link must schedule");
            };
            scheduler
                .schedule_at(delivery.deliver_at(), delivery)
                .expect("delivery fits");
        }
        let mut policy = DropOversized { max_bytes: 100 };
        let mut rng = ConstRng(0);
        let mut outcomes = Vec::new();
        while let Some(popped) = scheduler.pop_next() {
            let ctx = popped.event().fault_context();
            let decision = policy.decide(&popped.view(), &ctx, &mut rng);
            net.complete_delivery(popped.event().from().node(), popped.event().to());
            outcomes.push((ctx.size_bytes(), decision));
        }
        assert_eq!(
            outcomes,
            vec![(10, FaultDecision::Allow), (500, FaultDecision::Drop),]
        );
    }

    #[test]
    fn identities_display_for_trace_use() {
        assert_eq!(MsgId::from_u64(11).to_string(), "msg11");
        assert_eq!(ep(3).to_string(), "n3@1");
        assert_eq!(LinkStatus::Partitioned.to_string(), "partitioned");
    }
}
