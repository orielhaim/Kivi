//! Deterministic cluster scenario runner: network, storage, and node
//! lifecycle composed over one totally ordered scheduler.
//!
//! [`SimCluster`] owns a [`Scheduler`] of [`ClusterEvent`]s plus the three
//! domain simulators ([`NetSim`], per-node [`VirtualDisk`]s, [`NodeTable`])
//! and deterministic fault policies. Drivers build scenarios from explicit
//! actions — add nodes, connect links, send messages, partition/heal, crash,
//! restart, submit storage work, advance time — and run them with explicit
//! bounds (`run_until_idle(max_events)`, never an unbounded loop).
//!
//! Application logic plugs in through [`AppHandler`]: scheduled app events
//! (`E`, e.g. timers) and delivered network bytes both reach per-node state
//! `S` with an action sink for sends, storage work, and further scheduling.
//! Crashes discard `S` (volatile) while disks survive, exactly like the
//! modeled contract.
//!
//! Every fate is traced as [`ClusterTrace`] history: scheduling, deliveries,
//! policy/topology drops, deferrals, duplications, connectivity changes,
//! storage outcomes, lifecycle transitions, stale-incarnation rejections,
//! and app handling. Identical scenario + seed + policies replay to an equal
//! trace and equal terminal app states.

use core::fmt;
use core::time::Duration;
use std::collections::BTreeMap;

use kivi_core::{EventId, EventView, FaultDecision, FaultPolicy, RandomSource};
use kivi_types::{NodeId, NodeIncarnation, Ticks};

use crate::net::{
    Endpoint, MsgId, NetDelivery, NetDeliveryContext, NetDropReason, NetError, NetSim,
};
use crate::node::{NodeError, NodeState, NodeTable};
use crate::scheduler::{ScheduleError, Scheduler};
use crate::storage::{StorageError, StorageFault, StorageOp, StorageResult, VirtualDisk};

/// A node lifecycle step addressable as a scheduled event (timed crashes and
/// restarts ride the same total order as everything else).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NodeStep {
    /// Crash the node now.
    Crash(NodeId),
    /// Begin restart (mint the next incarnation, not serving yet).
    BeginRestart(NodeId),
    /// Finish restart (serving again under the new incarnation).
    FinishRestart(NodeId),
}

/// One schedulable cluster happening.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ClusterEvent<E> {
    /// A network delivery is due.
    NetDeliver(NetDelivery),
    /// A storage operation is due on one node's disk.
    StorageRun {
        /// Node whose disk serves the operation.
        node: NodeId,
        /// Operation to apply.
        op: StorageOp,
        /// Injected fault for this operation.
        fault: StorageFault,
    },
    /// A lifecycle transition is due.
    NodeStep(NodeStep),
    /// An application event is due on one node.
    App {
        /// Node that must handle the event (dropped if not running).
        target: NodeId,
        /// Opaque application payload (e.g. a timer).
        event: E,
    },
}

/// Machine-readable history of one cluster happening. Categories name what
/// the kernel did — never free-form human logging.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ClusterTrace<E> {
    /// An event entered the scheduler.
    Scheduled {
        /// Assigned scheduler identifier.
        id: EventId,
        /// Scheduled tick.
        at: Ticks,
    },
    /// A message was delivered to the currently running instance.
    Delivered {
        /// Message identity.
        msg: MsgId,
        /// Source endpoint (node + sender incarnation).
        from: Endpoint,
        /// Resolved receiving endpoint (node + current incarnation).
        to: Endpoint,
        /// Send tick.
        sent_at: Ticks,
        /// Delivery tick.
        at: Ticks,
    },
    /// An event was skipped.
    Dropped {
        /// Scheduler identifier of the skipped event, or `None` for
        /// send-time topology drops, which never entered the scheduler.
        id: Option<EventId>,
        /// Tick it would have run at (send tick for topology drops).
        at: Ticks,
        /// Why it was skipped.
        kind: DropKind,
    },
    /// An event was rescheduled for a later tick (fresh scheduler identity).
    Deferred {
        /// Original scheduler identifier.
        id: EventId,
        /// Original tick.
        at: Ticks,
        /// Rescheduled tick.
        until: Ticks,
    },
    /// An extra copy of an event was scheduled at the same tick under a
    /// fresh scheduler identity (fault-injected duplication). For network
    /// deliveries the copy shares the original message identity.
    Duplicated {
        /// Scheduler identifier of the duplicated event.
        event: EventId,
        /// Scheduler identifier of the copy.
        copy: EventId,
    },
    /// Connectivity was cut (symmetric, or one direction when `one_way`).
    Partitioned {
        /// First side.
        a: NodeId,
        /// Second side.
        b: NodeId,
        /// Whether only `a -> b` was cut.
        one_way: bool,
    },
    /// Partitioned links were healed.
    Healed {
        /// First side.
        a: NodeId,
        /// Second side.
        b: NodeId,
    },
    /// Links were administratively disconnected.
    Disconnected {
        /// First side.
        a: NodeId,
        /// Second side.
        b: NodeId,
    },
    /// Disconnected links were reconnected.
    Reconnected {
        /// First side.
        a: NodeId,
        /// Second side.
        b: NodeId,
    },
    /// A storage operation completed.
    StorageOperation {
        /// Node whose disk served it.
        node: NodeId,
        /// Operation applied.
        op: StorageOp,
        /// Outcome produced.
        result: StorageResult,
    },
    /// A storage operation failed (or a fault was injected).
    StorageFailure {
        /// Node whose disk was addressed.
        node: NodeId,
        /// Operation attempted.
        op: StorageOp,
        /// Failure produced.
        error: StorageError,
    },
    /// A node crashed (volatile state discarded, endpoint isolated).
    NodeCrashed {
        /// Crashed node.
        node: NodeId,
        /// Obsolete incarnation.
        incarnation: NodeIncarnation,
    },
    /// A node finished restarting under a new incarnation.
    NodeRestarted {
        /// Restarted node.
        node: NodeId,
        /// New serving incarnation.
        incarnation: NodeIncarnation,
    },
    /// A delivery from an obsolete sender incarnation was rejected. Such
    /// traffic never becomes valid after restart.
    StaleMessageRejected {
        /// Rejected message identity.
        msg: MsgId,
        /// Obsolete sender endpoint.
        from: Endpoint,
        /// Sender's current incarnation.
        current: NodeIncarnation,
    },
    /// An application event ran on a node.
    AppHandled {
        /// Handling node.
        node: NodeId,
        /// Payload handled.
        event: E,
    },
}

/// Why a scheduled event was skipped.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DropKind {
    /// A fault policy dropped it.
    Policy,
    /// The network dropped it at send time (reason recorded then).
    Topology(NetDropReason),
    /// Its target process was not running.
    TargetDown,
}

impl fmt::Display for DropKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Policy => write!(f, "policy"),
            Self::Topology(reason) => write!(f, "topology({reason})"),
            Self::TargetDown => write!(f, "target-down"),
        }
    }
}

/// Actions an application handler may emit. Applied by the driver after the
/// handler returns, so handlers never borrow simulator internals.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ClusterAction<E> {
    /// Send bytes to a node (sender is the handling node, current incarnation).
    NetSend {
        /// Destination node.
        to: NodeId,
        /// Bytes to transport.
        payload: Vec<u8>,
    },
    /// Run a storage operation on the handling node's disk (scheduled now,
    /// uniformly subject to the base fault policy at execution).
    Storage {
        /// Operation to apply.
        op: StorageOp,
        /// Injected fault for this operation.
        fault: StorageFault,
    },
    /// Schedule an app event on a node after a delay.
    AppAfter {
        /// Delay from the current tick.
        delay: Duration,
        /// Node that must handle it.
        target: NodeId,
        /// Payload.
        event: E,
    },
    /// Schedule an app event on a node at an explicit tick.
    AppAt {
        /// Tick to run at (must not precede now).
        at: Ticks,
        /// Node that must handle it.
        target: NodeId,
        /// Payload.
        event: E,
    },
}

/// Application logic inside the simulated cluster.
///
/// `E` is opaque scheduled payload (timers, commands); network receipts
/// arrive separately as bytes so each scenario owns its wire decoding
/// explicitly. `S` is volatile per-node state, discarded on crash. Both
/// handler methods receive the shared simulation RNG for deterministic
/// choices (peer selection, jitter) — never ambient randomness.
pub trait AppHandler<E, S> {
    /// Handles a scheduled app event on `node` with its volatile `state`.
    fn handle_app(
        &mut self,
        node: NodeId,
        event: E,
        state: &mut S,
        actions: &mut Vec<ClusterAction<E>>,
        rng: &mut dyn RandomSource,
    );

    /// Handles network bytes delivered to `node` from `from`.
    fn handle_net(
        &mut self,
        from: Endpoint,
        to: NodeId,
        bytes: &[u8],
        state: &mut S,
        actions: &mut Vec<ClusterAction<E>>,
        rng: &mut dyn RandomSource,
    );
}

/// Deterministic invariant check over cluster state, run after every stepped
/// event. Returns `Err(detail)` on violation.
pub type InvariantFn<E, S> = Box<dyn Fn(&SimCluster<E, S>) -> Result<(), String>>;

/// Context identifying an invariant failure for reproduction.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("invariant {name} failed at {tick} (trace position {trace_pos}): {detail}")]
pub struct InvariantFailure {
    name: String,
    tick: Ticks,
    trace_pos: usize,
    detail: String,
}

impl InvariantFailure {
    /// Returns the violated invariant's registered name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns the tick of the event that broke it.
    #[must_use]
    pub const fn tick(&self) -> Ticks {
        self.tick
    }

    /// Returns the trace length at failure (position of the breaking event).
    #[must_use]
    pub const fn trace_pos(&self) -> usize {
        self.trace_pos
    }

    /// Returns the check's detail message.
    #[must_use]
    pub fn detail(&self) -> &str {
        &self.detail
    }
}

/// Cluster driver failure: misuse, exhaustion, or a broken invariant.
/// Reachability and operation outcomes are traced, never errors — only
/// scenario misuse, checked-arithmetic exhaustion, and invariant violations
/// fail a run.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum ClusterError {
    /// Network misuse or exhaustion.
    #[error("network: {0}")]
    Net(#[from] NetError),
    /// Lifecycle misuse or exhaustion.
    #[error("node: {0}")]
    Node(#[from] NodeError),
    /// Scheduler misuse or exhaustion.
    #[error("schedule: {0}")]
    Schedule(#[from] ScheduleError),
    /// An invariant check failed (run aborted with reproduction context).
    #[error("{0}")]
    Invariant(#[source] InvariantFailure),
    /// The internal processed-event counter cannot advance (practically
    /// unreachable; explicit rather than wrapping).
    #[error("processed-event counter exhausted")]
    TooManyEvents,
    /// A scheduled restart completed but no restart-state factory was set.
    #[error("no restart-state factory set for node {node}")]
    MissingRestartFactory {
        /// Node that finished restarting.
        node: NodeId,
    },
}

/// Simulation format/version tag recorded in every scenario result, so
/// future format changes are detectable rather than silently misread.
pub const SIM_FORMAT_VERSION: u16 = 1;

/// Bounded-run statistics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RunStats {
    /// Total events processed across all runs so far.
    pub processed: u64,
    /// Whether the scheduler drained (true) or the bound stopped the run.
    pub idle: bool,
    /// Virtual tick after the run.
    pub final_tick: Ticks,
}

/// Complete reproducible outcome of a scenario: format tag, seed, counters,
/// terminal app states, and full trace. Equal inputs replay to an equal
/// result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScenarioResult<E, S> {
    /// Simulation format version.
    pub format: u16,
    /// Seed driving all simulation randomness.
    pub seed: u64,
    /// Events processed in total.
    pub events_processed: u64,
    /// Final virtual tick.
    pub final_tick: Ticks,
    /// Terminal volatile app states by node (crashed nodes absent).
    pub apps: BTreeMap<NodeId, S>,
    /// Full machine-readable history in run order.
    pub trace: Vec<ClusterTrace<E>>,
}

/// Deterministic cluster scenario runner.
pub struct SimCluster<E, S> {
    scheduler: Scheduler<ClusterEvent<E>>,
    net: NetSim,
    disks: BTreeMap<NodeId, VirtualDisk>,
    nodes: NodeTable,
    states: BTreeMap<NodeId, S>,
    net_policy: Box<dyn FaultPolicy<NetDeliveryContext>>,
    base_policy: Box<dyn FaultPolicy<()>>,
    trace: Vec<ClusterTrace<E>>,
    invariants: Vec<(String, InvariantFn<E, S>)>,
    restart_factory: Option<Box<dyn Fn(NodeId) -> S>>,
    seed: u64,
    processed: u64,
}

impl<E, S> SimCluster<E, S> {
    /// Creates an empty cluster: no nodes, both fault policies allow
    /// everything, clock at `origin`. The `seed` is recorded in results;
    /// simulation randomness stays owned by the driver and is threaded
    /// through every run call explicitly.
    #[must_use]
    pub fn new(seed: u64, origin: Ticks) -> Self {
        Self {
            scheduler: Scheduler::new(origin),
            net: NetSim::new(),
            disks: BTreeMap::new(),
            nodes: NodeTable::new(),
            states: BTreeMap::new(),
            net_policy: Box::new(crate::faults::AllowAll),
            base_policy: Box::new(crate::faults::AllowAll),
            trace: Vec::new(),
            invariants: Vec::new(),
            restart_factory: None,
            seed,
            processed: 0,
        }
    }

    /// Replaces the network fault policy (drop/delay/duplicate over typed
    /// delivery context).
    pub fn set_net_policy(&mut self, policy: impl FaultPolicy<NetDeliveryContext> + 'static) {
        self.net_policy = Box::new(policy);
    }

    /// Replaces the base fault policy (app/storage/timer event fate).
    pub fn set_base_policy(&mut self, policy: impl FaultPolicy<()> + 'static) {
        self.base_policy = Box::new(policy);
    }

    /// Sets the factory producing fresh volatile state for scheduled
    /// restart completions. Immediate `finish_restart` calls take their
    /// state directly and never consult the factory.
    pub fn set_restart_factory(&mut self, factory: impl Fn(NodeId) -> S + 'static) {
        self.restart_factory = Some(Box::new(factory));
    }

    /// Registers a node with a fresh disk of `disk_capacity` bytes and
    /// initial volatile app `state`.
    ///
    /// # Errors
    ///
    /// Returns [`NodeError::DuplicateNode`] if already registered.
    pub fn add_node(
        &mut self,
        node: NodeId,
        disk_capacity: u64,
        state: S,
    ) -> Result<(), ClusterError> {
        self.nodes.add_node(node)?;
        self.net.add_node(node);
        self.disks.insert(node, VirtualDisk::new(disk_capacity));
        self.states.insert(node, state);
        Ok(())
    }

    /// Connects `a <-> b` symmetrically with one config both ways.
    ///
    /// # Errors
    ///
    /// Returns [`NetError`] on unknown nodes or zero bandwidth.
    pub fn connect(
        &mut self,
        a: NodeId,
        b: NodeId,
        config: crate::net::LinkConfig,
    ) -> Result<(), ClusterError> {
        self.net.set_link(a, b, config)?;
        self.net.set_link(b, a, config)?;
        Ok(())
    }

    /// Registers a deterministic invariant check run after every stepped event.
    pub fn add_invariant(
        &mut self,
        name: &str,
        check: impl Fn(&SimCluster<E, S>) -> Result<(), String> + 'static,
    ) {
        self.invariants.push((name.to_owned(), Box::new(check)));
    }

    /// Returns the scenario seed.
    #[must_use]
    pub const fn seed(&self) -> u64 {
        self.seed
    }

    /// Returns the current virtual tick.
    #[must_use]
    pub fn now(&self) -> Ticks {
        self.scheduler.now()
    }

    /// Returns total events processed so far.
    #[must_use]
    pub const fn processed(&self) -> u64 {
        self.processed
    }

    /// Returns the full trace in run order.
    #[must_use]
    pub fn trace(&self) -> &[ClusterTrace<E>] {
        &self.trace
    }

    /// Returns the node lifecycle table.
    #[must_use]
    pub fn nodes(&self) -> &NodeTable {
        &self.nodes
    }

    /// Returns the network simulator.
    #[must_use]
    pub fn net(&self) -> &NetSim {
        &self.net
    }

    /// Returns all disks by node.
    #[must_use]
    pub fn disks(&self) -> &BTreeMap<NodeId, VirtualDisk> {
        &self.disks
    }

    /// Returns one disk for inspection, if the node exists.
    #[must_use]
    pub fn disk(&self, node: NodeId) -> Option<&VirtualDisk> {
        self.disks.get(&node)
    }

    /// Returns volatile app states by node (crashed nodes absent).
    #[must_use]
    pub fn app_states(&self) -> &BTreeMap<NodeId, S> {
        &self.states
    }

    /// Sends bytes from `from` to `to` at the current tick (scenario action).
    ///
    /// Reachability outcomes are traced; only misuse/exhaustion fail.
    ///
    /// # Errors
    ///
    /// Returns [`ClusterError`] on unknown nodes, tick overflow, or
    /// identifier exhaustion.
    pub fn send(
        &mut self,
        from: NodeId,
        to: NodeId,
        payload: Vec<u8>,
    ) -> Result<Option<MsgId>, ClusterError> {
        let now = self.scheduler.now();
        let endpoint = self
            .nodes
            .current_endpoint(from)
            .ok_or(NodeError::UnknownNode { node: from })?;
        match self.net.send(endpoint, to, payload, now)? {
            crate::net::NetSendOutcome::Scheduled { delivery } => {
                let msg = delivery.id();
                let at = delivery.deliver_at();
                let id = self
                    .scheduler
                    .schedule_at(at, ClusterEvent::NetDeliver(delivery))?;
                self.trace.push(ClusterTrace::Scheduled { id, at });
                Ok(Some(msg))
            }
            crate::net::NetSendOutcome::Dropped { reason } => {
                self.trace.push(ClusterTrace::Dropped {
                    id: None,
                    at: now,
                    kind: DropKind::Topology(reason),
                });
                Ok(None)
            }
        }
    }

    /// Schedules an app event (scenario action).
    ///
    /// # Errors
    ///
    /// Returns [`ClusterError`] on unknown targets or scheduler exhaustion.
    pub fn schedule_app(
        &mut self,
        at: Ticks,
        target: NodeId,
        event: E,
    ) -> Result<EventId, ClusterError> {
        self.require_node(target)?;
        let id = self
            .scheduler
            .schedule_at(at, ClusterEvent::App { target, event })?;
        self.trace.push(ClusterTrace::Scheduled { id, at });
        Ok(id)
    }

    /// Submits storage work on one node's disk at the current tick: the
    /// operation is scheduled (uniformly subject to the base policy at
    /// execution), never executed inline.
    ///
    /// # Errors
    ///
    /// Returns [`ClusterError`] on unknown nodes or scheduler exhaustion.
    pub fn submit_storage(
        &mut self,
        node: NodeId,
        op: StorageOp,
        fault: StorageFault,
    ) -> Result<EventId, ClusterError> {
        self.require_node(node)?;
        let at = self.scheduler.now();
        let id = self
            .scheduler
            .schedule_at(at, ClusterEvent::StorageRun { node, op, fault })?;
        self.trace.push(ClusterTrace::Scheduled { id, at });
        Ok(id)
    }

    /// Schedules a lifecycle step at a tick (timed crash/restart scenarios).
    ///
    /// # Errors
    ///
    /// Returns [`ClusterError`] on unknown nodes or scheduler exhaustion.
    pub fn schedule_node_step(
        &mut self,
        at: Ticks,
        step: NodeStep,
    ) -> Result<EventId, ClusterError> {
        let node = match step {
            NodeStep::Crash(node)
            | NodeStep::BeginRestart(node)
            | NodeStep::FinishRestart(node) => node,
        };
        self.require_node(node)?;
        let id = self
            .scheduler
            .schedule_at(at, ClusterEvent::NodeStep(step))?;
        self.trace.push(ClusterTrace::Scheduled { id, at });
        Ok(id)
    }

    /// Applies a partition scenario action immediately (traced when changed).
    ///
    /// # Errors
    ///
    /// Returns [`ClusterError`] on unknown nodes.
    pub fn partition(&mut self, a: NodeId, b: NodeId) -> Result<(), ClusterError> {
        if self.net.partition(a, b)? {
            self.trace.push(ClusterTrace::Partitioned {
                a,
                b,
                one_way: false,
            });
        }
        Ok(())
    }

    /// Applies a one-way partition `from -> to` immediately (traced when changed).
    ///
    /// # Errors
    ///
    /// Returns [`ClusterError`] on unknown nodes.
    pub fn partition_one_way(&mut self, from: NodeId, to: NodeId) -> Result<(), ClusterError> {
        if self.net.partition_one_way(from, to)? {
            self.trace.push(ClusterTrace::Partitioned {
                a: from,
                b: to,
                one_way: true,
            });
        }
        Ok(())
    }

    /// Heals partitioned links immediately (traced when changed).
    ///
    /// # Errors
    ///
    /// Returns [`ClusterError`] on unknown nodes.
    pub fn heal(&mut self, a: NodeId, b: NodeId) -> Result<(), ClusterError> {
        if self.net.heal(a, b)? {
            self.trace.push(ClusterTrace::Healed { a, b });
        }
        Ok(())
    }

    /// Disconnects links immediately (traced when changed).
    ///
    /// # Errors
    ///
    /// Returns [`ClusterError`] on unknown nodes.
    pub fn disconnect(&mut self, a: NodeId, b: NodeId) -> Result<(), ClusterError> {
        if self.net.disconnect(a, b)? {
            self.trace.push(ClusterTrace::Disconnected { a, b });
        }
        Ok(())
    }

    /// Reconnects disconnected links immediately (traced when changed).
    ///
    /// # Errors
    ///
    /// Returns [`ClusterError`] on unknown nodes.
    pub fn reconnect(&mut self, a: NodeId, b: NodeId) -> Result<(), ClusterError> {
        if self.net.reconnect(a, b)? {
            self.trace.push(ClusterTrace::Reconnected { a, b });
        }
        Ok(())
    }

    /// Crashes a node immediately: lifecycle to `Crashed`, endpoint isolated
    /// from the network, volatile app state discarded. Disks survive.
    /// Future events addressed to the node drop lazily at execution.
    ///
    /// # Errors
    ///
    /// Returns [`ClusterError`] for unknown nodes or non-running states.
    pub fn crash_node(&mut self, node: NodeId) -> Result<(), ClusterError> {
        let endpoint = self.nodes.crash(node)?;
        self.net.isolate(endpoint);
        self.states.remove(&node);
        self.trace.push(ClusterTrace::NodeCrashed {
            node,
            incarnation: endpoint.incarnation(),
        });
        Ok(())
    }

    /// Begins a restart (new incarnation minted, not serving yet).
    ///
    /// # Errors
    ///
    /// Returns [`ClusterError`] for unknown nodes, non-crashed states, or an
    /// exhausted incarnation counter.
    pub fn begin_restart(&mut self, node: NodeId) -> Result<Endpoint, ClusterError> {
        Ok(self.nodes.begin_restart(node)?)
    }

    /// Finishes a restart with fresh volatile state: the node serves again
    /// under its new incarnation.
    ///
    /// # Errors
    ///
    /// Returns [`ClusterError`] for unknown nodes or non-restarting states.
    pub fn finish_restart(&mut self, node: NodeId, state: S) -> Result<Endpoint, ClusterError> {
        let endpoint = self.nodes.finish_restart(node)?;
        self.states.insert(node, state);
        self.trace.push(ClusterTrace::NodeRestarted {
            node,
            incarnation: endpoint.incarnation(),
        });
        Ok(endpoint)
    }

    /// Runs one scheduled event (or reports idle), then all invariants.
    ///
    /// # Errors
    ///
    /// Returns [`ClusterError`] on driver misuse/exhaustion, or the first
    /// invariant failure with reproduction context.
    pub fn run_next<H: AppHandler<E, S>>(
        &mut self,
        handler: &mut H,
        rng: &mut dyn RandomSource,
    ) -> Result<StepOutcome, ClusterError>
    where
        E: Clone,
    {
        let Some(scheduled) = self.scheduler.pop_next() else {
            return Ok(StepOutcome::Idle);
        };
        self.processed = self
            .processed
            .checked_add(1)
            .ok_or(ClusterError::TooManyEvents)?;
        let id = scheduled.id();
        let at = scheduled.at();
        match scheduled.into_event() {
            ClusterEvent::NetDeliver(delivery) => self.deliver_net(id, at, delivery, handler, rng),
            ClusterEvent::StorageRun { node, op, fault } => {
                self.execute_storage(id, at, node, op, fault, rng)
            }
            ClusterEvent::NodeStep(step) => self.execute_node_step(step),
            ClusterEvent::App { target, event } => {
                self.execute_app(id, at, target, event, handler, rng)
            }
        }?;
        self.check_invariants()?;
        Ok(StepOutcome::Stepped)
    }

    /// Runs up to `max_events` events, stopping early when idle.
    ///
    /// # Errors
    ///
    /// Returns [`ClusterError`] on the first driver failure or invariant
    /// violation; reaching the bound without draining is reported in
    /// [`RunStats::idle`], never an infinite loop.
    pub fn run_n<H: AppHandler<E, S>>(
        &mut self,
        max_events: usize,
        handler: &mut H,
        rng: &mut dyn RandomSource,
    ) -> Result<RunStats, ClusterError>
    where
        E: Clone,
    {
        for _ in 0..max_events {
            if matches!(self.run_next(handler, rng)?, StepOutcome::Idle) {
                return Ok(self.stats(true));
            }
        }
        Ok(self.stats(self.scheduler.is_empty()))
    }

    /// Runs until the scheduler drains or `max_events` events processed.
    ///
    /// # Errors
    ///
    /// Same failure modes as [`run_n`](Self::run_n).
    pub fn run_until_idle<H: AppHandler<E, S>>(
        &mut self,
        max_events: usize,
        handler: &mut H,
        rng: &mut dyn RandomSource,
    ) -> Result<RunStats, ClusterError>
    where
        E: Clone,
    {
        self.run_n(max_events, handler, rng)
    }

    /// Runs events scheduled at or before `tick` (plus `max_events` bound).
    ///
    /// # Errors
    ///
    /// Same failure modes as [`run_n`](Self::run_n).
    pub fn run_until_tick<H: AppHandler<E, S>>(
        &mut self,
        tick: Ticks,
        max_events: usize,
        handler: &mut H,
        rng: &mut dyn RandomSource,
    ) -> Result<RunStats, ClusterError>
    where
        E: Clone,
    {
        let mut ran = 0usize;
        while ran < max_events {
            let Some((_, at)) = self.scheduler.peek() else {
                return Ok(self.stats(true));
            };
            if at > tick {
                return Ok(self.stats(false));
            }
            if matches!(self.run_next(handler, rng)?, StepOutcome::Idle) {
                return Ok(self.stats(true));
            }
            ran += 1;
        }
        Ok(self.stats(self.scheduler.is_empty()))
    }

    /// Snapshots the complete reproducible outcome of the scenario so far.
    #[must_use]
    pub fn snapshot(&self) -> ScenarioResult<E, S>
    where
        E: Clone,
        S: Clone,
    {
        ScenarioResult {
            format: SIM_FORMAT_VERSION,
            seed: self.seed,
            events_processed: self.processed,
            final_tick: self.scheduler.now(),
            apps: self.states.clone(),
            trace: self.trace.clone(),
        }
    }

    fn stats(&self, idle: bool) -> RunStats {
        RunStats {
            processed: self.processed,
            idle,
            final_tick: self.scheduler.now(),
        }
    }

    fn require_node(&self, node: NodeId) -> Result<(), ClusterError> {
        if self.nodes.get(node).is_some() {
            Ok(())
        } else {
            Err(NodeError::UnknownNode { node }.into())
        }
    }

    fn check_invariants(&self) -> Result<(), ClusterError> {
        for (name, check) in &self.invariants {
            if let Err(detail) = check(self) {
                return Err(ClusterError::Invariant(InvariantFailure {
                    name: name.clone(),
                    tick: self.scheduler.now(),
                    trace_pos: self.trace.len(),
                    detail,
                }));
            }
        }
        Ok(())
    }

    fn deliver_net<H: AppHandler<E, S>>(
        &mut self,
        id: EventId,
        at: Ticks,
        delivery: NetDelivery,
        handler: &mut H,
        rng: &mut dyn RandomSource,
    ) -> Result<(), ClusterError> {
        let from = delivery.from();
        // Sender incarnation must still be current: obsolete senders never
        // become valid after restart. A merely crashed (not restarted) sender
        // keeps its incarnation, so its in-flight messages still deliver.
        match self.nodes.get(from.node()) {
            None => {
                self.trace.push(ClusterTrace::Dropped {
                    id: Some(id),
                    at,
                    kind: DropKind::TargetDown,
                });
                self.net.complete_delivery(from.node(), delivery.to());
                return Ok(());
            }
            Some(info) if info.incarnation() != from.incarnation() => {
                self.trace.push(ClusterTrace::StaleMessageRejected {
                    msg: delivery.id(),
                    from,
                    current: info.incarnation(),
                });
                self.net.complete_delivery(from.node(), delivery.to());
                return Ok(());
            }
            Some(_) => {}
        }
        // Receiver must be running now.
        let to_endpoint = match self.nodes.get(delivery.to()) {
            Some(info) if info.state() == NodeState::Running => info.endpoint(),
            _ => {
                self.trace.push(ClusterTrace::Dropped {
                    id: Some(id),
                    at,
                    kind: DropKind::TargetDown,
                });
                self.net.complete_delivery(from.node(), delivery.to());
                return Ok(());
            }
        };
        // Slot discipline: the popped delivery holds exactly one capacity
        // slot, reported exactly once below. Copies and deferrals take a
        // fresh slot first (congestion drops the copy, not the original
        // decision). Duplicate delivers the original AND schedules an
        // identical copy (same message identity): the policy is consulted
        // exactly once per event; receivers observe the message twice.
        let ctx = delivery.fault_context();
        let decision = self.net_policy.decide(&EventView::new(id, at), &ctx, rng);
        let duplicate = matches!(decision, FaultDecision::Duplicate);
        match decision {
            FaultDecision::Allow | FaultDecision::Duplicate => {
                self.net.complete_delivery(from.node(), delivery.to());
                let payload = delivery.payload().to_vec();
                self.trace.push(ClusterTrace::Delivered {
                    msg: delivery.id(),
                    from,
                    to: to_endpoint,
                    sent_at: delivery.sent_at(),
                    at,
                });
                if let Some(state) = self.states.get_mut(&delivery.to()) {
                    let mut actions = Vec::new();
                    handler.handle_net(from, delivery.to(), &payload, state, &mut actions, rng);
                    self.drain_actions(delivery.to(), actions)?;
                }
                if duplicate {
                    if self.net.duplicate_slot(from.node(), delivery.to()) {
                        let copy = self
                            .scheduler
                            .schedule_at(at, ClusterEvent::NetDeliver(delivery))?;
                        self.trace
                            .push(ClusterTrace::Duplicated { event: id, copy });
                        self.trace.push(ClusterTrace::Scheduled { id: copy, at });
                    } else {
                        self.trace.push(ClusterTrace::Dropped {
                            id: Some(id),
                            at,
                            kind: DropKind::Topology(NetDropReason::Congested),
                        });
                    }
                }
            }
            FaultDecision::Drop => {
                self.net.complete_delivery(from.node(), delivery.to());
                self.trace.push(ClusterTrace::Dropped {
                    id: Some(id),
                    at,
                    kind: DropKind::Policy,
                });
            }
            FaultDecision::Defer { until } => {
                self.net.complete_delivery(from.node(), delivery.to());
                if self.net.duplicate_slot(from.node(), delivery.to()) {
                    let event = ClusterEvent::NetDeliver(delivery);
                    let copy = self.scheduler.schedule_at(until, event)?;
                    self.trace.push(ClusterTrace::Deferred { id, at, until });
                    self.trace.push(ClusterTrace::Scheduled {
                        id: copy,
                        at: until,
                    });
                } else {
                    self.trace.push(ClusterTrace::Dropped {
                        id: Some(id),
                        at,
                        kind: DropKind::Topology(NetDropReason::Congested),
                    });
                }
            }
        }
        Ok(())
    }

    fn execute_storage(
        &mut self,
        id: EventId,
        at: Ticks,
        node: NodeId,
        op: StorageOp,
        fault: StorageFault,
        rng: &mut dyn RandomSource,
    ) -> Result<(), ClusterError> {
        let running = self
            .nodes
            .get(node)
            .is_some_and(|info| info.state() == NodeState::Running);
        if !running {
            self.trace.push(ClusterTrace::Dropped {
                id: Some(id),
                at,
                kind: DropKind::TargetDown,
            });
            return Ok(());
        }
        // Storage completions share the base (domain-neutral) policy and the
        // run RNG, so faulted storage replays exactly like faulted delivery.
        let decision = self.base_policy.decide(&EventView::new(id, at), &(), rng);
        self.apply_storage_decision(id, at, node, op, fault, decision)
    }

    fn apply_storage_decision(
        &mut self,
        id: EventId,
        at: Ticks,
        node: NodeId,
        op: StorageOp,
        fault: StorageFault,
        decision: FaultDecision,
    ) -> Result<(), ClusterError> {
        match decision {
            FaultDecision::Allow | FaultDecision::Duplicate => {
                // Cloned before the outcome trace moves the originals.
                let replay = matches!(decision, FaultDecision::Duplicate)
                    .then(|| (op.clone(), fault.clone()));
                let disk = self
                    .disks
                    .get_mut(&node)
                    .ok_or(NodeError::UnknownNode { node })?;
                match disk.apply(&op, &fault) {
                    Ok(result) => {
                        self.trace
                            .push(ClusterTrace::StorageOperation { node, op, result });
                    }
                    Err(error) => {
                        self.trace
                            .push(ClusterTrace::StorageFailure { node, op, error });
                    }
                }
                if let Some((op_copy, fault_copy)) = replay {
                    let event = ClusterEvent::StorageRun {
                        node,
                        op: op_copy,
                        fault: fault_copy,
                    };
                    let copy = self.scheduler.schedule_at(at, event)?;
                    self.trace
                        .push(ClusterTrace::Duplicated { event: id, copy });
                    self.trace.push(ClusterTrace::Scheduled { id: copy, at });
                }
            }
            FaultDecision::Drop => {
                self.trace.push(ClusterTrace::Dropped {
                    id: Some(id),
                    at,
                    kind: DropKind::Policy,
                });
            }
            FaultDecision::Defer { until } => {
                let event = ClusterEvent::StorageRun { node, op, fault };
                let copy = self.scheduler.schedule_at(until, event)?;
                self.trace.push(ClusterTrace::Deferred { id, at, until });
                self.trace.push(ClusterTrace::Scheduled {
                    id: copy,
                    at: until,
                });
            }
        }
        Ok(())
    }

    fn execute_node_step(&mut self, step: NodeStep) -> Result<(), ClusterError> {
        match step {
            NodeStep::Crash(node) => self.crash_node(node),
            NodeStep::BeginRestart(node) => {
                self.begin_restart(node)?;
                Ok(())
            }
            NodeStep::FinishRestart(node) => {
                let factory = self
                    .restart_factory
                    .as_ref()
                    .ok_or(ClusterError::MissingRestartFactory { node })?;
                let state = factory(node);
                self.finish_restart(node, state)?;
                Ok(())
            }
        }
    }

    fn execute_app<H: AppHandler<E, S>>(
        &mut self,
        id: EventId,
        at: Ticks,
        target: NodeId,
        event: E,
        handler: &mut H,
        rng: &mut dyn RandomSource,
    ) -> Result<(), ClusterError>
    where
        E: Clone,
    {
        let running = self
            .nodes
            .get(target)
            .is_some_and(|info| info.state() == NodeState::Running);
        if !running {
            self.trace.push(ClusterTrace::Dropped {
                id: Some(id),
                at,
                kind: DropKind::TargetDown,
            });
            return Ok(());
        }
        let decision = self.base_policy.decide(&EventView::new(id, at), &(), rng);
        let duplicate = matches!(decision, FaultDecision::Duplicate);
        match decision {
            FaultDecision::Allow | FaultDecision::Duplicate => {
                // Cloned before the handler moves the original.
                let replay = duplicate.then(|| event.clone());
                self.trace.push(ClusterTrace::AppHandled {
                    node: target,
                    event: event.clone(),
                });
                if let Some(state) = self.states.get_mut(&target) {
                    let mut actions = Vec::new();
                    handler.handle_app(target, event, state, &mut actions, rng);
                    self.drain_actions(target, actions)?;
                }
                if let Some(copy_event) = replay {
                    let copy = self.scheduler.schedule_at(
                        at,
                        ClusterEvent::App {
                            target,
                            event: copy_event,
                        },
                    )?;
                    self.trace
                        .push(ClusterTrace::Duplicated { event: id, copy });
                    self.trace.push(ClusterTrace::Scheduled { id: copy, at });
                }
            }
            FaultDecision::Drop => {
                self.trace.push(ClusterTrace::Dropped {
                    id: Some(id),
                    at,
                    kind: DropKind::Policy,
                });
            }
            FaultDecision::Defer { until } => {
                let copy = self.scheduler.schedule_at(
                    until,
                    ClusterEvent::App {
                        target,
                        event: event.clone(),
                    },
                )?;
                self.trace.push(ClusterTrace::Deferred { id, at, until });
                self.trace.push(ClusterTrace::Scheduled {
                    id: copy,
                    at: until,
                });
            }
        }
        Ok(())
    }

    /// Applies post-handler actions: sends (traced), storage submissions
    /// (scheduled), and app scheduling.
    fn drain_actions(
        &mut self,
        from: NodeId,
        actions: Vec<ClusterAction<E>>,
    ) -> Result<(), ClusterError> {
        for action in actions {
            match action {
                ClusterAction::NetSend { to, payload } => {
                    self.send(from, to, payload)?;
                }
                ClusterAction::Storage { op, fault } => {
                    self.submit_storage(from, op, fault)?;
                }
                ClusterAction::AppAfter {
                    delay,
                    target,
                    event,
                } => {
                    self.require_node(target)?;
                    let now = self.scheduler.now();
                    let delta = u64::try_from(delay.as_micros()).unwrap_or(u64::MAX);
                    let at = now
                        .as_micros()
                        .checked_add(delta)
                        .ok_or(ScheduleError::TickOverflow)?;
                    let id = self
                        .scheduler
                        .schedule_at(Ticks::from_micros(at), ClusterEvent::App { target, event })?;
                    self.trace.push(ClusterTrace::Scheduled {
                        id,
                        at: Ticks::from_micros(at),
                    });
                }
                ClusterAction::AppAt { at, target, event } => {
                    self.require_node(target)?;
                    let id = self
                        .scheduler
                        .schedule_at(at, ClusterEvent::App { target, event })?;
                    self.trace.push(ClusterTrace::Scheduled { id, at });
                }
            }
        }
        Ok(())
    }
}

impl<E, S> fmt::Debug for SimCluster<E, S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SimCluster")
            .field("seed", &self.seed)
            .field("now", &self.scheduler.now())
            .field("processed", &self.processed)
            .field("queued", &self.scheduler.len())
            .field("trace_len", &self.trace.len())
            .finish_non_exhaustive()
    }
}

/// Outcome of one driver step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StepOutcome {
    /// The scheduler was empty; nothing ran.
    Idle,
    /// One event ran (plus invariant checks).
    Stepped,
}
