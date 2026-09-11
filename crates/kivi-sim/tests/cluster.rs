//! Integrated cluster scenarios: mechanics, combined network+storage state
//! machines, and the multi-seed deterministic cluster regression harness.

use core::num::NonZeroU64;
use core::time::Duration;

use kivi_core::{EventView, FaultPolicy, RandomSource};
use kivi_sim::{
    AppHandler, ClusterAction, ClusterError, ClusterTrace, DropEveryNth, Endpoint, LinkConfig,
    MsgId, NetError, NodeStep, SimCluster, SimRng, StepOutcome, StorageFault, StorageOp,
};
use kivi_types::{NodeId, NodeIncarnation, Ticks};
use rstest::{fixture, rstest};

fn node(n: u64) -> NodeId {
    NodeId::from_u64(n)
}

fn link() -> LinkConfig {
    LinkConfig::new(Duration::from_millis(5), 1_000_000, 64).expect("valid link")
}

// ---------------------------------------------------------------------------
// Shared tiny app: timers count, network bytes are recorded.
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct TinyState {
    received: Vec<(u64, Vec<u8>)>,
    timer_fires: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum TinyEv {
    Tick,
}

struct TinyApp;

impl AppHandler<TinyEv, TinyState> for TinyApp {
    fn handle_app(
        &mut self,
        _node: NodeId,
        _event: TinyEv,
        state: &mut TinyState,
        _actions: &mut Vec<ClusterAction<TinyEv>>,
        _rng: &mut dyn RandomSource,
    ) {
        state.timer_fires += 1;
    }

    fn handle_net(
        &mut self,
        from: Endpoint,
        _to: NodeId,
        bytes: &[u8],
        state: &mut TinyState,
        _actions: &mut Vec<ClusterAction<TinyEv>>,
        _rng: &mut dyn RandomSource,
    ) {
        state.received.push((from.node().as_u64(), bytes.to_vec()));
    }
}

#[fixture]
fn two_nodes() -> SimCluster<TinyEv, TinyState> {
    let mut cluster = SimCluster::new(1, Ticks::from_micros(0));
    cluster
        .add_node(node(1), 1 << 20, TinyState::default())
        .expect("node 1");
    cluster
        .add_node(node(2), 1 << 20, TinyState::default())
        .expect("node 2");
    cluster.connect(node(1), node(2), link()).expect("mesh");
    cluster
}

#[test]
fn send_delivers_with_full_metadata() {
    let mut cluster = two_nodes();
    let mut app = TinyApp;
    let mut rng = SimRng::seed_from_u64(1);
    let msg = cluster
        .send(node(1), node(2), b"hello".to_vec())
        .expect("send")
        .expect("scheduled");
    assert_eq!(msg, MsgId::from_u64(0));
    cluster
        .schedule_app(Ticks::from_micros(0), node(1), TinyEv::Tick)
        .expect("timer");
    let stats = cluster
        .run_until_idle(100, &mut app, &mut rng)
        .expect("run");
    assert!(stats.idle);
    assert_eq!(
        cluster.app_states().get(&node(1)).expect("a").timer_fires,
        1
    );
    let apps = cluster.app_states();
    assert_eq!(
        apps.get(&node(2)).expect("b").received,
        vec![(1, b"hello".to_vec())]
    );
    assert!(apps.get(&node(1)).expect("a").received.is_empty());
    let delivered = cluster
        .trace()
        .iter()
        .find_map(|entry| match entry {
            ClusterTrace::Delivered {
                msg,
                from,
                to,
                sent_at,
                at,
            } => Some((*msg, *from, *to, *sent_at, *at)),
            _ => None,
        })
        .expect("delivery traced");
    assert_eq!(delivered.0, msg);
    assert_eq!(delivered.1.node(), node(1));
    assert_eq!(delivered.2.node(), node(2));
    assert_eq!(
        delivered.2.incarnation(),
        NodeIncarnation::INITIAL,
        "receiver incarnation resolved at delivery"
    );
    assert!(delivered.3 <= delivered.4);
    assert_eq!(cluster.now(), delivered.4);
}

#[test]
fn stale_sender_incarnation_is_rejected_after_restart() {
    // The critical scenario: A sends while at incarnation N, crashes, and
    // restarts as N+1 before the in-flight message arrives. Arrival must
    // reject the message as stale — it can never become valid.
    let mut cluster = two_nodes();
    let mut app = TinyApp;
    let mut rng = SimRng::seed_from_u64(2);
    let old_ep = cluster.nodes().current_endpoint(node(1)).expect("a");
    cluster
        .send(node(1), node(2), b"delayed".to_vec())
        .expect("send");
    cluster.crash_node(node(1)).expect("crash");
    cluster.begin_restart(node(1)).expect("begin");
    let fresh = cluster
        .finish_restart(node(1), TinyState::default())
        .expect("restart");
    assert_ne!(old_ep.incarnation(), fresh.incarnation());
    cluster
        .run_until_idle(100, &mut app, &mut rng)
        .expect("run");
    // The old message arrived but was rejected as stale, never delivered.
    let stale = cluster
        .trace()
        .iter()
        .find_map(|entry| match entry {
            ClusterTrace::StaleMessageRejected { msg, from, current } => {
                Some((*msg, *from, *current))
            }
            _ => None,
        })
        .expect("stale rejection traced");
    assert_eq!(stale.1, old_ep);
    assert_eq!(stale.2, fresh.incarnation());
    assert!(
        !cluster
            .trace()
            .iter()
            .any(|entry| matches!(entry, ClusterTrace::Delivered { .. })),
        "stale message must never deliver"
    );
    assert!(
        cluster
            .app_states()
            .get(&node(2))
            .expect("b")
            .received
            .is_empty()
    );
}

#[test]
fn crash_discards_volatile_state_but_keeps_durable_disk() {
    let mut cluster = two_nodes();
    let mut app = TinyApp;
    let mut rng = SimRng::seed_from_u64(3);
    for op in [
        StorageOp::Create {
            path: "state".to_owned(),
        },
        StorageOp::Write {
            path: "state".to_owned(),
            offset: 0,
            data: b"v1".to_vec(),
        },
        StorageOp::SyncFile {
            path: "state".to_owned(),
        },
        StorageOp::SyncDir { dir: String::new() },
    ] {
        cluster
            .submit_storage(node(1), op, StorageFault::None)
            .expect("submit");
    }
    cluster
        .schedule_app(Ticks::from_micros(0), node(1), TinyEv::Tick)
        .expect("timer");
    cluster.run_until_idle(20, &mut app, &mut rng).expect("run");
    assert_eq!(
        cluster.app_states().get(&node(1)).expect("a").timer_fires,
        1
    );
    cluster.crash_node(node(1)).expect("crash");
    // Volatile app state is gone with the process ...
    assert!(!cluster.app_states().contains_key(&node(1)));
    // ... while the synced disk image survives the crash.
    assert_eq!(
        cluster.disk(node(1)).expect("disk").durable_image("state"),
        Some(b"v1".to_vec())
    );
    // Restart brings a fresh volatile state; the disk is untouched.
    cluster.begin_restart(node(1)).expect("begin");
    cluster
        .finish_restart(node(1), TinyState::default())
        .expect("restart");
    assert_eq!(
        cluster.app_states().get(&node(1)).expect("a").timer_fires,
        0
    );
    assert_eq!(
        cluster.disk(node(1)).expect("disk").durable_image("state"),
        Some(b"v1".to_vec())
    );
}

/// Self-perpetuating timer app: every tick schedules another tick, so any
/// run over it would loop forever without an explicit event bound.
struct LoopApp;

impl AppHandler<TinyEv, TinyState> for LoopApp {
    fn handle_app(
        &mut self,
        node: NodeId,
        _event: TinyEv,
        _state: &mut TinyState,
        actions: &mut Vec<ClusterAction<TinyEv>>,
        _rng: &mut dyn RandomSource,
    ) {
        actions.push(ClusterAction::AppAfter {
            delay: Duration::from_micros(1),
            target: node,
            event: TinyEv::Tick,
        });
    }

    fn handle_net(
        &mut self,
        _from: Endpoint,
        _to: NodeId,
        _bytes: &[u8],
        _state: &mut TinyState,
        _actions: &mut Vec<ClusterAction<TinyEv>>,
        _rng: &mut dyn RandomSource,
    ) {
    }
}

#[test]
fn run_bounds_and_idle_reporting() {
    let mut cluster = two_nodes();
    let mut app = TinyApp;
    let mut rng = SimRng::seed_from_u64(4);
    // Self-perpetuating timer would loop forever unbounded.
    let mut looping = LoopApp;
    cluster
        .schedule_app(Ticks::from_micros(0), node(1), TinyEv::Tick)
        .expect("seed timer");
    let stats = cluster
        .run_until_idle(100, &mut looping, &mut rng)
        .expect("run");
    assert!(!stats.idle, "bound stops the infinite loop");
    assert_eq!(stats.processed, 100);
    // run_until_tick stops at the tick boundary.
    let stats = cluster
        .run_until_tick(Ticks::from_micros(150), 10_000, &mut looping, &mut rng)
        .expect("run");
    assert!(!stats.idle);
    assert!(cluster.now() <= Ticks::from_micros(150));
    // An empty scheduler is idle immediately.
    let mut empty: SimCluster<TinyEv, TinyState> = SimCluster::new(0, Ticks::from_micros(0));
    assert_eq!(
        empty.run_next(&mut app, &mut rng).expect("idle"),
        StepOutcome::Idle
    );
}

#[test]
fn error_conversions_wrap_sources_with_text() {
    use std::error::Error;
    let net: ClusterError = NetError::ZeroBandwidth.into();
    assert_eq!(net, ClusterError::Net(NetError::ZeroBandwidth));
    assert!(net.source().is_some());
    assert_eq!(net.to_string(), "network: link bandwidth must be nonzero");
    let missing: ClusterError = ClusterError::MissingRestartFactory { node: node(1) };
    assert!(missing.source().is_none());
    assert_eq!(
        missing.to_string(),
        format!("no restart-state factory set for node {}", node(1))
    );
}

#[test]
fn invariant_failures_carry_reproduction_context() {
    let mut cluster = two_nodes();
    let mut app = TinyApp;
    let mut rng = SimRng::seed_from_u64(5);
    cluster.add_invariant("always-no", |_| Err("nope".to_owned()));
    cluster
        .schedule_app(Ticks::from_micros(0), node(1), TinyEv::Tick)
        .expect("timer");
    let failure = match cluster.run_until_idle(10, &mut app, &mut rng) {
        Err(ClusterError::Invariant(failure)) => failure,
        other => panic!("expected invariant failure, got {other:?}"),
    };
    assert_eq!(failure.name(), "always-no");
    assert_eq!(failure.tick(), Ticks::from_micros(0));
    assert_eq!(failure.trace_pos(), cluster.trace().len());
    assert_eq!(failure.detail(), "nope");
}

#[test]
fn duplicate_policy_produces_two_deliveries_one_identity() {
    struct DupOnce {
        done: bool,
    }
    impl FaultPolicy<kivi_sim::NetDeliveryContext> for DupOnce {
        fn decide(
            &mut self,
            _: &EventView,
            _: &kivi_sim::NetDeliveryContext,
            _: &mut dyn RandomSource,
        ) -> kivi_core::FaultDecision {
            if self.done {
                kivi_core::FaultDecision::Allow
            } else {
                self.done = true;
                kivi_core::FaultDecision::Duplicate
            }
        }
    }
    let mut cluster = two_nodes();
    cluster.set_net_policy(DupOnce { done: false });
    let mut app = TinyApp;
    let mut rng = SimRng::seed_from_u64(6);
    cluster.send(node(1), node(2), b"x".to_vec()).expect("send");
    cluster.run_until_idle(10, &mut app, &mut rng).expect("run");
    let delivered: Vec<_> = cluster
        .trace()
        .iter()
        .filter_map(|entry| match entry {
            ClusterTrace::Delivered { msg, .. } => Some(*msg),
            _ => None,
        })
        .collect();
    assert_eq!(delivered.len(), 2, "original plus one copy");
    assert_eq!(
        delivered[0], delivered[1],
        "duplicates share the message identity"
    );
    assert_eq!(
        cluster
            .app_states()
            .get(&node(2))
            .expect("b")
            .received
            .len(),
        2
    );
    assert!(
        cluster
            .trace()
            .iter()
            .any(|entry| matches!(entry, ClusterTrace::Duplicated { .. }))
    );
}

// ---------------------------------------------------------------------------
// Synthetic combined machine: command → write → sync → ack, under faults.
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct MailboxState {
    acked: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum MailboxEv {
    Command,
    Synced,
}

struct MailboxApp;

impl AppHandler<MailboxEv, MailboxState> for MailboxApp {
    fn handle_app(
        &mut self,
        this: NodeId,
        event: MailboxEv,
        _state: &mut MailboxState,
        actions: &mut Vec<ClusterAction<MailboxEv>>,
        _rng: &mut dyn RandomSource,
    ) {
        match event {
            MailboxEv::Command => {
                // Phase 1: create + write the state file (sync comes later,
                // so a crash can land between the two phases).
                actions.push(ClusterAction::Storage {
                    op: StorageOp::Create {
                        path: "state".to_owned(),
                    },
                    fault: StorageFault::None,
                });
                actions.push(ClusterAction::Storage {
                    op: StorageOp::Write {
                        path: "state".to_owned(),
                        offset: 0,
                        data: b"cmd".to_vec(),
                    },
                    fault: StorageFault::None,
                });
                actions.push(ClusterAction::AppAfter {
                    delay: Duration::from_micros(10),
                    target: this,
                    event: MailboxEv::Synced,
                });
            }
            MailboxEv::Synced => {
                // Phase 2: make it durable, then acknowledge to node 2.
                actions.push(ClusterAction::Storage {
                    op: StorageOp::SyncFile {
                        path: "state".to_owned(),
                    },
                    fault: StorageFault::None,
                });
                actions.push(ClusterAction::Storage {
                    op: StorageOp::SyncDir { dir: String::new() },
                    fault: StorageFault::None,
                });
                actions.push(ClusterAction::NetSend {
                    to: node(2),
                    payload: b"ack".to_vec(),
                });
            }
        }
    }

    fn handle_net(
        &mut self,
        _from: Endpoint,
        to: NodeId,
        bytes: &[u8],
        state: &mut MailboxState,
        _actions: &mut Vec<ClusterAction<MailboxEv>>,
        _rng: &mut dyn RandomSource,
    ) {
        assert_eq!(to, node(2));
        assert_eq!(bytes, b"ack");
        state.acked = true;
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MailboxFault {
    None,
    CrashBeforeSync,
    CrashAfterSync,
    LostAck,
    DelayedAck,
    OneWayPartition,
}

fn run_mailbox(fault: MailboxFault) -> SimCluster<MailboxEv, MailboxState> {
    let mut cluster = SimCluster::new(77, Ticks::from_micros(0));
    cluster
        .add_node(node(1), 1 << 20, MailboxState::default())
        .expect("a");
    cluster
        .add_node(node(2), 1 << 20, MailboxState::default())
        .expect("b");
    cluster.connect(node(1), node(2), link()).expect("mesh");
    cluster.set_restart_factory(|_| MailboxState::default());
    let mut app = MailboxApp;
    let mut rng = SimRng::seed_from_u64(77);
    // Command runs at t=100: write executes ~100, Synced at ~110.
    cluster
        .schedule_app(Ticks::from_micros(100), node(1), MailboxEv::Command)
        .expect("cmd");
    match fault {
        MailboxFault::None => {}
        MailboxFault::CrashBeforeSync => {
            cluster
                .schedule_node_step(Ticks::from_micros(105), NodeStep::Crash(node(1)))
                .expect("crash");
        }
        MailboxFault::CrashAfterSync => {
            // The ack delivers (~115) long before the crash: sync and send
            // both completed, so restart changes nothing observable.
            cluster
                .schedule_node_step(Ticks::from_micros(6_000), NodeStep::Crash(node(1)))
                .expect("crash");
            cluster
                .schedule_node_step(Ticks::from_micros(6_100), NodeStep::BeginRestart(node(1)))
                .expect("begin");
            cluster
                .schedule_node_step(Ticks::from_micros(6_200), NodeStep::FinishRestart(node(1)))
                .expect("finish");
        }
        MailboxFault::LostAck => {
            cluster.set_net_policy(DropEveryNth::new(NonZeroU64::new(1).expect("1")));
        }
        MailboxFault::DelayedAck => {
            struct DeferFirst {
                done: bool,
            }
            impl FaultPolicy<kivi_sim::NetDeliveryContext> for DeferFirst {
                fn decide(
                    &mut self,
                    _: &EventView,
                    ctx: &kivi_sim::NetDeliveryContext,
                    _: &mut dyn RandomSource,
                ) -> kivi_core::FaultDecision {
                    if self.done {
                        kivi_core::FaultDecision::Allow
                    } else {
                        self.done = true;
                        kivi_core::FaultDecision::Defer {
                            until: Ticks::from_micros(ctx.deliver_at().as_micros() + 1_000),
                        }
                    }
                }
            }
            cluster.set_net_policy(DeferFirst { done: false });
        }
        MailboxFault::OneWayPartition => {
            cluster
                .partition_one_way(node(1), node(2))
                .expect("partition");
        }
    }
    cluster
        .run_until_idle(1_000, &mut app, &mut rng)
        .expect("run");
    cluster
}

#[rstest]
#[case(MailboxFault::None, true, true)]
#[case(MailboxFault::CrashBeforeSync, false, false)]
#[case(MailboxFault::CrashAfterSync, true, true)]
#[case(MailboxFault::LostAck, false, true)]
#[case(MailboxFault::DelayedAck, true, true)]
#[case(MailboxFault::OneWayPartition, false, true)]
fn mailbox_composes_network_storage_and_lifecycle(
    #[case] fault: MailboxFault,
    #[case] expect_ack: bool,
    #[case] expect_durable: bool,
) {
    let cluster = run_mailbox(fault);
    assert_eq!(
        cluster.app_states().get(&node(2)).is_some_and(|s| s.acked),
        expect_ack,
        "fault {fault:?}"
    );
    assert_eq!(
        cluster
            .disk(node(1))
            .expect("disk")
            .durable_image("state")
            .is_some(),
        expect_durable,
        "fault {fault:?}"
    );
    // Replay equivalence: the same faulted scenario rebuilds equal state.
    let replay = run_mailbox(fault);
    assert_eq!(
        replay.snapshot(),
        cluster.snapshot(),
        "fault {fault:?} must replay"
    );
}
