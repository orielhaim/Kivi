//! Four-node deterministic mesh regression harness: the first real Kivi
//! cluster simulation test.
//!
//! Four synthetic nodes exchange pings/pongs over a full mesh while writing
//! every receipt to durable storage, under faults (probabilistic drop,
//! duplication, deferral, one-way partitions, healing, disconnects,
//! crashes, restarts, injected storage faults). The same seed rebuilt twice
//! must produce an equal scenario result; three fixed seeds run via rstest.

use core::time::Duration;

use kivi_core::RandomSource;
use kivi_sim::{
    AppHandler, ClusterAction, ClusterTrace, Endpoint, LinkConfig, SimCluster, SimRng,
    StorageFault, StorageOp,
};
use kivi_types::{NodeId, Ticks};
use rstest::rstest;

fn node(n: u64) -> NodeId {
    NodeId::from_u64(n)
}

fn mesh_link() -> LinkConfig {
    LinkConfig::new(Duration::from_millis(5), 1_000_000, 64).expect("valid link")
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct BigState {
    sent: u64,
    received: u64,
    pongs: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum BigEv {
    Tick { remaining: u64 },
}

fn encode_ping(seq: u64) -> Vec<u8> {
    let mut bytes = vec![1u8];
    bytes.extend_from_slice(&seq.to_le_bytes());
    bytes
}

fn encode_pong(seq: u64) -> Vec<u8> {
    let mut bytes = vec![2u8];
    bytes.extend_from_slice(&seq.to_le_bytes());
    bytes
}

/// Decodes `(is_ping, seq)`; unknown shapes are ignored by the app.
fn decode(bytes: &[u8]) -> Option<(bool, u64)> {
    if bytes.len() != 9 {
        return None;
    }
    let mut seq = [0u8; 8];
    seq.copy_from_slice(&bytes[1..9]);
    match bytes[0] {
        1 => Some((true, u64::from_le_bytes(seq))),
        2 => Some((false, u64::from_le_bytes(seq))),
        _ => None,
    }
}

struct BigApp;

impl AppHandler<BigEv, BigState> for BigApp {
    fn handle_app(
        &mut self,
        this: NodeId,
        event: BigEv,
        state: &mut BigState,
        actions: &mut Vec<ClusterAction<BigEv>>,
        rng: &mut dyn RandomSource,
    ) {
        let BigEv::Tick { remaining } = event;
        // Deterministic peer choice among the other three nodes.
        let peers: Vec<NodeId> = [1u64, 2, 3, 4]
            .into_iter()
            .map(node)
            .filter(|peer| *peer != this)
            .collect();
        let peer = peers[usize::try_from(rng.below_u64(3)).expect("small index")];
        actions.push(ClusterAction::NetSend {
            to: peer,
            payload: encode_ping(state.sent),
        });
        state.sent += 1;
        if remaining > 0 {
            actions.push(ClusterAction::AppAfter {
                delay: Duration::from_micros(50),
                target: this,
                event: BigEv::Tick {
                    remaining: remaining - 1,
                },
            });
        }
    }

    fn handle_net(
        &mut self,
        from: Endpoint,
        _to: NodeId,
        bytes: &[u8],
        state: &mut BigState,
        actions: &mut Vec<ClusterAction<BigEv>>,
        _rng: &mut dyn RandomSource,
    ) {
        let Some((is_ping, seq)) = decode(bytes) else {
            return;
        };
        if is_ping {
            state.received += 1;
            // Durable receipt log (single slot); every third write is lost
            // explicitly to model lost updates, then still synced.
            let fault = if state.received.is_multiple_of(3) {
                StorageFault::LostWrite
            } else {
                StorageFault::None
            };
            actions.push(ClusterAction::Storage {
                op: StorageOp::Write {
                    path: "log".to_owned(),
                    offset: 0,
                    data: seq.to_le_bytes().to_vec(),
                },
                fault,
            });
            actions.push(ClusterAction::Storage {
                op: StorageOp::SyncFile {
                    path: "log".to_owned(),
                },
                fault: StorageFault::None,
            });
            actions.push(ClusterAction::NetSend {
                to: from.node(),
                payload: encode_pong(seq),
            });
        } else {
            state.pongs += 1;
        }
    }
}

/// Lossy, duplicating, deferring network: 10% drop, 4% duplicate, 4% defer
/// by 50µs, else deliver. Deterministic given the RNG stream.
struct Chaos;

impl kivi_core::FaultPolicy<kivi_sim::NetDeliveryContext> for Chaos {
    fn decide(
        &mut self,
        _: &kivi_core::EventView,
        ctx: &kivi_sim::NetDeliveryContext,
        rng: &mut dyn RandomSource,
    ) -> kivi_core::FaultDecision {
        match rng.below_u64(100) {
            0..10 => kivi_core::FaultDecision::Drop,
            10..14 => kivi_core::FaultDecision::Duplicate,
            14..18 => kivi_core::FaultDecision::Defer {
                until: Ticks::from_micros(ctx.deliver_at().as_micros() + 50),
            },
            _ => kivi_core::FaultDecision::Allow,
        }
    }
}

/// No delivery may reach a process that is not running.
fn delivered_only_to_running<E, S>(cluster: &SimCluster<E, S>) -> Result<(), String> {
    use std::collections::BTreeMap;
    let mut running: BTreeMap<NodeId, bool> = BTreeMap::new();
    for info in cluster.nodes().iter() {
        running.insert(info.id(), info.state() == kivi_sim::NodeState::Running);
    }
    // Replay lifecycle from history: the check itself folds the trace.
    running.clear();
    for info in cluster.nodes().iter() {
        running.insert(info.id(), true);
    }
    for entry in cluster.trace() {
        match entry {
            ClusterTrace::NodeCrashed { node, .. } => {
                running.insert(*node, false);
            }
            ClusterTrace::NodeRestarted { node, .. } => {
                running.insert(*node, true);
            }
            ClusterTrace::Delivered { to, .. } if running.get(&to.node()) != Some(&true) => {
                return Err(format!("delivery to non-running {}", to.node().as_u64()));
            }
            _ => {}
        }
    }
    Ok(())
}

/// Every rejected sender incarnation must predate the current one.
fn stale_rejections_are_older<E, S>(cluster: &SimCluster<E, S>) -> Result<(), String> {
    for entry in cluster.trace() {
        if let ClusterTrace::StaleMessageRejected { from, current, .. } = entry
            && from.incarnation() >= *current
        {
            return Err("stale rejection names a non-older incarnation".to_owned());
        }
    }
    Ok(())
}

fn build_mesh(seed: u64) -> SimCluster<BigEv, BigState> {
    let mut cluster = SimCluster::new(seed, Ticks::from_micros(0));
    for n in 1..=4 {
        cluster
            .add_node(node(n), 1 << 20, BigState::default())
            .expect("add node");
    }
    for a in 1..=4 {
        for b in 1..=4 {
            if a != b {
                cluster
                    .connect(node(a), node(b), mesh_link())
                    .expect("mesh link");
            }
        }
    }
    // Receipt-log files exist before traffic lands.
    for n in 1..=4 {
        cluster
            .submit_storage(
                node(n),
                StorageOp::Create {
                    path: "log".to_owned(),
                },
                StorageFault::None,
            )
            .expect("create log");
    }
    cluster.set_net_policy(Chaos);
    cluster.set_restart_factory(|_| BigState::default());
    cluster.add_invariant("delivered-only-to-running", delivered_only_to_running);
    cluster.add_invariant("stale-rejections-are-older", stale_rejections_are_older);
    // Staggered tick chains: 150 ticks per node, 50µs apart.
    for (index, n) in (1..=4).enumerate() {
        cluster
            .schedule_app(
                Ticks::from_micros(index as u64 * 7),
                node(n),
                BigEv::Tick { remaining: 150 },
            )
            .expect("seed ticks");
    }
    // Fault script (identical every run): one-way partition, crash cycle
    // with restart, heal, disconnect cycle, injected storage faults.
    cluster
        .partition_one_way(node(1), node(2))
        .expect("partition");
    cluster
        .submit_storage(
            node(4),
            StorageOp::Write {
                path: "log".to_owned(),
                offset: 0,
                data: vec![0xAA; 8],
            },
            StorageFault::TornWrite { applied_bytes: 3 },
        )
        .expect("torn write");
    cluster
        .submit_storage(
            node(4),
            StorageOp::SyncFile {
                path: "log".to_owned(),
            },
            StorageFault::EIO,
        )
        .expect("failing sync");
    cluster
        .schedule_node_step(Ticks::from_micros(300), kivi_sim::NodeStep::Crash(node(3)))
        .expect("crash");
    cluster
        .schedule_node_step(
            Ticks::from_micros(400),
            kivi_sim::NodeStep::BeginRestart(node(3)),
        )
        .expect("begin");
    cluster
        .schedule_node_step(
            Ticks::from_micros(500),
            kivi_sim::NodeStep::FinishRestart(node(3)),
        )
        .expect("finish");
    // The restarted node rejoins the tick chains explicitly.
    cluster
        .schedule_app(
            Ticks::from_micros(500),
            node(3),
            BigEv::Tick { remaining: 150 },
        )
        .expect("rejoin");
    cluster.heal(node(1), node(2)).expect("heal");
    cluster.disconnect(node(2), node(4)).expect("disconnect");
    cluster.reconnect(node(2), node(4)).expect("reconnect");
    cluster
}

fn drive_mesh(seed: u64) -> kivi_sim::ScenarioResult<BigEv, BigState> {
    let mut cluster = build_mesh(seed);
    let mut app = BigApp;
    let mut rng = SimRng::seed_from_u64(seed);
    let stats = cluster
        .run_until_idle(30_000, &mut app, &mut rng)
        .expect("run drains");
    assert!(stats.idle, "seed {seed}: scenario must drain within budget");
    cluster.snapshot()
}

#[rstest]
#[case(11)]
#[case(22)]
#[case(33)]
fn mesh_regressions_replay_exactly(#[case] seed: u64) {
    let first = drive_mesh(seed);
    let second = drive_mesh(seed);
    assert_eq!(first, second, "seed {seed} must replay bit-for-bit");
    assert!(
        first.events_processed > 2_000,
        "must run thousands of events"
    );
    assert_eq!(first.format, kivi_sim::SIM_FORMAT_VERSION);
    assert_eq!(first.seed, seed);
    // Every fault family actually fired in the run.
    for (present, what) in [
        (
            first
                .trace
                .iter()
                .any(|e| matches!(e, ClusterTrace::Duplicated { .. })),
            "duplicates",
        ),
        (
            first
                .trace
                .iter()
                .any(|e| matches!(e, ClusterTrace::Deferred { .. })),
            "deferrals",
        ),
        (
            first
                .trace
                .iter()
                .any(|e| matches!(e, ClusterTrace::Partitioned { .. })),
            "partitions",
        ),
        (
            first
                .trace
                .iter()
                .any(|e| matches!(e, ClusterTrace::Healed { .. })),
            "heals",
        ),
        (
            first
                .trace
                .iter()
                .any(|e| matches!(e, ClusterTrace::NodeCrashed { .. })),
            "crashes",
        ),
        (
            first
                .trace
                .iter()
                .any(|e| matches!(e, ClusterTrace::NodeRestarted { .. })),
            "restarts",
        ),
        (
            first
                .trace
                .iter()
                .any(|e| matches!(e, ClusterTrace::StaleMessageRejected { .. })),
            "stale rejections",
        ),
        (
            first
                .trace
                .iter()
                .any(|e| matches!(e, ClusterTrace::StorageFailure { .. })),
            "storage failures",
        ),
    ] {
        assert!(present, "seed {seed}: expected {what} in trace");
    }
}
