//! `OpenRaft` 0.10 group-density experiment on the Compio runtime.
//!
//! Spawns N minimal in-memory single-voter groups on one Compio reactor
//! and reports startup, election, idle, and shutdown costs. This answers
//! whether `OpenRaft`'s "one `RaftCore` task per group, one replication
//! task per target on the leader" model is acceptable before Kivi depends
//! on it at scale.
//!
//! ```text
//! cargo run --release -p kivi-consensus --example density -- 100
//! cargo run --release -p kivi-consensus --example density -- 1000
//! cargo run --release -p kivi-consensus --example density -- 10000
//! cargo run --release -p kivi-consensus --example density -- 100 3
//! ```
//!
//! The optional second argument selects voters per group: `1` (default) is
//! a lone voter with the unreachable stub network; `3` wires three
//! in-process voters per group through ONE shared [`SharedRegistry`]
//! implementing the real [`openraft_multi::GroupRouter`] (group ids route
//! to the right instance), so leaders hold real replication streams and
//! followers receive real heartbeats — the honest proxy for mostly-idle
//! Multi-Raft tablet groups, with production-faithful shared routing
//! (never a connection object per group).
//!
//! RSS and idle CPU are sampled in-process via `sysinfo` (a
//! dev-dependency: measurement only, never production). Task counts are
//! not reported: single-threaded Compio exposes no task census like
//! Tokio's unstable metrics did; RSS and CPU are the density signals.
//!
//! Deliberately unoptimized: stub/in-memory transports, production-shaped
//! election timeouts. The numbers must reflect the architecture, not
//! tuning.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::io::Cursor;
use std::rc::Rc;
use std::sync::Arc;
use std::time::{Duration, Instant};

use kivi_consensus::config::{KiviTypeConfig, spike_config};
use kivi_consensus::spike::{MemLogStore, MemStateMachine, StubNetworkFactory};
use kivi_consensus::types::ConsensusGroupId;
use kivi_types::TabletId;
use openraft::Raft;
use openraft::type_config::async_runtime::watch::WatchReceiver as _;

fn usage() -> ! {
    eprintln!("usage: density <groups> [voters=1|3] [hold_secs]");
    std::process::exit(2);
}

type DensityRaft = Raft<KiviTypeConfig, MemStateMachine>;

/// Shared in-process router for the 3-voter mode: one registry keyed by
/// `(group, node)`, implementing the real [`openraft_multi::GroupRouter`]
/// every group shares. Missing targets (not yet spawned) report
/// `Unreachable`, which makes `OpenRaft` back off instead of failing.
/// Group ids prove routing correctness: convergence means every RPC
/// reached the right instance.
#[derive(Clone, Default)]
struct SharedRegistry {
    rafts: Rc<RefCell<BTreeMap<(u64, u64), DensityRaft>>>,
}

impl SharedRegistry {
    fn lookup(&self, group: u64, target: u64) -> Option<DensityRaft> {
        self.rafts.borrow().get(&(group, target)).cloned()
    }
}

fn unreachable(target: u64, group: u64) -> openraft::errors::Unreachable<KiviTypeConfig> {
    let source = std::io::Error::new(
        std::io::ErrorKind::NotConnected,
        format!("spike registry: node {target} of group {group} not spawned yet"),
    );
    openraft::errors::Unreachable::new(&source)
}

impl openraft_multi::GroupRouter<KiviTypeConfig, ConsensusGroupId> for SharedRegistry {
    type SnapshotData = Cursor<Vec<u8>>;

    async fn append_entries(
        &self,
        target: u64,
        group_id: ConsensusGroupId,
        rpc: openraft::raft::AppendEntriesRequest<KiviTypeConfig>,
        _option: openraft::network::RPCOption,
    ) -> Result<
        openraft::raft::AppendEntriesResponse<KiviTypeConfig>,
        openraft::errors::RPCError<KiviTypeConfig>,
    > {
        use openraft::errors::{RPCError, Unreachable};
        let group = group_id.tablet().as_u64();
        let Some(raft) = self.lookup(group, target) else {
            return Err(RPCError::Unreachable(unreachable(target, group)));
        };
        raft.append_entries(rpc).await.map_err(|error| {
            // Direct in-process service refused (shutting down): the
            // retry-safe error backs the caller off, never a verdict.
            let source = std::io::Error::other(error.to_string());
            RPCError::Unreachable(Unreachable::new(&source))
        })
    }

    async fn vote(
        &self,
        target: u64,
        group_id: ConsensusGroupId,
        rpc: openraft::raft::VoteRequest<KiviTypeConfig>,
        _option: openraft::network::RPCOption,
    ) -> Result<
        openraft::raft::VoteResponse<KiviTypeConfig>,
        openraft::errors::RPCError<KiviTypeConfig>,
    > {
        use openraft::errors::{RPCError, Unreachable};
        let group = group_id.tablet().as_u64();
        let Some(raft) = self.lookup(group, target) else {
            return Err(RPCError::Unreachable(unreachable(target, group)));
        };
        raft.vote(rpc).await.map_err(|error| {
            let source = std::io::Error::other(error.to_string());
            RPCError::Unreachable(Unreachable::new(&source))
        })
    }

    #[allow(clippy::unused_async_trait_impl)]
    async fn full_snapshot(
        &self,
        target: u64,
        group_id: ConsensusGroupId,
        _vote: openraft::type_config::alias::VoteOf<KiviTypeConfig>,
        _snapshot: openraft::type_config::alias::SnapshotOf<KiviTypeConfig, Cursor<Vec<u8>>>,
        _cancel: impl Future<Output = openraft::errors::ReplicationClosed>
        + openraft::OptionalSend
        + 'static,
        _option: openraft::network::RPCOption,
    ) -> Result<
        openraft::raft::SnapshotResponse<KiviTypeConfig>,
        openraft::errors::StreamingError<KiviTypeConfig>,
    > {
        // Snapshots never trigger in the experiment (tiny logs); a call
        // here would be a harness bug, failed loudly.
        Err(openraft::errors::StreamingError::Unreachable(unreachable(
            target,
            group_id.tablet().as_u64(),
        )))
    }
}

/// RSS of this process in MiB, sampled fresh.
fn rss_mib(sys: &mut sysinfo::System) -> f64 {
    use sysinfo::{Pid, ProcessesToUpdate};
    sys.refresh_processes(ProcessesToUpdate::All, true);
    let pid = Pid::from(std::process::id() as usize);
    sys.process(pid).map_or(0.0, |process| {
        // Bytes-to-MiB at these magnitudes (MiB to low GiB) never approach
        // the 2^53 exact-integer limit of `f64`.
        #[allow(clippy::cast_precision_loss)]
        let bytes = process.memory() as f64;
        bytes / 1_048_576.0
    })
}

fn main() {
    let mut args = std::env::args().skip(1);
    let groups: usize = match args.next().and_then(|raw| raw.parse().ok()) {
        Some(groups) if groups > 0 && groups <= 100_000 => groups,
        _ => usage(),
    };
    let voters: usize = match args.next() {
        None => 1,
        Some(raw) => match raw.parse() {
            Ok(voters @ (1 | 3)) => voters,
            _ => usage(),
        },
    };
    let hold_secs: u64 = match args.next() {
        None => 0,
        Some(raw) => match raw.parse() {
            Ok(secs) => secs,
            Err(_) => usage(),
        },
    };
    if args.next().is_some() {
        usage();
    }

    compio::runtime::Runtime::new()
        .expect("compio density runtime")
        .block_on(run(groups, voters, hold_secs));
}

/// Spawns one lone voter: unreachable stub network, self-electing.
async fn spawn_lone_voter(config: &Arc<openraft::Config>, node_id: u64) -> DensityRaft {
    let raft = Raft::new(
        node_id,
        Arc::clone(config),
        StubNetworkFactory,
        MemLogStore::new(),
        MemStateMachine::new(),
    )
    .await
    .expect("raft spawns");
    raft.initialize(BTreeMap::from([(
        node_id,
        openraft::BasicNode::new(format!("density-{node_id}")),
    )]))
    .await
    .expect("lone voter initializes");
    raft
}

/// Spawns one 3-voter group on the shared registry. All three
/// initialize; a `NotAllowed` from a late voter is safe to ignore per
/// `OpenRaft`'s contract — it means a peer already won an election and
/// replicated to this node, i.e. the cluster is up. The agreement wait
/// proves convergence either way.
async fn spawn_voter_group(
    config: &Arc<openraft::Config>,
    registry: &SharedRegistry,
    group: u64,
) -> Vec<DensityRaft> {
    use openraft::errors::{InitializeError, RaftError};
    let group_id = ConsensusGroupId::of_tablet(TabletId::from_u64(group));
    let members: BTreeMap<u64, openraft::BasicNode> = [0u64, 1, 2]
        .into_iter()
        .map(|node| {
            (
                node,
                openraft::BasicNode::new(format!("density-{group}-{node}")),
            )
        })
        .collect();
    let mut voters_rafts = Vec::with_capacity(3);
    for node_id in [0u64, 1, 2] {
        let raft = Raft::new(
            node_id,
            Arc::clone(config),
            kivi_consensus::router::GroupNetworkFactory::new(registry.clone(), group_id),
            MemLogStore::new(),
            MemStateMachine::new(),
        )
        .await
        .expect("raft spawns");
        registry
            .rafts
            .borrow_mut()
            .insert((group, node_id), raft.clone());
        if let Err(error) = raft.initialize(members.clone()).await
            && !matches!(error, RaftError::APIError(InitializeError::NotAllowed(_)))
        {
            panic!("group {group} voter {node_id} initializes: {error:?}");
        }
        voters_rafts.push(raft);
    }
    voters_rafts
}

/// Waits until every voter agrees on its group's leader (proves the setup
/// converged, including `NotAllowed`-tolerated initializations).
async fn wait_for_leaders(rafts: &[DensityRaft], groups: usize, voters: usize) {
    let elected = Instant::now();
    let deadline = Duration::from_secs(180);
    loop {
        let mut settled = 0usize;
        for chunk in rafts.chunks(voters) {
            let mut leader = None;
            let mut agreed = true;
            for raft in chunk {
                match raft.metrics().borrow_watched().current_leader {
                    Some(id) if leader.is_none_or(|l| l == id) => leader = Some(id),
                    _ => {
                        agreed = false;
                        break;
                    }
                }
            }
            if agreed && leader.is_some() {
                settled += 1;
            }
        }
        if settled == groups {
            return;
        }
        assert!(
            elected.elapsed() <= deadline,
            "only {settled}/{groups} groups settled after {deadline:?}"
        );
        compio::time::sleep(Duration::from_millis(100)).await;
    }
}

async fn run(groups: usize, voters: usize, hold_secs: u64) {
    let mut sys = sysinfo::System::new();
    let config = spike_config().expect("spike config validates");
    let registry = SharedRegistry::default();

    // --- Startup: spawn every group and initialize its voters. ---
    let started = Instant::now();
    let mut rafts = Vec::with_capacity(groups * voters);
    for group in 0..groups {
        if voters == 1 {
            rafts.push(spawn_lone_voter(&config, group as u64).await);
        } else {
            rafts.extend(spawn_voter_group(&config, &registry, group as u64).await);
        }
        if (group + 1) % 1000 == 0 {
            eprintln!("spawned {}/{}", group + 1, groups);
        }
    }
    let startup_ms = started.elapsed().as_millis();
    let rss_spawn_mib = rss_mib(&mut sys);

    // --- Election: wait until every voter agrees on its group's leader. ---
    let elected = Instant::now();
    wait_for_leaders(&rafts, groups, voters).await;
    let elect_ms = elected.elapsed().as_millis();

    // --- Idle window: 5 s of mostly-idle ticking. `cpu_usage` covers
    // the window since the previous refresh, so exactly one refresh
    // happens after the sleep and both RSS and CPU are read from it.
    // (An extra refresh between sleep-end and the CPU read would shrink
    // the window to milliseconds and sample heartbeat bursts instead of
    // the average — a methodology bug the old spike shared.)
    sys.refresh_processes(sysinfo::ProcessesToUpdate::All, true);
    compio::time::sleep(Duration::from_secs(5)).await;
    let (rss_idle_mib, idle_cpu_pct) = idle_sample(&mut sys);

    // --- Shutdown: every group leaves cleanly. ---
    let stopping = Instant::now();
    for raft in &rafts {
        raft.shutdown().await.expect("group shuts down");
    }
    let shutdown_ms = stopping.elapsed().as_millis();

    println!(
        "DENSITY groups={groups} voters={voters} startup_ms={startup_ms} elect_ms={elect_ms} \
         rss_spawn_mib={rss_spawn_mib:.1} rss_idle_mib={rss_idle_mib:.1} \
         idle_cpu_pct={idle_cpu_pct:.1} shutdown_ms={shutdown_ms}",
    );
    if hold_secs > 0 {
        println!("HOLDING {hold_secs}s for external sampling");
        compio::time::sleep(Duration::from_secs(hold_secs)).await;
    }
}

/// This process's CPU load as percent of one core over the window since
/// the caller's previous refresh, plus current RSS. The caller sleeps
/// between its baseline refresh and this call, so the CPU result is the
/// true average over the window — not an instantaneous sample.
fn idle_sample(sys: &mut sysinfo::System) -> (f64, f64) {
    use sysinfo::{Pid, ProcessesToUpdate};
    sys.refresh_processes(ProcessesToUpdate::All, true);
    let pid = Pid::from(std::process::id() as usize);
    sys.process(pid).map_or((0.0, 0.0), |process| {
        // Bytes-to-MiB at these magnitudes (MiB to low GiB) never approach
        // the 2^53 exact-integer limit of `f64`.
        #[allow(clippy::cast_precision_loss)]
        let bytes = process.memory() as f64;
        (bytes / 1_048_576.0, f64::from(process.cpu_usage()))
    })
}
