//! `kivi-server`: runnable single-node Kivi server with native endpoints.
//!
//! Starts engine workers (each with its own native TCP endpoint), serves the
//! read-only admin/control HTTP plane on Tokio, and shuts down orderly on
//! Ctrl-C. Explicitly **in-memory / non-durable** in this stage:
//! acknowledged writes survive process crashes only after the durability
//! fabric lands.

mod admin;

use std::net::{IpAddr, SocketAddr};

use admin::AdminState;
use anyhow::Context;
use clap::Parser;
use kivi_engine::{
    AffinityMode, ConnLimits, EngineConfig, EngineNetwork, LocalEngine, Placement, TurnBudget,
};
use kivi_state::PartitionHasher;
use kivi_tablet::{DirectorySnapshot, HashPrefix, PartitionRange};
use kivi_types::{
    ClusterId, NamespaceId, NodeId, TabletEpoch, TabletId, WorkerId, WriteGuardGeneration,
};
use tracing_subscriber::EnvFilter;

const NS: NamespaceId = NamespaceId::from_u64(1);

#[derive(Debug, Parser)]
#[command(
    name = "kivi-server",
    about = "Kivi single-node native server (in-memory)"
)]
struct Args {
    /// IP to bind worker listeners on.
    #[arg(long, default_value = "127.0.0.1")]
    bind: IpAddr,
    /// Base TCP port; worker i listens on base + i (0 selects ephemeral ports).
    #[arg(long, default_value_t = 9000)]
    port: u16,
    /// OS worker threads (each owns tablets plus one native endpoint).
    #[arg(long, default_value_t = 2)]
    workers: usize,
    /// Tablet count: 1 (root) or a power of two for even top-bit splits.
    #[arg(long, default_value_t = 1, value_parser = parse_tablets)]
    tablets: usize,
    /// CPU pinning: auto, none, or explicit cores like "0,2-4,7".
    #[arg(long, default_value = "auto", value_parser = parse_affinity)]
    pin: AffinityMode,
    /// Per-worker bounded request queue depth.
    #[arg(long, default_value_t = 1024)]
    queue: usize,
    /// Maximum accepted frame in bytes (protocol ceiling 64 MiB).
    #[arg(long, default_value_t = 64 * 1024 * 1024, value_parser = parse_max_frame)]
    max_frame: usize,
    /// Node identity for handshakes.
    #[arg(long, default_value_t = 1)]
    node: u64,
    /// Cluster identity for handshakes.
    #[arg(long, default_value_t = 1)]
    cluster: u128,
    /// Admin/control HTTP listen address (loopback by default; `:0` selects
    /// an ephemeral port and prints it).
    #[arg(long, default_value = "127.0.0.1:19080")]
    admin: SocketAddr,
}

/// Parses CPU pinning without panicking: `auto`, `none`, or explicit cores
/// like `0,2-4,7`. A Clap value parser, so bad input is a usage error
/// (exit code 2), never a panic.
fn parse_affinity(spec: &str) -> Result<AffinityMode, String> {
    match spec {
        "none" => Ok(AffinityMode::Disabled),
        "auto" => Ok(AffinityMode::Auto),
        other => {
            let mut cores = Vec::new();
            for part in other.split(',') {
                if let Some((from, to)) = part.split_once('-') {
                    let from: usize = from
                        .trim()
                        .parse()
                        .map_err(|_| format!("invalid core range {part:?}"))?;
                    let to: usize = to
                        .trim()
                        .parse()
                        .map_err(|_| format!("invalid core range {part:?}"))?;
                    if from > to {
                        return Err(format!("inverted core range {part:?}"));
                    }
                    cores.extend(from..=to);
                } else {
                    cores.push(
                        part.trim()
                            .parse()
                            .map_err(|_| format!("invalid core index {part:?}"))?,
                    );
                }
            }
            if cores.is_empty() {
                return Err("no cores listed".to_owned());
            }
            Ok(AffinityMode::Explicit(cores))
        }
    }
}

/// Parses the tablet count: 1 (root) or a power of two for even splits.
fn parse_tablets(spec: &str) -> Result<usize, String> {
    let count: usize = spec
        .parse()
        .map_err(|_| format!("invalid tablet count {spec:?}"))?;
    if count == 1 || (count.is_power_of_two() && count.trailing_zeros() > 0) {
        Ok(count)
    } else {
        Err(format!("tablets must be 1 or a power of two, got {count}"))
    }
}

/// Parses the frame bound against the 64 MiB protocol ceiling.
fn parse_max_frame(spec: &str) -> Result<usize, String> {
    let bound: usize = spec
        .parse()
        .map_err(|_| format!("invalid frame bound {spec:?}"))?;
    if bound == 0 {
        return Err("frame bound must be nonzero".to_owned());
    }
    if bound > kivi_protocol::DEFAULT_MAX_FRAME {
        return Err(format!(
            "frame bound {bound} exceeds the 64 MiB protocol ceiling"
        ));
    }
    Ok(bound)
}

/// Builds an evenly split directory with `2^levels` active leaves plus a
/// round-robin placement, using only validated directory transitions.
fn even_split(levels: u32, workers: usize) -> (DirectorySnapshot, Placement) {
    let mut next_id = 1u64;
    let mut new_tablet = || {
        next_id += 1;
        TabletId::from_u64(next_id)
    };
    let genesis = DirectorySnapshot::bootstrap(
        NS,
        TabletId::from_u64(1),
        PartitionRange::Hash(HashPrefix::new(0, 0).expect("root")),
        TabletEpoch::INITIAL,
        WriteGuardGeneration::INITIAL,
    )
    .expect("genesis");
    let mut active = vec![TabletId::from_u64(1)];
    let mut directory = genesis
        .stage(TabletId::from_u64(1))
        .and_then(|s| s.activate(TabletId::from_u64(1)))
        .expect("root active");
    for _ in 0..levels {
        let mut next_active = Vec::with_capacity(active.len() * 2);
        for parent in active {
            let parent_range = match directory.get(parent).expect("parent").range() {
                PartitionRange::Hash(prefix) => *prefix,
                PartitionRange::Ordered(_) => panic!("even splits need the hash layout"),
            };
            let len = parent_range.prefix_len() + 1;
            let left = new_tablet();
            let right = new_tablet();
            // The children differ in bit (len-1) from the top: left clears
            // it, right sets it. Both stay canonical (low bits zero).
            let half = 1u128 << (128 - len);
            directory = directory
                .allocate(
                    left,
                    PartitionRange::Hash(HashPrefix::new(parent_range.bits(), len).expect("left")),
                    TabletEpoch::INITIAL,
                    WriteGuardGeneration::INITIAL,
                )
                .and_then(|s| s.stage(left))
                .expect("left staged");
            directory = directory
                .allocate(
                    right,
                    PartitionRange::Hash(
                        HashPrefix::new(parent_range.bits() | half, len).expect("right"),
                    ),
                    TabletEpoch::INITIAL,
                    WriteGuardGeneration::INITIAL,
                )
                .and_then(|s| s.stage(right))
                .expect("right staged");
            directory = directory.seal(parent).expect("seal parent");
            directory = directory.activate(left).expect("activate left");
            directory = directory.activate(right).expect("activate right");
            directory = directory
                .retire(parent, kivi_tablet::Redirect::new(vec![left, right]))
                .expect("retire parent");
            next_active.push(left);
            next_active.push(right);
        }
        active = next_active;
    }
    let placement = Placement::new(
        active
            .iter()
            .enumerate()
            .map(|(index, tablet)| (*tablet, WorkerId::from_u64((index % workers) as u64))),
    );
    (directory, placement)
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();
    let args = Args::parse();
    if args.workers == 0 {
        anyhow::bail!("workers must be nonzero");
    }
    if args.queue == 0 {
        anyhow::bail!("queue depth must be nonzero");
    }
    if let AffinityMode::Explicit(cores) = &args.pin
        && cores.len() < args.workers
    {
        tracing::warn!(
            cores = cores.len(),
            workers = args.workers,
            "fewer cores listed than workers; wrapping round-robin"
        );
    }
    let node = NodeId::from_u64(args.node);
    let cluster = ClusterId::from_u128(args.cluster);
    let levels = args.tablets.trailing_zeros();
    let (directory, placement) = even_split(levels, args.workers);
    let tablet_count = directory
        .tablets()
        .iter()
        .filter(|tablet| tablet.state().is_writable())
        .count();
    let engine = LocalEngine::start(EngineConfig {
        namespace: NS,
        directory,
        placement,
        worker_count: args.workers,
        request_capacity: args.queue,
        network: Some(EngineNetwork {
            base_port: args.port,
            ports: Vec::new(),
            bind_ip: args.bind,
            max_frame: args.max_frame,
            affinity: args.pin.clone(),
            node_id: node,
            cluster_id: cluster,
            conn: ConnLimits::default(),
            turn: TurnBudget::default(),
        }),
    })
    .context("engine failed to start")?;
    tracing::warn!("kivi-server: IN-MEMORY / NON-DURABLE build (no WAL yet)");
    tracing::info!(
        tablets = tablet_count,
        workers = args.workers,
        namespace = NS.as_u64(),
        partitioning = %PartitionHasher::V1.algorithm(),
        "serving native endpoints"
    );
    for (index, addr) in engine.worker_addrs().iter().enumerate() {
        tracing::info!(worker = index, endpoint = %addr, "worker listening");
    }
    let state = AdminState {
        node,
        cluster,
        engine: engine.admin_handle(),
    };
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .thread_name("kivi-admin")
        .build()
        .context("admin runtime failed to start")?;
    let listener = runtime
        .block_on(tokio::net::TcpListener::bind(args.admin))
        .with_context(|| format!("admin bind failed on {}", args.admin))?;
    let admin_addr = listener.local_addr().context("admin listener address")?;
    tracing::info!(endpoint = %admin_addr, "admin plane listening");
    runtime.block_on(admin::serve(listener, state, async {
        if let Err(error) = tokio::signal::ctrl_c().await {
            tracing::warn!(%error, "shutdown signal watch failed; exiting");
        }
    }))?;
    let report = engine.shutdown().context("engine shutdown failed")?;
    tracing::info!(
        workers_joined = report.workers_joined,
        "kivi-server stopped"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[test]
    fn affinity_parser_accepts_all_forms() {
        assert_eq!(parse_affinity("none"), Ok(AffinityMode::Disabled));
        assert_eq!(parse_affinity("auto"), Ok(AffinityMode::Auto));
        assert_eq!(
            parse_affinity("0,2-4,7"),
            Ok(AffinityMode::Explicit(vec![0, 2, 3, 4, 7]))
        );
        assert!(parse_affinity("x").is_err());
        assert!(parse_affinity("4-2").is_err());
        assert!(parse_affinity("").is_err());
    }

    #[test]
    fn tablet_count_parser_enforces_power_of_two() {
        assert_eq!(parse_tablets("1"), Ok(1));
        assert_eq!(parse_tablets("4"), Ok(4));
        assert!(parse_tablets("0").is_err());
        assert!(parse_tablets("3").is_err());
        assert!(parse_tablets("lots").is_err());
    }

    #[test]
    fn max_frame_parser_enforces_ceiling() {
        assert_eq!(parse_max_frame("1024"), Ok(1024));
        assert!(parse_max_frame("0").is_err());
        assert!(parse_max_frame("134217729").is_err());
    }

    #[test]
    fn server_args_parse_end_to_end() {
        let args = Args::try_parse_from([
            "kivi-server",
            "--workers",
            "4",
            "--tablets",
            "4",
            "--pin",
            "0,1",
            "--admin",
            "127.0.0.1:0",
        ])
        .expect("args parse");
        assert_eq!(args.workers, 4);
        assert_eq!(args.tablets, 4);
        assert_eq!(args.pin, AffinityMode::Explicit(vec![0, 1]));
        assert_eq!(args.admin.port(), 0);
    }

    #[test]
    fn server_args_reject_bad_input_without_panicking() {
        assert!(Args::try_parse_from(["kivi-server", "--pin", "banana"]).is_err());
        assert!(Args::try_parse_from(["kivi-server", "--tablets", "6"]).is_err());
    }
}
