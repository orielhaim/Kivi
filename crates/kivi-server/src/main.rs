//! `kivi-server`: runnable single-node Kivi server with native endpoints.
//!
//! Starts engine workers (each with its own native TCP endpoint), serves the
//! read-only admin/control HTTP plane on Tokio, and shuts down orderly on
//! Ctrl-C. Two durability modes, chosen explicitly (never by default):
//!
//! * `--ephemeral`: pure in-memory (benchmarks, development).
//! * `--data-dir <path>`: crash-recoverable local database. The directory
//!   is locked, node identity loaded or minted, the incarnation advanced,
//!   WAL lanes recovered and replayed — all before any listener binds.

mod admin;
mod cluster;
mod compound;
mod control;
mod logging;
mod redundancy_admin;
#[cfg(feature = "redis-compat")]
mod resp;
#[cfg(feature = "redis-compat")]
use std::sync::Arc;

use std::io::Write as _;
use std::net::{IpAddr, SocketAddr};
use std::path::PathBuf;

use admin::AdminState;
use anyhow::Context;
use clap::Parser;
use kivi_engine::{
    AffinityMode, ConnLimits, DurabilityMode, DurableConfig, EngineConfig, EngineNetwork,
    LocalEngine, Placement, TurnBudget,
};
use kivi_state::PartitionHasher;
use kivi_tablet::DirectorySnapshot;
use kivi_types::{
    ClusterId, NamespaceId, NodeId, NodeIncarnation, TabletEpoch, TabletId, WorkerId,
    WriteGuardGeneration,
};
use logging::{LogConfig, LogGuard};

const NS: NamespaceId = NamespaceId::from_u64(1);

/// CLI flags are inherently boolean-heavy (one per toggle); the struct
/// groups them instead of scattering subcommands.
#[allow(clippy::struct_excessive_bools)]
#[derive(Debug, Parser)]
#[command(
    name = "kivi-server",
    about = "Kivi single-node native server (ephemeral or durable)"
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
    /// Namespace physical layout: `hash` (point-lookup partitioning) or
    /// `ordered` (range scans, secondary indexes). Applies to single-node
    /// `--tablets` and cluster `--cluster-tablets` alike.
    #[arg(long, default_value = "hash", value_parser = parse_layout)]
    layout: String,
    /// CPU pinning: auto, none, or explicit cores like "0,2-4,7".
    #[arg(long, default_value = "auto", value_parser = parse_affinity)]
    pin: AffinityMode,
    /// Per-worker bounded request queue depth.
    #[arg(long, default_value_t = 1024)]
    queue: usize,
    /// Maximum accepted frame in bytes (protocol ceiling 64 MiB).
    #[arg(long, default_value_t = 64 * 1024 * 1024, value_parser = parse_max_frame)]
    pub(crate) max_frame: usize,
    /// Run without persistence (in-memory; benchmarks and development).
    /// Exactly one of `--ephemeral` and `--data-dir` is required.
    #[arg(long, conflicts_with = "data_dir")]
    ephemeral: bool,
    /// Data directory for crash-recoverable durable mode. Created with
    /// node identity and WAL lanes on first startup.
    #[arg(long, conflicts_with = "ephemeral")]
    pub(crate) data_dir: Option<PathBuf>,
    /// Node identity for handshakes (ephemeral mode only; durable mode
    /// always uses the data directory's persisted identity).
    #[arg(long, default_value_t = 1, conflicts_with = "data_dir")]
    node: u64,
    /// Cluster identity for handshakes (ephemeral mode only).
    #[arg(long, default_value_t = 1, conflicts_with = "data_dir")]
    cluster: u128,
    /// WAL segment rotation target in bytes (durable mode only).
    #[arg(long, default_value_t = kivi_durability::wal::DEFAULT_SEGMENT_TARGET_BYTES)]
    pub(crate) wal_segment_target: u64,
    /// Share one WAL lane across all workers behind a mutex instead of
    /// private per-worker lanes (group-commit experiment arm; durable
    /// mode only).
    #[arg(long)]
    shared_wal: bool,
    /// Maximum mutations per WAL batch (durable mode only). `1` reproduces
    /// immediate per-mutation durability on the same pipeline — the
    /// baseline group commit measures against.
    #[arg(long, default_value_t = kivi_engine::BatchPolicy::DEFAULT_MAX_OPS)]
    batch_max_ops: usize,
    /// Maximum encoded record bytes per WAL batch (durable mode only).
    #[arg(long, default_value_t = kivi_engine::BatchPolicy::DEFAULT_MAX_BYTES)]
    batch_max_bytes: u64,
    /// Maximum age of the oldest unsealed batch entry in microseconds
    /// before forced seal (durable mode only).
    #[arg(long, default_value_t = 500)]
    batch_max_wait_us: u64,
    /// Drained-batch linger in microseconds: how long a non-full batch
    /// waits for concurrent arrivals before sealing (durable mode only).
    #[arg(long, default_value_t = kivi_engine::BatchPolicy::DEFAULT_LINGER.as_micros() as u64)]
    batch_linger_us: u64,
    /// WAL bytes per worker since its last checkpoint that trigger an
    /// automatic checkpoint (durable mode only).
    #[arg(long, default_value_t = kivi_engine::CheckpointConfig::DEFAULT_WAL_BYTES)]
    checkpoint_wal_bytes: u64,
    /// Applied commits per tablet since its last cut that trigger an
    /// automatic checkpoint (durable mode only).
    #[arg(long, default_value_t = kivi_engine::CheckpointConfig::DEFAULT_MUTATIONS)]
    checkpoint_mutations: u64,
    /// Minimum checkpoint age in seconds before the elapsed-time trigger
    /// may fire, and only with dirty state (durable mode only).
    #[arg(long, default_value_t = 300)]
    checkpoint_interval_secs: u64,
    /// Disable automatic checkpoints (manual triggers still work).
    /// Benchmarks use this to isolate commit-pipeline variables.
    #[arg(long)]
    no_checkpoint: bool,
    /// Values strictly larger stage as chunked roots through per-worker
    /// chunk lanes; the rest stay inline. Physical only: logical `Bytes`
    /// semantics never change. Both durability modes.
    #[arg(long, default_value_t = kivi_engine::ChunkFabricConfig::DEFAULT_INLINE_THRESHOLD)]
    inline_threshold: u64,
    /// Chunk pack rotation target in bytes per lane (tests use small
    /// targets to exercise rotation; production stays large).
    #[arg(long, default_value_t = kivi_engine::ChunkFabricConfig::DEFAULT_PACK_TARGET_BYTES)]
    chunk_pack_target: u64,
    /// Chunk-cache bound in bytes per lane (immutable entries, safe
    /// eviction at any time).
    #[arg(long, default_value_t = kivi_engine::ChunkFabricConfig::DEFAULT_CACHE_BYTES)]
    chunk_cache_bytes: u64,
    /// DRAM arena bound in bytes per worker for medium values (both
    /// durability modes). Smaller bounds demote cold data to `NVMe`
    /// sooner; the commit path fails fast with overload past it.
    #[arg(long, default_value_t = kivi_engine::FabricConfig::DEFAULT_ARENA_BYTES)]
    fabric_arena_bytes: u64,
    /// Demotion capacity in bytes per worker (optimization copies to
    /// `NVMe`; durable records live in the material file regardless).
    #[arg(long, default_value_t = kivi_engine::FabricConfig::DEFAULT_DEMOTION_BYTES)]
    fabric_demotion_bytes: u64,
    /// Admin/control HTTP listen address (loopback by default; `:0` selects
    /// an ephemeral port and prints it).
    #[arg(long, default_value = "127.0.0.1:19080")]
    pub(crate) admin: SocketAddr,
    /// Optional Redis/RESP compatibility listen address (feature
    /// `redis-compat` only). Absent means native-only: no RESP service
    /// starts even when compiled in. `:0` selects an ephemeral port.
    #[cfg(feature = "redis-compat")]
    #[arg(long)]
    pub(crate) redis_listen: Option<SocketAddr>,
    /// Namespace id the RESP frontend serves (Redis DB 0 maps here).
    /// Single-namespace stage: must equal the engine namespace (1).
    #[cfg(feature = "redis-compat")]
    #[arg(long, default_value_t = 1)]
    redis_namespace: u64,
    /// Enable replicated cluster mode: one statically configured tablet
    /// group instead of single-node workers. Requires `--data-dir`.
    #[arg(long = "cluster-mode")]
    cluster_mode: bool,
    /// Cluster identity for cluster mode.
    #[arg(long)]
    pub(crate) cluster_id: Option<u128>,
    /// This process's node identity for cluster mode.
    #[arg(long)]
    pub(crate) node_id: Option<u64>,
    /// Tablet count in cluster mode: the static directory tiles the whole
    /// hash space with this many tablets (any nonzero count; splits halve
    /// the largest piece, so non-powers of two mix prefix lengths).
    #[arg(long, default_value_t = 1, value_parser = parse_cluster_tablets)]
    pub(crate) cluster_tablets: usize,
    /// Consensus worker (Compio reactor) count in cluster mode: tablet
    /// groups stripe across these deterministically (`(tablet-1) % workers`
    /// on tablet ids). Thread count scales with workers, never tablets.
    #[arg(long, default_value_t = 2, value_parser = parse_cluster_workers)]
    pub(crate) cluster_workers: usize,
    /// Static peer endpoints `node=host:port,...` for cluster mode.
    #[arg(long, value_delimiter = ',', value_parser = parse_node_endpoint)]
    pub(crate) cluster_peers: Vec<(u64, SocketAddr)>,
    /// Static native endpoints `node=host:port,...` for cluster mode
    /// (client redirect targets; required, and distinct from the peer
    /// endpoints).
    #[arg(long, value_delimiter = ',', value_parser = parse_node_endpoint)]
    pub(crate) cluster_natives: Vec<(u64, SocketAddr)>,
    /// Native client listen address in cluster mode (`:0` ephemeral).
    #[arg(long, default_value = "127.0.0.1:9000")]
    pub(crate) cluster_native: SocketAddr,
    /// Admin endpoints `node=host:port,...` for cluster mode
    /// (reconciler forwarding targets; required, and distinct per
    /// node — the registry records each node's own admin endpoint,
    /// never the local flag).
    #[arg(long, value_delimiter = ',', value_parser = parse_node_endpoint)]
    pub(crate) cluster_admins: Vec<(u64, SocketAddr)>,
    /// Static peer TLS certificates `node=path/to/cert.der,...` for
    /// cluster mode (each file the peer's DER certificate as printed by
    /// `print-peer-cert`). Absent means `KIVI_INSECURE_PEER_TLS=1` must be
    /// set (lab tests only, never production).
    #[arg(long, value_delimiter = ',', value_parser = parse_node_cert_path)]
    pub(crate) cluster_peer_certs: Vec<(u64, PathBuf)>,
    /// Control-plane voter ids for cluster mode (comma-separated).
    /// Absent means the three lowest `--cluster-peers` ids: the initial
    /// product keeps the first three nodes as control voters while added
    /// data nodes join the control group as learners (control state for
    /// routing, no vote). Restarting nodes ignore this (durable control
    /// membership is truth, flags are seeds).
    #[arg(long, value_delimiter = ',', value_parser = parse_node_u64)]
    pub(crate) control_voters: Vec<u64>,
    /// Control-plane seed endpoints (`host:port,...`) proving the
    /// cluster already exists. Absent on founding voters (a fresh
    /// directory initializes the control group); REQUIRED on a joining
    /// node (a fresh directory waits for `add_learner` instead of
    /// forking history with a second initialize). Ignored on restart
    /// (durable control state is truth).
    #[arg(long, value_delimiter = ',')]
    pub(crate) control_seeds: Vec<SocketAddr>,
}

/// Parses one control-voter node id.
fn parse_node_u64(spec: &str) -> Result<u64, String> {
    let node: u64 = spec
        .trim()
        .parse()
        .map_err(|_| format!("invalid node id in {spec:?}"))?;
    if node == 0 {
        return Err(format!("node id 0 is reserved, got {spec:?}"));
    }
    Ok(node)
}

/// Parses one `node=path/to/cert.der` peer certificate mapping.
fn parse_node_cert_path(spec: &str) -> Result<(u64, PathBuf), String> {
    let (node, path) = spec
        .split_once('=')
        .ok_or_else(|| format!("peer cert mapping must be node=path, got {spec:?}"))?;
    let node: u64 = node
        .trim()
        .parse()
        .map_err(|_| format!("invalid node id in {spec:?}"))?;
    if node == 0 {
        return Err(format!("node id 0 is reserved, got {spec:?}"));
    }
    Ok((node, PathBuf::from(path.trim())))
}

/// Parses one `node=host:port` endpoint mapping.
fn parse_node_endpoint(spec: &str) -> Result<(u64, SocketAddr), String> {
    let (node, addr) = spec
        .split_once('=')
        .ok_or_else(|| format!("endpoint mapping must be node=host:port, got {spec:?}"))?;
    let node: u64 = node
        .trim()
        .parse()
        .map_err(|_| format!("invalid node id in {spec:?}"))?;
    if node == 0 {
        return Err(format!("node id 0 is reserved, got {spec:?}"));
    }
    let addr: SocketAddr = addr
        .trim()
        .parse()
        .map_err(|_| format!("invalid socket address in {spec:?}"))?;
    Ok((node, addr))
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

/// Parses the cluster tablet count: any nonzero count (static buddy
/// tiling covers non-powers of two with mixed prefix lengths).
fn parse_cluster_tablets(spec: &str) -> Result<usize, String> {
    let count: usize = spec
        .parse()
        .map_err(|_| format!("invalid cluster tablet count {spec:?}"))?;
    if count == 0 {
        return Err("cluster tablet count must be nonzero".to_owned());
    }
    Ok(count)
}

/// Parses the cluster worker count: any nonzero count.
fn parse_cluster_workers(spec: &str) -> Result<usize, String> {
    let count: usize = spec
        .parse()
        .map_err(|_| format!("invalid cluster worker count {spec:?}"))?;
    if count == 0 {
        return Err("cluster worker count must be nonzero".to_owned());
    }
    Ok(count)
}

/// Parses the tablet count: any nonzero count (hash tilings mix prefix
/// lengths for non-powers of two; ordered tilings spread first-byte
/// boundaries uniformly).
fn parse_tablets(spec: &str) -> Result<usize, String> {
    let count: usize = spec
        .parse()
        .map_err(|_| format!("invalid tablet count {spec:?}"))?;
    if count == 0 {
        return Err("tablet count must be nonzero".to_owned());
    }
    Ok(count)
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

/// Parses the namespace layout flag.
fn parse_layout(spec: &str) -> Result<String, String> {
    match spec {
        "hash" | "ordered" => Ok(spec.to_owned()),
        _ => Err(format!(
            "layout must be \"hash\" or \"ordered\", got {spec:?}"
        )),
    }
}

/// Builds an ordered directory with `count` active tablets plus a
/// round-robin placement. Initial boundaries spread the first key byte
/// uniformly (`count` tablets → boundaries at `i * 256 / count`); the
/// auto-split policy refines by data from there.
fn even_split_ordered(count: usize, workers: usize) -> (DirectorySnapshot, Placement) {
    assert!(count >= 1, "ordered tiling needs at least one tablet");
    let mut split_keys = Vec::new();
    for index in 1..count {
        #[allow(clippy::cast_possible_truncation)]
        let boundary = (index * 256 / count) as u8;
        // Boundaries must be strictly increasing and nonzero (the first
        // tile starts at empty): dedup guards degenerate counts.
        if boundary != 0
            && split_keys
                .last()
                .is_none_or(|last: &Vec<u8>| *last != vec![boundary])
        {
            split_keys.push(vec![boundary]);
        }
    }
    let directory =
        DirectorySnapshot::static_ordered_tiles(NS, &split_keys).expect("ordered tiling builds");
    let active: Vec<TabletId> = directory
        .tablets()
        .iter()
        .filter(|tablet| tablet.state().is_writable())
        .map(kivi_tablet::TabletDescriptor::id)
        .collect();
    let placement = Placement::new(
        active
            .iter()
            .enumerate()
            .map(|(index, tablet)| (*tablet, WorkerId::from_u64((index % workers) as u64))),
    );
    (directory, placement)
}

/// Opens the durability layer: explicit mode choice, data-directory lock,
/// identity, and incarnation advance — all before any listener binds.
/// Returns node/cluster/incarnation, the engine durability mode, and the
/// lock guard (held for the whole process).
fn open_durability(
    args: &Args,
) -> anyhow::Result<(
    NodeId,
    ClusterId,
    NodeIncarnation,
    DurabilityMode,
    Option<kivi_durability::OpenDir>,
)> {
    match (&args.ephemeral, &args.data_dir) {
        (true, None) => {
            tracing::warn!(
                "kivi-server: EPHEMERAL mode (no WAL; acknowledged writes die with the process)"
            );
            Ok((
                NodeId::from_u64(args.node),
                ClusterId::from_u128(args.cluster),
                NodeIncarnation::INITIAL,
                DurabilityMode::Ephemeral,
                None,
            ))
        }
        (false, Some(dir)) => {
            // Lock, identity, incarnation advance, WAL root — all before
            // any listener binds (see `open_data_dir`).
            let opened = kivi_durability::open_data_dir(dir)
                .with_context(|| format!("open data directory {}", dir.display()))?;
            let (node, cluster, incarnation) = (
                opened.meta.node,
                opened.meta.cluster,
                opened.meta.incarnation,
            );
            tracing::info!(
                mode = "durable",
                dir = %dir.display(),
                node = node.as_u64(),
                cluster = cluster.as_u128(),
                incarnation = incarnation.as_u64(),
                fresh = opened.fresh,
                "durable mode: data directory open"
            );
            let batch = kivi_engine::BatchPolicy {
                max_ops: args.batch_max_ops,
                max_bytes: args.batch_max_bytes,
                max_wait: std::time::Duration::from_micros(args.batch_max_wait_us),
                linger: std::time::Duration::from_micros(args.batch_linger_us),
            };
            batch
                .validate()
                .map_err(|reason| anyhow::anyhow!("invalid batch policy: {reason}"))?;
            let checkpoint = kivi_engine::CheckpointConfig {
                wal_bytes_threshold: args.checkpoint_wal_bytes,
                mutations_threshold: args.checkpoint_mutations,
                min_interval: std::time::Duration::from_secs(args.checkpoint_interval_secs),
                policy: kivi_checkpoint::CheckpointPolicy::DEFAULT,
                disabled: args.no_checkpoint,
            };
            let durability = DurabilityMode::Durable(DurableConfig {
                data_dir: dir.clone(),
                segment_target_bytes: args.wal_segment_target,
                node,
                cluster,
                incarnation,
                shared_wal: args.shared_wal,
                batch,
                checkpoint,
            });
            Ok((node, cluster, incarnation, durability, Some(opened)))
        }
        _ => {
            anyhow::bail!(
                "choose exactly one durability mode: --ephemeral (in-memory) or --data-dir <path> (crash-recoverable)"
            );
        }
    }
}

/// Coherent startup modes: exactly one is active per process. The CLI
/// exposes many flags, but only one of these states is valid; contradictory
/// combinations fail loudly before any listener binds or state recovers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum StartupMode {
    /// Pure in-memory single node (benchmarks, development).
    EphemeralSingle,
    /// Crash-recoverable single node.
    DurableSingle,
    /// Replicated cluster member (one tablet group, peer mesh).
    Cluster,
}

/// Resolves the startup mode and rejects contradictory flag combinations
/// early (before locks, recovery, or binds).
fn startup_mode(args: &Args) -> anyhow::Result<StartupMode> {
    if args.cluster_mode {
        if args.ephemeral {
            anyhow::bail!("--cluster-mode and --ephemeral are mutually exclusive");
        }
        if args.data_dir.is_none() {
            anyhow::bail!("cluster mode requires --data-dir <path>");
        }
        if args.cluster_id.is_none() {
            anyhow::bail!("cluster mode requires --cluster-id <u128>");
        }
        if args.node_id.is_none() {
            anyhow::bail!("cluster mode requires --node-id <u64>");
        }
        // Cluster mode honors --cluster-tablets/--cluster-workers (one
        // replica of every tablet per process); single-node --workers and
        // --tablets do not apply here.
        if args.workers != 2 || args.tablets != 1 {
            tracing::warn!(
                workers = args.workers,
                tablets = args.tablets,
                "cluster mode uses --cluster-tablets/--cluster-workers, not --workers/--tablets"
            );
        }
        return Ok(StartupMode::Cluster);
    }
    match (&args.ephemeral, &args.data_dir) {
        (true, None) => Ok(StartupMode::EphemeralSingle),
        (false, Some(_)) => Ok(StartupMode::DurableSingle),
        _ => anyhow::bail!(
            "choose exactly one durability mode: --ephemeral (in-memory) or --data-dir <path> (crash-recoverable)"
        ),
    }
}

#[allow(clippy::too_many_lines)]
fn main() -> anyhow::Result<()> {
    // Bounded production logging: size-rotated files behind a non-blocking
    // worker (see `logging.rs` for the crate decision). Default keeps
    // Kivi's own `info` narration but quiets `openraft` election/tick spam
    // (`openraft=warn`; ~300k lines/2min at 1000 groups otherwise).
    // The guard lives for the full process so shutdown flushes.
    let _log_guard = LogGuard::init(&LogConfig::from_env());
    let args = Args::parse();
    // Replicated cluster mode bypasses the single-node engine entirely:
    // one tablet group, peer mesh, native/admin/RESP edges on Tokio.
    match startup_mode(&args)? {
        StartupMode::Cluster => return cluster::run_from_args(&args),
        StartupMode::EphemeralSingle | StartupMode::DurableSingle => {}
    }
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
    // The lock guard lives for the whole process: dropping it would
    // release the data directory to a second process mid-run.
    let (node, cluster, incarnation, durability, _data_dir_guard) = open_durability(&args)?;
    let (directory, placement) = if args.layout == "ordered" {
        even_split_ordered(args.tablets.max(1), args.workers)
    } else {
        let directory = DirectorySnapshot::static_tiles(
            NS,
            args.tablets.max(1),
            TabletEpoch::INITIAL,
            WriteGuardGeneration::INITIAL,
        )
        .expect("hash tiling builds");
        let active: Vec<TabletId> = directory
            .tablets()
            .iter()
            .filter(|tablet| tablet.state().is_writable())
            .map(kivi_tablet::TabletDescriptor::id)
            .collect();
        let placement =
            Placement::new(active.iter().enumerate().map(|(index, tablet)| {
                (*tablet, WorkerId::from_u64((index % args.workers) as u64))
            }));
        (directory, placement)
    };
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
        chunks: kivi_engine::ChunkFabricConfig {
            inline_threshold: args.inline_threshold,
            pack_target_bytes: args.chunk_pack_target,
            cache_bytes: args.chunk_cache_bytes,
        },
        fabric: kivi_engine::FabricConfig {
            arena_bytes_per_worker: args.fabric_arena_bytes,
            demotion_bytes_per_worker: args.fabric_demotion_bytes,
        },
        network: Some(EngineNetwork {
            base_port: args.port,
            ports: Vec::new(),
            bind_ip: args.bind,
            max_frame: args.max_frame,
            affinity: args.pin.clone(),
            node_id: node,
            cluster_id: cluster,
            incarnation,
            conn: ConnLimits::default(),
            turn: TurnBudget::default(),
        }),
        durability,
    })
    .context("engine failed to start")?;
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
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .thread_name("kivi-admin")
        .build()
        .context("admin runtime failed to start")?;
    // Optional RESP frontend: bound before the ready line prints so the
    // reported endpoint is authoritative (including `:0` ephemeral ports).
    // Absent `--redis-listen` means native-only even when compiled in.
    // The shutdown sender outlives `admin::serve`; dropping it afterwards
    // trips the frontend's watch so RESP tasks exit before the engine.
    #[cfg(feature = "redis-compat")]
    let (resp_admin, resp_shutdown): (
        Option<resp::RespAdmin>,
        Option<tokio::sync::watch::Sender<bool>>,
    ) = match args.redis_listen {
        None => (None, None),
        Some(addr) => {
            if args.redis_namespace != NS.as_u64() {
                anyhow::bail!(
                    "single-namespace stage serves namespace 1; --redis-namespace {} has no route",
                    args.redis_namespace
                );
            }
            let stats = std::sync::Arc::new(resp::RespStats::default());
            let listener = runtime
                .block_on(tokio::net::TcpListener::bind(addr))
                .with_context(|| format!("redis bind failed on {addr}"))?;
            let endpoint = listener.local_addr().context("redis listener address")?;
            tracing::info!(endpoint = %endpoint, namespace = args.redis_namespace, "resp frontend listening");
            let (shutdown_tx, shutdown_rx) = tokio::sync::watch::channel(false);
            runtime.spawn(resp::serve(
                listener,
                engine.client(),
                Arc::clone(&stats),
                shutdown_rx,
            ));
            (
                Some(resp::RespAdmin {
                    endpoint,
                    namespace: args.redis_namespace,
                    stats,
                }),
                Some(shutdown_tx),
            )
        }
    };
    let state = AdminState {
        node,
        cluster,
        incarnation,
        engine: engine.admin_handle(),
        #[cfg(feature = "redis-compat")]
        resp: resp_admin.clone(),
    };
    let listener = runtime
        .block_on(tokio::net::TcpListener::bind(args.admin))
        .with_context(|| format!("admin bind failed on {}", args.admin))?;
    let admin_addr = listener.local_addr().context("admin listener address")?;
    tracing::info!(endpoint = %admin_addr, "admin plane listening");
    // Machine-readable startup report for process supervisors and the
    // integration-test harness: the actual bound endpoints (authoritative
    // when `--port 0` / `--admin 127.0.0.1:0` select ephemeral ports).
    // One line, fixed shape, printed once. Workers are listed in worker
    // index order, so the first entry is the seed endpoint.
    //
    // ```text
    // KIVI_READY workers=127.0.0.1:9000,127.0.0.1:9001 admin=127.0.0.1:19080
    // KIVI_READY workers=... admin=... redis=127.0.0.1:6379 (with --redis-listen)
    // ```
    let workers = engine
        .worker_addrs()
        .iter()
        .map(std::string::ToString::to_string)
        .collect::<Vec<_>>()
        .join(",");
    #[cfg(feature = "redis-compat")]
    {
        match &state.resp {
            Some(resp) => {
                println!(
                    "KIVI_READY workers={workers} admin={admin_addr} redis={}",
                    resp.endpoint
                );
            }
            None => {
                println!("KIVI_READY workers={workers} admin={admin_addr}");
            }
        }
    }
    #[cfg(not(feature = "redis-compat"))]
    {
        println!("KIVI_READY workers={workers} admin={admin_addr}");
    }
    let _ = std::io::stdout().flush();
    runtime.block_on(admin::serve(listener, state, async {
        if let Err(error) = tokio::signal::ctrl_c().await {
            tracing::warn!(%error, "shutdown signal watch failed; exiting");
        }
    }))?;
    #[cfg(feature = "redis-compat")]
    drop(resp_shutdown);
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
    fn tablet_count_parser_accepts_any_nonzero_count() {
        assert_eq!(parse_tablets("1"), Ok(1));
        assert_eq!(parse_tablets("4"), Ok(4));
        assert_eq!(parse_tablets("3"), Ok(3));
        assert!(parse_tablets("0").is_err());
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
        assert!(Args::try_parse_from(["kivi-server", "--tablets", "0"]).is_err());
        assert!(Args::try_parse_from(["kivi-server", "--layout", "b-tree"]).is_err());
    }

    #[test]
    fn layout_flag_selects_ordered_tiling() {
        let args = Args::try_parse_from(["kivi-server", "--layout", "ordered", "--tablets", "4"])
            .expect("args parse");
        assert_eq!(args.layout, "ordered");
        let (directory, _) = even_split_ordered(4, 2);
        assert!(directory.ordered_coverage_complete());
        assert_eq!(parse_layout("hash").expect("hash"), "hash");
        assert_eq!(parse_layout("ordered").expect("ordered"), "ordered");
    }

    #[test]
    fn cluster_tablet_count_accepts_any_nonzero_count() {
        // Unlike single-node even splits, static buddy tiling covers
        // non-powers of two with mixed prefix lengths.
        assert_eq!(parse_cluster_tablets("1"), Ok(1));
        assert_eq!(parse_cluster_tablets("16"), Ok(16));
        assert_eq!(parse_cluster_tablets("100"), Ok(100));
        assert!(parse_cluster_tablets("0").is_err());
        assert!(parse_cluster_tablets("lots").is_err());
    }

    #[test]
    fn cluster_worker_count_must_be_nonzero() {
        assert_eq!(parse_cluster_workers("1"), Ok(1));
        assert_eq!(parse_cluster_workers("4"), Ok(4));
        assert!(parse_cluster_workers("0").is_err());
        assert!(parse_cluster_workers("many").is_err());
    }

    #[test]
    fn cluster_args_parse_end_to_end() {
        let args = Args::try_parse_from([
            "kivi-server",
            "--cluster-mode",
            "--cluster-id",
            "12345",
            "--node-id",
            "1",
            "--cluster-tablets",
            "16",
            "--cluster-workers",
            "4",
            "--data-dir",
            "./data-a",
            "--cluster-peers",
            "1=127.0.0.1:9101",
            "--cluster-natives",
            "1=127.0.0.1:9201",
        ])
        .expect("cluster args parse");
        assert_eq!(args.cluster_tablets, 16);
        assert_eq!(args.cluster_workers, 4);
    }
}
