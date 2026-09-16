//! `kivi-cli`: development/operator client for the native protocol.
//!
//! Thin typed wrapper over [`NativeClient`]: parses arguments, prints
//! human-readable results, and exits nonzero with the stable machine error
//! on failure. Not Redis compatibility — native semantics only.

use bytes::Bytes;
use clap::{Parser, Subcommand};
use kivi_client::{ClientConfig, NativeClient};
use kivi_state::Key;
use kivi_types::{NamespaceId, UnixMicros};

#[derive(Debug, Parser)]
#[command(name = "kivi-cli", about = "Native Kivi development client")]
struct Cli {
    /// Seed endpoint (worker 0 doubles as the seed initially).
    #[arg(long, default_value = "127.0.0.1:9000")]
    server: String,
    /// Extra seed endpoints for clusters (comma-separated or repeated).
    /// The client starts at `--server` plus these, follows leader hints,
    /// and preserves mutation identities across redirects.
    #[arg(long, value_delimiter = ',')]
    seed: Vec<String>,
    /// Target namespace id.
    #[arg(long, default_value_t = 1)]
    namespace: u64,
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Fetch bytes.
    Get {
        /// Key to read.
        key: String,
    },
    /// Store bytes.
    Set {
        /// Key to write.
        key: String,
        /// Value to store.
        value: String,
    },
    /// Remove a key.
    Del {
        /// Key to remove.
        key: String,
    },
    /// Test for a live value.
    Exists {
        /// Key to probe.
        key: String,
    },
    /// Read a counter.
    CounterGet {
        /// Key to read.
        key: String,
    },
    /// Add to a counter.
    CounterAdd {
        /// Key to update.
        key: String,
        /// Signed addend.
        delta: i64,
    },
    /// Attach an absolute expiry (microseconds since the Unix epoch).
    ExpireAt {
        /// Key to expire.
        key: String,
        /// Expiration timestamp in microseconds.
        micros: u64,
    },
    /// Clear any expiry.
    Persist {
        /// Key to persist.
        key: String,
    },
    /// Show seconds until expiry (`none` when absent or immortal).
    Ttl {
        /// Key to inspect.
        key: String,
    },
    /// Read a byte slice `[offset, offset + len)`.
    GetRange {
        /// Key to slice.
        key: String,
        /// Start offset.
        offset: u64,
        /// Maximum bytes.
        len: u64,
    },
    /// Report the logical byte length (`(nil)` when absent).
    BytesLength {
        /// Key to measure.
        key: String,
    },
    /// Conditionally store bytes (`NX`/`XX`/always with expiry policy).
    SetConditional {
        /// Key to write.
        key: String,
        /// Value to store when the condition holds.
        value: String,
        /// Presence condition: `always`, `nx` (absent), `xx` (present).
        #[arg(long, default_value = "always")]
        condition: String,
        /// Expiry policy: `clear`, `keep`, or `at:<micros>`.
        #[arg(long, default_value = "clear")]
        expiry: String,
    },
    /// Cluster control-plane operations (node admission, placement,
    /// migration) against a node's admin endpoint.
    Cluster {
        /// Admin endpoint of any cluster node.
        #[arg(long, default_value = "127.0.0.1:19080")]
        admin: String,
        #[command(subcommand)]
        command: ClusterCommand,
    },
}

/// Cluster operator actions. These speak to the admin plane (typed
/// control-plane API), never raw Raft messages.
#[derive(Debug, Subcommand)]
enum ClusterCommand {
    /// Show cluster readiness, nodes, and migration progress.
    Status,
    /// List admitted nodes and their lifecycle states.
    NodeList,
    /// Admit a node (`Joining` then `Active` after commit).
    NodeAdd {
        /// New node id.
        id: u64,
        /// Its consensus (H3 peer) endpoint.
        #[arg(long)]
        peer: String,
        /// Its native client endpoint.
        #[arg(long)]
        native: String,
        /// Its admin endpoint.
        #[arg(long)]
        admin_node: String,
        /// Certificate fingerprint (64 hex chars); absent means
        /// unenforced (lab-insecure only).
        #[arg(long)]
        fingerprint: Option<String>,
        /// Failure-domain label (zone/rack/host-group).
        #[arg(long, default_value = "")]
        domain: String,
    },
    /// Drain a node: migrate every replica away, then mark `Drained`.
    NodeDrain {
        /// Node id.
        id: u64,
    },
    /// Remove a safely drained node (fails loudly otherwise).
    NodeRemove {
        /// Node id.
        id: u64,
    },
    /// List tablets with desired vs actual placement.
    TabletList,
    /// Show one tablet's membership and desired placement.
    TabletShow {
        /// Tablet id.
        id: u64,
    },
    /// Move one tablet replica through the persisted plan pathway.
    TabletMove {
        /// Tablet id.
        #[arg(long)]
        tablet: u64,
        /// Replica leaving.
        #[arg(long)]
        from: u64,
        /// Replica joining (must be `Active`).
        #[arg(long)]
        to: u64,
    },
    /// Compute desired placements and create bounded migration plans.
    Rebalance,
    /// Rebalance tablet leadership with graceful handoffs (explicit,
    /// never continuous).
    Leadership,
    /// List migration plans.
    Migrations,
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    if let Command::Cluster { admin, command } = &cli.command {
        return run_cluster(admin, command);
    }
    let mut seeds = vec![cli.server.clone()];
    seeds.extend(cli.seed.iter().cloned());
    let client = NativeClient::new(ClientConfig {
        seeds,
        namespace: NamespaceId::from_u64(cli.namespace),
        ..ClientConfig::default()
    })?;
    run(&client, &cli.command)?;
    Ok(())
}

/// Minimal synchronous admin HTTP client (Content-Length framing both
/// ways; the admin plane always replies with one JSON body).
fn admin_roundtrip(
    admin: &str,
    method: &str,
    path: &str,
    body: Option<&[u8]>,
) -> anyhow::Result<serde_json::Value> {
    use std::io::{Read, Write};
    let mut stream = std::net::TcpStream::connect(admin)?;
    let len = body.map_or(0, <[u8]>::len);
    let head = format!(
        "{method} {path} HTTP/1.1\r\nhost: {admin}\r\ncontent-type: application/json\r\ncontent-length: {len}\r\nconnection: close\r\n\r\n"
    );
    stream.write_all(head.as_bytes())?;
    if let Some(body) = body {
        stream.write_all(body)?;
    }
    let mut raw = Vec::new();
    stream.read_to_end(&mut raw)?;
    let header_end = raw
        .windows(4)
        .position(|window| window == b"\r\n\r\n")
        .ok_or_else(|| anyhow::anyhow!("admin reply without headers"))?;
    let body = &raw[header_end + 4..];
    Ok(serde_json::from_slice(body)?)
}

/// Runs one cluster operator command against the admin plane.
#[allow(clippy::too_many_lines)]
fn run_cluster(admin: &str, command: &ClusterCommand) -> anyhow::Result<()> {
    match command {
        ClusterCommand::Status => {
            let ready: serde_json::Value = admin_roundtrip(admin, "GET", "/ready", None)?;
            let control: serde_json::Value = admin_roundtrip(admin, "GET", "/v1/control", None)?;
            println!(
                "ready: {}",
                ready
                    .get("ready")
                    .and_then(serde_json::Value::as_bool)
                    .unwrap_or(false)
            );
            println!(
                "tablets: {}/{} healthy, {} leaders known",
                ready
                    .get("tablets_healthy")
                    .and_then(serde_json::Value::as_u64)
                    .unwrap_or(0),
                ready
                    .get("tablets_total")
                    .and_then(serde_json::Value::as_u64)
                    .unwrap_or(0),
                ready
                    .get("leaders_known")
                    .and_then(serde_json::Value::as_u64)
                    .unwrap_or(0),
            );
            if control
                .get("present")
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(false)
            {
                println!(
                    "placement version: {}",
                    control
                        .get("placement_version")
                        .and_then(serde_json::Value::as_u64)
                        .unwrap_or(0)
                );
                let nodes = control
                    .get("nodes")
                    .and_then(serde_json::Value::as_array)
                    .cloned()
                    .unwrap_or_default();
                let mut active = 0;
                let mut draining = 0;
                for node in &nodes {
                    match node
                        .get("state")
                        .and_then(serde_json::Value::as_str)
                        .unwrap_or("?")
                    {
                        "Active" => active += 1,
                        "Draining" | "Drained" => draining += 1,
                        _ => {}
                    }
                }
                println!(
                    "nodes: {} active, {} draining/drained ({} total)",
                    active,
                    draining,
                    nodes.len()
                );
                let live = control
                    .get("migrations")
                    .and_then(serde_json::Value::as_array)
                    .map_or(0, |plans| {
                        plans
                            .iter()
                            .filter(|plan| {
                                !matches!(
                                    plan.get("phase").and_then(serde_json::Value::as_str),
                                    Some("Completed" | "Failed")
                                )
                            })
                            .count()
                    });
                println!("migrations live: {live}");
            } else {
                println!("control plane: not present on this node");
            }
            Ok(())
        }
        ClusterCommand::NodeList => {
            let control: serde_json::Value = admin_roundtrip(admin, "GET", "/v1/control", None)?;
            let nodes = control
                .get("nodes")
                .and_then(serde_json::Value::as_array)
                .cloned()
                .unwrap_or_default();
            println!(
                "{:<6} {:<21} {:<21} {:<10} domain",
                "node", "peer", "native", "state"
            );
            for node in nodes {
                println!(
                    "{:<6} {:<21} {:<21} {:<10} {}",
                    node.get("node")
                        .and_then(serde_json::Value::as_u64)
                        .unwrap_or(0),
                    node.get("peer")
                        .and_then(serde_json::Value::as_str)
                        .unwrap_or("?"),
                    node.get("native")
                        .and_then(serde_json::Value::as_str)
                        .unwrap_or("?"),
                    node.get("state")
                        .and_then(serde_json::Value::as_str)
                        .unwrap_or("?"),
                    node.get("domain")
                        .and_then(serde_json::Value::as_str)
                        .unwrap_or(""),
                );
            }
            Ok(())
        }
        ClusterCommand::NodeAdd {
            id,
            peer,
            native,
            admin_node,
            fingerprint,
            domain,
        } => {
            let mut body = serde_json::json!({
                "node": id,
                "peer": peer,
                "native": native,
                "admin": admin_node,
                "domain": domain,
            });
            if let Some(fingerprint) = fingerprint {
                body["fingerprint"] = serde_json::Value::String(fingerprint.clone());
            }
            let raw = serde_json::to_vec(&body)?;
            let reply = admin_roundtrip(admin, "POST", "/v1/control/nodes", Some(&raw))?;
            if reply
                .get("ok")
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(false)
            {
                println!("admitted node {id}");
                Ok(())
            } else {
                anyhow::bail!("node add refused: {reply}");
            }
        }
        ClusterCommand::NodeDrain { id } => {
            let reply = admin_roundtrip(
                admin,
                "POST",
                &format!("/v1/control/nodes/{id}/drain"),
                Some(b"{}"),
            )?;
            if reply
                .get("ok")
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(false)
            {
                println!(
                    "draining node {id}; plans: {}",
                    reply
                        .get("plans")
                        .map_or_else(String::new, ToString::to_string)
                );
                Ok(())
            } else {
                anyhow::bail!("node drain refused: {reply}");
            }
        }
        ClusterCommand::NodeRemove { id } => {
            let reply = admin_roundtrip(
                admin,
                "POST",
                &format!("/v1/control/nodes/{id}/remove"),
                Some(b"{}"),
            )?;
            if reply
                .get("ok")
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(false)
            {
                println!("removed node {id}");
                Ok(())
            } else {
                anyhow::bail!("node remove refused: {reply}");
            }
        }
        ClusterCommand::TabletList => {
            let tablets: serde_json::Value = admin_roundtrip(admin, "GET", "/v1/tablets", None)?;
            let tablets = tablets.as_array().cloned().unwrap_or_default();
            println!(
                "{:<8} {:<8} {:<10} {:<24} voters",
                "tablet", "role", "placement", "desired"
            );
            for tablet in tablets {
                println!(
                    "{:<8} {:<8} {:<10} {:<24} {}",
                    tablet
                        .get("group")
                        .and_then(serde_json::Value::as_u64)
                        .unwrap_or(0),
                    tablet
                        .get("role")
                        .and_then(serde_json::Value::as_str)
                        .unwrap_or("?"),
                    tablet
                        .get("placement")
                        .and_then(serde_json::Value::as_str)
                        .unwrap_or("?"),
                    tablet
                        .get("desired")
                        .map_or_else(|| String::from("-"), ToString::to_string),
                    tablet
                        .get("voters")
                        .map_or_else(|| String::from("-"), ToString::to_string),
                );
            }
            Ok(())
        }
        ClusterCommand::TabletShow { id } => {
            let membership: serde_json::Value =
                admin_roundtrip(admin, "GET", &format!("/v1/tablets/{id}/membership"), None)?;
            println!("membership: {membership}");
            let control: serde_json::Value = admin_roundtrip(admin, "GET", "/v1/control", None)?;
            if let Some(placements) = control
                .get("placements")
                .and_then(serde_json::Value::as_array)
            {
                for placement in placements {
                    if placement.get("tablet").and_then(serde_json::Value::as_u64) == Some(*id) {
                        println!("desired: {placement}");
                    }
                }
            }
            Ok(())
        }
        ClusterCommand::TabletMove { tablet, from, to } => {
            let body = serde_json::json!({ "tablet": tablet, "from": from, "to": to });
            let raw = serde_json::to_vec(&body)?;
            let reply =
                admin_roundtrip(admin, "POST", "/v1/control/migrations/create", Some(&raw))?;
            if reply
                .get("ok")
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(false)
            {
                println!(
                    "plan created: {}",
                    reply
                        .get("plans")
                        .map_or_else(String::new, ToString::to_string)
                );
                Ok(())
            } else {
                anyhow::bail!("tablet move refused: {reply}");
            }
        }
        ClusterCommand::Rebalance => {
            let reply = admin_roundtrip(admin, "POST", "/v1/control/rebalance", Some(b"{}"))?;
            if reply
                .get("ok")
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(false)
            {
                println!(
                    "plans: {}",
                    reply
                        .get("plans")
                        .map_or_else(String::new, ToString::to_string)
                );
                Ok(())
            } else {
                anyhow::bail!("rebalance refused: {reply}");
            }
        }
        ClusterCommand::Leadership => {
            let reply = admin_roundtrip(admin, "POST", "/v1/control/leadership", Some(b"{}"))?;
            if reply
                .get("ok")
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(false)
            {
                println!(
                    "transfers: {}",
                    reply
                        .get("leadership_transfers")
                        .map_or_else(|| String::from("0"), ToString::to_string)
                );
                Ok(())
            } else {
                anyhow::bail!("leadership rebalance refused: {reply}");
            }
        }
        ClusterCommand::Migrations => {
            let reply: serde_json::Value =
                admin_roundtrip(admin, "GET", "/v1/control/migrations", None)?;
            let plans = reply
                .get("migrations")
                .and_then(serde_json::Value::as_array)
                .cloned()
                .unwrap_or_default();
            println!(
                "{:<6} {:<8} {:<6} {:<6} phase",
                "id", "tablet", "from", "to"
            );
            for plan in plans {
                println!(
                    "{:<6} {:<8} {:<6} {:<6} {}",
                    plan.get("id")
                        .and_then(serde_json::Value::as_u64)
                        .unwrap_or(0),
                    plan.get("tablet")
                        .and_then(serde_json::Value::as_u64)
                        .unwrap_or(0),
                    plan.get("from")
                        .and_then(serde_json::Value::as_u64)
                        .unwrap_or(0),
                    plan.get("to")
                        .and_then(serde_json::Value::as_u64)
                        .unwrap_or(0),
                    plan.get("phase")
                        .and_then(serde_json::Value::as_str)
                        .unwrap_or("?"),
                );
            }
            Ok(())
        }
    }
}

fn run(client: &NativeClient, command: &Command) -> Result<(), kivi_client::ClientError> {
    match command {
        // Diverted in `main` before any client connects; unreachable here.
        Command::Cluster { .. } => return Err(kivi_client::ClientError::InvalidRequest),
        Command::Get { key } => match client.get(&Key::from(key.clone()))? {
            Some(value) => println!("{}", String::from_utf8_lossy(&value)),
            None => println!("(nil)"),
        },
        Command::Set { key, value } => {
            client.set(&Key::from(key.clone()), Bytes::from(value.clone()))?;
            println!("OK");
        }
        Command::Del { key } => {
            println!("{}", u8::from(client.delete(&Key::from(key.clone()))?));
        }
        Command::Exists { key } => {
            println!("{}", u8::from(client.exists(&Key::from(key.clone()))?));
        }
        Command::CounterGet { key } => match client.counter_get(&Key::from(key.clone()))? {
            Some(value) => println!("{value}"),
            None => println!("(nil)"),
        },
        Command::CounterAdd { key, delta } => {
            println!("{}", client.counter_add(&Key::from(key.clone()), *delta)?);
        }
        Command::ExpireAt { key, micros } => {
            println!(
                "{}",
                u8::from(
                    client.expire_at(&Key::from(key.clone()), UnixMicros::from_micros(*micros))?
                )
            );
        }
        Command::Persist { key } => {
            println!(
                "{}",
                u8::from(client.persist_expiry(&Key::from(key.clone()))?)
            );
        }
        Command::Ttl { key } => match client.get_expiry(&Key::from(key.clone()))? {
            None => println!("(nil)"),
            Some(expiry) => match expiry.as_stamp() {
                None => println!("immortal"),
                Some(stamp) => {
                    let now = kivi_engine_clock_now();
                    if stamp.as_micros() <= now {
                        println!("expired");
                    } else {
                        #[allow(clippy::cast_precision_loss)]
                        let secs = (stamp.as_micros() - now) as f64 / 1_000_000.0;
                        println!("{secs:.3}");
                    }
                }
            },
        },
        Command::GetRange { key, offset, len } => {
            match client.get_range(&Key::from(key.clone()), *offset, *len)? {
                Some(value) => println!("{}", String::from_utf8_lossy(&value)),
                None => println!("(nil)"),
            }
        }
        Command::BytesLength { key } => match client.bytes_length(&Key::from(key.clone()))? {
            Some(len) => println!("{len}"),
            None => println!("(nil)"),
        },
        Command::SetConditional {
            key,
            value,
            condition,
            expiry,
        } => {
            let condition = match condition.as_str() {
                "always" => kivi_state::SetCondition::Always,
                "nx" => kivi_state::SetCondition::IfAbsent,
                "xx" => kivi_state::SetCondition::IfPresent,
                _ => return Err(kivi_client::ClientError::InvalidRequest),
            };
            let expiry = match expiry.strip_prefix("at:") {
                Some(stamp) => {
                    let Ok(micros) = stamp.parse::<u64>() else {
                        return Err(kivi_client::ClientError::InvalidRequest);
                    };
                    kivi_state::ExpiryPolicy::ExpireAt(UnixMicros::from_micros(micros))
                }
                None => match expiry.as_str() {
                    "clear" => kivi_state::ExpiryPolicy::Clear,
                    "keep" => kivi_state::ExpiryPolicy::Keep,
                    _ => return Err(kivi_client::ClientError::InvalidRequest),
                },
            };
            let (applied, _) = client.set_conditional(
                &Key::from(key.clone()),
                Bytes::from(value.clone()),
                condition,
                expiry,
            )?;
            println!("{}", if applied { "OK" } else { "(nil)" });
        }
    }
    Ok(())
}

fn kivi_engine_clock_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |span| {
            u64::try_from(span.as_micros()).unwrap_or(u64::MAX)
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[test]
    fn subcommands_parse_with_typed_arguments() {
        let cli = Cli::try_parse_from(["kivi-cli", "counter-add", "k", "41"]).expect("parses");
        assert!(matches!(
            cli.command,
            Command::CounterAdd { ref key, delta: 41 } if key == "k"
        ));
        let cli = Cli::try_parse_from(["kivi-cli", "--server", "10.0.0.2:9000", "get", "k"])
            .expect("parses");
        assert_eq!(cli.server, "10.0.0.2:9000");
        assert_eq!(cli.namespace, 1);
    }

    #[test]
    fn unknown_subcommands_and_missing_args_fail() {
        assert!(Cli::try_parse_from(["kivi-cli", "frobnicate"]).is_err());
        assert!(Cli::try_parse_from(["kivi-cli", "get"]).is_err());
        assert!(Cli::try_parse_from(["kivi-cli", "counter-add", "k", "lots"]).is_err());
    }

    #[test]
    fn cluster_operator_commands_parse() {
        let cli = Cli::try_parse_from(["kivi-cli", "cluster", "status"]).expect("parses");
        assert!(matches!(
            cli.command,
            Command::Cluster { ref command, .. } if matches!(command, ClusterCommand::Status)
        ));
        let cli = Cli::try_parse_from([
            "kivi-cli",
            "--admin",
            "10.0.0.9:19080",
            "cluster",
            "node",
            "drain",
            "3",
        ]);
        // `node drain` is `cluster node-drain` style: two words fail, the
        // hyphenated subcommand parses.
        assert!(cli.is_err());
        let cli = Cli::try_parse_from([
            "kivi-cli",
            "cluster",
            "--admin",
            "10.0.0.9:19080",
            "node-drain",
            "3",
        ])
        .expect("parses");
        match cli.command {
            Command::Cluster { admin, command } => {
                assert_eq!(admin, "10.0.0.9:19080");
                assert!(matches!(command, ClusterCommand::NodeDrain { id: 3 }));
            }
            _ => panic!("cluster command parses"),
        }
        let cli = Cli::try_parse_from([
            "kivi-cli",
            "cluster",
            "tablet-move",
            "--tablet",
            "7",
            "--from",
            "1",
            "--to",
            "4",
        ])
        .expect("parses");
        assert!(matches!(
            cli.command,
            Command::Cluster {
                command: ClusterCommand::TabletMove {
                    tablet: 7,
                    from: 1,
                    to: 4
                },
                ..
            }
        ));
    }
}
