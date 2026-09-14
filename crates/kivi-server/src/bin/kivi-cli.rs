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
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    let client = NativeClient::new(ClientConfig {
        seeds: vec![cli.server],
        namespace: NamespaceId::from_u64(cli.namespace),
        ..ClientConfig::default()
    })?;
    run(&client, &cli.command)?;
    Ok(())
}

fn run(client: &NativeClient, command: &Command) -> Result<(), kivi_client::ClientError> {
    match command {
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
}
