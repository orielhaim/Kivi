//! Prints this node's QUIC peer certificate (hex DER) for static cluster
//! configuration (`--cluster-peer-certs node=path,...` expects DER files;
//! this tool prints hex so operators can eyeball fingerprints, and writes
//! nothing). The certificate is generated on first use and reused after.

use std::path::PathBuf;

use clap::Parser;
use kivi_types::NodeId;

#[derive(Debug, Parser)]
struct Cli {
    /// Data-directory root (peer certificate lives under `peer-tls/`).
    #[arg(long)]
    data_dir: PathBuf,
    /// This process's node identity.
    #[arg(long)]
    node_id: u64,
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    if cli.node_id == 0 {
        anyhow::bail!("node id 0 is reserved");
    }
    let cert = kivi_consensus::tls::NodeCert::load_or_generate(
        &cli.data_dir,
        NodeId::from_u64(cli.node_id),
    )
    .map_err(|error| anyhow::anyhow!("peer certificate: {error}"))?;
    println!(
        "node={} fingerprint={}",
        cli.node_id,
        cert.fingerprint.hex()
    );
    print!("{}", hex(&cert.cert_der));
    Ok(())
}

/// Lowercase hex rendering.
fn hex(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        use std::fmt::Write as _;
        let _ = write!(out, "{byte:02x}");
    }
    out
}
