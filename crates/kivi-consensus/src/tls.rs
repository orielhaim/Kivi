//! QUIC/H3 peer TLS identity: self-signed node certificates with
//! fingerprint pinning.
//!
//! Prototype PKI, deliberately minimal (§11–12): every node generates one
//! self-signed certificate on first open, persisted under
//! `<data_dir>/peer-tls/` (`cert.der`, `key.der`; Kivi-owned operational
//! files, never durable state formats). Peers authenticate by comparing the
//! presented certificate's fingerprint against statically configured
//! expectations:
//!
//! ```text
//! fingerprint = BLAKE3(cert DER)
//! ```
//!
//! BLAKE3 (not SHA-256) because the workspace already pins it and the
//! fingerprint is an opaque Kivi-local pin, never a standards-facing TLS
//! artifact. The algorithm is pinned alongside the pin files.
//!
//! Normal cluster mode always verifies: [`NodeConfig`](crate::node::NodeConfig)
//! carries `peer_certs` (every other member's certificate DER) and refuses
//! to connect otherwise. Only `kivi-lab` sets `insecure_skip_verify`
//! (test-only name, never production): the lab harness uses ephemeral ports
//! and fresh data directories per run, so static pins are impractical there.
//!
//! Kivi identity beyond TLS (cluster, node, incarnation, capabilities)
//! rides authenticated H3 request headers validated per request — TLS
//! proves *which certificate*, headers prove *which Kivi peer*. Socket
//! tuples are never identity (§33).

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use kivi_types::NodeId;

/// Fingerprint of a peer certificate: `BLAKE3(cert DER)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CertFingerprint(pub [u8; 32]);

impl CertFingerprint {
    /// Computes the fingerprint of DER bytes.
    #[must_use]
    pub fn of_der(der: &[u8]) -> Self {
        Self(kivi_codec::integrity::blake3_256(der))
    }

    /// Hex rendering for logs and config.
    #[must_use]
    pub fn hex(&self) -> String {
        let mut out = String::with_capacity(64);
        for byte in self.0 {
            use std::fmt::Write as _;
            let _ = write!(out, "{byte:02x}");
        }
        out
    }
}

/// Why peer TLS material could not be prepared.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum TlsError {
    /// Filesystem failure around the cert directory.
    #[error("peer TLS I/O during {op}: {detail}")]
    Io {
        /// What was attempted.
        op: &'static str,
        /// Human-readable cause.
        detail: String,
    },
    /// A configured fingerprint is malformed.
    #[error("bad peer fingerprint for node {node}: {detail}")]
    BadFingerprint {
        /// Node the pin names.
        node: u64,
        /// Human-readable cause.
        detail: String,
    },
}

/// This node's certificate and key (DER), loaded or freshly generated.
#[derive(Debug, Clone)]
pub struct NodeCert {
    /// Certificate DER.
    pub cert_der: Vec<u8>,
    /// Private key DER (PKCS#8).
    pub key_der: Vec<u8>,
    /// Fingerprint peers pin.
    pub fingerprint: CertFingerprint,
}

impl NodeCert {
    /// File holding the certificate DER.
    #[must_use]
    pub fn cert_path(dir: &Path) -> PathBuf {
        dir.join("peer-tls").join("cert.der")
    }

    /// File holding the private-key DER.
    #[must_use]
    pub fn key_path(dir: &Path) -> PathBuf {
        dir.join("peer-tls").join("key.der")
    }

    /// Loads the node certificate, generating and persisting a self-signed
    /// one (CN `kivi-n{node}`) on first open.
    ///
    /// # Errors
    ///
    /// Returns [`TlsError::Io`] on filesystem or generation failures.
    pub fn load_or_generate(data_dir: &Path, node: NodeId) -> Result<Self, TlsError> {
        let dir = data_dir.join("peer-tls");
        let cert_path = Self::cert_path(data_dir);
        let key_path = Self::key_path(data_dir);
        if let (Ok(cert_der), Ok(key_der)) = (std::fs::read(&cert_path), std::fs::read(&key_path))
            && !cert_der.is_empty()
            && !key_der.is_empty()
        {
            let fingerprint = CertFingerprint::of_der(&cert_der);
            return Ok(Self {
                cert_der,
                key_der,
                fingerprint,
            });
        }
        let name = format!("kivi-n{}", node.as_u64());
        let certified =
            rcgen::generate_simple_self_signed(vec![name]).map_err(|error| TlsError::Io {
                op: "generate peer certificate",
                detail: error.to_string(),
            })?;
        let cert_der = certified.cert.der().to_vec();
        let secret = certified.signing_key.serialize_der();
        let key_der: Vec<u8> = Vec::from(&secret[..]);
        std::fs::create_dir_all(&dir).map_err(|error| TlsError::Io {
            op: "create peer-tls directory",
            detail: error.to_string(),
        })?;
        // Best-effort atomicity: temp + rename per file; a crash leaves the
        // previous complete pair or none (regeneration is idempotent only
        // before first use — after peers pin us, operator repair applies).
        for (path, bytes) in [(&cert_path, &cert_der), (&key_path, &key_der)] {
            let tmp = path.with_extension("tmp");
            std::fs::write(&tmp, bytes).map_err(|error| TlsError::Io {
                op: "write peer TLS file",
                detail: error.to_string(),
            })?;
            std::fs::rename(&tmp, path).map_err(|error| TlsError::Io {
                op: "publish peer TLS file",
                detail: error.to_string(),
            })?;
        }
        let fingerprint = CertFingerprint::of_der(&cert_der);
        tracing::info!(node = node.as_u64(), fingerprint = %fingerprint.hex(), "generated peer TLS certificate");
        Ok(Self {
            cert_der,
            key_der,
            fingerprint,
        })
    }
}

/// Server name presented when dialing `peer` (`kivi-n{id}` matches the
/// CN minted in [`NodeCert::load_or_generate`]).
#[must_use]
pub fn server_name_for(peer: NodeId) -> String {
    format!("kivi-n{}", peer.as_u64())
}

/// Builds a rustls client config trusting exactly the given peer
/// certificates as anchors (self-signed; each peer's cert is its own
/// root). Standard `WebPKI` verification then enforces signatures and the
/// `kivi-n{id}` SAN honestly — no custom signature logic, no impersonation
/// hole. Which peer the certificate belongs to is established by the
/// authenticated Kivi headers per request and cross-checked against the
/// dialed peer.
///
/// 0-RTT is disabled (`enable_early_data = false`): mutating Raft RPCs must
/// never rely on replayable semantics.
///
/// # Errors
///
/// Returns a human-readable reason when a configured cert is malformed.
#[allow(clippy::implicit_hasher)]
pub fn client_config_pinned(
    expected: &HashMap<NodeId, Vec<u8>>,
) -> Result<rustls::ClientConfig, String> {
    let mut roots = rustls::RootCertStore::empty();
    let mut order: Vec<NodeId> = expected.keys().copied().collect();
    order.sort_by_key(|node| node.as_u64());
    for node in order {
        let der = &expected[&node];
        roots
            .add(rustls::pki_types::CertificateDer::from(der.clone()))
            .map_err(|error| format!("peer {} cert invalid: {error}", node.as_u64()))?;
    }
    let mut config = rustls::ClientConfig::builder()
        .with_root_certificates(roots)
        .with_no_client_auth();
    config.alpn_protocols = vec![b"h3".to_vec()];
    config.enable_early_data = false;
    Ok(config)
}

/// Test-only rustls client config that skips server verification.
///
/// Used exclusively when `insecure_skip_verify` is set, which only
/// `kivi-lab` does (fresh data directories and ephemeral ports per run).
/// Never production: loopback tests have no attacker, but any real
/// deployment without verification admits impersonation.
#[must_use]
pub fn client_config_insecure() -> rustls::ClientConfig {
    #[derive(Debug)]
    struct NoVerify;
    impl rustls::client::danger::ServerCertVerifier for NoVerify {
        fn verify_server_cert(
            &self,
            _end_entity: &rustls::pki_types::CertificateDer<'_>,
            _intermediates: &[rustls::pki_types::CertificateDer<'_>],
            _server_name: &rustls::pki_types::ServerName<'_>,
            _ocsp_response: &[u8],
            _now: rustls::pki_types::UnixTime,
        ) -> Result<rustls::client::danger::ServerCertVerified, rustls::Error> {
            Ok(rustls::client::danger::ServerCertVerified::assertion())
        }
        fn verify_tls12_signature(
            &self,
            _message: &[u8],
            _cert: &rustls::pki_types::CertificateDer<'_>,
            _dss: &rustls::DigitallySignedStruct,
        ) -> Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
            Ok(rustls::client::danger::HandshakeSignatureValid::assertion())
        }
        fn verify_tls13_signature(
            &self,
            _message: &[u8],
            _cert: &rustls::pki_types::CertificateDer<'_>,
            _dss: &rustls::DigitallySignedStruct,
        ) -> Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
            Ok(rustls::client::danger::HandshakeSignatureValid::assertion())
        }
        fn supported_verify_schemes(&self) -> Vec<rustls::SignatureScheme> {
            vec![rustls::SignatureScheme::ECDSA_NISTP256_SHA256]
        }
    }
    let mut config = rustls::ClientConfig::builder()
        .dangerous()
        .with_custom_certificate_verifier(Arc::new(NoVerify))
        .with_no_client_auth();
    config.alpn_protocols = vec![b"h3".to_vec()];
    config.enable_early_data = false;
    config
}
