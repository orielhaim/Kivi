//! Immutable artifact identity and layout: the small seam future stores
//! implement.
//!
//! Every immutable checkpoint artifact (band, dedup component, manifest)
//! is identified by the BLAKE3 hash of its payload and stored under that
//! hash. Verification is structural: filename equals content hash, header
//! CRC covers the header, body CRC plus content hash cover the body.
//!
//! Current provider: the local filesystem under `<data-dir>/checkpoints`.
//! Future providers (`NVMe` packs, object store, peer sources, coded
//! storage) implement this same identity — hence the seam stays tiny:
//! hashes, kinds, and paths, never a storage framework.
//!
//! ```text
//! <data-dir>/checkpoints/
//!     bands/<blake3hex>.band
//!     comp/<blake3hex>.dedup
//!     tablet-<id>/
//!         CURRENT
//!         manifests/<blake3hex>.manifest
//! ```

use std::path::{Path, PathBuf};

/// Length of a BLAKE3 content hash in bytes.
pub const HASH_LEN: usize = 32;

/// Content identity of one immutable artifact.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ArtifactHash([u8; HASH_LEN]);

impl ArtifactHash {
    /// Hashes a payload into its content identity.
    #[must_use]
    pub fn of(payload: &[u8]) -> Self {
        Self(kivi_codec::integrity::blake3_256(payload))
    }

    /// Hashes pre-image chunks incrementally (header identity plus body
    /// without concatenating them). Checkpoint content identities always
    /// cover the artifact's identity header (minus its CRC field) plus
    /// the stored body — bodies alone do not identify an artifact (empty
    /// bands of different ids share an empty body).
    #[must_use]
    pub fn of_chunks(chunks: &[&[u8]]) -> Self {
        let mut hasher = blake3::Hasher::new();
        for chunk in chunks {
            hasher.update(chunk);
        }
        Self(*hasher.finalize().as_bytes())
    }

    /// Wraps raw hash bytes (e.g. decoded from a manifest).
    #[must_use]
    pub const fn from_bytes(bytes: [u8; HASH_LEN]) -> Self {
        Self(bytes)
    }

    /// Returns the raw hash bytes.
    #[must_use]
    pub const fn as_bytes(&self) -> &[u8; HASH_LEN] {
        &self.0
    }

    /// Lowercase hex filename stem.
    #[must_use]
    pub fn hex(&self) -> String {
        const HEX: &[u8; 16] = b"0123456789abcdef";
        let mut out = String::with_capacity(HASH_LEN * 2);
        for byte in self.0 {
            out.push(HEX[(byte >> 4) as usize] as char);
            out.push(HEX[(byte & 0x0F) as usize] as char);
        }
        out
    }

    /// Parses a lowercase hex stem back into a hash.
    #[must_use]
    pub fn parse_hex(text: &str) -> Option<Self> {
        let bytes = text.as_bytes();
        if bytes.len() != HASH_LEN * 2 {
            return None;
        }
        let (chunks, remainder) = bytes.as_chunks::<2>();
        if !remainder.is_empty() {
            return None;
        }
        let mut raw = [0u8; HASH_LEN];
        for (index, chunk) in chunks.iter().enumerate() {
            let hi = hex_val(chunk[0])?;
            let lo = hex_val(chunk[1])?;
            raw[index] = (hi << 4) | lo;
        }
        Some(Self(raw))
    }
}

fn hex_val(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        _ => None,
    }
}

impl core::fmt::Display for ArtifactHash {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(&self.hex())
    }
}

/// Which immutable store an artifact lives in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ArtifactKind {
    /// Checkpoint band (`bands/<hash>.band`).
    Band,
    /// Dedup component (`comp/<hash>.dedup`).
    DedupComponent,
    /// Manifest (`tablet-<id>/manifests/<hash>.manifest`).
    Manifest,
}

/// Checkpoint directory layout roots.
pub const CHECKPOINTS_DIR_NAME: &str = "checkpoints";
/// Content-addressed band store.
pub const BANDS_DIR_NAME: &str = "bands";
/// Content-addressed dedup-component store.
pub const COMP_DIR_NAME: &str = "comp";
/// Per-tablet installed-pointer plus manifest store.
pub const MANIFESTS_DIR_NAME: &str = "manifests";
/// Installed-checkpoint pointer filename.
pub const CURRENT_FILE_NAME: &str = "CURRENT";
/// Durable WAL-floor record filename.
pub const WAL_FLOOR_FILE_NAME: &str = "WAL_FLOOR";

/// Roots `<data-dir>/checkpoints`.
#[must_use]
pub fn checkpoints_dir(data_dir: &Path) -> PathBuf {
    data_dir.join(CHECKPOINTS_DIR_NAME)
}

/// Per-tablet directory `checkpoints/tablet-<id>`.
#[must_use]
pub fn tablet_dir(data_dir: &Path, tablet: kivi_types::TabletId) -> PathBuf {
    checkpoints_dir(data_dir).join(format!("tablet-{}", tablet.as_u64()))
}

/// Full path of one artifact. Manifests are per-tablet; bands and dedup
/// components are global (content-addressed sharing across tablets).
#[must_use]
pub fn artifact_path(
    data_dir: &Path,
    tablet: kivi_types::TabletId,
    kind: ArtifactKind,
    hash: ArtifactHash,
) -> PathBuf {
    match kind {
        ArtifactKind::Band => checkpoints_dir(data_dir)
            .join(BANDS_DIR_NAME)
            .join(format!("{}.band", hash.hex())),
        ArtifactKind::DedupComponent => checkpoints_dir(data_dir)
            .join(COMP_DIR_NAME)
            .join(format!("{}.dedup", hash.hex())),
        ArtifactKind::Manifest => tablet_dir(data_dir, tablet)
            .join(MANIFESTS_DIR_NAME)
            .join(format!("{}.manifest", hash.hex())),
    }
}

/// Path of one tablet's installed-checkpoint pointer.
#[must_use]
pub fn current_path(data_dir: &Path, tablet: kivi_types::TabletId) -> PathBuf {
    tablet_dir(data_dir, tablet).join(CURRENT_FILE_NAME)
}

/// Path of the durable WAL-floor record.
#[must_use]
pub fn wal_floor_path(data_dir: &Path) -> PathBuf {
    checkpoints_dir(data_dir).join(WAL_FLOOR_FILE_NAME)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hash_hex_round_trips() {
        let hash = ArtifactHash::of(b"band-payload");
        let parsed = ArtifactHash::parse_hex(&hash.hex()).expect("round trip");
        assert_eq!(parsed, hash);
        assert!(ArtifactHash::parse_hex("zz").is_none());
        assert!(ArtifactHash::parse_hex(&hash.hex()[..10]).is_none());
        assert!(ArtifactHash::parse_hex(&"0".repeat(64)).is_some());
        assert!(ArtifactHash::parse_hex(&"G".repeat(64)).is_none());
    }

    #[test]
    fn artifact_paths_share_content_addressed_stores() {
        let root = Path::new("/data");
        let first = kivi_types::TabletId::from_u64(1);
        let second = kivi_types::TabletId::from_u64(2);
        let hash = ArtifactHash::of(b"x");
        // Bands and dedup components are global across tablets...
        assert_eq!(
            artifact_path(root, first, ArtifactKind::Band, hash),
            artifact_path(root, second, ArtifactKind::Band, hash)
        );
        // ...but manifests (and CURRENT pointers) are per-tablet.
        assert_ne!(
            artifact_path(root, first, ArtifactKind::Manifest, hash),
            artifact_path(root, second, ArtifactKind::Manifest, hash)
        );
        assert_ne!(current_path(root, first), current_path(root, second));
    }
}
