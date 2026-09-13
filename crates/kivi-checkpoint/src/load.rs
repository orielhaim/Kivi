//! Installed-checkpoint loading: CURRENT to restored state.
//!
//! Recovery loads the installed checkpoint (current, loudly falling back
//! to the retained previous when the current chain is damaged), verifies
//! every referenced artifact, and returns plain-data state the engine
//! installs before replaying the WAL tail.
//!
//! Fallback rule (spec Y): falling back is LOUD (`fell_back`) and only
//! safe because reclamation never removes WAL the previous checkpoint
//! needs — the caller guarantees that invariant, and the loader checks
//! the catalog chain is monotonic (a previous that leads the current is
//! catalog corruption, refused without guessing).

use std::path::Path;

use crate::artifact::{ArtifactKind, artifact_path};
use crate::band::{LoadedBand, load_band};
use crate::catalog::CurrentRecord;
use crate::dedup::{LoadedDedup, load_dedup};
use crate::error::CheckpointError;
use crate::manifest::{LoadedManifest, load_manifest};
use crate::publish::read_current;

/// One fully verified installed checkpoint, ready to install.
#[derive(Debug)]
pub struct InstalledCheckpoint {
    /// The CURRENT record that installed it (previous when fell back).
    pub current: CurrentRecord,
    /// The verified manifest.
    pub manifest: LoadedManifest,
    /// All bands in band order (verified: identity, CRCs, ordering).
    pub bands: Vec<LoadedBand>,
    /// The verified dedup component.
    pub dedup: LoadedDedup,
    /// True when the retained previous checkpoint saved recovery (loud).
    pub fell_back: bool,
}

/// Loads the installed checkpoint for one tablet (`None` when never
/// published — recovery replays the WAL from genesis).
///
/// Tries the current chain first, then the retained previous chain with a
/// loud fallback. Both chains failing is a hard error: history is never
/// silently truncated and partial state is never served.
///
/// # Errors
///
/// Returns [`CheckpointError`] when no chain loads (with the previous
/// failure chained into the message when both fail).
pub fn load_installed(
    data_dir: &Path,
    tablet: kivi_types::TabletId,
) -> Result<Option<InstalledCheckpoint>, CheckpointError> {
    let Some(current) = read_current(data_dir, tablet)? else {
        return Ok(None);
    };
    match load_chain(data_dir, &current, false) {
        Ok(installed) => Ok(Some(installed)),
        Err(current_error) => {
            let Some(previous_hash) = current.previous else {
                return Err(current_error);
            };
            // Loud fallback: the previous chain must be strictly older
            // (a leading previous is catalog corruption, not an option).
            let previous_bytes = read_manifest_bytes(data_dir, tablet, previous_hash)?;
            let previous_manifest = load_manifest(previous_hash, &previous_bytes)?;
            if previous_manifest.cut > current.cut {
                return Err(CheckpointError::Format {
                    detail: format!(
                        "checkpoint catalog corrupt: previous cut {} leads current cut {}",
                        previous_manifest.cut, current.cut
                    ),
                });
            }
            let previous_current = CurrentRecord {
                tablet,
                cut: previous_manifest.cut,
                manifest: previous_hash,
                previous: previous_manifest.previous,
                created_wall_micros: current.created_wall_micros,
            };
            match load_chain(data_dir, &previous_current, true) {
                Ok(installed) => {
                    tracing::warn!(
                        tablet = tablet.as_u64(),
                        current_cut = current.cut,
                        previous_cut = previous_manifest.cut,
                        error = ?current_error,
                        "installed checkpoint damaged; recovered from retained previous"
                    );
                    Ok(Some(installed))
                }
                Err(previous_error) => Err(CheckpointError::Corrupt {
                    detail: format!(
                        "installed and previous checkpoints both unloadable: current: {current_error:?}; previous: {previous_error:?}"
                    ),
                }),
            }
        }
    }
}

/// Loads and verifies one CURRENT chain (manifest, all bands, dedup) with
/// full binding checks: manifest identity matches the pointer, band
/// provenance matches the manifest, cuts line up.
fn load_chain(
    data_dir: &Path,
    current: &CurrentRecord,
    fell_back: bool,
) -> Result<InstalledCheckpoint, CheckpointError> {
    let manifest_bytes = read_manifest_bytes(data_dir, current.tablet, current.manifest)?;
    let manifest = load_manifest(current.manifest, &manifest_bytes)?;
    if manifest.tablet != current.tablet {
        return Err(CheckpointError::Format {
            detail: "manifest names a different tablet than its CURRENT pointer".to_owned(),
        });
    }
    if manifest.cut != current.cut {
        return Err(CheckpointError::Format {
            detail: format!(
                "manifest cut {} disagrees with CURRENT cut {}",
                manifest.cut, current.cut
            ),
        });
    }
    let mut bands = Vec::with_capacity(manifest.bands.len());
    for descriptor in &manifest.bands {
        let path = artifact_path(
            data_dir,
            current.tablet,
            ArtifactKind::Band,
            descriptor.hash,
        );
        let bytes = std::fs::read(&path)
            .map_err(|error| CheckpointError::io("read checkpoint band", &path, &error))?;
        if bytes.len() as u64 != descriptor.len {
            return Err(CheckpointError::Corrupt {
                detail: format!(
                    "band {} length {} disagrees with manifest {}",
                    descriptor.hash,
                    bytes.len(),
                    descriptor.len
                ),
            });
        }
        bands.push(load_band(
            manifest.tablet,
            descriptor.band,
            manifest.layout,
            descriptor.hash,
            &bytes,
        )?);
    }
    let dedup_path = artifact_path(
        data_dir,
        current.tablet,
        ArtifactKind::DedupComponent,
        manifest.dedup.hash,
    );
    let dedup_bytes = std::fs::read(&dedup_path).map_err(|error| {
        CheckpointError::io("read checkpoint dedup component", &dedup_path, &error)
    })?;
    let dedup = load_dedup(manifest.tablet, manifest.dedup.hash, &dedup_bytes)?;
    Ok(InstalledCheckpoint {
        current: current.clone(),
        manifest,
        bands,
        dedup,
        fell_back,
    })
}

fn read_manifest_bytes(
    data_dir: &Path,
    tablet: kivi_types::TabletId,
    hash: crate::artifact::ArtifactHash,
) -> Result<Vec<u8>, CheckpointError> {
    let path = artifact_path(data_dir, tablet, ArtifactKind::Manifest, hash);
    std::fs::read(&path)
        .map_err(|error| CheckpointError::io("read checkpoint manifest", &path, &error))
}
