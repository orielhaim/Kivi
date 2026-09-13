//! Crash-safe checkpoint publication: bands, dedup, manifest, CURRENT.
//!
//! Publication order (spec V):
//!
//! ```text
//! write missing immutable bands (+ read-back verify new writes)
//!         ↓
//! sync band directories (batched: one sync per directory)
//!         ↓
//! write dedup component (+ read-back verify)
//!         ↓
//! write manifest temp (+ read-back verify)
//!         ↓
//! publish manifest atomically (rename) + sync parent
//!         ↓
//! durably publish CURRENT (atomic record: new current + previous)
//!         ↓
//! checkpoint becomes installed
//!         ↓
//! sweep unreferenced artifacts (best-effort retention GC)
//! ```
//!
//! An interrupted publication is invisible: CURRENT still points at the
//! previous checkpoint, and orphaned content-addressed files are either
//! reused (identical bytes) or swept by the next publish. Reused bands
//! are never rewritten — but a referenced-yet-missing band fails loudly
//! (rebuilding it at a new cut would fork its identity).

use std::collections::HashSet;
use std::path::Path;

use kivi_durability::fs::{create_dir_all_sync, sync_dir, tmp_path, write_atomic_sync};

use crate::artifact::{ArtifactKind, artifact_path, checkpoints_dir, current_path, tablet_dir};
use crate::builder::{BuiltBandRef, BuiltTablet};
use crate::catalog::{CurrentRecord, encode_current, wall_micros_now};
use crate::error::CheckpointError;

/// Publication accounting for operators and tests.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PublishStats {
    /// Bands freshly written (dirty).
    pub bands_written: usize,
    /// Bands skipped (byte-identical file already present).
    pub bands_skipped_present: usize,
    /// Dedup component written (`false` when already present).
    pub dedup_written: bool,
    /// Manifest written (`false` when already present — identical rebuild).
    pub manifest_written: bool,
    /// Artifact files removed by retention GC.
    pub garbage_collected: usize,
}

/// Publishes one built tablet checkpoint: writes missing artifacts,
/// installs the manifest reference in CURRENT (demoting the previous
/// CURRENT to the retention slot), and sweeps unreferenced artifacts.
/// Returns the installed CURRENT record plus accounting.
///
/// `previous` must be the CURRENT record this publish supersedes (`None`
/// on first install); a cut below it fails loudly instead of forking
/// history.
///
/// # Errors
///
/// Returns [`CheckpointError`] on missing reused bands, I/O failures, or
/// cut regression. GC failures never fail publication (disk usage, not
/// corruption).
// Orchestration across artifact kinds (bands, dedup, manifest, CURRENT,
// GC): linear stages, kept together so the order is reviewable in one
// place.
#[allow(clippy::too_many_lines)]
pub fn publish_tablet(
    data_dir: &Path,
    built: &BuiltTablet,
    previous: Option<&CurrentRecord>,
) -> Result<(CurrentRecord, PublishStats), CheckpointError> {
    if let Some(current) = previous {
        if current.tablet != built.tablet {
            return Err(CheckpointError::Format {
                detail: "publish supersedes a different tablet's CURRENT".to_owned(),
            });
        }
        if built.cut < current.cut {
            return Err(CheckpointError::Format {
                detail: format!(
                    "publish cut {} regresses installed {}",
                    built.cut, current.cut
                ),
            });
        }
    }
    let tablet = built.tablet;
    let bands_dir = checkpoints_dir(data_dir).join(crate::artifact::BANDS_DIR_NAME);
    let comp_dir = checkpoints_dir(data_dir).join(crate::artifact::COMP_DIR_NAME);
    let manifests_dir = tablet_dir(data_dir, tablet).join(crate::artifact::MANIFESTS_DIR_NAME);
    for dir in [&bands_dir, &comp_dir, &manifests_dir] {
        let created = create_dir_all_sync(dir)
            .map_err(|error| CheckpointError::io("create checkpoint directory", dir, &error))?;
        tracing::debug!(dir = %dir.display(), ?created, "checkpoint directory ready");
    }
    let mut stats = PublishStats::default();
    // Bands: write missing, verify new writes by read-back, stat-check
    // reused references (a referenced-yet-missing band is a loud error).
    for reference in &built.bands {
        match reference {
            BuiltBandRef::New(band) => {
                let path = artifact_path(data_dir, tablet, ArtifactKind::Band, band.hash);
                if file_matches(&path, band.bytes.len() as u64) {
                    stats.bands_skipped_present += 1;
                } else {
                    write_file_sync(&path, &band.bytes)?;
                    verify_file_exact(&path, &band.bytes)?;
                    stats.bands_written += 1;
                }
            }
            BuiltBandRef::Reused(descriptor) => {
                let path = artifact_path(data_dir, tablet, ArtifactKind::Band, descriptor.hash);
                let len = std::fs::metadata(&path)
                    .map_err(|error| {
                        CheckpointError::io("reused checkpoint band missing", &path, &error)
                    })?
                    .len();
                if len != descriptor.len {
                    return Err(CheckpointError::Corrupt {
                        detail: format!(
                            "reused band {} length {len} disagrees with manifest {}",
                            descriptor.hash, descriptor.len
                        ),
                    });
                }
                stats.bands_skipped_present += 1;
            }
        }
    }
    let synced = sync_dir(&bands_dir).map_err(|error| {
        CheckpointError::io("sync checkpoint band directory", &bands_dir, &error)
    })?;
    tracing::debug!(dir = %bands_dir.display(), ?synced, "checkpoint bands synced");
    // Dedup component: same content-addressed protocol (always freshly
    // built, but identical bytes skip the write naturally).
    let dedup_path = artifact_path(
        data_dir,
        tablet,
        ArtifactKind::DedupComponent,
        built.dedup.hash,
    );
    if file_matches(&dedup_path, built.dedup.bytes.len() as u64) {
        stats.dedup_written = false;
    } else {
        write_file_sync(&dedup_path, &built.dedup.bytes)?;
        verify_file_exact(&dedup_path, &built.dedup.bytes)?;
        stats.dedup_written = true;
    }
    let synced = sync_dir(&comp_dir).map_err(|error| {
        CheckpointError::io("sync checkpoint comp directory", &comp_dir, &error)
    })?;
    tracing::debug!(dir = %comp_dir.display(), ?synced, "checkpoint components synced");
    // Manifest: content-addressed like everything else.
    let manifest_path = artifact_path(
        data_dir,
        tablet,
        ArtifactKind::Manifest,
        built.manifest.hash,
    );
    if file_matches(&manifest_path, built.manifest.bytes.len() as u64) {
        stats.manifest_written = false;
    } else {
        write_file_sync(&manifest_path, &built.manifest.bytes)?;
        verify_file_exact(&manifest_path, &built.manifest.bytes)?;
        stats.manifest_written = true;
    }
    let synced = sync_dir(&manifests_dir).map_err(|error| {
        CheckpointError::io("sync checkpoint manifest directory", &manifests_dir, &error)
    })?;
    tracing::debug!(dir = %manifests_dir.display(), ?synced, "checkpoint manifests synced");
    // CURRENT: the atomic install. One record names both the new current
    // and the demoted previous, so retention never observes a half-step.
    let installed = CurrentRecord {
        tablet,
        cut: built.cut,
        manifest: built.manifest.hash,
        previous: previous.map(|current| current.manifest),
        created_wall_micros: wall_micros_now(),
    };
    write_atomic_sync(&current_path(data_dir, tablet), &encode_current(&installed)).map_err(
        |error| {
            CheckpointError::io(
                "publish checkpoint CURRENT",
                &current_path(data_dir, tablet),
                &error,
            )
        },
    )?;
    // Retention GC: keep exactly what current + previous reference.
    stats.garbage_collected = sweep_unreferenced(data_dir, tablet, &installed);
    Ok((installed, stats))
}

/// Reads one tablet's installed CURRENT (`None` when never published).
///
/// # Errors
///
/// Returns [`CheckpointError`] on unreadable or corrupt records.
pub fn read_current(
    data_dir: &Path,
    tablet: kivi_types::TabletId,
) -> Result<Option<CurrentRecord>, CheckpointError> {
    let path = current_path(data_dir, tablet);
    match std::fs::read(&path) {
        Ok(bytes) => Ok(Some(crate::catalog::decode_current(tablet, &bytes)?)),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(CheckpointError::io(
            "read checkpoint CURRENT",
            &path,
            &error,
        )),
    }
}

/// Writes the durable WAL-floor record atomically (reclaim promises).
///
/// # Errors
///
/// Returns [`CheckpointError`] when the record cannot be published.
pub fn write_wal_floor(
    data_dir: &Path,
    floors: &[kivi_durability::LaneFloor],
) -> Result<(), CheckpointError> {
    use crate::artifact::wal_floor_path;
    use crate::catalog::encode_wal_floor;
    let path = wal_floor_path(data_dir);
    if let Some(parent) = path.parent() {
        let created = create_dir_all_sync(parent)
            .map_err(|error| CheckpointError::io("create checkpoint directory", parent, &error))?;
        tracing::debug!(dir = %parent.display(), ?created, "checkpoint directory ready");
    }
    write_atomic_sync(&path, &encode_wal_floor(floors.to_vec())?)
        .map_err(|error| CheckpointError::io("publish WAL floor", &path, &error))?;
    Ok(())
}

/// Reads the durable WAL-floor record (`None` when never published —
/// recovery replays from genesis).
///
/// # Errors
///
/// Returns [`CheckpointError`] on unreadable or corrupt records.
pub fn read_wal_floor(
    data_dir: &Path,
) -> Result<Option<Vec<kivi_durability::LaneFloor>>, CheckpointError> {
    use crate::artifact::wal_floor_path;
    use crate::catalog::decode_wal_floor;
    let path = wal_floor_path(data_dir);
    match std::fs::read(&path) {
        Ok(bytes) => Ok(Some(decode_wal_floor(&bytes)?)),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(CheckpointError::io("read WAL floor", &path, &error)),
    }
}

fn file_matches(path: &Path, len: u64) -> bool {
    std::fs::metadata(path).map_or(u64::MAX, |meta| meta.len()) == len
}

/// Writes `bytes` to `path` through temp + sync + atomic rename (without
/// the final directory sync — the publisher batches one sync per
/// directory after all renames).
fn write_file_sync(path: &Path, bytes: &[u8]) -> Result<(), CheckpointError> {
    use kivi_durability::fs::sync_file;
    use std::io::Write;
    let tmp = tmp_path(path);
    let mut file = std::fs::File::create(&tmp)
        .map_err(|error| CheckpointError::io("create checkpoint artifact temp", &tmp, &error))?;
    file.write_all(bytes)
        .map_err(|error| CheckpointError::io("write checkpoint artifact temp", &tmp, &error))?;
    // Sync through the writable handle: reopening read-only and syncing
    // fails on Windows (flush needs write access).
    sync_file(&file)
        .map_err(|error| CheckpointError::io("sync checkpoint artifact temp", &tmp, &error))?;
    drop(file);
    std::fs::rename(&tmp, path)
        .map_err(|error| CheckpointError::io("publish checkpoint artifact", path, &error))?;
    Ok(())
}

/// Reads back a published artifact and checks byte-exact identity with
/// what was written. Page-cache hot (just written), so this costs a
/// memcpy, not disk I/O. Content-hash identity is verified separately at
/// load time by every reader.
fn verify_file_exact(path: &Path, expected: &[u8]) -> Result<(), CheckpointError> {
    let bytes = std::fs::read(path)
        .map_err(|error| CheckpointError::io("read back checkpoint artifact", path, &error))?;
    if bytes != expected {
        return Err(CheckpointError::Corrupt {
            detail: format!(
                "published checkpoint artifact fails read-back: {}",
                path.display()
            ),
        });
    }
    Ok(())
}

/// Deletes artifact files no installed checkpoint references (current +
/// previous). Best-effort: any failure keeps files (disk usage, never
/// corruption), and an unreadable directory skips GC entirely.
fn sweep_unreferenced(
    data_dir: &Path,
    tablet: kivi_types::TabletId,
    installed: &CurrentRecord,
) -> usize {
    // Resolve the full reference set first; if any manifest in the chain
    // is unreadable, keep everything (keep-on-doubt).
    let mut referenced_bands: HashSet<String> = HashSet::new();
    let mut referenced_comp: HashSet<String> = HashSet::new();
    let mut referenced_manifests: HashSet<String> = HashSet::new();
    let mut chain = vec![installed.manifest];
    if let Some(previous) = installed.previous {
        chain.push(previous);
    }
    for hash in &chain {
        referenced_manifests.insert(format!("{}.manifest", hash.hex()));
        let path = artifact_path(data_dir, tablet, ArtifactKind::Manifest, *hash);
        let Ok(bytes) = std::fs::read(&path) else {
            return 0;
        };
        let Ok(manifest) = crate::manifest::load_manifest(*hash, &bytes) else {
            return 0;
        };
        for band in &manifest.bands {
            referenced_bands.insert(format!("{}.band", band.hash.hex()));
        }
        referenced_comp.insert(format!("{}.dedup", manifest.dedup.hash.hex()));
    }
    let mut removed = 0usize;
    removed += sweep_dir(
        &checkpoints_dir(data_dir).join(crate::artifact::BANDS_DIR_NAME),
        &referenced_bands,
    );
    removed += sweep_dir(
        &checkpoints_dir(data_dir).join(crate::artifact::COMP_DIR_NAME),
        &referenced_comp,
    );
    removed += sweep_dir(
        &tablet_dir(data_dir, tablet).join(crate::artifact::MANIFESTS_DIR_NAME),
        &referenced_manifests,
    );
    removed
}

fn sweep_dir(dir: &Path, referenced: &HashSet<String>) -> usize {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return 0;
    };
    let mut removed = 0usize;
    for entry in entries.flatten() {
        let path = entry.path();
        // Stale publish temps always go; unreferenced artifacts go only
        // when they are known artifact files (never touch foreign files).
        // Extension checks use `Path` (case-sensitive by construction on
        // the platforms Kivi targets for local data).
        let is_tmp = path.extension().is_some_and(|ext| ext == "tmp");
        let is_artifact = ["band", "dedup", "manifest"]
            .iter()
            .any(|kind| path.extension().is_some_and(|ext| ext == *kind));
        let name = entry.file_name().to_string_lossy().into_owned();
        if (is_tmp || (is_artifact && !referenced.contains(&name)))
            && std::fs::remove_file(&path).is_ok()
        {
            removed += 1;
        }
    }
    let _ = sync_dir(dir);
    removed
}
