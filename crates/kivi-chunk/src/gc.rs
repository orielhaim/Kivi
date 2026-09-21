//! Mark-based immutable-information garbage collection.
//!
//! No reference counting: the collector marks everything reachable from an
//! explicit root set, and every sealed pack with zero reachable records
//! becomes a deletion candidate. Deletion itself is best-effort (a failed
//! delete costs disk, never correctness), exactly like WAL reclamation.
//!
//! ## Root set (§37)
//!
//! The engine merges at least: live tablet roots, the current and previous
//! retained checkpoints, manifests in required WAL history, in-flight
//! staged uploads (pins), and checkpoints currently building. This module
//! takes the merged set as input; assembling it is engine work.
//!
//! ## Why concurrent staging cannot lose (§40)
//!
//! Two facts compose the argument:
//!
//! 1. Staging appends to the *active* pack only. GC never proposes the
//!    active pack (nor any pack at or above the active sequence snapshotted
//!    when the plan was computed), so bytes staged during or after a mark
//!    phase are unreachable by that phase's deletion list.
//! 2. Bytes staged *before* the mark into an older pack are covered by
//!    staging pins: uploaders pin every chunk and manifest id they intend
//!    to stage *before* the first append, and unpin after commit or abort.
//!    The plan unions pins into its roots, so a pack holding staged-but-
//!    uncommitted data is never fully dead.
//!
//! A pack is therefore proposed only when it was sealed before the plan
//! began *and* nothing reachable — live, retained, or in flight — names
//! any record in it. Deleting it cannot strand a future commit.
//!
//! ## Reclamation scope (§38, §39)
//!
//! This stage deletes only fully-dead sealed packs. Mixed live/dead packs
//! are kept; mark-copy compaction of cold packs is a later optimization
//! once pack-lifetime measurements justify it.

use std::collections::{HashMap, HashSet};

use kivi_types::{ChunkId, ManifestId};

use crate::error::ChunkError;
use crate::store::ChunkStore;

/// GC inputs assembled by the engine. Everything here is an address, never
/// bytes: the plan reads the manifests it needs through the store.
#[derive(Debug, Clone, Default)]
pub struct GcInputs {
    /// Manifests referenced by live tablet roots, retained checkpoints
    /// (current + previous), required WAL history, and building
    /// checkpoints. A missing member is corruption (fail loudly).
    pub live_manifests: HashSet<ManifestId>,
    /// In-flight staged chunk addresses (pinned before their first append,
    /// unpinned after commit or abort).
    pub pinned_chunks: HashSet<ChunkId>,
    /// In-flight staged manifest addresses (same pin discipline).
    pub pinned_manifests: HashSet<ManifestId>,
}

/// What one mark pass found.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct GcReport {
    /// Reachable chunk addresses.
    pub live_chunks: usize,
    /// Reachable manifest addresses.
    pub live_manifests: usize,
    /// Stored bytes proven reachable (records, framing included).
    pub marked_bytes: u64,
    /// Sealed pack sequences with zero reachable records: safe to delete
    /// via [`ChunkStore::delete_packs`].
    pub dead_sealed_packs: Vec<u64>,
    /// Sealed packs retained (nonempty or too new).
    pub retained_packs: usize,
}

/// Marks from `inputs` over `store` and reports fully-dead sealed packs.
///
/// Manifest bodies are read through the store (maintenance context may
/// block). A *live* manifest with no representation is corruption and
/// fails the whole plan; a merely *pinned* one may simply not be staged
/// yet and is skipped (its bytes live in packs this plan can never
/// propose — see the module docs).
///
/// # Errors
///
/// Returns [`ChunkError`] when a live root is missing or unreadable.
pub fn plan(store: &ChunkStore, inputs: &GcInputs) -> Result<GcReport, ChunkError> {
    let active_seq = store.active_seq();
    // Pins join the mark set directly: pinned chunks are staged (or about
    // to be) and pinned manifests name them, so no manifest read is needed
    // — or even possible — for not-yet-staged pins.
    let mut live_chunks: HashSet<ChunkId> = inputs.pinned_chunks.clone();
    let mut live_manifests: HashSet<ManifestId> = inputs.pinned_manifests.clone();
    live_manifests.extend(inputs.live_manifests.iter().copied());
    // Expand every live manifest one level. A live root with no
    // representation is corruption: recovery itself could not serve it.
    for id in &inputs.live_manifests {
        let (manifest, _) = store.read_manifest(*id)?;
        for entry in &manifest.entries {
            live_chunks.insert(entry.id);
        }
    }
    // Census per pack from the derived index.
    let mut live_per_pack: HashMap<u64, u64> = HashMap::new();
    let mut marked_bytes = 0u64;
    for (id, pack, _, bytes) in store.chunk_locations() {
        if live_chunks.contains(&id) {
            *live_per_pack.entry(pack).or_insert(0) += 1;
            marked_bytes += bytes;
        }
    }
    for (id, pack, _, bytes) in store.manifest_locations() {
        if live_manifests.contains(&id) {
            *live_per_pack.entry(pack).or_insert(0) += 1;
            marked_bytes += bytes;
        }
    }
    let mut dead_sealed_packs = Vec::new();
    let mut retained_packs = 0usize;
    for pack in store.pack_seqs() {
        if pack >= active_seq {
            continue;
        }
        if live_per_pack.get(&pack).copied().unwrap_or(0) == 0 {
            dead_sealed_packs.push(pack);
        } else {
            retained_packs += 1;
        }
    }
    dead_sealed_packs.sort_unstable();
    Ok(GcReport {
        live_chunks: live_chunks.len(),
        live_manifests: live_manifests.len(),
        marked_bytes,
        dead_sealed_packs,
        retained_packs,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifest::build_manifest;
    use crate::policy::{CHUNKING_V1, Chunking};
    use kivi_codec::integrity::chunk_id;
    use kivi_types::{SecurityDomainId, WallTimestamp};

    const DOMAIN: SecurityDomainId = SecurityDomainId::from_u64(5);

    fn chunking(size: usize) -> Chunking {
        Chunking {
            version: CHUNKING_V1,
            chunk_size: size,
        }
    }

    fn stage_value(store: &mut ChunkStore, byte: u8, len: usize) -> (ManifestId, u64) {
        let bytes = vec![byte; len];
        let id = chunk_id(DOMAIN, &bytes);
        store.stage_chunk(id, &bytes).expect("stages chunk");
        let (manifest, manifest_id) = build_manifest(
            DOMAIN,
            chunking(len),
            crate::codec::ChunkCodecId::NONE,
            len as u64,
            vec![crate::manifest::ChunkEntry {
                id,
                len: len as u64,
            }],
        )
        .expect("builds");
        let (canonical, recomputed) = manifest.canonical_with_id(DOMAIN);
        assert_eq!(manifest_id, recomputed);
        store
            .stage_manifest(manifest_id, &canonical)
            .expect("stages manifest");
        (manifest_id, len as u64)
    }

    #[test]
    fn dead_sealed_packs_are_collected_live_ones_kept() {
        let scratch = tempfile::tempdir().expect("scratch");
        // Tiny packs: one value per pack.
        let (mut store, _) = ChunkStore::open(
            scratch.path(),
            0,
            DOMAIN,
            512,
            WallTimestamp::from_micros(1),
        )
        .expect("opens");
        let (keep, _) = stage_value(&mut store, 1, 100);
        let (drop_me, _) = stage_value(&mut store, 2, 100);
        store.sync().expect("syncs");
        store.seal_active().expect("seals");
        // Third value keeps the active pack alive-and-excluded.
        let (live_active, _) = stage_value(&mut store, 3, 100);
        store.sync().expect("syncs");
        let inputs = GcInputs {
            live_manifests: [keep, live_active].into_iter().collect(),
            ..GcInputs::default()
        };
        let report = plan(&store, &inputs).expect("plans");
        assert_eq!(report.live_manifests, 2);
        assert_eq!(report.live_chunks, 2);
        // The dropped value's pack is the only fully-dead sealed pack.
        assert_eq!(report.dead_sealed_packs.len(), 1);
        let deleted = store
            .delete_packs(&report.dead_sealed_packs)
            .expect("deletes");
        assert_eq!(deleted, report.dead_sealed_packs);
        // Survivors still read; the deleted address is gone loudly.
        assert!(store.read_manifest(keep).is_ok());
        assert!(store.read_manifest(live_active).is_ok());
        assert!(matches!(
            store.read_manifest(drop_me),
            Err(ChunkError::MissingManifest { .. })
        ));
    }

    #[test]
    fn pins_protect_staged_but_uncommitted_data() {
        let scratch = tempfile::tempdir().expect("scratch");
        let (mut store, _) = ChunkStore::open(
            scratch.path(),
            0,
            DOMAIN,
            512,
            WallTimestamp::from_micros(1),
        )
        .expect("opens");
        let (keep, _) = stage_value(&mut store, 1, 100);
        // Staged into the first pack, then sealed before any commit.
        store.sync().expect("syncs");
        store.seal_active().expect("seals");
        // The uploader staged (but has not committed) a second value whose
        // pack sealed underneath it; pins keep it alive anyway.
        let in_flight_bytes = vec![9u8; 100];
        let in_flight_chunk = chunk_id(DOMAIN, &in_flight_bytes);
        store
            .stage_chunk(in_flight_chunk, &in_flight_bytes)
            .expect("stages");
        store.sync().expect("syncs");
        store.seal_active().expect("seals");
        let inputs = GcInputs {
            live_manifests: [keep].into_iter().collect(),
            pinned_chunks: [in_flight_chunk].into_iter().collect(),
            pinned_manifests: HashSet::new(),
        };
        let report = plan(&store, &inputs).expect("plans");
        assert!(
            report.dead_sealed_packs.is_empty(),
            "pinned data must hold its pack, got {:?}",
            report.dead_sealed_packs
        );
        // Without the pin the same pack would be collected.
        let unpinned = GcInputs {
            live_manifests: [keep].into_iter().collect(),
            ..GcInputs::default()
        };
        let report = plan(&store, &unpinned).expect("plans");
        assert_eq!(report.dead_sealed_packs.len(), 1);
    }

    #[test]
    fn missing_live_root_fails_the_plan_loudly() {
        let scratch = tempfile::tempdir().expect("scratch");
        let (store, _) = ChunkStore::open(
            scratch.path(),
            0,
            DOMAIN,
            1024 * 1024,
            WallTimestamp::from_micros(1),
        )
        .expect("opens");
        let ghost = ManifestId::from_bytes([0x41; 32]);
        let inputs = GcInputs {
            live_manifests: [ghost].into_iter().collect(),
            ..GcInputs::default()
        };
        assert!(matches!(
            plan(&store, &inputs),
            Err(ChunkError::MissingManifest { .. })
        ));
    }
}
