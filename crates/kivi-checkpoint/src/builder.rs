//! Tablet checkpoint builder: snapshot to immutable artifacts.
//!
//! Pure function of its inputs (no I/O, no threads): the engine runs it
//! on a background thread while `DataWorkers` keep serving. Incremental
//! rule: a clean band is reused descriptor-for-descriptor from the
//! previous manifest (byte-for-byte identical — committed state for its
//! keys provably unchanged); a dirty band is rebuilt even when empty (an
//! emptied band is explicit state, never inferred absence).
//!
//! First checkpoints build all bands (mostly empty); later ones rewrite
//! only what changed. Cut discipline: reused band cuts never lead the new
//! manifest cut (enforced at manifest build); a regressing previous cut
//! fails the build loudly instead of forking history.

use std::collections::HashMap;

use kivi_state::PartitionHasher;
use kivi_types::{ClusterId, NamespaceId, NodeId, NodeIncarnation};

use crate::artifact::ArtifactHash;
use crate::band::{BandRecord, BuiltBand, CompressionCodecId, build_band};
use crate::dedup::{BuiltDedup, build_dedup};
use crate::error::CheckpointError;
use crate::layout::{BANDS_PER_TABLET, BandLayout};
use crate::manifest::{
    BandDescriptor, BuiltManifest, DedupRef, LoadedManifest, ManifestInput, build_manifest,
};
use crate::snapshot::TabletSnapshot;

/// Builder policy: layout plus body codec. The codec default is NONE;
/// LZ4 is selected explicitly for the compression experiment only.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CheckpointPolicy {
    /// Physical band layout.
    pub layout: BandLayout,
    /// Band body codec.
    pub codec: CompressionCodecId,
}

impl CheckpointPolicy {
    /// Production default: V1 layout, no compression.
    pub const DEFAULT: Self = Self {
        layout: BandLayout::V1,
        codec: CompressionCodecId::NONE,
    };
}

impl Default for CheckpointPolicy {
    fn default() -> Self {
        Self::DEFAULT
    }
}

/// Checkpoint publisher identity (stamped into the manifest).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CheckpointIdentity {
    /// Owning cluster.
    pub cluster: ClusterId,
    /// Owning node.
    pub node: NodeId,
    /// Writer incarnation (forensics).
    pub incarnation: NodeIncarnation,
    /// Namespace served.
    pub namespace: NamespaceId,
    /// Tablet checkpointed.
    pub tablet: kivi_types::TabletId,
}

/// One band of a built checkpoint: fresh bytes or a reused descriptor.
#[derive(Debug, Clone)]
pub enum BuiltBandRef {
    /// Freshly built band bytes (publish unless present).
    New(BuiltBand),
    /// Byte-identical reuse from the previous manifest (skip the write).
    Reused(BandDescriptor),
}

impl BuiltBandRef {
    /// The descriptor the manifest references.
    #[must_use]
    pub fn descriptor(&self, band: u16, cut: u64) -> BandDescriptor {
        match self {
            Self::New(built) => BandDescriptor {
                band,
                hash: built.hash,
                len: built.bytes.len() as u64,
                records: built.records,
                codec: built.codec,
                cut,
            },
            Self::Reused(descriptor) => descriptor.clone(),
        }
    }
}

/// One built tablet checkpoint: artifacts plus manifest plus stats.
#[derive(Debug, Clone)]
pub struct BuiltTablet {
    /// Tablet checkpointed.
    pub tablet: kivi_types::TabletId,
    /// Checkpoint cut.
    pub cut: u64,
    /// All bands in band order (new or reused).
    pub bands: Vec<BuiltBandRef>,
    /// Fresh dedup component (always rebuilt; content addressing skips
    /// identical bytes at publish).
    pub dedup: BuiltDedup,
    /// Manifest binding the cut to its artifacts.
    pub manifest: BuiltManifest,
    /// Build accounting.
    pub stats: BuildStats,
}

/// Build accounting for operators and benchmarks.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct BuildStats {
    /// Bands rebuilt (dirty).
    pub bands_new: usize,
    /// Bands reused byte-for-byte.
    pub bands_reused: usize,
    /// Objects encoded.
    pub objects: usize,
    /// Sessions encoded.
    pub sessions: usize,
    /// Outcomes encoded.
    pub outcomes: usize,
    /// Uncompressed band body bytes (new bands).
    pub raw_bytes: u64,
    /// Stored band file bytes (new bands, framing included).
    pub stored_bytes: u64,
}

/// Builds one tablet checkpoint from a capture view.
///
/// `previous` carries the last published manifest plus its content hash
/// (the retention-chain link). Clean bands reuse its descriptors; a
/// missing descriptor for a clean band fails loudly (the previous
/// manifest is incomplete — never fall back to rebuilding silently, which
/// would fork band identity).
///
/// # Errors
///
/// Returns [`CheckpointError`] on unmappable keys, cut regression,
/// incomplete previous manifests, or encoding failures.
// Long linear orchestration (group, build, reference, manifest): the
// stages read top to bottom; splitting would scatter the pipeline.
#[allow(clippy::too_many_lines)]
pub fn build_tablet(
    snapshot: &TabletSnapshot,
    previous: Option<(&LoadedManifest, ArtifactHash)>,
    policy: &CheckpointPolicy,
    identity: &CheckpointIdentity,
) -> Result<BuiltTablet, CheckpointError> {
    if identity.namespace != snapshot.namespace || identity.tablet != snapshot.tablet {
        return Err(CheckpointError::Format {
            detail: "checkpoint identity does not match the snapshot".to_owned(),
        });
    }
    if let Some((manifest, _)) = previous {
        if manifest.cut > snapshot.cut.as_u64() {
            return Err(CheckpointError::Format {
                detail: format!(
                    "checkpoint cut regressed: previous {} exceeds snapshot {}",
                    manifest.cut,
                    snapshot.cut.as_u64()
                ),
            });
        }
        if manifest.tablet != snapshot.tablet {
            return Err(CheckpointError::Format {
                detail: "previous manifest names a different tablet".to_owned(),
            });
        }
    }
    // Group objects into bands (partition hash computed once per key).
    let mut grouped: HashMap<u16, Vec<BandRecord>> = HashMap::new();
    for (key, object) in &snapshot.objects {
        let hash = PartitionHasher::V1
            .hash(snapshot.namespace, key.as_bytes())
            .ok_or_else(|| CheckpointError::Format {
                detail: "partition hash unavailable for checkpoint banding".to_owned(),
            })?;
        let band = policy
            .layout
            .band_of(snapshot.namespace, key.as_bytes())
            .ok_or_else(|| CheckpointError::Unsupported {
                detail: format!("checkpoint band layout {}", policy.layout.id()),
            })?;
        grouped.entry(band).or_default().push(BandRecord {
            hash: hash.as_u128(),
            key: key.clone(),
            object: object.clone(),
        });
    }
    let mut bands = Vec::with_capacity(BANDS_PER_TABLET);
    let mut stats = BuildStats {
        objects: snapshot.object_count(),
        sessions: snapshot.sessions.len(),
        outcomes: snapshot.outcome_count(),
        ..BuildStats::default()
    };
    for band in 0..crate::layout::BAND_COUNT_U16 {
        let dirty = snapshot.dirty.is_dirty(band);
        let reference = match previous {
            Some((manifest, _)) => manifest.bands.iter().find(|entry| entry.band == band),
            None => None,
        };
        if !dirty {
            if let Some(descriptor) = reference {
                stats.bands_reused += 1;
                bands.push(BuiltBandRef::Reused(descriptor.clone()));
                continue;
            }
            if previous.is_some() {
                return Err(CheckpointError::Format {
                    detail: format!("previous manifest is missing clean band {band}"),
                });
            }
        }
        let records = grouped.remove(&band).unwrap_or_default();
        let built = build_band(
            snapshot.tablet,
            snapshot.cut.as_u64(),
            band,
            policy.layout.id(),
            policy.codec,
            records,
        )?;
        stats.bands_new += 1;
        stats.raw_bytes += built.raw_bytes;
        stats.stored_bytes += built.bytes.len() as u64;
        bands.push(BuiltBandRef::New(built));
    }
    let dedup = build_dedup(
        snapshot.tablet,
        snapshot.cut.as_u64(),
        snapshot.sessions.clone(),
        snapshot.intents.clone(),
    )?;
    let descriptors: Vec<BandDescriptor> = bands
        .iter()
        .zip(0..crate::layout::BAND_COUNT_U16)
        .map(|(reference, band)| reference.descriptor(band, snapshot.cut.as_u64()))
        .collect();
    let manifest = build_manifest(ManifestInput {
        cluster: identity.cluster,
        node: identity.node,
        incarnation: identity.incarnation,
        namespace: snapshot.namespace,
        tablet: snapshot.tablet,
        epoch: snapshot.epoch,
        guard: snapshot.guard,
        cut: snapshot.cut.as_u64(),
        layout: policy.layout.id(),
        bands: descriptors,
        dedup: DedupRef {
            hash: dedup.hash,
            len: dedup.bytes.len() as u64,
            sessions: dedup.sessions,
            outcomes: u32::try_from(dedup.outcomes).map_err(|_| CheckpointError::Format {
                detail: "dedup component exceeds u32 outcomes".to_owned(),
            })?,
        },
        previous: previous.map(|(_, hash)| hash),
    })?;
    Ok(BuiltTablet {
        tablet: snapshot.tablet,
        cut: snapshot.cut.as_u64(),
        bands,
        dedup,
        manifest,
        stats,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use kivi_state::{Key, LogicalValue, ObjectVersion, StoredObject};
    use kivi_types::{
        CommitPosition, Expiry, RequestSeq, SessionId, TabletEpoch, TabletId, WriteGuardGeneration,
    };

    use crate::dedup::OutcomeCheckpoint;
    use crate::layout::BandDirty;
    use crate::manifest::load_manifest;

    const NS: NamespaceId = NamespaceId::from_u64(11);
    fn identity() -> CheckpointIdentity {
        CheckpointIdentity {
            cluster: ClusterId::from_u128(1),
            node: NodeId::from_u64(2),
            incarnation: NodeIncarnation::from_u64(3),
            namespace: NS,
            tablet: TabletId::from_u64(1),
        }
    }

    fn object(value: i64) -> StoredObject {
        StoredObject::restore(
            LogicalValue::StrictCounter(value),
            ObjectVersion::FIRST,
            Expiry::NEVER,
        )
    }

    fn snapshot(
        entries: Vec<(&str, StoredObject)>,
        dirty_bands: &[u16],
        cut: u64,
    ) -> TabletSnapshot {
        let mut dirty = BandDirty::clean();
        for band in dirty_bands {
            dirty.mark(*band);
        }
        TabletSnapshot {
            namespace: NS,
            tablet: TabletId::from_u64(1),
            epoch: TabletEpoch::INITIAL,
            guard: WriteGuardGeneration::INITIAL,
            cut: CommitPosition::from_u64(cut),
            objects: entries
                .into_iter()
                .map(|(key, object)| (Key::from(key), object))
                .collect(),
            sessions: Vec::new(),
            intents: Vec::new(),
            dirty,
        }
    }

    fn band_of(key: &str) -> u16 {
        BandLayout::V1
            .band_of(NS, key.as_bytes())
            .expect("mappable")
    }

    #[test]
    fn first_checkpoint_builds_all_bands() {
        let built = build_tablet(
            &snapshot(vec![("a", object(1)), ("b", object(2))], &[], 7),
            None,
            &CheckpointPolicy::DEFAULT,
            &identity(),
        )
        .expect("builds");
        assert_eq!(built.bands.len(), BANDS_PER_TABLET);
        assert_eq!(built.stats.objects, 2);
        // No previous: everything is new (dirty flags are meaningless
        // without a baseline, so all bands build).
        assert_eq!(built.stats.bands_new, BANDS_PER_TABLET);
        assert_eq!(built.stats.bands_reused, 0);
        let loaded = load_manifest(built.manifest.hash, &built.manifest.bytes).expect("loads");
        assert_eq!(loaded.cut, 7);
    }

    #[test]
    fn incremental_checkpoint_reuses_clean_bands_hash_for_hash() {
        let first = build_tablet(
            &snapshot(vec![("a", object(1)), ("b", object(2))], &[], 7),
            None,
            &CheckpointPolicy::DEFAULT,
            &identity(),
        )
        .expect("builds");
        let first_manifest =
            load_manifest(first.manifest.hash, &first.manifest.bytes).expect("loads");
        // Mutate exactly one band: only that band is dirty at the new cut.
        let dirty = band_of("a");
        assert_ne!(band_of("b"), dirty, "test needs two bands");
        let second = build_tablet(
            &snapshot(vec![("a", object(10)), ("b", object(2))], &[dirty], 9),
            Some((&first_manifest, first.manifest.hash)),
            &CheckpointPolicy::DEFAULT,
            &identity(),
        )
        .expect("builds");
        assert_eq!(second.stats.bands_new, 1);
        assert_eq!(second.stats.bands_reused, BANDS_PER_TABLET - 1);
        // Reused descriptors are identical references (same hash).
        for (index, reference) in second.bands.iter().enumerate() {
            match reference {
                BuiltBandRef::New(_) => {
                    assert_eq!(u16::try_from(index).expect("band index fits u16"), dirty);
                }
                BuiltBandRef::Reused(descriptor) => {
                    assert_eq!(descriptor.hash, first_manifest.bands[index].hash);
                    assert_eq!(descriptor.cut, 7, "reused bands keep their seal cut");
                }
            }
        }
        let loaded = load_manifest(second.manifest.hash, &second.manifest.bytes).expect("loads");
        assert_eq!(loaded.cut, 9);
        assert_eq!(loaded.previous, Some(first.manifest.hash));
    }

    #[test]
    fn cut_regression_fails_loudly() {
        let first = build_tablet(
            &snapshot(vec![], &[], 9),
            None,
            &CheckpointPolicy::DEFAULT,
            &identity(),
        )
        .expect("builds");
        let first_manifest =
            load_manifest(first.manifest.hash, &first.manifest.bytes).expect("loads");
        assert!(
            build_tablet(
                &snapshot(vec![], &[], 8),
                Some((&first_manifest, first.manifest.hash)),
                &CheckpointPolicy::DEFAULT,
                &identity(),
            )
            .is_err(),
            "history must not fork backwards"
        );
    }

    #[test]
    fn missing_clean_band_in_previous_fails_loudly() {
        let first = build_tablet(
            &snapshot(vec![], &[], 7),
            None,
            &CheckpointPolicy::DEFAULT,
            &identity(),
        )
        .expect("builds");
        let mut incomplete =
            load_manifest(first.manifest.hash, &first.manifest.bytes).expect("loads");
        incomplete.bands.pop();
        assert!(
            build_tablet(
                &snapshot(vec![], &[], 8),
                Some((&incomplete, first.manifest.hash)),
                &CheckpointPolicy::DEFAULT,
                &identity(),
            )
            .is_err(),
            "incomplete previous manifests never silently rebuild"
        );
    }

    #[test]
    fn dedup_state_rides_the_checkpoint() {
        let mut snap = snapshot(vec![], &[], 5);
        snap.sessions.push(crate::dedup::SessionCheckpoint {
            session: SessionId::from_u128(0xD0),
            floor: RequestSeq::from_u64(4),
            outcomes: vec![OutcomeCheckpoint {
                seq: RequestSeq::from_u64(5),
                commit: CommitPosition::from_u64(5),
                opcode: 6,
                outcome: kivi_state::DurableOutcome::Completed(
                    kivi_state::OperationResult::CounterUpdated {
                        value: 1,
                        version: ObjectVersion::FIRST,
                    },
                ),
            }],
        });
        let built =
            build_tablet(&snap, None, &CheckpointPolicy::DEFAULT, &identity()).expect("builds");
        assert_eq!(built.stats.sessions, 1);
        assert_eq!(built.stats.outcomes, 1);
        let loaded = load_manifest(built.manifest.hash, &built.manifest.bytes).expect("loads");
        assert_eq!(loaded.dedup.sessions, 1);
    }

    /// Deterministic PRNG (splitmix64) for representative incompressible
    /// bytes — no test-only RNG dependency, same bytes every run.
    fn splitmix_next(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Representative tablet state (not zeros): one third compressible
    /// JSON-like text, one third deterministic incompressible bytes, one
    /// third counters. Spans all 256 bands via the V1 layout.
    fn representative_snapshot(count: usize, cut: u64) -> TabletSnapshot {
        use kivi_state::LogicalValue;
        let mut objects = Vec::with_capacity(count);
        let mut rng = 0x1234_5678_9ABC_DEF0u64;
        for i in 0..count {
            let key = Key::from(format!("user:{i:05}"));
            let object = match i % 3 {
                0 => {
                    // Compressible: repeated structured text (~180B).
                    let text = format!(
                        "{{\"id\":{i},\"name\":\"user-{i}\",\"role\":\"member\",\"active\":true,\"tags\":[\"a\",\"b\",\"c\"]}}"
                    )
                    .repeat(2);
                    StoredObject::restore(
                        LogicalValue::Bytes(bytes::Bytes::copy_from_slice(text.as_bytes())),
                        ObjectVersion::FIRST,
                        Expiry::NEVER,
                    )
                }
                1 => {
                    // Incompressible: deterministic pseudo-random 96B.
                    let mut bytes = vec![0u8; 96];
                    for chunk in bytes.chunks_mut(8) {
                        chunk
                            .copy_from_slice(&splitmix_next(&mut rng).to_le_bytes()[..chunk.len()]);
                    }
                    StoredObject::restore(
                        LogicalValue::Bytes(bytes::Bytes::from(bytes)),
                        ObjectVersion::FIRST,
                        Expiry::NEVER,
                    )
                }
                _ => StoredObject::restore(
                    LogicalValue::StrictCounter(i64::try_from(i).expect("test count fits i64")),
                    ObjectVersion::FIRST,
                    Expiry::NEVER,
                ),
            };
            objects.push((key, object));
        }
        TabletSnapshot {
            namespace: NS,
            tablet: TabletId::from_u64(1),
            epoch: TabletEpoch::INITIAL,
            guard: WriteGuardGeneration::INITIAL,
            cut: CommitPosition::from_u64(cut),
            objects,
            sessions: Vec::new(),
            intents: Vec::new(),
            dirty: BandDirty::clean(),
        }
    }

    #[test]
    #[allow(clippy::too_many_lines)]
    fn lz4_vs_none_on_representative_state() {
        use std::time::Instant;

        let snapshot = representative_snapshot(3000, 3000);
        let logical_bytes: u64 = snapshot
            .objects
            .iter()
            .map(|(key, object)| {
                key.as_bytes().len() as u64
                    + match object.value() {
                        kivi_state::LogicalValue::Bytes(value) => value.len() as u64,
                        // Chunked roots count logical bytes, not the
                        // resident reference: the benchmark sizes the
                        // logical value either representation holds.
                        kivi_state::LogicalValue::Chunked(chunked) => chunked.logical_len,
                        kivi_state::LogicalValue::StrictCounter(_) => 8,
                    }
            })
            .sum();
        // NONE baseline.
        let none_policy = CheckpointPolicy::DEFAULT;
        let started = Instant::now();
        let none = build_tablet(&snapshot, None, &none_policy, &identity()).expect("none builds");
        let none_build = started.elapsed();
        // LZ4 experiment arm (same snapshot, same cut, same layout).
        let lz4_policy = CheckpointPolicy {
            layout: crate::layout::BandLayout::V1,
            codec: crate::band::CompressionCodecId::LZ4,
        };
        let started = Instant::now();
        let lz4 = build_tablet(&snapshot, None, &lz4_policy, &identity()).expect("lz4 builds");
        let lz4_build = started.elapsed();
        // Both bind the same cut; manifests differ (codec + hashes).
        assert_eq!(none.cut, 3000);
        assert_eq!(lz4.cut, 3000);
        assert_eq!(none.bands.len(), lz4.bands.len());
        // Restore timing: manifest + every new band loads and verifies.
        let started = Instant::now();
        let none_manifest =
            load_manifest(none.manifest.hash, &none.manifest.bytes).expect("none loads");
        let mut none_bands = 0usize;
        for (reference, band) in none.bands.iter().zip(0..) {
            match reference {
                BuiltBandRef::New(built) => {
                    crate::band::load_band(
                        TabletId::from_u64(1),
                        band,
                        crate::layout::BandLayoutId::HASH_PREFIX_8,
                        built.hash,
                        &built.bytes,
                    )
                    .expect("none band loads");
                    none_bands += 1;
                }
                BuiltBandRef::Reused(_) => panic!("first checkpoint has no reuse"),
            }
        }
        let none_restore = started.elapsed();
        let started = Instant::now();
        let lz4_manifest =
            load_manifest(lz4.manifest.hash, &lz4.manifest.bytes).expect("lz4 loads");
        let mut lz4_bands = 0usize;
        for (reference, band) in lz4.bands.iter().zip(0..) {
            match reference {
                BuiltBandRef::New(built) => {
                    crate::band::load_band(
                        TabletId::from_u64(1),
                        band,
                        crate::layout::BandLayoutId::HASH_PREFIX_8,
                        built.hash,
                        &built.bytes,
                    )
                    .expect("lz4 band loads");
                    lz4_bands += 1;
                }
                BuiltBandRef::Reused(_) => panic!("first checkpoint has no reuse"),
            }
        }
        let lz4_restore = started.elapsed();
        assert_eq!(none_manifest.bands.len(), lz4_manifest.bands.len());
        assert_eq!(none_bands, lz4_bands);
        // Publish timing (end-to-end write): temp dirs, real files, sync +
        // read-back verify. Measures what operators pay on checkpoint.
        let none_dir = tempfile::tempdir().expect("scratch");
        let started = Instant::now();
        let (none_installed, none_publish) =
            crate::publish::publish_tablet(none_dir.path(), &none, None).expect("none publishes");
        let none_write = started.elapsed();
        let lz4_dir = tempfile::tempdir().expect("scratch");
        let started = Instant::now();
        let (lz4_installed, lz4_publish) =
            crate::publish::publish_tablet(lz4_dir.path(), &lz4, None).expect("lz4 publishes");
        let lz4_write = started.elapsed();
        // Disk restore: load_installed verifies the full chain from files.
        let started = Instant::now();
        let none_disk = crate::load::load_installed(none_dir.path(), TabletId::from_u64(1))
            .expect("none disk loads")
            .expect("none installed");
        let none_disk_restore = started.elapsed();
        let started = Instant::now();
        let lz4_disk = crate::load::load_installed(lz4_dir.path(), TabletId::from_u64(1))
            .expect("lz4 disk loads")
            .expect("lz4 installed");
        let lz4_disk_restore = started.elapsed();
        assert_eq!(none_disk.current.cut, none_installed.cut);
        assert_eq!(lz4_disk.current.cut, lz4_installed.cut);
        let _ = (none_publish, lz4_publish);
        // Report (use `cargo test -- --nocapture` to read): the test itself
        // asserts only correctness + the compressible win below (timings
        // are informational, never asserted — too platform-sensitive).
        println!(
            "representative 3000 objects logical={logical_bytes}B \
             none stored={}B raw={}B build={:?} restore={:?} write={:?} disk_restore={:?} | \
             lz4 stored={}B raw={}B build={:?} restore={:?} write={:?} disk_restore={:?}",
            none.stats.stored_bytes,
            none.stats.raw_bytes,
            none_build,
            none_restore,
            none_write,
            none_disk_restore,
            lz4.stats.stored_bytes,
            lz4.stats.raw_bytes,
            lz4_build,
            lz4_restore,
            lz4_write,
            lz4_disk_restore,
        );
        // Mixed representative state must still shrink (compressible third
        // outweighs incompressible overhead + framing).
        assert!(
            lz4.stats.stored_bytes < none.stats.stored_bytes,
            "lz4 {} should beat none {} on representative state",
            lz4.stats.stored_bytes,
            none.stats.stored_bytes
        );
    }

    #[test]
    #[allow(clippy::too_many_lines)]
    fn incremental_dirty_band_costs() {
        use std::time::Instant;

        // Smaller representative set for fast incremental timing (1000
        // objects still spans all 256 bands uniformly).
        let base = representative_snapshot(1000, 1000);
        let first = build_tablet(&base, None, &CheckpointPolicy::DEFAULT, &identity())
            .expect("first builds");
        assert_eq!(first.stats.bands_new, 256);
        let first_manifest =
            load_manifest(first.manifest.hash, &first.manifest.bytes).expect("first loads");
        let dir = tempfile::tempdir().expect("scratch");
        let (mut current, _) =
            crate::publish::publish_tablet(dir.path(), &first, None).expect("first publishes");

        // Helper: clone base objects, bump values in `dirty_bands` distinct
        // bands, mark exactly those bands dirty at the new cut.
        let plan = [(1usize, 1001u64), (26usize, 1002u64), (256usize, 1003u64)];
        for (wanted_dirty, cut) in plan {
            // Pick one key per wanted band (distinct bands via layout).
            let mut seen_bands = std::collections::HashSet::new();
            let mut dirty = BandDirty::clean();
            let mut objects = base.objects.clone();
            // Mutate objects until `wanted_dirty` distinct bands covered
            // (1000 objects over 256 bands ≈ 4 per band, so this terminates;
            // 256 wants every band — mark all directly).
            if wanted_dirty >= 256 {
                for band in 0..crate::layout::BAND_COUNT_U16 {
                    dirty.mark(band);
                }
                // Bump every counter so dirty bands actually differ.
                for (_, object) in objects.iter_mut().step_by(3) {
                    if let kivi_state::LogicalValue::StrictCounter(value) = object.value() {
                        *object = kivi_state::StoredObject::restore(
                            kivi_state::LogicalValue::StrictCounter(value + 1),
                            object.version(),
                            object.expiry(),
                        );
                    }
                }
            } else {
                for (key, object) in &mut objects {
                    if seen_bands.len() >= wanted_dirty {
                        break;
                    }
                    let band = crate::layout::BandLayout::V1
                        .band_of(NS, key.as_bytes())
                        .expect("mappable");
                    if seen_bands.insert(band) {
                        dirty.mark(band);
                        // Bump the object so the band content differs.
                        if let kivi_state::LogicalValue::StrictCounter(value) = object.value() {
                            *object = kivi_state::StoredObject::restore(
                                kivi_state::LogicalValue::StrictCounter(value + 1000),
                                object.version(),
                                object.expiry(),
                            );
                        } else if let kivi_state::LogicalValue::Bytes(value) = object.value() {
                            let mut bytes = value.to_vec();
                            bytes.extend_from_slice(b"+");
                            *object = kivi_state::StoredObject::restore(
                                kivi_state::LogicalValue::Bytes(bytes::Bytes::from(bytes)),
                                object.version(),
                                object.expiry(),
                            );
                        }
                    }
                }
                assert_eq!(
                    seen_bands.len(),
                    wanted_dirty,
                    "enough distinct bands present"
                );
            }
            let snapshot = TabletSnapshot {
                namespace: NS,
                tablet: TabletId::from_u64(1),
                epoch: TabletEpoch::INITIAL,
                guard: WriteGuardGeneration::INITIAL,
                cut: CommitPosition::from_u64(cut),
                objects,
                sessions: Vec::new(),
                intents: Vec::new(),
                dirty,
            };
            let previous = (
                load_manifest(
                    current.manifest,
                    &std::fs::read(crate::artifact::artifact_path(
                        dir.path(),
                        TabletId::from_u64(1),
                        crate::artifact::ArtifactKind::Manifest,
                        current.manifest,
                    ))
                    .expect("manifest file"),
                )
                .expect("previous loads"),
                current.manifest,
            );
            // Sanity: previous manifest for reuse must match the installed
            // chain head (first iteration: the first build; later: the last
            // incremental). We rebuild the chain via `current` each time.
            let _ = &first_manifest;
            let started = Instant::now();
            let built = build_tablet(
                &snapshot,
                Some((&previous.0, previous.1)),
                &CheckpointPolicy::DEFAULT,
                &identity(),
            )
            .expect("incremental builds");
            let build_time = started.elapsed();
            assert_eq!(
                built.stats.bands_new, wanted_dirty,
                "exactly the dirty bands rebuild"
            );
            assert_eq!(
                built.stats.bands_reused,
                256 - wanted_dirty,
                "clean bands reuse"
            );
            let started = Instant::now();
            let (installed, stats) =
                crate::publish::publish_tablet(dir.path(), &built, Some(&current))
                    .expect("incremental publishes");
            let publish_time = started.elapsed();
            println!(
                "incremental dirty={wanted_dirty} cut={cut} new_bytes={} reused={} \
                 build={build_time:?} publish={publish_time:?} written={} skipped={}",
                built.stats.stored_bytes,
                built.stats.bands_reused,
                stats.bands_written,
                stats.bands_skipped_present,
            );
            current = installed;
        }
    }
}
