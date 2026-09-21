//! Lane-local immutable chunk store: packs, derived index, staging.
//!
//! One [`ChunkStore`] owns one lane directory (`chunks/lane-XXXX/`): it
//! appends staged records to the active pack, syncs them in batches, and
//! serves verified reads from a derived in-memory index. The index is a
//! cache of pack contents, never the truth: startup rebuilds it from a
//! full scan, so no second crash-consistent database exists.
//!
//! Ownership mirrors the rest of Kivi: one lane per worker, single-threaded
//! access (the owner drives staging, syncs, reads, and GC planning; the
//! engine's async read lane only borrows verified decoding).

use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};

use bytes::Bytes;
use kivi_types::{ChunkId, ManifestId, SecurityDomainId};

use crate::error::ChunkError;
use crate::manifest::{ChunkManifest, verify_manifest};
use crate::pack::{
    PACK_HEADER_LEN, PACK_SIZE_SLACK, RecordId, ScannedRecord, append_record, decode_pack_header,
    encode_chunk_record, encode_manifest_record, encode_pack_header, lane_dir_name, list_packs,
    pack_path, scan_pack,
};

/// Proof that a chunk survived the durability barrier: only
/// [`ChunkStore::durable_chunk`] mints it, and only for index entries at or
/// below the last synced generation. Carries the address, never the bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DurableChunk(ChunkId);

/// Proof that a manifest survived the durability barrier. See
/// [`DurableChunk`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DurableManifest(ManifestId);

impl DurableChunk {
    /// Returns the proven address.
    #[must_use]
    pub const fn id(self) -> ChunkId {
        self.0
    }
}

impl DurableManifest {
    /// Returns the proven address.
    #[must_use]
    pub const fn id(self) -> ManifestId {
        self.0
    }
}

/// Where one verified record lives.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Location {
    /// Pack sequence holding the record.
    pack: u64,
    /// Byte offset of the record start.
    offset: u64,
    /// Full record bytes (header + body + CRC).
    total: u64,
    /// Logical bytes (chunk body length; 0 for manifests).
    logical_len: u64,
    /// Staging generation: durable once `<= durable_gen`.
    generation: u64,
}

/// Removes headerless newest files (rotation crashes between create and
/// header sync prove nothing was published) and returns the active pack
/// sequence (`0` when no headed pack exists and one must be created).
/// Mutates `seqs` in place so the caller scans exactly what remains.
fn drop_headerless_newest(
    dir: &Path,
    seqs: &mut Vec<u64>,
    summary: &mut ChunkRecovery,
) -> Result<u64, ChunkError> {
    while let Some(&seq) = seqs.last() {
        let path = pack_path(dir, seq);
        let len = path
            .metadata()
            .map_err(|error| ChunkError::io("stat chunk pack", &path, &error))?
            .len();
        if len >= PACK_HEADER_LEN as u64 {
            return Ok(seq);
        }
        std::fs::remove_file(&path)
            .map_err(|error| ChunkError::io("remove torn headerless pack", &path, &error))?;
        let _ = kivi_durability::fs::sync_dir(dir);
        seqs.pop();
        summary.truncated_bytes += len;
    }
    Ok(0)
}

/// Opens the active pack for appending, creating it (header, sync, dir
/// sync) when no headed pack exists. Returns the file, its sequence, and
/// its current length.
fn open_active_pack(
    dir: &Path,
    lane: u16,
    seqs: &[u64],
    active_seq: u64,
    created_at: kivi_types::WallTimestamp,
    target_bytes: u64,
) -> Result<(File, u64, u64), ChunkError> {
    if active_seq != 0 {
        let path = pack_path(dir, active_seq);
        let file = File::options()
            .read(true)
            .append(true)
            .open(&path)
            .map_err(|error| ChunkError::io("reopen active pack", &path, &error))?;
        let len = file
            .metadata()
            .map_err(|error| ChunkError::io("stat active pack", &path, &error))?
            .len();
        return Ok((file, active_seq, len));
    }
    let seq = seqs.last().copied().unwrap_or(0) + 1;
    let path = pack_path(dir, seq);
    let mut file = File::create_new(&path)
        .map_err(|error| ChunkError::io("create chunk pack", &path, &error))?;
    let header = encode_pack_header(lane, seq, created_at, target_bytes);
    file.write_all(&header)
        .map_err(|error| ChunkError::io("write pack header", &path, &error))?;
    kivi_durability::fs::sync_file(&file)
        .map_err(|error| ChunkError::io("sync pack header", &path, &error))?;
    let _ = kivi_durability::fs::sync_dir(dir);
    Ok((file, seq, PACK_HEADER_LEN as u64))
}

/// Index entries recovered from one pack file, oldest record first.
struct PackIndex {
    chunks: Vec<(ChunkId, Location)>,
    manifests: Vec<(ManifestId, Location)>,
    unique_stored_bytes: u64,
}

/// Scans one pack file, repairs a torn active tail in place, and returns
/// its index entries. Sealed damage fails; the caller removes headerless
/// newest files before calling (a rotation crash between create and
/// header sync proves nothing was published).
#[allow(clippy::too_many_arguments)]
fn recover_pack(
    dir: &Path,
    lane: u16,
    seq: u64,
    domain: SecurityDomainId,
    target_bytes: u64,
    active: bool,
    summary: &mut ChunkRecovery,
) -> Result<PackIndex, ChunkError> {
    let path = pack_path(dir, seq);
    let mut file = File::options()
        .read(true)
        .write(true)
        .open(&path)
        .map_err(|error| ChunkError::io("open chunk pack", &path, &error))?;
    let file_len = file
        .metadata()
        .map_err(|error| ChunkError::io("stat chunk pack", &path, &error))?
        .len();
    if file_len > target_bytes + PACK_SIZE_SLACK {
        return Err(ChunkError::TooLarge {
            len: file_len,
            max: target_bytes + PACK_SIZE_SLACK,
            context: "chunk pack file",
        });
    }
    let scan = scan_pack(&path, lane, seq, domain, file_len, active, &mut file)?;
    summary.packs_scanned += 1;
    if scan.truncated_bytes > 0 {
        // Torn active tail only (sealed packs fail inside the scan):
        // truncate to the last complete boundary and sync.
        file.set_len(scan.kept_len)
            .map_err(|error| ChunkError::io("truncate torn pack tail", &path, &error))?;
        kivi_durability::fs::sync_file(&file)
            .map_err(|error| ChunkError::io("sync truncated pack", &path, &error))?;
        summary.truncated_bytes += scan.truncated_bytes;
    }
    summary.records_verified += scan.records.len() as u64;
    let mut index = PackIndex {
        chunks: Vec::new(),
        manifests: Vec::new(),
        unique_stored_bytes: 0,
    };
    for record in scan.records {
        index.unique_stored_bytes += record.total_len;
        let location = Location {
            pack: seq,
            offset: record.offset,
            total: record.total_len,
            logical_len: record.logical_len,
            generation: 0,
        };
        match record.id {
            RecordId::Chunk(id) => index.chunks.push((id, location)),
            RecordId::Manifest(id) => index.manifests.push((id, location)),
        }
    }
    Ok(index)
}

/// Outcome of one staging call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StageOutcome {
    /// Whether bytes were physically appended (`false` = dedup hit).
    pub inserted: bool,
    /// Whether the address is durable now (dedup hit on synced data, or
    /// this call synced — staging alone never implies durability).
    pub durable: bool,
}

/// Cumulative store counters for the admin plane (§62). Owned locally,
/// snapshotted on request — never a hot-path lock.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ChunkStats {
    /// Chunk records indexed (unique addresses in this lane).
    pub chunks: u64,
    /// Manifest records indexed.
    pub manifests: u64,
    /// Pack files present (active + sealed).
    pub packs: u64,
    /// Active pack bytes on disk.
    pub active_bytes: u64,
    /// Sealed pack bytes on disk.
    pub sealed_bytes: u64,
    /// Sum of logical chunk bytes referenced.
    pub logical_chunk_bytes: u64,
    /// Sum of stored record bytes (framing included).
    pub unique_stored_bytes: u64,
    /// Staging calls answered from existing durable data.
    pub dedup_hits: u64,
    /// Logical bytes those hits saved from rewriting.
    pub dedup_bytes_saved: u64,
    /// Pack sync barriers issued.
    pub syncs: u64,
    /// Bytes staged but not yet synced.
    pub volatile_bytes: u64,
    /// Physical duplicate records skipped at recovery (same id, later pack).
    pub duplicate_records: u64,
    /// Bytes truncated from a torn active tail at recovery.
    pub truncated_bytes: u64,
}

/// What recovery rebuilt, for operators and startup timing (§56).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ChunkRecovery {
    /// Pack files scanned.
    pub packs_scanned: u64,
    /// Records verified (framing, CRCs, content hashes).
    pub records_verified: u64,
    /// Chunk addresses indexed.
    pub chunks_indexed: u64,
    /// Manifest addresses indexed.
    pub manifests_indexed: u64,
    /// Bytes truncated from a torn active tail.
    pub truncated_bytes: u64,
    /// Duplicate physical records skipped (first copy kept).
    pub duplicates_skipped: u64,
}

/// One lane's immutable chunk store. Single-threaded by construction: the
/// owning worker (or its maintenance task) is the only caller.
#[derive(Debug)]
pub struct ChunkStore {
    dir: PathBuf,
    lane: u16,
    domain: SecurityDomainId,
    target_bytes: u64,
    wall_micros: kivi_types::WallTimestamp,
    file: File,
    active_seq: u64,
    active_len: u64,
    chunks: HashMap<ChunkId, Location>,
    manifests: HashMap<ManifestId, Location>,
    current_gen: u64,
    durable_gen: u64,
    stats: ChunkStats,
    sync_dirty: bool,
}

impl ChunkStore {
    /// Opens (creating if needed) one lane directory, recovers every pack,
    /// and rebuilds the derived index. Returns the store plus what recovery
    /// did. See the module docs for the exact repair rules.
    ///
    /// # Errors
    ///
    /// Returns [`ChunkError`] on I/O failure, oversize files, or any damage
    /// outside a torn active tail.
    pub fn open(
        root: &Path,
        lane: u16,
        domain: SecurityDomainId,
        target_bytes: u64,
        wall_micros: kivi_types::WallTimestamp,
    ) -> Result<(Self, ChunkRecovery), ChunkError> {
        if target_bytes == 0 {
            return Err(ChunkError::Invalid {
                detail: "chunk pack target must be nonzero".to_owned(),
            });
        }
        let dir = root.join("chunks").join(lane_dir_name(lane));
        let _ = kivi_durability::fs::create_dir_all_sync(&dir)
            .map_err(|error| ChunkError::io("create chunk lane directory", &dir, &error))?;
        let mut seqs = list_packs(&dir)?;
        let mut summary = ChunkRecovery::default();
        // Oldest first: the first physical copy of an address wins the
        // index (all copies verify identically, so the choice is stable).
        let mut chunks: HashMap<ChunkId, Location> = HashMap::new();
        let mut manifests: HashMap<ManifestId, Location> = HashMap::new();
        let mut logical_chunk_bytes = 0u64;
        let mut unique_stored_bytes = 0u64;
        // Resolve the active pack: highest sequence present. A headerless
        // newest file proves nothing was published — remove it and fall
        // back (a rotation crash between create and header sync).
        let active_seq = drop_headerless_newest(&dir, &mut seqs, &mut summary)?;
        for &seq in &seqs {
            let active = seq == active_seq;
            let index = recover_pack(&dir, lane, seq, domain, target_bytes, active, &mut summary)?;
            unique_stored_bytes += index.unique_stored_bytes;
            // Oldest pack first, so the first physical copy of an address
            // wins the index (all copies verify identically).
            for (id, location) in index.chunks {
                if let std::collections::hash_map::Entry::Vacant(slot) = chunks.entry(id) {
                    logical_chunk_bytes += location.logical_len;
                    slot.insert(location);
                } else {
                    summary.duplicates_skipped += 1;
                }
            }
            for (id, location) in index.manifests {
                if let std::collections::hash_map::Entry::Vacant(slot) = manifests.entry(id) {
                    slot.insert(location);
                } else {
                    summary.duplicates_skipped += 1;
                }
            }
        }
        summary.chunks_indexed = chunks.len() as u64;
        summary.manifests_indexed = manifests.len() as u64;
        // Open (or create) the active pack for appending.
        let (file, active_seq, active_len) =
            open_active_pack(&dir, lane, &seqs, active_seq, wall_micros, target_bytes)?;
        // Byte accounting across present files.
        let mut sealed_bytes = 0u64;
        for &seq in &seqs {
            if seq == active_seq {
                continue;
            }
            sealed_bytes += pack_path(&dir, seq).metadata().map_or(0, |meta| meta.len());
        }
        let stats = ChunkStats {
            chunks: chunks.len() as u64,
            manifests: manifests.len() as u64,
            packs: seqs.len().max(1) as u64,
            active_bytes: active_len,
            sealed_bytes,
            logical_chunk_bytes,
            unique_stored_bytes,
            dedup_hits: 0,
            dedup_bytes_saved: 0,
            syncs: 0,
            volatile_bytes: 0,
            duplicate_records: summary.duplicates_skipped,
            truncated_bytes: summary.truncated_bytes,
        };
        tracing::info!(
            lane,
            packs = stats.packs,
            chunks = stats.chunks,
            manifests = stats.manifests,
            truncated_bytes = summary.truncated_bytes,
            duplicates = summary.duplicates_skipped,
            "chunk lane recovered"
        );
        Ok((
            Self {
                dir,
                lane,
                domain,
                target_bytes,
                wall_micros,
                file,
                active_seq,
                active_len,
                chunks,
                manifests,
                current_gen: 1,
                durable_gen: 0,
                stats,
                sync_dirty: false,
            },
            summary,
        ))
    }

    /// Returns the lane index.
    #[must_use]
    pub const fn lane(&self) -> u16 {
        self.lane
    }

    /// Returns the security domain this lane was opened under. Lanes are
    /// single-domain: every id minted or verified here binds this domain.
    #[must_use]
    pub const fn domain(&self) -> SecurityDomainId {
        self.domain
    }

    /// Returns the active pack sequence.
    #[must_use]
    pub const fn active_seq(&self) -> u64 {
        self.active_seq
    }

    /// Returns all pack sequences present (active included), oldest first.
    /// Best-effort directory listing for GC and admin; I/O failure yields
    /// the active sequence alone rather than failing the caller.
    #[must_use]
    pub fn pack_seqs(&self) -> Vec<u64> {
        list_packs(&self.dir).unwrap_or_else(|_| vec![self.active_seq])
    }

    /// Returns a snapshot of the cumulative counters.
    #[must_use]
    pub const fn stats(&self) -> ChunkStats {
        self.stats
    }

    /// Whether a chunk address is indexed in this lane (no I/O).
    /// Recovery and GC planning use this for dependency checks; serving
    /// reads use [`read_chunk`](Self::read_chunk), which re-verifies.
    #[must_use]
    pub fn has_chunk(&self, id: &ChunkId) -> bool {
        self.chunks.contains_key(id)
    }

    /// Whether a manifest address is indexed in this lane (no I/O).
    #[must_use]
    pub fn has_manifest(&self, id: &ManifestId) -> bool {
        self.manifests.contains_key(id)
    }

    /// Whether an address is durable in this lane (indexed at or below the
    /// last synced generation).
    fn is_durable_gen(&self, generation: u64) -> bool {
        generation <= self.durable_gen
    }

    /// Stages one chunk: appends it to the active pack unless an identical
    /// address is already indexed (dedup hit, §35). Never syncs: call
    /// [`sync`](Self::sync) to make staged data durable in one barrier
    /// (§34). Cross-lane duplicates stay possible by design (§9).
    ///
    /// # Errors
    ///
    /// Returns [`ChunkError`] on oversize bodies or append I/O failure.
    pub fn stage_chunk(&mut self, id: ChunkId, bytes: &[u8]) -> Result<StageOutcome, ChunkError> {
        if let Some(location) = self.chunks.get(&id) {
            let durable = self.is_durable_gen(location.generation);
            if durable {
                self.stats.dedup_hits += 1;
                self.stats.dedup_bytes_saved += bytes.len() as u64;
            }
            return Ok(StageOutcome {
                inserted: false,
                durable,
            });
        }
        // Defensive: the address must name these exact bytes, or the
        // caller hashed wrong — never index a lie.
        debug_assert_eq!(kivi_codec::integrity::chunk_id(self.domain, bytes), id);
        let record = encode_chunk_record(id, bytes.len() as u64, bytes);
        self.append_record(&record)?;
        self.chunks.insert(
            id,
            Location {
                pack: self.active_seq,
                offset: self.active_len - record.len() as u64,
                total: record.len() as u64,
                logical_len: bytes.len() as u64,
                generation: self.current_gen,
            },
        );
        self.stats.chunks += 1;
        self.stats.logical_chunk_bytes += bytes.len() as u64;
        self.stats.unique_stored_bytes += record.len() as u64;
        self.stats.volatile_bytes += record.len() as u64;
        Ok(StageOutcome {
            inserted: true,
            durable: false,
        })
    }

    /// Stages one canonical manifest under its verified identity. Same
    /// durability contract as [`stage_chunk`](Self::stage_chunk) (§36).
    ///
    /// # Errors
    ///
    /// Returns [`ChunkError`] when the bytes do not verify under `id`, or
    /// on append I/O failure.
    pub fn stage_manifest(
        &mut self,
        id: ManifestId,
        canonical: &[u8],
    ) -> Result<StageOutcome, ChunkError> {
        if let Some(location) = self.manifests.get(&id) {
            let durable = self.is_durable_gen(location.generation);
            if durable {
                self.stats.dedup_hits += 1;
                self.stats.dedup_bytes_saved += canonical.len() as u64;
            }
            return Ok(StageOutcome {
                inserted: false,
                durable,
            });
        }
        let manifest = verify_manifest(id, canonical)?;
        if manifest.domain != self.domain {
            return Err(ChunkError::CorruptManifest {
                id,
                detail: "staged manifest names a foreign security domain".to_owned(),
            });
        }
        let record = encode_manifest_record(id, canonical);
        self.append_record(&record)?;
        self.manifests.insert(
            id,
            Location {
                pack: self.active_seq,
                offset: self.active_len - record.len() as u64,
                total: record.len() as u64,
                logical_len: 0,
                generation: self.current_gen,
            },
        );
        self.stats.manifests += 1;
        self.stats.unique_stored_bytes += record.len() as u64;
        self.stats.volatile_bytes += record.len() as u64;
        Ok(StageOutcome {
            inserted: true,
            durable: false,
        })
    }

    /// Makes everything staged so far durable with one file sync, then
    /// advances the durability generation: index entries at or below it
    /// prove durable for [`durable_chunk`](Self::durable_chunk) and
    /// [`durable_manifest`](Self::durable_manifest). Skips the sync when
    /// nothing was staged (no empty barriers).
    ///
    /// # Errors
    ///
    /// Returns [`ChunkError`] when the sync fails; staged data stays
    /// volatile and the generation does not advance.
    pub fn sync(&mut self) -> Result<(), ChunkError> {
        if !self.sync_dirty {
            return Ok(());
        }
        let path = pack_path(&self.dir, self.active_seq);
        kivi_durability::fs::sync_file(&self.file)
            .map_err(|error| ChunkError::io("sync active chunk pack", &path, &error))?;
        self.durable_gen = self.current_gen;
        self.current_gen += 1;
        self.stats.syncs += 1;
        self.stats.volatile_bytes = 0;
        self.sync_dirty = false;
        Ok(())
    }

    /// Returns the durability proof for an indexed chunk, if it is durable.
    /// The engine converts this into the small WAL root mutation only after
    /// this returns `Some`: chunks-first ordering made type-level (§10).
    #[must_use]
    pub fn durable_chunk(&self, id: ChunkId) -> Option<DurableChunk> {
        self.chunks
            .get(&id)
            .filter(|location| self.is_durable_gen(location.generation))
            .map(|_| DurableChunk(id))
    }

    /// Returns the durability proof for an indexed manifest, if durable.
    #[must_use]
    pub fn durable_manifest(&self, id: ManifestId) -> Option<DurableManifest> {
        self.manifests
            .get(&id)
            .filter(|location| self.is_durable_gen(location.generation))
            .map(|_| DurableManifest(id))
    }

    /// Reads and fully verifies one chunk: framing, CRCs, and the content
    /// hash recomputed under this lane's domain. Blocking: maintenance,
    /// tests, and the embedded path call this; the native data path uses
    /// the async lane over the same verification (never blocks its
    /// reactor on it).
    ///
    /// # Errors
    ///
    /// Returns [`ChunkError::MissingChunk`] without a representation, or
    /// [`ChunkError::CorruptChunk`] when verification fails.
    pub fn read_chunk(&self, id: ChunkId) -> Result<Bytes, ChunkError> {
        let location = self
            .chunks
            .get(&id)
            .copied()
            .ok_or(ChunkError::MissingChunk { id })?;
        let path = pack_path(&self.dir, location.pack);
        let mut file = File::options()
            .read(true)
            .open(&path)
            .map_err(|error| ChunkError::io("open chunk pack for read", &path, &error))?;
        let body = crate::pack::verify_record_at(
            &mut file,
            &path,
            &ScannedRecord {
                id: RecordId::Chunk(id),
                logical_len: 0,
                stored_len: 0,
                offset: location.offset,
                total_len: location.total,
            },
            self.domain,
            RecordId::Chunk(id),
        )?;
        Ok(Bytes::from(body))
    }

    /// Reads and fully verifies one manifest, returning the decoded
    /// manifest plus its canonical bytes. Same blocking contract as
    /// [`read_chunk`](Self::read_chunk).
    ///
    /// # Errors
    ///
    /// Returns [`ChunkError::MissingManifest`] or [`ChunkError::CorruptManifest`].
    pub fn read_manifest(&self, id: ManifestId) -> Result<(ChunkManifest, Vec<u8>), ChunkError> {
        let location = self
            .manifests
            .get(&id)
            .copied()
            .ok_or(ChunkError::MissingManifest { id })?;
        let path = pack_path(&self.dir, location.pack);
        let mut file = File::options()
            .read(true)
            .open(&path)
            .map_err(|error| ChunkError::io("open chunk pack for read", &path, &error))?;
        let body = crate::pack::verify_record_at(
            &mut file,
            &path,
            &ScannedRecord {
                id: RecordId::Manifest(id),
                logical_len: 0,
                stored_len: 0,
                offset: location.offset,
                total_len: location.total,
            },
            self.domain,
            RecordId::Manifest(id),
        )?;
        let manifest = verify_manifest(id, &body)?;
        Ok((manifest, body))
    }

    /// Seals the active pack now (checkpoint/GC-time rotation): the next
    /// staged record opens a fresh pack. No-op when the active pack holds
    /// only its header.
    ///
    /// # Errors
    ///
    /// Returns [`ChunkError`] when the new pack cannot be published.
    pub fn seal_active(&mut self) -> Result<(), ChunkError> {
        if self.active_len <= PACK_HEADER_LEN as u64 {
            return Ok(());
        }
        self.rotate()
    }

    /// Deletes sealed packs by sequence (never the active tail). Missing
    /// files are already-gone (idempotent success); other I/O failures
    /// propagate so the GC planner keeps the pack. Index entries that lose
    /// their last representation are dropped.
    ///
    /// # Errors
    ///
    /// Returns [`ChunkError`] on refusal (active tail) or undeletable files.
    pub fn delete_packs(&mut self, seqs: &[u64]) -> Result<Vec<u64>, ChunkError> {
        let mut deleted = Vec::new();
        for &seq in seqs {
            if seq == self.active_seq {
                return Err(ChunkError::Invalid {
                    detail: "chunk GC must never remove the active pack".to_owned(),
                });
            }
            let path = pack_path(&self.dir, seq);
            match std::fs::remove_file(&path) {
                Ok(()) => {}
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                Err(error) => {
                    return Err(ChunkError::io("delete reclaimed chunk pack", &path, &error));
                }
            }
            deleted.push(seq);
        }
        if !deleted.is_empty() {
            let _ = kivi_durability::fs::sync_dir(&self.dir);
            let gone: HashSet<u64> = deleted.iter().copied().collect();
            self.chunks
                .retain(|_, location| !gone.contains(&location.pack));
            self.manifests
                .retain(|_, location| !gone.contains(&location.pack));
            self.stats.packs = self.stats.packs.saturating_sub(deleted.len() as u64);
            self.stats.chunks = self.chunks.len() as u64;
            self.stats.manifests = self.manifests.len() as u64;
            // Byte accounting is best-effort (deleted sizes are gone with
            // the files); recompute sealed bytes from what remains.
            self.stats.sealed_bytes = 0;
            for &seq in &self.pack_seqs() {
                if seq != self.active_seq {
                    self.stats.sealed_bytes += pack_path(&self.dir, seq)
                        .metadata()
                        .map_or(0, |meta| meta.len());
                }
            }
            self.stats.active_bytes = self.active_len;
        }
        Ok(deleted)
    }

    /// Iterates chunk addresses with pack, staging generation, and record
    /// bytes (GC census input).
    pub fn chunk_locations(&self) -> impl Iterator<Item = (ChunkId, u64, u64, u64)> + '_ {
        self.chunks
            .iter()
            .map(|(id, location)| (*id, location.pack, location.generation, location.total))
    }

    /// Iterates manifest addresses with pack, staging generation, and
    /// record bytes.
    pub fn manifest_locations(&self) -> impl Iterator<Item = (ManifestId, u64, u64, u64)> + '_ {
        self.manifests
            .iter()
            .map(|(id, location)| (*id, location.pack, location.generation, location.total))
    }

    /// Appends one framed record to the active pack, rotating first when it
    /// would overflow the target (never mid-record).
    fn append_record(&mut self, record: &[u8]) -> Result<(), ChunkError> {
        if self.active_len > PACK_HEADER_LEN as u64
            && self.active_len + record.len() as u64 > self.target_bytes
        {
            self.rotate()?;
        }
        let path = pack_path(&self.dir, self.active_seq);
        append_record(&mut self.file, &path, record)?;
        self.active_len += record.len() as u64;
        self.stats.active_bytes = self.active_len;
        self.sync_dirty = true;
        Ok(())
    }

    /// Seals the active pack and opens the next sequence: create, header,
    /// sync file, sync directory, switch.
    fn rotate(&mut self) -> Result<(), ChunkError> {
        let seq = self.active_seq + 1;
        let path = pack_path(&self.dir, seq);
        let mut file = File::create_new(&path)
            .map_err(|error| ChunkError::io("create chunk pack", &path, &error))?;
        let header = encode_pack_header(self.lane, seq, self.wall_micros, self.target_bytes);
        file.write_all(&header)
            .map_err(|error| ChunkError::io("write pack header", &path, &error))?;
        kivi_durability::fs::sync_file(&file)
            .map_err(|error| ChunkError::io("sync pack header", &path, &error))?;
        let _ = kivi_durability::fs::sync_dir(&self.dir);
        self.file = file;
        self.active_seq = seq;
        self.active_len = PACK_HEADER_LEN as u64;
        self.stats.packs += 1;
        self.stats.active_bytes = self.active_len;
        tracing::debug!(lane = self.lane, pack = seq, "chunk pack rotated");
        Ok(())
    }
}

/// Validates that a pack header on disk names the expected lane/sequence.
/// Thin wrapper for engine-level checks without a full scan.
///
/// # Errors
///
/// Returns [`ChunkError`] on I/O failure or header mismatch.
pub fn check_pack_header(dir: &Path, lane: u16, seq: u64) -> Result<(), ChunkError> {
    use std::io::Read as _;
    let path = pack_path(dir, seq);
    let mut file = File::options()
        .read(true)
        .open(&path)
        .map_err(|error| ChunkError::io("open chunk pack header", &path, &error))?;
    let mut header = [0u8; PACK_HEADER_LEN];
    file.read_exact(&mut header)
        .map_err(|error| ChunkError::io("read chunk pack header", &path, &error))?;
    decode_pack_header(&header, lane, seq)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifest::build_manifest;
    use crate::policy::{Chunking, DEFAULT_CHUNK_SIZE};
    use kivi_codec::integrity::chunk_id;
    use kivi_types::WallTimestamp;

    const DOMAIN: SecurityDomainId = SecurityDomainId::from_u64(3);

    fn open_store(dir: &Path) -> ChunkStore {
        ChunkStore::open(
            dir,
            0,
            DOMAIN,
            1024 * 1024,
            WallTimestamp::from_micros(1_000_000),
        )
        .expect("store opens")
        .0
    }

    #[test]
    fn stage_sync_then_prove_durability_ordering() {
        let scratch = tempfile::tempdir().expect("scratch");
        let mut store = open_store(scratch.path());
        let bytes = vec![7u8; 4096];
        let id = chunk_id(DOMAIN, &bytes);
        // Staged but unsynced: no durability proof may exist.
        let staged = store.stage_chunk(id, &bytes).expect("stages");
        assert!(staged.inserted);
        assert!(!staged.durable);
        assert!(store.durable_chunk(id).is_none());
        // One barrier makes everything staged so far durable.
        store.sync().expect("syncs");
        assert!(store.durable_chunk(id).is_some());
        // Restaging the same address is a dedup hit on durable data.
        let again = store.stage_chunk(id, &bytes).expect("restages");
        assert!(!again.inserted);
        assert!(again.durable);
        assert_eq!(store.stats().dedup_hits, 1);
        assert_eq!(store.stats().dedup_bytes_saved, 4096);
    }

    #[test]
    fn reads_verify_and_missing_reads_loudly() {
        let scratch = tempfile::tempdir().expect("scratch");
        let mut store = open_store(scratch.path());
        let bytes = vec![9u8; 512];
        let id = chunk_id(DOMAIN, &bytes);
        store.stage_chunk(id, &bytes).expect("stages");
        store.sync().expect("syncs");
        let back = store.read_chunk(id).expect("reads");
        assert_eq!(back.as_ref(), bytes.as_slice());
        let missing = chunk_id(DOMAIN, b"never staged");
        assert!(matches!(
            store.read_chunk(missing),
            Err(ChunkError::MissingChunk { .. })
        ));
    }

    #[test]
    fn recovery_rebuilds_the_index_and_repairs_torn_tails() {
        let scratch = tempfile::tempdir().expect("scratch");
        let mut store = open_store(scratch.path());
        let first = vec![1u8; 100];
        let second = vec![2u8; 200];
        let first_id = chunk_id(DOMAIN, &first);
        let second_id = chunk_id(DOMAIN, &second);
        store.stage_chunk(first_id, &first).expect("stages");
        store.stage_chunk(second_id, &second).expect("stages");
        store.sync().expect("syncs");
        drop(store);
        // Simulate a torn tail: append garbage past the last record.
        let pack = scratch
            .path()
            .join("chunks")
            .join("lane-0000")
            .join("00000000000000000001.pack");
        let mut raw = std::fs::read(&pack).expect("pack readable");
        raw.extend_from_slice(&[0xFF; 37]);
        std::fs::write(&pack, &raw).expect("torn");
        let (store, summary) = ChunkStore::open(
            scratch.path(),
            0,
            DOMAIN,
            1024 * 1024,
            WallTimestamp::from_micros(2_000_000),
        )
        .expect("recovers");
        assert_eq!(summary.truncated_bytes, 37);
        assert_eq!(summary.records_verified, 2);
        assert_eq!(
            store.read_chunk(first_id).expect("reads").as_ref(),
            first.as_slice()
        );
        assert_eq!(
            store.read_chunk(second_id).expect("reads").as_ref(),
            second.as_slice()
        );
    }

    #[test]
    fn sealed_damage_fails_recovery_loudly() {
        let scratch = tempfile::tempdir().expect("scratch");
        let mut store = open_store(scratch.path());
        let bytes = vec![3u8; 300];
        let id = chunk_id(DOMAIN, &bytes);
        store.stage_chunk(id, &bytes).expect("stages");
        store.sync().expect("syncs");
        store.seal_active().expect("seals");
        drop(store);
        // Corrupt a sealed body byte (past the 64-byte header).
        let pack = scratch
            .path()
            .join("chunks")
            .join("lane-0000")
            .join("00000000000000000001.pack");
        let mut raw = std::fs::read(&pack).expect("pack readable");
        raw[100] ^= 0xFF;
        std::fs::write(&pack, &raw).expect("corrupted");
        assert!(
            ChunkStore::open(
                scratch.path(),
                0,
                DOMAIN,
                1024 * 1024,
                WallTimestamp::from_micros(3_000_000)
            )
            .is_err()
        );
    }

    #[test]
    fn manifests_stage_verify_and_recover() {
        let scratch = tempfile::tempdir().expect("scratch");
        let mut store = open_store(scratch.path());
        let chunk = vec![5u8; 64];
        let chunk_id = kivi_codec::integrity::chunk_id(DOMAIN, &chunk);
        store.stage_chunk(chunk_id, &chunk).expect("stages chunk");
        let chunking = Chunking::DEFAULT;
        assert_eq!(chunking.chunk_size, DEFAULT_CHUNK_SIZE);
        let (manifest, id) = build_manifest(
            DOMAIN,
            Chunking {
                version: crate::policy::CHUNKING_V1,
                chunk_size: 64,
            },
            crate::codec::ChunkCodecId::NONE,
            64,
            vec![crate::manifest::ChunkEntry {
                id: chunk_id,
                len: 64,
            }],
        )
        .expect("builds");
        let (canonical, recomputed) = manifest.canonical_with_id(DOMAIN);
        assert_eq!(id, recomputed);
        let staged = store
            .stage_manifest(id, &canonical)
            .expect("stages manifest");
        assert!(staged.inserted && !staged.durable);
        store.sync().expect("syncs");
        assert!(store.durable_manifest(id).is_some());
        drop(store);
        let (store, _) = ChunkStore::open(
            scratch.path(),
            0,
            DOMAIN,
            1024 * 1024,
            WallTimestamp::from_micros(4_000_000),
        )
        .expect("recovers");
        let (back, _) = store.read_manifest(id).expect("reads manifest");
        assert_eq!(back, manifest);
    }

    #[test]
    fn rotation_spills_into_fresh_packs() {
        let scratch = tempfile::tempdir().expect("scratch");
        // Tiny target: two 164-byte chunk records fit per pack past the
        // 64-byte header (64 + 164 + 164 = 392 ≤ 512 < 556), so six chunks
        // span exactly three packs.
        let (mut store, _) = ChunkStore::open(
            scratch.path(),
            0,
            DOMAIN,
            512,
            WallTimestamp::from_micros(5_000_000),
        )
        .expect("opens");
        for byte in 0u8..6 {
            let bytes = vec![byte; 100];
            store
                .stage_chunk(chunk_id(DOMAIN, &bytes), &bytes)
                .expect("stages");
        }
        store.sync().expect("syncs");
        assert_eq!(store.pack_seqs(), vec![1, 2, 3]);
        for byte in 0u8..6 {
            let bytes = vec![byte; 100];
            assert_eq!(
                store
                    .read_chunk(chunk_id(DOMAIN, &bytes))
                    .expect("reads")
                    .as_ref(),
                bytes.as_slice()
            );
        }
    }
}
