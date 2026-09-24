//! Node-local fragment storage: one node's durable fragment view.
//!
//! [`LocalFragmentStore`] is the per-node basement of the redundancy plane:
//! it stages hash-verified fragment bytes, durably accepts published layout
//! bytes, and serves only bytes that verify now. It never plans, never
//! places, never publishes authority — the control plane owns which
//! generation is current, so this store keeps **no `CURRENT` file**: every
//! accepted generation's `gen-N.layout` simply persists, and
//! [`FragmentStore::probe_asset`] reports the newest one.
//!
//! On-disk layout (under `<root>/assets/<kind>-<hex>/`, shared with the
//! embedded [`crate::RedundancyFabric`] file helpers also owned here):
//! fragments as `frag-<gen>-<idx>.bin`, layouts as `gen-<N>.layout`, every
//! write through temp + rename + sync so a torn write never looks complete.
//!
//! Fencing, everywhere: fragment bytes are hash- and length-verified
//! **before** storing (lies are rejected, never indexed); a generation is
//! accepted only past the cataloged maximum (or as an idempotent retry of
//! it); incarnation mismatches fail closed with
//! [`crate::RedundancyError::StaleIncarnation`] (`target == 0` is the
//! bootstrap probe and is always allowed).

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use kivi_codec::integrity::blake3_256;

use crate::{AssetId, FragmentRole, RedundancyError, RedundancyLayout, proto::FragmentKey};

/// Node-local fragment storage contract: stage, serve, probe, retire.
///
/// All methods are `&self` (interior mutability): holders serve RPCs while
/// staging concurrently. Every method taking `target_incarnation` refuses
/// with [`crate::RedundancyError::StaleIncarnation`] when `target` is
/// neither `0` (bootstrap probe, always allowed) nor the store's current
/// incarnation.
pub trait FragmentStore: Send + Sync + std::fmt::Debug {
    /// Stages verified fragment bytes, returning the verified content hash.
    ///
    /// Hash and length are verified **before** anything is stored; a
    /// generation is accepted only past the cataloged maximum for the asset
    /// (or as an idempotent retry of it). Layout association arrives lazily
    /// through [`Self::put_layout`]; fragments staged ahead of their layout
    /// are reported by [`Self::orphans`].
    ///
    /// # Errors
    ///
    /// Returns [`RedundancyError`] on incarnation mismatch, invalid role,
    /// oversize payloads, hash/length mismatch, stale generations, or I/O.
    // Seven wire-shaped parameters: the RPC fields map one-to-one, so a
    // struct would only rename the lint without removing the coupling.
    #[allow(clippy::too_many_arguments)]
    fn stage_fragment(
        &self,
        key: &FragmentKey,
        role: u8,
        params_tag: u32,
        content: [u8; 32],
        stored_len: u64,
        bytes: &[u8],
        target_incarnation: u64,
    ) -> Result<[u8; 32], RedundancyError>;

    /// Fetches fragment bytes, hash-verified at serve time.
    ///
    /// # Errors
    ///
    /// Returns [`RedundancyError::MissingFragment`] when absent and
    /// [`RedundancyError::CorruptFragment`] when the stored bytes fail
    /// verification — bad bytes are never served.
    fn fetch_fragment(
        &self,
        key: &FragmentKey,
        target_incarnation: u64,
    ) -> Result<Vec<u8>, RedundancyError>;

    /// Probes presence and health: `(present, healthy)`. Health re-verifies
    /// the hash now. Incarnation mismatches report `(false, false)`.
    fn has_fragment(&self, key: &FragmentKey, target_incarnation: u64) -> (bool, bool);

    /// Removes one fragment (idempotent: absent counts as removed).
    /// Incarnation mismatches remove nothing and report `false`.
    fn remove_fragment(&self, key: &FragmentKey, target_incarnation: u64) -> bool;

    /// Durably accepts a published layout's opaque bytes.
    ///
    /// Returns `true` when newly accepted, `false` on an idempotent retry
    /// of identical bytes. Same-generation conflicts and older generations
    /// are rejected (fencing, never silent supersede).
    ///
    /// # Errors
    ///
    /// Returns [`RedundancyError`] on incarnation mismatch, malformed
    /// layouts, generation conflicts, stale generations, or I/O.
    fn put_layout(
        &self,
        asset: &AssetId,
        generation: u64,
        control_generation: u64,
        layout: &[u8],
        target_incarnation: u64,
    ) -> Result<bool, RedundancyError>;

    /// Returns accepted layout bytes for one generation.
    ///
    /// # Errors
    ///
    /// Returns [`RedundancyError`] on incarnation mismatch, when no layout
    /// is accepted for the generation, or on I/O.
    fn get_layout(
        &self,
        asset_kind: u8,
        asset_domain: u64,
        asset_hash: [u8; 32],
        generation: u64,
        target_incarnation: u64,
    ) -> Result<Vec<u8>, RedundancyError>;

    /// Reports the newest accepted generation plus the BLAKE3 of its layout
    /// bytes (`(0, zero)` when unknown; `(0, zero)` on incarnation mismatch).
    fn probe_asset(
        &self,
        asset_kind: u8,
        asset_domain: u64,
        asset_hash: [u8; 32],
        target_incarnation: u64,
    ) -> (u64, [u8; 32]);

    /// Lists staged fragments whose `(asset, generation)` has no accepted
    /// layout (staged ahead of — or surviving past — authority).
    fn orphans(&self) -> Vec<FragmentKey>;

    /// Removes orphan fragments (files plus index), returning how many were
    /// swept.
    fn sweep_orphans(&self) -> u64;

    /// Returns the current incarnation.
    fn incarnation(&self) -> u64;

    /// Sets the incarnation (node restart, operator epoch).
    fn set_incarnation(&self, incarnation: u64);
}

/// What [`LocalFragmentStore::open`] recovered (operators, startup timing).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct StoreRecovery {
    /// Asset directories indexed.
    pub assets: u64,
    /// Fragment files verified and indexed.
    pub fragments: u64,
    /// Layout files decoded and accepted.
    pub layouts: u64,
    /// Staged fragments without an accepted layout.
    pub orphans: u64,
}

/// One accepted layout's operator facts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AcceptedLayout {
    /// Asset family discriminant.
    pub asset_kind: u8,
    /// Raw security-domain value.
    pub asset_domain: u64,
    /// Content hash minted by the owning fabric.
    pub asset_hash: [u8; 32],
    /// Accepted generation.
    pub generation: u64,
    /// Control generation that published it.
    pub control_generation: u64,
    /// BLAKE3 of the accepted layout bytes.
    pub layout_hash: [u8; 32],
    /// Accepted layout bytes length.
    pub layout_len: u64,
}

/// One staged fragment's operator facts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StagedFragment {
    /// Fragment coordinates.
    pub key: FragmentKey,
    /// Role discriminant at stage time.
    pub role: u8,
    /// Encoding parameters tag at stage time.
    pub params_tag: u32,
    /// BLAKE3 the staged bytes verified against.
    pub content: [u8; 32],
    /// Declared stored length.
    pub stored_len: u64,
    /// Whether the fragment's generation has an accepted layout.
    pub layout_accepted: bool,
}

/// One accepted layout: opaque bytes plus their hash.
#[derive(Debug, Clone)]
struct LayoutEntry {
    /// Exact accepted bytes.
    bytes: Vec<u8>,
    /// BLAKE3 of `bytes` (probe answers without re-reading).
    hash: [u8; 32],
    /// Control generation that published it (retained for operators; the
    /// store neither fences nor acts on it — authority lives in the
    /// control plane).
    control_generation: u64,
}

/// One staged fragment: verification facts plus storage length.
#[derive(Debug, Clone)]
struct FragmentEntry {
    /// Role discriminant at stage time (observability only).
    role: u8,
    /// Encoding parameters tag (observability only).
    params_tag: u32,
    /// BLAKE3 the staged bytes verified against.
    content: [u8; 32],
    /// Declared stored length.
    stored_len: u64,
}

/// Per-asset state: accepted layouts plus staged fragments.
#[derive(Debug, Default)]
struct AssetState {
    /// Accepted layouts by generation.
    layouts: BTreeMap<u64, LayoutEntry>,
    /// Staged fragments by `(generation, index)`.
    fragments: HashMap<(u64, u32), FragmentEntry>,
}

/// Mutable store state behind the lock.
#[derive(Debug)]
struct Inner {
    /// Store root (`<root>/assets/...`).
    root: PathBuf,
    /// Indexed state by `(kind, domain, hash)`.
    assets: HashMap<(u8, u64, [u8; 32]), AssetState>,
    /// Current incarnation (restart epoch).
    incarnation: u64,
}

/// Dir-backed node-local fragment store.
///
/// Separate tempdir per node in tests means separate stores with no shared
/// filesystem; real-network coverage lives in `kivi-consensus` and
/// `kivi-lab` suites.
#[derive(Debug)]
pub struct LocalFragmentStore {
    /// Guarded state (root lives inside for path derivation).
    inner: Mutex<Inner>,
}

impl LocalFragmentStore {
    /// Opens (creating if needed) the store root and recovers indexed state:
    /// layout files are decoded and accepted, fragment files are read and
    /// hash-checked against their layout records where one exists, and
    /// staged fragments without an accepted layout are counted as orphans.
    /// Undecodable files are skipped (never served, never indexed).
    ///
    /// # Errors
    ///
    /// Returns [`RedundancyError`] when the root cannot be created or listed.
    pub fn open(root: &Path, incarnation: u64) -> Result<(Self, StoreRecovery), RedundancyError> {
        std::fs::create_dir_all(root)
            .map_err(|error| RedundancyError::io("create fragment store root", root, &error))?;
        let assets_dir = root.join("assets");
        std::fs::create_dir_all(&assets_dir).map_err(|error| {
            RedundancyError::io("create fragment store assets", &assets_dir, &error)
        })?;
        let mut inner = Inner {
            root: root.to_path_buf(),
            assets: HashMap::new(),
            incarnation,
        };
        let mut recovery = StoreRecovery::default();
        let entries = std::fs::read_dir(&assets_dir).map_err(|error| {
            RedundancyError::io("list fragment store assets", &assets_dir, &error)
        })?;
        for entry in entries.flatten() {
            let dir = entry.path();
            if !dir.is_dir() {
                continue;
            }
            let name = entry.file_name().to_string_lossy().into_owned();
            let Some((kind, hash)) = parse_asset_dir(&name) else {
                continue;
            };
            // The domain is not encoded in the directory name; recover it
            // from the first decodable layout (fragments alone cannot name
            // it, so domain-less orphans stay keyed under `0` until their
            // layout arrives — coordinates still match by kind+hash).
            let coords = (kind.as_u8(), 0u64, hash);
            Self::recover_asset_dir(&mut inner, &coords, &dir, &mut recovery);
        }
        recovery.orphans = count_orphans(&inner);
        let store = Self {
            inner: Mutex::new(inner),
        };
        Ok((store, recovery))
    }

    /// Recovers one asset directory into the index.
    /// Recovers one asset directory into the index (layouts first, then
    /// fragments against their records).
    fn recover_asset_dir(
        inner: &mut Inner,
        coords: &(u8, u64, [u8; 32]),
        dir: &Path,
        recovery: &mut StoreRecovery,
    ) {
        let entries: Vec<std::path::PathBuf> = std::fs::read_dir(dir)
            .map(|entries| entries.flatten().map(|e| e.path()).collect())
            .unwrap_or_default();
        Self::recover_layouts(inner, coords, &entries, recovery);
        Self::recover_fragments(inner, coords, &entries, recovery);
        recovery.assets += 1;
    }

    /// Recovers layout files: decode, accept, and name the true domain.
    fn recover_layouts(
        inner: &mut Inner,
        coords: &(u8, u64, [u8; 32]),
        entries: &[std::path::PathBuf],
        recovery: &mut StoreRecovery,
    ) {
        // Layouts first: fragment verification needs their records.
        for path in entries {
            let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
                continue;
            };
            // Temp files never name durable state (we mint lowercase `.tmp`
            // only, like the embedded fabric's recovery).
            #[allow(clippy::case_sensitive_file_extension_comparisons)]
            let is_tmp = name.ends_with(".tmp");
            if is_tmp {
                let _ = std::fs::remove_file(path);
                continue;
            }
            let Some(generation) = parse_layout_name(name) else {
                continue;
            };
            if generation == 0 {
                continue;
            }
            let Ok(bytes) = std::fs::read(path) else {
                continue;
            };
            let Ok(layout) = RedundancyLayout::decode(&bytes) else {
                continue;
            };
            if layout.generation != generation || layout.asset.hash != coords.2 {
                continue;
            }
            let key = (
                layout.asset.kind.as_u8(),
                layout.asset.domain,
                layout.asset.hash,
            );
            // First layout names the true domain; merge any domain-less
            // fragments already indexed under `(kind, 0, hash)`.
            if key != *coords
                && let Some(staged) = inner.assets.remove(coords)
            {
                inner
                    .assets
                    .entry(key)
                    .or_default()
                    .fragments
                    .extend(staged.fragments);
            }
            let state = inner.assets.entry(key).or_default();
            state.layouts.entry(generation).or_insert(LayoutEntry {
                bytes: bytes.clone(),
                hash: blake3_256(&bytes),
                control_generation: 0,
            });
            recovery.layouts += 1;
        }
    }

    /// Recovers fragment files: verify against layout records where
    /// accepted, else index as orphans by their staged hash.
    fn recover_fragments(
        inner: &mut Inner,
        coords: &(u8, u64, [u8; 32]),
        entries: &[std::path::PathBuf],
        recovery: &mut StoreRecovery,
    ) {
        for path in entries {
            let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
                continue;
            };
            let Some((generation, index)) = parse_fragment_name(name) else {
                continue;
            };
            if generation == 0 {
                continue;
            }
            let Ok(bytes) = std::fs::read(path) else {
                continue;
            };
            if Self::place_verified(inner, coords, path, &bytes, generation, index, recovery) {
                continue;
            }
            let state = inner.assets.entry(*coords).or_default();
            let content = blake3_256(&bytes);
            state
                .fragments
                .entry((generation, index))
                .or_insert(FragmentEntry {
                    role: 0,
                    params_tag: 0,
                    content,
                    stored_len: bytes.len() as u64,
                });
            recovery.fragments += 1;
        }
    }

    /// Indexes one fragment file against its layout record where accepted
    /// (deleting hash failures); reports whether the file was placed.
    fn place_verified(
        inner: &mut Inner,
        coords: &(u8, u64, [u8; 32]),
        path: &Path,
        bytes: &[u8],
        generation: u64,
        index: u32,
        recovery: &mut StoreRecovery,
    ) -> bool {
        // Candidate keys are limited to this directory's hash (kind may
        // vary only by construction; layouts name it exactly).
        let candidates: Vec<(u8, u64, [u8; 32])> = inner
            .assets
            .keys()
            .filter(|k| k.2 == coords.2)
            .copied()
            .collect();
        for candidate in candidates {
            let Some(state) = inner.assets.get_mut(&candidate) else {
                continue;
            };
            let Some(layout_entry) = state.layouts.get(&generation) else {
                continue;
            };
            let Ok(layout) = RedundancyLayout::decode(&layout_entry.bytes) else {
                continue;
            };
            // Index < total <= 40, fits `usize` on every target.
            let slot = index as usize;
            let Some(record) = layout.fragments.get(slot) else {
                continue;
            };
            if record.id.index != index {
                continue;
            }
            if blake3_256(bytes) != record.id.content || bytes.len() as u64 != record.stored_len {
                let _ = std::fs::remove_file(path);
                return true;
            }
            state
                .fragments
                .entry((generation, index))
                .or_insert(FragmentEntry {
                    role: record.id.role.as_u8(),
                    params_tag: record.id.params_tag,
                    content: record.id.content,
                    stored_len: record.stored_len,
                });
            recovery.fragments += 1;
            return true;
        }
        false
    }

    /// Locks guarded state, mapping poisoning to an I/O-shaped error.
    fn lock(&self) -> Result<std::sync::MutexGuard<'_, Inner>, RedundancyError> {
        self.inner.lock().map_err(|_| RedundancyError::Unreadable {
            detail: "fragment store lock poisoned".to_owned(),
        })
    }

    /// Directory for one asset's layouts and fragments.
    fn asset_dir(root: &Path, kind: crate::AssetKind, hash: &[u8; 32]) -> PathBuf {
        root.join("assets")
            .join(format!("{}-{}", kind.name(), hex_of(hash)))
    }

    /// Lists accepted layouts' operator facts (generation, control
    /// generation, hashes), sorted by asset then generation.
    #[must_use]
    pub fn accepted_layouts(&self) -> Vec<AcceptedLayout> {
        let Ok(guard) = self.inner.lock() else {
            return Vec::new();
        };
        let mut out = Vec::new();
        for ((kind, domain, hash), state) in &guard.assets {
            for (generation, entry) in &state.layouts {
                out.push(AcceptedLayout {
                    asset_kind: *kind,
                    asset_domain: *domain,
                    asset_hash: *hash,
                    generation: *generation,
                    control_generation: entry.control_generation,
                    layout_hash: entry.hash,
                    layout_len: entry.bytes.len() as u64,
                });
            }
        }
        out.sort_by(|a, b| {
            (a.asset_kind, a.asset_domain, a.asset_hash, a.generation).cmp(&(
                b.asset_kind,
                b.asset_domain,
                b.asset_hash,
                b.generation,
            ))
        });
        out
    }

    /// Reclaims staged fragments that no accepted layout names, fenced by
    /// `target_incarnation` so a stale writer can never sweep under a live
    /// incarnation. Orphans were never authoritative and never answer a
    /// read: this only reclaims disk.
    ///
    /// # Errors
    ///
    /// Returns [`RedundancyError::StaleIncarnation`] when addressed to a
    /// superseded incarnation and [`RedundancyError::Unreadable`] when the
    /// store lock is poisoned.
    pub fn sweep_orphans_at(&self, target_incarnation: u64) -> Result<u64, RedundancyError> {
        let guard = self.lock()?;
        check_incarnation(guard.incarnation, target_incarnation)?;
        drop(guard);
        Ok(self.sweep_orphans())
    }

    /// Lists staged fragments' operator facts, sorted by coordinates.
    #[must_use]
    pub fn staged_fragments(&self) -> Vec<StagedFragment> {
        let Ok(guard) = self.inner.lock() else {
            return Vec::new();
        };
        let mut out = Vec::new();
        for ((kind, domain, hash), state) in &guard.assets {
            for ((generation, index), entry) in &state.fragments {
                out.push(StagedFragment {
                    key: FragmentKey {
                        asset_kind: *kind,
                        asset_domain: *domain,
                        asset_hash: *hash,
                        generation: *generation,
                        index: *index,
                    },
                    role: entry.role,
                    params_tag: entry.params_tag,
                    content: entry.content,
                    stored_len: entry.stored_len,
                    layout_accepted: state.layouts.contains_key(generation),
                });
            }
        }
        out.sort_by(|a, b| {
            (
                a.key.asset_kind,
                a.key.asset_domain,
                a.key.asset_hash,
                a.key.generation,
                a.key.index,
            )
                .cmp(&(
                    b.key.asset_kind,
                    b.key.asset_domain,
                    b.key.asset_hash,
                    b.key.generation,
                    b.key.index,
                ))
        });
        out
    }
}

/// Counts staged fragments without an accepted layout.
fn count_orphans(inner: &Inner) -> u64 {
    let mut orphans = 0u64;
    for state in inner.assets.values() {
        for (generation, _) in state.fragments.keys() {
            if !state.layouts.contains_key(generation) {
                orphans += 1;
            }
        }
    }
    orphans
}

impl FragmentStore for LocalFragmentStore {
    fn stage_fragment(
        &self,
        key: &FragmentKey,
        role: u8,
        params_tag: u32,
        content: [u8; 32],
        stored_len: u64,
        bytes: &[u8],
        target_incarnation: u64,
    ) -> Result<[u8; 32], RedundancyError> {
        let mut guard = self.lock()?;
        check_incarnation(guard.incarnation, target_incarnation)?;
        if key.generation == 0 {
            return Err(RedundancyError::BadLayout {
                detail: "generation 0 is the invalid sentinel".to_owned(),
            });
        }
        FragmentRole::from_u8(role).map_err(|_| RedundancyError::BadLayout {
            detail: format!("unknown fragment role {role}"),
        })?;
        if bytes.len() > crate::proto::MAX_FRAGMENT_WIRE_BYTES {
            return Err(RedundancyError::Overloaded {
                detail: format!("fragment holds {} bytes", bytes.len()),
            });
        }
        if bytes.len() as u64 != stored_len {
            return Err(RedundancyError::CorruptFragment {
                detail: format!(
                    "fragment {} bytes disagree with stored length {stored_len}",
                    bytes.len()
                ),
            });
        }
        if blake3_256(bytes) != content {
            return Err(RedundancyError::CorruptFragment {
                detail: "fragment content hash mismatch".to_owned(),
            });
        }
        let asset = key.to_asset().map_err(|error| RedundancyError::BadLayout {
            detail: format!("fragment key: {error}"),
        })?;
        let coords = (key.asset_kind, key.asset_domain, key.asset_hash);
        let current = guard
            .assets
            .get(&coords)
            .and_then(|state| state.layouts.keys().next_back().copied())
            .unwrap_or(0);
        if key.generation < current {
            return Err(RedundancyError::StaleGeneration {
                generation: key.generation,
                current,
            });
        }
        if let Some(state) = guard.assets.get(&coords)
            && let Some(existing) = state.fragments.get(&(key.generation, key.index))
        {
            // Idempotent retry of identical bytes; conflicting bytes for
            // fenced coordinates are corruption, never a merge.
            if existing.content != content || existing.stored_len != stored_len {
                return Err(RedundancyError::CorruptFragment {
                    detail: "conflicting bytes for fenced fragment coordinates".to_owned(),
                });
            }
            // The in-memory record can be intact while the payload on disk
            // rotted (bit rot, torn write, external damage). Repair re-puts
            // the authoritative bytes, so this is the moment to restore the
            // file: a matching hash must never leave corrupt bytes in place.
            let dir = Self::asset_dir(&guard.root, asset.kind, &asset.hash);
            let path = dir.join(fragment_name(key.generation, key.index, false));
            let healthy = std::fs::read(&path).is_ok_and(|current| {
                current.len() as u64 == stored_len && blake3_256(&current) == content
            });
            if !healthy {
                std::fs::create_dir_all(&dir)
                    .map_err(|error| RedundancyError::io("create asset directory", &dir, &error))?;
                write_fragment(&dir, key.generation, key.index, bytes, false)?;
                sync_dir_best_effort(&dir);
            }
            return Ok(content);
        }
        let dir = Self::asset_dir(&guard.root, asset.kind, &asset.hash);
        std::fs::create_dir_all(&dir)
            .map_err(|error| RedundancyError::io("create asset directory", &dir, &error))?;
        write_fragment(&dir, key.generation, key.index, bytes, false)?;
        sync_dir_best_effort(&dir);
        guard.assets.entry(coords).or_default().fragments.insert(
            (key.generation, key.index),
            FragmentEntry {
                role,
                params_tag,
                content,
                stored_len,
            },
        );
        Ok(content)
    }

    fn fetch_fragment(
        &self,
        key: &FragmentKey,
        target_incarnation: u64,
    ) -> Result<Vec<u8>, RedundancyError> {
        let guard = self.lock()?;
        check_incarnation(guard.incarnation, target_incarnation)?;
        let coords = (key.asset_kind, key.asset_domain, key.asset_hash);
        let missing = || RedundancyError::MissingFragment {
            asset: coords_string(&coords),
            generation: key.generation,
            index: key.index,
        };
        let resolved = resolve_coords(&guard.assets, coords).ok_or_else(missing)?;
        let state = guard.assets.get(&resolved).ok_or_else(missing)?;
        let entry = state
            .fragments
            .get(&(key.generation, key.index))
            .ok_or_else(missing)?;
        let asset = key.to_asset().map_err(|_| missing())?;
        let dir = Self::asset_dir(&guard.root, asset.kind, &asset.hash);
        let bytes = read_fragment(&dir, key.generation, key.index)?;
        if bytes.len() as u64 != entry.stored_len || blake3_256(&bytes) != entry.content {
            return Err(RedundancyError::CorruptFragment {
                detail: "stored fragment fails verification".to_owned(),
            });
        }
        Ok(bytes)
    }

    fn has_fragment(&self, key: &FragmentKey, target_incarnation: u64) -> (bool, bool) {
        let Ok(guard) = self.inner.lock() else {
            return (false, false);
        };
        if check_incarnation(guard.incarnation, target_incarnation).is_err() {
            return (false, false);
        }
        let coords = (key.asset_kind, key.asset_domain, key.asset_hash);
        let Some(resolved) = resolve_coords(&guard.assets, coords) else {
            return (false, false);
        };
        let Some(state) = guard.assets.get(&resolved) else {
            return (false, false);
        };
        let Some(entry) = state.fragments.get(&(key.generation, key.index)) else {
            return (false, false);
        };
        let Ok(asset) = key.to_asset() else {
            return (false, false);
        };
        let dir = Self::asset_dir(&guard.root, asset.kind, &asset.hash);
        let Ok(bytes) = read_fragment(&dir, key.generation, key.index) else {
            return (false, false);
        };
        if bytes.len() as u64 != entry.stored_len || blake3_256(&bytes) != entry.content {
            return (true, false);
        }
        (true, true)
    }

    fn remove_fragment(&self, key: &FragmentKey, target_incarnation: u64) -> bool {
        let Ok(mut guard) = self.inner.lock() else {
            return false;
        };
        if check_incarnation(guard.incarnation, target_incarnation).is_err() {
            return false;
        }
        let coords = (key.asset_kind, key.asset_domain, key.asset_hash);
        let mut removed = false;
        if let Some(resolved) = resolve_coords(&guard.assets, coords)
            && let Some(state) = guard.assets.get_mut(&resolved)
        {
            removed = state
                .fragments
                .remove(&(key.generation, key.index))
                .is_some();
        }
        if let Ok(asset) = key.to_asset() {
            let dir = Self::asset_dir(&guard.root, asset.kind, &asset.hash);
            let path = dir.join(fragment_name(key.generation, key.index, false));
            if std::fs::remove_file(&path).is_ok() {
                removed = true;
            }
        }
        removed
    }

    fn put_layout(
        &self,
        asset: &AssetId,
        generation: u64,
        control_generation: u64,
        layout: &[u8],
        target_incarnation: u64,
    ) -> Result<bool, RedundancyError> {
        let mut guard = self.lock()?;
        check_incarnation(guard.incarnation, target_incarnation)?;
        if generation == 0 {
            return Err(RedundancyError::BadLayout {
                detail: "generation 0 is the invalid sentinel".to_owned(),
            });
        }
        if layout.len() > crate::proto::MAX_LAYOUT_WIRE_BYTES {
            return Err(RedundancyError::Overloaded {
                detail: format!("layout holds {} bytes", layout.len()),
            });
        }
        let decoded = RedundancyLayout::decode(layout)?;
        if decoded.asset != *asset || decoded.generation != generation {
            return Err(RedundancyError::BadLayout {
                detail: "layout bytes disagree with addressed asset/generation".to_owned(),
            });
        }
        let coords = (asset.kind.as_u8(), asset.domain, asset.hash);
        let current = guard
            .assets
            .get(&coords)
            .and_then(|state| state.layouts.keys().next_back().copied())
            .unwrap_or(0);
        if generation < current {
            return Err(RedundancyError::StaleGeneration {
                generation,
                current,
            });
        }
        if let Some(state) = guard.assets.get(&coords)
            && let Some(existing) = state.layouts.get(&generation)
        {
            if existing.bytes == layout {
                return Ok(false);
            }
            return Err(RedundancyError::BadLayout {
                detail: "conflicting layout bytes for a fenced generation".to_owned(),
            });
        }
        let dir = Self::asset_dir(&guard.root, asset.kind, &asset.hash);
        std::fs::create_dir_all(&dir)
            .map_err(|error| RedundancyError::io("create asset directory", &dir, &error))?;
        write_layout_bytes(&dir, generation, layout)?;
        sync_dir_best_effort(&dir);
        guard.assets.entry(coords).or_default().layouts.insert(
            generation,
            LayoutEntry {
                bytes: layout.to_vec(),
                hash: blake3_256(layout),
                control_generation,
            },
        );
        // Authority names the true domain: adopt pre-authority stages,
        // then bind them to the accepted records (strays quarantined).
        adopt_prestaged(&mut guard, coords);
        let dir = Self::asset_dir(&guard.root, asset.kind, &asset.hash);
        if let Some(state) = guard.assets.get_mut(&coords) {
            bind_staged_inline(&decoded, &dir, state);
        }
        Ok(true)
    }

    fn get_layout(
        &self,
        asset_kind: u8,
        asset_domain: u64,
        asset_hash: [u8; 32],
        generation: u64,
        target_incarnation: u64,
    ) -> Result<Vec<u8>, RedundancyError> {
        let guard = self.lock()?;
        check_incarnation(guard.incarnation, target_incarnation)?;
        let coords = (asset_kind, asset_domain, asset_hash);
        guard
            .assets
            .get(&coords)
            .and_then(|state| state.layouts.get(&generation))
            .map(|entry| entry.bytes.clone())
            .ok_or_else(|| RedundancyError::MissingFragment {
                asset: coords_string(&coords),
                generation,
                index: 0,
            })
    }

    fn probe_asset(
        &self,
        asset_kind: u8,
        asset_domain: u64,
        asset_hash: [u8; 32],
        target_incarnation: u64,
    ) -> (u64, [u8; 32]) {
        let Ok(guard) = self.inner.lock() else {
            return (0, [0; 32]);
        };
        if check_incarnation(guard.incarnation, target_incarnation).is_err() {
            return (0, [0; 32]);
        }
        let coords = (asset_kind, asset_domain, asset_hash);
        guard
            .assets
            .get(&coords)
            .and_then(|state| {
                state.layouts.keys().next_back().and_then(|generation| {
                    state
                        .layouts
                        .get(generation)
                        .map(|entry| (*generation, entry.hash))
                })
            })
            .unwrap_or((0, [0; 32]))
    }

    fn orphans(&self) -> Vec<FragmentKey> {
        let Ok(guard) = self.inner.lock() else {
            return Vec::new();
        };
        let mut out = Vec::new();
        for ((kind, domain, hash), state) in &guard.assets {
            for (generation, index) in state.fragments.keys() {
                if !state.layouts.contains_key(generation) {
                    out.push(FragmentKey {
                        asset_kind: *kind,
                        asset_domain: *domain,
                        asset_hash: *hash,
                        generation: *generation,
                        index: *index,
                    });
                }
            }
        }
        out.sort_by(|a, b| {
            (
                a.asset_kind,
                a.asset_domain,
                a.asset_hash,
                a.generation,
                a.index,
            )
                .cmp(&(
                    b.asset_kind,
                    b.asset_domain,
                    b.asset_hash,
                    b.generation,
                    b.index,
                ))
        });
        out
    }

    fn sweep_orphans(&self) -> u64 {
        let Ok(mut guard) = self.inner.lock() else {
            return 0;
        };
        let mut swept = 0u64;
        let coords_list: Vec<(u8, u64, [u8; 32])> = guard.assets.keys().copied().collect();
        for coords in coords_list {
            let orphan_keys: Vec<(u64, u32)> = guard
                .assets
                .get(&coords)
                .map(|state| {
                    state
                        .fragments
                        .keys()
                        .filter(|(generation, _)| !state.layouts.contains_key(generation))
                        .copied()
                        .collect()
                })
                .unwrap_or_default();
            if orphan_keys.is_empty() {
                continue;
            }
            let dir = asset_dir_name(&guard.root, coords.0, &coords.2);
            for (generation, index) in orphan_keys {
                if let Some(state) = guard.assets.get_mut(&coords) {
                    state.fragments.remove(&(generation, index));
                }
                let _ = std::fs::remove_file(dir.join(fragment_name(generation, index, false)));
                swept += 1;
            }
            sync_dir_best_effort(&dir);
        }
        swept
    }

    fn incarnation(&self) -> u64 {
        self.inner.lock().map_or(0, |guard| guard.incarnation)
    }

    fn set_incarnation(&self, incarnation: u64) {
        if let Ok(mut guard) = self.inner.lock() {
            guard.incarnation = incarnation;
        }
    }
}

/// Resolves index coordinates: the exact domain first, then the
/// pre-authority domain-`0` key (recovery cannot learn the domain without a
/// layout; exact domains always win once known).
fn resolve_coords(
    assets: &HashMap<(u8, u64, [u8; 32]), AssetState>,
    coords: (u8, u64, [u8; 32]),
) -> Option<(u8, u64, [u8; 32])> {
    if assets.contains_key(&coords) {
        return Some(coords);
    }
    let prestaged = (coords.0, 0, coords.2);
    if prestaged != coords && assets.contains_key(&prestaged) {
        return Some(prestaged);
    }
    None
}

/// Adopts pre-authority (domain-`0`) staged fragments into the true-domain
/// state once authority names it.
fn adopt_prestaged(inner: &mut Inner, coords: (u8, u64, [u8; 32])) {
    if coords.1 == 0 {
        return;
    }
    let prestaged = (coords.0, 0, coords.2);
    if prestaged != coords
        && let Some(staged) = inner.assets.remove(&prestaged)
    {
        inner
            .assets
            .entry(coords)
            .or_default()
            .fragments
            .extend(staged.fragments);
    }
}

/// Refuses stale incarnations (`0` is the bootstrap probe, always allowed).
fn check_incarnation(current: u64, target: u64) -> Result<(), RedundancyError> {
    if target != 0 && target != current {
        return Err(RedundancyError::StaleIncarnation {
            expected: current,
            current: target,
        });
    }
    Ok(())
}

/// Quarantines staged fragments that disagree with an accepted layout
/// (role, params tag, content, or length mismatch — or no covering record
/// at all): strays staged under false pretenses are dropped, never served
/// under authority. Runs inline in [`LocalFragmentStore::put_layout`].
fn bind_staged_inline(decoded: &RedundancyLayout, dir: &Path, state: &mut AssetState) {
    let generation = decoded.generation;
    let mut strays = Vec::new();
    for ((staged_gen, index), entry) in &state.fragments {
        if *staged_gen != generation {
            continue;
        }
        let agrees = decoded
            .fragments
            .iter()
            .find(|record| record.id.index == *index)
            .is_some_and(|record| {
                entry.role == record.id.role.as_u8()
                    && entry.params_tag == record.id.params_tag
                    && entry.content == record.id.content
                    && entry.stored_len == record.stored_len
            });
        if !agrees {
            strays.push(*index);
        }
    }
    for index in strays {
        state.fragments.remove(&(generation, index));
        let _ = std::fs::remove_file(dir.join(fragment_name(generation, index, false)));
    }
    sync_dir_best_effort(dir);
}

/// Human-readable coordinates for error context.
fn coords_string(coords: &(u8, u64, [u8; 32])) -> String {
    format!("kind:{}:{}", coords.0, hex_of(&coords.2))
}

/// Lowercase hex of 32 bytes (file names, diagnostics).
fn hex_of(hash: &[u8; 32]) -> String {
    use core::fmt::Write as _;
    let mut out = String::with_capacity(64);
    for byte in hash {
        let _ = write!(out, "{byte:02x}");
    }
    out
}

/// Decodes two lowercase hex digits.
fn hex_val(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        _ => None,
    }
}

/// Parses an asset directory name (`<kind-name>-<64 hex>`).
fn parse_asset_dir(name: &str) -> Option<(crate::AssetKind, [u8; 32])> {
    let (kind_name, hex) = name.rsplit_once('-')?;
    if hex.len() != 64 {
        return None;
    }
    let kind = match kind_name {
        "chunk" => crate::AssetKind::Chunk,
        "chunk-manifest" => crate::AssetKind::ChunkManifest,
        "checkpoint-band" => crate::AssetKind::CheckpointBand,
        "checkpoint-manifest" => crate::AssetKind::CheckpointManifest,
        "checkpoint-dedup" => crate::AssetKind::CheckpointDedup,
        "generic-immutable" => crate::AssetKind::GenericImmutable,
        _ => return None,
    };
    let bytes = hex.as_bytes();
    let mut hash = [0u8; 32];
    for (index, chunk) in bytes.chunks(2).enumerate() {
        hash[index] = hex_val(chunk[0])? << 4 | hex_val(chunk[1])?;
    }
    Some((kind, hash))
}

/// Asset directory path without a decoded domain (recovery helper).
fn asset_dir_name(root: &Path, kind: u8, hash: &[u8; 32]) -> PathBuf {
    let name = crate::AssetKind::from_u8(kind)
        .map_or_else(|_| format!("unknown-{kind}"), |kind| kind.name().to_owned());
    root.join("assets").join(format!("{name}-{}", hex_of(hash)))
}

/// Parses a fragment file name (`frag-<gen>-<idx>.bin`).
fn parse_fragment_name(name: &str) -> Option<(u64, u32)> {
    let rest = name.strip_prefix("frag-")?.strip_suffix(".bin")?;
    let (generation, index) = rest.rsplit_once('-')?;
    Some((generation.parse().ok()?, index.parse().ok()?))
}

/// Parses a published layout file name (`gen-<N>.layout`).
fn parse_layout_name(name: &str) -> Option<u64> {
    name.strip_prefix("gen-")?
        .strip_suffix(".layout")?
        .parse()
        .ok()
}

// ---------------------------------------------------------------------------
// File helpers (single implementation shared with the embedded fabric).
// ---------------------------------------------------------------------------

/// File names for fragment storage.
pub(crate) fn fragment_name(generation: u64, index: u32, pending: bool) -> String {
    if pending {
        format!("pending-{generation}-{index:04}.bin")
    } else {
        format!("frag-{generation}-{index:04}.bin")
    }
}

/// Writes one fragment file (sync file; dir sync batched by the caller).
pub(crate) fn write_fragment(
    dir: &Path,
    generation: u64,
    index: u32,
    bytes: &[u8],
    pending: bool,
) -> Result<(), RedundancyError> {
    // Partial writes must never look complete: write temp, sync, rename.
    let name = fragment_name(generation, index, pending);
    let path = dir.join(&name);
    let tmp = dir.join(format!("{name}.tmp"));
    std::fs::write(&tmp, bytes)
        .map_err(|error| RedundancyError::io("write fragment temp", &tmp, &error))?;
    sync_file_best_effort(&tmp);
    std::fs::rename(&tmp, &path)
        .map_err(|error| RedundancyError::io("publish fragment", &path, &error))?;
    sync_file_best_effort(&path);
    Ok(())
}

/// Reads one published fragment file.
pub(crate) fn read_fragment(
    dir: &Path,
    generation: u64,
    index: u32,
) -> Result<Vec<u8>, RedundancyError> {
    let path = dir.join(fragment_name(generation, index, false));
    match std::fs::read(&path) {
        Ok(bytes) => Ok(bytes),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            Err(RedundancyError::MissingFragment {
                asset: dir
                    .file_name()
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_default(),
                generation,
                index,
            })
        }
        Err(error) => Err(RedundancyError::Unreadable {
            detail: format!("fragment {}: {error}", path.display()),
        }),
    }
}

/// Writes a pending layout file (unpublished).
pub(crate) fn write_layout(
    dir: &Path,
    generation: u64,
    layout: &RedundancyLayout,
    pending: bool,
) -> Result<(), RedundancyError> {
    let name = if pending {
        format!("pending-{generation}.layout")
    } else {
        format!("gen-{generation}.layout")
    };
    let path = dir.join(&name);
    let tmp = dir.join(format!("{name}.tmp"));
    std::fs::write(&tmp, layout.encode())
        .map_err(|error| RedundancyError::io("write layout temp", &tmp, &error))?;
    sync_file_best_effort(&tmp);
    std::fs::rename(&tmp, &path)
        .map_err(|error| RedundancyError::io("publish layout temp", &path, &error))?;
    sync_file_best_effort(&path);
    Ok(())
}

/// Writes opaque layout bytes durably (store path: no pending concept).
pub(crate) fn write_layout_bytes(
    dir: &Path,
    generation: u64,
    bytes: &[u8],
) -> Result<(), RedundancyError> {
    let name = format!("gen-{generation}.layout");
    let path = dir.join(&name);
    let tmp = dir.join(format!("{name}.tmp"));
    std::fs::write(&tmp, bytes)
        .map_err(|error| RedundancyError::io("write layout temp", &tmp, &error))?;
    sync_file_best_effort(&tmp);
    std::fs::rename(&tmp, &path)
        .map_err(|error| RedundancyError::io("publish layout temp", &path, &error))?;
    sync_file_best_effort(&path);
    Ok(())
}

/// Publishes a verified layout: durable `gen-N.layout` plus atomic CURRENT.
///
/// CURRENT names the published generation for the embedded fabric only;
/// cluster stores ([`LocalFragmentStore`]) never use it — authority lives
/// in the control plane.
pub(crate) fn publish_layout(
    dir: &Path,
    generation: u64,
    layout: &RedundancyLayout,
) -> Result<(), RedundancyError> {
    write_layout(dir, generation, layout, false)?;
    // Promote staged fragments to published names (pending -> frag).
    // Reads serve only published names, so deleting pending files here
    // would leave reads with nothing to find.
    for index in 0..layout.params.total_fragments() {
        let pending_path = dir.join(fragment_name(generation, index, true));
        if pending_path.exists() {
            let published_path = dir.join(fragment_name(generation, index, false));
            std::fs::rename(&pending_path, &published_path).map_err(|error| {
                RedundancyError::io("promote staged fragment", &published_path, &error)
            })?;
            sync_file_best_effort(&published_path);
        }
    }
    sync_dir_best_effort(dir);
    // Remove the pending layout file now that published files prove durable.
    let _ = std::fs::remove_file(dir.join(format!("pending-{generation}.layout")));
    // CURRENT names the published generation (8-byte LE, atomic).
    let current = dir.join("CURRENT");
    let tmp = dir.join("CURRENT.tmp");
    std::fs::write(&tmp, generation.to_le_bytes())
        .map_err(|error| RedundancyError::io("write CURRENT temp", &tmp, &error))?;
    sync_file_best_effort(&tmp);
    std::fs::rename(&tmp, &current)
        .map_err(|error| RedundancyError::io("publish CURRENT", &current, &error))?;
    sync_dir_best_effort(dir);
    Ok(())
}

/// Retires a superseded generation (best-effort: failures cost disk).
pub(crate) fn retire_generation(dir: &Path, generation: u64, total: u32) {
    let _ = std::fs::remove_file(dir.join(format!("gen-{generation}.layout")));
    for index in 0..total {
        let _ = std::fs::remove_file(dir.join(fragment_name(generation, index, false)));
    }
    sync_dir_best_effort(dir);
}

/// Syncs one file's data (best-effort where the platform forbids it).
pub(crate) fn sync_file_best_effort(path: &Path) {
    if let Ok(file) = std::fs::File::options().read(true).open(path) {
        let _ = file.sync_all();
    }
}

/// Syncs one directory (best-effort: Windows may refuse dir opens).
pub(crate) fn sync_dir_best_effort(dir: &Path) {
    #[cfg(unix)]
    {
        if let Ok(file) = std::fs::File::open(dir) {
            let _ = file.sync_all();
        }
    }
    #[cfg(not(unix))]
    {
        let _ = dir;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn store(incarnation: u64) -> (LocalFragmentStore, tempfile::TempDir) {
        let dir = tempfile::tempdir().expect("scratch");
        let (store, recovery) =
            LocalFragmentStore::open(&dir.path().join("store"), incarnation).expect("opens");
        assert_eq!(recovery.assets, 0);
        (store, dir)
    }

    fn test_key() -> FragmentKey {
        FragmentKey {
            asset_kind: 0,
            asset_domain: 7,
            asset_hash: [11; 32],
            generation: 1,
            index: 0,
        }
    }

    /// Test asset identity matching [`test_key`].
    fn test_asset() -> AssetId {
        AssetId::new(crate::AssetKind::Chunk, 7, [11; 32]).expect("asset")
    }

    fn stage(
        store: &LocalFragmentStore,
        key: &FragmentKey,
        bytes: &[u8],
        incarnation: u64,
    ) -> [u8; 32] {
        let content = blake3_256(bytes);
        // Role 0 (`Replica`) matches the replication test layouts, so
        // layout arrival binds (rather than quarantines) these stages.
        store
            .stage_fragment(key, 0, 0, content, bytes.len() as u64, bytes, incarnation)
            .expect("stages")
    }

    #[test]
    fn stage_fetch_has_remove_round_trip() {
        let (store, _guard) = store(5);
        let key = test_key();
        let bytes = b"fragment bytes".to_vec();
        let content = stage(&store, &key, &bytes, 5);
        assert_eq!(store.has_fragment(&key, 5), (true, true));
        assert_eq!(store.fetch_fragment(&key, 5).expect("fetches"), bytes);
        assert!(store.remove_fragment(&key, 5));
        assert_eq!(store.has_fragment(&key, 5), (false, false));
        assert!(!store.remove_fragment(&key, 5));
        let _ = content;
    }

    #[test]
    fn hash_mismatch_rejected_before_store() {
        let (store, _guard) = store(1);
        let key = test_key();
        let error = store
            .stage_fragment(&key, 1, 0, [9; 32], 3, b"abc", 1)
            .expect_err("lies rejected");
        assert!(matches!(error, RedundancyError::CorruptFragment { .. }));
        assert_eq!(store.has_fragment(&key, 1), (false, false));
        let length_lie = store
            .stage_fragment(&key, 1, 0, blake3_256(b"abc"), 99, b"abc", 1)
            .expect_err("length lies rejected");
        assert!(matches!(
            length_lie,
            RedundancyError::CorruptFragment { .. }
        ));
    }

    #[test]
    fn stale_generation_rejected() {
        let (store, _guard) = store(1);
        let asset = test_asset();
        let mut new_key = test_key();
        new_key.generation = 3;
        // Staged bytes agree with the gen-3 layout (bound on accept).
        stage(&store, &new_key, b"bytes", 1);
        let layout = store_layout_bytes(3);
        assert!(
            store
                .put_layout(&asset, 3, 10, &layout, 1)
                .expect("publishes")
        );
        // Older generation fragments are stale now.
        let old = test_key();
        let error = store
            .stage_fragment(&old, 1, 0, blake3_256(b"old"), 3, b"old", 1)
            .expect_err("stale rejected");
        assert!(matches!(error, RedundancyError::StaleGeneration { .. }));
        // Older layouts are stale too.
        let stale_layout = store_layout_bytes(1);
        assert!(store.put_layout(&asset, 1, 11, &stale_layout, 1).is_err());
    }

    #[test]
    fn incarnation_refused_but_bootstrap_zero_allowed() {
        let (store, _guard) = store(7);
        let key = test_key();
        // Bootstrap probes (target 0) work without knowing the epoch.
        stage(&store, &key, b"bytes", 0);
        assert_eq!(store.has_fragment(&key, 0), (true, true));
        // Wrong epochs fail closed.
        let error = store
            .stage_fragment(&key, 1, 0, blake3_256(b"bytes"), 5, b"bytes", 6)
            .expect_err("stale incarnation refused");
        assert!(matches!(error, RedundancyError::StaleIncarnation { .. }));
        assert_eq!(store.has_fragment(&key, 6), (false, false));
        assert!(!store.remove_fragment(&key, 6));
        assert_eq!(store.probe_asset(0, 7, [11; 32], 6), (0, [0; 32]));
        // The live epoch works.
        assert_eq!(store.has_fragment(&key, 7), (true, true));
        assert_eq!(store.incarnation(), 7);
        store.set_incarnation(8);
        assert_eq!(store.incarnation(), 8);
        assert_eq!(store.has_fragment(&key, 7), (false, false));
        assert_eq!(store.has_fragment(&key, 8), (true, true));
    }

    #[test]
    fn restart_reopens_and_reports_orphans() {
        let dir = tempfile::tempdir().expect("scratch");
        let root = dir.path().join("store");
        let asset = test_asset();
        let key = test_key();
        {
            let (store, _) = LocalFragmentStore::open(&root, 2).expect("opens");
            // Staged bytes agree with the gen-1 layout (bound on accept).
            stage(&store, &key, b"bytes", 2);
            assert_eq!(store.orphans().len(), 1);
            assert_eq!(store.probe_asset(0, 7, [11; 32], 2), (0, [0; 32]));
        }
        let (store, recovery) = LocalFragmentStore::open(&root, 2).expect("reopens");
        assert_eq!(recovery.fragments, 1);
        assert_eq!(recovery.orphans, 1);
        assert_eq!(store.orphans().len(), 1);
        // Layout arrival resolves the orphan; bytes survive the restart.
        let layout = store_layout_bytes(1);
        assert!(
            store
                .put_layout(&asset, 1, 5, &layout, 2)
                .expect("publishes")
        );
        assert!(store.orphans().is_empty());
        assert_eq!(store.probe_asset(0, 7, [11; 32], 2).0, 1);
        assert_eq!(store.fetch_fragment(&key, 2).expect("survives"), b"bytes");
        // Sweep with no orphans removes nothing.
        assert_eq!(store.sweep_orphans(), 0);
        // A newer orphan sweeps cleanly.
        let mut second = test_key();
        second.generation = 2;
        second.index = 1;
        stage(&store, &second, b"second", 2);
        assert_eq!(store.sweep_orphans(), 1);
        assert_eq!(store.has_fragment(&second, 2), (false, false));
    }

    #[test]
    fn layout_arrival_quarantines_strays() {
        let (store, _guard) = store(1);
        let asset = test_asset();
        let key = test_key();
        // Correct bytes but a lying role: the layout names a replica.
        let content = blake3_256(b"bytes");
        store
            .stage_fragment(&key, 2, 0, content, 5, b"bytes", 1)
            .expect("stages");
        assert_eq!(store.staged_fragments().len(), 1);
        let layout = store_layout_bytes(1);
        assert!(
            store
                .put_layout(&asset, 1, 5, &layout, 1)
                .expect("publishes")
        );
        // The stray is quarantined (never served under authority); the
        // layout itself is accepted and reported.
        assert_eq!(store.has_fragment(&key, 1), (false, false));
        assert!(store.staged_fragments().is_empty());
        let accepted = store.accepted_layouts();
        assert_eq!(accepted.len(), 1);
        assert_eq!(accepted[0].generation, 1);
        assert_eq!(accepted[0].control_generation, 5);
    }

    #[test]
    fn torn_files_read_as_missing_or_corrupt_never_valid() {
        let dir = tempfile::tempdir().expect("scratch");
        let root = dir.path().join("store");
        let asset = test_asset();
        let key = test_key();
        let path = root
            .join("assets")
            .join(format!("chunk-{}", key_to_hex(&key)));
        let frag = path.join("frag-1-0000.bin");
        {
            let (store, _) = LocalFragmentStore::open(&root, 1).expect("opens");
            // Stage plus layout: the store knows the true hash from here on.
            stage(&store, &key, b"bytes", 1);
            let layout = store_layout_bytes(1);
            assert!(
                store
                    .put_layout(&asset, 1, 5, &layout, 1)
                    .expect("publishes")
            );
            // Tear the file mid-bytes: present but unhealthy, fetch fails
            // closed (bad bytes are never served as valid).
            std::fs::write(&frag, b"by").expect("tears");
            assert_eq!(store.has_fragment(&key, 1), (true, false));
            assert!(store.fetch_fragment(&key, 1).is_err());
        }
        // Restart recovery deletes bytes that disagree with authority:
        // torn files never verify, never index, never serve.
        let (store, _) = LocalFragmentStore::open(&root, 1).expect("reopens");
        assert!(!frag.exists(), "torn file swept by recovery");
        assert_eq!(store.has_fragment(&key, 1), (false, false));
        assert!(store.fetch_fragment(&key, 1).is_err());
    }

    /// Builds layout bytes for generation `generation` over `[11; 32]`.
    fn store_layout_bytes(generation: u64) -> Vec<u8> {
        let asset = AssetId::new(crate::AssetKind::Chunk, 7, [11; 32]).expect("asset");
        let params = crate::SchemeParams::Replication(crate::ReplicationParams { copies: 1 });
        let fragments = vec![crate::FragmentRecord {
            id: crate::FragmentId::for_bytes(asset, generation, 0, params, b"bytes"),
            node: kivi_types::NodeId::from_u64(1),
            stored_len: 5,
        }];
        crate::RedundancyLayout::new(asset, 5, generation, params, fragments)
            .expect("builds")
            .encode()
    }

    /// Hex of a test key's hash.
    fn key_to_hex(key: &FragmentKey) -> String {
        use core::fmt::Write as _;
        let mut out = String::with_capacity(64);
        for byte in key.asset_hash {
            let _ = write!(out, "{byte:02x}");
        }
        out
    }
}
