//! The Memory Fabric: one orchestrator over arenas, providers, planner,
//! transitions, and bounded movement.
//!
//! [`MemoryFabric`] owns worker-local [`WorkerArenas`], a dynamic
//! [`ProviderRegistry`], a [`Calibrator`], a [`CriticalityPlanner`], an
//! [`OffcoreQueue`], and a [`MovementScheduler`]. Objects enter via
//! [`insert`](MemoryFabric::insert) with a [`MaterializationIntent`] and
//! a [`BehaviorClass`]; reads via [`get`](MemoryFabric::get) resolve the
//! primary residence transparently, returning [`GetOutcome::Pending`]
//! (never blocking) when the primary is `NVMe`-resident without a
//! synchronous shadow. Background work runs in [`tick`](MemoryFabric::tick)
//! under explicit budgets.
//!
//! The safety invariant holds at every step: [`ObjectMaterialization`]
//! proves a second representation or reconstruction source before any
//! reclamation, transitions build/verify/publish before retiring, and
//! mutations fence in-flight builds by version.

use std::collections::HashMap;
use std::path::PathBuf;

use bytes::Bytes;

use crate::arena::WorkerArenas;
use crate::calibrate::Calibrator;
use crate::compressed::CompressedBlock;
use crate::criticality::{AccessSignals, ExecutionCriticality, Mutability, criticality_of};
use crate::error::MemoryError;
use crate::handle::BehaviorClass;
use crate::intent::MaterializationIntent;
use crate::movement::{BoundedMoveQueue, MovementBudget, MovementScheduler};
use crate::nvme::{Admission, NvmeOptions, NvmeProvider};
use crate::offcore::{OffcoreCompletion, OffcoreOpId, OffcoreQueue};
use crate::placement::{CriticalityPlanner, PlacementInput, Planner};
use crate::provider::{ProviderCaps, ProviderDescriptor, ProviderId, ProviderRegistry};
use crate::repr::{ObjectMaterialization, ReconstructionSource, Residence};
use crate::sim::{SimFault, SimProvider};
use crate::transition::{Transition, TransitionTarget};

/// `NVMe` backend owned by the fabric.
///
/// Production uses [`NvmeBackend::File`] (same record format as the
/// async Compio path); deterministic tests and simulation use
/// [`NvmeBackend::Sim`]. Placement, transition, and movement logic never
/// branches on this enum for correctness — only latency and fault
/// behavior differ.
#[derive(Debug)]
pub enum NvmeBackend {
    /// Deterministic in-memory simulated `NVMe`.
    Sim(SimProvider),
    /// Synchronous file-backed `NVMe` (correctness baseline).
    File(NvmeProvider),
    /// No `NVMe` provider (DRAM-only deployment).
    None,
}

impl NvmeBackend {
    fn write_record(&mut self, bytes: &[u8]) -> Result<(u64, u64, u32), MemoryError> {
        match self {
            Self::Sim(provider) => {
                let (offset, _) = provider.write(bytes)?;
                Ok((offset, bytes.len() as u64, crc32c::crc32c(bytes)))
            }
            Self::File(provider) => provider.append(bytes),
            Self::None => Err(MemoryError::ProviderFailed {
                provider: "nvme",
                object: u64::MAX,
                detail: "no nvme backend configured".to_owned(),
            }),
        }
    }

    fn read_record(&mut self, offset: u64, len: u64) -> Result<Vec<u8>, MemoryError> {
        match self {
            Self::Sim(provider) => {
                let (bytes, _) = provider.read(offset)?;
                if bytes.len() as u64 != len {
                    return Err(MemoryError::CorruptRepresentation {
                        object: u64::MAX,
                        detail: "sim record length changed".to_owned(),
                    });
                }
                Ok(bytes)
            }
            Self::File(provider) => provider.read_at(offset, len),
            Self::None => Err(MemoryError::ProviderFailed {
                provider: "nvme",
                object: u64::MAX,
                detail: "no nvme backend configured".to_owned(),
            }),
        }
    }

    /// Injects a scripted fault (sim backend only; file backend
    /// failures surface as real I/O errors).
    pub fn inject_fault(&mut self, fault: SimFault) {
        if let Self::Sim(provider) = self {
            provider.inject(fault);
        }
    }
}

/// Configuration for one fabric instance.
#[derive(Debug, Clone)]
pub struct MemoryFabricConfig {
    /// DRAM arena capacity bound in bytes.
    pub arena_capacity_bytes: u64,
    /// `NVMe` capacity bound in bytes (sim backend).
    pub nvme_capacity_bytes: u64,
    /// Background move queue capacity (operations).
    pub move_queue_capacity: usize,
    /// Per-tenant queued byte share.
    pub max_bytes_per_tenant: u64,
    /// Off-core queue capacity (operations).
    pub offcore_queue_capacity: usize,
    /// Off-core queued byte bound.
    pub offcore_max_bytes: u64,
    /// Per-tick background budget.
    pub budget: MovementBudget,
    /// `NVMe` admission options.
    pub nvme_options: NvmeOptions,
    /// Optional file-backed `NVMe` directory. When `None`, the fabric
    /// uses the deterministic sim backend.
    pub nvme_dir: Option<PathBuf>,
}

impl Default for MemoryFabricConfig {
    fn default() -> Self {
        Self {
            arena_capacity_bytes: 64 << 20,
            nvme_capacity_bytes: 1 << 30,
            move_queue_capacity: 256,
            max_bytes_per_tenant: 16 << 20,
            offcore_queue_capacity: 64,
            offcore_max_bytes: 64 << 20,
            budget: MovementBudget::production(),
            nvme_options: NvmeOptions::default(),
            nvme_dir: None,
        }
    }
}

/// One device operation for an external executor (single-writer
/// discipline): the fabric planned it under budget, resolved the payload
/// or locator synchronously, and the executor performs the device I/O,
/// then reports back through [`install_demote_record`](MemoryFabric::install_demote_record)
/// or [`install_promote_bytes`](MemoryFabric::install_promote_bytes).
/// Version fences apply at install time, so a raced mutation discards
/// the result without touching state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExternalOp {
    /// Write these bytes to the device for `object` at `version`.
    Demote {
        /// Fabric object served.
        object: u64,
        /// Logical version staged at.
        version: u64,
        /// Payload bytes.
        bytes: Vec<u8>,
    },
    /// Read the off-core record at `offset`/`len` for `object`.
    Promote {
        /// Fabric object served.
        object: u64,
        /// Logical version issued for.
        version: u64,
        /// Device offset of the record.
        offset: u64,
        /// Expected payload length.
        len: u64,
    },
}

/// One cold record to import at recovery: locator plus the logical
/// metadata the journal vouches for.
#[derive(Debug, Clone)]
pub struct OffcoreImport {
    /// Fabric object id to recreate (must be unused).
    pub id: u64,
    /// Logical version the record was sealed at.
    pub version: u64,
    /// Device offset of the record.
    pub offset: u64,
    /// Stored payload length.
    pub len: u64,
    /// CRC32C of the payload.
    pub checksum: u32,
    /// Materialization intent for planning.
    pub intent: MaterializationIntent,
    /// Behavior class for accounting.
    pub class: BehaviorClass,
}

/// Outcome of a synchronous read.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GetOutcome {
    /// Bytes served from a synchronous residence (or a shadow).
    Ready(Bytes),
    /// Primary is `NVMe`-resident with no synchronous copy; a promotion
    /// was enqueued. Re-issue [`get`](MemoryFabric::get) after the
    /// completion is applied.
    Pending {
        /// The promotion operation serving this read.
        op: OffcoreOpId,
    },
}

/// One tracked object.
#[derive(Debug)]
struct ObjectEntry {
    materialization: ObjectMaterialization,
    intent: MaterializationIntent,
    class: BehaviorClass,
    criticality: ExecutionCriticality,
    signals: AccessSignals,
    ticks_since_move: u64,
    pending_promotion: Option<OffcoreOpId>,
    transition: Option<Transition>,
}

/// Aggregate statistics for observability and benchmarks.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct FabricStats {
    /// Tracked objects.
    pub objects: usize,
    /// Inline bytes.
    pub inline_bytes: u64,
    /// DRAM arena bytes.
    pub dram_bytes: u64,
    /// Compressed stored bytes.
    pub compressed_bytes: u64,
    /// `NVMe` stored bytes.
    pub nvme_bytes: u64,
    /// Synchronous read hits.
    pub hits: u64,
    /// Reads that required promotion.
    pub promotion_misses: u64,
    /// Completed demotions.
    pub demotions: u64,
    /// Completed promotions.
    pub promotions: u64,
    /// Payload bytes moved in the background.
    pub migration_bytes: u64,
    /// Compression ratio in basis points (stored/logical, 0 = none).
    pub compression_ratio_bps: u64,
    /// Cancelled transitions.
    pub cancelled: u64,
    /// Arena compactions completed (slot remaps published).
    pub compactions: u64,
}

/// Bytes attributed to one id set, by residence (per-tablet footprint).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Footprint {
    /// Objects with at least one tracked residence.
    pub objects: usize,
    /// Inline bytes.
    pub inline_bytes: u64,
    /// DRAM arena bytes.
    pub dram_bytes: u64,
    /// Compressed stored bytes.
    pub compressed_bytes: u64,
    /// `NVMe` stored bytes.
    pub nvme_bytes: u64,
}

/// The Memory Fabric.
#[derive(Debug)]
pub struct MemoryFabric {
    config: MemoryFabricConfig,
    arenas: WorkerArenas,
    registry: ProviderRegistry,
    calibrator: Calibrator,
    planner: CriticalityPlanner,
    dram_provider: ProviderId,
    compressed_provider: ProviderId,
    nvme_provider: ProviderId,
    backend: NvmeBackend,
    objects: HashMap<u64, ObjectEntry>,
    moves: BoundedMoveQueue,
    scheduler: MovementScheduler,
    offcore: OffcoreQueue,
    now_ticks: u64,
    next_object: u64,
    stats: FabricStats,
    compressed_logical: u64,
    compressed_stored: u64,
    /// Outstanding-operation pins per object: while nonzero the object
    /// must not be retired or have its last synchronous residence
    /// reclaimed. Distinct from reconstruction sources (which name where
    /// bytes can be rebuilt from); pins name who is still looking at them.
    pins: HashMap<u64, usize>,
    /// Round-robin planning cursor (object id): each tick plans at most
    /// `plan_slice` objects starting here, amortizing the scan.
    plan_cursor: u64,
    /// Maximum objects planned per tick (`usize::MAX` = full scan).
    plan_slice: usize,
    /// External executor mode: when true, `Nvme`/DRAM-promote moves are
    /// collected as [`ExternalOp`]s for an outside device executor
    /// (single-writer discipline) instead of the internal off-core queue.
    external_executor: bool,
    /// In-flight operations held by the external executor (counts toward
    /// the movement concurrency bound alongside submitted queue depth).
    external_in_flight: usize,
    /// Outbox of external operations filled by [`drain_moves`](Self::drain_moves).
    external_outbox: Vec<ExternalOp>,
}

impl MemoryFabric {
    /// Creates a fabric from a configuration.
    ///
    /// # Errors
    ///
    /// Returns [`MemoryError::Io`] when a file-backed `NVMe` directory
    /// cannot be opened.
    pub fn new(config: MemoryFabricConfig) -> Result<Self, MemoryError> {
        let mut registry = ProviderRegistry::new();
        let dram_provider =
            registry.register(ProviderCaps::dram(Some(0), config.arena_capacity_bytes));
        let compressed_provider = registry.register(ProviderCaps::compressed(
            Some(0),
            config.arena_capacity_bytes / 2,
        ));
        let nvme_provider = registry.register(ProviderCaps::nvme(config.nvme_capacity_bytes));
        let backend = if let Some(dir) = &config.nvme_dir {
            NvmeBackend::File(NvmeProvider::open(dir, config.nvme_options)?)
        } else {
            let provider =
                SimProvider::nvme(nvme_provider, config.nvme_capacity_bytes, 0x00C0_FFEE);
            NvmeBackend::Sim(provider)
        };
        Ok(Self {
            arenas: WorkerArenas::new(config.arena_capacity_bytes),
            registry,
            calibrator: Calibrator::new(),
            planner: CriticalityPlanner::production(),
            dram_provider,
            compressed_provider,
            nvme_provider,
            backend,
            objects: HashMap::new(),
            moves: BoundedMoveQueue::new(config.move_queue_capacity, config.max_bytes_per_tenant),
            scheduler: MovementScheduler::new(config.budget),
            offcore: OffcoreQueue::new(config.offcore_queue_capacity, config.offcore_max_bytes),
            config,
            now_ticks: 0,
            next_object: 1,
            stats: FabricStats::default(),
            compressed_logical: 0,
            compressed_stored: 0,
            pins: HashMap::new(),
            plan_cursor: 0,
            plan_slice: usize::MAX,
            external_executor: false,
            external_in_flight: 0,
            external_outbox: Vec::new(),
        })
    }

    /// Registers the authoritative-store reconstruction source for an
    /// object (soft-state eviction requires it).
    pub fn pin_authoritative(&mut self, object: u64) {
        if let Some(entry) = self.objects.get_mut(&object)
            && !entry
                .materialization
                .reconstruction
                .contains(&ReconstructionSource::AuthoritativeStore)
        {
            entry
                .materialization
                .reconstruction
                .push(ReconstructionSource::AuthoritativeStore);
        }
    }

    /// Explicit staging admission plan: proves arena headroom for
    /// `bytes` before the caller does staging work. Fails when the
    /// arena cannot fit `bytes` (the insert would fail fast anyway;
    /// this answers overload before any work and feeds backpressure
    /// observability). Background-queue saturation is enforced at each
    /// enqueue point, not here: staging itself never queues work. Never
    /// consulted by consensus (prepare stays pure deterministic).
    ///
    /// # Errors
    ///
    /// Returns [`MemoryError::Overloaded`] when the arena cannot fit
    /// `bytes`, or [`MemoryError::TooLarge`] above `MEDIUM_MAX`.
    pub fn stage_plan(&self, bytes: u64) -> Result<(), MemoryError> {
        if bytes > crate::MEDIUM_MAX as u64 {
            return Err(MemoryError::TooLarge {
                len: bytes,
                max: crate::MEDIUM_MAX as u64,
                context: "fabric insert: use chunk fabric",
            });
        }
        if bytes > crate::TINY_INLINE_MAX as u64
            && self.arenas.used_bytes().saturating_add(bytes) > self.config.arena_capacity_bytes
        {
            return Err(MemoryError::Overloaded { queue: "arena" });
        }
        Ok(())
    }

    /// Inserts an object, routing by size: tiny values stay inline,
    /// medium values enter the behavior-class arena, large values are
    /// rejected to the chunk fabric.
    ///
    /// Returns the fabric-scoped object id.
    ///
    /// # Errors
    ///
    /// Returns [`MemoryError::TooLarge`] for values above `MEDIUM_MAX`
    /// (large-object path owns them), or arena/queue failures.
    pub fn insert(
        &mut self,
        bytes: Bytes,
        intent: MaterializationIntent,
        class: BehaviorClass,
        mutability: Mutability,
    ) -> Result<u64, MemoryError> {
        if bytes.len() > crate::MEDIUM_MAX {
            return Err(MemoryError::TooLarge {
                len: bytes.len() as u64,
                max: crate::MEDIUM_MAX as u64,
                context: "fabric insert: use chunk fabric",
            });
        }
        let object = self.next_object;
        self.next_object += 1;
        let signals = AccessSignals {
            logical_bytes: bytes.len() as u64,
            mutability,
            ..AccessSignals::cold(bytes.len() as u64)
        };
        let criticality = criticality_of(&signals, 90.0);
        let primary = if bytes.len() <= crate::TINY_INLINE_MAX {
            Residence::Inline(bytes)
        } else {
            let handle = self.arenas.alloc(class, bytes)?;
            Residence::Arena {
                class,
                handle,
                provider: self.dram_provider,
            }
        };
        let mut materialization = ObjectMaterialization::new(object, 1, primary);
        materialization
            .reconstruction
            .push(ReconstructionSource::AuthoritativeStore);
        self.objects.insert(
            object,
            ObjectEntry {
                materialization,
                intent,
                class,
                criticality,
                signals,
                ticks_since_move: u64::MAX / 2,
                pending_promotion: None,
                transition: None,
            },
        );
        self.stats.objects = self.objects.len();
        Ok(object)
    }

    /// Reads an object transparently across representations.
    ///
    /// Never blocks on `NVMe`: when the primary is off-core with no
    /// synchronous shadow, a promotion is enqueued and
    /// [`GetOutcome::Pending`] returned.
    ///
    /// # Errors
    ///
    /// Returns [`MemoryError::NoRepresentation`] for unknown objects and
    /// [`MemoryError::CorruptRepresentation`] when a residence fails
    /// verification (the object keeps its other representations).
    pub fn get(&mut self, object: u64) -> Result<GetOutcome, MemoryError> {
        // Clone the primary first so resolution borrows only the arenas
        // (disjoint from the object table borrow below).
        let primary = {
            let entry = self
                .objects
                .get_mut(&object)
                .ok_or(MemoryError::NoRepresentation { object })?;
            entry.signals.reads = entry.signals.reads.saturating_add(1);
            entry.materialization.primary.clone()
        };
        // Fast path: primary synchronous.
        match Self::resolve(&self.arenas, &primary) {
            Ok(bytes) => {
                self.stats.hits += 1;
                let provider = Self::provider_of(self.dram_provider, &primary);
                self.calibrator
                    .for_provider(provider)
                    .observe_read(bytes.len() as u64, 100);
                Ok(GetOutcome::Ready(bytes))
            }
            Err(MemoryError::CorruptRepresentation { detail, .. }) => {
                // Quarantine the corrupt primary when a shadow exists.
                let fallback = {
                    let entry = self
                        .objects
                        .get_mut(&object)
                        .ok_or(MemoryError::NoRepresentation { object })?;
                    if entry.materialization.shadows.is_empty() {
                        return Err(MemoryError::CorruptRepresentation { object, detail });
                    }
                    let corrupt = std::mem::replace(
                        &mut entry.materialization.primary,
                        entry.materialization.shadows.remove(0),
                    );
                    let _ = corrupt;
                    entry.materialization.primary.clone()
                };
                self.stats.hits += 1;
                let bytes = Self::resolve(&self.arenas, &fallback).map_err(|_| {
                    MemoryError::CorruptRepresentation {
                        object,
                        detail: detail.clone(),
                    }
                })?;
                Ok(GetOutcome::Ready(bytes))
            }
            Err(MemoryError::StaleHandle { .. }) => Err(MemoryError::CorruptRepresentation {
                object,
                detail: "arena handle went stale under a live object".to_owned(),
            }),
            Err(MemoryError::ProviderFailed { .. }) => {
                // Off-core or chunked primary: serve a synchronous shadow
                // when one exists (non-exclusive tiering), else enqueue a
                // promotion and report pending (never block).
                let shadow = {
                    let entry = self
                        .objects
                        .get_mut(&object)
                        .ok_or(MemoryError::NoRepresentation { object })?;
                    entry
                        .materialization
                        .shadows
                        .iter()
                        .find(|residence| residence.is_synchronous())
                        .cloned()
                };
                if let Some(shadow) = shadow
                    && let Ok(bytes) = Self::resolve(&self.arenas, &shadow)
                {
                    self.stats.hits += 1;
                    return Ok(GetOutcome::Ready(bytes));
                }
                let version = self
                    .objects
                    .get(&object)
                    .map_or(0, |entry| entry.materialization.version);
                match self.offcore.enqueue_promote(object, version) {
                    Ok(op) => {
                        if let Some(entry) = self.objects.get_mut(&object) {
                            entry.pending_promotion = Some(op);
                        }
                        self.stats.promotion_misses += 1;
                        Ok(GetOutcome::Pending { op })
                    }
                    Err(error) => Err(error),
                }
            }
            Err(other) => Err(other),
        }
    }

    /// Mutates an object: replaces its logical bytes, bumps the version,
    /// and cancels any in-flight transition or queued move (the build was
    /// for the old version).
    ///
    /// # Errors
    ///
    /// Returns [`MemoryError::NoRepresentation`] for unknown objects.
    pub fn mutate(&mut self, object: u64, bytes: Bytes) -> Result<(), MemoryError> {
        if bytes.len() > crate::MEDIUM_MAX {
            return Err(MemoryError::TooLarge {
                len: bytes.len() as u64,
                max: crate::MEDIUM_MAX as u64,
                context: "fabric mutate: use chunk fabric",
            });
        }
        let (old, class, signals) = {
            let entry = self
                .objects
                .get_mut(&object)
                .ok_or(MemoryError::NoRepresentation { object })?;
            entry.materialization.version += 1;
            entry.signals.writes = entry.signals.writes.saturating_add(1);
            entry.signals.logical_bytes = bytes.len() as u64;
            entry.transition = None;
            entry.pending_promotion = None;
            let old = std::mem::replace(
                &mut entry.materialization.primary,
                Residence::Inline(Bytes::new()),
            );
            entry.materialization.shadows.clear();
            (old, entry.class, entry.signals)
        };
        self.moves.cancel_for_object(object);
        self.offcore.cancel_for_object(object);
        self.stats.cancelled += 1;
        // Free the old residence (proven safe: the authoritative source
        // plus the incoming bytes are the new truth).
        Self::release_residence(&mut self.arenas, &old);
        let primary = if bytes.len() <= crate::TINY_INLINE_MAX {
            Residence::Inline(bytes)
        } else {
            let handle = self.arenas.alloc(class, bytes)?;
            Residence::Arena {
                class,
                handle,
                provider: self.dram_provider,
            }
        };
        let entry = self
            .objects
            .get_mut(&object)
            .ok_or(MemoryError::NoRepresentation { object })?;
        entry.materialization.primary = primary;
        entry.criticality = criticality_of(&signals, 90.0);
        Ok(())
    }

    /// Removes an object, releasing all its residences. Reconstruction
    /// sources make this always safe (removal is a logical delete, not a
    /// reclamation optimization).
    ///
    /// # Errors
    ///
    /// Returns [`MemoryError::NoRepresentation`] for unknown objects.
    pub fn remove(&mut self, object: u64) -> Result<(), MemoryError> {
        let entry = self
            .objects
            .remove(&object)
            .ok_or(MemoryError::NoRepresentation { object })?;
        Self::release_residence(&mut self.arenas, &entry.materialization.primary);
        for shadow in &entry.materialization.shadows {
            Self::release_residence(&mut self.arenas, shadow);
        }
        self.moves.cancel_for_object(object);
        self.offcore.cancel_for_object(object);
        self.pins.remove(&object);
        self.stats.objects = self.objects.len();
        Ok(())
    }

    /// Reads an object pinned to an expected logical version: the
    /// engine-facing resolve used for version-pinned reads. Behaves like
    /// [`get`](Self::get) (including `Pending` promotion enqueueing) but
    /// first proves the materialization still names `expected_version` —
    /// a delayed promotion can never serve bytes the read was not
    /// authorized to observe.
    ///
    /// # Errors
    ///
    /// Returns [`MemoryError::MutatedDuringBuild`] when the object moved
    /// past the pinned version (the caller re-prepares), plus every
    /// [`get`](Self::get) failure mode.
    pub fn get_pinned(
        &mut self,
        object: u64,
        expected_version: u64,
    ) -> Result<GetOutcome, MemoryError> {
        let current = self
            .objects
            .get(&object)
            .map(|entry| entry.materialization.version)
            .ok_or(MemoryError::NoRepresentation { object })?;
        if current != expected_version {
            return Err(MemoryError::MutatedDuringBuild {
                object,
                build: expected_version,
                current,
            });
        }
        self.get(object)
    }

    /// Pins an object for an outstanding operation (suspended read,
    /// in-flight checkpoint stream, parked migration). While pinned the
    /// object is never retired and its last synchronous residence is
    /// never reclaimed. Pins nest by count; each `pin` needs one `unpin`.
    pub fn pin(&mut self, object: u64) {
        *self.pins.entry(object).or_insert(0) += 1;
    }

    /// Releases one pin previously taken with [`pin`](Self::pin).
    pub fn unpin(&mut self, object: u64) {
        if let Some(count) = self.pins.get_mut(&object) {
            *count = count.saturating_sub(1);
            if *count == 0 {
                self.pins.remove(&object);
            }
        }
    }

    /// Whether an object currently holds outstanding-operation pins.
    #[must_use]
    pub fn is_pinned(&self, object: u64) -> bool {
        self.pins.get(&object).is_some_and(|count| *count > 0)
    }

    /// Retires an object from the fabric, failing closed while pinned.
    /// The engine calls this only after the logical root no longer names
    /// the id *and* no reconstruction reference (retained band, journal
    /// coverage, live root) needs it; the pin check is defense in depth.
    ///
    /// # Errors
    ///
    /// Returns [`MemoryError::NoRepresentation`] for unknown objects and
    /// [`MemoryError::Cancelled`] while pinned.
    pub fn retire_object(&mut self, object: u64) -> Result<(), MemoryError> {
        if self.is_pinned(object) {
            return Err(MemoryError::Cancelled {
                object,
                reason: "retire while pinned",
            });
        }
        self.remove(object)
    }

    /// Reads bytes from any valid residence without moving them: sync
    /// primary/shadows first, else a direct backend read of the off-core
    /// record. Used by checkpoint streaming and tablet migration so cold
    /// objects never promote into DRAM merely to be serialized or moved.
    /// Never enqueues work and never mutates placement.
    ///
    /// # Errors
    ///
    /// Returns [`MemoryError::NoRepresentation`] for unknown objects and
    /// [`MemoryError::CorruptRepresentation`] when no valid residence
    /// remains.
    pub fn read_anywhere(&mut self, object: u64) -> Result<Bytes, MemoryError> {
        let primary = self
            .objects
            .get(&object)
            .map(|entry| entry.materialization.primary.clone())
            .ok_or(MemoryError::NoRepresentation { object })?;
        if let Ok(bytes) = Self::resolve(&self.arenas, &primary) {
            return Ok(bytes);
        }
        let shadows: Vec<Residence> = self
            .objects
            .get(&object)
            .map(|entry| entry.materialization.shadows.clone())
            .unwrap_or_default();
        for shadow in &shadows {
            if let Ok(bytes) = Self::resolve(&self.arenas, shadow) {
                return Ok(bytes);
            }
        }
        // Last resort: read the off-core record straight off the device.
        let offcore = std::iter::once(&primary)
            .chain(shadows.iter())
            .find_map(|residence| match residence {
                Residence::Offcore { offset, len, .. } => Some((*offset, *len)),
                _ => None,
            })
            .ok_or(MemoryError::NoRepresentation { object })?;
        let bytes = self
            .backend
            .read_record(offcore.0, offcore.1)
            .map_err(|_| MemoryError::CorruptRepresentation {
                object,
                detail: "no valid residence for direct read".to_owned(),
            })?;
        Ok(Bytes::from(bytes))
    }

    /// Writes an object's current bytes to the backend synchronously,
    /// returning the record locator for the engine's durability journal.
    /// The seal-time flush path calls this for every staged id before the
    /// WAL barrier, establishing durable-before-reference ordering.
    ///
    /// # Errors
    ///
    /// Propagates backend write failures; the WAL barrier must not seal
    /// while any staged id lacks a durable record.
    pub fn flush_object_to_backend(&mut self, object: u64) -> Result<(u64, u64, u32), MemoryError> {
        let primary = self
            .objects
            .get(&object)
            .map(|entry| entry.materialization.primary.clone())
            .ok_or(MemoryError::NoRepresentation { object })?;
        let bytes = Self::resolve(&self.arenas, &primary).map_err(|_| {
            MemoryError::CorruptRepresentation {
                object,
                detail: "flush needs a synchronous residence".to_owned(),
            }
        })?;
        let locator = self.backend.write_record(&bytes)?;
        self.calibrator
            .for_provider(self.nvme_provider)
            .observe_write(bytes.len() as u64, 30_000);
        Ok(locator)
    }

    /// Imports one off-core record as a cold object with a fixed id and
    /// version (recovery path). The id must be unused; `next_object`
    /// advances past it so later inserts never collide.
    ///
    /// # Errors
    ///
    /// Returns [`MemoryError::Invalid`] when the id is already tracked.
    pub fn import_offcore(&mut self, import: OffcoreImport) -> Result<(), MemoryError> {
        let id = import.id;
        if self.objects.contains_key(&id) {
            return Err(MemoryError::Invalid {
                detail: "import collides with a live object".to_owned(),
            });
        }
        let mut materialization = ObjectMaterialization::new(
            id,
            import.version,
            Residence::Offcore {
                provider: self.nvme_provider,
                offset: import.offset,
                len: import.len,
                checksum: import.checksum,
            },
        );
        materialization
            .reconstruction
            .push(ReconstructionSource::AuthoritativeStore);
        let signals = AccessSignals::cold(import.len);
        let criticality = criticality_of(&signals, 90.0);
        self.objects.insert(
            id,
            ObjectEntry {
                materialization,
                intent: import.intent,
                class: import.class,
                criticality,
                signals,
                ticks_since_move: u64::MAX / 2,
                pending_promotion: None,
                transition: None,
            },
        );
        self.next_object = self.next_object.max(id.saturating_add(1));
        self.stats.objects = self.objects.len();
        Ok(())
    }

    /// All tracked object ids, for GC marking (live-root collection).
    #[must_use]
    pub fn live_ids(&self) -> Vec<u64> {
        self.objects.keys().copied().collect()
    }

    /// The materialization version of one object, if tracked.
    #[must_use]
    pub fn object_version(&self, object: u64) -> Option<u64> {
        self.objects
            .get(&object)
            .map(|entry| entry.materialization.version)
    }

    /// Queued vs in-flight off-core operation counts (observability).
    #[must_use]
    pub fn offcore_depth(&self) -> (usize, usize) {
        (self.offcore.queued_len(), self.offcore.submitted_len())
    }

    /// Queued background move count (observability).
    #[must_use]
    pub fn move_depth(&self) -> usize {
        self.moves.len()
    }

    /// Current provider capability snapshot (observability).
    #[must_use]
    pub fn provider_caps(&self) -> Vec<ProviderDescriptor> {
        self.registry.all().to_vec()
    }

    /// Tries a synchronous resolve without enqueueing work and without
    /// touching the backend: `Ok` on any valid arena residence, `Err`
    /// otherwise (off-core primaries report `ProviderFailed` so the
    /// caller can route through the external executor). No version
    /// check - pair with [`object_version`](Self::object_version).
    ///
    /// # Errors
    ///
    /// Returns [`MemoryError::NoRepresentation`] for unknown objects and
    /// [`MemoryError::ProviderFailed`] when no arena residence exists
    /// (the caller must promote through the lane first).
    pub fn try_resolve_sync(&self, object: u64) -> Result<Bytes, MemoryError> {
        let entry = self
            .objects
            .get(&object)
            .ok_or(MemoryError::NoRepresentation { object })?;
        if let Ok(bytes) = Self::resolve(&self.arenas, &entry.materialization.primary) {
            return Ok(bytes);
        }
        for shadow in &entry.materialization.shadows {
            if let Ok(bytes) = Self::resolve(&self.arenas, shadow) {
                return Ok(bytes);
            }
        }
        Err(MemoryError::ProviderFailed {
            provider: "offcore",
            object,
            detail: "no synchronous residence; promote first".to_owned(),
        })
    }

    /// Locates the first off-core residence of an object for the external
    /// executor: `(offset, len, materialization version)`. Pure lookup,
    /// no I/O. `None` when the object is unknown or fully resident.
    #[must_use]
    pub fn offcore_locator(&self, object: u64) -> Option<(u64, u64, u64)> {
        let entry = self.objects.get(&object)?;
        let version = entry.materialization.version;
        std::iter::once(&entry.materialization.primary)
            .chain(entry.materialization.shadows.iter())
            .find_map(|residence| match residence {
                Residence::Offcore { offset, len, .. } => Some((*offset, *len, version)),
                _ => None,
            })
    }

    /// Number of currently pinned objects (observability).
    #[must_use]
    pub fn pinned_count(&self) -> usize {
        self.pins.len()
    }

    /// Per-class arena statistics (observability).
    #[must_use]
    pub fn arena_stats(&self) -> [crate::arena::ArenaStats; 7] {
        self.arenas.stats()
    }

    /// Advances virtual time by one tick: reevaluates placement, queues
    /// moves, drains the movement queue under budget, and submits
    /// off-core operations. Deterministic in the tick count. Statistics
    /// refresh every 8th tick and arena compaction runs every 64th tick
    /// at most (both amortized; planning tolerates slightly stale
    /// numbers, compaction is gated on material fragmentation anyway).
    pub fn tick(&mut self) {
        self.now_ticks += 1;
        if let NvmeBackend::Sim(provider) = &mut self.backend {
            provider.advance_to(self.now_ticks);
        }
        self.scheduler.reset_tick();
        self.plan_all();
        self.drain_moves();
        if self.now_ticks.is_multiple_of(8) {
            self.refresh_stats();
        }
        if self.now_ticks.is_multiple_of(64) {
            self.compact_arenas();
        }
    }

    /// Compacts fragmented arenas, publishing slot remaps into the object
    /// table atomically (same `&mut` borrow: no observer ever sees a
    /// half-remapped table). `WorkerArenas::compact` only returns remaps
    /// for materially fragmented arenas (>25% dead), so quiet ticks cost
    /// nothing past the call.
    fn compact_arenas(&mut self) {
        let remap = self.arenas.compact();
        if remap.is_empty() {
            return;
        }
        let mut table = std::collections::HashMap::with_capacity(remap.len() * 2);
        for (class, old, new) in remap {
            table.insert((class, old.slot, old.generation), new);
        }
        for entry in self.objects.values_mut() {
            for residence in std::iter::once(&mut entry.materialization.primary)
                .chain(entry.materialization.shadows.iter_mut())
            {
                let handle = match residence {
                    Residence::Arena { class, handle, .. }
                    | Residence::Compressed { class, handle, .. } => (*class, *handle),
                    _ => continue,
                };
                if let Some(new) = table.get(&(handle.0, handle.1.slot, handle.1.generation)) {
                    match residence {
                        Residence::Arena { handle, .. } | Residence::Compressed { handle, .. } => {
                            *handle = *new;
                        }
                        _ => {}
                    }
                }
            }
        }
        self.stats.compactions += 1;
    }

    /// Applies an off-core completion to its object.
    ///
    /// # Errors
    ///
    /// Returns [`MemoryError::NoRepresentation`] for unknown operations
    /// or objects, and version-fence failures when the object mutated
    /// while the op was in flight.
    pub fn apply_completion(&mut self, completion: OffcoreCompletion) -> Result<(), MemoryError> {
        if !self.offcore.complete(completion.id) {
            return Err(MemoryError::NoRepresentation {
                object: completion.object,
            });
        }
        // Version fence: the object may have mutated while the op was
        // in flight (queued ops are cancelled on mutate, but submitted
        // ones race). A stale completion is discarded without touching
        // state; the authoritative bytes already reflect the mutation.
        if let Some(entry) = self.objects.get(&completion.object)
            && entry.materialization.version != completion.version
        {
            return Err(MemoryError::MutatedDuringBuild {
                object: completion.object,
                build: completion.version,
                current: entry.materialization.version,
            });
        }
        // Release admission for file backend.
        if let NvmeBackend::File(provider) = &mut self.backend {
            provider.release();
        }
        if !completion.ok {
            self.calibrator
                .for_provider(self.nvme_provider)
                .observe_error();
            return Err(MemoryError::ProviderFailed {
                provider: "nvme",
                object: completion.object,
                detail: completion.detail,
            });
        }
        let pressured = self.arenas.pressure() > 0.75;
        // Demote completion installs a verified shadow, then publishes
        // it (non-exclusive: the DRAM copy stays until retired).
        if let (Some(offset), Some(len), Some(checksum)) =
            (completion.offset, completion.len, completion.checksum)
        {
            self.apply_demote_record(completion.object, offset, len, checksum, pressured)?;
        } else if !completion.bytes.is_empty() {
            self.apply_promote_bytes(completion.object, completion.bytes);
        }
        Ok(())
    }

    /// Installs a demoted `NVMe` record as a published shadow residence.
    ///
    /// # Errors
    ///
    /// Returns [`MemoryError::NoRepresentation`] for unknown objects and
    /// [`MemoryError::Invalid`] when shadow publication fails.
    fn apply_demote_record(
        &mut self,
        object: u64,
        offset: u64,
        len: u64,
        checksum: u32,
        pressured: bool,
    ) -> Result<(), MemoryError> {
        let nvme_provider = self.nvme_provider;
        let retired = {
            let entry = self
                .objects
                .get_mut(&object)
                .ok_or(MemoryError::NoRepresentation { object })?;
            let shadow = Residence::Offcore {
                provider: nvme_provider,
                offset,
                len,
                checksum,
            };
            entry.materialization.add_shadow(shadow);
            entry.materialization.publish_shadow(0).map_err(|error| {
                let _ = error;
                MemoryError::Invalid {
                    detail: "publish demote shadow".to_owned(),
                }
            })?;
            // Retire policy: keep the DRAM shadow when pressure is
            // low (non-exclusive insurance); drop it when pressured.
            let mut retired = None;
            let dram_index = entry
                .materialization
                .shadows
                .iter()
                .position(Residence::is_synchronous);
            if pressured
                && let Some(position) = dram_index
                && entry.materialization.can_reclaim(position + 1).is_ok()
            {
                retired = entry.materialization.retire(position + 1).ok();
            }
            entry.ticks_since_move = 0;
            // The planned move settled: clear its mark so the planner
            // sees the object again (cancel paths clear it the same way).
            entry.transition = None;
            retired
        };
        if let Some(retired) = &retired {
            Self::release_residence(&mut self.arenas, retired);
        }
        self.stats.demotions += 1;
        self.stats.migration_bytes += len;
        self.calibrator
            .for_provider(self.nvme_provider)
            .observe_write(len, 30_000);
        Ok(())
    }

    /// Installs promoted bytes as a synchronous shadow so the next read
    /// hits without I/O.
    fn apply_promote_bytes(&mut self, object: u64, bytes: Vec<u8>) {
        let bytes = Bytes::from(bytes);
        let inline = bytes.len() <= crate::TINY_INLINE_MAX;
        let (class, dram_provider) = {
            let Some(entry) = self.objects.get_mut(&object) else {
                return;
            };
            entry.pending_promotion = None;
            entry.ticks_since_move = 0;
            (entry.class, self.dram_provider)
        };
        if inline {
            if let Some(entry) = self.objects.get_mut(&object) {
                entry.materialization.add_shadow(Residence::Inline(bytes));
            }
        } else if let Ok(handle) = self.arenas.alloc(class, bytes)
            && let Some(entry) = self.objects.get_mut(&object)
        {
            entry.materialization.add_shadow(Residence::Arena {
                class,
                handle,
                provider: dram_provider,
            });
        }
        self.stats.promotions += 1;
    }

    /// Whether a submittable op is already moot (removed object,
    /// version fence tripped, duplicate demote, or vacuous promote).
    fn pump_is_redundant(&self, op: &crate::offcore::OffcoreOp) -> bool {
        let Some(entry) = self.objects.get(&op.object) else {
            return true;
        };
        if entry.materialization.version != op.version {
            return true;
        }
        let has_offcore = std::iter::once(&entry.materialization.primary)
            .chain(entry.materialization.shadows.iter())
            .any(|residence| matches!(residence, Residence::Offcore { .. }));
        match op.kind {
            crate::offcore::OffcoreKind::Demote => has_offcore,
            crate::offcore::OffcoreKind::Promote => !has_offcore,
            crate::offcore::OffcoreKind::Delete => false,
        }
    }

    /// Takes the next submittable off-core op for the device backend.
    #[must_use]
    pub fn take_submittable(&mut self) -> Option<crate::offcore::OffcoreOp> {
        self.offcore.take_submittable()
    }

    /// Executes one submittable off-core op synchronously against the
    /// owned backend (deterministic tests and the file baseline) and
    /// applies its completion. Returns `false` when no op was ready.
    ///
    /// # Errors
    ///
    /// Propagates version-fence and object-lifecycle failures from
    /// [`apply_completion`](Self::apply_completion).
    pub fn pump_offcore(&mut self) -> Result<bool, MemoryError> {
        let Some(op) = self.offcore.take_submittable() else {
            return Ok(false);
        };
        // Redundant-work skip: a demote is moot when the object is gone,
        // mutated past the op version, or already holds an off-core
        // residence at this version (duplicate queued demotes must not
        // re-publish stale shadows). A promote is moot under the same
        // conditions or when no off-core residence remains. Skipping
        // saves device I/O and keeps the state machine quiet; the
        // authoritative bytes are already correct either way.
        if self.pump_is_redundant(&op) {
            self.offcore.complete(op.id);
            self.stats.cancelled += 1;
            return Ok(true);
        }
        let completion = self.execute_offcore_op(&op);
        let had_completion = completion.ok || !completion.detail.is_empty();
        // Apply even failures so error accounting stays exact; ignore
        // unknown-object races from concurrent remove().
        match self.apply_completion(completion) {
            Ok(()) => Ok(true),
            Err(MemoryError::NoRepresentation { .. }) => Ok(had_completion),
            Err(error) => Err(error),
        }
    }

    /// Executes one op against the owned backend, building its
    /// completion. Pure device I/O: no object-table mutation happens
    /// here (that is [`apply_completion`](Self::apply_completion)).
    fn execute_offcore_op(&mut self, op: &crate::offcore::OffcoreOp) -> OffcoreCompletion {
        match op.kind {
            crate::offcore::OffcoreKind::Demote => match self.backend.write_record(&op.bytes) {
                Ok((offset, len, checksum)) => OffcoreCompletion {
                    id: op.id,
                    object: op.object,
                    version: op.version,
                    bytes: Vec::new(),
                    offset: Some(offset),
                    len: Some(len),
                    checksum: Some(checksum),
                    ok: true,
                    detail: String::new(),
                },
                Err(error) => OffcoreCompletion {
                    id: op.id,
                    object: op.object,
                    version: op.version,
                    bytes: Vec::new(),
                    offset: None,
                    len: None,
                    checksum: None,
                    ok: false,
                    detail: error.to_string(),
                },
            },
            crate::offcore::OffcoreKind::Promote => self.execute_promote_op(op),
            crate::offcore::OffcoreKind::Delete => OffcoreCompletion {
                id: op.id,
                object: op.object,
                version: op.version,
                bytes: Vec::new(),
                offset: None,
                len: None,
                checksum: None,
                ok: true,
                detail: String::new(),
            },
        }
    }

    /// Reads the off-core residence for a promote op.
    fn execute_promote_op(&mut self, op: &crate::offcore::OffcoreOp) -> OffcoreCompletion {
        // Find the off-core residence to know what to read.
        let residence = self.objects.get(&op.object).and_then(|entry| {
            std::iter::once(&entry.materialization.primary)
                .chain(entry.materialization.shadows.iter())
                .find(|residence| !residence.is_synchronous())
                .cloned()
        });
        let Some(Residence::Offcore { offset, len, .. }) = residence else {
            return OffcoreCompletion {
                id: op.id,
                object: op.object,
                version: op.version,
                bytes: Vec::new(),
                offset: None,
                len: None,
                checksum: None,
                ok: false,
                detail: "no offcore residence for promotion".to_owned(),
            };
        };
        match self.backend.read_record(offset, len) {
            Ok(bytes) => OffcoreCompletion {
                id: op.id,
                object: op.object,
                version: op.version,
                bytes,
                offset: None,
                len: None,
                checksum: None,
                ok: true,
                detail: String::new(),
            },
            Err(error) => OffcoreCompletion {
                id: op.id,
                object: op.object,
                version: op.version,
                bytes: Vec::new(),
                offset: None,
                len: None,
                checksum: None,
                ok: false,
                detail: error.to_string(),
            },
        }
    }

    /// Injects a scripted fault into the `NVMe` backend (sim only).
    pub fn inject_fault(&mut self, fault: SimFault) {
        self.backend.inject_fault(fault);
    }

    /// Current configuration.
    #[must_use]
    pub const fn config(&self) -> &MemoryFabricConfig {
        &self.config
    }

    /// Current statistics snapshot, refreshed by the last
    /// [`tick`](Self::tick). Call `tick` after bulk inserts for fresh
    /// byte accounting.
    #[must_use]
    pub const fn stats(&self) -> FabricStats {
        self.stats
    }

    /// Arena pressure in `[0, 1]`.
    #[must_use]
    pub fn pressure(&self) -> f64 {
        self.arenas.pressure()
    }

    /// Explains one object's placement for `EXPLAIN`-style output.
    #[must_use]
    pub fn explain(&self, object: u64) -> Option<String> {
        let entry = self.objects.get(&object)?;
        Some(format!(
            "object {object} v{} primary={} shadows={} criticality={:.1} class={:?} moves_ago={}",
            entry.materialization.version,
            entry.materialization.primary.name(),
            entry.materialization.shadows.len(),
            entry.criticality.score,
            entry.class,
            entry.ticks_since_move,
        ))
    }

    /// Bytes attributed to one id set, by residence (per-tablet footprint input).
    #[must_use]
    pub fn footprint_for(&self, ids: &std::collections::HashSet<u64>) -> Footprint {
        let mut out = Footprint::default();
        for id in ids {
            let Some(entry) = self.objects.get(id) else {
                continue;
            };
            out.objects += 1;
            for residence in std::iter::once(&entry.materialization.primary)
                .chain(entry.materialization.shadows.iter())
            {
                match residence {
                    Residence::Inline(bytes) => {
                        out.inline_bytes = out.inline_bytes.saturating_add(bytes.len() as u64);
                    }
                    Residence::Arena { class, handle, .. } => {
                        if let Ok(bytes) = self.arenas.get(*class, *handle) {
                            out.dram_bytes = out.dram_bytes.saturating_add(bytes.len() as u64);
                        }
                    }
                    Residence::Compressed { stored_len, .. } => {
                        out.compressed_bytes =
                            out.compressed_bytes.saturating_add(u64::from(*stored_len));
                    }
                    Residence::Offcore { len, .. } => {
                        out.nvme_bytes = out.nvme_bytes.saturating_add(*len);
                    }
                    Residence::Chunked { .. } => {}
                }
            }
        }
        out
    }

    /// Moves all of one tablet's objects into envelopes for migration
    /// (see [`crate::tablet`]). Queued moves and transitions for those
    /// objects are cancelled; residences stay valid across the move.
    /// Bytes resolve from any valid residence (including direct device
    /// reads) so cold tablets migrate without promoting into DRAM.
    pub fn export_tablet(&mut self, objects: &[u64]) -> Vec<(u64, Vec<u8>)> {
        let mut out = Vec::new();
        for object in objects {
            self.moves.cancel_for_object(*object);
            self.offcore.cancel_for_object(*object);
            if let Some(entry) = self.objects.get_mut(object) {
                entry.transition = None;
                entry.pending_promotion = None;
            }
            if let Ok(bytes) = self.read_anywhere(*object) {
                out.push((*object, bytes.to_vec()));
            }
        }
        out
    }
}

impl MemoryFabric {
    /// Maps a residence to its accounting provider. Associated function
    /// (not a method) so callers can hold disjoint borrows of the object
    /// table and the provider ids simultaneously.
    fn provider_of(dram: ProviderId, residence: &Residence) -> ProviderId {
        match residence {
            Residence::Arena { provider, .. }
            | Residence::Compressed { provider, .. }
            | Residence::Offcore { provider, .. } => *provider,
            Residence::Inline(_) | Residence::Chunked { .. } => dram,
        }
    }

    /// Resolves a residence to bytes. Associated function over explicit
    /// arenas so reads never borrow the whole fabric.
    fn resolve(arenas: &WorkerArenas, residence: &Residence) -> Result<Bytes, MemoryError> {
        match residence {
            Residence::Inline(bytes) => Ok(bytes.clone()),
            Residence::Arena { class, handle, .. } => arenas.get(*class, *handle),
            Residence::Compressed {
                class,
                handle,
                logical_len,
                ..
            } => {
                let stored = arenas.get(*class, *handle)?;
                let decoded = crate::compressed::decode_stored(&stored)?;
                if u32::try_from(decoded.len()).unwrap_or(u32::MAX) != *logical_len {
                    return Err(MemoryError::CorruptRepresentation {
                        object: u64::MAX,
                        detail: "compressed length changed".to_owned(),
                    });
                }
                Ok(decoded)
            }
            Residence::Offcore { .. } => Err(MemoryError::ProviderFailed {
                provider: "nvme",
                object: u64::MAX,
                detail: "offcore residence needs promotion".to_owned(),
            }),
            Residence::Chunked { .. } => Err(MemoryError::ProviderFailed {
                provider: "chunk",
                object: u64::MAX,
                detail: "chunked residence resolves in the chunk fabric".to_owned(),
            }),
        }
    }

    /// Releases one residence's arena slot. Associated function over
    /// explicit arenas so mutation paths can hold disjoint borrows.
    fn release_residence(arenas: &mut WorkerArenas, residence: &Residence) {
        match residence {
            Residence::Arena { class, handle, .. }
            | Residence::Compressed { class, handle, .. } => {
                let _ = arenas.free(*class, *handle);
            }
            // Inline and chunked names need no release; off-core records
            // are append-only and reclaim by retention policy (future
            // phase), not by immediate delete.
            Residence::Inline(_) | Residence::Chunked { .. } | Residence::Offcore { .. } => {}
        }
    }

    /// Cancels the in-flight transition and queued moves for an object
    /// (external callers use this when they fence or supersede planned
    /// work: mutation, migration, removal).
    pub fn cancel_transition_for(&mut self, object: u64) {
        self.cancel_transition(object, "external fence");
    }

    /// Bounds the planning scan: at most `slice` objects are planned per
    /// tick, round-robin from a cursor. Replaces the full O(n) scan on
    /// every tick for production-path integration; `usize::MAX` restores
    /// the full scan (tests, small deployments).
    pub fn set_plan_slice(&mut self, slice: usize) {
        self.plan_slice = slice.max(1);
    }

    /// Enables external-executor mode: `Nvme`/promote moves are collected
    /// as [`ExternalOp`]s (drained with
    /// [`drain_external_ops`](Self::drain_external_ops)) instead of the
    /// internal off-core queue, so one outside writer owns all device
    /// appends. Compressed/evict moves still execute inline.
    pub fn set_external_executor(&mut self, external: bool) {
        self.external_executor = external;
    }

    /// Reports how many operations the external executor currently holds
    /// (counts toward the movement concurrency bound).
    pub fn set_external_in_flight(&mut self, in_flight: usize) {
        self.external_in_flight = in_flight;
    }

    /// Drains operations collected for the external executor.
    #[must_use]
    pub fn drain_external_ops(&mut self) -> Vec<ExternalOp> {
        std::mem::take(&mut self.external_outbox)
    }

    /// Releases one admission slot acquired while collecting an external
    /// demote (file-backend endurance accounting). Call once per external
    /// demote completion.
    pub fn release_admission(&mut self) {
        if let NvmeBackend::File(provider) = &mut self.backend {
            provider.release();
        }
    }

    /// Installs a demoted record produced by the external executor:
    /// version-fenced shadow install, publish, and conditional retire —
    /// the same verify/publish phases
    /// [`apply_completion`](Self::apply_completion) runs for internal
    /// completions.
    ///
    /// # Errors
    ///
    /// Returns [`MemoryError::NoRepresentation`] for unknown objects and
    /// [`MemoryError::MutatedDuringBuild`] when the object moved past
    /// `version` while the device op was in flight.
    pub fn install_demote_record(
        &mut self,
        object: u64,
        version: u64,
        offset: u64,
        len: u64,
        checksum: u32,
    ) -> Result<(), MemoryError> {
        let current = self
            .objects
            .get(&object)
            .map(|entry| entry.materialization.version)
            .ok_or(MemoryError::NoRepresentation { object })?;
        if current != version {
            return Err(MemoryError::MutatedDuringBuild {
                object,
                build: version,
                current,
            });
        }
        let pressured = self.arenas.pressure() > 0.75;
        self.apply_demote_record(object, offset, len, checksum, pressured)
    }

    /// Installs promoted bytes produced by the external executor as a
    /// synchronous shadow (same publish phase as internal completions).
    ///
    /// # Errors
    ///
    /// Returns [`MemoryError::NoRepresentation`] for unknown objects and
    /// [`MemoryError::MutatedDuringBuild`] on version races.
    pub fn install_promote_bytes(
        &mut self,
        object: u64,
        version: u64,
        bytes: Vec<u8>,
    ) -> Result<(), MemoryError> {
        let current = self
            .objects
            .get(&object)
            .map(|entry| entry.materialization.version)
            .ok_or(MemoryError::NoRepresentation { object })?;
        if current != version {
            return Err(MemoryError::MutatedDuringBuild {
                object,
                build: version,
                current,
            });
        }
        self.apply_promote_bytes(object, bytes);
        Ok(())
    }

    fn plan_all(&mut self) {
        // Collect candidates first (borrow discipline), then plan a
        // bounded round-robin slice starting at the cursor.
        let mut ids: Vec<u64> = self.objects.keys().copied().collect();
        ids.sort_unstable();
        let start = ids.partition_point(|id| *id < self.plan_cursor);
        let mut ordered = Vec::with_capacity(ids.len());
        ordered.extend_from_slice(&ids[start..]);
        ordered.extend_from_slice(&ids[..start]);
        // Age every object every tick (cheap counters, not planning).
        for object in &ordered {
            if let Some(entry) = self.objects.get_mut(object) {
                entry.ticks_since_move = entry.ticks_since_move.saturating_add(1);
            }
        }
        let slice = self.plan_slice.min(ordered.len().max(1));
        for object in ordered.iter().take(slice) {
            self.plan_one(*object);
        }
        if let Some(last) = ordered.iter().take(slice).next_back() {
            self.plan_cursor = last.saturating_add(1);
        }
        // Memory pressure forces bounded safe reclamation of the least
        // critical residents even when the planner stays quiet.
        if self.arenas.pressure() > 0.85 {
            self.reclaim_under_pressure();
        }
    }

    fn plan_one(&mut self, object: u64) {
        let (intent, criticality, logical_bytes, current, ticks_since_move, migration_cost) =
            match self.objects.get(&object) {
                Some(entry) => {
                    let bytes = self.logical_bytes_of(entry);
                    (
                        entry.intent.clone(),
                        entry.criticality,
                        bytes,
                        self.current_provider(entry),
                        entry.ticks_since_move,
                        bytes.min(1 << 20),
                    )
                }
                None => return,
            };
        // Skip objects mid-transition or mid-promotion.
        if let Some(entry) = self.objects.get(&object)
            && (entry.transition.is_some() || entry.pending_promotion.is_some())
        {
            return;
        }
        let input = PlacementInput {
            object,
            intent: &intent,
            criticality,
            logical_bytes,
            current,
            pressure: self.arenas.pressure(),
            migration_cost_ns: migration_cost * 2,
            ticks_since_move,
        };
        let decision = self.planner.plan(&input, &self.registry);
        if decision.target == ProviderId::NONE {
            return;
        }
        let target = self.transition_target_for(decision.target);
        let Some(target) = target else { return };
        // Compression requires a measured win; the planner proposes,
        // the fabric disposes after measuring the actual block.
        if target == TransitionTarget::Compressed && !criticality.compression_worthy {
            // Still allow demotion pressure to compress cold data: check
            // the class gate instead of the full worthiness flag.
            let entry = self.objects.get(&object).expect("object present");
            if !entry.class.compression_candidate() {
                return;
            }
        }
        let request = crate::movement::MoveRequest {
            object,
            target,
            bytes: logical_bytes,
            cpu_ns: if target == TransitionTarget::Compressed {
                logical_bytes * 3
            } else {
                500
            },
            tenant: 0,
        };
        let entry = self.objects.get_mut(&object).expect("object present");
        entry.transition = Some(Transition::begin(
            object,
            entry.materialization.version,
            target,
        ));
        if self.moves.push(request).is_err() {
            // Queue full: drop the plan and its mark together. Leaving
            // a transition behind a move that was never queued would
            // blind the planner (and reclaim, which also skips
            // in-flight objects) to this object forever: no install or
            // cancel would ever clear it.
            self.cancel_transition(object, "move queue saturated");
        }
    }

    fn current_provider(&self, entry: &ObjectEntry) -> Option<ProviderId> {
        match &entry.materialization.primary {
            Residence::Inline(_) => Some(self.dram_provider),
            Residence::Arena { provider, .. }
            | Residence::Compressed { provider, .. }
            | Residence::Offcore { provider, .. } => Some(*provider),
            Residence::Chunked { .. } => None,
        }
    }

    fn transition_target_for(&self, provider: ProviderId) -> Option<TransitionTarget> {
        let caps = self.registry.get(provider)?;
        if caps.kind == crate::provider::ProviderKind::DRAM {
            Some(TransitionTarget::Dram)
        } else if caps.kind == crate::provider::ProviderKind::COMPRESSED {
            Some(TransitionTarget::Compressed)
        } else if caps.kind == crate::provider::ProviderKind::NVME {
            Some(TransitionTarget::Nvme)
        } else {
            None
        }
    }

    fn logical_bytes_of(&self, entry: &ObjectEntry) -> u64 {
        match Self::resolve(&self.arenas, &entry.materialization.primary) {
            Ok(bytes) => bytes.len() as u64,
            Err(_) => entry.signals.logical_bytes,
        }
    }

    fn drain_moves(&mut self) {
        loop {
            let in_flight = self
                .offcore
                .submitted_len()
                .saturating_add(self.external_in_flight);
            let Some(request) = self.scheduler.next_move(&mut self.moves, in_flight) else {
                break;
            };
            // Version-fence: object may have mutated since planning.
            let version = match self.objects.get(&request.object) {
                Some(entry) => entry.materialization.version,
                None => continue,
            };
            match request.target {
                TransitionTarget::Compressed => self.execute_compress(request.object, version),
                TransitionTarget::Nvme => {
                    if self.external_executor {
                        self.collect_external_demote(request.object, version);
                    } else {
                        self.execute_demote(request.object, version);
                    }
                }
                TransitionTarget::Dram => {
                    if self.external_executor {
                        self.collect_external_promote(request.object, version);
                    } else {
                        self.execute_promote(request.object, version);
                    }
                }
                TransitionTarget::Evicted => self.execute_evict(request.object),
            }
        }
        // Off-core submission is explicit: the caller pumps via
        // pump_offcore() (tests, file baseline) or drains
        // drain_external_ops() (production single-writer executor). This
        // keeps device I/O off the planning path and preserves per-object
        // ordering in the queue until execution.
    }

    /// Resolves a demote payload into the external outbox (admission
    /// accounted; the executor releases it per completion). Returns
    /// whether an operation was queued: unresolvable objects take no
    /// slot, so callers can spend bounded budgets elsewhere.
    fn collect_external_demote(&mut self, object: u64, version: u64) -> bool {
        let primary = match self.objects.get(&object) {
            Some(entry) if entry.materialization.version == version => {
                entry.materialization.primary.clone()
            }
            _ => {
                self.cancel_transition(object, "version fence");
                return false;
            }
        };
        let Ok(bytes) = Self::resolve(&self.arenas, &primary) else {
            return false;
        };
        let admitted = match &mut self.backend {
            NvmeBackend::Sim(_) => true,
            NvmeBackend::File(provider) => {
                matches!(provider.admit(bytes.len() as u64), Admission::Accept)
            }
            NvmeBackend::None => false,
        };
        if !admitted {
            // Navy endurance says no device writes: compress candidates
            // instead of leaving pressure unrelieved (relief without a
            // single device byte). The compress path fences versions and
            // clears the transition mark itself.
            let candidate = self.objects.get(&object).is_some_and(|entry| {
                entry.materialization.version == version && entry.class.compression_candidate()
            });
            if candidate {
                self.execute_compress(object, version);
                return false;
            }
            self.cancel_transition(object, "admission rejected");
            return false;
        }
        // Bound the outbox like the internal queue: saturation defers,
        // never grows without limit.
        if self.external_outbox.len() >= self.config.offcore_queue_capacity {
            self.cancel_transition(object, "external outbox saturated");
            self.release_admission();
            return false;
        }
        self.external_outbox.push(ExternalOp::Demote {
            object,
            version,
            bytes: bytes.to_vec(),
        });
        // Mark the in-flight move so neither the planner nor the next
        // reclaim tick double-submits it; the install and cancel paths
        // clear the mark when the move settles.
        if let Some(entry) = self.objects.get_mut(&object) {
            entry.transition = Some(Transition::begin(object, version, TransitionTarget::Nvme));
        }
        true
    }

    /// Resolves a promote locator into the external outbox.
    fn collect_external_promote(&mut self, object: u64, version: u64) {
        let residence = self.objects.get(&object).and_then(|entry| {
            std::iter::once(&entry.materialization.primary)
                .chain(entry.materialization.shadows.iter())
                .find(|residence| !residence.is_synchronous())
                .cloned()
        });
        match residence {
            Some(Residence::Offcore { offset, len, .. }) => {
                if self.external_outbox.len() >= self.config.offcore_queue_capacity {
                    self.cancel_transition(object, "external outbox saturated");
                    return;
                }
                self.external_outbox.push(ExternalOp::Promote {
                    object,
                    version,
                    offset,
                    len,
                });
            }
            _ => self.cancel_transition(object, "nothing to promote"),
        }
    }

    fn execute_compress(&mut self, object: u64, version: u64) {
        let primary = match self.objects.get(&object) {
            Some(entry) if entry.materialization.version == version => {
                entry.materialization.primary.clone()
            }
            _ => {
                self.cancel_transition(object, "version fence");
                return;
            }
        };
        let Ok(bytes) = Self::resolve(&self.arenas, &primary) else {
            return;
        };
        let Some(block) = CompressedBlock::compress_if_worthwhile(&bytes) else {
            self.cancel_transition(object, "incompressible");
            return;
        };
        // Verify round-trip before publishing (build/verify phases).
        match block.decompress() {
            Ok(decoded) if decoded == bytes => {}
            _ => {
                self.cancel_transition(object, "round-trip mismatch");
                return;
            }
        }
        let logical = bytes.len() as u64;
        let stored = block.stored_len() as u64;
        // Publish: allocate the compressed block, install as shadow,
        // publish, then retire the old primary (safety proved: two
        // valid representations exist between publish and retire).
        let (class, compressed_provider) = match self.objects.get(&object) {
            Some(entry) => (entry.class, self.compressed_provider),
            None => return,
        };
        let Ok(handle) = self.arenas.alloc(class, block.stored.clone()) else {
            self.cancel_transition(object, "arena full");
            return;
        };
        let retired = {
            let Some(entry) = self.objects.get_mut(&object) else {
                let _ = self.arenas.free(class, handle);
                return;
            };
            entry.materialization.add_shadow(Residence::Compressed {
                class,
                handle,
                provider: compressed_provider,
                logical_len: block.logical_len,
                stored_len: u32::try_from(block.stored.len()).unwrap_or(u32::MAX),
            });
            if entry.materialization.publish_shadow(0).is_err() {
                let _ = self.arenas.free(class, handle);
                entry.transition = None;
                self.stats.cancelled += 1;
                return;
            }
            let retired = if entry.materialization.can_reclaim(1).is_ok() {
                entry.materialization.retire(1).ok()
            } else {
                None
            };
            entry.transition = None;
            entry.ticks_since_move = 0;
            retired
        };
        if let Some(retired) = &retired {
            Self::release_residence(&mut self.arenas, retired);
        }
        self.compressed_logical += logical;
        self.compressed_stored += stored;
        self.calibrator
            .for_provider(self.compressed_provider)
            .observe_compression(logical, stored, 2.0);
        self.stats.migration_bytes += logical;
        let _ = version;
    }

    fn execute_demote(&mut self, object: u64, version: u64) {
        let primary = match self.objects.get(&object) {
            Some(entry) if entry.materialization.version == version => {
                entry.materialization.primary.clone()
            }
            _ => {
                self.cancel_transition(object, "version fence");
                return;
            }
        };
        let Ok(bytes) = Self::resolve(&self.arenas, &primary) else {
            return;
        };
        // Admission first (Navy endurance): rejection defers, never blocks.
        let admitted = match &mut self.backend {
            NvmeBackend::Sim(_) => true,
            NvmeBackend::File(provider) => {
                matches!(provider.admit(bytes.len() as u64), Admission::Accept)
            }
            NvmeBackend::None => false,
        };
        if !admitted {
            // Navy endurance says no device writes: compress candidates
            // instead of leaving pressure unrelieved (relief without a
            // single device byte). The compress path fences versions and
            // clears the transition mark itself.
            let candidate = self.objects.get(&object).is_some_and(|entry| {
                entry.materialization.version == version && entry.class.compression_candidate()
            });
            if candidate {
                self.execute_compress(object, version);
                return;
            }
            self.cancel_transition(object, "admission rejected");
            return;
        }
        if self
            .offcore
            .enqueue_demote(object, version, bytes.to_vec())
            .is_err()
        {
            self.cancel_transition(object, "offcore saturated");
            if let NvmeBackend::File(provider) = &mut self.backend {
                provider.release();
            }
        }
        // The transition completes in apply_completion (verify/publish
        // phases run against the device record, not staged bytes).
    }

    fn execute_promote(&mut self, object: u64, version: u64) {
        let has_offcore = self.objects.get(&object).is_some_and(|entry| {
            std::iter::once(&entry.materialization.primary)
                .chain(entry.materialization.shadows.iter())
                .any(|residence| matches!(residence, Residence::Offcore { .. }))
        });
        if !has_offcore {
            self.cancel_transition(object, "nothing to promote");
            return;
        }
        if self.offcore.enqueue_promote(object, version).is_err() {
            self.cancel_transition(object, "offcore saturated");
        }
    }

    fn execute_evict(&mut self, object: u64) {
        let Some(entry) = self.objects.get_mut(&object) else {
            return;
        };
        if entry.materialization.reconstruction.is_empty() {
            self.cancel_transition(object, "no reconstruction source");
            return;
        }
        // Evict shadows first (proven safe one by one), keep primary.
        while entry.materialization.shadows.len() > 1 {
            let index = entry.materialization.shadows.len();
            if entry.materialization.can_reclaim(index).is_ok() {
                let retired = entry.materialization.retire(index).expect("proven safe");
                let _ = retired;
            } else {
                break;
            }
        }
        entry.transition = None;
    }

    fn cancel_transition(&mut self, object: u64, _reason: &'static str) {
        if let Some(entry) = self.objects.get_mut(&object) {
            entry.transition = None;
        }
        self.stats.cancelled += 1;
    }

    /// Retires one synchronous shadow of an object whose primary no
    /// longer needs DRAM (already off-core or otherwise non-synchronous),
    /// freeing arena bytes without device I/O. Reads still serve through
    /// promotion; the off-core record stays authoritative. Returns whether
    /// a copy was retired.
    fn retire_redundant_shadow(&mut self, object: u64) -> bool {
        let actionable = match self.objects.get(&object) {
            Some(entry) => {
                Self::resolve(&self.arenas, &entry.materialization.primary).is_err()
                    && entry
                        .materialization
                        .shadows
                        .iter()
                        .any(Residence::is_synchronous)
            }
            None => return false,
        };
        if !actionable {
            return false;
        }
        let retired = {
            let Some(entry) = self.objects.get_mut(&object) else {
                return false;
            };
            let Some(position) = entry
                .materialization
                .shadows
                .iter()
                .position(Residence::is_synchronous)
            else {
                return false;
            };
            // Shadow numbering starts past the primary.
            if entry.materialization.can_reclaim(position + 1).is_err() {
                return false;
            }
            entry.materialization.retire(position + 1).ok()
        };
        if let Some(retired) = &retired {
            Self::release_residence(&mut self.arenas, retired);
            true
        } else {
            false
        }
    }

    /// Sheds a synchronous primary with an off-core shadow, promoting the
    /// shadow: the mirror of [`retire_redundant_shadow`](Self::retire_redundant_shadow)
    /// for demotes that kept DRAM insurance. Demotion publishes off-core
    /// as a shadow and only retires the arena copy while pressured at
    /// install time; without this, a full arena of already-demoted
    /// objects never frees and every new stage fails forever (write
    /// starvation with reads still serving). Frees arena bytes without
    /// device I/O; later reads promote through the verified record.
    /// Pinned objects never shed (pin contract). Returns whether the
    /// primary was shed.
    fn shed_insured_primary(&mut self, object: u64) -> bool {
        if self.is_pinned(object) {
            return false;
        }
        let actionable = match self.objects.get(&object) {
            Some(entry) => {
                matches!(
                    entry.materialization.primary,
                    Residence::Arena { .. } | Residence::Compressed { .. }
                ) && entry
                    .materialization
                    .shadows
                    .iter()
                    .any(|residence| matches!(residence, Residence::Offcore { .. }))
            }
            None => return false,
        };
        if !actionable {
            return false;
        }
        let retired = {
            let Some(entry) = self.objects.get_mut(&object) else {
                return false;
            };
            // Promote the off-core copy specifically (it may sit behind
            // other shadows): the arena primary is what must go.
            if let Some(position) = entry
                .materialization
                .shadows
                .iter()
                .position(|residence| matches!(residence, Residence::Offcore { .. }))
            {
                entry.materialization.shadows.swap(0, position);
            }
            // Retiring the primary promotes the first shadow; the proof
            // needs one other valid representation, which the off-core
            // shadow is (verified at install, checksummed on promote).
            if entry.materialization.can_reclaim(0).is_ok() {
                entry.materialization.retire(0).ok()
            } else {
                None
            }
        };
        if let Some(retired) = &retired {
            Self::release_residence(&mut self.arenas, retired);
            true
        } else {
            false
        }
    }

    fn reclaim_under_pressure(&mut self) {
        // Least-critical first, bounded per tick.
        let mut ids: Vec<u64> = self.objects.keys().copied().collect();
        ids.sort_by(|left, right| {
            let left_score = self
                .objects
                .get(left)
                .map_or(f64::INFINITY, |entry| entry.criticality.score);
            let right_score = self
                .objects
                .get(right)
                .map_or(f64::INFINITY, |entry| entry.criticality.score);
            left_score
                .partial_cmp(&right_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut reclaimed = 0;
        for object in ids {
            if self.arenas.pressure() <= 0.85 || reclaimed >= 8 {
                break;
            }
            let (version, candidate) = match self.objects.get(&object) {
                Some(entry) => {
                    // Skip objects with a move already in flight
                    // (planned or reclaimed): re-pushing them every tick
                    // burns the bounded budget on duplicates while other
                    // residents starve.
                    if entry.transition.is_some() {
                        continue;
                    }
                    (
                        entry.materialization.version,
                        entry.class.compression_candidate(),
                    )
                }
                None => continue,
            };
            // Primary already off-core with a synchronous shadow still
            // pinning arena bytes: shed the redundant copy without any
            // device round-trip. Neither planning (which moves
            // primaries) nor collection (which resolves primaries) can
            // see this state, so without this the bytes pin the arena
            // forever under sustained pressure.
            if self.retire_redundant_shadow(object) {
                reclaimed += 1;
                continue;
            }
            // Synchronous primary with an off-core shadow (demote kept
            // DRAM insurance): shed the arena copy the same way. Without
            // this a full arena of already-demoted objects never frees
            // and every new stage fails forever.
            if self.shed_insured_primary(object) {
                reclaimed += 1;
                continue;
            }
            // Prefer compression for candidates, demotion otherwise.
            // External-executor deployments never pump the internal
            // off-core queue, so pressure demotions must join the
            // external outbox like planned moves: queuing them
            // internally would stall the arena (and the movement
            // scheduler that counts them in flight) forever.
            // Unresolvable objects take no budget slot, so reclaimable
            // residents behind them in the order still get served.
            if candidate {
                self.execute_compress(object, version);
                reclaimed += 1;
            } else if self.external_executor {
                if self.collect_external_demote(object, version) {
                    reclaimed += 1;
                }
            } else {
                self.execute_demote(object, version);
                reclaimed += 1;
            }
        }
    }

    fn refresh_stats(&mut self) {
        let mut inline_bytes = 0u64;
        let mut dram_bytes = 0u64;
        let mut compressed_bytes = 0u64;
        let mut nvme_bytes = 0u64;
        for entry in self.objects.values() {
            for residence in std::iter::once(&entry.materialization.primary)
                .chain(entry.materialization.shadows.iter())
            {
                match residence {
                    Residence::Inline(bytes) => {
                        inline_bytes = inline_bytes.saturating_add(bytes.len() as u64);
                    }
                    Residence::Arena { class, handle, .. } => {
                        if let Ok(bytes) = self.arenas.get(*class, *handle) {
                            dram_bytes = dram_bytes.saturating_add(bytes.len() as u64);
                        }
                    }
                    Residence::Compressed { stored_len, .. } => {
                        compressed_bytes += u64::from(*stored_len);
                    }
                    Residence::Offcore { len, .. } => nvme_bytes += *len,
                    Residence::Chunked { .. } => {}
                }
            }
        }
        self.stats.inline_bytes = inline_bytes;
        self.stats.dram_bytes = dram_bytes;
        self.stats.compressed_bytes = compressed_bytes;
        self.stats.nvme_bytes = nvme_bytes;
        self.stats.objects = self.objects.len();
        self.stats.compression_ratio_bps = self
            .compressed_stored
            .saturating_mul(10_000)
            .checked_div(self.compressed_logical)
            .unwrap_or(0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_fabric() -> MemoryFabric {
        MemoryFabric::new(MemoryFabricConfig {
            arena_capacity_bytes: 1 << 20,
            nvme_capacity_bytes: 16 << 20,
            ..MemoryFabricConfig::default()
        })
        .expect("fabric")
    }

    #[test]
    fn tiny_values_stay_inline_without_arena() {
        let mut fabric = test_fabric();
        let object = fabric
            .insert(
                Bytes::from_static(b"tiny"),
                MaterializationIntent::hot(),
                BehaviorClass::HotMutable,
                Mutability::Mutable,
            )
            .expect("insert");
        let entry = fabric.objects.get(&object).expect("entry");
        assert!(matches!(
            entry.materialization.primary,
            Residence::Inline(_)
        ));
        assert_eq!(fabric.arenas.live_count(), 0);
    }

    #[test]
    fn medium_values_enter_behavior_arenas() {
        let mut fabric = test_fabric();
        let bytes = Bytes::from(vec![0xABu8; 1024]);
        let object = fabric
            .insert(
                bytes.clone(),
                MaterializationIntent::hot(),
                BehaviorClass::HotReadMostly,
                Mutability::ReadMostly,
            )
            .expect("insert");
        assert_eq!(fabric.arenas.live_count(), 1);
        let outcome = fabric.get(object).expect("get");
        assert_eq!(outcome, GetOutcome::Ready(bytes));
    }

    #[test]
    fn large_values_are_rejected_to_the_chunk_fabric() {
        let mut fabric = test_fabric();
        let bytes = Bytes::from(vec![0u8; crate::MEDIUM_MAX + 1]);
        let error = fabric
            .insert(
                bytes,
                MaterializationIntent::hot(),
                BehaviorClass::HotMutable,
                Mutability::Mutable,
            )
            .expect_err("must reject");
        assert!(matches!(error, MemoryError::TooLarge { .. }));
    }

    #[test]
    fn mutation_fences_in_flight_work() {
        let mut fabric = test_fabric();
        let object = fabric
            .insert(
                Bytes::from(vec![1u8; 1024]),
                MaterializationIntent::hot(),
                BehaviorClass::HotMutable,
                Mutability::Mutable,
            )
            .expect("insert");
        fabric
            .mutate(object, Bytes::from(vec![2u8; 1024]))
            .expect("mutate");
        let outcome = fabric.get(object).expect("get");
        assert_eq!(outcome, GetOutcome::Ready(Bytes::from(vec![2u8; 1024])));
    }

    #[test]
    fn arena_compaction_remaps_live_handles_transparently() {
        let mut fabric = test_fabric();
        let mut live = Vec::new();
        for index in 0u8..10 {
            let object = fabric
                .insert(
                    Bytes::from(vec![index; 1024]),
                    MaterializationIntent::hot(),
                    BehaviorClass::HotMutable,
                    Mutability::Mutable,
                )
                .expect("insert");
            live.push((object, index));
        }
        // Free half the slots (past the 25% fragmentation gate), then
        // run a full compaction cadence: the slot remap must publish
        // into the object table atomically, so every survivor still
        // reads while the freed ids fail closed.
        let mut freed = std::collections::HashSet::new();
        for (object, _) in live.iter().step_by(2) {
            fabric.remove(*object).expect("remove");
            freed.insert(*object);
        }
        for _ in 0..64 {
            fabric.tick();
        }
        assert!(fabric.stats().compactions >= 1, "compact ran");
        for (object, index) in &live {
            if freed.contains(object) {
                assert!(fabric.get(*object).is_err(), "freed id fails closed");
            } else {
                let outcome = fabric.get(*object).expect("survivor reads");
                assert_eq!(outcome, GetOutcome::Ready(Bytes::from(vec![*index; 1024])));
            }
        }
    }
}
