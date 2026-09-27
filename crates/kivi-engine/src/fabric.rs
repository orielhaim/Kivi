//! Engine Memory Fabric integration: logical authority stays in
//! [`ObjectStore`](kivi_state::ObjectStore); physical bytes live here.
//!
//! Ownership split (never two truths):
//!
//! ```text
//! LOGICAL (kivi-state, deterministic, replicated, WAL'd):
//!     StoredObject{LogicalValue::Fabric(FabricRef{id,len,version}), version, expiry}
//!     Mutation::ReplaceFabricRoot{id,len,version} (small root only)
//!
//! PHYSICAL (this module + kivi-memory, worker-local, never replicated):
//!     MemoryFabric arenas/compressed/planning, demotion lane records,
//!     journal locators, parked-operation pins
//!
//! DURABLE SOURCES (reconstruction, consulted in order):
//!     live fabric residence → fabric journal record → WAL root history →
//!     checkpoint band reference
//! ```
//!
//! A fabric id, arena slot, `NVMe` offset, or provider id never enters
//! Mutation IR beyond the opaque [`FabricRef`](kivi_state::FabricRef)
//! name — exactly like [`ChunkedRef`](kivi_state::ChunkedRef) names a
//! manifest without naming packs. Topology moves logical bytes and
//! re-stages fresh ids; single-writer rules keep both append files
//! consistent without locks.

use std::collections::{HashMap, HashSet, VecDeque};
use std::path::{Path, PathBuf};

use bytes::Bytes;
use kivi_hardware::{PlacementRole, WorkerSlot};

use crate::placement::ThreadPlacement;

use kivi_memory::{
    BehaviorClass, ExternalOp, MemoryError, MemoryFabric, MemoryFabricConfig, Mutability,
};
use kivi_state::{FabricRef, Key};
use kivi_types::{TabletId, WorkerId};

use kivi_memory::offcore_lane::{OffcoreLaneGuard, OffcoreLaneHandle, OffcoreReply, PromotedBytes};

/// Values at or below this stay inline in the WAL record (no arena slot,
/// no off-core locator, no second durable representation).
///
/// The floor is a *value* judgement, not a technical limit. A small value
/// costs the WAL only the bytes themselves; routing it through the fabric
/// instead buys a second copy on the write path and an indirection on the
/// read path, in exchange for nothing. The fabric earns its keep on real
/// bulk, where keeping bytes out of the durable log and out of the object
/// store is worth an arena slot and an off-core locator. One kibibyte is
/// where that trade stops being favourable: below it the copy and the
/// indirection dominate, above it the avoided bytes do.
pub const FABRIC_INLINE_MAX: usize = 1024;

/// Version stamped on a fabric object that has no committed version yet.
///
/// A freshly staged value is not published under any version until its
/// mutation seals, so its locator cannot carry one. Reads are located by
/// offset and length and the authoritative version lives on the state root,
/// so this is metadata rather than a fence. It matches the version
/// [`MemoryFabric::insert`] gives an arena-resident object, which keeps one
/// rule for "not yet published" across both tiers.
const STAGED_FABRIC_VERSION: u64 = 1;

/// Memory Fabric configuration: capacity policy per worker. One value
/// shared by every worker; per-worker fabrics open beneath it.
#[derive(Debug, Clone)]
pub struct FabricConfig {
    /// DRAM arena bound in bytes per worker.
    pub arena_bytes_per_worker: u64,
    /// Demotion capacity in bytes per worker (optimization copies).
    pub demotion_bytes_per_worker: u64,
}

impl FabricConfig {
    /// Default DRAM arena bound per worker: 256 MiB.
    pub const DEFAULT_ARENA_BYTES: u64 = 256 * 1024 * 1024;
    /// Default demotion capacity per worker: 4 GiB.
    pub const DEFAULT_DEMOTION_BYTES: u64 = 4 * 1024 * 1024 * 1024;
}

impl Default for FabricConfig {
    fn default() -> Self {
        Self {
            arena_bytes_per_worker: Self::DEFAULT_ARENA_BYTES,
            demotion_bytes_per_worker: Self::DEFAULT_DEMOTION_BYTES,
        }
    }
}

/// Journal filename inside the worker fabric directory.
const JOURNAL_FILE: &str = "fabric.journal";
/// Durable record filename inside the worker fabric directory.
const MATERIAL_FILE_DIR: &str = "material";
/// Demotion lane subdirectory name.
const DEMOTION_DIR: &str = "demotion";

/// Fabric integration failure modes. `Overloaded` is backpressure (retry
/// with backoff); everything else fails the single operation closed.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum FabricError {
    /// A bounded queue, budget, or lane is saturated.
    #[error("memory fabric saturated")]
    Overloaded,
    /// The fabric id is unknown or the version moved (stale completion,
    /// retired id, torn reference). Never served as bytes.
    #[error("fabric reference unavailable")]
    Unavailable,
    /// A record failed verification (checksum, length, framing).
    /// Quarantined; other representations still stand.
    #[error("fabric record corrupt")]
    Corrupt,
}

impl From<MemoryError> for FabricError {
    fn from(error: MemoryError) -> Self {
        match error {
            MemoryError::Overloaded { .. } | MemoryError::BudgetExhausted { .. } => {
                Self::Overloaded
            }
            MemoryError::CorruptRepresentation { .. } => Self::Corrupt,
            _ => Self::Unavailable,
        }
    }
}

/// One medium value staged at admission: the fabric id plus the bytes
/// the durability lane persists before the WAL barrier. Shared by the
/// worker admission paths and the commit pipeline (which fills the
/// predicted version at prepare and moves bytes into the seal).
#[derive(Debug, Clone)]
pub struct StagedSeal {
    /// Fabric-scoped object id.
    pub fabric_id: u64,
    /// Target key.
    pub key: Key,
    /// Staged payload bytes.
    pub bytes: Bytes,
}

/// One staged payload riding a seal to the durability lane: bytes plus
/// the logical coordinates the journal records. Bounded by the batch
/// byte cap like the WAL records themselves.
#[derive(Debug, Clone)]
pub struct FabricSealPayload {
    /// Fabric-scoped object id staged at admission.
    pub fabric_id: u64,
    /// Owning tablet.
    pub tablet: TabletId,
    /// Logical key (journal observability).
    pub key: Key,
    /// Predicted authoritative version (equals the applied version by
    /// predict/apply agreement; recovery re-links on it).
    pub version: u64,
    /// Staged payload bytes.
    pub bytes: Bytes,
}

/// One sealed locator riding home on the seal outcome: the durable
/// reconstruction source the worker mirrors into memory.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FabricSealLocator {
    /// Fabric-scoped object id.
    pub fabric_id: u64,
    /// Owning tablet.
    pub tablet: TabletId,
    /// Logical key.
    pub key: Key,
    /// Sealed version.
    pub version: u64,
    /// Device offset of the durable record.
    pub offset: u64,
    /// Stored payload length.
    pub len: u64,
    /// CRC32C of the payload.
    pub checksum: u32,
}

/// One parked off-core wait: a suspended operation plus everything the
/// resume path needs to reject it as stale (tablet authority triple is
/// re-checked by the caller; key/version/id reject value staleness).
#[derive(Debug)]
pub struct ParkedOp {
    /// Tablet the operation runs against.
    pub tablet: TabletId,
    /// Key being read (for root re-validation on resume).
    pub key: Key,
    /// Logical version the read was authorized for.
    pub version: u64,
    /// Fabric id the read was authorized for.
    pub fabric_id: u64,
    /// Promotion reply awaited (bridge polls, channel worker blocks).
    pub reply: OffcoreReply<Result<PromotedBytes, MemoryError>>,
}

/// Returns the pinned version when the current root still names exactly
/// `(reference.id, reference.version)`: the single re-fencing predicate
/// every resolve/promote/install path shares. Anything else means the
/// value moved and the caller must re-prepare, never serve old bytes.
fn pinned_root_version(
    current: Option<&kivi_state::StoredObject>,
    reference: &FabricRef,
) -> Option<u64> {
    let object = current?;
    let live = object.fabric_ref()?;
    if live == *reference && object.version().as_u64() == reference.version {
        Some(live.version)
    } else {
        None
    }
}

/// One maintenance-submitted lane operation awaiting collection.
#[derive(Debug)]
struct MaintWait {
    /// Fabric object served.
    object: u64,
    /// Logical version issued for.
    version: u64,
    /// Demote replies carry record locators; promote replies bytes.
    demote: Option<OffcoreReply<Result<kivi_memory::offcore_lane::DemotedRecord, MemoryError>>>,
    /// Promote reply (exactly one of the two is `Some`).
    promote: Option<OffcoreReply<Result<PromotedBytes, MemoryError>>>,
}

/// Maximum maintenance-awaited lane operations (backpressure: when full,
/// maintenance submits nothing new until completions drain).
const MAX_MAINT_WAITS: usize = 64;

/// Per-worker fabric directories.
#[derive(Debug, Clone)]
pub struct FabricPaths {
    /// Worker fabric root (`<data_dir>/fabric-worker-N` or `TempDir`).
    pub root: PathBuf,
}

impl FabricPaths {
    /// Durable-mode paths for one worker index.
    #[must_use]
    pub fn worker(data_dir: &Path, index: usize) -> Self {
        Self {
            root: data_dir.join(format!("fabric-worker-{index}")),
        }
    }

    /// Journal file path.
    #[must_use]
    pub fn journal(&self) -> PathBuf {
        self.root.join(JOURNAL_FILE)
    }

    /// Durable record directory (durability-lane single writer).
    #[must_use]
    pub fn material_dir(&self) -> PathBuf {
        self.root.join(MATERIAL_FILE_DIR)
    }

    /// Demotion lane directory (offcore-lane single writer).
    #[must_use]
    pub fn demotion_dir(&self) -> PathBuf {
        self.root.join(DEMOTION_DIR)
    }
}

/// Worker-owned Memory Fabric integration: planning/resident state plus
/// the demotion lane, the durable-locator mirror, parked waits, deferred
/// retirements, and bounded maintenance.
#[derive(Debug)]
pub struct TabletFabric {
    fabric: MemoryFabric,
    lane: OffcoreLaneHandle,
    /// Parked promotion waits keyed by park sequence.
    parked: HashMap<u64, ParkedOp>,
    next_park: u64,
    /// Maintenance-submitted lane operations awaiting collection.
    maint_waits: VecDeque<MaintWait>,
    /// Fabric ids whose logical roots are gone but which were pinned at
    /// retire time; swept on unpin and maintenance.
    retire_pending: HashSet<u64>,
    /// Ticks since last maintenance (amortizes planning cadence).
    since_maintenance: u64,
    /// Stale completions discarded (observability).
    pub stale_completions: u64,
    /// Promotions served from parked waits (observability).
    pub parked_resumes: u64,
    /// Staging admission rejections (observability: backpressure events).
    pub admission_rejects: u64,
    /// Preferred NUMA node for this worker's materializations (`None`
    /// on single-node topologies): recorded into every staging intent
    /// as an advisory preference, never a hard constraint.
    numa_node: Option<u32>,
}

impl TabletFabric {
    /// Opens a worker fabric with no thread placement, for tests and for
    /// embedders that manage their own threads. Real callers pass the node's
    /// [`ThreadPlacement`] so the fabric's locality matches the thread that
    /// owns it.
    ///
    /// # Errors
    ///
    /// Returns [`FabricError::Unavailable`] when the memory fabric or the
    /// demotion lane cannot be opened.
    pub fn open_unbound(
        worker: WorkerId,
        paths: &FabricPaths,
        arena_bytes: u64,
        demotion_bytes: u64,
    ) -> Result<(Self, OffcoreLaneGuard), FabricError> {
        Self::open(
            worker,
            paths,
            arena_bytes,
            demotion_bytes,
            &ThreadPlacement::unbound(),
        )
    }

    /// Opens a worker fabric: memory fabric (external-executor mode,
    /// bounded plan slices), demotion lane thread, and the memory node the
    /// owner thread was placed on.
    ///
    /// # Errors
    ///
    /// Returns [`FabricError::Unavailable`] when the memory fabric or
    /// the demotion lane cannot be opened.
    pub fn open(
        worker: WorkerId,
        paths: &FabricPaths,
        arena_bytes: u64,
        demotion_bytes: u64,
        placement: &ThreadPlacement,
    ) -> Result<(Self, OffcoreLaneGuard), FabricError> {
        std::fs::create_dir_all(&paths.root).map_err(|_| FabricError::Unavailable)?;

        // Locality comes from the *planned* core of the owning thread, not from
        // the worker index: the index says nothing about which node the
        // scheduler placed the thread on. It must arrive *before* construction,
        // because the fabric registers one provider per memory node so the
        // planner can score remote residency.
        let ordinal = usize::try_from(worker.as_u64()).unwrap_or(0);
        let owner_slot = WorkerSlot::new(PlacementRole::DataWorker, ordinal);
        let topology = &kivi_hardware::snapshot().topology;
        let numa_node = placement
            .plan()
            .placement(owner_slot)
            .and_then(|p| p.numa_node);

        let mut fabric = MemoryFabric::new(MemoryFabricConfig {
            arena_capacity_bytes: arena_bytes,
            nvme_capacity_bytes: demotion_bytes,
            nvme_dir: Some(paths.demotion_dir()),
            numa_node: numa_node.map(|node| node.0),
            numa_node_count: u32::try_from(topology.memory_nodes.len()).unwrap_or(u32::MAX),
            ..MemoryFabricConfig::default()
        })
        .map_err(FabricError::from)?;
        fabric.set_external_executor(true);
        fabric.set_plan_slice(64);
        if let Some(node) = numa_node.map(|node| node.0) {
            // The invariant that matters: when placement named a node, the
            // fabric's resident provider is on it. An unplanned fabric is not
            // wrong, it just has no preference and falls back to node 0.
            debug_assert_eq!(
                fabric.locality().numa_node,
                Some(node),
                "the fabric's resident provider must be the owner's planned node",
            );
        }

        let bind = placement.clone();
        let (lane, guard) = kivi_memory::offcore_lane::spawn_offcore_lane(
            worker,
            &paths.root,
            kivi_memory::NvmeOptions::default(),
            Some(Box::new(move || {
                let slot = WorkerSlot::new(PlacementRole::OffcoreLane, ordinal);
                if let Err(error) = bind.bind(slot) {
                    tracing::warn!(?error, worker = worker.as_u64(), "offcore lane unbound");
                }
            })),
        )
        .map_err(FabricError::from)?;
        Ok((
            Self {
                fabric,
                lane,
                parked: HashMap::new(),
                next_park: 1,
                maint_waits: VecDeque::new(),
                retire_pending: HashSet::new(),
                since_maintenance: 0,
                stale_completions: 0,
                parked_resumes: 0,
                admission_rejects: 0,
                numa_node: numa_node.map(|node| node.0),
            },
            guard,
        ))
    }

    /// Stages medium bytes into DRAM residency, returning `(id, len)`
    /// descriptor inputs for `SetFabric`. Synchronous and bounded: arena
    /// exhaustion or saturation fails fast with `Overloaded`, never
    /// partial state.
    ///
    /// Callers keep `bytes` for the seal payload; the fabric holds its
    /// own copy for serving.
    ///
    /// # Errors
    ///
    /// Returns [`FabricError::Unavailable`] when the value is above the
    /// medium limit, or [`FabricError::Overloaded`] when neither the arena
    /// nor the off-core device can take the bytes.
    pub fn stage(
        &mut self,
        tablet: TabletId,
        bytes: &Bytes,
        hot: bool,
    ) -> Result<(u64, u64), FabricError> {
        let len = bytes.len() as u64;
        // Explicit admission plan before any staging work, so a commit that
        // reaches the seal can never meet an unrepresentable payload.
        // Consensus determinism never depends on this (prepare stays pure).
        //
        // Saturation is a *tiering* signal, not a refusal: the arena is a
        // cache of hot medium values and the off-core device holds the rest of
        // the working set.
        match self.fabric.stage_plan(len) {
            Ok(()) => {}
            Err(kivi_memory::MemoryError::Overloaded { .. }) => self.admission_rejects += 1,
            Err(error) => return Err(FabricError::from(error)),
        }
        let mut intent = if hot {
            kivi_memory::MaterializationIntent::hot()
        } else {
            kivi_memory::MaterializationIntent::cold()
        };
        intent.tablet = Some(tablet);
        if let Some(node) = self.numa_node {
            intent.locality = kivi_memory::LocalityRequirement::PreferredNuma { node };
        }
        match self.fabric.insert(
            bytes.clone(),
            intent.clone(),
            BehaviorClass::HotMutable,
            Mutability::Mutable,
        ) {
            Ok(id) => Ok((id, len)),
            // The arena bound moved between the plan and the insert (or the
            // plan was optimistic). Same answer: take the cold tier.
            Err(kivi_memory::MemoryError::Overloaded { .. }) => {
                self.admission_rejects += 1;
                self.stage_offcore(bytes, len, intent)
            }
            Err(error) => Err(FabricError::from(error)),
        }
    }

    /// Stages a value on the off-core tier because the hot arena cannot take
    /// it.
    ///
    /// The write is a relaxed append — no fsync — because a demotion record
    /// is an optimization copy that nothing references durably. The bytes
    /// still ride the WAL as a `MaterialRecord` when the mutation seals, so
    /// the cold tier changes where a value lives, never whether it survives.
    fn stage_offcore(
        &mut self,
        bytes: &Bytes,
        len: u64,
        intent: kivi_memory::MaterializationIntent,
    ) -> Result<(u64, u64), FabricError> {
        let id = self.fabric.reserve_object();
        let record = self
            .lane
            .demote_blocking(id, STAGED_FABRIC_VERSION, bytes.to_vec())
            .map_err(FabricError::from)?;
        self.fabric
            .import_offcore(kivi_memory::OffcoreImport {
                id,
                version: STAGED_FABRIC_VERSION,
                offset: record.offset,
                len: record.len,
                checksum: record.checksum,
                intent,
                class: BehaviorClass::ColdCandidate,
            })
            .map_err(FabricError::from)?;
        Ok((id, len))
    }

    /// Re-homes a value recovered from the WAL under the id its durable
    /// record published it as.
    ///
    /// A replayed root names a fixed fabric id, so recovery cannot let
    /// [`Self::stage`] mint a fresh one: the bytes have to land on the id the
    /// log says they were published under or the root resolves to nothing.
    ///
    /// Residence mirrors the pressure the write path would have faced. The
    /// value goes into the arena when there is room, and to the off-core
    /// device when there is not — recovery must not be the moment a bounded
    /// fabric discovers it cannot hold a working set it already accepted, so
    /// the arena is a preference and the device is the fallback. Either way
    /// the first read promotes back on demand.
    ///
    /// # Errors
    ///
    /// Returns [`FabricError`] when neither the arena nor the off-core
    /// device can take the bytes. An id that is already resident is success:
    /// a value sealed twice under one id (transaction prepare then
    /// finalize) is normal, not corruption.
    pub fn import_recovered(
        &mut self,
        tablet: TabletId,
        fabric_id: u64,
        version: u64,
        bytes: &Bytes,
    ) -> Result<(), FabricError> {
        let mut intent = kivi_memory::MaterializationIntent::cold();
        intent.tablet = Some(tablet);
        if let Some(node) = self.numa_node {
            intent.locality = kivi_memory::LocalityRequirement::PreferredNuma { node };
        }
        match self.fabric.insert_at(
            fabric_id,
            bytes.clone(),
            version,
            intent.clone(),
            BehaviorClass::HotMutable,
            Mutability::Mutable,
        ) {
            // A second seal of one id (transaction prepare then finalize) is
            // expected, not corruption: the first import is already resident.
            Ok(()) | Err(kivi_memory::MemoryError::Invalid { .. }) => return Ok(()),
            Err(kivi_memory::MemoryError::Overloaded { .. }) => {}
            Err(error) => return Err(FabricError::from(error)),
        }
        // Arena is full. The bytes are already in hand, so park them on the
        // off-core device and register the locator: same durable truth, a
        // residence the bounded arena does not have to hold.
        let demoted = self
            .lane
            .demote_blocking(fabric_id, version, bytes.to_vec())
            .map_err(FabricError::from)?;
        self.fabric
            .import_offcore(kivi_memory::OffcoreImport {
                id: fabric_id,
                version,
                offset: demoted.offset,
                len: demoted.len,
                checksum: demoted.checksum,
                intent,
                class: BehaviorClass::ColdCandidate,
            })
            .map_err(FabricError::from)
    }

    /// Resolves a fabric reference synchronously when a valid arena
    /// residence exists AND still names the pinned version. Returns
    /// `None` when the caller must suspend for promotion, and
    /// `Err(Unavailable)` when the root moved (caller re-prepares).
    ///
    /// # Errors
    ///
    /// Returns [`FabricError::Unavailable`] when the root moved or the
    /// object is unknown, and [`FabricError::Corrupt`] on length mismatch.
    pub fn resolve_sync(
        &mut self,
        reference: &FabricRef,
        current: Option<&kivi_state::StoredObject>,
    ) -> Result<Option<Bytes>, FabricError> {
        if pinned_root_version(current, reference) != Some(reference.version) {
            return Err(FabricError::Unavailable);
        }
        match self.fabric.try_resolve_sync(reference.id) {
            Ok(bytes) => {
                if bytes.len() as u64 != reference.logical_len {
                    return Err(FabricError::Corrupt);
                }
                Ok(Some(bytes))
            }
            Err(MemoryError::ProviderFailed { .. }) => Ok(None),
            Err(error) => Err(FabricError::from(error)),
        }
    }

    /// Submits a lane promotion for an off-core residence and parks the
    /// wait. The caller polls (bridge) or blocks (channel worker);
    /// connection tasks compose [`begin_promote`](Self::begin_promote)
    /// with an await plus [`finish_promote`](Self::finish_promote) and
    /// hold no park.
    ///
    /// # Errors
    ///
    /// Returns `Overloaded` when the lane queue is saturated and
    /// `Unavailable` when no off-core residence exists or the version
    /// already moved.
    pub fn submit_parked_promote(
        &mut self,
        tablet: TabletId,
        key: Key,
        reference: &FabricRef,
        current: Option<&kivi_state::StoredObject>,
    ) -> Result<u64, FabricError> {
        if pinned_root_version(current, reference) != Some(reference.version) {
            return Err(FabricError::Unavailable);
        }
        let Some((offset, len, version)) = self.fabric.offcore_locator(reference.id) else {
            return Err(FabricError::Unavailable);
        };
        let reply = self
            .lane
            .submit_promote(reference.id, version, offset, len)
            .map_err(FabricError::from)?;
        self.fabric.pin(reference.id);
        let park = self.next_park;
        self.next_park += 1;
        self.parked.insert(
            park,
            ParkedOp {
                tablet,
                key,
                version: reference.version,
                fabric_id: reference.id,
                reply,
            },
        );
        Ok(park)
    }

    /// Direct lane promotion split into two synchronous phases so async
    /// callers never hold a fabric borrow across an await: `begin`
    /// submits, the caller awaits the reply, then `finish` fences and
    /// installs. `promote_direct` composes both for blocking callers.
    ///
    /// # Errors
    ///
    /// Returns `Overloaded` on lane saturation and `Unavailable` when the
    /// reference moved or no residence exists.
    pub fn begin_promote(
        &mut self,
        reference: &FabricRef,
        current_version: u64,
    ) -> Result<OffcoreReply<Result<PromotedBytes, MemoryError>>, FabricError> {
        if reference.version != current_version {
            return Err(FabricError::Unavailable);
        }
        // Fast path first: a synchronous residence needs no lane at all.
        // (The caller checks this with try_resolve_sync before begin;
        // begin assumes the miss.)
        let Some((offset, len, version)) = self.fabric.offcore_locator(reference.id) else {
            return Err(FabricError::Unavailable);
        };
        self.lane
            .submit_promote(reference.id, version, offset, len)
            .map_err(FabricError::from)
    }

    /// Finishes a direct promotion: version-fences the reply against the
    /// current root and installs the bytes as a synchronous shadow.
    ///
    /// # Errors
    ///
    /// Returns `Unavailable` when the root moved under the promotion
    /// (stale completion, counted) and `Corrupt` on length mismatch.
    pub fn finish_promote(
        &mut self,
        reference: &FabricRef,
        current: Option<&kivi_state::StoredObject>,
        outcome: Result<PromotedBytes, MemoryError>,
    ) -> Result<Bytes, FabricError> {
        let promoted = outcome.map_err(FabricError::from)?;
        let live = current.and_then(kivi_state::StoredObject::fabric_ref);
        let Some(live) = live else {
            self.stale_completions += 1;
            return Err(FabricError::Unavailable);
        };
        if live != *reference
            || current.map(|object| object.version().as_u64()) != Some(reference.version)
        {
            self.stale_completions += 1;
            return Err(FabricError::Unavailable);
        }
        if promoted.bytes.len() as u64 != live.logical_len {
            return Err(FabricError::Corrupt);
        }
        self.fabric.install_promote_bytes(
            reference.id,
            promoted.version,
            promoted.bytes.clone(),
        )?;
        Ok(Bytes::from(promoted.bytes))
    }

    /// Direct lane promotion for blocking callers (plain worker threads,
    /// tests): submit, block, fence, install. Never used on the reactor.
    ///
    /// # Errors
    ///
    /// Returns `Overloaded` on lane saturation, `Unavailable` when the
    /// reference moved or no residence exists, and `Corrupt` on length
    /// or verification mismatch.
    pub fn promote_blocking(
        &mut self,
        reference: &FabricRef,
        current: Option<&kivi_state::StoredObject>,
    ) -> Result<Bytes, FabricError> {
        if pinned_root_version(current, reference) != Some(reference.version) {
            return Err(FabricError::Unavailable);
        }
        if let Ok(bytes) = self.fabric.try_resolve_sync(reference.id) {
            if bytes.len() as u64 != reference.logical_len {
                return Err(FabricError::Corrupt);
            }
            return Ok(bytes);
        }
        let reply = self.begin_promote(reference, reference.version)?;
        let outcome = reply.recv_blocking().map_err(|_| FabricError::Overloaded)?;
        self.finish_promote(reference, current, outcome)
    }

    /// Polls one parked wait (bridge turns). `Ok(None)` means still
    /// pending; `Ok(Some(bytes))` delivers verified bytes after
    /// re-fencing against the current root; `Err` fails closed and
    /// unparks.
    ///
    /// # Errors
    ///
    /// Returns [`FabricError::Unavailable`] when the park id is unknown or
    /// the root moved under the promotion, [`FabricError::Overloaded`]
    /// when the lane closed, and [`FabricError::Corrupt`] on length
    /// mismatch.
    ///
    /// # Panics
    ///
    /// Panics if the park entry is missing after the existence check above
    /// (unreachable while holding `&mut self` without re-entrancy).
    pub fn poll_parked(
        &mut self,
        park: u64,
        current: Option<&kivi_state::StoredObject>,
    ) -> Result<Option<Bytes>, FabricError> {
        let Some(wait) = self.parked.get(&park) else {
            return Err(FabricError::Unavailable);
        };
        let outcome = match wait.reply.try_recv() {
            Err(async_channel::TryRecvError::Empty) => return Ok(None),
            Err(async_channel::TryRecvError::Closed) => Err(FabricError::Overloaded),
            Ok(outcome) => outcome.map_err(FabricError::from),
        };
        let wait = self.parked.remove(&park).expect("park checked above");
        self.fabric.unpin(wait.fabric_id);
        self.fabric.release_admission();
        match outcome {
            Err(error) => Err(error),
            Ok(promoted) => {
                // Re-fence: the root must still name this id at this
                // version, or the bytes are stale history.
                let live = current
                    .and_then(kivi_state::StoredObject::fabric_ref)
                    .filter(|live| live.id == wait.fabric_id && live.version == wait.version);
                let Some(live) = live else {
                    self.stale_completions += 1;
                    return Err(FabricError::Unavailable);
                };
                if promoted.bytes.len() as u64 != live.logical_len {
                    return Err(FabricError::Corrupt);
                }
                // Install as a synchronous shadow so follow-up reads hit
                // without another promotion.
                self.fabric.install_promote_bytes(
                    wait.fabric_id,
                    promoted.version,
                    promoted.bytes.clone(),
                )?;
                self.parked_resumes += 1;
                Ok(Some(Bytes::from(promoted.bytes)))
            }
        }
    }

    /// Cancels all parked waits for a tablet (migration/split/merge
    /// fencing). Completions arriving later find no park and count stale.
    pub fn fence_tablet(&mut self, tablet: TabletId) {
        let parks: Vec<u64> = self
            .parked
            .iter()
            .filter(|(_, wait)| wait.tablet == tablet)
            .map(|(park, _)| *park)
            .collect();
        for park in parks {
            if let Some(wait) = self.parked.remove(&park) {
                self.fabric.unpin(wait.fabric_id);
            }
        }
    }

    /// Pins a fabric id for an outstanding reservation (prepared 2PC
    /// intent): while pinned the id is never retired and its last
    /// synchronous residence is never reclaimed, so a later finalize
    /// always finds the bytes. Pins nest by count; every `pin` needs
    /// one unpin (finalize-commit and finalize-abort both release).
    pub fn pin(&mut self, id: u64) {
        self.fabric.pin(id);
    }

    /// Retires a fabric id whose logical root is gone, deferring while
    /// pinned. Deferred ids sweep on unpin and maintenance.
    pub fn retire_or_defer(&mut self, id: u64) {
        if self.fabric.is_pinned(id) {
            self.retire_pending.insert(id);
            return;
        }
        if self.fabric.retire_object(id).is_err() {
            self.retire_pending.insert(id);
        }
    }

    /// Unpins an id and sweeps retirements it unblocks.
    pub fn unpin_and_sweep(&mut self, id: u64) {
        self.fabric.unpin(id);
        if !self.fabric.is_pinned(id) && self.retire_pending.remove(&id) {
            if self.fabric.retire_object(id).is_ok() {
            } else {
                self.retire_pending.insert(id);
            }
        }
    }

    /// Bounded maintenance, amortized: runs at most every 16 calls so
    /// hot wakes never pay planning on every request.
    pub fn maintenance_if_due(&mut self) {
        self.since_maintenance += 1;
        if self.since_maintenance >= 16 {
            self.since_maintenance = 0;
            self.maintenance();
        }
    }

    /// Bounded maintenance: external-executor drain → lane submission,
    /// lane completion collection with version-fenced installs, a fabric
    /// planning tick, deferred-retire sweep. No unbounded loops; planning
    /// scans at most the configured slice per call.
    pub fn maintenance(&mut self) {
        self.since_maintenance += 1;
        let queued = self.lane.stats_blocking().queued;
        self.fabric.set_external_in_flight(queued);
        self.fabric.tick();
        self.collect_maint_waits();
        if self.maint_waits.len() < MAX_MAINT_WAITS {
            for op in self.fabric.drain_external_ops() {
                if self.maint_waits.len() >= MAX_MAINT_WAITS {
                    break;
                }
                match op {
                    ExternalOp::Demote {
                        object,
                        version,
                        bytes,
                    } => {
                        if let Ok(reply) = self.lane.submit_demote(object, version, bytes) {
                            self.maint_waits.push_back(MaintWait {
                                object,
                                version,
                                demote: Some(reply),
                                promote: None,
                            });
                        } else {
                            self.fabric.release_admission();
                            self.fabric.cancel_transition_for(object);
                        }
                    }
                    ExternalOp::Promote {
                        object,
                        version,
                        offset,
                        len,
                    } => match self.lane.submit_promote(object, version, offset, len) {
                        Ok(reply) => self.maint_waits.push_back(MaintWait {
                            object,
                            version,
                            demote: None,
                            promote: Some(reply),
                        }),
                        Err(_) => {
                            self.fabric.cancel_transition_for(object);
                        }
                    },
                }
            }
        }
        let pending: Vec<u64> = self.retire_pending.iter().copied().collect();
        for id in pending {
            if !self.fabric.is_pinned(id) {
                self.retire_pending.remove(&id);
                if self.fabric.retire_object(id).is_ok() {
                } else {
                    self.retire_pending.insert(id);
                }
            }
        }
    }

    /// Collects ready maintenance lane replies, installing results with
    /// version fencing. Stale or failed completions count observability
    /// and never touch state.
    fn collect_maint_waits(&mut self) {
        // Poll each reply exactly once, in reverse order so removals
        // never shift pending indexes: `try_recv` consumes a ready
        // outcome, so a readiness probe followed by a second poll would
        // mistake every completion for a failure and drop it as stale.
        for index in (0..self.maint_waits.len()).rev() {
            let Some(wait) = self.maint_waits.get(index) else {
                continue;
            };
            if wait.demote.is_none() && wait.promote.is_none() {
                // Defensive: a wait with no reply can never complete.
                self.maint_waits.remove(index);
                continue;
            }
            if wait.demote.is_some() {
                let outcome = self.maint_waits[index]
                    .demote
                    .as_ref()
                    .map(OffcoreReply::try_recv);
                match outcome {
                    Some(Err(async_channel::TryRecvError::Empty)) | None => {}
                    Some(Err(_)) => {
                        self.maint_waits.remove(index);
                        self.stale_completions += 1;
                        self.fabric.release_admission();
                    }
                    Some(Ok(outcome)) => {
                        let Some(wait) = self.maint_waits.remove(index) else {
                            continue;
                        };
                        if let Ok(record) = outcome {
                            if self
                                .fabric
                                .install_demote_record(
                                    wait.object,
                                    wait.version,
                                    record.offset,
                                    record.len,
                                    record.checksum,
                                )
                                .is_err()
                            {
                                self.stale_completions += 1;
                            }
                        } else {
                            self.stale_completions += 1;
                        }
                        self.fabric.release_admission();
                    }
                }
            } else {
                let outcome = self.maint_waits[index]
                    .promote
                    .as_ref()
                    .map(OffcoreReply::try_recv);
                match outcome {
                    Some(Err(async_channel::TryRecvError::Empty)) | None => {}
                    Some(Err(_)) => {
                        self.maint_waits.remove(index);
                        self.stale_completions += 1;
                    }
                    Some(Ok(outcome)) => {
                        let Some(wait) = self.maint_waits.remove(index) else {
                            continue;
                        };
                        if let Ok(promoted) = outcome {
                            if self
                                .fabric
                                .install_promote_bytes(wait.object, wait.version, promoted.bytes)
                                .is_err()
                            {
                                self.stale_completions += 1;
                            }
                        } else {
                            self.stale_completions += 1;
                        }
                    }
                }
            }
        }
    }

    /// Reads bytes from any residence without promoting (checkpoint
    /// streaming, migration export). Never moves placement.
    ///
    /// # Errors
    ///
    /// Returns [`FabricError::Unavailable`] when `id` is unknown or has no
    /// residence, and [`FabricError::Corrupt`] when no valid residence
    /// remains.
    pub fn read_anywhere(&mut self, id: u64) -> Result<Bytes, FabricError> {
        self.fabric.read_anywhere(id).map_err(FabricError::from)
    }

    /// Synchronous arena-only resolve passthrough (reactor-safe: never
    /// touches the backend).
    ///
    /// # Errors
    ///
    /// Returns [`FabricError::Unavailable`] when the object is unknown or
    /// has no synchronous arena residence (the caller must promote first).
    pub fn try_resolve_sync(&self, object: u64) -> Result<Bytes, FabricError> {
        self.fabric
            .try_resolve_sync(object)
            .map_err(FabricError::from)
    }

    /// Off-core locator passthrough (pure lookup, no I/O).
    #[must_use]
    pub fn offcore_locator(&self, object: u64) -> Option<(u64, u64, u64)> {
        self.fabric.offcore_locator(object)
    }

    /// Retires a superseded fabric id after a successful write: when the
    /// pre-image root named `previous` and the post-image no longer does,
    /// the old materialization is retired (or deferred while pinned).
    /// Call with the pre-apply root and the post-apply object.
    pub fn retire_superseded(
        &mut self,
        previous: Option<FabricRef>,
        post: &kivi_state::StoredObject,
    ) {
        let Some(previous) = previous else {
            return;
        };
        let still_live = post.fabric_ref().is_some_and(|live| live.id == previous.id);
        if !still_live {
            self.retire_or_defer(previous.id);
        }
    }

    /// Observability snapshot.
    #[must_use]
    pub fn stats_snapshot(&self) -> FabricStatsSnapshot {
        let stats = self.fabric.stats();
        FabricStatsSnapshot {
            objects: stats.objects,
            inline_bytes: stats.inline_bytes,
            dram_bytes: stats.dram_bytes,
            compressed_bytes: stats.compressed_bytes,
            nvme_bytes: stats.nvme_bytes,
            hits: stats.hits,
            promotion_misses: stats.promotion_misses,
            demotions: stats.demotions,
            promotions: stats.promotions,
            migration_bytes: stats.migration_bytes,
            compression_ratio_bps: stats.compression_ratio_bps,
            cancelled: stats.cancelled,
            compactions: stats.compactions,
            pinned: self.fabric.pinned_count(),
            parked: self.parked.len(),
            retire_pending: self.retire_pending.len(),
            stale_completions: self.stale_completions,
            parked_resumes: self.parked_resumes,
            admission_rejects: self.admission_rejects,
        }
    }

    /// Direct access for commit application (retire-old hooks) and
    /// recovery (imports). All mutation goes through fenced helpers.
    pub fn fabric_mut(&mut self) -> &mut MemoryFabric {
        &mut self.fabric
    }

    /// Applies a validated bounded memory control policy on the worker owner.
    pub fn set_control_policy(&mut self, policy: kivi_memory::MemoryControlPolicy) {
        self.fabric.set_control_policy(policy);
    }

    /// Returns the worker's current bounded memory control policy.
    #[must_use]
    pub fn control_policy(&self) -> &kivi_memory::MemoryControlPolicy {
        self.fabric.control_policy()
    }

    /// Drains worker-local access telemetry for the observation fabric.
    pub fn drain_access_signals(&mut self) -> Vec<(u64, kivi_memory::AccessSignals)> {
        self.fabric.drain_access_signals()
    }

    /// Lane handle for async submission sites.
    #[must_use]
    pub const fn lane(&self) -> &OffcoreLaneHandle {
        &self.lane
    }
}

/// Flat observability snapshot: everything an operator needs to answer
/// why an object is in `NVMe` and why memory cannot be reclaimed.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct FabricStatsSnapshot {
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
    /// Compression ratio in basis points.
    pub compression_ratio_bps: u64,
    /// Cancelled transitions.
    pub cancelled: u64,
    /// Arena compactions completed.
    pub compactions: u64,
    /// Currently pinned objects (reclamation blockers).
    pub pinned: usize,
    /// Parked promotion waits.
    pub parked: usize,
    /// Deferred retirements (pinned at retire time).
    pub retire_pending: usize,
    /// Stale completions discarded.
    pub stale_completions: u64,
    /// Parked waits resumed with bytes.
    pub parked_resumes: u64,
    /// Staging admission rejections (backpressure events).
    pub admission_rejects: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The inline floor is a policy choice, not a mirror of the arena's
    /// tiny-value path: it decides whether a value is worth an arena slot and
    /// an off-core locator. Pinning it to the arena's internal tiny constant
    /// would make that policy a side effect of a storage detail.
    #[test]
    fn inline_floor_is_the_policy_value_not_the_arena_tiny_path() {
        const { assert!(FABRIC_INLINE_MAX > kivi_memory::TINY_INLINE_MAX) };
        assert_eq!(FABRIC_INLINE_MAX, 1024);
    }

    #[test]
    fn stage_and_resolve_roundtrip() {
        let dir = tempfile::tempdir().expect("tempdir");
        let paths = FabricPaths {
            root: dir.path().to_owned(),
        };
        let worker = WorkerId::from_u64(0);
        let (mut fabric, guard) =
            TabletFabric::open_unbound(worker, &paths, 1 << 20, 16 << 20).expect("fabric opens");
        let tablet = TabletId::from_u64(1);
        let bytes = Bytes::from(vec![0xABu8; 1024]);
        let (staging_id, staged_len) = fabric.stage(tablet, &bytes, true).expect("stage");
        let reference = FabricRef {
            id: staging_id,
            logical_len: staged_len,
            version: 1,
        };
        // Fabric-internal versions start at 1; the authoritative version
        // here coincides because this is the first write.
        let pinned = kivi_state::StoredObject::restore(
            kivi_state::LogicalValue::Fabric(reference),
            kivi_state::ObjectVersion::FIRST,
            kivi_types::Expiry::NEVER,
        );
        let resolved = fabric
            .resolve_sync(&reference, Some(&pinned))
            .expect("resolves")
            .expect("synchronous");
        assert_eq!(resolved, bytes);
        guard.shutdown();
    }

    /// Drives maintenance until `id` gains an off-core residence,
    /// bounded (the lane completes in microseconds; failure means the
    /// planner never demotes, which is itself the bug).
    fn await_offcore(fabric: &mut TabletFabric, id: u64) {
        for _ in 0..2000 {
            fabric.maintenance();
            if fabric.offcore_locator(id).is_some() {
                return;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        panic!("off-core residence never materialized for {id}");
    }

    fn pinned_object(version: u64, reference: FabricRef) -> kivi_state::StoredObject {
        kivi_state::StoredObject::restore(
            kivi_state::LogicalValue::Fabric(reference),
            kivi_state::ObjectVersion::from_u64(version),
            kivi_types::Expiry::NEVER,
        )
    }

    #[test]
    fn finish_promote_uses_physical_version_and_propagates_install_errors() {
        let dir = tempfile::tempdir().expect("tempdir");
        let paths = FabricPaths {
            root: dir.path().to_owned(),
        };
        let (mut fabric, guard) =
            TabletFabric::open_unbound(WorkerId::from_u64(0), &paths, 1 << 20, 16 << 20)
                .expect("fabric opens");
        let tablet = TabletId::from_u64(1);
        let bytes = Bytes::from(vec![0xABu8; 1024]);
        let (staging_id, staged_len) = fabric.stage(tablet, &bytes, true).expect("stage");
        let physical_version = fabric
            .fabric_mut()
            .object_version(staging_id)
            .expect("physical version");
        let reference = FabricRef {
            id: staging_id,
            logical_len: staged_len,
            version: physical_version + 7,
        };
        let current = pinned_object(reference.version, reference);
        let promotions_before = fabric.stats_snapshot().promotions;
        let resolved = fabric
            .finish_promote(
                &reference,
                Some(&current),
                Ok(PromotedBytes {
                    object: staging_id,
                    version: physical_version,
                    bytes: bytes.to_vec(),
                }),
            )
            .expect("finish");
        assert_eq!(resolved, bytes);
        assert_eq!(fabric.stats_snapshot().promotions, promotions_before + 1);
        let error = fabric
            .finish_promote(
                &reference,
                Some(&current),
                Ok(PromotedBytes {
                    object: staging_id,
                    version: physical_version + 1,
                    bytes: bytes.to_vec(),
                }),
            )
            .expect_err("installation error propagates");
        assert_eq!(error, FabricError::Unavailable);
        guard.shutdown();
    }

    #[test]
    fn poll_parked_uses_physical_version() {
        let dir = tempfile::tempdir().expect("tempdir");
        let paths = FabricPaths {
            root: dir.path().to_owned(),
        };
        let (mut fabric, guard) =
            TabletFabric::open_unbound(WorkerId::from_u64(0), &paths, 1 << 12, 16 << 20)
                .expect("fabric opens");
        let tablet = TabletId::from_u64(1);
        let bytes = Bytes::from(vec![0xCDu8; 1024]);
        let (staging_id, staged_len) = fabric.stage(tablet, &bytes, false).expect("stage");
        await_offcore(&mut fabric, staging_id);
        let physical_version = fabric
            .fabric_mut()
            .object_version(staging_id)
            .expect("physical version");
        let reference = FabricRef {
            id: staging_id,
            logical_len: staged_len,
            version: physical_version + 7,
        };
        let current = pinned_object(reference.version, reference);
        let promotions_before = fabric.stats_snapshot().promotions;
        let park = fabric
            .submit_parked_promote(tablet, Key::from("k"), &reference, Some(&current))
            .expect("parks");
        let mut resolved = None;
        for _ in 0..2000 {
            match fabric.poll_parked(park, Some(&current)) {
                Ok(None) => std::thread::sleep(std::time::Duration::from_millis(5)),
                Ok(Some(bytes)) => {
                    resolved = Some(bytes);
                    break;
                }
                Err(error) => panic!("park promotion failed: {error:?}"),
            }
        }
        assert_eq!(resolved.expect("park completes"), bytes);
        assert_eq!(fabric.stats_snapshot().promotions, promotions_before + 1);
        guard.shutdown();
    }

    #[test]
    fn fence_tablet_cancels_parks_and_fails_closed() {
        let dir = tempfile::tempdir().expect("tempdir");
        let paths = FabricPaths {
            root: dir.path().to_owned(),
        };
        let worker = WorkerId::from_u64(0);
        // Tiny arena so one cold object demotes promptly.
        let (mut fabric, guard) =
            TabletFabric::open_unbound(worker, &paths, 1 << 12, 16 << 20).expect("fabric opens");
        let tablet = TabletId::from_u64(1);
        let bytes = Bytes::from(vec![0xCDu8; 1024]);
        let (staging_id, staged_len) = fabric.stage(tablet, &bytes, false).expect("stage");
        await_offcore(&mut fabric, staging_id);
        let reference = FabricRef {
            id: staging_id,
            logical_len: staged_len,
            version: 1,
        };
        let pinned = pinned_object(1, reference);
        // An off-core residence exists now (the arena shadow may still
        // serve synchronously; parking exercises the lane path anyway).
        let park = fabric
            .submit_parked_promote(tablet, Key::from("k"), &reference, Some(&pinned))
            .expect("parks");
        // Authority change fences the tablet: the park vanishes and the
        // late lane reply finds nothing (disconnect-tolerant, no panic).
        fabric.fence_tablet(tablet);
        assert!(matches!(
            fabric.poll_parked(park, Some(&pinned)),
            Err(FabricError::Unavailable)
        ));
        // Stale root (moved/deleted while promoted) counts and fails
        // closed instead of serving history.
        let park = fabric
            .submit_parked_promote(tablet, Key::from("k"), &reference, Some(&pinned))
            .expect("re-parks");
        let stale_before = fabric.stats_snapshot().stale_completions;
        // Wait for the lane reply, then answer against a gone root.
        for _ in 0..2000 {
            match fabric.poll_parked(park, None) {
                Ok(None) => std::thread::sleep(std::time::Duration::from_millis(5)),
                Err(FabricError::Unavailable) => break,
                outcome => panic!("unexpected park outcome: {outcome:?}"),
            }
        }
        assert_eq!(
            fabric.stats_snapshot().stale_completions,
            stale_before + 1,
            "stale completion counted"
        );
        guard.shutdown();
    }
}
