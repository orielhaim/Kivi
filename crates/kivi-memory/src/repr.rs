//! Physical representations and the reconstruction safety invariant.
//!
//! Logical state (bytes + version) is exact; representations are
//! reconstructible physical forms. [`ObjectMaterialization`] tracks one
//! primary residence plus zero or more shadow residences (Nomad-style
//! non-exclusive tiering). The invariant, enforced by
//! [`ObjectMaterialization::can_reclaim`], is:
//!
//! > Before reclaiming a representation, prove another valid
//! > representation or reconstruction source exists.
//!
//! Large values reuse the existing immutable chunk architecture: the
//! [`Residence::Chunked`] variant names a chunk reference owned by
//! `kivi-chunk`; this crate never stores large bulk bytes itself.

use bytes::Bytes;

use crate::error::MemoryError;
use crate::handle::{BehaviorClass, ObjectHandle};
use crate::provider::ProviderId;

/// Where one copy of an object's bytes physically lives.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Residence {
    /// Tiny value inline in the object root (no handle, no provider).
    Inline(Bytes),
    /// Medium value in a worker-local arena.
    Arena {
        /// Which behavior class arena owns the slot.
        class: BehaviorClass,
        /// Generational handle naming the slot.
        handle: ObjectHandle,
        /// Provider the arena memory is accounted to (usually DRAM).
        provider: ProviderId,
    },
    /// Compressed DRAM representation (explicit, not swap).
    Compressed {
        /// Arena slot holding the compressed block.
        class: BehaviorClass,
        /// Handle of the compressed block.
        handle: ObjectHandle,
        /// Provider accounting (compressed-DRAM provider).
        provider: ProviderId,
        /// Logical (uncompressed) length for pre-allocation guards.
        logical_len: u32,
        /// Stored (compressed) length.
        stored_len: u32,
    },
    /// NVMe-resident copy written by an [`OffcoreOp`](crate::offcore::OffcoreOp).
    Offcore {
        /// Provider that owns the record.
        provider: ProviderId,
        /// Offset of the record in the provider's file.
        offset: u64,
        /// Stored length.
        len: u64,
        /// Content checksum (CRC32C of the stored bytes).
        checksum: u32,
    },
    /// Large value addressed by an immutable chunk manifest owned by
    /// `kivi-chunk`. The fabric never moves these bytes; it only keeps
    /// the reference alive across transitions.
    Chunked {
        /// Opaque manifest identifier bytes (40-byte `ChunkedRef`
        /// encoding owned by `kivi-state`; stored opaquely here).
        manifest: [u8; 32],
        /// Logical length.
        logical_len: u64,
    },
}

impl Residence {
    /// Short diagnostic name for `EXPLAIN`-style output.
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::Inline(_) => "inline",
            Self::Arena { .. } => "arena",
            Self::Compressed { .. } => "compressed",
            Self::Offcore { .. } => "nvme",
            Self::Chunked { .. } => "chunked",
        }
    }

    /// Whether this residence is directly readable without I/O.
    #[must_use]
    pub const fn is_synchronous(&self) -> bool {
        matches!(
            self,
            Self::Inline(_) | Self::Arena { .. } | Self::Chunked { .. }
        )
    }
}

/// A non-fabric source that can reconstruct the bytes: the authoritative
/// store, a chunk manifest, a checkpoint band, or a peer replica.
///
/// Eviction (Midas-style `unmap-and-reconstruct`) is legal only when at
/// least one of these exists alongside the remaining representations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReconstructionSource {
    /// The authoritative object store still holds the logical value.
    AuthoritativeStore,
    /// An immutable chunk manifest addresses the bytes.
    ChunkManifest,
    /// A sealed checkpoint band contains the bytes.
    CheckpointBand,
    /// A peer replica can serve the bytes.
    PeerReplica,
}

/// One object's materialization: primary plus shadows, version-fenced.
///
/// The `version` is the logical version the cached bytes correspond to.
/// Any mutation bumps `version` and invalidates in-flight transition
/// builds (which record the version they started from).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ObjectMaterialization {
    /// Internal object index (fabric-scoped).
    pub object: u64,
    /// Logical version of the cached bytes.
    pub version: u64,
    /// Primary residence serving synchronous reads.
    pub primary: Residence,
    /// Shadow copies kept during non-exclusive transitions or as
    /// insurance before reclamation.
    pub shadows: Vec<Residence>,
    /// Reconstruction sources independent of the tracked residences.
    pub reconstruction: Vec<ReconstructionSource>,
}

impl ObjectMaterialization {
    /// Creates a materialization with a single primary residence.
    #[must_use]
    pub const fn new(object: u64, version: u64, primary: Residence) -> Self {
        Self {
            object,
            version,
            primary,
            shadows: Vec::new(),
            reconstruction: Vec::new(),
        }
    }

    /// All residences: primary first, then shadows.
    #[must_use]
    pub fn all_residences(&self) -> Vec<&Residence> {
        let mut out = Vec::with_capacity(1 + self.shadows.len());
        out.push(&self.primary);
        out.extend(&self.shadows);
        out
    }

    /// Whether reclaiming (dropping) `index` (0 = primary, 1+ = shadows)
    /// is safe: at least one *other* valid representation or a
    /// reconstruction source must remain.
    ///
    /// # Errors
    ///
    /// Returns [`MemoryError::NoRepresentation`] when the index is out of
    /// bounds, and [`MemoryError::Invalid`] when reclamation would leave
    /// the object with no valid representation and no reconstruction
    /// source.
    pub fn can_reclaim(&self, index: usize) -> Result<(), MemoryError> {
        let total = 1 + self.shadows.len();
        if index >= total {
            return Err(MemoryError::NoRepresentation {
                object: self.object,
            });
        }
        if total >= 2 || !self.reconstruction.is_empty() {
            return Ok(());
        }
        Err(MemoryError::Invalid {
            detail: "reclaiming the last representation without a reconstruction source".to_owned(),
        })
    }

    /// Adds a shadow copy (built and verified, not yet published).
    pub fn add_shadow(&mut self, residence: Residence) {
        self.shadows.push(residence);
    }

    /// Publishes shadow `index` as the new primary, demoting the old
    /// primary to a shadow. The old primary is NOT destroyed: the caller
    /// retires it after a hysteresis window or keeps it as insurance.
    ///
    /// # Errors
    ///
    /// Returns [`MemoryError::NoRepresentation`] for an out-of-bounds
    /// shadow index.
    pub fn publish_shadow(&mut self, index: usize) -> Result<Residence, MemoryError> {
        if index >= self.shadows.len() {
            return Err(MemoryError::NoRepresentation {
                object: self.object,
            });
        }
        let next = self.shadows.remove(index);
        let previous = std::mem::replace(&mut self.primary, next);
        self.shadows.push(previous);
        Ok(self.primary.clone())
    }

    /// Retires (drops) residence `index` after proving safety with
    /// [`can_reclaim`](Self::can_reclaim).
    ///
    /// # Errors
    ///
    /// Propagates the safety check failure without mutating anything.
    pub fn retire(&mut self, index: usize) -> Result<Residence, MemoryError> {
        self.can_reclaim(index)?;
        if index == 0 {
            // Retiring the primary promotes the first shadow, or fails
            // when no shadow exists (can_reclaim already proved a
            // reconstruction source exists; the caller evicts fully).
            if self.shadows.is_empty() {
                return Err(MemoryError::Invalid {
                    detail: "cannot retire a lone primary without a shadow".to_owned(),
                });
            }
            let next = self.shadows.remove(0);
            Ok(std::mem::replace(&mut self.primary, next))
        } else {
            Ok(self.shadows.remove(index - 1))
        }
    }

    /// Fully evicts the object from the fabric (all residences dropped).
    /// Legal only with a reconstruction source present.
    ///
    /// # Errors
    ///
    /// Returns [`MemoryError::Invalid`] without a reconstruction source.
    pub fn evict_all(&mut self) -> Result<Vec<Residence>, MemoryError> {
        if self.reconstruction.is_empty() {
            return Err(MemoryError::Invalid {
                detail: "evicting without a reconstruction source loses soft state".to_owned(),
            });
        }
        let mut out = Vec::with_capacity(1 + self.shadows.len());
        out.push(std::mem::replace(
            &mut self.primary,
            Residence::Inline(bytes::Bytes::new()),
        ));
        out.append(&mut self.shadows);
        Ok(out)
    }
}

/// Physical representation tag for diagnostics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PhysicalRepr {
    /// Tiny inline bytes.
    Inline,
    /// Arena-resident medium object.
    ArenaResident,
    /// Compressed DRAM block.
    CompressedMemory,
    /// NVMe-resident record.
    LocalStorage,
    /// Chunked large value (external fabric).
    Chunked,
    /// Future extension (CXL, remote, ...).
    Remote,
}

impl PhysicalRepr {
    /// Derives the tag from a residence.
    #[must_use]
    pub const fn of(residence: &Residence) -> Self {
        match residence {
            Residence::Inline(_) => Self::Inline,
            Residence::Arena { .. } => Self::ArenaResident,
            Residence::Compressed { .. } => Self::CompressedMemory,
            Residence::Offcore { .. } => Self::LocalStorage,
            Residence::Chunked { .. } => Self::Chunked,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bytes::Bytes;

    fn inline_object() -> ObjectMaterialization {
        ObjectMaterialization::new(1, 1, Residence::Inline(Bytes::from_static(b"x")))
    }

    #[test]
    fn lone_primary_cannot_be_reclaimed_without_source() {
        let materialization = inline_object();
        assert!(materialization.can_reclaim(0).is_err());
    }

    #[test]
    fn shadow_makes_reclamation_safe() {
        let mut materialization = inline_object();
        materialization.add_shadow(Residence::Inline(Bytes::from_static(b"x")));
        assert!(materialization.can_reclaim(0).is_ok());
        let retired = materialization.retire(1).expect("retire shadow");
        assert_eq!(retired.name(), "inline");
    }

    #[test]
    fn reconstruction_source_permits_eviction() {
        let mut materialization = inline_object();
        materialization
            .reconstruction
            .push(ReconstructionSource::AuthoritativeStore);
        assert!(materialization.can_reclaim(0).is_ok());
        let evicted = materialization.evict_all().expect("evict");
        assert_eq!(evicted.len(), 1);
    }

    #[test]
    fn publish_keeps_old_primary_as_shadow() {
        let mut materialization = inline_object();
        materialization.add_shadow(Residence::Chunked {
            manifest: [7u8; 32],
            logical_len: 1 << 20,
        });
        materialization.publish_shadow(0).expect("publish");
        assert_eq!(materialization.primary.name(), "chunked");
        assert_eq!(materialization.shadows.len(), 1);
        assert_eq!(materialization.shadows[0].name(), "inline");
    }
}
