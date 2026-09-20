//! Compact generational handles and behavior classes.
//!
//! Medium objects live in worker-local moving arenas (OBASE insight: group
//! by behavior, not size alone). Handles — not raw pointers — name them,
//! so relocation, compaction, and tier movement only bump a generation
//! instead of chasing pointers. No stable raw pointer ever escapes the
//! materialization boundary: callers resolve a handle to bytes through the
//! arena or fabric, and stale handles fail loudly as
//! [`MemoryError::StaleHandle`](crate::error::MemoryError::StaleHandle).

/// Compact handle naming one medium object in its worker-local arena.
///
/// 8 bytes total: 32-bit slot index plus 32-bit generation. Copyable,
/// orderable, and safe to store in object roots, checkpoint bands, and
/// transition records.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ObjectHandle {
    /// Slot index within the owning arena class.
    pub slot: u32,
    /// Generation bumped on every reuse of the slot. A handle is valid
    /// only when its generation matches the slot's current generation.
    pub generation: u32,
}

impl ObjectHandle {
    /// Well-known invalid handle (slot and generation zero is never
    /// issued: generations start at 1).
    pub const INVALID: Self = Self {
        slot: u32::MAX,
        generation: 0,
    };

    /// Creates a handle. The arena is the only production caller.
    #[must_use]
    pub const fn new(slot: u32, generation: u32) -> Self {
        Self { slot, generation }
    }

    /// Whether this is the invalid sentinel.
    #[must_use]
    pub const fn is_invalid(self) -> bool {
        self.slot == u32::MAX && self.generation == 0
    }
}

/// Behavior class of a medium object (RFC §38).
///
/// Physical pages/extents are grouped by these classes so hot/cold,
/// mutable/read-mostly, lifetime, and criticality classes never
/// intermingle on the same pages (OBASE hotness-fragmentation fix).
/// The exact set is internal and adaptive: the planner assigns classes,
/// arenas enforce grouping, and future policies may split or merge
/// classes without changing handle semantics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BehaviorClass {
    /// Frequently mutated hot objects. Never co-located with cold data;
    /// never compressed (write churn would burn CPU).
    HotMutable,
    /// Frequently read, rarely written. Compression candidates when cold.
    HotReadMostly,
    /// Lukewarm: accessed but not critical. First demotion candidates.
    Warm,
    /// Expected to go cold. Grouped for cheap bulk demotion/compaction.
    ColdCandidate,
    /// Very short expected lifetime (transaction intents, staging).
    /// Segregated so death does not fragment long-lived arenas.
    ShortLived,
    /// Long-lived metadata (roots, manifests references, directory).
    LongLivedMetadata,
    /// Large-root metadata (chunk references for large values).
    LargeRootMetadata,
}

impl BehaviorClass {
    /// All classes in a fixed order for iteration and reporting.
    pub const ALL: [Self; 7] = [
        Self::HotMutable,
        Self::HotReadMostly,
        Self::Warm,
        Self::ColdCandidate,
        Self::ShortLived,
        Self::LongLivedMetadata,
        Self::LargeRootMetadata,
    ];

    /// Whether objects of this class are compression candidates.
    #[must_use]
    pub const fn compression_candidate(self) -> bool {
        matches!(
            self,
            Self::HotReadMostly | Self::Warm | Self::ColdCandidate | Self::LongLivedMetadata
        )
    }

    /// Whether objects of this class are expected to be mutated often.
    #[must_use]
    pub const fn mutable_hot(self) -> bool {
        matches!(self, Self::HotMutable | Self::ShortLived)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invalid_handle_is_not_issued() {
        assert!(ObjectHandle::INVALID.is_invalid());
        assert!(!ObjectHandle::new(0, 1).is_invalid());
    }

    #[test]
    fn hot_mutable_is_never_a_compression_candidate() {
        assert!(!BehaviorClass::HotMutable.compression_candidate());
        assert!(BehaviorClass::HotReadMostly.compression_candidate());
    }
}
