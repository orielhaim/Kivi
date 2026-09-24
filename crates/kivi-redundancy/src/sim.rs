//! Deterministic simulation hooks: fault injection for the fabric.
//!
//! The simulation layer drives these hooks with a seeded RNG (never wall
//! time, never global randomness): each [`FaultKind`] names one failure the
//! Phase 10 suite must cover — missing or corrupted fragments, failures
//! within and beyond tolerance, failures during encoding/reconstruction,
//! before publish, after publish before retirement, stale completions, drain
//! during repair, restart during transition, duplicate repair work, and
//! partial fragment writes. Hooks apply faults to an in-memory fragment view
//! so `kivi-sim` scenarios stay deterministic and replayable.

use std::collections::HashMap;

/// Every fault the deterministic suite injects.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FaultKind {
    /// One fragment goes missing.
    MissingFragment,
    /// One fragment's bytes are bit-flipped (checksum mismatch).
    CorruptedFragment,
    /// Failures within the advertised tolerance (recoverable).
    WithinTolerance,
    /// Failures beyond the advertised tolerance (fails closed).
    BeyondTolerance,
    /// Crash during encoding (pending build discarded).
    DuringEncoding,
    /// Crash during reconstruction (read retries cleanly).
    DuringReconstruction,
    /// Crash before publish (pending invisible, old layout serves).
    BeforePublish,
    /// Crash after publish before old-layout retirement (extra retained).
    AfterPublishBeforeRetire,
    /// Stale repair completion arrives after a newer generation published.
    StaleRepair,
    /// Node drain starts mid-repair.
    DrainDuringRepair,
    /// Restart lands mid-transition.
    RestartDuringTransition,
    /// Duplicate repair work is queued twice (idempotent coalesce).
    DuplicateRepair,
    /// Partial fragment write (torn file, detected as missing/corrupt).
    PartialWrite,
}

impl FaultKind {
    /// All faults in suite order.
    #[must_use]
    pub const fn all() -> [Self; 13] {
        [
            Self::MissingFragment,
            Self::CorruptedFragment,
            Self::WithinTolerance,
            Self::BeyondTolerance,
            Self::DuringEncoding,
            Self::DuringReconstruction,
            Self::BeforePublish,
            Self::AfterPublishBeforeRetire,
            Self::StaleRepair,
            Self::DrainDuringRepair,
            Self::RestartDuringTransition,
            Self::DuplicateRepair,
            Self::PartialWrite,
        ]
    }

    /// Human-readable name.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::MissingFragment => "missing-fragment",
            Self::CorruptedFragment => "corrupted-fragment",
            Self::WithinTolerance => "within-tolerance",
            Self::BeyondTolerance => "beyond-tolerance",
            Self::DuringEncoding => "during-encoding",
            Self::DuringReconstruction => "during-reconstruction",
            Self::BeforePublish => "before-publish",
            Self::AfterPublishBeforeRetire => "after-publish-before-retire",
            Self::StaleRepair => "stale-repair",
            Self::DrainDuringRepair => "drain-during-repair",
            Self::RestartDuringTransition => "restart-during-transition",
            Self::DuplicateRepair => "duplicate-repair",
            Self::PartialWrite => "partial-write",
        }
    }
}

/// In-memory fragment view a simulation mutates (index → bytes or tombstone).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SimFragments {
    /// Present fragments.
    pub present: HashMap<u32, Vec<u8>>,
}

impl SimFragments {
    /// Creates a view from encoded shards.
    #[must_use]
    pub fn from_shards(shards: Vec<Vec<u8>>) -> Self {
        Self {
            present: shards
                .into_iter()
                .enumerate()
                .map(|(i, s)| {
                    // Index < total <=40, fits `u32`.
                    #[allow(clippy::cast_possible_truncation)]
                    let index = i as u32;
                    (index, s)
                })
                .collect(),
        }
    }

    /// Applies one deterministic fault to the view.
    ///
    /// `seed` selects victims deterministically (no RNG needed): missing
    /// removes `seed % len`, corruption flips one byte there, partial write
    /// truncates there.
    pub fn apply(&mut self, fault: FaultKind, seed: u64) {
        if self.present.is_empty() {
            return;
        }
        let mut indexes: Vec<u32> = self.present.keys().copied().collect();
        indexes.sort_unstable();
        // Truncation is fine: `seed` only selects a deterministic victim.
        #[allow(clippy::cast_possible_truncation)]
        let slot = seed as usize % indexes.len();
        let victim = indexes[slot];
        match fault {
            FaultKind::MissingFragment | FaultKind::WithinTolerance | FaultKind::BeforePublish => {
                self.present.remove(&victim);
            }
            FaultKind::CorruptedFragment => {
                if let Some(bytes) = self.present.get_mut(&victim)
                    && !bytes.is_empty()
                {
                    bytes[0] ^= 0xFF;
                }
            }
            FaultKind::PartialWrite => {
                if let Some(bytes) = self.present.get_mut(&victim) {
                    bytes.truncate(bytes.len() / 2);
                }
            }
            FaultKind::BeyondTolerance => {
                self.present.clear();
            }
            FaultKind::DuplicateRepair
            | FaultKind::DuringEncoding
            | FaultKind::DuringReconstruction
            | FaultKind::AfterPublishBeforeRetire
            | FaultKind::StaleRepair
            | FaultKind::DrainDuringRepair
            | FaultKind::RestartDuringTransition => {}
        }
    }

    /// Present pairs sorted by index (decoder input).
    #[must_use]
    pub fn present_sorted(&self) -> Vec<(u32, Vec<u8>)> {
        let mut out: Vec<(u32, Vec<u8>)> =
            self.present.iter().map(|(i, b)| (*i, b.clone())).collect();
        out.sort_by_key(|(i, _)| *i);
        out
    }
}
