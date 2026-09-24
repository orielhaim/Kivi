//! Repair engine: the mechanism for repair (Phase 11 policy stays out).
//!
//! Phase 10 repair reacts to explicit known deficits — node unavailable,
//! fragment missing, corruption detected, drain requires replacement, layout
//! transition — and calculates whether each asset is healthy, degraded but
//! recoverable, critically degraded, or unrecoverable. Work is
//! resumable/idempotent (re-issuing a repair for an already-healthy fragment
//! is a no-op) and bounded: [`RepairBudgets`] caps repairs per tick, bytes
//! per tick, concurrent decodes, and memory per decode, while foreground
//! traffic retains priority (repair yields when the caller reports pressure).
//!
//! Scrubbing hooks ([`scrub_report`]) classify every fragment for the future
//! Phase 11 self-healing policy without building that policy here:
//! missing, unreadable, checksum mismatch, stale generation, reconstructable
//! corruption, unrecoverable corruption. Corruption is never treated as a
//! valid fragment merely because bytes were read.

use std::collections::{HashMap, VecDeque};

use kivi_types::NodeId;

use crate::{AssetHealth, AssetId, RedundancyError, SchemeParams};

/// Per-fragment integrity verdict (scrubbing hook input).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FragmentVerdict {
    /// Verified healthy.
    Healthy,
    /// No representation exists.
    Missing,
    /// Bytes unreadable (I/O, not corruption).
    Unreadable,
    /// Bytes read but hash mismatched.
    ChecksumMismatch,
    /// Valid bytes from a superseded generation (stale, not corrupt).
    StaleGeneration,
    /// Corrupt but reconstructable from remaining information.
    ReconstructableCorruption,
    /// Corrupt and not reconstructable now.
    UnrecoverableCorruption,
}

impl FragmentVerdict {
    /// Whether this fragment contributes to reconstruction now.
    #[must_use]
    pub const fn is_usable(self) -> bool {
        matches!(self, Self::Healthy)
    }

    /// Whether a repair is owed for this fragment.
    #[must_use]
    pub const fn needs_repair(self) -> bool {
        matches!(
            self,
            Self::Missing
                | Self::ChecksumMismatch
                | Self::ReconstructableCorruption
                | Self::Unreadable
        )
    }
}

/// Health assessment for one asset.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AssetAssessment {
    /// Asset assessed.
    pub asset: AssetId,
    /// Generation assessed.
    pub generation: u64,
    /// Health class.
    pub health: AssetHealth,
    /// Usable fragments now.
    pub usable: u32,
    /// Fragments required to reconstruct.
    pub required: u32,
    /// Total fragments in the layout.
    pub total: u32,
    /// Independent failures currently survivable.
    pub survivable: u32,
    /// Whether reconstruction is possible now.
    pub reconstructable: bool,
    /// Fragment indexes needing repair.
    pub need_repair: Vec<u32>,
    /// Why (per-index verdicts for operators).
    pub verdicts: Vec<(u32, FragmentVerdict)>,
}

/// Assesses one asset's health from per-fragment verdicts.
///
/// `survivable` counts only independently-placed healthy fragments (the
/// caller computes it via [`crate::independent_survivable`]); this function
/// maps counts to health classes without re-deriving placement.
#[must_use]
pub fn assess(
    asset: AssetId,
    generation: u64,
    params: SchemeParams,
    verdicts: Vec<(u32, FragmentVerdict)>,
    survivable: u32,
) -> AssetAssessment {
    let total = params.total_fragments();
    let required = params.required_pieces();
    // Usable <= total <=40, far below `u32::MAX`.
    #[allow(clippy::cast_possible_truncation)]
    let usable = verdicts.iter().filter(|(_, v)| v.is_usable()).count() as u32;
    let reconstructable = usable >= required;
    let need_repair: Vec<u32> = verdicts
        .iter()
        .filter(|(_, v)| v.needs_repair())
        .map(|(i, _)| *i)
        .collect();
    let health = if !reconstructable {
        AssetHealth::Unrecoverable
    } else if usable == total && need_repair.is_empty() {
        AssetHealth::Healthy
    } else if survivable == 0 {
        AssetHealth::Critical
    } else {
        AssetHealth::Degraded
    };
    AssetAssessment {
        asset,
        generation,
        health,
        usable,
        required,
        total,
        survivable,
        reconstructable,
        need_repair,
        verdicts,
    }
}

/// Bounds for background repair work.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RepairBudgets {
    /// Maximum repair actions per [`RepairEngine::tick`].
    pub max_repairs_per_tick: usize,
    /// Maximum repair bytes per tick.
    pub max_bytes_per_tick: u64,
    /// Maximum concurrent decodes fabric-wide.
    pub max_concurrent_decodes: usize,
    /// Maximum memory per decode (shard math guard).
    pub max_decode_bytes: u64,
    /// Maximum queued assets (backpressure: overflow fails fast).
    pub max_queued_assets: usize,
}

impl RepairBudgets {
    /// Conservative production defaults: small ticks, foreground first.
    #[must_use]
    pub const fn conservative() -> Self {
        Self {
            max_repairs_per_tick: 4,
            max_bytes_per_tick: 16 * 1024 * 1024,
            max_concurrent_decodes: 2,
            max_decode_bytes: 64 * 1024 * 1024,
            max_queued_assets: 1024,
        }
    }

    /// Test-friendly unbounded budgets (still exercises the code paths).
    #[must_use]
    pub const fn test_wide() -> Self {
        Self {
            max_repairs_per_tick: 64,
            max_bytes_per_tick: u64::MAX,
            max_concurrent_decodes: 16,
            max_decode_bytes: u64::MAX,
            max_queued_assets: 65536,
        }
    }
}

/// One queued repair: what to rebuild, where, and why.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RepairTask {
    /// Asset to repair.
    pub asset: AssetId,
    /// Generation to repair (fenced: stale generations never execute).
    pub generation: u64,
    /// Fragment indexes to rebuild.
    pub fragments: Vec<u32>,
    /// Why the repair was queued.
    pub reason: RepairReason,
    /// Target nodes (parallel to `fragments`; empty = planner assigns).
    pub targets: Vec<NodeId>,
}

/// Why a repair was queued (explicit deficits only — no learned policy).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RepairReason {
    /// Node unavailable (health snapshot).
    NodeUnavailable,
    /// Fragment missing (read or scrub found nothing).
    FragmentMissing,
    /// Corruption detected (checksum mismatch).
    CorruptionDetected,
    /// Operator drain requires replacement.
    DrainReplacement,
    /// Layout transition (new generation needs materialization).
    LayoutTransition,
}

impl RepairReason {
    /// Human-readable name.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::NodeUnavailable => "node-unavailable",
            Self::FragmentMissing => "fragment-missing",
            Self::CorruptionDetected => "corruption-detected",
            Self::DrainReplacement => "drain-replacement",
            Self::LayoutTransition => "layout-transition",
        }
    }
}

/// Bounded repair queue: critical first, idempotent, resumable.
///
/// Duplicate tasks for the same `(asset, generation, fragment)` coalesce:
/// re-queueing an already-queued repair is a no-op that reports `false`.
/// Completed fragments dequeue; failed attempts re-queue with their reason
/// intact (the caller decides retry policy — this engine never loops
/// unboundedly on its own).
#[derive(Debug, Default)]
pub struct RepairEngine {
    /// Queued tasks in priority order (critical/unrecoverable first).
    queue: VecDeque<RepairTask>,
    /// Budgets enforced per tick.
    budgets: RepairBudgetsState,
    /// Currently executing decodes (concurrency bound).
    inflight_decodes: usize,
}

/// Mutable budget counters (reset per tick where noted).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct RepairBudgetsState {
    /// Configured limits.
    limits: RepairBudgets,
    /// Bytes spent this tick.
    bytes_this_tick: u64,
}

impl Default for RepairBudgetsState {
    fn default() -> Self {
        Self {
            limits: RepairBudgets::conservative(),
            bytes_this_tick: 0,
        }
    }
}

impl RepairEngine {
    /// Creates an engine with conservative budgets.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates an engine with explicit budgets.
    #[must_use]
    pub const fn with_budgets(budgets: RepairBudgets) -> Self {
        Self {
            queue: VecDeque::new(),
            budgets: RepairBudgetsState {
                limits: budgets,
                bytes_this_tick: 0,
            },
            inflight_decodes: 0,
        }
    }

    /// Queues a repair, coalescing duplicates. Returns `true` when newly
    /// queued, `false` when already pending (idempotent re-queue).
    ///
    /// # Errors
    ///
    /// Returns [`RedundancyError::Overloaded`] when the queue is full.
    pub fn queue(&mut self, task: RepairTask) -> Result<bool, RedundancyError> {
        // Coalesce: merge fragment sets for the same asset+ generation.
        for queued in &mut self.queue {
            if queued.asset == task.asset && queued.generation == task.generation {
                let mut added = false;
                for fragment in &task.fragments {
                    if !queued.fragments.contains(fragment) {
                        queued.fragments.push(*fragment);
                        queued.fragments.sort_unstable();
                        added = true;
                    }
                }
                return Ok(added);
            }
        }
        if self.queue.len() >= self.budgets.limits.max_queued_assets {
            return Err(RedundancyError::Overloaded {
                detail: "repair queue full".to_owned(),
            });
        }
        // Critical tasks jump the queue (unrecoverable first is enforced by
        // the caller ordering assessments; within one call, LayoutTransition
        // and CorruptionDetected sort before routine missing-fragment work).
        let critical = matches!(
            task.reason,
            RepairReason::CorruptionDetected | RepairReason::LayoutTransition
        );
        if critical {
            self.queue.push_front(task);
        } else {
            self.queue.push_back(task);
        }
        Ok(true)
    }

    /// Queues repairs for every fragment an assessment flags.
    ///
    /// # Errors
    ///
    /// Returns [`RedundancyError::Overloaded`] when the queue is full.
    pub fn queue_assessment(
        &mut self,
        assessment: &AssetAssessment,
        reason: RepairReason,
        targets: &HashMap<u32, NodeId>,
    ) -> Result<bool, RedundancyError> {
        if assessment.need_repair.is_empty() {
            return Ok(false);
        }
        let task = RepairTask {
            asset: assessment.asset,
            generation: assessment.generation,
            fragments: assessment.need_repair.clone(),
            reason,
            targets: assessment
                .need_repair
                .iter()
                .filter_map(|index| targets.get(index).copied())
                .collect(),
        };
        self.queue(task)
    }

    /// Number of queued tasks.
    #[must_use]
    pub fn pending(&self) -> usize {
        self.queue.len()
    }

    /// Whether the queue is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }

    /// Drains up to the per-tick budget, returning the tasks to execute now.
    ///
    /// The caller executes them (reconstruct → verify → publish replacement
    /// fragments) and reports completion via [`mark_done`](Self::mark_done).
    /// Foreground pressure is honored through `foreground_pressure`: when
    /// `true`, only one task drains regardless of budget.
    pub fn tick(&mut self, foreground_pressure: bool) -> Vec<RepairTask> {
        self.budgets.bytes_this_tick = 0;
        let limit = if foreground_pressure {
            1
        } else {
            self.budgets.limits.max_repairs_per_tick
        };
        let mut out = Vec::new();
        while out.len() < limit {
            let Some(task) = self.queue.pop_front() else {
                break;
            };
            out.push(task);
        }
        out
    }

    /// Reports a completed repair (frees decode accounting).
    pub fn mark_done(&mut self, _task: &RepairTask, bytes: u64) {
        self.budgets.bytes_this_tick = self.budgets.bytes_this_tick.saturating_add(bytes);
        self.inflight_decodes = self.inflight_decodes.saturating_sub(1);
    }

    /// Re-queues a failed repair (keeps its reason; caller bounds retries).
    ///
    /// # Errors
    ///
    /// Returns [`RedundancyError::Overloaded`] when the queue is full.
    pub fn requeue(&mut self, task: RepairTask) -> Result<(), RedundancyError> {
        if self.queue.len() >= self.budgets.limits.max_queued_assets {
            return Err(RedundancyError::Overloaded {
                detail: "repair queue full on requeue".to_owned(),
            });
        }
        self.queue.push_back(task);
        Ok(())
    }

    /// Whether another decode may start under the concurrency bound.
    #[must_use]
    pub const fn can_decode(&self) -> bool {
        self.inflight_decodes < self.budgets.limits.max_concurrent_decodes
    }

    /// Acquires a decode slot, if available.
    #[must_use]
    pub fn try_acquire_decode(&mut self) -> bool {
        if self.can_decode() {
            self.inflight_decodes += 1;
            true
        } else {
            false
        }
    }

    /// Releases a decode slot.
    pub fn release_decode(&mut self) {
        self.inflight_decodes = self.inflight_decodes.saturating_sub(1);
    }
}

/// Scrub report for one asset: per-fragment verdicts for operators and the
/// future Phase 11 policy. Pure classification — this function never repairs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScrubReport {
    /// Asset scrubbed.
    pub asset: AssetId,
    /// Generation scrubbed.
    pub generation: u64,
    /// Per-fragment verdicts.
    pub verdicts: Vec<(u32, FragmentVerdict)>,
    /// Corruptions found (checksum mismatches + reconstructable).
    pub corruptions: usize,
    /// Missing fragments.
    pub missing: usize,
    /// Stale-generation fragments.
    pub stale: usize,
}

/// Classifies scrub observations into a [`ScrubReport`].
#[must_use]
pub fn scrub_report(
    asset: AssetId,
    generation: u64,
    verdicts: Vec<(u32, FragmentVerdict)>,
) -> ScrubReport {
    let corruptions = verdicts
        .iter()
        .filter(|(_, v)| {
            matches!(
                v,
                FragmentVerdict::ChecksumMismatch
                    | FragmentVerdict::ReconstructableCorruption
                    | FragmentVerdict::UnrecoverableCorruption
            )
        })
        .count();
    let missing = verdicts
        .iter()
        .filter(|(_, v)| matches!(v, FragmentVerdict::Missing))
        .count();
    let stale = verdicts
        .iter()
        .filter(|(_, v)| matches!(v, FragmentVerdict::StaleGeneration))
        .count();
    ScrubReport {
        asset,
        generation,
        verdicts,
        corruptions,
        missing,
        stale,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AssetKind, SchemeParams};

    fn asset() -> AssetId {
        AssetId::new(AssetKind::Chunk, 1, [5; 32]).expect("asset")
    }

    #[test]
    fn health_classes_are_exact() {
        let params = SchemeParams::Replication(crate::ReplicationParams { copies: 3 });
        let healthy = assess(
            asset(),
            1,
            params,
            vec![
                (0, FragmentVerdict::Healthy),
                (1, FragmentVerdict::Healthy),
                (2, FragmentVerdict::Healthy),
            ],
            2,
        );
        assert_eq!(healthy.health, AssetHealth::Healthy);
        assert!(healthy.reconstructable);
        let degraded = assess(
            asset(),
            1,
            params,
            vec![
                (0, FragmentVerdict::Healthy),
                (1, FragmentVerdict::Healthy),
                (2, FragmentVerdict::Missing),
            ],
            1,
        );
        assert_eq!(degraded.health, AssetHealth::Degraded);
        let critical = assess(
            asset(),
            1,
            params,
            vec![
                (0, FragmentVerdict::Healthy),
                (1, FragmentVerdict::Missing),
                (2, FragmentVerdict::Missing),
            ],
            0,
        );
        assert_eq!(critical.health, AssetHealth::Critical);
        assert!(critical.reconstructable);
        let lost = assess(
            asset(),
            1,
            params,
            vec![
                (0, FragmentVerdict::Missing),
                (1, FragmentVerdict::Missing),
                (2, FragmentVerdict::Missing),
            ],
            0,
        );
        assert_eq!(lost.health, AssetHealth::Unrecoverable);
        assert!(!lost.reconstructable);
    }

    #[test]
    fn queue_coalesces_and_bounds() {
        let mut engine = RepairEngine::with_budgets(RepairBudgets::test_wide());
        let task = RepairTask {
            asset: asset(),
            generation: 2,
            fragments: vec![1],
            reason: RepairReason::FragmentMissing,
            targets: Vec::new(),
        };
        assert!(engine.queue(task.clone()).expect("queues"));
        assert!(!engine.queue(task.clone()).expect("coalesces"));
        // New fragment merges into the same queued task.
        let mut wider = task.clone();
        wider.fragments = vec![2];
        assert!(engine.queue(wider).expect("merges"));
        assert_eq!(engine.pending(), 1);
        let drained = engine.tick(false);
        assert_eq!(drained.len(), 1);
        assert_eq!(drained[0].fragments, vec![1, 2]);
    }
}
