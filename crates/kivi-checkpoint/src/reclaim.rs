//! WAL reclamation planning: which sealed segments may go.
//!
//! Safety rule (specs AD–AG, Y): a sealed segment may be reclaimed only
//! when EVERY logical record it contains is covered by the oldest retained
//! recovery point. The oldest promise is the PREVIOUS checkpoint's cut —
//! and only a verified previous counts. A tablet with a single checkpoint
//! (or an unverifiable previous) makes no promise: its records block
//! reclamation of every segment holding them. This keeps genesis replay
//! viable until two checkpoints exist, so losing the sole CURRENT can
//! never brick the database — the WAL still covers everything.
//!
//! ```text
//! reclaimable(segment) ⟺ ∀ tablets T in segment:
//!     floor(T) exists AND max_commit(segment, T) ≤ floor(T)
//! where floor(T) = verified previous_cut(T)
//! ```
//!
//! Deletion itself is best-effort (a failed delete costs disk, never
//! correctness); an unscannable segment is kept along with everything the
//! planner has not yet proven obsolete. The active tail is never a
//! candidate. Reappearing deleted segments are harmless: recovery resumes
//! at the durable floor and the checkpoint cut skips their records.

use std::collections::HashMap;

use kivi_durability::{LaneFloor, SealedSegmentSummary};
use kivi_types::TabletId;

/// One lane's reclaim computation: the new durable floor plus the sealed
/// segments proven obsolete under it.
#[derive(Debug, Clone)]
pub struct LaneReclaim {
    /// The lane.
    pub lane: u16,
    /// New durable floor (first retained segment + watermark).
    pub floor: LaneFloor,
    /// Sealed segments safe to delete, oldest first.
    pub obsolete: Vec<u64>,
}

/// Plans reclamation across lanes.
///
/// * `summaries`: strictly scanned sealed segments per lane (oldest first;
///   any scan failure must have aborted the whole plan before this call).
/// * `active`: active (still-written) segment per lane — never a
///   candidate, but anchors the floor when everything sealed is gone.
/// * `active_first_batch`: first batch of each active segment (or the
///   upcoming sequence for an empty tail).
/// * `next_batch`: one-past-issued batch watermark per lane.
/// * `floors`: oldest recovery promise (commit position) per tablet.
///
/// Returns per-lane plans sorted by lane. A lane with no summaries keeps a
/// floor pointing at its active segment.
#[must_use]
pub fn plan_reclaim<S: std::hash::BuildHasher>(
    summaries: &std::collections::HashMap<u16, Vec<SealedSegmentSummary>, S>,
    active: &std::collections::HashMap<u16, u64, S>,
    active_first_batch: &std::collections::HashMap<u16, u64, S>,
    next_batch: &std::collections::HashMap<u16, u64, S>,
    floors: &std::collections::HashMap<TabletId, u64, S>,
) -> Vec<LaneReclaim> {
    let mut lanes: Vec<u16> = summaries.keys().copied().collect();
    for lane in active.keys() {
        if !lanes.contains(lane) {
            lanes.push(*lane);
        }
    }
    lanes.sort_unstable();
    lanes
        .into_iter()
        .map(|lane| {
            let lane_summaries = summaries.get(&lane);
            let active_segment = active.get(&lane).copied().unwrap_or(0);
            let watermark = next_batch.get(&lane).copied().unwrap_or(1).max(1);
            let active_first = active_first_batch
                .get(&lane)
                .copied()
                .unwrap_or(watermark)
                .max(1);
            let mut obsolete = Vec::new();
            let mut first_retained: Option<(u64, u64)> = None;
            if let Some(summaries) = lane_summaries {
                for summary in summaries {
                    // The active tail is never a candidate even if a
                    // caller lists it (defense in depth alongside the
                    // lane's own refusal).
                    if summary.segment == active_segment {
                        first_retained =
                            Some((summary.segment, summary_first_batch(summary, watermark)));
                        break;
                    }
                    if segment_covered(summary, floors) {
                        obsolete.push(summary.segment);
                    } else {
                        first_retained =
                            Some((summary.segment, summary_first_batch(summary, watermark)));
                        break;
                    }
                }
            }
            // Everything sealed is gone (or nothing was ever sealed): the
            // floor resumes at the active tail.
            let (first_segment, first_batch) =
                first_retained.unwrap_or((active_segment.max(1), active_first));
            LaneReclaim {
                lane,
                floor: LaneFloor {
                    lane,
                    first_segment,
                    first_batch,
                    next_batch: watermark,
                },
                obsolete,
            }
        })
        .collect()
}

/// First batch sequence of a scanned segment (the upcoming watermark when
/// header-only — vacuously consistent, since no batch precedes it).
fn summary_first_batch(summary: &SealedSegmentSummary, watermark: u64) -> u64 {
    summary.first_batch_seq.unwrap_or(watermark).max(1)
}

/// Whether every record in the segment sits at or below its tablet's
/// floor. Empty (header-only) segments hold no logical records and are
/// vacuously covered.
fn segment_covered<S: std::hash::BuildHasher>(
    summary: &SealedSegmentSummary,
    floors: &HashMap<TabletId, u64, S>,
) -> bool {
    let mut max_per_tablet: HashMap<TabletId, u64> = HashMap::new();
    for (tablet, commit) in &summary.records {
        let entry = max_per_tablet.entry(*tablet).or_insert(0);
        *entry = (*entry).max(commit.as_u64());
    }
    max_per_tablet
        .iter()
        .all(|(tablet, max)| floors.get(tablet).is_some_and(|floor| *max <= *floor))
}

#[cfg(test)]
mod tests {
    use super::*;
    use kivi_types::CommitPosition;

    fn summary(segment: u64, first_batch: u64, records: Vec<(u64, u64)>) -> SealedSegmentSummary {
        SealedSegmentSummary {
            segment,
            first_batch_seq: Some(first_batch),
            records: records
                .into_iter()
                .map(|(tablet, commit)| {
                    (TabletId::from_u64(tablet), CommitPosition::from_u64(commit))
                })
                .collect(),
        }
    }

    fn lane_maps(
        active: u64,
        first_batch: u64,
        next: u64,
    ) -> (HashMap<u16, u64>, HashMap<u16, u64>, HashMap<u16, u64>) {
        (
            HashMap::from([(0, active)]),
            HashMap::from([(0, first_batch)]),
            HashMap::from([(0, next)]),
        )
    }

    #[test]
    fn covered_prefix_is_reclaimed_and_floor_advances() {
        let summaries = HashMap::from([(
            0,
            vec![
                summary(1, 1, vec![(1, 10), (1, 11)]),
                summary(2, 3, vec![(1, 12), (1, 20)]),
                summary(3, 5, vec![(1, 21)]),
            ],
        )]);
        let (active, first_batch, next) = lane_maps(4, 7, 8);
        let floors = HashMap::from([(TabletId::from_u64(1), 20)]);
        let plans = plan_reclaim(&summaries, &active, &first_batch, &next, &floors);
        assert_eq!(plans.len(), 1);
        assert_eq!(plans[0].obsolete, vec![1, 2]);
        assert_eq!(plans[0].floor.first_segment, 3);
        assert_eq!(plans[0].floor.first_batch, 5);
        assert_eq!(plans[0].floor.next_batch, 8);
    }

    #[test]
    fn spanning_segment_blocks_and_anchors_the_floor() {
        let summaries = HashMap::from([(
            0,
            vec![
                summary(1, 1, vec![(1, 10)]),
                // Max commit 30 > floor 20: kept, and everything after it
                // is unreachable for reclaim (ordered scan stops here).
                summary(2, 2, vec![(1, 30), (1, 5)]),
                summary(3, 4, vec![(1, 11)]),
            ],
        )]);
        let (active, first_batch, next) = lane_maps(4, 7, 8);
        let floors = HashMap::from([(TabletId::from_u64(1), 20)]);
        let plans = plan_reclaim(&summaries, &active, &first_batch, &next, &floors);
        assert_eq!(plans[0].obsolete, vec![1]);
        assert_eq!(plans[0].floor.first_segment, 2);
        assert_eq!(plans[0].floor.first_batch, 2);
    }

    #[test]
    fn mixed_tablets_need_every_tablet_covered() {
        let summaries = HashMap::from([(0, vec![summary(1, 1, vec![(1, 10), (2, 99)])])]);
        let (active, first_batch, next) = lane_maps(2, 3, 4);
        // Tablet 2's floor is below the segment max: the whole segment stays.
        let floors = HashMap::from([(TabletId::from_u64(1), 50), (TabletId::from_u64(2), 90)]);
        let plans = plan_reclaim(&summaries, &active, &first_batch, &next, &floors);
        assert!(plans[0].obsolete.is_empty());
        assert_eq!(plans[0].floor.first_segment, 1);
        // Tablet 2 catches up: now reclaimable.
        let floors = HashMap::from([(TabletId::from_u64(1), 50), (TabletId::from_u64(2), 99)]);
        let plans = plan_reclaim(&summaries, &active, &first_batch, &next, &floors);
        assert_eq!(plans[0].obsolete, vec![1]);
        assert_eq!(plans[0].floor.first_segment, 2);
        assert_eq!(plans[0].floor.first_batch, 3);
    }

    #[test]
    fn empty_segments_are_vacuously_covered() {
        let summaries = HashMap::from([(
            0,
            vec![SealedSegmentSummary {
                segment: 1,
                first_batch_seq: None,
                records: Vec::new(),
            }],
        )]);
        let (active, first_batch, next) = lane_maps(2, 3, 4);
        let plans = plan_reclaim(&summaries, &active, &first_batch, &next, &HashMap::new());
        assert_eq!(plans[0].obsolete, vec![1]);
    }

    #[test]
    fn unknown_tablet_blocks_reclaim() {
        // No floor for tablet 9 (no checkpoint covers it): keep the
        // segment. Floors only exist for checkpointed tablets.
        let summaries = HashMap::from([(0, vec![summary(1, 1, vec![(9, 3)])])]);
        let (active, first_batch, next) = lane_maps(2, 3, 4);
        let plans = plan_reclaim(&summaries, &active, &first_batch, &next, &HashMap::new());
        assert!(plans[0].obsolete.is_empty());
    }
}
