//! Deterministic bounded `SpaceSaving` hot-key sketch.

use crate::signals::HotKeySignal;

/// Fixed-width producer identity for a hot key.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct HotKeyId(u128);

impl HotKeyId {
    /// Wraps a raw 128-bit key identity.
    #[must_use]
    pub const fn from_u128(value: u128) -> Self {
        Self(value)
    }

    /// Returns the raw key identity.
    #[must_use]
    pub const fn as_u128(self) -> u128 {
        self.0
    }
}

/// Frequency classification for a retained hot-key entry.
///
/// A key is hot when its estimated count is at least one percent of all
/// observations in the sketch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HotnessClass {
    /// Seen once and below the hot threshold.
    Cold,
    /// Seen more than once but below the hot threshold.
    Warm,
    /// At least one percent of observed accesses.
    Hot,
}

/// Execution-path classification independent of access frequency.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ExecutionClass {
    /// Neither critical-path nor tail evidence reaches one half of retained accesses.
    Ordinary,
    /// At least one half of retained accesses affected tail latency.
    TailSensitive,
    /// At least one half of retained accesses were on a critical path.
    CriticalPath,
}

/// Combined frequency and execution-path classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HotKeyClassification {
    /// Frequency-derived classification.
    pub hotness: HotnessClass,
    /// Execution-path-derived classification.
    pub execution: ExecutionClass,
}

/// One deterministic entry emitted by the hot-key sketch.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HotKeyEntry {
    /// Fixed-width key identity.
    pub key: HotKeyId,
    /// `SpaceSaving` frequency estimate.
    pub estimated_count: u64,
    /// Global additive estimate bound at the current observation count.
    pub error_bound: u64,
    /// Estimated share of all retained observations in `[0, 1]`.
    pub share: f64,
    /// Critical-path accesses since this key entered the sketch.
    pub critical_path_hits: u64,
    /// Tail-contributing accesses since this key entered the sketch.
    pub tail_hits: u64,
    /// Attributed stall nanoseconds since this key entered the sketch.
    pub stall_time_ns: u64,
    /// Combined hotness and execution classification.
    pub classification: HotKeyClassification,
}

/// Concentration summary over the sketch's estimated frequencies.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HotKeyConcentration {
    /// Estimated share of the most frequent retained key.
    pub top_one_share: f64,
    /// Estimated combined share of the ten most frequent retained keys.
    pub top_ten_share: f64,
    /// Sum of squared estimated key shares.
    pub herfindahl: f64,
}

impl HotKeyConcentration {
    /// Returns zero concentration for an empty sketch.
    #[must_use]
    pub const fn empty() -> Self {
        Self {
            top_one_share: 0.0,
            top_ten_share: 0.0,
            herfindahl: 0.0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct Entry {
    key: HotKeyId,
    count: u64,
    critical_path_hits: u64,
    tail_hits: u64,
    stall_time_ns: u64,
}

impl Entry {
    #[allow(clippy::cast_precision_loss)]
    fn snapshot(self, total: u64, error_bound: u64) -> HotKeyEntry {
        let share = if total == 0 {
            0.0
        } else {
            self.count as f64 / total as f64
        };
        let hotness = if self.count == 1 {
            HotnessClass::Cold
        } else if u128::from(self.count).saturating_mul(100) >= u128::from(total) {
            HotnessClass::Hot
        } else {
            HotnessClass::Warm
        };
        let execution = if self.critical_path_hits.saturating_mul(2) >= self.count {
            ExecutionClass::CriticalPath
        } else if self.tail_hits.saturating_mul(2) >= self.count {
            ExecutionClass::TailSensitive
        } else {
            ExecutionClass::Ordinary
        };
        HotKeyEntry {
            key: self.key,
            estimated_count: self.count,
            error_bound,
            share,
            critical_path_hits: self.critical_path_hits,
            tail_hits: self.tail_hits,
            stall_time_ns: self.stall_time_ns,
            classification: HotKeyClassification { hotness, execution },
        }
    }
}

/// Failure while incrementing a fixed-capacity frequency counter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HotKeyError {
    /// A per-key or total observation count reached `u64::MAX`.
    CounterOverflow,
}

impl core::fmt::Display for HotKeyError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter.write_str("hot-key observation counter exhausted")
    }
}

impl std::error::Error for HotKeyError {}

/// Deterministic bounded `SpaceSaving` sketch for hot keys.
///
/// For capacity `m` and `n` total observations, every retained estimate for
/// a key satisfies `true_count <= estimate <= true_count + floor(n / m)`.
/// A key with true frequency greater than `n / m` is retained. These are
/// standard `SpaceSaving` bounds.
///
/// Output is sorted by descending estimated count and then ascending
/// [`HotKeyId`]. The sketch is deterministic for a given observation order and
/// contains at most `capacity` keys.
#[derive(Debug, Clone)]
pub struct HotKeySketch {
    entries: Vec<Entry>,
    capacity: usize,
    capacity_u64: u64,
    total: u64,
}

impl HotKeySketch {
    /// Creates an empty sketch when `capacity` is nonzero and fits `u64`.
    #[must_use]
    pub fn new(capacity: usize) -> Option<Self> {
        let capacity_u64 = u64::try_from(capacity).ok()?;
        if capacity == 0 {
            return None;
        }
        Some(Self {
            entries: Vec::with_capacity(capacity),
            capacity,
            capacity_u64,
            total: 0,
        })
    }

    /// Records one key access.
    ///
    /// When full, `SpaceSaving` replaces the least frequent entry. Frequency
    /// ties choose the smallest [`HotKeyId`]. Per-key execution evidence is
    /// reset when an entry is replaced because earlier evidence is no longer
    /// retained.
    ///
    /// # Errors
    ///
    /// Returns [`HotKeyError::CounterOverflow`] without mutating the sketch if
    /// the total or replacement counter cannot advance.
    pub fn observe(&mut self, signal: &HotKeySignal) -> Result<(), HotKeyError> {
        let next_total = self
            .total
            .checked_add(1)
            .ok_or(HotKeyError::CounterOverflow)?;
        if let Some(entry) = self
            .entries
            .iter_mut()
            .find(|entry| entry.key == signal.key)
        {
            entry.count = entry
                .count
                .checked_add(1)
                .ok_or(HotKeyError::CounterOverflow)?;
            Self::add_evidence(entry, signal);
            self.total = next_total;
            return Ok(());
        }

        let index = if self.entries.len() < self.capacity {
            self.entries.push(Entry {
                key: signal.key,
                count: 1,
                critical_path_hits: 0,
                tail_hits: 0,
                stall_time_ns: 0,
            });
            self.entries.len() - 1
        } else {
            let index = self.replacement_index();
            let minimum = self.entries[index].count;
            let count = minimum.checked_add(1).ok_or(HotKeyError::CounterOverflow)?;
            self.entries[index] = Entry {
                key: signal.key,
                count,
                critical_path_hits: 0,
                tail_hits: 0,
                stall_time_ns: 0,
            };
            index
        };
        Self::add_evidence(&mut self.entries[index], signal);
        self.total = next_total;
        Ok(())
    }

    fn add_evidence(entry: &mut Entry, signal: &HotKeySignal) {
        entry.critical_path_hits = entry
            .critical_path_hits
            .saturating_add(u64::from(signal.on_critical_path));
        entry.tail_hits = entry
            .tail_hits
            .saturating_add(u64::from(signal.affects_tail));
        entry.stall_time_ns = entry.stall_time_ns.saturating_add(signal.stall_time_ns);
    }

    fn replacement_index(&self) -> usize {
        let mut best = 0;
        for index in 1..self.entries.len() {
            let candidate = self.entries[index];
            let incumbent = self.entries[best];
            if candidate.count < incumbent.count
                || (candidate.count == incumbent.count && candidate.key < incumbent.key)
            {
                best = index;
            }
        }
        best
    }

    /// Returns configured key capacity.
    #[must_use]
    pub const fn capacity(&self) -> usize {
        self.capacity
    }

    /// Returns the number of retained keys.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether no key is retained.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Returns the total number of accepted observations.
    #[must_use]
    pub const fn total_count(&self) -> u64 {
        self.total
    }

    /// Returns the global additive `SpaceSaving` error bound `floor(n / m)`.
    #[must_use]
    pub const fn error_bound(&self) -> u64 {
        self.total / self.capacity_u64
    }

    /// Returns deterministic sorted entries and their shared error bound.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn entries(&self) -> Vec<HotKeyEntry> {
        let error_bound = self.error_bound();
        let mut entries: Vec<_> = self
            .entries
            .iter()
            .copied()
            .map(|entry| entry.snapshot(self.total, error_bound))
            .collect();
        entries.sort_unstable_by(|left, right| {
            right
                .estimated_count
                .cmp(&left.estimated_count)
                .then_with(|| left.key.cmp(&right.key))
        });
        entries
    }

    /// Returns concentration over the top one, top ten, and all entries.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn concentration(&self) -> HotKeyConcentration {
        if self.total == 0 {
            return HotKeyConcentration::empty();
        }
        let entries = self.entries();
        let top_one_share = entries.first().map_or(0.0, |entry| {
            entry.estimated_count as f64 / self.total as f64
        });
        let top_ten: u64 = entries.iter().take(10).fold(0_u64, |sum, entry| {
            sum.saturating_add(entry.estimated_count)
        });
        let top_ten_share = top_ten as f64 / self.total as f64;
        let herfindahl = entries.iter().fold(0.0, |sum, entry| {
            let share = entry.estimated_count as f64 / self.total as f64;
            sum + (share * share)
        });
        HotKeyConcentration {
            top_one_share,
            top_ten_share,
            herfindahl,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;

    fn access(key: u128, critical: bool, tail: bool) -> HotKeySignal {
        HotKeySignal {
            key: HotKeyId::from_u128(key),
            on_critical_path: critical,
            affects_tail: tail,
            stall_time_ns: if critical { 100 } else { 0 },
        }
    }

    #[test]
    fn capacity_error_bound_and_sorted_output_are_bounded() {
        let counts = [
            (1_u128, 1_000_u64),
            (2, 500),
            (3, 250),
            (4, 100),
            (5, 50),
            (6, 25),
            (7, 10),
            (8, 5),
        ];
        let mut sketch = HotKeySketch::new(4).expect("capacity");
        let mut truth = BTreeMap::new();
        for (key, count) in counts {
            truth.insert(HotKeyId::from_u128(key), count);
            for _ in 0..count {
                sketch.observe(&access(key, false, false)).expect("observe");
            }
        }
        let entries = sketch.entries();
        assert_eq!(sketch.len(), sketch.capacity());
        assert_eq!(entries.len(), 4);
        assert_eq!(sketch.total_count(), 1_940);
        let bound =
            sketch.total_count() / u64::try_from(sketch.capacity()).expect("capacity fits u64");
        assert_eq!(bound, 485);
        for pair in entries.windows(2) {
            assert!(
                pair[0].estimated_count > pair[1].estimated_count
                    || (pair[0].estimated_count == pair[1].estimated_count
                        && pair[0].key < pair[1].key)
            );
        }
        for entry in entries {
            let actual = truth.get(&entry.key).copied().expect("retained true key");
            assert!(entry.estimated_count >= actual);
            assert!(entry.estimated_count <= actual + bound);
            assert_eq!(entry.error_bound, bound);
        }
    }

    #[test]
    fn execution_classification_is_independent_of_hotness() {
        let mut sketch = HotKeySketch::new(2).expect("capacity");
        sketch
            .observe(&access(7, true, false))
            .expect("critical access");
        sketch
            .observe(&access(8, false, false))
            .expect("ordinary access");
        for _ in 0..200 {
            sketch
                .observe(&access(8, false, false))
                .expect("hot access");
        }
        let entries = sketch.entries();
        let rare = entries
            .iter()
            .find(|entry| entry.key == HotKeyId::from_u128(7))
            .expect("rare key retained");
        assert_eq!(rare.classification.hotness, HotnessClass::Cold);
        assert_eq!(rare.classification.execution, ExecutionClass::CriticalPath);
        let hot = entries
            .iter()
            .find(|entry| entry.key == HotKeyId::from_u128(8))
            .expect("hot key retained");
        assert_eq!(hot.classification.hotness, HotnessClass::Hot);
        assert_eq!(hot.classification.execution, ExecutionClass::Ordinary);
    }

    #[test]
    fn identical_replay_produces_identical_sorted_entries() {
        let observations = [
            access(20, false, true),
            access(10, true, false),
            access(20, false, true),
            access(10, false, false),
            access(30, false, false),
        ];
        let mut first = HotKeySketch::new(3).expect("capacity");
        let mut second = HotKeySketch::new(3).expect("capacity");
        for observation in observations {
            first.observe(&observation).expect("first");
            second.observe(&observation).expect("second");
        }
        assert_eq!(first.entries(), second.entries());
        assert_eq!(first.concentration(), second.concentration());
        assert_eq!(first.entries()[0].key, HotKeyId::from_u128(10));
        assert_eq!(first.entries()[1].key, HotKeyId::from_u128(20));
    }
}
