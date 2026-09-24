//! Source, confidence, and freshness evidence for observations.

use core::time::Duration;

use kivi_types::Ticks;

/// Producer or derivation path that emitted an observation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ObservationSource {
    /// In-process counters updated on an execution path.
    RuntimeCounter,
    /// Sampled execution traces or PMU events.
    RuntimeTrace,
    /// Periodic sampled telemetry.
    SampledTelemetry,
    /// A report received from another Kivi node.
    PeerReport,
    /// State derived by the control plane.
    ControlPlane,
    /// Deterministic simulation telemetry.
    Simulator,
    /// A bounded aggregation or estimator derived inside the fabric.
    Derived,
}

impl ObservationSource {
    const fn bit(self) -> u8 {
        match self {
            Self::RuntimeCounter => 1 << 0,
            Self::RuntimeTrace => 1 << 1,
            Self::SampledTelemetry => 1 << 2,
            Self::PeerReport => 1 << 3,
            Self::ControlPlane => 1 << 4,
            Self::Simulator => 1 << 5,
            Self::Derived => 1 << 6,
        }
    }

    const ALL: [(Self, u8); 7] = [
        (Self::RuntimeCounter, 1 << 0),
        (Self::RuntimeTrace, 1 << 1),
        (Self::SampledTelemetry, 1 << 2),
        (Self::PeerReport, 1 << 3),
        (Self::ControlPlane, 1 << 4),
        (Self::Simulator, 1 << 5),
        (Self::Derived, 1 << 6),
    ];
}

/// Bounded set of observation sources represented as a fixed bit mask.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct ObservationSources(u8);

impl ObservationSources {
    /// An empty source set.
    #[must_use]
    pub const fn empty() -> Self {
        Self(0)
    }

    /// Returns a set containing one source.
    #[must_use]
    pub const fn from_source(source: ObservationSource) -> Self {
        Self(source.bit())
    }

    /// Adds a source to the set.
    pub fn insert(&mut self, source: ObservationSource) {
        self.0 |= source.bit();
    }

    /// Whether the set contains `source`.
    #[must_use]
    pub const fn contains(self, source: ObservationSource) -> bool {
        self.0 & source.bit() != 0
    }

    /// Whether the set contains no sources.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// Returns sources in declaration order.
    pub fn iter(self) -> impl Iterator<Item = ObservationSource> {
        ObservationSource::ALL
            .into_iter()
            .filter_map(move |(source, bit)| (self.0 & bit != 0).then_some(source))
    }

    /// Returns the stable bit representation.
    #[must_use]
    pub const fn bits(self) -> u8 {
        self.0
    }
}

impl FromIterator<ObservationSource> for ObservationSources {
    fn from_iter<I: IntoIterator<Item = ObservationSource>>(iter: I) -> Self {
        let mut sources = Self::empty();
        for source in iter {
            sources.insert(source);
        }
        sources
    }
}

/// Confidence attached by the producer to an observation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Confidence {
    /// Coarse or heavily sampled evidence.
    Low,
    /// Useful evidence with known but material uncertainty.
    Medium,
    /// Direct or strongly corroborated evidence.
    High,
}

/// Per-sample source, confidence, and producer timestamp.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ObservationMetadata {
    /// Monotonic producer timestamp at which evidence was captured.
    pub observed_at: Ticks,
    /// Producer or derivation path.
    pub source: ObservationSource,
    /// Producer confidence in the evidence.
    pub confidence: Confidence,
}

impl ObservationMetadata {
    /// Creates sample evidence metadata.
    #[must_use]
    pub const fn new(
        observed_at: Ticks,
        source: ObservationSource,
        confidence: Confidence,
    ) -> Self {
        Self {
            observed_at,
            source,
            confidence,
        }
    }
}

/// Freshness classification relative to a caller-supplied observation time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FreshnessStatus {
    /// Evidence age is within the configured budget.
    Current,
    /// Evidence age exceeds the configured budget.
    Stale,
    /// Producer timestamp is ahead of the snapshot clock.
    Future,
}

/// Evidence age and classification at snapshot time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Freshness {
    /// Nonnegative age of the newest evidence.
    pub age: Duration,
    /// Classification relative to the configured freshness budget.
    pub status: FreshnessStatus,
}

impl Freshness {
    /// Classifies evidence without reading a clock.
    #[must_use]
    pub fn classify(observed_at: Ticks, as_of: Ticks, budget: Duration) -> Self {
        if observed_at > as_of {
            return Self {
                age: Duration::ZERO,
                status: FreshnessStatus::Future,
            };
        }
        let age = as_of.saturating_since(observed_at);
        let status = if age <= budget {
            FreshnessStatus::Current
        } else {
            FreshnessStatus::Stale
        };
        Self { age, status }
    }

    /// Whether evidence is inside its freshness budget.
    #[must_use]
    pub const fn is_current(self) -> bool {
        matches!(self.status, FreshnessStatus::Current)
    }
}

/// Aggregated provenance for all retained samples in one scope.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AggregateEvidence {
    /// Oldest producer timestamp in the retained window.
    pub oldest_observed_at: Ticks,
    /// Newest producer timestamp in the retained window.
    pub latest_observed_at: Ticks,
    /// Bounded set of contributing sources.
    pub sources: ObservationSources,
    /// Lowest confidence among retained samples.
    pub confidence: Confidence,
    /// Freshness of the newest evidence at snapshot time.
    pub freshness: Freshness,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn freshness_distinguishes_current_stale_and_future_evidence() {
        let observed = Ticks::from_micros(100);
        let current = Freshness::classify(
            observed,
            Ticks::from_micros(1_100),
            Duration::from_millis(1),
        );
        let stale = Freshness::classify(
            observed,
            Ticks::from_micros(1_101),
            Duration::from_millis(1),
        );
        let future =
            Freshness::classify(observed, Ticks::from_micros(99), Duration::from_millis(1));
        assert_eq!(current.status, FreshnessStatus::Current);
        assert!(current.is_current());
        assert_eq!(stale.status, FreshnessStatus::Stale);
        assert_eq!(future.status, FreshnessStatus::Future);
        assert_eq!(future.age, Duration::ZERO);
    }

    #[test]
    fn source_sets_are_fixed_and_iterate_deterministically() {
        let mut sources = ObservationSources::empty();
        sources.insert(ObservationSource::Simulator);
        sources.insert(ObservationSource::RuntimeCounter);
        sources.insert(ObservationSource::RuntimeCounter);
        assert!(sources.contains(ObservationSource::RuntimeCounter));
        assert!(sources.contains(ObservationSource::Simulator));
        assert!(!sources.contains(ObservationSource::PeerReport));
        assert!(!sources.is_empty());
        assert_eq!(
            sources.iter().collect::<Vec<_>>(),
            vec![
                ObservationSource::RuntimeCounter,
                ObservationSource::Simulator
            ]
        );
    }
}
