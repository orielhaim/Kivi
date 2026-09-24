//! Observation window, sequence, and period identity.

use core::time::Duration;

/// Stable identity of one producer observation window.
///
/// Window IDs are opaque to the fabric. Producers may reuse a window for
/// multiple typed samples, skip IDs, or restart their local numbering.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ObservationWindowId(u64);

impl ObservationWindowId {
    /// First window in a producer's numbering space.
    pub const FIRST: Self = Self(1);

    /// Wraps a raw window identifier.
    #[must_use]
    pub const fn from_u64(value: u64) -> Self {
        Self(value)
    }

    /// Returns the raw window identifier.
    #[must_use]
    pub const fn as_u64(self) -> u64 {
        self.0
    }

    /// Returns the next identifier.
    ///
    /// # Errors
    ///
    /// Returns [`WindowIdExhausted`] at `u64::MAX` instead of wrapping.
    pub const fn next(self) -> Result<Self, WindowIdExhausted> {
        match self.0.checked_add(1) {
            Some(value) => Ok(Self(value)),
            None => Err(WindowIdExhausted),
        }
    }
}

/// A producer observation window exhausted its `u64` identifier space.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WindowIdExhausted;

impl core::fmt::Display for WindowIdExhausted {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter.write_str("observation window identifier space exhausted")
    }
}

impl std::error::Error for WindowIdExhausted {}

/// Globally monotonic acceptance sequence within one fabric.
///
/// The fabric assigns no sequence itself. Producers provide one, and
/// [`ObservationFabric`](crate::ObservationFabric) accepts only strictly
/// increasing values, making replay order explicit and observable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ObservationSequence(u64);

impl ObservationSequence {
    /// First valid acceptance sequence.
    pub const FIRST: Self = Self(1);

    /// Wraps a raw sequence value.
    #[must_use]
    pub const fn from_u64(value: u64) -> Self {
        Self(value)
    }

    /// Returns the raw sequence value.
    #[must_use]
    pub const fn as_u64(self) -> u64 {
        self.0
    }

    /// Returns the next sequence.
    ///
    /// # Errors
    ///
    /// Returns [`SequenceExhausted`] at `u64::MAX` instead of wrapping.
    pub const fn next(self) -> Result<Self, SequenceExhausted> {
        match self.0.checked_add(1) {
            Some(value) => Ok(Self(value)),
            None => Err(SequenceExhausted),
        }
    }
}

/// A monotonic observation sequence exhausted its `u64` value space.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SequenceExhausted;

impl core::fmt::Display for SequenceExhausted {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter.write_str("observation sequence space exhausted")
    }
}

impl std::error::Error for SequenceExhausted {}

/// Position of a sample in its producer window and global replay order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ObservationStamp {
    /// Producer-defined observation window.
    pub window: ObservationWindowId,
    /// Fabric-wide strictly increasing acceptance sequence.
    pub sequence: ObservationSequence,
}

/// Nonzero duration represented by the samples in one observation window.
///
/// Rates divide by this period, so a zero period is not representable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ObservationPeriod(Duration);

impl ObservationPeriod {
    /// Creates a nonzero observation period.
    #[must_use]
    pub const fn new(period: Duration) -> Option<Self> {
        if period.is_zero() {
            None
        } else {
            Some(Self(period))
        }
    }

    /// Returns the represented duration.
    #[must_use]
    pub const fn duration(self) -> Duration {
        self.0
    }

    /// Returns the duration in microseconds.
    #[must_use]
    pub fn micros(self) -> u128 {
        self.0.as_micros()
    }
}
