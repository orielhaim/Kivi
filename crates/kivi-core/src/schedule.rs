//! Scheduler vocabulary: stable event identifiers.
//!
//! The scheduler assigns identifiers in schedule-call order and orders ready
//! events by `(scheduled tick, identifier)`, giving a total order with no
//! ties. Identifiers therefore serve two roles at once — deterministic
//! tie-break and stable cancellation handle — from a single checked counter,
//! so neither role can wrap or saturate independently.

use core::fmt;

/// Stable identifier of one scheduled event.
///
/// Minted by the scheduler in schedule-call order starting at
/// [`FIRST`](Self::FIRST). Advancement is checked: minting past `u64::MAX`
/// fails with [`EventIdExhausted`] instead of reusing an identifier (a reused
/// identifier could cancel or order the wrong event).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct EventId(u64);

impl EventId {
    /// First identifier minted by a fresh scheduler.
    pub const FIRST: Self = Self(0);

    /// Wraps a raw identifier value.
    #[must_use]
    pub const fn from_u64(value: u64) -> Self {
        Self(value)
    }

    /// Returns the raw value.
    #[must_use]
    pub const fn as_u64(self) -> u64 {
        self.0
    }

    /// Successor identifier for the next scheduled event.
    ///
    /// # Errors
    ///
    /// Returns [`EventIdExhausted`] if `self` is already `u64::MAX`.
    pub const fn next(self) -> Result<Self, EventIdExhausted> {
        match self.0.checked_add(1) {
            Some(value) => Ok(Self(value)),
            None => Err(EventIdExhausted),
        }
    }
}

impl From<EventId> for u64 {
    /// Returns the raw identifier value.
    fn from(id: EventId) -> Self {
        id.0
    }
}

impl From<u64> for EventId {
    /// Wraps a raw identifier value.
    fn from(value: u64) -> Self {
        EventId(value)
    }
}

impl fmt::Display for EventId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ev{}", self.0)
    }
}

/// Reported when the scheduler cannot mint another event identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, thiserror::Error)]
#[error("event identifier space exhausted")]
pub struct EventIdExhausted;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identifiers_advance_and_fail_explicitly_at_the_top() {
        assert_eq!(EventId::FIRST.as_u64(), 0);
        assert_eq!(
            EventId::FIRST.next().expect("advances"),
            EventId::from_u64(1)
        );
        assert_eq!(EventId::from_u64(u64::MAX).next(), Err(EventIdExhausted));
    }

    #[test]
    fn identifiers_order_and_display() {
        assert!(EventId::from_u64(3) < EventId::from_u64(4));
        assert_eq!(EventId::from_u64(7).to_string(), "ev7");
    }
}
