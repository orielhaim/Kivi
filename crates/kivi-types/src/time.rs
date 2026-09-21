//! Time value types.
//!
//! These are inert stamps and spans: nothing here reads a clock. Callers pass
//! timestamps in explicitly so production can supply hardware time while
//! deterministic simulation supplies virtual time (RFC §176, §177).
//!
//! Three concepts stay structurally distinct:
//!
//! * [`Ticks`] — deterministic virtual monotonic microseconds since a
//!   clock-defined origin. Simulation and correctness-sensitive timing.
//! * [`WallTimestamp`] — exact wall-clock instants as signed Unix
//!   microseconds. Expiry and other logical metadata (RFC §163).
//! * `std::time::Instant` (not defined here) — monotonic process time for
//!   local timeouts, latency measurement, and scheduler deadlines. A system
//!   clock adjustment must never break a local timeout, so wall and
//!   monotonic time never share a type.
//!
//! [`WallTimestamp`] owns Kivi's durable semantics (signed Unix
//! microseconds, microsecond precision). Jiff owns the datetime mechanics:
//! parsing, formatting, and signed arithmetic all delegate to
//! [`jiff::Timestamp`]/[`jiff::SignedDuration`] at explicit, fallible
//! boundaries. Jiff's nanosecond precision never leaks into storage:
//! sub-microsecond remainders truncate toward zero on the way in, and
//! conversion of out-of-range sentinels fails instead of saturating.

use core::fmt;
use core::time::Duration;

use jiff::{SignedDuration, Timestamp};

/// Monotonic virtual timestamp in microseconds.
///
/// The origin is defined by the clock implementation (`kivi-core` provides the
/// `Clock` trait and a manual clock for simulation); only differences and
/// ordering are meaningful across implementations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Ticks(u64);

impl Ticks {
    /// Wraps a raw microsecond count.
    #[must_use]
    pub const fn from_micros(value: u64) -> Self {
        Self(value)
    }

    /// Returns the raw microsecond count.
    #[must_use]
    pub const fn as_micros(self) -> u64 {
        self.0
    }

    /// Advances by `delta`, saturating instead of wrapping or truncating.
    ///
    /// Durations wider than 64 bits clamp to `u64::MAX` (fail closed: time
    /// jumps forward to the end rather than landing somewhere arbitrary).
    #[must_use]
    pub fn advance_by(self, delta: Duration) -> Self {
        let wide = delta.as_micros();
        let micros = u64::try_from(wide).unwrap_or(u64::MAX);
        Self(self.0.saturating_add(micros))
    }

    /// Elapsed microseconds from `earlier` to `self`, saturating at zero if
    /// `earlier` is newer (clock skew fails closed to "no time passed").
    #[must_use]
    pub const fn saturating_since(self, earlier: Self) -> Duration {
        Duration::from_micros(self.0.saturating_sub(earlier.0))
    }
}

impl fmt::Display for Ticks {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}us", self.0)
    }
}

impl From<Ticks> for u64 {
    /// Returns the raw microsecond count.
    fn from(ticks: Ticks) -> Self {
        ticks.0
    }
}

impl From<u64> for Ticks {
    /// Wraps a raw microsecond count.
    fn from(value: u64) -> Self {
        Self(value)
    }
}

/// Exact wall-clock instant as signed microseconds since the Unix epoch.
///
/// The project-owned durable representation is the inner `i64`
/// (little-endian on every wire/durable format); Jiff never defines the
/// encoding. Negative values are pre-epoch instants and compare before
/// [`EPOCH`](Self::EPOCH). Durable precision stays at microseconds even
/// though Jiff itself resolves nanoseconds: [`from_jiff`](Self::from_jiff)
/// truncates sub-microsecond remainders toward zero, and values outside
/// Jiff's ±9999-year range (ordering sentinels such as [`MAX`](Self::MAX))
/// fail conversion explicitly instead of saturating.
///
/// There is deliberately no `From<i64>`: raw integers cross this boundary
/// only at codecs and clock edges via [`from_micros`](Self::from_micros),
/// so wall time and virtual [`Ticks`] cannot mix by accident.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct WallTimestamp(i64);

impl WallTimestamp {
    /// The Unix epoch itself.
    pub const EPOCH: Self = Self(0);

    /// Largest storable stamp. An ordering sentinel, not a real instant:
    /// conversion to Jiff fails with [`TimeError::OutOfRange`].
    pub const MAX: Self = Self(i64::MAX);

    /// Smallest storable stamp (pre-epoch bound). Like [`MAX`](Self::MAX),
    /// an ordering sentinel outside Jiff's range.
    pub const MIN: Self = Self(i64::MIN);

    /// Wraps signed microseconds since the Unix epoch (negative is
    /// pre-epoch). Total: every `i64` is a valid stamp; Jiff-range
    /// membership is checked at the conversion boundary, not here.
    #[must_use]
    pub const fn from_micros(value: i64) -> Self {
        Self(value)
    }

    /// Returns signed microseconds since the Unix epoch.
    #[must_use]
    pub const fn as_micros(self) -> i64 {
        self.0
    }

    /// Converts a Jiff timestamp, truncating sub-microsecond remainders
    /// toward zero.
    ///
    /// # Errors
    ///
    /// Returns [`TimeError::OutOfRange`] if the instant needs more than 64
    /// bits of microseconds (unreachable for Jiff's ±9999-year range, but
    /// checked rather than assumed).
    pub fn from_jiff(stamp: Timestamp) -> Result<Self, TimeError> {
        let nanos = stamp.as_nanosecond();
        let micros = nanos.div_euclid(1_000);
        let value = i64::try_from(micros).map_err(|_| TimeError::OutOfRange {
            micros: stamp.as_microsecond(),
        })?;
        // `div_euclid` floors negatives; Jiff truncates toward zero, so
        // adjust a negative exact-floor overshoot back up by one microsecond.
        let value = if nanos < 0 && nanos % 1_000 != 0 {
            value.saturating_add(1)
        } else {
            value
        };
        Ok(Self(value))
    }

    /// Converts to a Jiff timestamp for parsing/formatting/arithmetic.
    ///
    /// # Errors
    ///
    /// Returns [`TimeError::OutOfRange`] for sentinels outside Jiff's
    /// ±9999-year range (e.g. [`MAX`](Self::MAX)). Ordering and expiry
    /// comparison still work on such values; only Jiff mechanics refuse
    /// them.
    pub fn to_jiff(self) -> Result<Timestamp, TimeError> {
        Timestamp::from_microsecond(self.0).map_err(|_| TimeError::OutOfRange { micros: self.0 })
    }

    /// Adds a signed duration with Jiff arithmetic.
    ///
    /// # Errors
    ///
    /// Returns [`TimeError::OutOfRange`] when either endpoint leaves Jiff's
    /// range or the sum overflows it. Callers that need total semantics on
    /// raw micros (e.g. saturating lease deadlines) say so explicitly at
    /// their own site instead of hiding it here.
    pub fn checked_add(self, delta: SignedDuration) -> Result<Self, TimeError> {
        let sum = self
            .to_jiff()
            .map_err(|_| TimeError::OutOfRange { micros: self.0 })?
            .checked_add(delta)
            .map_err(|_| TimeError::OutOfRange { micros: self.0 })?;
        Self::from_jiff(sum)
    }

    /// Signed difference `self - earlier` with Jiff arithmetic.
    ///
    /// # Errors
    ///
    /// Returns [`TimeError::OutOfRange`] when either endpoint leaves Jiff's
    /// range. Microsecond differences of in-range stamps always fit
    /// [`SignedDuration`], so a successful conversion never fails here.
    pub fn signed_duration_since(self, earlier: Self) -> Result<SignedDuration, TimeError> {
        let later = self.to_jiff()?;
        let before = earlier.to_jiff()?;
        Ok(later.duration_since(before))
    }

    /// Parses an RFC 3339 instant (`2024-07-11T01:14:00Z`), truncating
    /// sub-microsecond remainders toward zero.
    ///
    /// # Errors
    ///
    /// Returns [`TimeError::InvalidFormat`] for unparsable input and
    /// [`TimeError::OutOfRange`] for instants needing more than 64 bits of
    /// microseconds.
    pub fn parse_rfc3339(text: &str) -> Result<Self, TimeError> {
        let stamp: Timestamp =
            text.parse::<Timestamp>()
                .map_err(|error| TimeError::InvalidFormat {
                    detail: error.to_string(),
                })?;
        Self::from_jiff(stamp)
    }

    /// Formats as RFC 3339 (`2024-07-11T01:14:00Z`) for human-facing output.
    ///
    /// # Errors
    ///
    /// Returns [`TimeError::OutOfRange`] for sentinels outside Jiff's range;
    /// [`Display`](fmt::Display) falls back to raw microseconds there so
    /// diagnostics never fail.
    pub fn to_rfc3339(self) -> Result<String, TimeError> {
        Ok(self.to_jiff()?.to_string())
    }
}

impl Default for WallTimestamp {
    /// The Unix epoch. Fail-closed for expiry comparison: `now >= EPOCH`
    /// always holds, so default-constructed metadata reads as expired,
    /// never as immortal.
    fn default() -> Self {
        Self::EPOCH
    }
}

impl fmt::Display for WallTimestamp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.to_jiff() {
            Ok(stamp) => write!(f, "{stamp}"),
            Err(_) => write!(f, "{}us", self.0),
        }
    }
}

impl From<WallTimestamp> for i64 {
    /// Returns signed microseconds since the Unix epoch.
    fn from(stamp: WallTimestamp) -> Self {
        stamp.0
    }
}

/// Wall-clock conversion failure: explicit at every Jiff boundary, never a
/// silent clamp to zero or maximum time.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum TimeError {
    /// The stamp lies outside Jiff's ±9999-year representable range, so
    /// Jiff mechanics (formatting, parsing, arithmetic) refuse it. The
    /// raw microsecond value is preserved for diagnostics.
    #[error("wall timestamp {micros}us is outside the Jiff range")]
    OutOfRange {
        /// The offending raw microsecond count.
        micros: i64,
    },
    /// RFC 3339 parsing failed.
    #[error("invalid wall timestamp: {detail}")]
    InvalidFormat {
        /// The underlying parse failure.
        detail: String,
    },
}

/// Logical expiration attached to stored data (RFC §163).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Expiry(Option<WallTimestamp>);

impl Expiry {
    /// No expiration: the data never logically expires.
    pub const NEVER: Self = Self(None);

    /// Expires at the given logical timestamp.
    #[must_use]
    pub const fn at(when: WallTimestamp) -> Self {
        Self(Some(when))
    }

    /// Returns the expiration stamp, or `None` for [`NEVER`](Self::NEVER).
    #[must_use]
    pub const fn as_stamp(self) -> Option<WallTimestamp> {
        self.0
    }

    /// Whether the data is logically absent at `now`.
    ///
    /// Pure comparison: the caller supplies `now` from whatever clock (real
    /// or virtual) governs the execution context.
    #[must_use]
    pub const fn is_expired(self, now: WallTimestamp) -> bool {
        match self.0 {
            None => false,
            Some(when) => now.as_micros() >= when.as_micros(),
        }
    }
}

impl fmt::Display for Expiry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.0 {
            None => write!(f, "never"),
            Some(when) => write!(f, "at {when}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ticks_advance_and_measure_spans() {
        let start = Ticks::from_micros(1_000);
        let later = start.advance_by(Duration::from_millis(2));
        assert_eq!(later.as_micros(), 3_000);
        assert_eq!(later.saturating_since(start), Duration::from_millis(2));
        assert_eq!(start.saturating_since(later), Duration::ZERO);
    }

    #[test]
    fn expiry_is_never_or_a_stamp() {
        assert!(!Expiry::NEVER.is_expired(WallTimestamp::MAX));
        let deadline = WallTimestamp::from_micros(500);
        let expiry = Expiry::at(deadline);
        assert_eq!(expiry.as_stamp(), Some(deadline));
        assert!(!expiry.is_expired(WallTimestamp::from_micros(499)));
        assert!(expiry.is_expired(WallTimestamp::from_micros(500)));
        assert!(expiry.is_expired(WallTimestamp::from_micros(501)));
    }

    #[test]
    fn display_formats() {
        assert_eq!(Ticks::from_micros(7).to_string(), "7us");
        assert_eq!(Expiry::NEVER.to_string(), "never");
    }

    #[test]
    fn epoch_round_trips_through_jiff() {
        let epoch = WallTimestamp::EPOCH;
        assert_eq!(epoch.to_jiff().expect("epoch fits").as_microsecond(), 0);
        assert_eq!(
            WallTimestamp::from_jiff(epoch.to_jiff().expect("epoch fits")).expect("round trip"),
            epoch
        );
        assert_eq!(epoch.to_rfc3339().expect("formats"), "1970-01-01T00:00:00Z");
        assert_eq!(
            WallTimestamp::parse_rfc3339("1970-01-01T00:00:00Z").expect("parses"),
            epoch
        );
    }

    #[test]
    fn pre_epoch_stamps_order_before_epoch() {
        let before = WallTimestamp::from_micros(-1);
        assert!(before < WallTimestamp::EPOCH);
        assert_eq!(
            before.to_rfc3339().expect("formats"),
            "1969-12-31T23:59:59.999999Z"
        );
        assert_eq!(
            WallTimestamp::parse_rfc3339("1969-12-31T23:59:59.999999Z").expect("parses"),
            before
        );
        // Negative differences stay signed.
        let gap = WallTimestamp::EPOCH
            .signed_duration_since(before)
            .expect("in range");
        assert_eq!(gap.as_micros(), 1);
        assert_eq!(
            before
                .signed_duration_since(WallTimestamp::EPOCH)
                .expect("in range")
                .as_micros(),
            -1
        );
    }

    #[test]
    fn micros_round_trip_through_jiff() {
        for micros in [
            i64::MIN + 1,
            -1_000_001,
            -1,
            0,
            1,
            1_000_000,
            9_000_000_000_000_000,
        ] {
            let stamp = WallTimestamp::from_micros(micros);
            match stamp.to_jiff() {
                Ok(jiff) => {
                    assert_eq!(
                        WallTimestamp::from_jiff(jiff).expect("in range"),
                        stamp,
                        "micros {micros} must round-trip"
                    );
                }
                Err(TimeError::OutOfRange { .. }) => {}
                Err(error) => panic!("unexpected error for {micros}: {error}"),
            }
        }
    }

    #[test]
    fn sub_microsecond_truncates_toward_zero() {
        let positive = Timestamp::from_nanosecond(1_500).expect("small timestamp fits");
        assert_eq!(
            WallTimestamp::from_jiff(positive).expect("fits"),
            WallTimestamp::from_micros(1)
        );
        let negative = Timestamp::from_nanosecond(-1_500).expect("small timestamp fits");
        assert_eq!(
            WallTimestamp::from_jiff(negative).expect("fits"),
            WallTimestamp::from_micros(-1)
        );
    }

    #[test]
    fn sentinels_compare_but_do_not_convert() {
        assert!(WallTimestamp::MIN < WallTimestamp::EPOCH);
        assert!(WallTimestamp::EPOCH < WallTimestamp::MAX);
        assert!(matches!(
            WallTimestamp::MAX.to_jiff(),
            Err(TimeError::OutOfRange { .. })
        ));
        assert!(matches!(
            WallTimestamp::MIN.to_jiff(),
            Err(TimeError::OutOfRange { .. })
        ));
        // Diagnostics still render instead of failing.
        assert_eq!(WallTimestamp::MAX.to_string(), format!("{}us", i64::MAX));
        assert!(
            WallTimestamp::MAX
                .checked_add(SignedDuration::ZERO)
                .is_err()
        );
    }

    #[test]
    fn checked_add_uses_jiff_arithmetic() {
        let start = WallTimestamp::from_micros(1_000_000);
        let later = start
            .checked_add(SignedDuration::from_secs(2))
            .expect("in range");
        assert_eq!(later, WallTimestamp::from_micros(3_000_000));
        let back = later
            .checked_add(SignedDuration::from_secs(-5))
            .expect("in range");
        assert_eq!(back, WallTimestamp::from_micros(-2_000_000));
        assert_eq!(
            back.signed_duration_since(later).expect("in range"),
            SignedDuration::from_secs(-5)
        );
    }

    #[test]
    fn invalid_rfc3339_is_an_error_not_a_stamp() {
        assert!(matches!(
            WallTimestamp::parse_rfc3339("not a timestamp"),
            Err(TimeError::InvalidFormat { .. })
        ));
        assert!(matches!(
            WallTimestamp::parse_rfc3339("2024-13-45T99:99:99Z"),
            Err(TimeError::InvalidFormat { .. })
        ));
    }

    #[test]
    fn expiry_boundary_and_skew() {
        // Exactly at the deadline counts as expired (strict containment).
        let deadline = WallTimestamp::from_micros(2_000_000);
        let expiry = Expiry::at(deadline);
        assert!(!expiry.is_expired(WallTimestamp::from_micros(1_999_999)));
        assert!(expiry.is_expired(deadline));
        // A skewed (early) clock observes live data; a late clock expires
        // early. Either way the comparison is total and deterministic.
        assert!(!expiry.is_expired(WallTimestamp::from_micros(0)));
        assert!(expiry.is_expired(WallTimestamp::MAX));
        // Readable diagnostics for operators.
        assert_eq!(expiry.to_string(), "at 1970-01-01T00:00:02Z");
    }
}
