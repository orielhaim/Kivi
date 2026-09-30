//! Primitive strongly typed identifiers.
//!
//! All identifiers are transparent, `Copy` newtypes with value semantics only.
//! Canonical binary encoding (little-endian, fixed width) is defined in
//! `kivi-codec`, never here, so this crate stays dependency-free.

use core::fmt;

/// Reported when a fencing generation cannot advance past `u64::MAX`.
///
/// Fencing generations must never wrap or silently stop advancing: reaching
/// the end of the numbering space is an explicit, operator-visible exhaustion
/// condition, not a quiet saturation. In practice the space is inexhaustible
/// (`2^64` topology transitions); the error exists so the type system, not
/// convention, guarantees no silent reuse of generation values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, thiserror::Error)]
#[non_exhaustive]
pub enum GenerationExhausted {
    /// A tablet epoch reached the end of its numbering space.
    #[error("tablet epoch space exhausted")]
    TabletEpoch,
    /// A write-guard generation reached the end of its numbering space.
    #[error("write-guard generation space exhausted")]
    WriteGuardGeneration,
    /// A node incarnation reached the end of its numbering space.
    #[error("node incarnation space exhausted")]
    NodeIncarnation,
    /// A roster generation reached the end of its numbering space.
    #[error("roster generation space exhausted")]
    RosterGeneration,
}

/// Reported when an ordering or deduplication counter cannot advance past
/// `u64::MAX`.
///
/// Like fencing generations, these counters must never wrap or silently
/// saturate: [`RequestSeq`] saturation would reuse a mutation identity and
/// return a stale deduplicated outcome for a genuinely new mutation, while
/// [`CommitPosition`] saturation would assign one log slot twice. Exhaustion
/// is explicit and operator-visible instead.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, thiserror::Error)]
#[non_exhaustive]
pub enum SequenceExhausted {
    /// A per-session request sequence reached the end of its numbering space.
    #[error("request sequence space exhausted")]
    RequestSeq,
    /// A tablet log position reached the end of its numbering space.
    #[error("commit position space exhausted")]
    CommitPosition,
}

/// Identity of a whole Kivi cluster (exact control-plane state).
///
/// 128-bit globally unique value assigned at cluster creation. Displayed as
/// 32 lowercase hexadecimal digits.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ClusterId(u128);

impl ClusterId {
    /// Wraps a raw 128-bit value.
    #[must_use]
    pub const fn from_u128(value: u128) -> Self {
        Self(value)
    }

    /// Returns the raw 128-bit value.
    #[must_use]
    pub const fn as_u128(self) -> u128 {
        self.0
    }
}

impl fmt::Display for ClusterId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:032x}", self.0)
    }
}

/// Identity of a single process/host member of the cluster.
///
/// Assigned by the control plane; never reused across different processes.
/// Combined with [`NodeIncarnation`] to reject stale restarts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NodeId(u64);

impl NodeId {
    /// Wraps a raw control-plane-assigned value.
    #[must_use]
    pub const fn from_u64(value: u64) -> Self {
        Self(value)
    }

    /// Returns the raw value.
    #[must_use]
    pub const fn as_u64(self) -> u64 {
        self.0
    }
}

impl fmt::Display for NodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Identity of a worker pinned to one CPU.
///
/// A worker owns a set of tablet replicas; tablet state stays with its owner
/// worker. Distinct from the hardware's own processor identity
/// (`kivi_hardware::topology::CpuId`): CPUs are hardware, workers are
/// execution owners that can be re-pinned.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct WorkerId(u64);

impl WorkerId {
    /// Wraps a raw worker index.
    #[must_use]
    pub const fn from_u64(value: u64) -> Self {
        Self(value)
    }

    /// Returns the raw value.
    #[must_use]
    pub const fn as_u64(self) -> u64 {
        self.0
    }
}

impl fmt::Display for WorkerId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Identity of a namespace, the main semantic and policy boundary.
///
/// The durable canonical form is this control-plane-assigned 64-bit value, not
/// the human-readable [`crate::NamespaceName`]: names are normalized at the
/// edge, IDs are what replication, durability, and routing store.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NamespaceId(u64);

impl NamespaceId {
    /// Wraps a raw control-plane-assigned value.
    #[must_use]
    pub const fn from_u64(value: u64) -> Self {
        Self(value)
    }

    /// Returns the raw value.
    #[must_use]
    pub const fn as_u64(self) -> u64 {
        self.0
    }
}

impl fmt::Display for NamespaceId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Identity of a tablet: the unit of ownership, routing, consensus, movement,
/// replication, split, merge, and load accounting.
///
/// Ordered so deterministic choices such as "lowest [`TabletId`] coordinates"
/// are well-defined.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TabletId(u64);

impl TabletId {
    /// Wraps a raw control-plane-assigned value.
    #[must_use]
    pub const fn from_u64(value: u64) -> Self {
        Self(value)
    }

    /// Returns the raw value.
    #[must_use]
    pub const fn as_u64(self) -> u64 {
        self.0
    }
}

impl fmt::Display for TabletId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Monotonic fencing epoch of a tablet.
///
/// Every topology transition (split, merge, migration, ownership move) advances
/// the epoch. A mutation carrying an older epoch than the current authority
/// must be rejected: a stale owner may still execute code but can never commit
/// authoritative mutations under a newer epoch.
///
/// `0` is the invalid sentinel and never authorizes anything.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TabletEpoch(u64);

impl TabletEpoch {
    /// Invalid sentinel. Never authorizes; zeroed memory fails closed.
    pub const INVALID: Self = Self(0);
    /// First valid epoch of a newly created tablet.
    pub const INITIAL: Self = Self(1);

    /// Wraps a raw epoch value.
    #[must_use]
    pub const fn from_u64(value: u64) -> Self {
        Self(value)
    }

    /// Returns the raw value.
    #[must_use]
    pub const fn as_u64(self) -> u64 {
        self.0
    }

    /// Whether this is a real epoch rather than the invalid sentinel.
    #[must_use]
    pub const fn is_valid(self) -> bool {
        self.0 != 0
    }

    /// Successor epoch for the next topology transition.
    ///
    /// Fails with [`GenerationExhausted::TabletEpoch`] at `u64::MAX` instead
    /// of wrapping or saturating: reusing an epoch value would resurrect a
    /// stale authority, so exhaustion must be explicit.
    ///
    /// # Errors
    ///
    /// Returns [`GenerationExhausted::TabletEpoch`] if `self` is already
    /// `u64::MAX`.
    pub const fn next(self) -> Result<Self, GenerationExhausted> {
        match self.0.checked_add(1) {
            Some(value) => Ok(Self(value)),
            None => Err(GenerationExhausted::TabletEpoch),
        }
    }

    /// Whether `self` strictly supersedes `other` (is newer than it).
    #[must_use]
    pub const fn supersedes(self, other: Self) -> bool {
        self.0 > other.0
    }

    /// Whether `self` is stale relative to `current` (older than it).
    #[must_use]
    pub const fn is_stale_relative_to(self, current: Self) -> bool {
        self.0 < current.0
    }
}

impl fmt::Display for TabletEpoch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Fencing generation bound to one authoritative tablet range.
///
/// Derived from ownership state; every topology transition changes it.
/// Stale writes carrying an older generation are rejected without extra
/// coordination on ordinary reads. `0` is the invalid sentinel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct WriteGuardGeneration(u64);

impl WriteGuardGeneration {
    /// Invalid sentinel. Never authorizes; zeroed memory fails closed.
    pub const INVALID: Self = Self(0);
    /// First valid generation of a newly created tablet range.
    pub const INITIAL: Self = Self(1);

    /// Wraps a raw generation value.
    #[must_use]
    pub const fn from_u64(value: u64) -> Self {
        Self(value)
    }

    /// Returns the raw value.
    #[must_use]
    pub const fn as_u64(self) -> u64 {
        self.0
    }

    /// Whether this is a real generation rather than the invalid sentinel.
    #[must_use]
    pub const fn is_valid(self) -> bool {
        self.0 != 0
    }

    /// Successor generation. Fails with
    /// [`GenerationExhausted::WriteGuardGeneration`] at `u64::MAX` instead of
    /// wrapping or saturating (see [`TabletEpoch::next`]).
    ///
    /// # Errors
    ///
    /// Returns [`GenerationExhausted::WriteGuardGeneration`] if `self` is
    /// already `u64::MAX`.
    pub const fn next(self) -> Result<Self, GenerationExhausted> {
        match self.0.checked_add(1) {
            Some(value) => Ok(Self(value)),
            None => Err(GenerationExhausted::WriteGuardGeneration),
        }
    }

    /// Whether `self` strictly supersedes `other`.
    #[must_use]
    pub const fn supersedes(self, other: Self) -> bool {
        self.0 > other.0
    }

    /// Whether `self` is stale relative to `current`.
    #[must_use]
    pub const fn is_stale_relative_to(self, current: Self) -> bool {
        self.0 < current.0
    }
}

impl fmt::Display for WriteGuardGeneration {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Durable per-process incarnation counter.
///
/// Incremented on every process restart and persisted before the node serves
/// traffic. Peers reject traffic from an older incarnation so a stale
/// restarted process cannot pose as the current node instance. `0` is the
/// invalid sentinel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NodeIncarnation(u64);

impl NodeIncarnation {
    /// Invalid sentinel. Never accepted from a peer.
    pub const INVALID: Self = Self(0);
    /// First incarnation of a freshly admitted node.
    pub const INITIAL: Self = Self(1);

    /// Wraps a raw incarnation value.
    #[must_use]
    pub const fn from_u64(value: u64) -> Self {
        Self(value)
    }

    /// Returns the raw value.
    #[must_use]
    pub const fn as_u64(self) -> u64 {
        self.0
    }

    /// Whether this is a real incarnation rather than the invalid sentinel.
    #[must_use]
    pub const fn is_valid(self) -> bool {
        self.0 != 0
    }

    /// Successor incarnation recorded across a process restart.
    ///
    /// Fails with [`GenerationExhausted::NodeIncarnation`] at `u64::MAX`
    /// instead of wrapping or saturating (see [`TabletEpoch::next`]).
    ///
    /// # Errors
    ///
    /// Returns [`GenerationExhausted::NodeIncarnation`] if `self` is already
    /// `u64::MAX`.
    pub const fn next(self) -> Result<Self, GenerationExhausted> {
        match self.0.checked_add(1) {
            Some(value) => Ok(Self(value)),
            None => Err(GenerationExhausted::NodeIncarnation),
        }
    }

    /// Whether `self` strictly supersedes `other`.
    #[must_use]
    pub const fn supersedes(self, other: Self) -> bool {
        self.0 > other.0
    }

    /// Whether `self` is stale relative to `current` and must be rejected
    /// by peers.
    #[must_use]
    pub const fn is_stale_relative_to(self, current: Self) -> bool {
        self.0 < current.0
    }
}

impl fmt::Display for NodeIncarnation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Position of a mutation within one tablet's ordered log.
///
/// `0` means unassigned (no log slot yet); the first committed entry is `1`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CommitPosition(u64);

impl CommitPosition {
    /// Sentinel for "not yet assigned a log slot".
    pub const UNASSIGNED: Self = Self(0);
    /// First assigned log position.
    pub const FIRST: Self = Self(1);

    /// Wraps a raw position value.
    #[must_use]
    pub const fn from_u64(value: u64) -> Self {
        Self(value)
    }

    /// Returns the raw value.
    #[must_use]
    pub const fn as_u64(self) -> u64 {
        self.0
    }

    /// Whether a real log slot has been assigned.
    #[must_use]
    pub const fn is_assigned(self) -> bool {
        self.0 != 0
    }

    /// Successor position. [`UNASSIGNED`](Self::UNASSIGNED)`/next()` yields
    /// [`FIRST`](Self::FIRST).
    ///
    /// Fails with [`SequenceExhausted::CommitPosition`] at `u64::MAX` instead
    /// of wrapping or saturating: reusing a log slot would alias two distinct
    /// committed mutations.
    ///
    /// # Errors
    ///
    /// Returns [`SequenceExhausted::CommitPosition`] if `self` is already
    /// `u64::MAX`.
    pub const fn next(self) -> Result<Self, SequenceExhausted> {
        match self.0.checked_add(1) {
            Some(value) => Ok(Self(value)),
            None => Err(SequenceExhausted::CommitPosition),
        }
    }
}

impl fmt::Display for CommitPosition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Identity of a client session.
///
/// 128-bit value issued at session establishment. Combined with
/// [`RequestSeq`] it makes every mutation uniquely addressable for
/// lost-reply deduplication. Displayed as 32 lowercase hex digits.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SessionId(u128);

impl SessionId {
    /// Wraps a raw 128-bit value.
    #[must_use]
    pub const fn from_u128(value: u128) -> Self {
        Self(value)
    }

    /// Returns the raw 128-bit value.
    #[must_use]
    pub const fn as_u128(self) -> u128 {
        self.0
    }
}

impl fmt::Display for SessionId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:032x}", self.0)
    }
}

/// Per-session monotonic sequence number.
///
/// Assigned by the client adapter; the core treats (`SessionId`, `RequestSeq`)
/// as an opaque deduplication key: a retry carrying the same pair returns the
/// recorded outcome instead of executing again.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RequestSeq(u64);

impl RequestSeq {
    /// First sequence number of a new session.
    pub const FIRST: Self = Self(0);

    /// Wraps a raw sequence value.
    #[must_use]
    pub const fn from_u64(value: u64) -> Self {
        Self(value)
    }

    /// Returns the raw value.
    #[must_use]
    pub const fn as_u64(self) -> u64 {
        self.0
    }

    /// Successor sequence number.
    ///
    /// Fails with [`SequenceExhausted::RequestSeq`] at `u64::MAX` instead of
    /// wrapping or saturating: saturating would alias every further mutation
    /// in the session to one retained identity, returning stale deduplicated
    /// outcomes for genuinely new mutations and breaking retry safety.
    ///
    /// # Errors
    ///
    /// Returns [`SequenceExhausted::RequestSeq`] if `self` is already
    /// `u64::MAX`.
    pub const fn next(self) -> Result<Self, SequenceExhausted> {
        match self.0.checked_add(1) {
            Some(value) => Ok(Self(value)),
            None => Err(SequenceExhausted::RequestSeq),
        }
    }
}

impl fmt::Display for RequestSeq {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Security domain scoping content-addressed deduplication.
///
/// Deduplication is constrained to one domain; cross-tenant deduplication is
/// disabled by default. There is deliberately no `DEFAULT` value: every chunk
/// derivation must name its domain explicitly. Distinct from [`NamespaceId`]:
/// one tenant may own many namespaces sharing a domain, or isolate each.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SecurityDomainId(u64);

impl SecurityDomainId {
    /// Wraps a raw control-plane-assigned value.
    #[must_use]
    pub const fn from_u64(value: u64) -> Self {
        Self(value)
    }

    /// Returns the raw value.
    #[must_use]
    pub const fn as_u64(self) -> u64 {
        self.0
    }
}

impl fmt::Display for SecurityDomainId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

// Standard conversions between identifiers and their raw values, so generic
// code (codecs, CLIs, tests) can use `Into`/`From` without learning one-off
// accessor names.

/// Implements `From` in both directions for one identifier.
macro_rules! impl_raw_conversions {
    ($type:ty, $raw:ty, $get:ident, $wrap:ident) => {
        impl From<$type> for $raw {
            fn from(id: $type) -> Self {
                id.$get()
            }
        }

        impl From<$raw> for $type {
            fn from(value: $raw) -> Self {
                Self::$wrap(value)
            }
        }
    };
}

impl_raw_conversions!(NodeId, u64, as_u64, from_u64);
impl_raw_conversions!(WorkerId, u64, as_u64, from_u64);
impl_raw_conversions!(NamespaceId, u64, as_u64, from_u64);
impl_raw_conversions!(TabletId, u64, as_u64, from_u64);
impl_raw_conversions!(TabletEpoch, u64, as_u64, from_u64);
impl_raw_conversions!(WriteGuardGeneration, u64, as_u64, from_u64);
impl_raw_conversions!(NodeIncarnation, u64, as_u64, from_u64);
impl_raw_conversions!(CommitPosition, u64, as_u64, from_u64);
impl_raw_conversions!(RequestSeq, u64, as_u64, from_u64);
impl_raw_conversions!(SecurityDomainId, u64, as_u64, from_u64);
impl_raw_conversions!(ClusterId, u128, as_u128, from_u128);
impl_raw_conversions!(SessionId, u128, as_u128, from_u128);

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    /// The fencing types must fail closed at their zero value: none of them
    /// implements `Default`, so the only way to reach an invalid generation is
    /// to name [`INVALID`](TabletEpoch::INVALID) or
    /// [`UNASSIGNED`](CommitPosition::UNASSIGNED) deliberately. A zeroed value
    /// that counted as valid would authorize a write with no fence at all.
    #[test]
    fn zero_is_invalid_for_every_fencing_type() {
        assert!(!TabletEpoch::INVALID.is_valid());
        assert!(!WriteGuardGeneration::INVALID.is_valid());
        assert!(!NodeIncarnation::INVALID.is_valid());
        assert!(TabletEpoch::INITIAL.is_valid());
        assert!(!CommitPosition::UNASSIGNED.is_assigned());
        assert!(CommitPosition::FIRST.is_assigned());
    }

    /// Fencing comparisons are strict: a generation supersedes only a strictly
    /// lower one, and is stale only relative to a strictly higher one. An equal
    /// generation is neither, so a retried request on the same fence is not
    /// mistaken for a superseded one.
    #[test]
    fn fencing_ordering_is_strict() {
        let epoch = TabletEpoch::INITIAL;
        let next = epoch.next().expect("epoch advances");
        assert!(next.supersedes(epoch));
        assert!(epoch.is_stale_relative_to(next));
        assert!(!epoch.supersedes(epoch));
        assert!(!epoch.is_stale_relative_to(epoch));
        assert!(
            WriteGuardGeneration::INITIAL
                .next()
                .expect("guard advances")
                .supersedes(WriteGuardGeneration::INITIAL)
        );
        assert!(
            NodeIncarnation::INITIAL
                .next()
                .expect("incarnation advances")
                .supersedes(NodeIncarnation::INITIAL)
        );
        // The invalid sentinel is stale relative to any real value, so a
        // default-constructed fence can never win a comparison.
        assert!(NodeIncarnation::INVALID.is_stale_relative_to(NodeIncarnation::INITIAL));
    }

    /// Every checked counter must report exhaustion rather than wrap or
    /// saturate: a reused epoch, guard, incarnation, or sequence would fence
    /// the wrong holder. The message must name *which* counter, because the
    /// operator reading it cannot otherwise tell which one to widen.
    #[rstest]
    #[case::epoch(TabletEpoch::from_u64(u64::MAX).next().map(|_| ()), "tablet epoch")]
    #[case::guard(WriteGuardGeneration::from_u64(u64::MAX).next().map(|_| ()), "write-guard generation")]
    #[case::incarnation(NodeIncarnation::from_u64(u64::MAX).next().map(|_| ()), "node incarnation")]
    #[case::request_seq(RequestSeq::from_u64(u64::MAX).next().map(|_| ()), "request sequence")]
    #[case::commit_position(CommitPosition::from_u64(u64::MAX).next().map(|_| ()), "commit position")]
    fn counters_fail_explicitly_at_the_top(
        #[case] actual: Result<(), impl core::fmt::Display>,
        #[case] expected_message: &str,
    ) {
        let message = actual.expect_err("exhaustion must be reported, not wrapped");
        let rendered = message.to_string();
        assert!(
            rendered.contains(expected_message),
            "{rendered:?} does not name {expected_message:?}"
        );
    }

    /// One below the top still advances onto the final value: the check must
    /// reject the overflow, not the last usable counter.
    #[test]
    fn the_penultimate_value_still_advances() {
        assert_eq!(
            TabletEpoch::from_u64(u64::MAX - 1)
                .next()
                .expect("penultimate advances"),
            TabletEpoch::from_u64(u64::MAX)
        );
        assert_eq!(
            RequestSeq::from_u64(u64::MAX - 1)
                .next()
                .expect("penultimate advances"),
            RequestSeq::from_u64(u64::MAX)
        );
    }

    /// `RequestSeq` and `CommitPosition` both start below their first real
    /// value, so a client and a tablet cannot mint a sequence that collides
    /// with a reserved endpoint.
    #[test]
    fn sequence_sentinels_are_below_the_first_value() {
        assert_eq!(RequestSeq::FIRST.next().expect("advances").as_u64(), 1);
        assert_eq!(
            CommitPosition::UNASSIGNED.next().expect("advances"),
            CommitPosition::FIRST
        );
    }
}
