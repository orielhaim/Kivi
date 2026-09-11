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

/// Identity of a whole Kivi cluster (RFC §151, exact control-plane state).
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

/// Identity of a single process/host member of the cluster (RFC §6, §151).
///
/// Assigned by the control plane; never reused across different processes.
/// Combined with [`NodeIncarnation`] to reject stale restarts (RFC §153).
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

/// Identity of a NUMA domain within a node (RFC §6, §15).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NumaId(u32);

impl NumaId {
    /// Wraps a raw OS-reported NUMA node index.
    #[must_use]
    pub const fn from_u32(value: u32) -> Self {
        Self(value)
    }

    /// Returns the raw value.
    #[must_use]
    pub const fn as_u32(self) -> u32 {
        self.0
    }
}

impl fmt::Display for NumaId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Identity of a single CPU within a node (RFC §15).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CpuId(u32);

impl CpuId {
    /// Wraps a raw OS-reported CPU index.
    #[must_use]
    pub const fn from_u32(value: u32) -> Self {
        Self(value)
    }

    /// Returns the raw value.
    #[must_use]
    pub const fn as_u32(self) -> u32 {
        self.0
    }
}

impl fmt::Display for CpuId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Identity of a worker pinned to one CPU (RFC §15).
///
/// A worker owns a set of tablet replicas; tablet state stays with its owner
/// worker (RFC §16, §20). Distinct from [`CpuId`]: CPUs are hardware, workers
/// are execution owners that can be re-pinned.
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

/// Identity of a namespace, the main semantic and policy boundary (RFC §7).
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
/// replication, split, merge, and load accounting (RFC §8).
///
/// Ordered so deterministic choices such as "lowest [`TabletId`] coordinates"
/// (RFC §83) are well-defined.
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

/// Monotonic fencing epoch of a tablet (RFC §70, §194, §251).
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

/// Fencing generation bound to one authoritative tablet range (RFC §70).
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

/// Durable per-process incarnation counter (RFC §153).
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
    /// by peers (RFC §153).
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

/// Position of a mutation within one tablet's ordered log (RFC §167).
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

/// Identity of a client session (RFC §63).
///
/// 128-bit value issued at session establishment. Combined with
/// [`RequestSeq`] it makes every mutation uniquely addressable for
/// lost-reply deduplication (RFC §64). Displayed as 32 lowercase hex digits.
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

/// Per-session monotonic sequence number (RFC §63).
///
/// Assigned by the client adapter; the core treats (`SessionId`, `RequestSeq`)
/// as an opaque deduplication key: a retry carrying the same pair returns the
/// recorded outcome instead of executing again (RFC §64).
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

/// Security domain scoping content-addressed deduplication (RFC §29).
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
impl_raw_conversions!(NumaId, u32, as_u32, from_u32);
impl_raw_conversions!(CpuId, u32, as_u32, from_u32);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn integer_ids_round_trip_through_raw_values() {
        assert_eq!(NodeId::from_u64(7).as_u64(), 7);
        assert_eq!(NumaId::from_u32(1).as_u32(), 1);
        assert_eq!(CpuId::from_u32(11).as_u32(), 11);
        assert_eq!(WorkerId::from_u64(11).as_u64(), 11);
        assert_eq!(NamespaceId::from_u64(3).as_u64(), 3);
        assert_eq!(TabletId::from_u64(918).as_u64(), 918);
        assert_eq!(SecurityDomainId::from_u64(42).as_u64(), 42);
    }

    #[test]
    fn wide_ids_round_trip_through_raw_values() {
        assert_eq!(ClusterId::from_u128(0x1234_5678).as_u128(), 0x1234_5678);
        assert_eq!(SessionId::from_u128(u128::MAX).as_u128(), u128::MAX);
    }

    #[test]
    fn display_formats_match_rfc_examples() {
        // RFC §148 shows decimal tablet/epoch and hex-ish opaque IDs.
        assert_eq!(TabletId::from_u64(918).to_string(), "918");
        assert_eq!(TabletEpoch::from_u64(381).to_string(), "381");
        assert_eq!(
            SessionId::from_u128(0).to_string(),
            "00000000000000000000000000000000"
        );
        assert_eq!(
            ClusterId::from_u128(0xab).to_string(),
            "000000000000000000000000000000ab"
        );
    }

    #[test]
    fn zero_is_invalid_for_fencing_types() {
        assert!(!TabletEpoch::INVALID.is_valid());
        assert!(!WriteGuardGeneration::INVALID.is_valid());
        assert!(!NodeIncarnation::INVALID.is_valid());
        assert!(TabletEpoch::INITIAL.is_valid());
        assert!(!CommitPosition::UNASSIGNED.is_assigned());
        assert!(CommitPosition::FIRST.is_assigned());
    }

    #[test]
    fn fencing_ordering_is_strict() {
        let e1 = TabletEpoch::INITIAL;
        let e2 = e1.next().expect("epoch advances");
        assert!(e2.supersedes(e1));
        assert!(e1.is_stale_relative_to(e2));
        assert!(!e1.supersedes(e1));
        assert!(!e1.is_stale_relative_to(e1));

        let g1 = WriteGuardGeneration::INITIAL;
        assert!(g1.next().expect("guard advances").supersedes(g1));
        let n1 = NodeIncarnation::INITIAL;
        assert!(n1.next().expect("incarnation advances").supersedes(n1));
        assert!(NodeIncarnation::INVALID.is_stale_relative_to(n1));
    }

    #[test]
    fn fencing_successors_fail_explicitly_at_u64_max() {
        assert_eq!(
            TabletEpoch::from_u64(u64::MAX).next(),
            Err(GenerationExhausted::TabletEpoch)
        );
        assert_eq!(
            WriteGuardGeneration::from_u64(u64::MAX).next(),
            Err(GenerationExhausted::WriteGuardGeneration)
        );
        assert_eq!(
            NodeIncarnation::from_u64(u64::MAX).next(),
            Err(GenerationExhausted::NodeIncarnation)
        );
        // One below the top still advances onto the final value.
        assert_eq!(
            TabletEpoch::from_u64(u64::MAX - 1)
                .next()
                .expect("penultimate advances"),
            TabletEpoch::from_u64(u64::MAX)
        );
    }

    #[test]
    fn tablet_ids_order_deterministically_for_coordinator_election() {
        // RFC §83: coordinator chosen deterministically, e.g. lowest TabletId.
        let mut tablets = [
            TabletId::from_u64(9),
            TabletId::from_u64(3),
            TabletId::from_u64(7),
        ];
        tablets.sort();
        assert_eq!(tablets[0], TabletId::from_u64(3));
    }

    #[test]
    fn request_seq_advances_monotonically() {
        let first = RequestSeq::FIRST;
        assert_eq!(first.next().expect("advances").as_u64(), 1);
        assert_eq!(
            CommitPosition::UNASSIGNED.next().expect("advances"),
            CommitPosition::FIRST
        );
    }

    #[test]
    fn sequence_counters_fail_explicitly_at_u64_max() {
        assert_eq!(
            RequestSeq::from_u64(u64::MAX).next(),
            Err(SequenceExhausted::RequestSeq)
        );
        assert_eq!(
            CommitPosition::from_u64(u64::MAX).next(),
            Err(SequenceExhausted::CommitPosition)
        );
        assert_eq!(
            RequestSeq::from_u64(u64::MAX - 1)
                .next()
                .expect("penultimate advances"),
            RequestSeq::from_u64(u64::MAX)
        );
    }

    #[test]
    fn distinct_id_types_do_not_compare_equal() {
        // Compile-time proof by construction: these are different types, so
        // `assert_ne!(TabletId::from_u64(1), WorkerId::from_u64(1))` does not
        // even compile. Runtime spot-check that same-type equality works.
        assert_eq!(TabletId::from_u64(1), TabletId::from_u64(1));
        assert_ne!(TabletId::from_u64(1), TabletId::from_u64(2));
    }
}
