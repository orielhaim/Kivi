//! Cross-tablet OCC + 2PC transaction substrate, V1 (RFC transaction fabric).
//!
//! Transaction V1 provides **serializable point-key mutation sets** across
//! tablets: OCC conflict detection via [`ObjectVersion`] validation plus
//! durable two-phase commit driven by a deterministic coordinator tablet.
//! It is deliberately NOT general transactions: no SQL, no range
//! predicates, no MVCC, no global snapshot reads. Transactional range/predicate
//! forms are rejected with [`TxnError::UnsupportedRange`], never silently
//! weakened.
//!
//! ## Flow
//!
//! ```text
//! coordinator persists Begun (ordinary system-key write)
//!     ↓
//! participants Prepare (Mutation::TxnPrepare via normal Raft propose)
//!     → validate expected ObjectVersions, detect conflicts,
//!       reserve keys with durable intents
//!     ↓ all Prepared
//! coordinator persists Commit decision (system-key write)
//!     ↓
//! participants FinalizeCommit (one Mutation::TxnFinalizeTablet per tablet,
//!                              atomic over that tablet's intents)
//! ```
//!
//! Any prepare failure → coordinator persists Abort → participants
//! `FinalizeAbort`. Only a durable coordinator decision resolves a prepared
//! transaction; participants never unilaterally abort on timeout.
//!
//! ## Identity and idempotency
//!
//! [`TxnId`] is a Kivi-owned 128-bit id derived with BLAKE3 over
//! `(domain, session, sequence, attempt)`: collision-resistant, durable,
//! and never an `OpenRaft` id. Every step (prepare, finalize, record write)
//! is idempotent by `TxnId`: replays overwrite the identical state.
//!
//! ## Coordinator selection
//!
//! The coordinator is the lowest [`TabletId`](kivi_types::TabletId) among
//! participants: deterministic for a given write *set* (independent of
//! request order), so idempotent re-drives and post-crash recovery derive
//! the same coordinator from the same participants with no control-plane
//! involvement in the data path. Single-tablet transactions coordinate on
//! their sole participant but take the cheap local atomic path (one ordered
//! tablet mutation, no 2PC waves); see
//! [`ObjectStore::commit_local`](crate::ObjectStore::commit_local).
//!
//! ## Isolation without MVCC
//!
//! Prepared intents hide: user state is untouched until finalize, and
//! conflicting local writes fail with [`TxnError::Conflict`] while an
//! intent is prepared. Commit finalization is atomic per tablet; across
//! tablets, readers under `LatestPerTablet` may transiently observe commit
//! convergence (documented, no global snapshot claimed). The supported
//! shapes (point reads/writes with exact version evidence, including
//! absence) are serializable; range/predicate isolation is unsupported and
//! rejected loudly.
//!
//! ## Bounds
//!
//! [`MAX_TXN_PARTICIPANTS`], [`MAX_TXN_KEYS`], [`MAX_TXN_BYTES`], and
//! [`MAX_TXN_LIFETIME_MICROS`] bound every transaction; oversize plans are
//! rejected before any intent is written. [`reservation_cost`] lets the
//! admission path reserve intent memory and payload capacity *before*
//! acknowledging prepare, so a prepared transaction can always be finalized.

use std::collections::{BTreeMap, BTreeSet};

use bytes::Bytes;
use kivi_codec::{CodecError, Decode, Encode, decode_byte_vec, encode_bytes};
use kivi_types::{NamespaceId, TabletId};

use crate::object::{Key, ObjectVersion};

/// Maximum participant tablets in one transaction.
pub const MAX_TXN_PARTICIPANTS: usize = 32;
/// Maximum keys in one transaction (sized so a transaction's steps fit one
/// session's retained-outcome window with headroom).
pub const MAX_TXN_KEYS: usize = 256;
/// Maximum total mutation bytes (keys + inline values) in one transaction.
pub const MAX_TXN_BYTES: usize = 1 << 20;
/// Maximum transaction lifetime in micros (leases intents, not a timeout
/// abort: expiry only lets the resolver prefer abort for vanished
/// coordinators after this long with no decision).
pub const MAX_TXN_LIFETIME_MICROS: u64 = 60_000_000;

/// Resolver grace in micros: a decided transaction's intents younger than
/// this are left to their (live) driver. Driver finalize waves complete in
/// milliseconds; converging them from the resolver in that window races
/// the wave's own finalizes through consensus verification (same intent
/// finalized twice from two pre-executions = deterministic-apply
/// divergence, fatal to the node). Only driverless (dead/stalled) intents
/// age past the grace, and only those the resolver converges.
pub const RESOLVE_GRACE_MICROS: u64 = 2_000_000;

/// Kivi-owned transaction identity: `BLAKE3("KIVI-TXN-V1" || session ||
/// seq || attempt)[..16]`, collision-resistant and never an `OpenRaft` id.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TxnId([u8; 16]);

impl TxnId {
    /// Domain tag for id derivation.
    pub const DOMAIN: &'static [u8] = b"KIVI-TXN-V1";

    /// Derives an id from client session, sequence, and attempt number.
    /// Same inputs → same id (transport retry); a new OCC attempt bumps
    /// `attempt` (transaction retry).
    #[must_use]
    pub fn derive(session: u128, seq: u64, attempt: u64) -> Self {
        let mut hasher = blake3::Hasher::new();
        hasher.update(Self::DOMAIN);
        hasher.update(&session.to_le_bytes());
        hasher.update(&seq.to_le_bytes());
        hasher.update(&attempt.to_le_bytes());
        let hash = hasher.finalize();
        let mut id = [0u8; 16];
        id.copy_from_slice(&hash.as_bytes()[..16]);
        Self(id)
    }

    /// Wraps raw id bytes (decode path).
    #[must_use]
    pub const fn from_bytes(bytes: [u8; 16]) -> Self {
        Self(bytes)
    }

    /// Returns the raw id bytes.
    #[must_use]
    pub const fn as_bytes(self) -> [u8; 16] {
        self.0
    }
}

impl core::fmt::Display for TxnId {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        for byte in self.0 {
            write!(f, "{byte:02x}")?;
        }
        Ok(())
    }
}

impl Encode for TxnId {
    fn encoded_len(&self) -> usize {
        16
    }

    fn encode(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.0);
    }
}

impl Decode for TxnId {
    fn decode(input: &[u8]) -> Result<(Self, usize), CodecError> {
        let (raw, consumed) = <[u8; 16]>::decode(input)?;
        Ok((Self::from_bytes(raw), consumed))
    }
}

/// Canonical tags for [`TxnExpect`]. Fixed forever within framing version 1.
const TAG_EXPECT_ANY: u8 = 0;
const TAG_EXPECT_ABSENT: u8 = 1;
const TAG_EXPECT_VERSION: u8 = 2;

impl Encode for TxnExpect {
    fn encoded_len(&self) -> usize {
        match self {
            Self::Any | Self::Absent => 1,
            Self::Version(_) => 1 + 8,
        }
    }

    fn encode(&self, out: &mut Vec<u8>) {
        match self {
            Self::Any => out.push(TAG_EXPECT_ANY),
            Self::Absent => out.push(TAG_EXPECT_ABSENT),
            Self::Version(version) => {
                out.push(TAG_EXPECT_VERSION);
                version.encode(out);
            }
        }
    }
}

impl Decode for TxnExpect {
    fn decode(input: &[u8]) -> Result<(Self, usize), CodecError> {
        let (tag, first) = u8::decode(input)?;
        match tag {
            TAG_EXPECT_ANY => Ok((Self::Any, first)),
            TAG_EXPECT_ABSENT => Ok((Self::Absent, first)),
            TAG_EXPECT_VERSION => {
                let (version, second) = ObjectVersion::decode(&input[first..])?;
                Ok((Self::Version(version), first + second))
            }
            other => Err(CodecError::InvalidTag {
                kind: "txn-expect",
                tag: other,
            }),
        }
    }
}

/// Canonical tags for [`TxnWriteKind`]. Fixed forever within framing v1.
const TAG_WRITE_PUT: u8 = 1;
const TAG_WRITE_DELETE: u8 = 2;
const TAG_WRITE_COUNTER_ADD: u8 = 3;
/// Chunked-put tag: a large value staged before prepare, referenced by
/// manifest. New tags never reuse old ones.
const TAG_WRITE_PUT_CHUNKED: u8 = 4;
/// Commutative-counter add tag. New tags never reuse old ones.
const TAG_WRITE_COMMUTATIVE_ADD: u8 = 5;
/// Bounded-counter add tag. New tags never reuse old ones.
const TAG_WRITE_BOUNDED_ADD: u8 = 6;
/// Escrow share-move tag (one side of a paired transfer).
const TAG_WRITE_ESCROW_SHARE: u8 = 7;
/// Fabric-put tag: a medium value staged in the fabric before prepare,
/// referenced by id. New tags never reuse old ones.
const TAG_WRITE_PUT_FABRIC: u8 = 8;

impl Encode for TxnWriteKind {
    fn encoded_len(&self) -> usize {
        match self {
            Self::Put(value) => 1 + 4 + value.len(),
            Self::Delete => 1,
            Self::CounterAdd(_) | Self::CommutativeAdd(_) | Self::BoundedAdd(_) => 1 + 8,
            Self::PutChunked { .. } => 1 + 32 + 8,
            Self::PutFabric { .. } => 1 + 8 + 8 + 8,
            Self::EscrowSetShare { .. } => 1 + 8 + 8,
        }
    }

    fn encode(&self, out: &mut Vec<u8>) {
        match self {
            Self::Put(value) => {
                out.push(TAG_WRITE_PUT);
                encode_bytes(out, value);
            }
            Self::Delete => out.push(TAG_WRITE_DELETE),
            Self::CounterAdd(delta) => {
                out.push(TAG_WRITE_COUNTER_ADD);
                out.extend_from_slice(&delta.to_le_bytes());
            }
            Self::PutChunked {
                manifest,
                logical_len,
            } => {
                out.push(TAG_WRITE_PUT_CHUNKED);
                out.extend_from_slice(manifest.as_bytes());
                out.extend_from_slice(&logical_len.to_le_bytes());
            }
            Self::PutFabric {
                fabric_id,
                logical_len,
                version,
            } => {
                out.push(TAG_WRITE_PUT_FABRIC);
                out.extend_from_slice(&fabric_id.to_le_bytes());
                out.extend_from_slice(&logical_len.to_le_bytes());
                out.extend_from_slice(&version.to_le_bytes());
            }
            Self::CommutativeAdd(delta) => {
                out.push(TAG_WRITE_COMMUTATIVE_ADD);
                out.extend_from_slice(&delta.to_le_bytes());
            }
            Self::BoundedAdd(delta) => {
                out.push(TAG_WRITE_BOUNDED_ADD);
                out.extend_from_slice(&delta.to_le_bytes());
            }
            Self::EscrowSetShare { min, max } => {
                out.push(TAG_WRITE_ESCROW_SHARE);
                out.extend_from_slice(&min.to_le_bytes());
                out.extend_from_slice(&max.to_le_bytes());
            }
        }
    }
}

impl Decode for TxnWriteKind {
    fn decode(input: &[u8]) -> Result<(Self, usize), CodecError> {
        let (tag, first) = u8::decode(input)?;
        match tag {
            TAG_WRITE_PUT => {
                let (value, second) = decode_byte_vec(&input[first..])?;
                Ok((Self::Put(Bytes::from(value)), first + second))
            }
            TAG_WRITE_DELETE => Ok((Self::Delete, first)),
            TAG_WRITE_COUNTER_ADD => {
                let (raw, second) = <[u8; 8]>::decode(&input[first..])?;
                Ok((Self::CounterAdd(i64::from_le_bytes(raw)), first + second))
            }
            TAG_WRITE_PUT_CHUNKED => {
                let (raw, second) = <[u8; 32]>::decode(&input[first..])?;
                let (len_raw, third) = <[u8; 8]>::decode(&input[first + second..])?;
                Ok((
                    Self::PutChunked {
                        manifest: kivi_types::ManifestId::from_bytes(raw),
                        logical_len: u64::from_le_bytes(len_raw),
                    },
                    first + second + third,
                ))
            }
            TAG_WRITE_PUT_FABRIC => {
                let (id_raw, second) = <[u8; 8]>::decode(&input[first..])?;
                let (len_raw, third) = <[u8; 8]>::decode(&input[first + second..])?;
                let (ver_raw, fourth) = <[u8; 8]>::decode(&input[first + second + third..])?;
                Ok((
                    Self::PutFabric {
                        fabric_id: u64::from_le_bytes(id_raw),
                        logical_len: u64::from_le_bytes(len_raw),
                        version: u64::from_le_bytes(ver_raw),
                    },
                    first + second + third + fourth,
                ))
            }
            TAG_WRITE_COMMUTATIVE_ADD => {
                let (raw, second) = <[u8; 8]>::decode(&input[first..])?;
                Ok((
                    Self::CommutativeAdd(i64::from_le_bytes(raw)),
                    first + second,
                ))
            }
            TAG_WRITE_BOUNDED_ADD => {
                let (raw, second) = <[u8; 8]>::decode(&input[first..])?;
                Ok((Self::BoundedAdd(i64::from_le_bytes(raw)), first + second))
            }
            TAG_WRITE_ESCROW_SHARE => {
                let (min_raw, second) = <[u8; 8]>::decode(&input[first..])?;
                let (max_raw, third) = <[u8; 8]>::decode(&input[first + second..])?;
                Ok((
                    Self::EscrowSetShare {
                        min: i64::from_le_bytes(min_raw),
                        max: i64::from_le_bytes(max_raw),
                    },
                    first + second + third,
                ))
            }
            other => Err(CodecError::InvalidTag {
                kind: "txn-write",
                tag: other,
            }),
        }
    }
}

/// One point write inside a transaction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TxnWrite {
    /// Target key.
    pub key: Key,
    /// Write kind.
    pub kind: TxnWriteKind,
    /// OCC expectation: `None` = blind write, `Some(v)` = abort unless the
    /// live version is exactly `v` (`Some(FIRST)` with no live object fails:
    /// use `None`-absent encoding below instead).
    pub expect: TxnExpect,
}

/// Version expectation for one transactional write.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TxnExpect {
    /// No expectation (blind write, still conflicts with prepared intents).
    Any,
    /// Key must be absent (live).
    Absent,
    /// Live version must equal this exactly.
    Version(ObjectVersion),
}

/// Transactional write kinds (V1: deterministic point ops only).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TxnWriteKind {
    /// Store bytes (SET semantics: overwrite, clear expiry).
    Put(Bytes),
    /// Store a staged large value by chunk reference (the chunked spelling
    /// of [`Put`](Self::Put)). The engine stages every referenced chunk and
    /// proves pack durability *before* prepare, so a later Commit can never
    /// meet a durable root with unavailable payload.
    PutChunked {
        /// Manifest addressing the immutable chunk sequence.
        manifest: kivi_types::ManifestId,
        /// Total logical bytes across the manifest.
        logical_len: u64,
    },
    /// Store a staged medium value by fabric reference (the fabric
    /// spelling of [`Put`](Self::Put)). The engine stages the bytes into
    /// the fabric and proves the materialization *before* prepare, so a
    /// later Commit can never meet a durable root with unavailable
    /// payload.
    PutFabric {
        /// Fabric-scoped object id assigned at staging.
        fabric_id: u64,
        /// Total logical bytes of the materialized value.
        logical_len: u64,
        /// Logical version the bytes were published at (pins reads).
        version: u64,
    },
    /// Remove the key.
    Delete,
    /// Add to a strict counter (prepared deterministically: overflow fails at
    /// prepare, identically on every replica).
    CounterAdd(i64),
    /// Add to a commutative counter (order-free; no ordinal is produced).
    CommutativeAdd(i64),
    /// Add to a bounded counter (escrow rights checked at prepare against
    /// the locally owned share, identically on every replica).
    BoundedAdd(i64),
    /// Move one bounded counter's escrow share to `[min, max]` (one side of
    /// a paired rights transfer). Well-formedness (value covered, inside
    /// `[0, capacity]`) is participant-checked; conservation across the pair
    /// is driver-checked with [`verify_escrow_widths`] before any intent
    /// prepares, so a half transfer never reserves.
    EscrowSetShare {
        /// New share lower bound (inclusive).
        min: i64,
        /// New share upper bound (inclusive).
        max: i64,
    },
}

impl TxnWrite {
    /// Estimated wire/intent bytes for bounding.
    #[must_use]
    pub fn estimated_bytes(&self) -> usize {
        let payload = match &self.kind {
            TxnWriteKind::Put(value) => value.len(),
            TxnWriteKind::PutChunked { .. } => 40,
            TxnWriteKind::PutFabric { .. } => 24,
            TxnWriteKind::Delete
            | TxnWriteKind::CounterAdd(_)
            | TxnWriteKind::CommutativeAdd(_)
            | TxnWriteKind::BoundedAdd(_)
            | TxnWriteKind::EscrowSetShare { .. } => 8,
        };
        self.key.len() + payload
    }
}

/// Durable coordinator decision record (persisted as an ordinary object
/// under a `\xff` system key on the coordinator tablet; see
/// [`txn_record_key`](crate::txn_record_key)).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TxnRecord {
    /// Transaction identity.
    pub id: TxnId,
    /// Coordinator tablet (persists this record).
    pub coordinator: TabletId,
    /// Participant tablets.
    pub participants: Vec<TabletId>,
    /// Current durable state.
    pub state: TxnState,
    /// Directory version the plan was routed against (fencing).
    pub dir_version: u64,
    /// Digest over the full write set (participants verify they were
    /// prepared for this exact transaction, never a confused deputy).
    pub digest: [u8; 32],
}

/// Durable coordinator decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TxnState {
    /// Intents may prepare; no outcome decided yet.
    Begun,
    /// All prepared: participants must finalize commit.
    Committed,
    /// Aborted: participants must discard intents.
    Aborted,
}

impl TxnState {
    /// Whether this state is a terminal decision.
    #[must_use]
    pub const fn is_decided(self) -> bool {
        match self {
            Self::Begun => false,
            Self::Committed | Self::Aborted => true,
        }
    }

    /// Applies a decision transition, enforcing the irreversible state
    /// machine structurally:
    ///
    /// ```text
    /// Begun → Committed | Begun → Aborted   (decide once)
    /// Begun → Begun, Committed → Committed, Aborted → Aborted (replay)
    /// Committed ⇄ Aborted                    (corruption, rejected)
    /// Committed/Aborted → Begun              (resurrection, rejected)
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`TxnError::DecisionConflict`] when `next` contradicts a
    /// durably decided outcome: conflicting final decisions are corruption,
    /// never a tie to break.
    pub const fn transition_to(self, next: Self) -> Result<Self, TxnError> {
        match (self, next) {
            (Self::Begun, Self::Begun)
            | (Self::Committed, Self::Committed)
            | (Self::Aborted, Self::Aborted) => Ok(self),
            (Self::Begun, decided @ (Self::Committed | Self::Aborted)) => Ok(decided),
            (Self::Committed, Self::Aborted)
            | (Self::Aborted, Self::Committed)
            | (Self::Committed | Self::Aborted, Self::Begun) => Err(TxnError::DecisionConflict),
        }
    }
}

/// One transaction participant: the tablet that must prepare and finalize.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TxnParticipant {
    /// Participant tablet.
    pub tablet: TabletId,
    /// Directory version the plan was routed against (fencing: a prepare
    /// against a different lineage is rejected, never remapped).
    pub dir_version: u64,
}

/// The deterministic coordinator: the lowest participant [`TabletId`].
/// Derivable after any failure from the participant set alone — never a
/// service, never control-plane state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TxnCoordinator {
    /// Coordinator tablet (owns the durable decision record).
    pub tablet: TabletId,
}

impl TxnCoordinator {
    /// Selects the coordinator for a participant set: the lowest tablet.
    #[must_use]
    pub fn select(participants: &BTreeSet<TabletId>) -> Option<Self> {
        participants.first().map(|tablet| Self { tablet: *tablet })
    }
}

/// Exact validation evidence captured by one transactional read: the live
/// version observed, or observed absence. Absence is evidence — a missing
/// key created concurrently must fail validation, or lost-create anomalies
/// become possible.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TxnReadVersion {
    /// The key was present at this version.
    Present(ObjectVersion),
    /// The key was absent.
    Absent,
}

/// One participant's prepare request: the exact mutation it must reserve,
/// with the read evidence it was validated against.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TxnPrepare {
    /// Transaction identity.
    pub id: TxnId,
    /// Coordinator owning the decision record.
    pub coordinator: TxnCoordinator,
    /// Participant being prepared.
    pub participant: TxnParticipant,
    /// Reserved write.
    pub write: TxnWrite,
    /// Read evidence captured at validation time.
    pub read: TxnReadVersion,
    /// Digest over the full write set (the participant verifies it was
    /// prepared for this exact transaction, never a confused deputy).
    pub digest: [u8; 32],
}

/// The durable coordinator decision: the single source of truth for the
/// transaction's outcome. Replicated as an ordinary system-key write on the
/// coordinator tablet, so it survives coordinator crash, participant crash,
/// and full-cluster restart.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TxnDecision {
    /// Commit: every participant must apply its prepared write.
    Commit,
    /// Abort: every participant must discard its prepared write.
    Abort,
}

impl TxnDecision {
    /// Maps the decision to its durable record state.
    #[must_use]
    pub const fn state(self) -> TxnState {
        match self {
            Self::Commit => TxnState::Committed,
            Self::Abort => TxnState::Aborted,
        }
    }
}

/// The durable, replayable observable outcome of a transaction: what a
/// client retry after a lost reply receives instead of re-executing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TxnOutcome {
    /// Committed: resulting versions in request order (`None` for deletes,
    /// which carry no version).
    Committed {
        /// Resulting versions in request order.
        versions: Vec<Option<ObjectVersion>>,
    },
    /// Aborted with the terminal reason.
    Aborted {
        /// Why the transaction aborted.
        reason: TxnError,
    },
}

/// One participant's durable intent: the prepared write plus the version
/// it was validated against, bound to the exact write-set digest it was
/// prepared for (a different digest under the same `TxnId` is a conflicting
/// driver, never a retry — rejected loudly, never overwritten silently).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TxnIntent {
    /// Transaction identity.
    pub id: TxnId,
    /// Coordinator tablet (owns the decision record).
    pub coordinator: TabletId,
    /// Reserved key.
    pub key: Key,
    /// Prepared write.
    pub write: TxnWrite,
    /// Digest over the full write set this intent was prepared for.
    pub digest: [u8; 32],
    /// Live version observed at prepare (`None` = absent).
    pub observed: Option<ObjectVersion>,
    /// Prepare timestamp (leader-materialized; resolver use only).
    pub prepared_at: kivi_types::WallTimestamp,
}

/// Transaction failure (Kivi-owned; never leaks Raft/storage internals).
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum TxnError {
    /// OCC validation failed (version moved) or a prepared intent / live
    /// owner blocks the key.
    #[error("transaction conflict")]
    Conflict,
    /// Unique index term already owned by another primary.
    #[error("unique index violation")]
    UniqueViolation,
    /// Transaction exceeds participant/key/byte bounds.
    #[error("transaction too large")]
    TooLarge,
    /// Coordinator record unreachable (participant cannot resolve yet;
    /// retry, never guess).
    #[error("transaction coordinator unavailable")]
    CoordinatorUnavailable,
    /// Routing changed mid-transaction (directory fence); replan.
    #[error("transaction routing changed; replan")]
    RoutingChanged,
    /// Transaction aborted (prepare rejected or coordinator decided abort).
    #[error("transaction aborted")]
    Aborted,
    /// A batch write is structurally invalid (unknown kind/expectation).
    #[error("invalid transaction write")]
    InvalidWrite,
    /// Conflicting final decisions for one transaction (durable corruption:
    /// both Commit and Abort recorded). Never resolved by guessing.
    #[error("transaction decision conflict: both commit and abort recorded")]
    DecisionConflict,
    /// Transactional range/predicate reads are unsupported: the point-key
    /// OCC mechanism cannot protect them, so they are rejected rather than
    /// silently offering weaker isolation.
    #[error("transactional range reads are unsupported")]
    UnsupportedRange,
}

impl TxnError {
    /// Whether the client may retry with a fresh attempt number.
    #[must_use]
    pub const fn is_retryable(self) -> bool {
        match self {
            Self::Conflict | Self::RoutingChanged | Self::CoordinatorUnavailable => true,
            Self::UniqueViolation
            | Self::TooLarge
            | Self::Aborted
            | Self::InvalidWrite
            | Self::DecisionConflict
            | Self::UnsupportedRange => false,
        }
    }
}

/// Validates transaction bounds before any intent is written.
///
/// # Errors
///
/// Returns [`TxnError::TooLarge`] when participants, keys, or bytes exceed
/// [`MAX_TXN_PARTICIPANTS`], [`MAX_TXN_KEYS`], or [`MAX_TXN_BYTES`].
pub fn check_bounds(participants: usize, writes: &[TxnWrite]) -> Result<(), TxnError> {
    if participants > MAX_TXN_PARTICIPANTS || writes.len() > MAX_TXN_KEYS {
        return Err(TxnError::TooLarge);
    }
    let bytes: usize = writes.iter().map(TxnWrite::estimated_bytes).sum();
    if bytes > MAX_TXN_BYTES {
        return Err(TxnError::TooLarge);
    }
    Ok(())
}

/// Computes the write-set digest covered by the coordinator record.
///
/// The digest binds the EFFECTIVE set, not its presentation: duplicate keys
/// collapse last-wins (matching the intent-overwrite rule and
/// [`ObjectStore::commit_local`](crate::store::ObjectStore::commit_local)),
/// then keys sort. Replicas, re-drives, and recovery probes presenting the
/// same transaction in any order — duplicates included — bind the same
/// digest, so a conflicting digest under one [`TxnId`] always means a
/// conflicting transaction, never a reshuffled one.
///
/// Expectations ([`TxnExpect`]) are deliberately unhashed: every prepare
/// re-validates its expectation against live state, so a re-presented
/// expectation is re-checked, never trusted from the digest.
#[must_use]
pub fn write_set_digest(writes: &[TxnWrite], namespace: NamespaceId) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"KIVI-TXN-SET-V1");
    hasher.update(&namespace.as_u64().to_le_bytes());
    let mut collapsed: BTreeMap<&[u8], &TxnWrite> = BTreeMap::new();
    for write in writes {
        collapsed.insert(write.key.as_bytes(), write);
    }
    for write in collapsed.values() {
        hasher.update(write.key.as_bytes());
        match &write.kind {
            TxnWriteKind::Put(value) => {
                hasher.update(&[1]);
                hasher.update(value);
            }
            TxnWriteKind::PutChunked {
                manifest,
                logical_len,
            } => {
                hasher.update(&[4]);
                hasher.update(manifest.as_bytes());
                hasher.update(&logical_len.to_le_bytes());
            }
            TxnWriteKind::PutFabric {
                fabric_id,
                logical_len,
                version,
            } => {
                hasher.update(&[8]);
                hasher.update(&fabric_id.to_le_bytes());
                hasher.update(&logical_len.to_le_bytes());
                hasher.update(&version.to_le_bytes());
            }
            TxnWriteKind::Delete => {
                hasher.update(&[2]);
            }
            TxnWriteKind::CounterAdd(delta) => {
                hasher.update(&[3]);
                hasher.update(&delta.to_le_bytes());
            }
            TxnWriteKind::CommutativeAdd(delta) => {
                hasher.update(&[5]);
                hasher.update(&delta.to_le_bytes());
            }
            TxnWriteKind::BoundedAdd(delta) => {
                hasher.update(&[6]);
                hasher.update(&delta.to_le_bytes());
            }
            TxnWriteKind::EscrowSetShare { min, max } => {
                hasher.update(&[7]);
                hasher.update(&min.to_le_bytes());
                hasher.update(&max.to_le_bytes());
            }
        }
    }
    *hasher.finalize().as_bytes()
}

/// Cost of reserving `writes` as prepared intents: intent memory plus write
/// payload capacity. Admission reserves this *before* acknowledging prepare,
/// so `prepared → commit → impossible local apply` (OOM, payload capacity)
/// is structurally unreachable rather than a runtime surprise.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TxnReservation {
    /// Intent-table entries required (one per key).
    pub intents: usize,
    /// Payload bytes required (keys + inline values + chunk roots).
    pub payload_bytes: usize,
}

/// Computes the reservation cost of a write set.
///
/// # Errors
///
/// Returns [`TxnError::TooLarge`] when the set exceeds
/// [`MAX_TXN_KEYS`] or [`MAX_TXN_BYTES`].
pub fn reservation_cost(writes: &[TxnWrite]) -> Result<TxnReservation, TxnError> {
    if writes.len() > MAX_TXN_KEYS {
        return Err(TxnError::TooLarge);
    }
    let payload_bytes: usize = writes.iter().map(TxnWrite::estimated_bytes).sum();
    if payload_bytes > MAX_TXN_BYTES {
        return Err(TxnError::TooLarge);
    }
    Ok(TxnReservation {
        intents: writes.len(),
        payload_bytes,
    })
}

/// Verifies a paired escrow transfer against live shares: `donor` narrows
/// by exactly `amount` and `recipient` widens by exactly `amount`, keeping
/// total width constant. Pure: inputs are the four share bounds. Used by
/// the same-tablet atomic transfer (both objects visible) and, later, by
/// cross-tablet transfer drivers once the geography dimension lands —
/// the per-side moves are already typed for it.
///
/// # Errors
///
/// Returns [`TxnError::InvalidWrite`] when the widths do not conserve.
pub fn verify_escrow_widths(
    donor_before: (i64, i64),
    donor_after: (i64, i64),
    recipient_before: (i64, i64),
    recipient_after: (i64, i64),
) -> Result<(), TxnError> {
    let width = |(min, max): (i64, i64)| -> i128 { i128::from(max) - i128::from(min) };
    let before = width(donor_before) + width(recipient_before);
    let after = width(donor_after) + width(recipient_after);
    if before == after {
        Ok(())
    } else {
        Err(TxnError::InvalidWrite)
    }
}

/// A planned paired escrow move: the donor's and recipient's new shares.
/// Width is conserved by construction (see [`verify_escrow_widths`]).
pub type PlannedEscrowMove = ((i64, i64), (i64, i64));

/// Plans a paired escrow transfer from read evidence: given both live
/// shares (captured by reads, revalidated at prepare) plus the live values,
/// computes the donor/recipient share moves shifting `amount` of rights.
/// The pair conserves by construction ([`verify_escrow_widths`] holds), so a
/// driver preparing both moves can never mint or destroy rights.
///
/// # Errors
///
/// Returns [`TxnError::InvalidWrite`] for a non-positive amount,
/// [`TxnError::Conflict`] when the donor cannot cover `amount` past its
/// live value (retry after rights move back), or [`TxnError::InvalidWrite`]
/// when the recipient would exceed its counter capacity.
pub fn plan_escrow_transfer(
    donor_share: (i64, i64),
    donor_value: i64,
    recipient_share: (i64, i64),
    recipient_capacity: u64,
    amount: u64,
) -> Result<PlannedEscrowMove, TxnError> {
    let amount = i64::try_from(amount).map_err(|_| TxnError::InvalidWrite)?;
    if amount <= 0 {
        return Err(TxnError::InvalidWrite);
    }
    let (donor_min, donor_max) = donor_share;
    let (recipient_min, recipient_max) = recipient_share;
    let donor_new_max = donor_max
        .checked_sub(amount)
        .ok_or(TxnError::InvalidWrite)?;
    if donor_new_max < donor_min || donor_value > donor_new_max {
        return Err(TxnError::Conflict);
    }
    let recipient_new_max = recipient_max
        .checked_add(amount)
        .ok_or(TxnError::InvalidWrite)?;
    let Ok(capacity) = i64::try_from(recipient_capacity) else {
        return Err(TxnError::InvalidWrite);
    };
    if recipient_new_max > capacity {
        return Err(TxnError::InvalidWrite);
    }
    Ok((
        (donor_min, donor_new_max),
        (recipient_min, recipient_new_max),
    ))
}

/// Groups writes by participant tablet for 2PC fan-out.
///
/// # Errors
///
/// Returns [`TxnError::RoutingChanged`] when any key routes nowhere (gap),
/// so half the transaction never uses stale routing.
pub fn group_by_tablet(
    writes: &[TxnWrite],
    route: impl Fn(&[u8]) -> Option<TabletId>,
) -> Result<BTreeMap<TabletId, Vec<usize>>, TxnError> {
    let mut groups: BTreeMap<TabletId, Vec<usize>> = BTreeMap::new();
    for (index, write) in writes.iter().enumerate() {
        let tablet = route(write.key.as_bytes()).ok_or(TxnError::RoutingChanged)?;
        groups.entry(tablet).or_default().push(index);
    }
    Ok(groups)
}

/// Selects the deterministic coordinator: the lowest [`TabletId`] among the
/// write set's tablets. Order-independent (unlike first-write selection),
/// so idempotent re-drives, recovery probes, and conflicting drivers all
/// derive the same coordinator from the same participant set.
#[must_use]
pub fn select_coordinator(
    writes: &[TxnWrite],
    route: impl Fn(&[u8]) -> Option<TabletId>,
) -> Option<TabletId> {
    writes
        .iter()
        .filter_map(|write| route(write.key.as_bytes()))
        .min()
}

/// Distinct tablets touched by `writes` (introspection for tests/metrics).
#[must_use]
pub fn participant_set(groups: &BTreeMap<TabletId, Vec<usize>>) -> BTreeSet<TabletId> {
    groups.keys().copied().collect()
}

/// Builds one transactional write from wire fields (the protocol validates
/// tags at decode; unknown tags here mean a peer bug, never user input).
///
/// # Errors
///
/// Returns [`TxnError::InvalidWrite`] for unknown kinds or expectations.
pub fn txn_write_from_wire(
    key: Vec<u8>,
    kind: u8,
    value: Vec<u8>,
    delta: i64,
    expect: u8,
    expect_version: u64,
) -> Result<TxnWrite, TxnError> {
    let kind = match kind {
        // Wire tags mirror `BATCH_*` in kivi-protocol (1/2/3); matched
        // numerically so kivi-state never depends on the protocol crate.
        // 4/5/6 extend the same space for chunked and semantic counters.
        // There is deliberately no wire tag for staged fabric values:
        // fabric ids are worker-local names and must never cross the
        // wire (staging happens past the boundary, on the owner).
        1 => TxnWriteKind::Put(Bytes::from(value)),
        2 => TxnWriteKind::Delete,
        3 => TxnWriteKind::CounterAdd(delta),
        4 => {
            if value.len() != 40 {
                return Err(TxnError::InvalidWrite);
            }
            let mut manifest = [0u8; 32];
            manifest.copy_from_slice(&value[..32]);
            let mut len = [0u8; 8];
            len.copy_from_slice(&value[32..]);
            TxnWriteKind::PutChunked {
                manifest: kivi_types::ManifestId::from_bytes(manifest),
                logical_len: u64::from_le_bytes(len),
            }
        }
        5 => TxnWriteKind::CommutativeAdd(delta),
        6 => TxnWriteKind::BoundedAdd(delta),
        7 => {
            if value.len() != 16 {
                return Err(TxnError::InvalidWrite);
            }
            let mut min = [0u8; 8];
            let mut max = [0u8; 8];
            min.copy_from_slice(&value[..8]);
            max.copy_from_slice(&value[8..]);
            TxnWriteKind::EscrowSetShare {
                min: i64::from_le_bytes(min),
                max: i64::from_le_bytes(max),
            }
        }
        _ => return Err(TxnError::InvalidWrite),
    };
    let expect = match expect {
        0 => TxnExpect::Any,
        1 => TxnExpect::Absent,
        2 => TxnExpect::Version(ObjectVersion::from_u64(expect_version)),
        _ => return Err(TxnError::InvalidWrite),
    };
    Ok(TxnWrite {
        key: Key::from(key),
        kind,
        expect,
    })
}

/// Validated, routable 2PC plan shared by every driver (embedded, engine
/// connection, cluster coordinator, client-driven): bounds checked,
/// participants grouped, coordinator selected, record built. Drivers differ
/// only in how they propose each step, never in what the steps are.
#[derive(Debug, Clone)]
pub struct TxnDriverPlan {
    /// Transaction identity.
    pub txn: TxnId,
    /// Writes in request order.
    pub writes: Vec<TxnWrite>,
    /// Participant tablet → write indexes (routing order).
    pub groups: BTreeMap<TabletId, Vec<usize>>,
    /// Deterministic coordinator (lowest participant tablet).
    pub coordinator: TabletId,
    /// Coordinator record to persist first (`Begun`).
    pub record: TxnRecord,
}

/// Builds a validated driver plan: bounds, grouping, coordinator, record.
///
/// # Errors
///
/// Returns [`TxnError::TooLarge`] for oversize plans,
/// [`TxnError::RoutingChanged`] when any key routes nowhere, or
/// [`TxnError::InvalidWrite`]... never here (writes are already typed);
/// empty write sets are rejected as too large (nothing to commit).
pub fn plan_transaction(
    txn: TxnId,
    writes: Vec<TxnWrite>,
    route: impl Fn(&[u8]) -> Option<TabletId>,
    dir_version: u64,
    namespace: NamespaceId,
) -> Result<TxnDriverPlan, TxnError> {
    if writes.is_empty() {
        return Err(TxnError::TooLarge);
    }
    let groups = group_by_tablet(&writes, &route)?;
    let coordinator = select_coordinator(&writes, &route).ok_or(TxnError::RoutingChanged)?;
    check_bounds(groups.len(), &writes)?;
    let participants: Vec<TabletId> = groups.keys().copied().collect();
    let digest = write_set_digest(&writes, namespace);
    Ok(TxnDriverPlan {
        txn,
        writes,
        groups,
        coordinator,
        record: TxnRecord {
            id: txn,
            coordinator,
            participants,
            state: TxnState::Begun,
            dir_version,
            digest,
        },
    })
}

/// Record encoding version for [`TxnRecord`].
pub const TXN_RECORD_VERSION: u16 = 1;

/// Length of a well-formed transaction record key in bytes:
/// `1 + 9 + 8 + 16` (`\xff`, `kivi/txn/`, coordinator BE64, txn id).
pub const TXN_RECORD_KEY_LEN: usize = 34;

/// Record-key prefix after the system byte: `kivi/txn/`.
pub const TXN_RECORD_PREFIX: &[u8; 9] = b"kivi/txn/";

/// Parses a transaction record key into `(coordinator, txn)`.
///
/// Returns `None` for ordinary keys (fast path: length + prefix checks
/// first, so user traffic pays two comparisons). Callers route parsed
/// keys to the embedded tablet while it is active, else normally.
#[must_use]
pub fn parse_txn_record_key(key: &[u8]) -> Option<(TabletId, TxnId)> {
    if key.len() != TXN_RECORD_KEY_LEN {
        return None;
    }
    if key[0] != crate::object::Key::SYSTEM_PREFIX || &key[1..10] != TXN_RECORD_PREFIX {
        return None;
    }
    let coordinator =
        TabletId::from_u64(u64::from_be_bytes(key[10..18].try_into().unwrap_or([0; 8])));
    let mut id = [0u8; 16];
    id.copy_from_slice(&key[18..34]);
    Some((coordinator, TxnId::from_bytes(id)))
}

impl TxnRecord {
    /// Records a decision transition on a loaded record, enforcing the
    /// irreversible machine: committing a begun record is final, aborting
    /// one is final, replaying the same decision is idempotent, and
    /// contradicting a decided record is corruption.
    ///
    /// # Errors
    ///
    /// Returns [`TxnError::DecisionConflict`] on a conflicting final
    /// decision or a resurrection to [`TxnState::Begun`].
    pub fn decide(&mut self, decision: TxnDecision) -> Result<(), TxnError> {
        self.state = self.state.transition_to(decision.state())?;
        Ok(())
    }
    /// Encodes the record value stored under [`crate::txn_record_key`].
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&TXN_RECORD_VERSION.to_le_bytes());
        out.extend_from_slice(&self.id.as_bytes());
        out.extend_from_slice(&self.coordinator.as_u64().to_le_bytes());
        out.push(match self.state {
            TxnState::Begun => 0,
            TxnState::Committed => 1,
            TxnState::Aborted => 2,
        });
        out.extend_from_slice(&self.dir_version.to_le_bytes());
        out.extend_from_slice(
            &u32::try_from(self.participants.len())
                .unwrap_or(u32::MAX)
                .to_le_bytes(),
        );
        for participant in &self.participants {
            out.extend_from_slice(&participant.as_u64().to_le_bytes());
        }
        out.extend_from_slice(&self.digest);
        out
    }

    /// Decodes a record value, requiring exact consumption.
    ///
    /// # Errors
    ///
    /// Returns [`TxnDecodeError`] on structural failure.
    pub fn decode(input: &[u8]) -> Result<Self, TxnDecodeError> {
        use TxnDecodeError as Fault;
        if input.len() < 2 + 16 + 8 + 1 + 8 + 4 + 32 {
            return Err(Fault::Truncated);
        }
        if u16::from_le_bytes([input[0], input[1]]) != TXN_RECORD_VERSION {
            return Err(Fault::BadVersion);
        }
        let mut id = [0u8; 16];
        id.copy_from_slice(&input[2..18]);
        let coordinator = TabletId::from_u64(u64::from_le_bytes(
            input[18..26].try_into().map_err(|_| Fault::Truncated)?,
        ));
        let state = match input[26] {
            0 => TxnState::Begun,
            1 => TxnState::Committed,
            2 => TxnState::Aborted,
            _ => return Err(Fault::BadState),
        };
        let dir_version =
            u64::from_le_bytes(input[27..35].try_into().map_err(|_| Fault::Truncated)?);
        let count =
            u32::from_le_bytes(input[35..39].try_into().map_err(|_| Fault::Truncated)?) as usize;
        if count > MAX_TXN_PARTICIPANTS + 1 {
            return Err(Fault::Oversized);
        }
        if input.len() != 39 + count * 8 + 32 {
            return Err(Fault::Truncated);
        }
        let mut participants = Vec::with_capacity(count);
        for index in 0..count {
            let at = 39 + index * 8;
            participants.push(TabletId::from_u64(u64::from_le_bytes(
                input[at..at + 8].try_into().map_err(|_| Fault::Truncated)?,
            )));
        }
        let mut digest = [0u8; 32];
        digest.copy_from_slice(&input[39 + count * 8..]);
        Ok(Self {
            id: TxnId::from_bytes(id),
            coordinator,
            participants,
            state,
            dir_version,
            digest,
        })
    }
}

/// Transaction record decode failure (structural only).
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum TxnDecodeError {
    /// Truncated input where framing promised bytes.
    #[error("truncated transaction record")]
    Truncated,
    /// Unknown record version.
    #[error("unsupported transaction record version")]
    BadVersion,
    /// Unknown decision state.
    #[error("unknown transaction state")]
    BadState,
    /// Absurd participant count (never allocated blindly).
    #[error("oversize transaction record")]
    Oversized,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write(key: &str) -> TxnWrite {
        TxnWrite {
            key: Key::from(key),
            kind: TxnWriteKind::Put(Bytes::from_static(b"v")),
            expect: TxnExpect::Any,
        }
    }

    #[test]
    fn txn_ids_derive_deterministically() {
        assert_eq!(TxnId::derive(7, 9, 0), TxnId::derive(7, 9, 0));
        assert_ne!(TxnId::derive(7, 9, 0), TxnId::derive(7, 9, 1));
        assert_ne!(TxnId::derive(7, 9, 0), TxnId::derive(8, 9, 0));
    }

    #[test]
    fn bounds_reject_oversize_plans() {
        assert!(check_bounds(33, &[]).is_err());
        let many: Vec<TxnWrite> = (0..513).map(|i| write(&format!("k{i}"))).collect();
        assert_eq!(check_bounds(1, &many), Err(TxnError::TooLarge));
        let big = TxnWrite {
            key: Key::from("k"),
            kind: TxnWriteKind::Put(Bytes::from(vec![0u8; MAX_TXN_BYTES])),
            expect: TxnExpect::Any,
        };
        assert_eq!(check_bounds(1, &[big]), Err(TxnError::TooLarge));
        assert!(check_bounds(2, &[write("a"), write("b")]).is_ok());
    }

    #[test]
    fn grouping_and_coordinator_are_deterministic() {
        let writes = vec![write("b"), write("a"), write("c")];
        let route = |key: &[u8]| {
            if key < b"b".as_slice() {
                Some(TabletId::from_u64(1))
            } else {
                Some(TabletId::from_u64(2))
            }
        };
        let groups = group_by_tablet(&writes, route).expect("groups");
        assert_eq!(groups.len(), 2);
        // Coordinator is the lowest participant tablet (order-independent:
        // any permutation of the same write set selects tablet 1).
        assert_eq!(
            select_coordinator(&writes, route),
            Some(TabletId::from_u64(1))
        );
        let reversed = vec![write("c"), write("b"), write("a")];
        assert_eq!(
            select_coordinator(&reversed, route),
            select_coordinator(&writes, route)
        );
        // Gap routes nowhere → fence error, never a half plan.
        let gap = |_: &[u8]| None;
        assert_eq!(group_by_tablet(&writes, gap), Err(TxnError::RoutingChanged));
    }

    #[test]
    fn digest_is_order_independent() {
        let ns = NamespaceId::from_u64(3);
        let forward = vec![write("a"), write("b")];
        let backward = vec![write("b"), write("a")];
        assert_eq!(
            write_set_digest(&forward, ns),
            write_set_digest(&backward, ns)
        );
    }

    #[test]
    fn state_machine_is_irreversible() {
        use TxnState as State;
        assert_eq!(
            State::Begun.transition_to(State::Committed),
            Ok(State::Committed)
        );
        assert_eq!(
            State::Begun.transition_to(State::Aborted),
            Ok(State::Aborted)
        );
        assert_eq!(
            State::Committed.transition_to(State::Committed),
            Ok(State::Committed)
        );
        assert_eq!(
            State::Aborted.transition_to(State::Aborted),
            Ok(State::Aborted)
        );
        assert_eq!(
            State::Committed.transition_to(State::Aborted),
            Err(TxnError::DecisionConflict)
        );
        assert_eq!(
            State::Aborted.transition_to(State::Committed),
            Err(TxnError::DecisionConflict)
        );
        assert_eq!(
            State::Committed.transition_to(State::Begun),
            Err(TxnError::DecisionConflict)
        );
    }

    #[test]
    fn record_decide_is_final() {
        let mut record = TxnRecord {
            id: TxnId::derive(1, 2, 0),
            coordinator: TabletId::from_u64(1),
            participants: vec![TabletId::from_u64(1)],
            state: TxnState::Begun,
            dir_version: 7,
            digest: [0xAB; 32],
        };
        record.decide(TxnDecision::Commit).expect("begun commits");
        assert_eq!(record.state, TxnState::Committed);
        assert_eq!(
            record.decide(TxnDecision::Abort),
            Err(TxnError::DecisionConflict)
        );
        // Replay of the same decision stays idempotent.
        record.decide(TxnDecision::Commit).expect("replay");
    }

    #[test]
    fn coordinator_selects_lowest_participant() {
        let participants: BTreeSet<TabletId> =
            [9, 3, 7].into_iter().map(TabletId::from_u64).collect();
        assert_eq!(
            TxnCoordinator::select(&participants),
            Some(TxnCoordinator {
                tablet: TabletId::from_u64(3)
            })
        );
        assert_eq!(TxnCoordinator::select(&BTreeSet::new()), None);
    }

    #[test]
    fn reservation_cost_bounds_before_prepare() {
        assert!(reservation_cost(&[write("a")]).is_ok());
        let many: Vec<TxnWrite> = (0..513).map(|i| write(&format!("k{i}"))).collect();
        assert_eq!(reservation_cost(&many), Err(TxnError::TooLarge));
    }
}
