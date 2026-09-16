//! Cross-tablet OCC + 2PC transaction substrate, V1 (RFC transaction fabric).
//!
//! Transaction V1 provides **atomic point-key mutation sets** across
//! tablets: OCC conflict detection via [`ObjectVersion`] validation plus
//! durable two-phase commit driven by a deterministic coordinator tablet.
//! It is deliberately NOT general transactions: no SQL, no range
//! predicates, no MVCC, no global snapshot reads.
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
//! The tablet owning the first write in request order coordinates
//! (deterministic for a given write order; no control-plane
//! involvement in the data path). Single-tablet transactions collapse to
//! the same intent machinery with coordinator == sole participant: no
//! cross-group coordination, one atomic finalize wave per key. Client
//! drivers discover the first write's tablet with a one-entry scan probe
//! (which also warms the route cache) before preparing.
//!
//! ## Isolation without MVCC
//!
//! Prepared intents hide: user state is untouched until finalize, and
//! conflicting local writes fail with [`TxnError::Conflict`] while an
//! intent is prepared. Commit finalization is atomic per tablet; across
//! tablets, readers under `LatestPerTablet` may transiently observe commit
//! convergence (documented, no global snapshot claimed).
//!
//! ## Bounds
//!
//! [`MAX_TXN_PARTICIPANTS`], [`MAX_TXN_KEYS`], [`MAX_TXN_BYTES`], and
//! [`MAX_TXN_LIFETIME_MICROS`] bound every transaction; oversize plans are
//! rejected before any intent is written.

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

impl Encode for TxnWriteKind {
    fn encoded_len(&self) -> usize {
        match self {
            Self::Put(value) => 1 + 4 + value.len(),
            Self::Delete => 1,
            Self::CounterAdd(_) => 1 + 8,
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
    /// Remove the key.
    Delete,
    /// Add to a counter (prepared deterministically: overflow fails at
    /// prepare, identically on every replica).
    CounterAdd(i64),
}

impl TxnWrite {
    /// Estimated wire/intent bytes for bounding.
    #[must_use]
    pub fn estimated_bytes(&self) -> usize {
        let payload = match &self.kind {
            TxnWriteKind::Put(value) => value.len(),
            TxnWriteKind::Delete | TxnWriteKind::CounterAdd(_) => 8,
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

/// One participant's durable intent: the prepared write plus the version
/// it was validated against.
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
    /// Live version observed at prepare (`None` = absent).
    pub observed: Option<ObjectVersion>,
    /// Prepare timestamp (micros, leader-materialized; resolver use only).
    pub prepared_at: u64,
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
}

impl TxnError {
    /// Whether the client may retry with a fresh attempt number.
    #[must_use]
    pub const fn is_retryable(self) -> bool {
        match self {
            Self::Conflict | Self::RoutingChanged | Self::CoordinatorUnavailable => true,
            Self::UniqueViolation | Self::TooLarge | Self::Aborted | Self::InvalidWrite => false,
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
#[must_use]
pub fn write_set_digest(writes: &[TxnWrite], namespace: NamespaceId) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"KIVI-TXN-SET-V1");
    hasher.update(&namespace.as_u64().to_le_bytes());
    let mut sorted: Vec<&TxnWrite> = writes.iter().collect();
    sorted.sort_by(|left, right| left.key.as_bytes().cmp(right.key.as_bytes()));
    for write in sorted {
        hasher.update(write.key.as_bytes());
        match &write.kind {
            TxnWriteKind::Put(value) => {
                hasher.update(&[1]);
                hasher.update(value);
            }
            TxnWriteKind::Delete => {
                hasher.update(&[2]);
            }
            TxnWriteKind::CounterAdd(delta) => {
                hasher.update(&[3]);
                hasher.update(&delta.to_le_bytes());
            }
        }
    }
    *hasher.finalize().as_bytes()
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

/// Selects the deterministic coordinator: the tablet owning the first
/// write in request order. Deterministic for a given write order, so
/// idempotent re-drives agree; single-write/single-tablet transactions
/// trivially coordinate on their sole participant (the fast path).
#[must_use]
pub fn select_coordinator(
    writes: &[TxnWrite],
    route: impl Fn(&[u8]) -> Option<TabletId>,
) -> Option<TabletId> {
    writes.first().and_then(|write| route(write.key.as_bytes()))
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
        1 => TxnWriteKind::Put(Bytes::from(value)),
        2 => TxnWriteKind::Delete,
        3 => TxnWriteKind::CounterAdd(delta),
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
    /// Deterministic coordinator (first write's tablet).
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
        // Coordinator is the first write's tablet (request order).
        assert_eq!(
            select_coordinator(&writes, route),
            Some(TabletId::from_u64(2))
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
}
