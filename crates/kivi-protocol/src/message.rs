//! Native message semantics: opcodes, handshake, requests, responses.
//!
//! Every type here has a fixed wire encoding (little-endian integers,
//! length-prefixed blobs, versioned tags) implemented by hand — never
//! derived struct layout, never enum discriminants. Decoding translates
//! immediately into project-owned domain types ([`Operation`],
//! [`PartitionRange`], tablet identities); this crate never redefines what
//! those mean.
//!
//! Human-readable diagnostics may accompany a response, but the stable
//! machine-readable [`Status`] code is the only semantic contract.

use kivi_state::{Key, Mutation, Operation};
use kivi_tablet::{DirectoryVersion, HashPrefix, OrderedRange, PartitionRange};
use kivi_types::{
    ClusterId, CommitPosition, CommitToken, NamespaceId, NodeId, NodeIncarnation, ReadContract,
    ReadReceipt, RequestIdentity, RequestSeq, TabletAuthority, TabletEpoch, TabletId, UnixMicros,
    WorkerId, WriteGuardGeneration,
};

use crate::frame::ProtocolError;

bitflags::bitflags! {
    /// Negotiated protocol capabilities: one `u64` holding two 32-bit spaces.
    ///
    /// Bits 0..32 are the required space (unknown bits fail negotiation);
    /// bits 32..64 are the optional space (unknown bits are silently ignored).
    /// No bits are minted beyond v1 yet, except streaming below. Future
    /// candidates with no assigned bits and no implementation: TLS/auth,
    /// compression, changefeeds, strong caches, QUIC.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct Capabilities: u64 {
        /// Base v1 protocol every peer must speak (required bit).
        const BASE_V1 = 1 << 0;
        /// Durable mutation dedup: the server persists every mutating
        /// request's outcome keyed by client identity before replying, so a
        /// retried identity returns its original outcome instead of
        /// re-executing. Servers advertise it only in durable mode; clients
        /// retry ambiguous mutating requests only when they saw it.
        const DURABLE_MUTATION_DEDUP = 1 << 1;
        /// Chunk-fabric streaming: upload (`Stream*`) and download
        /// (`ValueStream*`) frames plus the `SetRange`/`GetStream`
        /// opcodes. Servers advertise it when they serve streams; clients
        /// gate every streaming API on having seen it and answer
        /// `Unsupported` without sending a frame otherwise.
        const STREAMING = 1 << 2;
    }
}

impl Capabilities {
    /// Capability bits this build understands.
    pub const KNOWN: Self = Self::BASE_V1
        .union(Self::DURABLE_MUTATION_DEDUP)
        .union(Self::STREAMING);

    /// Bits in a required mask that this build does not understand (`empty`
    /// means negotiation can proceed). Unknown bits fail even in the
    /// optional space when placed in the required mask: required means the
    /// peer demands something undefined.
    #[must_use]
    pub fn unknown_required(self) -> Self {
        self - Self::KNOWN
    }
}

/// Native operation codes. Fixed wire values, never reordered.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Opcode {
    /// Fetch bytes.
    Get = 1,
    /// Store bytes.
    Set = 2,
    /// Remove a key.
    Delete = 3,
    /// Test for a live value.
    Exists = 4,
    /// Fetch a counter.
    CounterGet = 5,
    /// Add to a counter.
    CounterAdd = 6,
    /// Attach an absolute expiry.
    ExpireAt = 7,
    /// Clear any expiry.
    PersistExpiry = 8,
    /// Read a key's expiry.
    GetExpiry = 9,
    /// Patch a byte range of a value (partial update; Redis SETRANGE
    /// semantics with zero-padding past the end).
    SetRange = 10,
    /// Fetch a value as a stream (`ValueStream*` frames) instead of one
    /// response frame. Large reads never build giant frames either side.
    GetStream = 11,
    /// Read a byte slice `[offset, offset + len)` clamped to the length.
    GetRange = 12,
    /// Report the logical byte length without reading chunk payloads.
    BytesLength = 13,
    /// Conditionally store bytes with a Kivi-owned presence condition and
    /// expiry policy (atomic at the owning tablet).
    SetConditional = 14,
    /// Bounded single-tablet ordered scan (see the `scan_*` fields on
    /// [`Request`]): reads `[start, end)` in key order, never unbounded.
    /// Cross-tablet scans fan out from the client; the server answers only
    /// the slice owned by the routed tablet.
    Scan = 15,
    /// Atomic multi-key batch against ONE tablet (single-tablet fast path;
    /// multi-tablet batches use client-driven 2PC over `TxnPrepare` /
    /// `TxnFinalize`). See the `batch_*` fields on [`Request`].
    AtomicBatch = 16,
    /// Reserve one key for a transaction (2PC prepare): OCC validation
    /// plus a durable intent, no user-visible mutation yet.
    TxnPrepare = 17,
    /// Resolve one key's transaction intent: commit applies the prepared
    /// write, abort discards it. Idempotent by `TxnId`.
    TxnFinalize = 18,
    /// Read a key's logical version without fetching its value
    /// (`None` when absent; works for chunked values without resolving
    /// any bytes). Index maintenance and OCC planning use this.
    GetVersion = 19,
    /// Fetch a commutative counter's current sum (a state query, not an
    /// ordinal; additions stay order-free).
    CommutativeGet = 20,
    /// Add to a commutative counter. The result is `Applied`, never an
    /// ordinal, so additions commute observably.
    CommutativeAdd = 21,
    /// Create a bounded counter with a global capacity and a full local
    /// escrow share for the executing tablet.
    BoundedCounterCreate = 22,
    /// Add to a bounded counter within locally owned escrow rights.
    BoundedCounterAdd = 23,
    /// Read a bounded counter's value and capacity.
    BoundedCounterGet = 24,
    /// Narrow one bounded counter's escrow share (lone operations narrow
    /// only; widening pairs inside atomic transfers).
    EscrowTransfer = 25,
    /// Create a semaphore with a total permit capacity.
    SemaphoreCreate = 26,
    /// Acquire permits under a client-minted permit id (idempotent retry).
    SemaphoreAcquire = 27,
    /// Release the permits held under a permit id (idempotent, mints nothing).
    SemaphoreRelease = 28,
    /// Inspect a semaphore's outstanding load and capacity.
    SemaphoreInspect = 29,
    /// Acquire (or re-acquire after expiry) a fenced lease.
    LeaseAcquire = 30,
    /// Extend a live lease grant (exact fencing token required).
    LeaseRenew = 31,
    /// Release a lease grant (exact fencing token required).
    LeaseRelease = 32,
    /// Inspect a lease's live holder and next fencing token.
    LeaseInspect = 33,
    /// Create one shard of a sharded stream.
    StreamCreate = 34,
    /// Append one entry to a stream shard (shard-local offset assigned).
    StreamAppend = 35,
    /// Read one shard's entries from an offset (shard-local, no global order).
    StreamRead = 36,
    /// Trim one shard's retained prefix (cursor never rewinds).
    StreamTrim = 37,
}

impl Opcode {
    /// Decodes an opcode byte.
    #[must_use]
    pub const fn from_u8(value: u8) -> Option<Self> {
        match value {
            1 => Some(Self::Get),
            2 => Some(Self::Set),
            3 => Some(Self::Delete),
            4 => Some(Self::Exists),
            5 => Some(Self::CounterGet),
            6 => Some(Self::CounterAdd),
            7 => Some(Self::ExpireAt),
            8 => Some(Self::PersistExpiry),
            9 => Some(Self::GetExpiry),
            10 => Some(Self::SetRange),
            11 => Some(Self::GetStream),
            12 => Some(Self::GetRange),
            13 => Some(Self::BytesLength),
            14 => Some(Self::SetConditional),
            15 => Some(Self::Scan),
            16 => Some(Self::AtomicBatch),
            17 => Some(Self::TxnPrepare),
            18 => Some(Self::TxnFinalize),
            19 => Some(Self::GetVersion),
            20 => Some(Self::CommutativeGet),
            21 => Some(Self::CommutativeAdd),
            22 => Some(Self::BoundedCounterCreate),
            23 => Some(Self::BoundedCounterAdd),
            24 => Some(Self::BoundedCounterGet),
            25 => Some(Self::EscrowTransfer),
            26 => Some(Self::SemaphoreCreate),
            27 => Some(Self::SemaphoreAcquire),
            28 => Some(Self::SemaphoreRelease),
            29 => Some(Self::SemaphoreInspect),
            30 => Some(Self::LeaseAcquire),
            31 => Some(Self::LeaseRenew),
            32 => Some(Self::LeaseRelease),
            33 => Some(Self::LeaseInspect),
            34 => Some(Self::StreamCreate),
            35 => Some(Self::StreamAppend),
            36 => Some(Self::StreamRead),
            37 => Some(Self::StreamTrim),
            _ => None,
        }
    }

    /// Encodes the opcode discriminant.
    #[must_use]
    pub const fn as_u8(self) -> u8 {
        self as u8
    }

    /// Whether this opcode can change logical state (and therefore enters
    /// the WAL in durable mode). Reads never do, whatever their outcome.
    #[must_use]
    pub const fn is_mutating(self) -> bool {
        matches!(
            self,
            Self::Set
                | Self::Delete
                | Self::CounterAdd
                | Self::ExpireAt
                | Self::PersistExpiry
                | Self::SetRange
                | Self::SetConditional
                | Self::AtomicBatch
                | Self::TxnPrepare
                | Self::TxnFinalize
                | Self::CommutativeAdd
                | Self::BoundedCounterCreate
                | Self::BoundedCounterAdd
                | Self::EscrowTransfer
                | Self::SemaphoreCreate
                | Self::SemaphoreAcquire
                | Self::SemaphoreRelease
                | Self::LeaseAcquire
                | Self::LeaseRenew
                | Self::LeaseRelease
                | Self::StreamCreate
                | Self::StreamAppend
                | Self::StreamTrim
        )
    }
}

impl core::fmt::Display for Opcode {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Get => write!(f, "get"),
            Self::Set => write!(f, "set"),
            Self::Delete => write!(f, "delete"),
            Self::Exists => write!(f, "exists"),
            Self::CounterGet => write!(f, "counter-get"),
            Self::CounterAdd => write!(f, "counter-add"),
            Self::ExpireAt => write!(f, "expire-at"),
            Self::PersistExpiry => write!(f, "persist-expiry"),
            Self::GetExpiry => write!(f, "get-expiry"),
            Self::SetRange => write!(f, "set-range"),
            Self::GetStream => write!(f, "get-stream"),
            Self::GetRange => write!(f, "get-range"),
            Self::BytesLength => write!(f, "bytes-length"),
            Self::SetConditional => write!(f, "set-conditional"),
            Self::Scan => write!(f, "scan"),
            Self::AtomicBatch => write!(f, "atomic-batch"),
            Self::TxnPrepare => write!(f, "txn-prepare"),
            Self::TxnFinalize => write!(f, "txn-finalize"),
            Self::GetVersion => write!(f, "get-version"),
            Self::CommutativeGet => write!(f, "commutative-get"),
            Self::CommutativeAdd => write!(f, "commutative-add"),
            Self::BoundedCounterCreate => write!(f, "bounded-counter-create"),
            Self::BoundedCounterAdd => write!(f, "bounded-counter-add"),
            Self::BoundedCounterGet => write!(f, "bounded-counter-get"),
            Self::EscrowTransfer => write!(f, "escrow-transfer"),
            Self::SemaphoreCreate => write!(f, "semaphore-create"),
            Self::SemaphoreAcquire => write!(f, "semaphore-acquire"),
            Self::SemaphoreRelease => write!(f, "semaphore-release"),
            Self::SemaphoreInspect => write!(f, "semaphore-inspect"),
            Self::LeaseAcquire => write!(f, "lease-acquire"),
            Self::LeaseRenew => write!(f, "lease-renew"),
            Self::LeaseRelease => write!(f, "lease-release"),
            Self::LeaseInspect => write!(f, "lease-inspect"),
            Self::StreamCreate => write!(f, "stream-create"),
            Self::StreamAppend => write!(f, "stream-append"),
            Self::StreamRead => write!(f, "stream-read"),
            Self::StreamTrim => write!(f, "stream-trim"),
        }
    }
}

/// Stable machine-readable response status. The code is the contract;
/// diagnostics are commentary. Nothing here carries Rust error strings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Status {
    /// The operation completed; see the body.
    Ok = 0,
    /// Structurally invalid request (bad shape, bad lengths, bad pairing).
    InvalidRequest = 1,
    /// Operation or capability not implemented by this server.
    Unsupported = 2,
    /// Key holds a different logical type; nothing mutated.
    WrongType = 3,
    /// No live value under the key.
    NotFound = 4,
    /// Counter addition overflows `i64`; nothing mutated.
    CounterOverflow = 5,
    /// An object version cannot advance; nothing mutated.
    VersionExhausted = 6,
    /// The receiving worker is not the authority; redirect info attached.
    StaleRoute = 7,
    /// Correct worker, wrong tablet or epoch for the hint; redirect attached.
    NotLocal = 8,
    /// Bounded queues are full; safe to retry (ideally with backoff).
    Overloaded = 9,
    /// A resource bound (frame, value, connection state) was exceeded.
    ResourceExhausted = 10,
    /// A single key or value exceeds the configured per-value bound.
    ValueTooLarge = 11,
    /// Unexpected server-side failure; safe to retry, likely pointless.
    Internal = 12,
    /// A retried mutation identity fell below the session's durable floor:
    /// its outcome is gone and it will never execute again. Not retriable
    /// under the same identity; start a new mutation instead.
    DedupExpired = 13,
    /// The session holds too many unacknowledged outcomes on this tablet;
    /// acknowledge older mutations before sending new ones. Deterministic
    /// for the same state, so retrying is harmless but pointless.
    SessionOverloaded = 14,
    /// A transactional prepare failed OCC validation or met a prepared
    /// intent. Not a transport retry: the batch API decides (new attempt).
    TxnConflict = 15,
    /// The transaction was aborted (prepare rejected or coordinator
    /// decided abort). Retrying the identical batch replays the abort;
    /// start a new attempt instead.
    TxnAborted = 16,
    /// The transaction exceeds participant/key/byte bounds. Split it.
    TxnTooLarge = 17,
    /// A participant cannot reach the coordinator record yet. Safe to
    /// re-drive the identical batch (same `TxnId`).
    TxnCoordinatorUnavailable = 18,
    /// A unique index term is already owned by another primary.
    UniqueViolation = 19,
    /// A scan cursor no longer resolves (namespace dropped or cursor
    /// predates retention). Re-issue the scan from the start.
    ScanCursorStale = 20,
    /// An `AtLeast` token's lineage is foreign to the serving replica
    /// (wrong tablet, superseded epoch, or unassigned position). Waiting
    /// cannot cure it: refresh the token from a read on its own lineage.
    /// Never retried under the same token.
    StaleToken = 21,
    /// A strong write committed internally but responder coverage could
    /// not be established, so it was not acknowledged as complete: the
    /// outcome is uncertain — it may have applied. Never transparently
    /// retried (that would double-apply anonymous non-idempotent writes);
    /// identified writes may re-drive under the same identity, everyone
    /// else must read-verify before retrying.
    CoverageUncertain = 22,
    /// A bounded-counter addition would leave locally owned escrow rights
    /// (or the global bound). Move rights first, then retry.
    BoundedExceeded = 23,
    /// A semaphore acquisition would exceed capacity. Retry after releases.
    SemaphoreExhausted = 24,
    /// A lease acquisition met a live holder owned by someone else.
    LeaseConflict = 25,
    /// A lease renew/release named a fencing token that is not the current
    /// grant's. Re-acquire; never retry blindly with the stale token.
    StaleFencing = 26,
    /// A stream append would exceed the shard's entry or payload bounds.
    StreamFull = 27,
    /// Conflicting final decisions for one transaction (durable corruption).
    /// Never guessed; surfaced for operator repair.
    TxnDecisionConflict = 28,
    /// The batch spans tablets: drive it over `TxnPrepare`/`TxnFinalize`.
    /// Typed (never string-matched): oversize batches stay `TxnTooLarge`.
    TxnCrossTablet = 29,
}

impl Status {
    /// Decodes a status code.
    #[must_use]
    pub const fn from_u16(value: u16) -> Option<Self> {
        match value {
            0 => Some(Self::Ok),
            1 => Some(Self::InvalidRequest),
            2 => Some(Self::Unsupported),
            3 => Some(Self::WrongType),
            4 => Some(Self::NotFound),
            5 => Some(Self::CounterOverflow),
            6 => Some(Self::VersionExhausted),
            7 => Some(Self::StaleRoute),
            8 => Some(Self::NotLocal),
            9 => Some(Self::Overloaded),
            10 => Some(Self::ResourceExhausted),
            11 => Some(Self::ValueTooLarge),
            12 => Some(Self::Internal),
            13 => Some(Self::DedupExpired),
            14 => Some(Self::SessionOverloaded),
            15 => Some(Self::TxnConflict),
            16 => Some(Self::TxnAborted),
            17 => Some(Self::TxnTooLarge),
            18 => Some(Self::TxnCoordinatorUnavailable),
            19 => Some(Self::UniqueViolation),
            20 => Some(Self::ScanCursorStale),
            21 => Some(Self::StaleToken),
            22 => Some(Self::CoverageUncertain),
            23 => Some(Self::BoundedExceeded),
            24 => Some(Self::SemaphoreExhausted),
            25 => Some(Self::LeaseConflict),
            26 => Some(Self::StaleFencing),
            27 => Some(Self::StreamFull),
            28 => Some(Self::TxnDecisionConflict),
            29 => Some(Self::TxnCrossTablet),
            _ => None,
        }
    }

    /// Encodes the status code.
    #[must_use]
    pub const fn as_u16(self) -> u16 {
        self as u16
    }

    /// Whether retrying the identical request (after any attached redirect)
    /// can usefully succeed. Redirects and overload are retriable by design;
    /// semantic rejections are not. `TxnConflict` is deliberately NOT
    /// retriable at transport: the identical 2PC re-drive would replay the
    /// same contention, so the batch API mints a fresh attempt instead
    /// (transport retry vs transaction retry stay distinct).
    #[must_use]
    pub const fn is_retriable(self) -> bool {
        matches!(
            self,
            Self::StaleRoute
                | Self::NotLocal
                | Self::Overloaded
                | Self::TxnCoordinatorUnavailable
                | Self::ScanCursorStale
        )
    }
}

impl core::fmt::Display for Status {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Ok => write!(f, "ok"),
            Self::InvalidRequest => write!(f, "invalid-request"),
            Self::Unsupported => write!(f, "unsupported"),
            Self::WrongType => write!(f, "wrong-type"),
            Self::NotFound => write!(f, "not-found"),
            Self::CounterOverflow => write!(f, "counter-overflow"),
            Self::VersionExhausted => write!(f, "version-exhausted"),
            Self::StaleRoute => write!(f, "stale-route"),
            Self::NotLocal => write!(f, "not-local"),
            Self::Overloaded => write!(f, "overloaded"),
            Self::ResourceExhausted => write!(f, "resource-exhausted"),
            Self::ValueTooLarge => write!(f, "value-too-large"),
            Self::Internal => write!(f, "internal"),
            Self::DedupExpired => write!(f, "dedup-expired"),
            Self::SessionOverloaded => write!(f, "session-overloaded"),
            Self::TxnConflict => write!(f, "txn-conflict"),
            Self::TxnAborted => write!(f, "txn-aborted"),
            Self::TxnTooLarge => write!(f, "txn-too-large"),
            Self::TxnCoordinatorUnavailable => write!(f, "txn-coordinator-unavailable"),
            Self::UniqueViolation => write!(f, "unique-violation"),
            Self::ScanCursorStale => write!(f, "scan-cursor-stale"),
            Self::StaleToken => write!(f, "stale-token"),
            Self::CoverageUncertain => write!(f, "coverage-uncertain"),
            Self::BoundedExceeded => write!(f, "bounded-exceeded"),
            Self::SemaphoreExhausted => write!(f, "semaphore-exhausted"),
            Self::LeaseConflict => write!(f, "lease-conflict"),
            Self::StaleFencing => write!(f, "stale-fencing"),
            Self::StreamFull => write!(f, "stream-full"),
            Self::TxnDecisionConflict => write!(f, "txn-decision-conflict"),
            Self::TxnCrossTablet => write!(f, "txn-cross-tablet"),
        }
    }
}

/// Request identity on one connection: echoed in the matching response.
/// Allocated by the client from 1 upward (0 conventionally means "none").
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RequestId(u64);

impl RequestId {
    /// First client-allocated identity.
    pub const FIRST: Self = Self(1);

    /// Wraps a raw identity value.
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

impl core::fmt::Display for RequestId {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "req{}", self.0)
    }
}

/// Optional client route hint: where the client believes the authority is.
/// The server revalidates against its immutable snapshot on every request —
/// hints accelerate, never authorize.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RouteHint {
    /// Believed tablet.
    pub tablet: TabletId,
    /// Believed epoch.
    pub epoch: TabletEpoch,
    /// Believed owner worker.
    pub worker: WorkerId,
}

/// Redirection answer: everything needed to retry directly at the authority.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RedirectInfo {
    /// Authority's directory version (rejects obviously stale client routes).
    pub dir_version: DirectoryVersion,
    /// Authoritative tablet.
    pub tablet: TabletId,
    /// Its epoch.
    pub epoch: TabletEpoch,
    /// Its owner worker.
    pub worker: WorkerId,
    /// Dialable worker endpoint (`"ip:port"`, ≤ 255 bytes).
    pub endpoint: String,
    /// Owned range (lets the client cache a prefix, not just a point).
    pub range: PartitionRange,
}

impl RedirectInfo {
    /// Builds redirect info, validating the endpoint fits its length prefix.
    ///
    /// # Errors
    ///
    /// Returns [`ProtocolError::Malformed`] when the endpoint exceeds 255 bytes.
    pub fn new(
        dir_version: DirectoryVersion,
        tablet: TabletId,
        epoch: TabletEpoch,
        worker: WorkerId,
        endpoint: String,
        range: PartitionRange,
    ) -> Result<Self, ProtocolError> {
        if endpoint.len() > u8::MAX as usize {
            return Err(ProtocolError::Malformed {
                context: "redirect endpoint",
            });
        }
        Ok(Self {
            dir_version,
            tablet,
            epoch,
            worker,
            endpoint,
            range,
        })
    }
}

/// Client handshake opener: versions, capabilities, limits, affinity wish.
/// Plain wire DTO with public fields; validation lives in encode/decode
/// and the server's negotiation checks.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClientHello {
    /// Largest frame the client will accept (server replies with the min).
    pub max_frame: u32,
    /// Required capability bits (unknown → handshake fails).
    pub required_caps: Capabilities,
    /// Optional capability bits (unknown → ignored).
    pub optional_caps: Capabilities,
    /// Preferred worker, or `None` for seed routing.
    pub desired_worker: Option<WorkerId>,
}

/// Server handshake answer: negotiated versions, capabilities, and the
/// topology facts a smart client needs (cluster/node/worker identities,
/// directory version, frame bound, worker endpoints for direct dialing).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServerHello {
    /// Negotiated major version.
    pub major: u16,
    /// Negotiated minor version.
    pub minor: u16,
    /// Server capability bits.
    pub caps: Capabilities,
    /// Cluster identity.
    pub cluster: ClusterId,
    /// Serving node.
    pub node: NodeId,
    /// Its incarnation.
    pub incarnation: NodeIncarnation,
    /// The worker behind this connection.
    pub worker: WorkerId,
    /// Current directory version.
    pub dir_version: DirectoryVersion,
    /// Negotiated maximum frame.
    pub max_frame: u32,
    /// Dialable endpoints per worker for direct routing.
    pub endpoints: Vec<(WorkerId, String)>,
}

/// Wire values for [`kivi_state::SetCondition`] on `SetConditional`.
pub const COND_ALWAYS: u8 = 0;
/// Wire values for [`kivi_state::SetCondition`] on `SetConditional`.
pub const COND_IF_ABSENT: u8 = 1;
/// Wire values for [`kivi_state::SetCondition`] on `SetConditional`.
pub const COND_IF_PRESENT: u8 = 2;

/// Wire values for [`kivi_state::ExpiryPolicy`] on `SetConditional`.
pub const EXPIRY_CLEAR: u8 = 0;
/// Wire values for [`kivi_state::ExpiryPolicy`] on `SetConditional`.
pub const EXPIRY_KEEP: u8 = 1;
/// Wire values for [`kivi_state::ExpiryPolicy`] on `SetConditional`.
pub const EXPIRY_AT: u8 = 2;

/// Wire values for scan direction on `Scan`: ascending key order.
pub const SCAN_FORWARD: u8 = 0;
/// Wire values for scan direction on `Scan`: descending key order.
pub const SCAN_REVERSE: u8 = 1;

/// Wire values for scan projection on `Scan`: keys only.
pub const SCAN_KEYS_ONLY: u8 = 0;
/// Wire values for scan projection on `Scan`: keys with bounded values.
pub const SCAN_KEYS_AND_VALUES: u8 = 1;

/// Wire values for scan consistency on `Scan`: strong `Latest` per tablet.
/// The full multi-tablet scan is NOT one global snapshot (documented).
pub const SCAN_LATEST_PER_TABLET: u8 = 0;
/// Wire values for scan consistency on `Scan`: weakest local read per the
/// tablet's existing read contracts.
pub const SCAN_ANY: u8 = 1;

/// Wire values for batch write kinds on `AtomicBatch`.
pub const BATCH_PUT: u8 = 1;
/// Wire values for batch write kinds on `AtomicBatch`.
pub const BATCH_DELETE: u8 = 2;
/// Wire values for batch write kinds on `AtomicBatch`.
pub const BATCH_COUNTER_ADD: u8 = 3;
/// Chunked put by staged manifest reference (value blob carries 32-byte
/// manifest id plus 8-byte LE logical length).
pub const BATCH_PUT_CHUNKED: u8 = 4;
/// Commutative-counter add (order-free; no ordinal produced).
pub const BATCH_COMMUTATIVE_ADD: u8 = 5;
/// Bounded-counter add (escrow rights checked at prepare).
pub const BATCH_BOUNDED_ADD: u8 = 6;
/// Escrow share move (value blob carries 8-byte LE min plus 8-byte LE max;
/// one side of a paired, conservation-checked transfer).
pub const BATCH_ESCROW_SHARE: u8 = 7;

/// Wire values for batch OCC expectations on `AtomicBatch`: blind write.
pub const BATCH_EXPECT_ANY: u8 = 0;
/// Wire values for batch OCC expectations on `AtomicBatch`: key must be absent.
pub const BATCH_EXPECT_ABSENT: u8 = 1;
/// Wire values for batch OCC expectations on `AtomicBatch`: live version
/// must equal the carried `expect_version`.
pub const BATCH_EXPECT_VERSION: u8 = 2;

/// One write inside an `AtomicBatch` request: a point key, kind, payload,
/// and OCC expectation. The coordinator translates each into a
/// `TxnPrepare` against its owning tablet; all commit or none does.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BatchWrite {
    /// Target key.
    pub key: Vec<u8>,
    /// One of [`BATCH_PUT`], [`BATCH_DELETE`], [`BATCH_COUNTER_ADD`],
    /// [`BATCH_PUT_CHUNKED`], [`BATCH_COMMUTATIVE_ADD`],
    /// [`BATCH_BOUNDED_ADD`], [`BATCH_ESCROW_SHARE`].
    pub kind: u8,
    /// Put payload (meaningful for `BATCH_PUT` only).
    pub value: Vec<u8>,
    /// Addend (meaningful for `BATCH_COUNTER_ADD` only).
    pub delta: i64,
    /// One of [`BATCH_EXPECT_ANY`], [`BATCH_EXPECT_ABSENT`],
    /// [`BATCH_EXPECT_VERSION`].
    pub expect: u8,
    /// Expected live version (meaningful for `BATCH_EXPECT_VERSION` only).
    pub expect_version: u64,
}

/// One scanned entry in a `Scan` response page.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScanEntryBody {
    /// Scanned key.
    pub key: Vec<u8>,
    /// Projected value shape (see tags below).
    pub value: ScanValueBody,
}

/// Value payload of [`ScanEntryBody`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ScanValueBody {
    /// No value (`KeysOnly` projection).
    None,
    /// Resident bytes (fit the page budget).
    Inline(Vec<u8>),
    /// Exact counter value.
    Counter(i64),
    /// Chunked bytes by reference (resolve via `get_stream`).
    Chunked {
        /// Manifest addressing the immutable chunk sequence.
        manifest: [u8; 32],
        /// Total logical bytes across the manifest.
        logical_len: u64,
    },
    /// Resident bytes exceeding the page budget on an otherwise empty
    /// page: length only, fetch via point `get`/`get_stream`.
    Oversize {
        /// Logical length of the value.
        logical_len: u64,
    },
    /// Typed semantic value by small descriptor (object type tag plus
    /// canonical descriptor bytes; never inlined bulk).
    Semantic {
        /// Object type tag (mirrors `kivi-state`'s `ObjectType` tags).
        object: u8,
        /// Small canonical descriptor bytes for `object`.
        descriptor: Vec<u8>,
    },
}

/// Wire tags for [`ScanValueBody`]. Fixed, never reused.
pub const SCAN_VALUE_NONE: u8 = 0;
/// Wire tags for [`ScanValueBody`]. Fixed, never reused.
pub const SCAN_VALUE_INLINE: u8 = 1;
/// Wire tags for [`ScanValueBody`]. Fixed, never reused.
pub const SCAN_VALUE_COUNTER: u8 = 2;
/// Wire tags for [`ScanValueBody`]. Fixed, never reused.
pub const SCAN_VALUE_CHUNKED: u8 = 3;
/// Wire tags for [`ScanValueBody`]. Fixed, never reused.
pub const SCAN_VALUE_OVERSIZE: u8 = 4;
/// Typed semantic descriptor tag. Fixed, never reused.
pub const SCAN_VALUE_SEMANTIC: u8 = 5;

/// One typed native request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Request {
    /// Target namespace.
    pub namespace: NamespaceId,
    /// Operation to invoke.
    pub opcode: Opcode,
    /// Client's route belief, if any.
    pub hint: Option<RouteHint>,
    /// Target key.
    pub key: Vec<u8>,
    /// Set payload (meaningful for `Set` only).
    pub value: Option<Vec<u8>>,
    /// Addend (meaningful for `CounterAdd` only).
    pub delta: i64,
    /// Absolute expiry micros (meaningful for `ExpireAt` only, and for
    /// `SetConditional` when its expiry policy is `ExpireAt`).
    pub expiry: u64,
    /// Patch offset (meaningful for `SetRange` only): the logical byte
    /// index the patch overwrites from, zero-padding past the end.
    /// Reused as the slice start for `GetRange`.
    pub offset: u64,
    /// Slice length (meaningful for `GetRange` only): maximum bytes to
    /// return from `offset`.
    pub len: u64,
    /// Presence condition (meaningful for `SetConditional` only):
    /// [`COND_ALWAYS`], [`COND_IF_ABSENT`], or [`COND_IF_PRESENT`].
    pub condition: u8,
    /// Expiry policy (meaningful for `SetConditional` only):
    /// [`EXPIRY_CLEAR`], [`EXPIRY_KEEP`], or [`EXPIRY_AT`] (with `expiry`).
    pub expiry_policy: u8,
    /// Retry identity for mutating opcodes (`None` for reads and for
    /// clients that predate durable dedup). Durable servers require it on
    /// every mutating request; ephemeral servers ignore it.
    pub identity: Option<RequestIdentity>,
    /// Freshness contract for read opcodes (`Latest` when absent on the
    /// wire: pre-contract clients only knew linearizable reads, and
    /// `Latest` is the strongest contract, so defaulting is never a
    /// silent weakening). Ignored on mutating opcodes. Non-`Latest`
    /// contracts ride a trailer (see [`Request::encode`]).
    pub contract: ReadContract,
    /// Client acknowledgement watermark for the identity's session
    /// (meaningful only alongside `identity`; zero otherwise).
    pub ack_floor: RequestSeq,
    /// Scan start, inclusive (`None` = first key; `Scan` only).
    pub scan_start: Option<Vec<u8>>,
    /// Scan end, exclusive (`None` = last key; `Scan` only).
    pub scan_end: Option<Vec<u8>>,
    /// Scan direction ([`SCAN_FORWARD`] / [`SCAN_REVERSE`]; `Scan` only).
    pub scan_direction: u8,
    /// Maximum entries per page (`Scan` only).
    pub scan_max_items: u32,
    /// Budget for inline value bytes per page (`Scan` only).
    pub scan_max_bytes: u32,
    /// Projection ([`SCAN_KEYS_ONLY`] / [`SCAN_KEYS_AND_VALUES`]).
    pub scan_projection: u8,
    /// Consistency ([`SCAN_LATEST_PER_TABLET`] / [`SCAN_ANY`]).
    pub scan_consistency: u8,
    /// Client-minted transaction id (stable across transport retries;
    /// fresh per OCC attempt; `AtomicBatch` only).
    pub batch_txn: [u8; 16],
    /// Writes in the batch (`AtomicBatch` only, bound enforced server-side).
    pub batch_writes: Vec<BatchWrite>,
    /// Coordinator tablet for one prepare (`TxnPrepare` only).
    pub txn_coordinator: u64,
    /// Whether to apply (`true`) or discard (`false`) the intent
    /// (`TxnFinalize` only).
    pub txn_commit: bool,
    /// Digest over the full transaction write set (`TxnPrepare` /
    /// `TxnFinalize`): binds the step to the exact transaction, so a
    /// conflicting driver under one `TxnId` is rejected, never mixed in.
    pub txn_digest: [u8; 32],
    /// Total capacity (`BoundedCounterCreate` / `SemaphoreCreate` only).
    pub capacity: u64,
    /// Tablet holding a new bounded counter's initial full escrow share
    /// (`BoundedCounterCreate` only; filled by the driver after routing).
    pub holder: u64,
    /// Permit identity (16 bytes; `SemaphoreAcquire` / `SemaphoreRelease`).
    pub permit: [u8; 16],
    /// Owner/session identity (`SemaphoreAcquire`, lease ops).
    pub owner: u64,
    /// Permit quantity (`SemaphoreAcquire` only).
    pub qty: u64,
    /// Fencing token of the named grant (`LeaseRenew` / `LeaseRelease`).
    pub fencing: u64,
    /// Logical time-to-live in micros from now (`LeaseAcquire` /
    /// `LeaseRenew`; `0` means an immortal grant).
    pub ttl: u64,
    /// Stream identity (16 bytes; `StreamCreate` only).
    pub stream: [u8; 16],
    /// Shard index within the stream (`StreamCreate` only).
    pub shard: u32,
    /// Partition key that routed this entry (`StreamAppend` only; retained
    /// for audit, not a second payload).
    pub partition: Vec<u8>,
    /// New escrow share lower bound (`EscrowTransfer` only).
    pub share_min: i64,
    /// New escrow share upper bound (`EscrowTransfer` only).
    pub share_max: i64,
}

impl Request {
    /// Translates into the project-owned typed operation. Returns `None`
    /// for [`Scan`](Opcode::Scan) and [`AtomicBatch`](Opcode::AtomicBatch):
    /// those dispatch on their dedicated payloads before operation
    /// translation (a scan has no single key; a batch has many), so forcing
    /// them through a single-key operation would misroute. Every other
    /// decodable request maps to exactly one operation.
    #[must_use]
    #[allow(clippy::too_many_lines)]
    pub fn into_operation(self) -> Option<Operation> {
        use kivi_state::{ExpiryPolicy, SetCondition};
        let key = Key::from(self.key);
        let operation = match self.opcode {
            // A stream read executes the same read as `Get`; the opcode
            // only changes the delivery shape (`ValueStream*` frames
            // instead of one response frame), so translation stays total
            // here and the server picks the encoder after executing.
            Opcode::Get | Opcode::GetStream => Operation::Get { key },
            Opcode::Set => Operation::Set {
                key,
                value: bytes::Bytes::from(self.value.unwrap_or_default()),
            },
            Opcode::Delete => Operation::Delete { key },
            Opcode::Exists => Operation::Exists { key },
            Opcode::CounterGet => Operation::CounterGet { key },
            Opcode::CounterAdd => Operation::CounterAdd {
                key,
                delta: self.delta,
            },
            Opcode::ExpireAt => Operation::ExpireAt {
                key,
                expires_at: UnixMicros::from_micros(self.expiry),
            },
            Opcode::PersistExpiry => Operation::PersistExpiry { key },
            Opcode::GetExpiry => Operation::GetExpiry { key },
            Opcode::SetRange => Operation::SetRange {
                key,
                offset: self.offset,
                patch: bytes::Bytes::from(self.value.unwrap_or_default()),
            },
            Opcode::GetRange => Operation::GetRange {
                key,
                offset: self.offset,
                len: self.len,
            },
            Opcode::BytesLength => Operation::BytesLength { key },
            Opcode::GetVersion => Operation::GetVersion { key },
            Opcode::SetConditional => {
                let condition = match self.condition {
                    COND_IF_ABSENT => SetCondition::IfAbsent,
                    COND_IF_PRESENT => SetCondition::IfPresent,
                    _ => SetCondition::Always,
                };
                let expiry = match self.expiry_policy {
                    EXPIRY_KEEP => ExpiryPolicy::Keep,
                    EXPIRY_AT => ExpiryPolicy::ExpireAt(UnixMicros::from_micros(self.expiry)),
                    _ => ExpiryPolicy::Clear,
                };
                Operation::SetConditional {
                    key,
                    value: bytes::Bytes::from(self.value.unwrap_or_default()),
                    condition,
                    expiry,
                }
            }
            Opcode::Scan | Opcode::AtomicBatch => return None,
            Opcode::TxnPrepare => {
                if self.batch_writes.len() != 1 {
                    return None;
                }
                let write = &self.batch_writes[0];
                let kind = match write.kind {
                    BATCH_PUT => {
                        kivi_state::TxnWriteKind::Put(bytes::Bytes::from(write.value.clone()))
                    }
                    BATCH_DELETE => kivi_state::TxnWriteKind::Delete,
                    BATCH_COUNTER_ADD => kivi_state::TxnWriteKind::CounterAdd(write.delta),
                    BATCH_PUT_CHUNKED => {
                        if write.value.len() != 40 {
                            return None;
                        }
                        let mut manifest = [0u8; 32];
                        manifest.copy_from_slice(&write.value[..32]);
                        let mut len = [0u8; 8];
                        len.copy_from_slice(&write.value[32..]);
                        kivi_state::TxnWriteKind::PutChunked {
                            manifest: kivi_types::ManifestId::from_bytes(manifest),
                            logical_len: u64::from_le_bytes(len),
                        }
                    }
                    BATCH_COMMUTATIVE_ADD => kivi_state::TxnWriteKind::CommutativeAdd(write.delta),
                    BATCH_BOUNDED_ADD => kivi_state::TxnWriteKind::BoundedAdd(write.delta),
                    BATCH_ESCROW_SHARE => {
                        if write.value.len() != 16 {
                            return None;
                        }
                        let mut min = [0u8; 8];
                        let mut max = [0u8; 8];
                        min.copy_from_slice(&write.value[..8]);
                        max.copy_from_slice(&write.value[8..]);
                        kivi_state::TxnWriteKind::EscrowSetShare {
                            min: i64::from_le_bytes(min),
                            max: i64::from_le_bytes(max),
                        }
                    }
                    _ => return None,
                };
                let expect = match write.expect {
                    BATCH_EXPECT_ABSENT => kivi_state::TxnExpect::Absent,
                    BATCH_EXPECT_VERSION => kivi_state::TxnExpect::Version(
                        kivi_state::ObjectVersion::from_u64(write.expect_version),
                    ),
                    _ => kivi_state::TxnExpect::Any,
                };
                return Some(Operation::TxnPrepare {
                    txn: kivi_state::TxnId::from_bytes(self.batch_txn),
                    coordinator: TabletId::from_u64(self.txn_coordinator),
                    write: kivi_state::TxnWrite {
                        key: Key::from(write.key.clone()),
                        kind,
                        expect,
                    },
                    digest: self.txn_digest,
                });
            }
            Opcode::TxnFinalize => {
                return Some(Operation::TxnFinalize {
                    txn: kivi_state::TxnId::from_bytes(self.batch_txn),
                    key,
                    commit: self.txn_commit,
                    digest: self.txn_digest,
                });
            }
            Opcode::CommutativeGet => Operation::CommutativeGet { key },
            Opcode::CommutativeAdd => Operation::CommutativeAdd {
                key,
                delta: self.delta,
            },
            Opcode::BoundedCounterCreate => Operation::BoundedCounterCreate {
                key,
                capacity: self.capacity,
                holder: TabletId::from_u64(self.holder),
            },
            Opcode::BoundedCounterAdd => Operation::BoundedCounterAdd {
                key,
                delta: self.delta,
            },
            Opcode::BoundedCounterGet => Operation::BoundedCounterGet { key },
            Opcode::EscrowTransfer => Operation::EscrowTransfer {
                key,
                new_min: self.share_min,
                new_max: self.share_max,
            },
            Opcode::SemaphoreCreate => Operation::SemaphoreCreate {
                key,
                capacity: self.capacity,
            },
            Opcode::SemaphoreAcquire => Operation::SemaphoreAcquire {
                key,
                permit: kivi_state::PermitId::from_bytes(self.permit),
                owner: self.owner,
                qty: self.qty,
            },
            Opcode::SemaphoreRelease => Operation::SemaphoreRelease {
                key,
                permit: kivi_state::PermitId::from_bytes(self.permit),
            },
            Opcode::SemaphoreInspect => Operation::SemaphoreInspect { key },
            Opcode::LeaseAcquire => Operation::LeaseAcquire {
                key,
                owner: self.owner,
                ttl_micros: self.ttl,
            },
            Opcode::LeaseRenew => Operation::LeaseRenew {
                key,
                owner: self.owner,
                fencing: kivi_state::FencingToken::from_u64(self.fencing),
                ttl_micros: self.ttl,
            },
            Opcode::LeaseRelease => Operation::LeaseRelease {
                key,
                owner: self.owner,
                fencing: kivi_state::FencingToken::from_u64(self.fencing),
            },
            Opcode::LeaseInspect => Operation::LeaseInspect { key },
            Opcode::StreamCreate => Operation::StreamCreate {
                key,
                stream: self.stream,
                shard: self.shard,
            },
            Opcode::StreamAppend => Operation::StreamAppend {
                key,
                partition: self.partition,
                payload: bytes::Bytes::from(self.value.unwrap_or_default()),
            },
            Opcode::StreamRead => Operation::StreamRead {
                key,
                from_offset: self.offset,
                max_entries: u32::try_from(self.len).unwrap_or(u32::MAX),
            },
            Opcode::StreamTrim => Operation::StreamTrim {
                key,
                through_offset: self.offset,
            },
        };
        Some(operation)
    }
}

/// Maps a typed operation back to its opcode. Inverse of the operation
/// half of [`Request::into_operation`] with one deliberate many-to-one:
/// [`SetChunked`](Operation::SetChunked) shares [`Set`](Opcode::Set)'s
/// shape. Both store bytes and both answer `Stored{version}`; the response
/// carries no representation, and chunked commits arrive over the
/// streaming COMMIT frame (never as legacy `Set` frames), so no decoder
/// can confuse the two. Response shaping and WAL records never guess.
#[must_use]
pub fn operation_opcode(operation: &Operation) -> Opcode {
    match operation {
        Operation::Get { .. } => Opcode::Get,
        Operation::Set { .. } | Operation::SetChunked { .. } => Opcode::Set,
        Operation::Delete { .. } => Opcode::Delete,
        Operation::Exists { .. } => Opcode::Exists,
        Operation::CounterGet { .. } => Opcode::CounterGet,
        Operation::CounterAdd { .. } => Opcode::CounterAdd,
        Operation::ExpireAt { .. } => Opcode::ExpireAt,
        Operation::PersistExpiry { .. } => Opcode::PersistExpiry,
        Operation::GetExpiry { .. } => Opcode::GetExpiry,
        Operation::SetRange { .. } => Opcode::SetRange,
        Operation::GetRange { .. } => Opcode::GetRange,
        Operation::BytesLength { .. } => Opcode::BytesLength,
        Operation::GetVersion { .. } => Opcode::GetVersion,
        Operation::SetConditional { .. } | Operation::SetConditionalChunked { .. } => {
            Opcode::SetConditional
        }
        // Transaction steps ride the normal propose path with dedup and
        // response shaping; each shapes as its own opcode. Local commits
        // shape as their parent batch opcode.
        Operation::TxnPrepare { .. } => Opcode::TxnPrepare,
        Operation::TxnFinalize { .. } => Opcode::TxnFinalize,
        Operation::TxnCommitLocal { .. } => Opcode::AtomicBatch,
        Operation::CommutativeGet { .. } => Opcode::CommutativeGet,
        Operation::CommutativeAdd { .. } => Opcode::CommutativeAdd,
        Operation::BoundedCounterCreate { .. } => Opcode::BoundedCounterCreate,
        Operation::BoundedCounterAdd { .. } => Opcode::BoundedCounterAdd,
        Operation::BoundedCounterGet { .. } => Opcode::BoundedCounterGet,
        Operation::EscrowTransfer { .. } => Opcode::EscrowTransfer,
        Operation::SemaphoreCreate { .. } => Opcode::SemaphoreCreate,
        Operation::SemaphoreAcquire { .. } => Opcode::SemaphoreAcquire,
        Operation::SemaphoreRelease { .. } => Opcode::SemaphoreRelease,
        Operation::SemaphoreInspect { .. } => Opcode::SemaphoreInspect,
        Operation::LeaseAcquire { .. } => Opcode::LeaseAcquire,
        Operation::LeaseRenew { .. } => Opcode::LeaseRenew,
        Operation::LeaseRelease { .. } => Opcode::LeaseRelease,
        Operation::LeaseInspect { .. } => Opcode::LeaseInspect,
        Operation::StreamCreate { .. } => Opcode::StreamCreate,
        Operation::StreamAppend { .. } => Opcode::StreamAppend,
        Operation::StreamRead { .. } => Opcode::StreamRead,
        Operation::StreamTrim { .. } => Opcode::StreamTrim,
    }
}

/// Maps a persisted mutation back to its originating opcode discriminant
/// (response shaping + replay mapping). The single canonical copy: the
/// engine tablet path and the consensus state machine both derive from
/// the originating request through [`Mutation`], so they cannot
/// disagree. The `is_persist_expiry` flag is the same one
/// [`kivi_state::outcome_for`] takes (both derive from the single
/// originating request).
#[must_use]
pub fn mutation_opcode(mutation: &Mutation, is_persist_expiry: bool) -> Opcode {
    match mutation {
        // Chunked roots answer exactly like inline stores: the response
        // shapes `Stored{version}` either way. Range patches answer the
        // same `Stored{version}` shape.
        Mutation::PutBytes { .. } | Mutation::ReplaceChunkedRoot { .. } => Opcode::Set,
        Mutation::PutBytesWithExpiry { .. } | Mutation::ReplaceChunkedRootWithExpiry { .. } => {
            Opcode::SetConditional
        }
        Mutation::SpliceBytes { .. } => Opcode::SetRange,
        Mutation::Delete { .. } => Opcode::Delete,
        Mutation::CounterAdd { .. } => Opcode::CounterAdd,
        Mutation::SetExpiry { .. } => {
            if is_persist_expiry {
                Opcode::PersistExpiry
            } else {
                Opcode::ExpireAt
            }
        }
        // Transaction steps shape as their parent batch opcode.
        Mutation::TxnPrepare { .. }
        | Mutation::TxnFinalize { .. }
        | Mutation::TxnCommitLocal { .. } => Opcode::AtomicBatch,
        Mutation::CommutativeAdd { .. } => Opcode::CommutativeAdd,
        Mutation::BoundedCreate { .. } => Opcode::BoundedCounterCreate,
        Mutation::BoundedAdd { .. } => Opcode::BoundedCounterAdd,
        Mutation::EscrowSetShare { .. } => Opcode::EscrowTransfer,
        Mutation::SemaphoreCreate { .. } => Opcode::SemaphoreCreate,
        Mutation::SemaphoreAcquire { .. } => Opcode::SemaphoreAcquire,
        Mutation::SemaphoreRelease { .. } => Opcode::SemaphoreRelease,
        Mutation::LeaseAcquire { .. } => Opcode::LeaseAcquire,
        Mutation::LeaseRenew { .. } => Opcode::LeaseRenew,
        Mutation::LeaseRelease { .. } => Opcode::LeaseRelease,
        Mutation::StreamCreate { .. } => Opcode::StreamCreate,
        Mutation::StreamAppend { .. } => Opcode::StreamAppend,
        Mutation::StreamTrim { .. } => Opcode::StreamTrim,
    }
}

/// Typed response body. `Ok` shapes pair with the request opcode (also
/// encoded, so decoders never guess); redirect shapes pair with
/// `StaleRoute`/`NotLocal`; diagnostics pair with the remaining errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResponseBody {
    /// Found bytes.
    Value(Vec<u8>),
    /// Stored with the new version.
    Stored {
        /// Version after the store.
        version: u64,
    },
    /// Removal outcome.
    Deleted {
        /// Whether a live object existed.
        existed: bool,
    },
    /// Existence probe answer.
    Exists(bool),
    /// Found counter.
    Counter(i64),
    /// Counter stored with the new value and version.
    CounterUpdated {
        /// Value after the addition.
        value: i64,
        /// Version after the addition.
        version: u64,
    },
    /// Expiry attach outcome.
    ExpirySet {
        /// Whether the key took the expiry.
        applied: bool,
    },
    /// Expiry clear outcome.
    ExpiryPersisted {
        /// Whether an expiry was removed.
        removed: bool,
    },
    /// Found expiry timestamp (micros); never `NEVER` (see `ExpiryNever`).
    ExpiryAt(u64),
    /// The key is live and immortal (no expiry attached).
    ExpiryNever,
    /// Logical byte length (pairs with `BytesLength`).
    Length(u64),
    /// Logical version (pairs with `GetVersion`).
    Version(u64),
    /// Conditional-store outcome (pairs with `SetConditional`).
    ConditionalSet {
        /// Whether the condition held and the value was stored.
        applied: bool,
        /// Version after the store (zero when not applied).
        version: u64,
    },
    /// One bounded scan page (pairs with `Scan`).
    ScanPage {
        /// Entries in scan order.
        entries: Vec<ScanEntryBody>,
        /// Whether no further keys remain in the range in scan order.
        exhausted: bool,
        /// Last emitted key, if any (cursor resume point).
        last_key: Option<Vec<u8>>,
        /// Serving tablet (route hint for the next page; hints never
        /// authorize, the server re-routes every page by key).
        tablet: u64,
        /// Serving tablet's range start (inclusive).
        range_start: Vec<u8>,
        /// Serving tablet's range end (exclusive; `None` = `+∞`).
        range_end: Option<Vec<u8>>,
        /// Serving directory version (staleness hint only).
        dir_version: u64,
    },
    /// Atomic batch committed (pairs with `AtomicBatch`): one version per
    /// write in request order (`None` for deletes, which carry no version).
    AtomicCommitted {
        /// Resulting versions in request order.
        versions: Vec<Option<u64>>,
    },
    /// Transaction intent reserved (pairs with `TxnPrepare`). Carries the
    /// serving tablet so drivers learn key → tablet placement without
    /// extra probes.
    TxnPrepared {
        /// Serving tablet that reserved the intent.
        tablet: u64,
        /// Serving directory version (staleness hint only).
        dir_version: u64,
    },
    /// One key's transaction intent resolved (pairs with `TxnFinalize`).
    TxnFinalized {
        /// Whether a prepared write was applied (`false` for aborts and
        /// idempotent replays with no intent left).
        applied: bool,
        /// Version after the applied write (zero when nothing applied).
        version: u64,
        /// Serving tablet that resolved the intent (drivers verify lineage:
        /// a finalize served elsewhere means the tablet moved mid-txn).
        tablet: u64,
    },
    /// Commutative addition applied (pairs with `CommutativeAdd`): no
    /// ordinal, by design — permutations are indistinguishable.
    CommutativeApplied,
    /// Commutative counter sum (pairs with `CommutativeGet`).
    CommutativeValue(i64),
    /// Bounded-counter write outcome (pairs with `BoundedCounterAdd`).
    BoundedUpdated {
        /// Value after the addition.
        value: i64,
        /// Version after the update.
        version: u64,
    },
    /// Bounded-counter read answer (pairs with `BoundedCounterGet`;
    /// absent keys arrive with `NotFound` status instead).
    BoundedValue {
        /// Current value.
        value: i64,
        /// Global capacity.
        capacity: u64,
        /// Locally owned escrow share lower bound.
        share_min: i64,
        /// Locally owned escrow share upper bound.
        share_max: i64,
    },
    /// Semaphore acquisition applied (pairs with `SemaphoreAcquire`).
    SemaphoreAcquired,
    /// Semaphore release outcome (pairs with `SemaphoreRelease`).
    SemaphoreReleased {
        /// Whether a live permit was released.
        released: bool,
    },
    /// Semaphore load answer (pairs with `SemaphoreInspect`).
    SemaphoreLoad {
        /// Total outstanding quantity.
        outstanding: u64,
        /// Total capacity.
        capacity: u64,
    },
    /// Lease grant outcome (pairs with `LeaseAcquire`).
    LeaseAcquired {
        /// Fencing token issued with this grant.
        fencing: u64,
        /// Logical expiry of the grant.
        expires_at: u64,
    },
    /// Lease renewal outcome (pairs with `LeaseRenew`; same token).
    LeaseRenewed {
        /// Fencing token of the continuing grant.
        fencing: u64,
        /// New logical expiry of the grant.
        expires_at: u64,
    },
    /// Lease release outcome (pairs with `LeaseRelease`).
    LeaseReleased {
        /// Whether a live holding was released.
        released: bool,
    },
    /// Lease inspection answer (pairs with `LeaseInspect`).
    LeaseInfo {
        /// Live holder's owner (`None` when free or expired).
        owner: Option<u64>,
        /// Live holder's fencing token (zero when no live holder).
        fencing: u64,
        /// Live holder's logical expiry (zero when no live holder).
        expires_at: u64,
        /// Next fencing token to be issued.
        next_fencing: u64,
    },
    /// Stream shard created (pairs with `StreamCreate`).
    StreamCreated,
    /// Stream append outcome (pairs with `StreamAppend`).
    StreamAppended {
        /// Assigned shard-local offset.
        offset: u64,
    },
    /// Stream read page (pairs with `StreamRead`): entries in shard order.
    StreamEntries {
        /// Entries as `(offset, partition, payload)` triples.
        entries: Vec<StreamEntryBody>,
        /// Shard's next offset (exclusive upper bound of the log).
        next_offset: u64,
    },
    /// Stream trim outcome (pairs with `StreamTrim`).
    StreamTrimmed {
        /// Entries removed.
        removed: u64,
    },
    /// Retry directly at the attached authority.
    Redirect(RedirectInfo),
    /// Machine code plus optional human diagnostic (never Rust internals).
    Diagnostic(String),
}

/// One stream entry in a `StreamRead` response page.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StreamEntryBody {
    /// Shard-local offset.
    pub offset: u64,
    /// Partition key that routed this entry here.
    pub partition: Vec<u8>,
    /// Entry payload bytes.
    pub payload: Vec<u8>,
}

/// One typed native response: stable status plus body.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Response {
    /// Machine-readable outcome.
    pub status: Status,
    /// Outcome payload.
    pub body: ResponseBody,
    /// Read proof attached to served reads (`Ok`/`NotFound` on read
    /// opcodes): the authority and applied position the read served from,
    /// for `AtLeast` chaining and operator diagnosis. `None` on writes
    /// (which carry versions instead), on errors, and on responses from
    /// pre-proof servers (treated as no evidence, never as freshness).
    /// Rides a trailer (see [`Response::encode`]).
    pub proof: Option<ReadReceipt>,
}

// ---------------------------------------------------------------------------
// Wire codecs: explicit little-endian primitives, length-prefixed blobs.
// ---------------------------------------------------------------------------

fn push_u8(out: &mut Vec<u8>, value: u8) {
    out.push(value);
}

fn push_u16(out: &mut Vec<u8>, value: u16) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn push_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn push_u64(out: &mut Vec<u8>, value: u64) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn push_u128(out: &mut Vec<u8>, value: u128) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn push_i64(out: &mut Vec<u8>, value: i64) {
    out.extend_from_slice(&value.to_le_bytes());
}

/// Length-prefixed bytes (`u32` LE length + raw). Lengths of live allocations
/// always fit; the clamp only bounds absurd constructions on encode, while
/// decode strictly validates against actual input.
fn push_blob(out: &mut Vec<u8>, bytes: &[u8]) {
    push_u32(out, u32::try_from(bytes.len()).unwrap_or(u32::MAX));
    out.extend_from_slice(bytes);
}

/// Checked cursor over a payload under decode. Every read validates bounds
/// first: truncated input is an error, never a panic or an over-read.
struct Cursor<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    fn remaining(&self) -> usize {
        self.buf.len() - self.pos
    }

    fn take(&mut self, len: usize, context: &'static str) -> Result<&'a [u8], ProtocolError> {
        if self.remaining() < len {
            return Err(ProtocolError::Malformed { context });
        }
        let start = self.pos;
        self.pos += len;
        Ok(&self.buf[start..start + len])
    }

    fn u8(&mut self, context: &'static str) -> Result<u8, ProtocolError> {
        Ok(self.take(1, context)?[0])
    }

    fn u16(&mut self, context: &'static str) -> Result<u16, ProtocolError> {
        Ok(u16::from_le_bytes(
            self.take(2, context)?
                .try_into()
                .map_err(|_| ProtocolError::Malformed { context })?,
        ))
    }

    fn u32(&mut self, context: &'static str) -> Result<u32, ProtocolError> {
        Ok(u32::from_le_bytes(
            self.take(4, context)?
                .try_into()
                .map_err(|_| ProtocolError::Malformed { context })?,
        ))
    }

    fn u64(&mut self, context: &'static str) -> Result<u64, ProtocolError> {
        Ok(u64::from_le_bytes(
            self.take(8, context)?
                .try_into()
                .map_err(|_| ProtocolError::Malformed { context })?,
        ))
    }

    fn u128(&mut self, context: &'static str) -> Result<u128, ProtocolError> {
        Ok(u128::from_le_bytes(
            self.take(16, context)?
                .try_into()
                .map_err(|_| ProtocolError::Malformed { context })?,
        ))
    }

    fn i64(&mut self, context: &'static str) -> Result<i64, ProtocolError> {
        Ok(i64::from_le_bytes(
            self.take(8, context)?
                .try_into()
                .map_err(|_| ProtocolError::Malformed { context })?,
        ))
    }

    fn blob(&mut self, context: &'static str) -> Result<&'a [u8], ProtocolError> {
        let len = self.u32(context)? as usize;
        self.take(len, context)
    }

    fn blob_u8(&mut self, context: &'static str) -> Result<&'a [u8], ProtocolError> {
        let len = self.u8(context)? as usize;
        self.take(len, context)
    }

    fn end(&self, context: &'static str) -> Result<(), ProtocolError> {
        if self.remaining() == 0 {
            Ok(())
        } else {
            Err(ProtocolError::Malformed { context })
        }
    }
}

impl ClientHello {
    /// Encodes the handshake opener.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(4 + 8 + 8 + 8);
        push_u32(&mut out, self.max_frame);
        push_u64(&mut out, self.required_caps.bits());
        push_u64(&mut out, self.optional_caps.bits());
        push_u64(
            &mut out,
            self.desired_worker.map_or(u64::MAX, WorkerId::as_u64),
        );
        out
    }

    /// Decodes a handshake opener, requiring exact consumption.
    ///
    /// # Errors
    ///
    /// Returns [`ProtocolError`] on truncated or trailing input.
    pub fn decode(input: &[u8]) -> Result<Self, ProtocolError> {
        let mut cursor = Cursor::new(input);
        let hello = Self {
            max_frame: cursor.u32("client-hello")?,
            // Retained, never truncated: negotiation must see unknown bits.
            required_caps: Capabilities::from_bits_retain(cursor.u64("client-hello")?),
            optional_caps: Capabilities::from_bits_retain(cursor.u64("client-hello")?),
            desired_worker: {
                let raw = cursor.u64("client-hello")?;
                (raw != u64::MAX).then(|| WorkerId::from_u64(raw))
            },
        };
        cursor.end("client-hello")?;
        Ok(hello)
    }
}

impl ServerHello {
    /// Encodes the handshake answer.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(64 + self.endpoints.len() * 24);
        push_u16(&mut out, self.major);
        push_u16(&mut out, self.minor);
        push_u64(&mut out, self.caps.bits());
        push_u128(&mut out, self.cluster.as_u128());
        push_u64(&mut out, self.node.as_u64());
        push_u64(&mut out, self.incarnation.as_u64());
        push_u64(&mut out, self.worker.as_u64());
        push_u64(&mut out, self.dir_version.as_u64());
        push_u32(&mut out, self.max_frame);
        push_u16(
            &mut out,
            u16::try_from(self.endpoints.len()).unwrap_or(u16::MAX),
        );
        for (worker, endpoint) in &self.endpoints {
            push_u64(&mut out, worker.as_u64());
            let bytes = endpoint.as_bytes();
            push_u8(&mut out, u8::try_from(bytes.len()).unwrap_or(u8::MAX));
            out.extend_from_slice(&bytes[..bytes.len().min(255)]);
        }
        out
    }

    /// Decodes a handshake answer, requiring exact consumption.
    ///
    /// # Errors
    ///
    /// Returns [`ProtocolError`] on truncated, trailing, or non-UTF-8 input.
    pub fn decode(input: &[u8]) -> Result<Self, ProtocolError> {
        let mut cursor = Cursor::new(input);
        let hello = Self {
            major: cursor.u16("server-hello")?,
            minor: cursor.u16("server-hello")?,
            caps: Capabilities::from_bits_retain(cursor.u64("server-hello")?),
            cluster: ClusterId::from_u128(cursor.u128("server-hello")?),
            node: NodeId::from_u64(cursor.u64("server-hello")?),
            incarnation: NodeIncarnation::from_u64(cursor.u64("server-hello")?),
            worker: WorkerId::from_u64(cursor.u64("server-hello")?),
            dir_version: DirectoryVersion::from_u64(cursor.u64("server-hello")?),
            max_frame: cursor.u32("server-hello")?,
            endpoints: {
                let count = cursor.u16("server-hello")? as usize;
                let mut endpoints = Vec::with_capacity(count.min(1024));
                for _ in 0..count {
                    let worker = WorkerId::from_u64(cursor.u64("server-hello")?);
                    let endpoint = core::str::from_utf8(cursor.blob_u8("server-hello")?)
                        .map_err(|_| ProtocolError::Malformed {
                            context: "server-hello",
                        })?
                        .to_owned();
                    endpoints.push((worker, endpoint));
                }
                endpoints
            },
        };
        cursor.end("server-hello")?;
        Ok(hello)
    }
}

/// Encodes a partition range for redirect caching (hash prefixes as
/// `u128 LE + len`; ordered intervals as length-prefixed bounds with an
/// empty marker for `+∞`).
fn encode_range(range: &PartitionRange, out: &mut Vec<u8>) {
    match range {
        PartitionRange::Hash(prefix) => {
            push_u8(out, 1);
            push_u128(out, prefix.bits());
            push_u8(out, prefix.prefix_len());
        }
        PartitionRange::Ordered(interval) => {
            push_u8(out, 2);
            push_blob(out, interval.start());
            match interval.end() {
                None => push_u8(out, 0),
                Some(bound) => {
                    push_u8(out, 1);
                    push_blob(out, bound);
                }
            }
        }
    }
}

/// Decodes a redirect range.
///
/// # Errors
///
/// Returns [`ProtocolError`] on unknown tags, truncation, or invalid ranges.
fn decode_range(cursor: &mut Cursor<'_>) -> Result<PartitionRange, ProtocolError> {
    const CONTEXT: &str = "redirect range";
    match cursor.u8(CONTEXT)? {
        1 => {
            let bytes = cursor.take(16, CONTEXT)?;
            let bits = u128::from_le_bytes(
                bytes
                    .try_into()
                    .map_err(|_| ProtocolError::Malformed { context: CONTEXT })?,
            );
            let len = cursor.u8(CONTEXT)?;
            HashPrefix::new(bits, len)
                .map(PartitionRange::Hash)
                .map_err(|_| ProtocolError::Malformed { context: CONTEXT })
        }
        2 => {
            let start = cursor.blob(CONTEXT)?.to_vec();
            let end = match cursor.u8(CONTEXT)? {
                0 => None,
                1 => Some(cursor.blob(CONTEXT)?.to_vec()),
                _ => return Err(ProtocolError::Malformed { context: CONTEXT }),
            };
            OrderedRange::new(start, end)
                .map(PartitionRange::Ordered)
                .map_err(|_| ProtocolError::Malformed { context: CONTEXT })
        }
        _ => Err(ProtocolError::Malformed { context: CONTEXT }),
    }
}

impl RedirectInfo {
    /// Encodes redirect info for `StaleRoute`/`NotLocal` bodies.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(64 + self.endpoint.len());
        push_u64(&mut out, self.dir_version.as_u64());
        push_u64(&mut out, self.tablet.as_u64());
        push_u64(&mut out, self.epoch.as_u64());
        push_u64(&mut out, self.worker.as_u64());
        let bytes = self.endpoint.as_bytes();
        push_u8(&mut out, u8::try_from(bytes.len()).unwrap_or(u8::MAX));
        out.extend_from_slice(&bytes[..bytes.len().min(255)]);
        encode_range(&self.range, &mut out);
        out
    }

    /// Decodes redirect info.
    ///
    /// # Errors
    ///
    /// Returns [`ProtocolError`] on truncation or invalid contents.
    pub fn decode(input: &[u8]) -> Result<Self, ProtocolError> {
        let mut cursor = Cursor::new(input);
        let info = Self {
            dir_version: DirectoryVersion::from_u64(cursor.u64("redirect")?),
            tablet: TabletId::from_u64(cursor.u64("redirect")?),
            epoch: TabletEpoch::from_u64(cursor.u64("redirect")?),
            worker: WorkerId::from_u64(cursor.u64("redirect")?),
            endpoint: core::str::from_utf8(cursor.blob_u8("redirect")?)
                .map_err(|_| ProtocolError::Malformed {
                    context: "redirect",
                })?
                .to_owned(),
            range: decode_range(&mut cursor)?,
        };
        cursor.end("redirect")?;
        Ok(info)
    }
}

impl Request {
    /// Encodes one request payload (framing header added by the caller).
    #[must_use]
    #[allow(clippy::too_many_lines)]
    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(32 + self.key.len());
        push_u64(&mut out, self.namespace.as_u64());
        push_u8(&mut out, self.opcode.as_u8());
        match self.hint {
            None => push_u8(&mut out, 0),
            Some(hint) => {
                push_u8(&mut out, 1);
                push_u64(&mut out, hint.tablet.as_u64());
                push_u64(&mut out, hint.epoch.as_u64());
                push_u64(&mut out, hint.worker.as_u64());
            }
        }
        push_blob(&mut out, &self.key);
        match self.opcode {
            Opcode::Set => push_blob(&mut out, self.value.as_deref().unwrap_or_default()),
            // All three counters share the delta wire shape; the opcode
            // names the semantics, never the encoding.
            Opcode::CounterAdd | Opcode::CommutativeAdd | Opcode::BoundedCounterAdd => {
                push_i64(&mut out, self.delta);
            }
            Opcode::ExpireAt => push_u64(&mut out, self.expiry),
            Opcode::SetRange => {
                push_u64(&mut out, self.offset);
                push_blob(&mut out, self.value.as_deref().unwrap_or_default());
            }
            // Slices share the offset+length wire shape (byte ranges and
            // shard reads alike); the opcode names the semantics.
            Opcode::GetRange | Opcode::StreamRead => {
                push_u64(&mut out, self.offset);
                push_u64(&mut out, self.len);
            }
            Opcode::SetConditional => {
                push_u8(&mut out, self.condition);
                push_u8(&mut out, self.expiry_policy);
                // `expiry` rides only for `EXPIRY_AT`; the other policies
                // ignore it (zero on encode when unused).
                if self.expiry_policy == EXPIRY_AT {
                    push_u64(&mut out, self.expiry);
                }
                push_blob(&mut out, self.value.as_deref().unwrap_or_default());
            }
            Opcode::Scan => {
                match &self.scan_start {
                    None => push_u8(&mut out, 0),
                    Some(start) => {
                        push_u8(&mut out, 1);
                        push_blob(&mut out, start);
                    }
                }
                match &self.scan_end {
                    None => push_u8(&mut out, 0),
                    Some(end) => {
                        push_u8(&mut out, 1);
                        push_blob(&mut out, end);
                    }
                }
                push_u8(&mut out, self.scan_direction);
                push_u32(&mut out, self.scan_max_items);
                push_u32(&mut out, self.scan_max_bytes);
                push_u8(&mut out, self.scan_projection);
                push_u8(&mut out, self.scan_consistency);
            }
            Opcode::AtomicBatch => {
                out.extend_from_slice(&self.batch_txn);
                push_u16(
                    &mut out,
                    u16::try_from(self.batch_writes.len()).unwrap_or(u16::MAX),
                );
                for write in &self.batch_writes {
                    push_blob(&mut out, &write.key);
                    push_u8(&mut out, write.kind);
                    push_blob(&mut out, &write.value);
                    push_i64(&mut out, write.delta);
                    push_u8(&mut out, write.expect);
                    push_u64(&mut out, write.expect_version);
                }
            }
            Opcode::TxnPrepare => {
                // One prepare carries exactly one write: the batch tail
                // with a single entry plus the coordinator.
                out.extend_from_slice(&self.batch_txn);
                push_u64(&mut out, self.txn_coordinator);
                out.extend_from_slice(&self.txn_digest);
                push_u16(
                    &mut out,
                    u16::try_from(self.batch_writes.len()).unwrap_or(u16::MAX),
                );
                for write in &self.batch_writes {
                    push_blob(&mut out, &write.key);
                    push_u8(&mut out, write.kind);
                    push_blob(&mut out, &write.value);
                    push_i64(&mut out, write.delta);
                    push_u8(&mut out, write.expect);
                    push_u64(&mut out, write.expect_version);
                }
            }
            Opcode::TxnFinalize => {
                out.extend_from_slice(&self.batch_txn);
                push_u8(&mut out, u8::from(self.txn_commit));
                out.extend_from_slice(&self.txn_digest);
            }
            Opcode::BoundedCounterCreate => {
                push_u64(&mut out, self.capacity);
                push_u64(&mut out, self.holder);
            }
            Opcode::EscrowTransfer => {
                push_i64(&mut out, self.share_min);
                push_i64(&mut out, self.share_max);
            }
            Opcode::SemaphoreCreate => push_u64(&mut out, self.capacity),
            Opcode::SemaphoreAcquire => {
                out.extend_from_slice(&self.permit);
                push_u64(&mut out, self.owner);
                push_u64(&mut out, self.qty);
            }
            Opcode::SemaphoreRelease => {
                out.extend_from_slice(&self.permit);
            }
            Opcode::LeaseAcquire => {
                push_u64(&mut out, self.owner);
                push_u64(&mut out, self.ttl);
            }
            Opcode::LeaseRenew => {
                push_u64(&mut out, self.owner);
                push_u64(&mut out, self.fencing);
                push_u64(&mut out, self.ttl);
            }
            Opcode::LeaseRelease => {
                push_u64(&mut out, self.owner);
                push_u64(&mut out, self.fencing);
            }
            Opcode::StreamCreate => {
                out.extend_from_slice(&self.stream);
                push_u32(&mut out, self.shard);
            }
            Opcode::StreamAppend => {
                push_blob(&mut out, &self.partition);
                push_blob(&mut out, self.value.as_deref().unwrap_or_default());
            }
            Opcode::StreamTrim => {
                push_u64(&mut out, self.offset);
            }
            Opcode::Get
            | Opcode::Delete
            | Opcode::Exists
            | Opcode::CounterGet
            | Opcode::PersistExpiry
            | Opcode::GetExpiry
            | Opcode::GetStream
            | Opcode::GetVersion
            | Opcode::CommutativeGet
            | Opcode::BoundedCounterGet
            | Opcode::SemaphoreInspect
            | Opcode::LeaseInspect
            | Opcode::BytesLength => {}
        }
        // Retry identity rides last so pre-identity decoders fail cleanly on
        // length: flag 0 means "no identity, end of request".
        match self.identity {
            None => push_u8(&mut out, 0),
            Some(identity) => {
                push_u8(&mut out, 1);
                push_u128(&mut out, identity.session().as_u128());
                push_u64(&mut out, identity.seq().as_u64());
                push_u64(&mut out, self.ack_floor.as_u64());
            }
        }
        // Read-contract trailer: `Latest` writes nothing (byte-identical
        // to pre-contract encodings — old servers keep serving it as the
        // only contract they know). Any other contract appends its tag,
        // which old servers reject loudly as trailing bytes instead of
        // silently serving stronger-or-weaker semantics. Tags mirror
        // `kivi-codec`'s canonical `ReadContract` tags (0 is accepted on
        // decode as `Latest` but never written).
        match self.contract {
            ReadContract::Latest => {}
            ReadContract::AtLeast(token) => {
                push_u8(&mut out, 1);
                push_u64(&mut out, token.tablet().as_u64());
                push_u64(&mut out, token.epoch().as_u64());
                push_u64(&mut out, token.position().as_u64());
            }
            ReadContract::BoundedStale { max_staleness } => {
                push_u8(&mut out, 2);
                push_u64(
                    &mut out,
                    u64::try_from(max_staleness.as_micros()).unwrap_or(u64::MAX),
                );
            }
            ReadContract::Any => push_u8(&mut out, 3),
        }
        out
    }

    /// Decodes one request payload, requiring exact consumption and
    /// opcode-consistent shape.
    ///
    /// # Errors
    ///
    /// Returns [`ProtocolError`] on unknown opcodes, truncation, trailing
    /// bytes, or non-UTF-8 where strings are required (none here — keys
    /// stay byte-exact).
    #[allow(clippy::too_many_lines)]
    pub fn decode(input: &[u8]) -> Result<Self, ProtocolError> {
        const CONTEXT: &str = "request";
        let mut cursor = Cursor::new(input);
        let namespace = NamespaceId::from_u64(cursor.u64(CONTEXT)?);
        let raw_opcode = cursor.u8(CONTEXT)?;
        let opcode = Opcode::from_u8(raw_opcode)
            .ok_or(ProtocolError::UnknownOpcode { opcode: raw_opcode })?;
        let hint = match cursor.u8(CONTEXT)? {
            0 => None,
            1 => Some(RouteHint {
                tablet: TabletId::from_u64(cursor.u64(CONTEXT)?),
                epoch: TabletEpoch::from_u64(cursor.u64(CONTEXT)?),
                worker: WorkerId::from_u64(cursor.u64(CONTEXT)?),
            }),
            _ => return Err(ProtocolError::Malformed { context: CONTEXT }),
        };
        let key = cursor.blob(CONTEXT)?.to_vec();
        let mut request = Self {
            namespace,
            opcode,
            hint,
            key,
            contract: kivi_types::ReadContract::Latest,
            value: None,
            delta: 0,
            expiry: 0,
            offset: 0,
            len: 0,
            condition: COND_ALWAYS,
            expiry_policy: EXPIRY_CLEAR,
            identity: None,
            ack_floor: RequestSeq::from_u64(0),
            scan_start: None,
            scan_end: None,
            scan_direction: SCAN_FORWARD,
            scan_max_items: 0,
            scan_max_bytes: 0,
            scan_projection: SCAN_KEYS_ONLY,
            scan_consistency: SCAN_LATEST_PER_TABLET,
            batch_txn: [0u8; 16],
            batch_writes: Vec::new(),
            txn_coordinator: 0,
            txn_commit: false,
            txn_digest: [0u8; 32],
            capacity: 0,
            holder: 0,
            permit: [0u8; 16],
            owner: 0,
            qty: 0,
            fencing: 0,
            ttl: 0,
            stream: [0u8; 16],
            shard: 0,
            partition: Vec::new(),
            share_min: 0,
            share_max: 0,
        };
        match opcode {
            Opcode::Set => {
                request.value = Some(cursor.blob(CONTEXT)?.to_vec());
            }
            // All three counters share the delta wire shape.
            Opcode::CounterAdd | Opcode::CommutativeAdd | Opcode::BoundedCounterAdd => {
                request.delta = cursor.i64(CONTEXT)?;
            }
            Opcode::ExpireAt => {
                request.expiry = cursor.u64(CONTEXT)?;
            }
            Opcode::SetRange => {
                request.offset = cursor.u64(CONTEXT)?;
                request.value = Some(cursor.blob(CONTEXT)?.to_vec());
            }
            Opcode::GetRange | Opcode::StreamRead => {
                request.offset = cursor.u64(CONTEXT)?;
                request.len = cursor.u64(CONTEXT)?;
            }
            Opcode::SetConditional => {
                request.condition = cursor.u8(CONTEXT)?;
                if !matches!(
                    request.condition,
                    COND_ALWAYS | COND_IF_ABSENT | COND_IF_PRESENT
                ) {
                    return Err(ProtocolError::Malformed { context: CONTEXT });
                }
                request.expiry_policy = cursor.u8(CONTEXT)?;
                if !matches!(
                    request.expiry_policy,
                    EXPIRY_CLEAR | EXPIRY_KEEP | EXPIRY_AT
                ) {
                    return Err(ProtocolError::Malformed { context: CONTEXT });
                }
                if request.expiry_policy == EXPIRY_AT {
                    request.expiry = cursor.u64(CONTEXT)?;
                }
                request.value = Some(cursor.blob(CONTEXT)?.to_vec());
            }
            Opcode::Scan => {
                request.scan_start = match cursor.u8(CONTEXT)? {
                    0 => None,
                    1 => Some(cursor.blob(CONTEXT)?.to_vec()),
                    _ => return Err(ProtocolError::Malformed { context: CONTEXT }),
                };
                request.scan_end = match cursor.u8(CONTEXT)? {
                    0 => None,
                    1 => Some(cursor.blob(CONTEXT)?.to_vec()),
                    _ => return Err(ProtocolError::Malformed { context: CONTEXT }),
                };
                request.scan_direction = cursor.u8(CONTEXT)?;
                if !matches!(request.scan_direction, SCAN_FORWARD | SCAN_REVERSE) {
                    return Err(ProtocolError::Malformed { context: CONTEXT });
                }
                request.scan_max_items = cursor.u32(CONTEXT)?;
                request.scan_max_bytes = cursor.u32(CONTEXT)?;
                request.scan_projection = cursor.u8(CONTEXT)?;
                if !matches!(
                    request.scan_projection,
                    SCAN_KEYS_ONLY | SCAN_KEYS_AND_VALUES
                ) {
                    return Err(ProtocolError::Malformed { context: CONTEXT });
                }
                request.scan_consistency = cursor.u8(CONTEXT)?;
                if !matches!(request.scan_consistency, SCAN_LATEST_PER_TABLET | SCAN_ANY) {
                    return Err(ProtocolError::Malformed { context: CONTEXT });
                }
            }
            Opcode::AtomicBatch => {
                let raw = cursor.take(16, CONTEXT)?;
                request.batch_txn = raw
                    .try_into()
                    .map_err(|_| ProtocolError::Malformed { context: CONTEXT })?;
                request.batch_writes = decode_batch_writes(&mut cursor, 4096)?;
            }
            Opcode::TxnPrepare => {
                let raw = cursor.take(16, CONTEXT)?;
                request.batch_txn = raw
                    .try_into()
                    .map_err(|_| ProtocolError::Malformed { context: CONTEXT })?;
                request.txn_coordinator = cursor.u64(CONTEXT)?;
                let digest = cursor.take(32, CONTEXT)?;
                request.txn_digest = digest
                    .try_into()
                    .map_err(|_| ProtocolError::Malformed { context: CONTEXT })?;
                // One prepare reserves exactly one key: the batch tail
                // carries a single entry, never a set.
                let writes = decode_batch_writes(&mut cursor, 1)?;
                if writes.len() != 1 {
                    return Err(ProtocolError::Malformed { context: CONTEXT });
                }
                request.batch_writes = writes;
            }
            Opcode::TxnFinalize => {
                let raw = cursor.take(16, CONTEXT)?;
                request.batch_txn = raw
                    .try_into()
                    .map_err(|_| ProtocolError::Malformed { context: CONTEXT })?;
                request.txn_commit = match cursor.u8(CONTEXT)? {
                    0 => false,
                    1 => true,
                    _ => return Err(ProtocolError::Malformed { context: CONTEXT }),
                };
                let digest = cursor.take(32, CONTEXT)?;
                request.txn_digest = digest
                    .try_into()
                    .map_err(|_| ProtocolError::Malformed { context: CONTEXT })?;
            }
            Opcode::BoundedCounterCreate => {
                request.capacity = cursor.u64(CONTEXT)?;
                request.holder = cursor.u64(CONTEXT)?;
            }
            Opcode::EscrowTransfer => {
                request.share_min = cursor.i64(CONTEXT)?;
                request.share_max = cursor.i64(CONTEXT)?;
            }
            Opcode::SemaphoreCreate => {
                request.capacity = cursor.u64(CONTEXT)?;
            }
            Opcode::SemaphoreAcquire => {
                let raw = cursor.take(16, CONTEXT)?;
                request.permit = raw
                    .try_into()
                    .map_err(|_| ProtocolError::Malformed { context: CONTEXT })?;
                request.owner = cursor.u64(CONTEXT)?;
                request.qty = cursor.u64(CONTEXT)?;
            }
            Opcode::SemaphoreRelease => {
                let raw = cursor.take(16, CONTEXT)?;
                request.permit = raw
                    .try_into()
                    .map_err(|_| ProtocolError::Malformed { context: CONTEXT })?;
            }
            Opcode::LeaseAcquire => {
                request.owner = cursor.u64(CONTEXT)?;
                request.ttl = cursor.u64(CONTEXT)?;
            }
            Opcode::LeaseRenew => {
                request.owner = cursor.u64(CONTEXT)?;
                request.fencing = cursor.u64(CONTEXT)?;
                request.ttl = cursor.u64(CONTEXT)?;
            }
            Opcode::LeaseRelease => {
                request.owner = cursor.u64(CONTEXT)?;
                request.fencing = cursor.u64(CONTEXT)?;
            }
            Opcode::StreamCreate => {
                let raw = cursor.take(16, CONTEXT)?;
                request.stream = raw
                    .try_into()
                    .map_err(|_| ProtocolError::Malformed { context: CONTEXT })?;
                request.shard = cursor.u32(CONTEXT)?;
            }
            Opcode::StreamAppend => {
                request.partition = cursor.blob(CONTEXT)?.to_vec();
                request.value = Some(cursor.blob(CONTEXT)?.to_vec());
            }
            Opcode::StreamTrim => {
                request.offset = cursor.u64(CONTEXT)?;
            }
            Opcode::Get
            | Opcode::Delete
            | Opcode::Exists
            | Opcode::CounterGet
            | Opcode::PersistExpiry
            | Opcode::GetExpiry
            | Opcode::GetStream
            | Opcode::GetVersion
            | Opcode::CommutativeGet
            | Opcode::BoundedCounterGet
            | Opcode::SemaphoreInspect
            | Opcode::LeaseInspect
            | Opcode::BytesLength => {}
        }
        // Identity suffix: absent on pre-identity encodings (exact end),
        // otherwise flag 1 plus session/seq/ack. Anything else is malformed.
        if cursor.remaining() > 0 {
            match cursor.u8(CONTEXT)? {
                0 => {}
                1 => {
                    let session = kivi_types::SessionId::from_u128(cursor.u128(CONTEXT)?);
                    let seq = RequestSeq::from_u64(cursor.u64(CONTEXT)?);
                    let ack_floor = RequestSeq::from_u64(cursor.u64(CONTEXT)?);
                    request.identity = Some(RequestIdentity::new(session, seq));
                    request.ack_floor = ack_floor;
                }
                _ => return Err(ProtocolError::Malformed { context: CONTEXT }),
            }
        }
        // Contract trailer: absent means `Latest` (pre-contract clients
        // only knew linearizable reads). Present trailers decode by tag;
        // unknown tags are malformed, never guessed.
        if cursor.remaining() > 0 {
            request.contract = match cursor.u8(CONTEXT)? {
                0 => ReadContract::Latest,
                1 => {
                    let tablet = TabletId::from_u64(cursor.u64(CONTEXT)?);
                    let epoch = TabletEpoch::from_u64(cursor.u64(CONTEXT)?);
                    let position = CommitPosition::from_u64(cursor.u64(CONTEXT)?);
                    ReadContract::AtLeast(CommitToken::new(tablet, epoch, position))
                }
                2 => {
                    let micros = cursor.u64(CONTEXT)?;
                    ReadContract::BoundedStale {
                        max_staleness: core::time::Duration::from_micros(micros),
                    }
                }
                3 => ReadContract::Any,
                _ => return Err(ProtocolError::Malformed { context: CONTEXT }),
            };
        }
        cursor.end(CONTEXT)?;
        Ok(request)
    }
}

impl Response {
    /// Encodes one response payload. The opcode travels alongside the status
    /// so decoders never guess the body shape from connection state.
    #[must_use]
    #[allow(clippy::too_many_lines)]
    pub fn encode(&self, opcode: Opcode) -> Vec<u8> {
        let mut out = Vec::with_capacity(32);
        push_u16(&mut out, self.status.as_u16());
        push_u8(&mut out, opcode.as_u8());
        match &self.body {
            ResponseBody::Value(value) => push_blob(&mut out, value),
            ResponseBody::Stored { version } | ResponseBody::Version(version) => {
                push_u64(&mut out, *version);
            }
            ResponseBody::Deleted { existed } => push_u8(&mut out, u8::from(*existed)),
            ResponseBody::Exists(present) => push_u8(&mut out, u8::from(*present)),
            ResponseBody::Counter(value) | ResponseBody::CommutativeValue(value) => {
                push_i64(&mut out, *value);
            }
            ResponseBody::CounterUpdated { value, version }
            | ResponseBody::BoundedUpdated { value, version } => {
                push_i64(&mut out, *value);
                push_u64(&mut out, *version);
            }
            ResponseBody::ExpirySet { applied } => push_u8(&mut out, u8::from(*applied)),
            ResponseBody::ExpiryPersisted { removed } => push_u8(&mut out, u8::from(*removed)),
            ResponseBody::ExpiryAt(stamp) => {
                push_u8(&mut out, 1);
                push_u64(&mut out, *stamp);
            }
            ResponseBody::ExpiryNever => {
                push_u8(&mut out, 0);
            }
            ResponseBody::Length(len) => push_u64(&mut out, *len),
            ResponseBody::ConditionalSet { applied, version } => {
                push_u8(&mut out, u8::from(*applied));
                push_u64(&mut out, *version);
            }
            ResponseBody::TxnFinalized {
                applied,
                version,
                tablet,
            } => {
                push_u8(&mut out, u8::from(*applied));
                push_u64(&mut out, *version);
                push_u64(&mut out, *tablet);
            }
            ResponseBody::CommutativeApplied
            | ResponseBody::StreamCreated
            | ResponseBody::SemaphoreAcquired => {}
            ResponseBody::BoundedValue {
                value,
                capacity,
                share_min,
                share_max,
            } => {
                push_i64(&mut out, *value);
                push_u64(&mut out, *capacity);
                push_i64(&mut out, *share_min);
                push_i64(&mut out, *share_max);
            }
            ResponseBody::SemaphoreReleased { released }
            | ResponseBody::LeaseReleased { released } => {
                push_u8(&mut out, u8::from(*released));
            }
            ResponseBody::SemaphoreLoad {
                outstanding,
                capacity,
            } => {
                push_u64(&mut out, *outstanding);
                push_u64(&mut out, *capacity);
            }
            ResponseBody::LeaseAcquired {
                fencing,
                expires_at,
            }
            | ResponseBody::LeaseRenewed {
                fencing,
                expires_at,
            } => {
                push_u64(&mut out, *fencing);
                push_u64(&mut out, *expires_at);
            }
            ResponseBody::LeaseInfo {
                owner,
                fencing,
                expires_at,
                next_fencing,
            } => {
                match owner {
                    None => push_u8(&mut out, 0),
                    Some(owner) => {
                        push_u8(&mut out, 1);
                        push_u64(&mut out, *owner);
                    }
                }
                push_u64(&mut out, *fencing);
                push_u64(&mut out, *expires_at);
                push_u64(&mut out, *next_fencing);
            }
            ResponseBody::StreamAppended { offset } => {
                push_u64(&mut out, *offset);
            }
            ResponseBody::StreamEntries {
                entries,
                next_offset,
            } => {
                push_u32(&mut out, u32::try_from(entries.len()).unwrap_or(u32::MAX));
                for entry in entries {
                    push_u64(&mut out, entry.offset);
                    push_blob(&mut out, &entry.partition);
                    push_blob(&mut out, &entry.payload);
                }
                push_u64(&mut out, *next_offset);
            }
            ResponseBody::StreamTrimmed { removed } => {
                push_u64(&mut out, *removed);
            }
            ResponseBody::ScanPage {
                entries,
                exhausted,
                last_key,
                tablet,
                range_start,
                range_end,
                dir_version,
            } => {
                push_u32(&mut out, u32::try_from(entries.len()).unwrap_or(u32::MAX));
                for entry in entries {
                    push_blob(&mut out, &entry.key);
                    match &entry.value {
                        ScanValueBody::None => push_u8(&mut out, SCAN_VALUE_NONE),
                        ScanValueBody::Inline(bytes) => {
                            push_u8(&mut out, SCAN_VALUE_INLINE);
                            push_blob(&mut out, bytes);
                        }
                        ScanValueBody::Counter(counter) => {
                            push_u8(&mut out, SCAN_VALUE_COUNTER);
                            push_i64(&mut out, *counter);
                        }
                        ScanValueBody::Chunked {
                            manifest,
                            logical_len,
                        } => {
                            push_u8(&mut out, SCAN_VALUE_CHUNKED);
                            out.extend_from_slice(manifest);
                            push_u64(&mut out, *logical_len);
                        }
                        ScanValueBody::Oversize { logical_len } => {
                            push_u8(&mut out, SCAN_VALUE_OVERSIZE);
                            push_u64(&mut out, *logical_len);
                        }
                        ScanValueBody::Semantic { object, descriptor } => {
                            push_u8(&mut out, SCAN_VALUE_SEMANTIC);
                            push_u8(&mut out, *object);
                            push_blob(&mut out, descriptor);
                        }
                    }
                }
                push_u8(&mut out, u8::from(*exhausted));
                match last_key {
                    None => push_u8(&mut out, 0),
                    Some(key) => {
                        push_u8(&mut out, 1);
                        push_blob(&mut out, key);
                    }
                }
                push_u64(&mut out, *tablet);
                push_blob(&mut out, range_start);
                match range_end {
                    None => push_u8(&mut out, 0),
                    Some(end) => {
                        push_u8(&mut out, 1);
                        push_blob(&mut out, end);
                    }
                }
                push_u64(&mut out, *dir_version);
            }
            ResponseBody::AtomicCommitted { versions } => {
                push_u16(&mut out, u16::try_from(versions.len()).unwrap_or(u16::MAX));
                for version in versions {
                    match version {
                        None => push_u8(&mut out, 0),
                        Some(version) => {
                            push_u8(&mut out, 1);
                            push_u64(&mut out, *version);
                        }
                    }
                }
            }
            ResponseBody::TxnPrepared {
                tablet,
                dir_version,
            } => {
                push_u64(&mut out, *tablet);
                push_u64(&mut out, *dir_version);
            }
            ResponseBody::Redirect(info) => out.extend_from_slice(&info.encode()),
            ResponseBody::Diagnostic(message) => {
                push_u16(&mut out, u16::try_from(message.len()).unwrap_or(u16::MAX));
                out.extend_from_slice(message.as_bytes());
            }
        }
        // Read-proof trailer: absent when `None` (byte-identical to
        // pre-proof encodings), 40 bytes `(tablet, epoch, position,
        // guard, incarnation)` when `Some`. Old decoders reject proofed
        // responses loudly as trailing bytes; new decoders treat a
        // missing trailer as no evidence, never as freshness.
        if let Some(proof) = &self.proof {
            push_u64(&mut out, proof.authority.tablet().as_u64());
            push_u64(&mut out, proof.authority.epoch().as_u64());
            push_u64(&mut out, proof.position.as_u64());
            push_u64(&mut out, proof.authority.guard().as_u64());
            push_u64(&mut out, proof.incarnation.as_u64());
        }
        out
    }

    /// Decodes one response payload, requiring exact consumption.
    ///
    /// # Errors
    ///
    /// Returns [`ProtocolError`] on unknown statuses/opcodes, truncation,
    /// trailing bytes, or shapes inconsistent with the status/opcode pair.
    #[allow(clippy::too_many_lines)]
    pub fn decode(input: &[u8]) -> Result<(Self, Opcode), ProtocolError> {
        const CONTEXT: &str = "response";
        let mut cursor = Cursor::new(input);
        let status = Status::from_u16(cursor.u16(CONTEXT)?)
            .ok_or(ProtocolError::Malformed { context: CONTEXT })?;
        let opcode = Opcode::from_u8(cursor.u8(CONTEXT)?)
            .ok_or(ProtocolError::Malformed { context: CONTEXT })?;
        let body = match status {
            Status::Ok => match opcode {
                Opcode::Get | Opcode::GetRange => {
                    ResponseBody::Value(cursor.blob(CONTEXT)?.to_vec())
                }
                // Creates answer like ordinary stores: the version names
                // the new object either way, never the rights.
                Opcode::Set
                | Opcode::SetRange
                | Opcode::BoundedCounterCreate
                | Opcode::EscrowTransfer
                | Opcode::SemaphoreCreate => ResponseBody::Stored {
                    version: cursor.u64(CONTEXT)?,
                },
                Opcode::BytesLength => ResponseBody::Length(cursor.u64(CONTEXT)?),
                Opcode::GetVersion => ResponseBody::Version(cursor.u64(CONTEXT)?),
                Opcode::SetConditional => ResponseBody::ConditionalSet {
                    applied: flag(&mut cursor)?,
                    version: cursor.u64(CONTEXT)?,
                },
                // A stream read never completes as a single `Response`
                // frame: bytes arrive as `ValueStream*` frames under the
                // request id. An `Ok` shaped like a response is a peer
                // bug — fail closed, never guess a shape.
                Opcode::GetStream => {
                    return Err(ProtocolError::Malformed { context: CONTEXT });
                }
                Opcode::Delete => ResponseBody::Deleted {
                    existed: flag(&mut cursor)?,
                },
                Opcode::Exists => ResponseBody::Exists(flag(&mut cursor)?),
                Opcode::CounterGet => ResponseBody::Counter(cursor.i64(CONTEXT)?),
                Opcode::CounterAdd => ResponseBody::CounterUpdated {
                    value: cursor.i64(CONTEXT)?,
                    version: cursor.u64(CONTEXT)?,
                },
                Opcode::ExpireAt => ResponseBody::ExpirySet {
                    applied: flag(&mut cursor)?,
                },
                Opcode::PersistExpiry => ResponseBody::ExpiryPersisted {
                    removed: flag(&mut cursor)?,
                },
                Opcode::GetExpiry => match cursor.u8(CONTEXT)? {
                    0 => ResponseBody::ExpiryNever,
                    1 => ResponseBody::ExpiryAt(cursor.u64(CONTEXT)?),
                    _ => return Err(ProtocolError::Malformed { context: CONTEXT }),
                },
                Opcode::Scan => {
                    let count = cursor.u32(CONTEXT)? as usize;
                    if count > 100_000 {
                        return Err(ProtocolError::Malformed { context: CONTEXT });
                    }
                    let mut entries = Vec::with_capacity(count.min(1024));
                    for _ in 0..count {
                        let key = cursor.blob(CONTEXT)?.to_vec();
                        let value = match cursor.u8(CONTEXT)? {
                            SCAN_VALUE_NONE => ScanValueBody::None,
                            SCAN_VALUE_INLINE => {
                                ScanValueBody::Inline(cursor.blob(CONTEXT)?.to_vec())
                            }
                            SCAN_VALUE_COUNTER => ScanValueBody::Counter(cursor.i64(CONTEXT)?),
                            SCAN_VALUE_CHUNKED => {
                                let raw = cursor.take(32, CONTEXT)?;
                                let manifest: [u8; 32] = raw
                                    .try_into()
                                    .map_err(|_| ProtocolError::Malformed { context: CONTEXT })?;
                                let logical_len = cursor.u64(CONTEXT)?;
                                ScanValueBody::Chunked {
                                    manifest,
                                    logical_len,
                                }
                            }
                            SCAN_VALUE_OVERSIZE => ScanValueBody::Oversize {
                                logical_len: cursor.u64(CONTEXT)?,
                            },
                            SCAN_VALUE_SEMANTIC => ScanValueBody::Semantic {
                                object: cursor.u8(CONTEXT)?,
                                descriptor: cursor.blob(CONTEXT)?.to_vec(),
                            },
                            _ => return Err(ProtocolError::Malformed { context: CONTEXT }),
                        };
                        entries.push(ScanEntryBody { key, value });
                    }
                    let exhausted = flag(&mut cursor)?;
                    let last_key = match cursor.u8(CONTEXT)? {
                        0 => None,
                        1 => Some(cursor.blob(CONTEXT)?.to_vec()),
                        _ => return Err(ProtocolError::Malformed { context: CONTEXT }),
                    };
                    let tablet = cursor.u64(CONTEXT)?;
                    let range_start = cursor.blob(CONTEXT)?.to_vec();
                    let range_end = match cursor.u8(CONTEXT)? {
                        0 => None,
                        1 => Some(cursor.blob(CONTEXT)?.to_vec()),
                        _ => return Err(ProtocolError::Malformed { context: CONTEXT }),
                    };
                    let dir_version = cursor.u64(CONTEXT)?;
                    ResponseBody::ScanPage {
                        entries,
                        exhausted,
                        last_key,
                        tablet,
                        range_start,
                        range_end,
                        dir_version,
                    }
                }
                Opcode::AtomicBatch => {
                    let count = cursor.u16(CONTEXT)? as usize;
                    if count > 4096 {
                        return Err(ProtocolError::Malformed { context: CONTEXT });
                    }
                    let mut versions = Vec::with_capacity(count);
                    for _ in 0..count {
                        versions.push(match cursor.u8(CONTEXT)? {
                            0 => None,
                            1 => Some(cursor.u64(CONTEXT)?),
                            _ => return Err(ProtocolError::Malformed { context: CONTEXT }),
                        });
                    }
                    ResponseBody::AtomicCommitted { versions }
                }
                Opcode::TxnPrepare => ResponseBody::TxnPrepared {
                    tablet: cursor.u64(CONTEXT)?,
                    dir_version: cursor.u64(CONTEXT)?,
                },
                Opcode::TxnFinalize => ResponseBody::TxnFinalized {
                    applied: flag(&mut cursor)?,
                    version: cursor.u64(CONTEXT)?,
                    tablet: cursor.u64(CONTEXT)?,
                },
                Opcode::CommutativeGet => ResponseBody::CommutativeValue(cursor.i64(CONTEXT)?),
                Opcode::CommutativeAdd => ResponseBody::CommutativeApplied,
                Opcode::BoundedCounterAdd => ResponseBody::BoundedUpdated {
                    value: cursor.i64(CONTEXT)?,
                    version: cursor.u64(CONTEXT)?,
                },
                Opcode::BoundedCounterGet => ResponseBody::BoundedValue {
                    value: cursor.i64(CONTEXT)?,
                    capacity: cursor.u64(CONTEXT)?,
                    share_min: cursor.i64(CONTEXT)?,
                    share_max: cursor.i64(CONTEXT)?,
                },
                Opcode::SemaphoreAcquire => ResponseBody::SemaphoreAcquired,
                Opcode::SemaphoreRelease => ResponseBody::SemaphoreReleased {
                    released: flag(&mut cursor)?,
                },
                Opcode::SemaphoreInspect => ResponseBody::SemaphoreLoad {
                    outstanding: cursor.u64(CONTEXT)?,
                    capacity: cursor.u64(CONTEXT)?,
                },
                Opcode::LeaseAcquire => ResponseBody::LeaseAcquired {
                    fencing: cursor.u64(CONTEXT)?,
                    expires_at: cursor.u64(CONTEXT)?,
                },
                Opcode::LeaseRenew => ResponseBody::LeaseRenewed {
                    fencing: cursor.u64(CONTEXT)?,
                    expires_at: cursor.u64(CONTEXT)?,
                },
                Opcode::LeaseRelease => ResponseBody::LeaseReleased {
                    released: flag(&mut cursor)?,
                },
                Opcode::LeaseInspect => {
                    let owner = match cursor.u8(CONTEXT)? {
                        0 => None,
                        1 => Some(cursor.u64(CONTEXT)?),
                        _ => return Err(ProtocolError::Malformed { context: CONTEXT }),
                    };
                    ResponseBody::LeaseInfo {
                        owner,
                        fencing: cursor.u64(CONTEXT)?,
                        expires_at: cursor.u64(CONTEXT)?,
                        next_fencing: cursor.u64(CONTEXT)?,
                    }
                }
                Opcode::StreamCreate => ResponseBody::StreamCreated,
                Opcode::StreamAppend => ResponseBody::StreamAppended {
                    offset: cursor.u64(CONTEXT)?,
                },
                Opcode::StreamRead => {
                    let count = cursor.u32(CONTEXT)? as usize;
                    if count > 4096 {
                        return Err(ProtocolError::Malformed { context: CONTEXT });
                    }
                    let mut entries = Vec::with_capacity(count.min(256));
                    for _ in 0..count {
                        entries.push(StreamEntryBody {
                            offset: cursor.u64(CONTEXT)?,
                            partition: cursor.blob(CONTEXT)?.to_vec(),
                            payload: cursor.blob(CONTEXT)?.to_vec(),
                        });
                    }
                    ResponseBody::StreamEntries {
                        entries,
                        next_offset: cursor.u64(CONTEXT)?,
                    }
                }
                Opcode::StreamTrim => ResponseBody::StreamTrimmed {
                    removed: cursor.u64(CONTEXT)?,
                },
            },
            Status::NotFound => match opcode {
                Opcode::Get
                | Opcode::CounterGet
                | Opcode::CommutativeGet
                | Opcode::BoundedCounterGet
                | Opcode::SemaphoreInspect
                | Opcode::GetExpiry
                | Opcode::GetStream
                | Opcode::GetRange
                | Opcode::GetVersion
                | Opcode::BytesLength => {
                    let len = cursor.u16(CONTEXT)? as usize;
                    let bytes = cursor.take(len, CONTEXT)?;
                    ResponseBody::Diagnostic(String::from_utf8_lossy(bytes).into_owned())
                }
                _ => return Err(ProtocolError::Malformed { context: CONTEXT }),
            },
            Status::StaleRoute | Status::NotLocal => {
                let info = RedirectInfo::decode(cursor.rest())?;
                cursor.consume_all();
                ResponseBody::Redirect(info)
            }
            _ => {
                let len = cursor.u16(CONTEXT)? as usize;
                let bytes = cursor.take(len, CONTEXT)?;
                ResponseBody::Diagnostic(String::from_utf8_lossy(bytes).into_owned())
            }
        };
        // Proof trailer: only `Ok`/`NotFound` responses may carry one
        // (errors never vouch freshness). Absent means no evidence;
        // present must be exactly 40 bytes, else malformed.
        let proof = match status {
            Status::Ok | Status::NotFound => decode_proof_trailer(&mut cursor)?,
            _ => None,
        };
        cursor.end(CONTEXT)?;
        Ok((
            Self {
                status,
                body,
                proof,
            },
            opcode,
        ))
    }
}

/// Decodes the optional 40-byte read-proof trailer: empty remainder is
/// `None`, exactly 40 bytes is `Some`, anything else is malformed.
fn decode_proof_trailer(cursor: &mut Cursor<'_>) -> Result<Option<ReadReceipt>, ProtocolError> {
    const CONTEXT: &str = "response proof";
    if cursor.remaining() == 0 {
        return Ok(None);
    }
    if cursor.remaining() != 40 {
        return Err(ProtocolError::Malformed { context: CONTEXT });
    }
    let tablet = TabletId::from_u64(cursor.u64(CONTEXT)?);
    let epoch = TabletEpoch::from_u64(cursor.u64(CONTEXT)?);
    let position = CommitPosition::from_u64(cursor.u64(CONTEXT)?);
    let guard = WriteGuardGeneration::from_u64(cursor.u64(CONTEXT)?);
    let incarnation = kivi_types::NodeIncarnation::from_u64(cursor.u64(CONTEXT)?);
    Ok(Some(ReadReceipt::new(
        TabletAuthority::new(tablet, epoch, guard),
        position,
        incarnation,
    )))
}

/// Decodes a `0/1` flag byte.
fn flag(cursor: &mut Cursor<'_>) -> Result<bool, ProtocolError> {
    match cursor.u8("flag")? {
        0 => Ok(false),
        1 => Ok(true),
        _ => Err(ProtocolError::Malformed { context: "flag" }),
    }
}

/// Decodes one batch write list: `u16` count plus per-write
/// `(key, kind, value, delta, expect, expect_version)`. The count is capped
/// at `max` (framing-level allocation bound; servers enforce the tighter
/// transaction bound), so decoders never allocate blindly.
fn decode_batch_writes(
    cursor: &mut Cursor<'_>,
    max: usize,
) -> Result<Vec<BatchWrite>, ProtocolError> {
    const CONTEXT: &str = "request";
    let count = cursor.u16(CONTEXT)? as usize;
    if count > max {
        return Err(ProtocolError::Malformed { context: CONTEXT });
    }
    let mut writes = Vec::with_capacity(count);
    for _ in 0..count {
        let key = cursor.blob(CONTEXT)?.to_vec();
        let kind = cursor.u8(CONTEXT)?;
        if !matches!(
            kind,
            BATCH_PUT
                | BATCH_DELETE
                | BATCH_COUNTER_ADD
                | BATCH_PUT_CHUNKED
                | BATCH_COMMUTATIVE_ADD
                | BATCH_BOUNDED_ADD
                | BATCH_ESCROW_SHARE
        ) {
            return Err(ProtocolError::Malformed { context: CONTEXT });
        }
        let value = cursor.blob(CONTEXT)?.to_vec();
        let delta = cursor.i64(CONTEXT)?;
        let expect = cursor.u8(CONTEXT)?;
        if !matches!(
            expect,
            BATCH_EXPECT_ANY | BATCH_EXPECT_ABSENT | BATCH_EXPECT_VERSION
        ) {
            return Err(ProtocolError::Malformed { context: CONTEXT });
        }
        let expect_version = cursor.u64(CONTEXT)?;
        writes.push(BatchWrite {
            key,
            kind,
            value,
            delta,
            expect,
            expect_version,
        });
    }
    Ok(writes)
}

/// V1 streaming-upload bound: 1 GiB per stream (1024 chunks at the
/// default 1 MiB grid). Peers agree here so both sides enforce the same
/// ceiling — the server aborts past it, the client never opens past it.
/// Bulk beyond this is a future multi-stream or resumption story, never
/// silent truncation today.
pub const MAX_STREAM_UPLOAD_BYTES: u64 = 1024 * 1024 * 1024;

/// Opens a value upload: client → server `StreamBegin` frame payload.
/// The frame `request_id` is the client-chosen stream id, echoed on
/// every stream frame for this upload and on the final `Response`
/// (`Stored{version}`), so uploads multiplex freely with ordinary
/// requests on one connection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StreamBegin {
    /// Target namespace.
    pub namespace: NamespaceId,
    /// Target key.
    pub key: Vec<u8>,
    /// Declared total bytes (`None` = unknown upfront; the upload bound
    /// still applies, and a declared total mismatching delivery aborts
    /// the stream instead of committing short).
    pub total_len: Option<u64>,
    /// Retry identity, mirroring [`Request`]: a retried upload commits at
    /// most once per identity, dedup-hitting when the first attempt
    /// already committed.
    pub identity: Option<RequestIdentity>,
    /// Acknowledgement floor, mirroring [`Request`].
    pub ack_floor: RequestSeq,
}

impl StreamBegin {
    /// Encodes one begin payload (framing header added by the caller).
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(32 + self.key.len());
        push_u64(&mut out, self.namespace.as_u64());
        push_blob(&mut out, &self.key);
        match self.total_len {
            None => push_u8(&mut out, 0),
            Some(total) => {
                push_u8(&mut out, 1);
                push_u64(&mut out, total);
            }
        }
        match self.identity {
            None => push_u8(&mut out, 0),
            Some(identity) => {
                push_u8(&mut out, 1);
                push_u128(&mut out, identity.session().as_u128());
                push_u64(&mut out, identity.seq().as_u64());
                push_u64(&mut out, self.ack_floor.as_u64());
            }
        }
        out
    }

    /// Decodes one begin payload, requiring exact consumption.
    ///
    /// # Errors
    ///
    /// Returns [`ProtocolError`] on truncation, trailing bytes, or a bad
    /// discriminator.
    pub fn decode(input: &[u8]) -> Result<Self, ProtocolError> {
        const CONTEXT: &str = "stream-begin";
        let mut cursor = Cursor::new(input);
        let namespace = NamespaceId::from_u64(cursor.u64(CONTEXT)?);
        let key = cursor.blob(CONTEXT)?.to_vec();
        let total_len = match cursor.u8(CONTEXT)? {
            0 => None,
            1 => Some(cursor.u64(CONTEXT)?),
            _ => return Err(ProtocolError::Malformed { context: CONTEXT }),
        };
        let mut begin = Self {
            namespace,
            key,
            total_len,
            identity: None,
            ack_floor: RequestSeq::from_u64(0),
        };
        match cursor.u8(CONTEXT)? {
            0 => {}
            1 => {
                let session = kivi_types::SessionId::from_u128(cursor.u128(CONTEXT)?);
                let seq = RequestSeq::from_u64(cursor.u64(CONTEXT)?);
                begin.ack_floor = RequestSeq::from_u64(cursor.u64(CONTEXT)?);
                begin.identity = Some(RequestIdentity::new(session, seq));
            }
            _ => return Err(ProtocolError::Malformed { context: CONTEXT }),
        }
        cursor.end(CONTEXT)?;
        Ok(begin)
    }
}

/// Server accepts an upload: `StreamReady` frame payload. The client
/// must keep every `StreamData` payload at or under `max_data` bytes;
/// larger frames abort the stream (framing bounds still apply on top).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StreamReady {
    /// Maximum `StreamData` payload bytes per frame on this stream.
    pub max_data: u32,
}

impl StreamReady {
    /// Encodes one ready payload.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(4);
        push_u32(&mut out, self.max_data);
        out
    }

    /// Decodes one ready payload, requiring exact consumption.
    ///
    /// # Errors
    ///
    /// Returns [`ProtocolError`] on truncation or trailing bytes.
    pub fn decode(input: &[u8]) -> Result<Self, ProtocolError> {
        const CONTEXT: &str = "stream-ready";
        let mut cursor = Cursor::new(input);
        let ready = Self {
            max_data: cursor.u32(CONTEXT)?,
        };
        cursor.end(CONTEXT)?;
        Ok(ready)
    }
}

/// Either side aborts a stream: `StreamAbort` frame payload. The sender
/// drops all stream state; the receiver answers nothing. Commits never
/// follow an abort on the same id, and servers never commit a stream
/// they aborted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StreamAbort {
    /// Human-readable reason (diagnostic only, never a contract).
    pub reason: String,
}

impl StreamAbort {
    /// Encodes one abort payload (u16-prefixed UTF-8, like diagnostics).
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(2 + self.reason.len());
        push_u16(
            &mut out,
            u16::try_from(self.reason.len()).unwrap_or(u16::MAX),
        );
        out.extend_from_slice(self.reason.as_bytes());
        out
    }

    /// Decodes one abort payload, requiring exact consumption.
    ///
    /// # Errors
    ///
    /// Returns [`ProtocolError`] on truncation or trailing bytes.
    pub fn decode(input: &[u8]) -> Result<Self, ProtocolError> {
        const CONTEXT: &str = "stream-abort";
        let mut cursor = Cursor::new(input);
        let len = cursor.u16(CONTEXT)? as usize;
        let bytes = cursor.take(len, CONTEXT)?;
        let abort = Self {
            reason: String::from_utf8_lossy(bytes).into_owned(),
        };
        cursor.end(CONTEXT)?;
        Ok(abort)
    }
}

/// A streamed read starts: server → client `ValueStreamBegin` frame
/// payload under the originating request id. `total_len` data bytes
/// follow across `ValueStreamData` frames, then one empty
/// `ValueStreamEnd`. Raw `Data` payloads are verbatim bytes (no
/// envelope); `Commit`-side framing does not exist for reads — the
/// stream simply ends.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ValueStreamBegin {
    /// Total data bytes to follow across `ValueStreamData` frames.
    pub total_len: u64,
}

impl ValueStreamBegin {
    /// Encodes one begin payload.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(8);
        push_u64(&mut out, self.total_len);
        out
    }

    /// Decodes one begin payload, requiring exact consumption.
    ///
    /// # Errors
    ///
    /// Returns [`ProtocolError`] on truncation or trailing bytes.
    pub fn decode(input: &[u8]) -> Result<Self, ProtocolError> {
        const CONTEXT: &str = "value-stream-begin";
        let mut cursor = Cursor::new(input);
        let begin = Self {
            total_len: cursor.u64(CONTEXT)?,
        };
        cursor.end(CONTEXT)?;
        Ok(begin)
    }
}

impl Cursor<'_> {
    /// Borrows the unread remainder.
    fn rest(&self) -> &[u8] {
        &self.buf[self.pos..]
    }

    /// Advances past all remaining bytes (after a nested exact decoder ran).
    fn consume_all(&mut self) {
        self.pos = self.buf.len();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kivi_state::Key;
    use kivi_tablet::{DirectoryVersion, HashPrefix};
    use rstest::rstest;

    const NS: NamespaceId = NamespaceId::from_u64(7);

    fn request(opcode: Opcode) -> Request {
        Request {
            contract: kivi_types::ReadContract::Latest,
            namespace: NS,
            opcode,
            hint: Some(RouteHint {
                tablet: TabletId::from_u64(3),
                epoch: TabletEpoch::from_u64(9),
                worker: WorkerId::from_u64(1),
            }),
            key: b"user:1".to_vec(),
            value: Some(b"value".to_vec()),
            delta: -12,
            expiry: 123_456,
            offset: 77,
            len: 19,
            condition: COND_IF_ABSENT,
            expiry_policy: EXPIRY_AT,
            identity: Some(RequestIdentity::new(
                kivi_types::SessionId::from_u128(0x00C0_FFEE),
                RequestSeq::from_u64(41),
            )),
            ack_floor: RequestSeq::from_u64(40),
            scan_start: Some(b"a".to_vec()),
            scan_end: Some(b"z".to_vec()),
            scan_direction: SCAN_FORWARD,
            scan_max_items: 100,
            scan_max_bytes: 65536,
            scan_projection: SCAN_KEYS_AND_VALUES,
            scan_consistency: SCAN_LATEST_PER_TABLET,
            batch_txn: [0xAB; 16],
            batch_writes: vec![BatchWrite {
                key: b"k1".to_vec(),
                kind: BATCH_PUT,
                value: b"v1".to_vec(),
                delta: 0,
                expect: BATCH_EXPECT_ANY,
                expect_version: 0,
            }],
            txn_coordinator: 0,
            txn_commit: false,
            txn_digest: [0xD1; 32],
            capacity: 0,
            holder: 0,
            permit: [0u8; 16],
            owner: 0,
            qty: 0,
            fencing: 0,
            ttl: 0,
            stream: [0u8; 16],
            shard: 0,
            partition: Vec::new(),
            share_min: 0,
            share_max: 0,
        }
    }

    #[test]
    fn txn_prepare_finalize_round_trip() {
        let mut prepare = request(Opcode::TxnPrepare);
        prepare.txn_coordinator = 9;
        let encoded = prepare.encode();
        let decoded = Request::decode(&encoded).expect("round trip");
        assert_eq!(decoded.opcode, Opcode::TxnPrepare);
        assert_eq!(decoded.batch_txn, [0xAB; 16]);
        assert_eq!(decoded.txn_coordinator, 9);
        assert_eq!(decoded.batch_writes.len(), 1);
        let op = decoded.into_operation().expect("translatable");
        assert!(matches!(op, Operation::TxnPrepare { .. }));
        // A prepare carrying zero or two writes is malformed (exactly one).
        let mut multi = request(Opcode::TxnPrepare);
        multi.batch_writes.push(BatchWrite {
            key: b"k2".to_vec(),
            kind: BATCH_DELETE,
            value: Vec::new(),
            delta: 0,
            expect: BATCH_EXPECT_ANY,
            expect_version: 0,
        });
        assert!(Request::decode(&multi.encode()).is_err());
        // Finalize carries key + commit flag.
        let mut finalize = request(Opcode::TxnFinalize);
        finalize.txn_commit = true;
        let decoded = Request::decode(&finalize.encode()).expect("round trip");
        assert!(decoded.txn_commit);
        let op = decoded.into_operation().expect("translatable");
        assert!(matches!(op, Operation::TxnFinalize { commit: true, .. }));
        // Txn responses shape as their own opcodes.
        for (opcode, body) in [
            (
                Opcode::TxnPrepare,
                ResponseBody::TxnPrepared {
                    tablet: 4,
                    dir_version: 12,
                },
            ),
            (
                Opcode::TxnFinalize,
                ResponseBody::TxnFinalized {
                    applied: true,
                    version: 5,
                    tablet: 7,
                },
            ),
        ] {
            let response = Response {
                proof: None,
                status: Status::Ok,
                body,
            };
            let (decoded, back) = Response::decode(&response.encode(opcode)).expect("round trip");
            assert_eq!(back, opcode);
            assert_eq!(decoded, response);
        }
    }

    #[rstest]
    #[case(Opcode::Get)]
    #[case(Opcode::Set)]
    #[case(Opcode::Delete)]
    #[case(Opcode::Exists)]
    #[case(Opcode::CounterGet)]
    #[case(Opcode::CounterAdd)]
    #[case(Opcode::ExpireAt)]
    #[case(Opcode::PersistExpiry)]
    #[case(Opcode::GetExpiry)]
    #[case(Opcode::SetRange)]
    #[case(Opcode::GetStream)]
    #[case(Opcode::GetRange)]
    #[case(Opcode::BytesLength)]
    #[case(Opcode::GetVersion)]
    #[case(Opcode::SetConditional)]
    #[case(Opcode::CommutativeGet)]
    #[case(Opcode::CommutativeAdd)]
    #[case(Opcode::BoundedCounterCreate)]
    #[case(Opcode::BoundedCounterAdd)]
    #[case(Opcode::BoundedCounterGet)]
    #[case(Opcode::EscrowTransfer)]
    #[case(Opcode::SemaphoreCreate)]
    #[case(Opcode::SemaphoreAcquire)]
    #[case(Opcode::SemaphoreRelease)]
    #[case(Opcode::SemaphoreInspect)]
    #[case(Opcode::LeaseAcquire)]
    #[case(Opcode::LeaseRenew)]
    #[case(Opcode::LeaseRelease)]
    #[case(Opcode::LeaseInspect)]
    #[case(Opcode::StreamCreate)]
    #[case(Opcode::StreamAppend)]
    #[case(Opcode::StreamRead)]
    #[case(Opcode::StreamTrim)]
    #[case(Opcode::Scan)]
    #[case(Opcode::AtomicBatch)]
    fn every_operation_round_trips(#[case] opcode: Opcode) {
        let encoded = request(opcode).encode();
        let decoded = Request::decode(&encoded).expect("round trip");
        assert_eq!(decoded.opcode, opcode);
        assert_eq!(decoded.namespace, NS);
        assert_eq!(decoded.key, b"user:1");
        assert_eq!(decoded.hint.expect("hint").tablet, TabletId::from_u64(3));
        assert_eq!(
            decoded.identity.expect("identity").seq(),
            RequestSeq::from_u64(41)
        );
        assert_eq!(decoded.ack_floor, RequestSeq::from_u64(40));
        // Operation translation is total and typed for single-key opcodes;
        // Scan and AtomicBatch dispatch on dedicated payloads instead.
        if matches!(opcode, Opcode::Scan | Opcode::AtomicBatch) {
            assert_eq!(decoded.clone().into_operation(), None);
            if opcode == Opcode::Scan {
                assert_eq!(decoded.scan_start, Some(b"a".to_vec()));
                assert_eq!(decoded.scan_end, Some(b"z".to_vec()));
                assert_eq!(decoded.scan_direction, SCAN_FORWARD);
                assert_eq!(decoded.scan_max_items, 100);
                assert_eq!(decoded.scan_max_bytes, 65536);
                assert_eq!(decoded.scan_projection, SCAN_KEYS_AND_VALUES);
                assert_eq!(decoded.scan_consistency, SCAN_LATEST_PER_TABLET);
            } else {
                assert_eq!(decoded.batch_txn, [0xAB; 16]);
                assert_eq!(decoded.batch_writes.len(), 1);
                assert_eq!(decoded.batch_writes[0].key, b"k1");
                assert_eq!(decoded.batch_writes[0].kind, BATCH_PUT);
            }
            return;
        }
        let op = decoded.into_operation().expect("translatable");
        match opcode {
            Opcode::Get => assert!(matches!(op, Operation::Get { .. })),
            Opcode::Set => assert!(matches!(op, Operation::Set { .. })),
            Opcode::Delete => assert!(matches!(op, Operation::Delete { .. })),
            Opcode::Exists => assert!(matches!(op, Operation::Exists { .. })),
            Opcode::CounterGet => assert!(matches!(op, Operation::CounterGet { .. })),
            Opcode::CounterAdd => assert!(matches!(op, Operation::CounterAdd { .. })),
            Opcode::ExpireAt => assert!(matches!(op, Operation::ExpireAt { .. })),
            Opcode::PersistExpiry => assert!(matches!(op, Operation::PersistExpiry { .. })),
            Opcode::GetExpiry => assert!(matches!(op, Operation::GetExpiry { .. })),
            Opcode::SetRange => {
                let Operation::SetRange { offset, .. } = op else {
                    panic!("SetRange mistranslated")
                };
                assert_eq!(offset, 77);
            }
            Opcode::GetRange => {
                let Operation::GetRange { offset, len, .. } = op else {
                    panic!("GetRange mistranslated")
                };
                assert_eq!(offset, 77);
                assert_eq!(len, 19);
            }
            Opcode::BytesLength => assert!(matches!(op, Operation::BytesLength { .. })),
            Opcode::GetVersion => assert!(matches!(op, Operation::GetVersion { .. })),
            Opcode::SetConditional => {
                let Operation::SetConditional {
                    condition, expiry, ..
                } = op
                else {
                    panic!("SetConditional mistranslated")
                };
                assert_eq!(condition, kivi_state::SetCondition::IfAbsent);
                assert!(matches!(expiry, kivi_state::ExpiryPolicy::ExpireAt(_)));
            }
            // GetStream executes the same read as Get; only delivery differs.
            Opcode::GetStream => assert!(matches!(op, Operation::Get { .. })),
            Opcode::CommutativeGet => assert!(matches!(op, Operation::CommutativeGet { .. })),
            Opcode::CommutativeAdd => assert!(matches!(op, Operation::CommutativeAdd { .. })),
            Opcode::BoundedCounterCreate => {
                assert!(matches!(op, Operation::BoundedCounterCreate { .. }));
            }
            Opcode::BoundedCounterAdd => {
                assert!(matches!(op, Operation::BoundedCounterAdd { .. }));
            }
            Opcode::BoundedCounterGet => {
                assert!(matches!(op, Operation::BoundedCounterGet { .. }));
            }
            Opcode::EscrowTransfer => assert!(matches!(op, Operation::EscrowTransfer { .. })),
            Opcode::SemaphoreCreate => assert!(matches!(op, Operation::SemaphoreCreate { .. })),
            Opcode::SemaphoreAcquire => {
                assert!(matches!(op, Operation::SemaphoreAcquire { .. }));
            }
            Opcode::SemaphoreRelease => {
                assert!(matches!(op, Operation::SemaphoreRelease { .. }));
            }
            Opcode::SemaphoreInspect => {
                assert!(matches!(op, Operation::SemaphoreInspect { .. }));
            }
            Opcode::LeaseAcquire => assert!(matches!(op, Operation::LeaseAcquire { .. })),
            Opcode::LeaseRenew => assert!(matches!(op, Operation::LeaseRenew { .. })),
            Opcode::LeaseRelease => assert!(matches!(op, Operation::LeaseRelease { .. })),
            Opcode::LeaseInspect => assert!(matches!(op, Operation::LeaseInspect { .. })),
            Opcode::StreamCreate => assert!(matches!(op, Operation::StreamCreate { .. })),
            Opcode::StreamAppend => assert!(matches!(op, Operation::StreamAppend { .. })),
            Opcode::StreamRead => assert!(matches!(op, Operation::StreamRead { .. })),
            Opcode::StreamTrim => assert!(matches!(op, Operation::StreamTrim { .. })),
            Opcode::TxnPrepare => assert!(matches!(op, Operation::TxnPrepare { .. })),
            Opcode::TxnFinalize => assert!(matches!(op, Operation::TxnFinalize { .. })),
            // Scan and AtomicBatch return early above (no Operation).
            Opcode::Scan | Opcode::AtomicBatch => {
                panic!("scan/batch translate to no operation")
            }
        }
    }

    #[test]
    fn request_ids_and_hints() {
        assert_eq!(RequestId::FIRST.as_u64(), 1);
        assert_eq!(RequestId::from_u64(9).to_string(), "req9");
        let mut bare = request(Opcode::Get);
        bare.hint = None;
        let decoded = Request::decode(&bare.encode()).expect("hintless");
        assert_eq!(decoded.hint, None);
        // Unknown opcodes fail closed with their discriminant preserved.
        // (Layout: namespace u64, then the opcode byte at offset 8.)
        let mut bad = request(Opcode::Get).encode();
        bad[8] = 0xFF;
        assert_eq!(
            Request::decode(&bad),
            Err(ProtocolError::UnknownOpcode { opcode: 0xFF })
        );
    }

    fn redirect() -> RedirectInfo {
        RedirectInfo::new(
            DirectoryVersion::from_u64(11),
            TabletId::from_u64(3),
            TabletEpoch::from_u64(9),
            WorkerId::from_u64(1),
            "127.0.0.1:9101".to_owned(),
            PartitionRange::Hash(HashPrefix::new(1u128 << 127, 1).expect("prefix")),
        )
        .expect("valid redirect")
    }

    #[rstest]
    #[case(Status::Ok)]
    #[case(Status::InvalidRequest)]
    #[case(Status::Unsupported)]
    #[case(Status::WrongType)]
    #[case(Status::NotFound)]
    #[case(Status::CounterOverflow)]
    #[case(Status::VersionExhausted)]
    #[case(Status::StaleRoute)]
    #[case(Status::NotLocal)]
    #[case(Status::Overloaded)]
    #[case(Status::ResourceExhausted)]
    #[case(Status::ValueTooLarge)]
    #[case(Status::Internal)]
    #[case(Status::DedupExpired)]
    #[case(Status::SessionOverloaded)]
    #[case(Status::TxnConflict)]
    #[case(Status::TxnAborted)]
    #[case(Status::TxnTooLarge)]
    #[case(Status::TxnCoordinatorUnavailable)]
    #[case(Status::UniqueViolation)]
    #[case(Status::ScanCursorStale)]
    #[case(Status::StaleToken)]
    #[case(Status::CoverageUncertain)]
    fn every_status_round_trips(#[case] status: Status) {
        assert_eq!(Status::from_u16(status.as_u16()), Some(status));
        assert_eq!(
            status.is_retriable(),
            matches!(
                status,
                Status::StaleRoute
                    | Status::NotLocal
                    | Status::Overloaded
                    | Status::TxnCoordinatorUnavailable
                    | Status::ScanCursorStale
            )
        );
    }

    #[test]
    fn stale_token_is_terminal() {
        // A stale token never heals under the same token: retrying is
        // pointless (refresh the token instead), so it is not retriable.
        assert!(!Status::StaleToken.is_retriable());
        assert_eq!(Status::from_u16(21), Some(Status::StaleToken));
        assert_eq!(Status::StaleToken.to_string(), "stale-token");
    }

    #[test]
    fn coverage_uncertain_is_terminal_and_ambiguous() {
        // A coverage refusal is not a transport retry: the write may have
        // committed, so blind redelivery could double-apply anonymous
        // non-idempotent writes. Callers read-verify (or re-drive under
        // the same identity) instead.
        assert!(!Status::CoverageUncertain.is_retriable());
        assert_eq!(Status::from_u16(22), Some(Status::CoverageUncertain));
        assert_eq!(Status::CoverageUncertain.to_string(), "coverage-uncertain");
    }

    #[test]
    fn dedup_statuses_are_terminal() {
        // Dedup outcomes are deterministic for the same state: retrying is
        // harmless but pointless, so they are not marked retriable.
        assert!(!Status::DedupExpired.is_retriable());
        assert!(!Status::SessionOverloaded.is_retriable());
        assert_eq!(Status::from_u16(13), Some(Status::DedupExpired));
        assert_eq!(Status::from_u16(14), Some(Status::SessionOverloaded));
        assert_eq!(Status::DedupExpired.to_string(), "dedup-expired");
        assert_eq!(Status::SessionOverloaded.to_string(), "session-overloaded");
    }

    #[test]
    fn read_contracts_round_trip_with_stable_tags() {
        use core::time::Duration;
        // Latest writes no trailer: byte-identical to pre-contract
        // encodings, decoding back to Latest.
        let latest = request(Opcode::Get);
        assert_eq!(latest.contract, ReadContract::Latest);
        let decoded = Request::decode(&latest.encode()).expect("latest round trip");
        assert_eq!(decoded.contract, ReadContract::Latest);
        // Every other contract rides its tag and round-trips exactly.
        let token = CommitToken::new(
            TabletId::from_u64(3),
            TabletEpoch::from_u64(9),
            CommitPosition::from_u64(41),
        );
        for contract in [
            ReadContract::AtLeast(token),
            ReadContract::BoundedStale {
                max_staleness: Duration::from_millis(250),
            },
            ReadContract::Any,
        ] {
            let mut wire = request(Opcode::Get);
            wire.contract = contract;
            let decoded = Request::decode(&wire.encode()).expect("contract round trip");
            assert_eq!(decoded.contract, contract);
        }
        // Unknown contract tags fail closed, never guessed.
        let mut bad = request(Opcode::Get).encode();
        bad.push(0x7F);
        assert!(Request::decode(&bad).is_err());
        // A pre-contract encoding (no trailer at all) still decodes as
        // Latest: the strongest contract, never a silent weakening.
        // (Only the contract is compared: opcode tails never round-trip
        // across shapes, as `every_operation_round_trips` pins.)
        let truncated = request(Opcode::Get).encode();
        assert_eq!(
            Request::decode(&truncated)
                .expect("pre-contract decodes")
                .contract,
            ReadContract::Latest
        );
    }

    #[test]
    fn read_proofs_round_trip_as_response_trailers() {
        let proof = ReadReceipt::new(
            TabletAuthority::new(
                TabletId::from_u64(3),
                TabletEpoch::from_u64(9),
                WriteGuardGeneration::from_u64(12),
            ),
            CommitPosition::from_u64(41),
            kivi_types::NodeIncarnation::from_u64(7),
        );
        let response = Response {
            status: Status::Ok,
            body: ResponseBody::Exists(true),
            proof: Some(proof),
        };
        let (decoded, _) = Response::decode(&response.encode(Opcode::Exists)).expect("round trip");
        assert_eq!(decoded, response);
        assert_eq!(
            decoded.proof.expect("proof").token().position(),
            CommitPosition::from_u64(41)
        );
        // Proof-less responses are byte-identical to pre-proof encodings
        // and decode to `None` (no evidence, never freshness).
        let bare = Response {
            status: Status::Ok,
            body: ResponseBody::Exists(true),
            proof: None,
        };
        let (decoded, _) = Response::decode(&bare.encode(Opcode::Exists)).expect("round trip");
        assert_eq!(decoded.proof, None);
        // A truncated trailer (neither absent nor 40 bytes) is malformed.
        let mut bad = bare.encode(Opcode::Exists);
        bad.extend_from_slice(&[1u8; 7]);
        assert!(Response::decode(&bad).is_err());
    }

    #[test]
    fn identity_wire_layout_is_stable() {
        // Golden suffix: flag 1, session LE, seq LE, ack LE. The base layout
        // is pinned by the pre-identity tests; this pins the suffix.
        let mut bare = request(Opcode::Get);
        bare.identity = None;
        let mut identified = bare.clone();
        identified.identity = Some(RequestIdentity::new(
            kivi_types::SessionId::from_u128(0x0102_0304_0506_0708_090A_0B0C_0D0E_0F10),
            RequestSeq::from_u64(0x1112_1314_1516_1718),
        ));
        identified.ack_floor = RequestSeq::from_u64(0x2122_2324_2526_2728);
        let bare_bytes = bare.encode();
        let full_bytes = identified.encode();
        // Bare ends with flag 0 (1 byte); identified ends with flag 1 plus
        // session/seq/ack (33 bytes): the difference is exactly 32.
        assert_eq!(full_bytes.len(), bare_bytes.len() + 32);
        assert_eq!(
            &full_bytes[..bare_bytes.len() - 1],
            &bare_bytes[..bare_bytes.len() - 1]
        );
        assert_eq!(bare_bytes[bare_bytes.len() - 1], 0);
        let suffix = &full_bytes[bare_bytes.len() - 1..];
        assert_eq!(suffix.len(), 33);
        assert_eq!(suffix[0], 1);
        assert_eq!(
            &suffix[1..17],
            &0x0102_0304_0506_0708_090A_0B0C_0D0E_0F10u128.to_le_bytes()
        );
        assert_eq!(&suffix[17..25], &0x1112_1314_1516_1718u64.to_le_bytes());
        assert_eq!(&suffix[25..33], &0x2122_2324_2526_2728u64.to_le_bytes());
        let back = Request::decode(&full_bytes).expect("identity round trip");
        // Opcode tails are not preserved for Get (same as pre-identity
        // behavior); the identity suffix is what this pins.
        assert_eq!(back.identity, identified.identity);
        assert_eq!(back.ack_floor, identified.ack_floor);
        assert_eq!(back.key, identified.key);
        assert_eq!(back.hint, identified.hint);
    }

    #[test]
    fn pre_identity_requests_still_decode() {
        // A request encoded by a pre-identity client (no suffix at all, not
        // even the flag) decodes with no identity rather than failing: the
        // server decides policy, not the parser.
        let mut bare = request(Opcode::Set);
        bare.identity = None;
        let bytes = bare.encode();
        assert_eq!(bytes[bytes.len() - 1], 0, "flag 0 terminates");
        let old_style = &bytes[..bytes.len() - 1];
        let back = Request::decode(old_style).expect("suffix-less decodes");
        assert_eq!(back.identity, None);
        assert_eq!(back.ack_floor, RequestSeq::from_u64(0));
        // A garbage flag is malformed, not silently skipped.
        let mut bad = bytes.clone();
        let last = bad.len() - 1;
        bad[last] = 0x7F;
        assert!(matches!(
            Request::decode(&bad),
            Err(ProtocolError::Malformed { .. })
        ));
    }

    #[test]
    fn durable_dedup_capability_negotiates() {
        assert!(Capabilities::BASE_V1.unknown_required().is_empty());
        let both = Capabilities::BASE_V1 | Capabilities::DURABLE_MUTATION_DEDUP;
        assert!(both.unknown_required().is_empty());
        assert!(both.contains(Capabilities::DURABLE_MUTATION_DEDUP));
        // The known set additionally carries streaming (chunk-fabric
        // upload/download frames plus SetRange/GetStream opcodes).
        assert_eq!(Capabilities::KNOWN, both | Capabilities::STREAMING);
    }

    #[test]
    fn response_bodies_round_trip_per_opcode() {
        let bodies = vec![
            (Opcode::Get, ResponseBody::Value(b"v".to_vec())),
            (Opcode::Set, ResponseBody::Stored { version: 41 }),
            (Opcode::Delete, ResponseBody::Deleted { existed: true }),
            (Opcode::Exists, ResponseBody::Exists(false)),
            (Opcode::CounterGet, ResponseBody::Counter(-7)),
            (
                Opcode::CounterAdd,
                ResponseBody::CounterUpdated {
                    value: 8,
                    version: 2,
                },
            ),
            (Opcode::ExpireAt, ResponseBody::ExpirySet { applied: true }),
            (
                Opcode::PersistExpiry,
                ResponseBody::ExpiryPersisted { removed: false },
            ),
            (Opcode::GetExpiry, ResponseBody::ExpiryAt(99)),
            (
                Opcode::Scan,
                ResponseBody::ScanPage {
                    entries: vec![ScanEntryBody {
                        key: b"k".to_vec(),
                        value: ScanValueBody::Inline(b"v".to_vec()),
                    }],
                    exhausted: true,
                    last_key: Some(b"k".to_vec()),
                    tablet: 3,
                    range_start: b"a".to_vec(),
                    range_end: Some(b"z".to_vec()),
                    dir_version: 11,
                },
            ),
            (
                Opcode::AtomicBatch,
                ResponseBody::AtomicCommitted {
                    versions: vec![Some(3), None],
                },
            ),
        ];
        for (opcode, body) in bodies {
            let response = Response {
                proof: None,
                status: Status::Ok,
                body,
            };
            let (decoded, back) = Response::decode(&response.encode(opcode)).expect("round trip");
            assert_eq!(back, opcode);
            assert_eq!(decoded, response);
        }
        // Redirects and diagnostics ride their own statuses.
        let redirect = Response {
            proof: None,
            status: Status::StaleRoute,
            body: ResponseBody::Redirect(redirect()),
        };
        let (decoded, opcode) = Response::decode(&redirect.encode(Opcode::Get)).expect("redirect");
        assert_eq!(opcode, Opcode::Get);
        assert_eq!(decoded, redirect);
        let error = Response {
            proof: None,
            status: Status::WrongType,
            body: ResponseBody::Diagnostic("needs bytes".to_owned()),
        };
        let (decoded, _) = Response::decode(&error.encode(Opcode::Get)).expect("error");
        assert_eq!(decoded, error);
        // Missing values surface as NotFound, not empty Ok bodies.
        let missing = Response {
            proof: None,
            status: Status::NotFound,
            body: ResponseBody::Diagnostic(String::new()),
        };
        let (decoded, opcode) = Response::decode(&missing.encode(Opcode::CounterGet)).expect("404");
        assert_eq!(decoded.status, Status::NotFound);
        assert_eq!(opcode, Opcode::CounterGet);
        // Versions share the missing shape (absent keys read no version).
        let (decoded, opcode) = Response::decode(&missing.encode(Opcode::GetVersion)).expect("404");
        assert_eq!(decoded.status, Status::NotFound);
        assert_eq!(opcode, Opcode::GetVersion);
        let versioned = Response {
            proof: None,
            status: Status::Ok,
            body: ResponseBody::Version(41),
        };
        let (decoded, opcode) =
            Response::decode(&versioned.encode(Opcode::GetVersion)).expect("ok");
        assert_eq!(decoded, versioned);
        assert_eq!(opcode, Opcode::GetVersion);
    }

    /// Every semantic response body survives its opcode's codec. The
    /// client maps these (never strings), so a codec regression here is a
    /// silent contract break downstream.
    #[test]
    fn semantic_response_bodies_round_trip() {
        for (opcode, body) in semantic_bodies() {
            let response = Response {
                proof: None,
                status: Status::Ok,
                body,
            };
            let (decoded, back) = Response::decode(&response.encode(opcode)).expect("round trip");
            assert_eq!(back, opcode);
            assert_eq!(decoded, response);
        }
    }

    fn semantic_bodies() -> Vec<(Opcode, ResponseBody)> {
        vec![
            (Opcode::CommutativeAdd, ResponseBody::CommutativeApplied),
            (Opcode::CommutativeGet, ResponseBody::CommutativeValue(-9)),
            (
                Opcode::BoundedCounterCreate,
                ResponseBody::Stored { version: 1 },
            ),
            (
                Opcode::BoundedCounterAdd,
                ResponseBody::BoundedUpdated {
                    value: 6,
                    version: 2,
                },
            ),
            (
                Opcode::BoundedCounterGet,
                ResponseBody::BoundedValue {
                    value: 6,
                    capacity: 10,
                    share_min: 0,
                    share_max: 10,
                },
            ),
            (Opcode::SemaphoreCreate, ResponseBody::Stored { version: 1 }),
            (Opcode::SemaphoreAcquire, ResponseBody::SemaphoreAcquired),
            (
                Opcode::SemaphoreRelease,
                ResponseBody::SemaphoreReleased { released: true },
            ),
            (
                Opcode::SemaphoreInspect,
                ResponseBody::SemaphoreLoad {
                    outstanding: 1,
                    capacity: 2,
                },
            ),
            (
                Opcode::LeaseAcquire,
                ResponseBody::LeaseAcquired {
                    fencing: 3,
                    expires_at: 100,
                },
            ),
            (
                Opcode::LeaseRenew,
                ResponseBody::LeaseRenewed {
                    fencing: 3,
                    expires_at: 200,
                },
            ),
            (
                Opcode::LeaseRelease,
                ResponseBody::LeaseReleased { released: false },
            ),
            (
                Opcode::LeaseInspect,
                ResponseBody::LeaseInfo {
                    owner: Some(7),
                    fencing: 3,
                    expires_at: 100,
                    next_fencing: 4,
                },
            ),
            (Opcode::StreamCreate, ResponseBody::StreamCreated),
            (
                Opcode::StreamAppend,
                ResponseBody::StreamAppended { offset: 5 },
            ),
            (
                Opcode::StreamRead,
                ResponseBody::StreamEntries {
                    entries: vec![StreamEntryBody {
                        offset: 5,
                        partition: b"p".to_vec(),
                        payload: b"e".to_vec(),
                    }],
                    next_offset: 6,
                },
            ),
            (
                Opcode::StreamTrim,
                ResponseBody::StreamTrimmed { removed: 2 },
            ),
        ]
    }

    /// Typed semantic values page through scans without inlining bulk.
    #[test]
    fn semantic_scan_values_page_intact() {
        let page = Response {
            proof: None,
            status: Status::Ok,
            body: ResponseBody::ScanPage {
                entries: vec![ScanEntryBody {
                    key: b"k".to_vec(),
                    value: ScanValueBody::Semantic {
                        object: 9,
                        descriptor: vec![1, 2, 3],
                    },
                }],
                exhausted: true,
                last_key: Some(b"k".to_vec()),
                tablet: 3,
                range_start: b"a".to_vec(),
                range_end: Some(b"z".to_vec()),
                dir_version: 11,
            },
        };
        let (decoded, back) = Response::decode(&page.encode(Opcode::Scan)).expect("scan page");
        assert_eq!(back, Opcode::Scan);
        assert_eq!(decoded, page);
    }

    /// Every new machine-readable status survives the wire on a
    /// diagnostic body (clients match the status, never the text).
    #[test]
    fn semantic_statuses_ride_diagnostics() {
        for status in [
            Status::BoundedExceeded,
            Status::SemaphoreExhausted,
            Status::LeaseConflict,
            Status::StaleFencing,
            Status::StreamFull,
            Status::TxnDecisionConflict,
            Status::TxnCrossTablet,
        ] {
            let error = Response {
                proof: None,
                status,
                body: ResponseBody::Diagnostic("typed".to_owned()),
            };
            let (decoded, _) = Response::decode(&error.encode(Opcode::Get)).expect("status");
            assert_eq!(decoded.status, status);
            assert_eq!(decoded, error);
        }
    }

    #[test]
    fn malformed_responses_rejected() {
        // Truncated Ok body.
        let mut bytes = Response {
            proof: None,
            status: Status::Ok,
            body: ResponseBody::Stored { version: 1 },
        }
        .encode(Opcode::Set);
        bytes.pop();
        assert!(Response::decode(&bytes).is_err());
        // Ok body inconsistent with a redirect status.
        let mut bad = Response {
            proof: None,
            status: Status::StaleRoute,
            body: ResponseBody::Stored { version: 1 },
        }
        .encode(Opcode::Set);
        let _ = &mut bad;
        assert!(Response::decode(&bad).is_err());
        // Unknown status code.
        let mut bad = Response {
            proof: None,
            status: Status::Ok,
            body: ResponseBody::Exists(true),
        }
        .encode(Opcode::Exists);
        bad[0] = 0xFF;
        bad[1] = 0xFF;
        assert!(Response::decode(&bad).is_err());
    }

    #[test]
    fn handshake_round_trips_and_negotiates() {
        let hello = ClientHello {
            max_frame: 1 << 20,
            required_caps: Capabilities::BASE_V1,
            optional_caps: Capabilities::from_bits_retain(1 << 40),
            desired_worker: Some(WorkerId::from_u64(2)),
        };
        let decoded = ClientHello::decode(&hello.encode()).expect("round trip");
        assert_eq!(decoded, hello);
        assert!(Capabilities::BASE_V1.unknown_required().is_empty());
        assert_eq!(
            Capabilities::from_bits_retain(Capabilities::BASE_V1.bits() | (1 << 9))
                .unknown_required(),
            Capabilities::from_bits_retain(1 << 9)
        );
        // Even optional-space bits fail when placed in the required mask:
        // unknown-required means the client demands something undefined.
        assert_eq!(
            Capabilities::from_bits_retain(1 << 40).unknown_required(),
            Capabilities::from_bits_retain(1 << 40)
        );
        // Unknown bits survive the round trip (retained, never truncated),
        // so negotiation inspects what the peer actually sent.
        assert_eq!(decoded.optional_caps.bits(), 1 << 40);
        let reply = ServerHello {
            major: 1,
            minor: 0,
            caps: Capabilities::BASE_V1,
            cluster: ClusterId::from_u128(0xC1),
            node: NodeId::from_u64(7),
            incarnation: NodeIncarnation::from_u64(3),
            worker: WorkerId::from_u64(2),
            dir_version: DirectoryVersion::from_u64(11),
            max_frame: 1 << 20,
            endpoints: vec![
                (WorkerId::from_u64(0), "127.0.0.1:9100".to_owned()),
                (WorkerId::from_u64(1), "127.0.0.1:9101".to_owned()),
            ],
        };
        let decoded = ServerHello::decode(&reply.encode()).expect("round trip");
        assert_eq!(decoded, reply);
        assert_eq!(decoded.endpoints.len(), 2);
    }

    #[test]
    fn stream_control_messages_round_trip() {
        // Begin with a declared total and an identity.
        let begin = StreamBegin {
            namespace: NS,
            key: b"big:1".to_vec(),
            total_len: Some(1 << 20),
            identity: Some(RequestIdentity::new(
                kivi_types::SessionId::from_u128(0x5EED),
                RequestSeq::from_u64(9),
            )),
            ack_floor: RequestSeq::from_u64(8),
        };
        let decoded = StreamBegin::decode(&begin.encode()).expect("round trip");
        assert_eq!(decoded, begin);
        // Begin without total or identity (unknown length, at-most-once).
        let bare = StreamBegin {
            namespace: NS,
            key: Vec::new(),
            total_len: None,
            identity: None,
            ack_floor: RequestSeq::from_u64(0),
        };
        assert_eq!(
            StreamBegin::decode(&bare.encode()).expect("round trip"),
            bare
        );
        // Ready, abort, and value-begin are fixed small shapes.
        let ready = StreamReady { max_data: 1 << 20 };
        assert_eq!(
            StreamReady::decode(&ready.encode()).expect("round trip"),
            ready
        );
        let abort = StreamAbort {
            reason: "over the upload bound".to_owned(),
        };
        assert_eq!(
            StreamAbort::decode(&abort.encode()).expect("round trip"),
            abort
        );
        let value = ValueStreamBegin {
            total_len: 3_000_000,
        };
        assert_eq!(
            ValueStreamBegin::decode(&value.encode()).expect("round trip"),
            value
        );
        // Truncation and trailing bytes fail closed on every shape.
        let begin_bytes = begin.encode();
        assert!(StreamBegin::decode(&begin_bytes[..begin_bytes.len() - 1]).is_err());
        let ready_bytes = ready.encode();
        assert!(StreamReady::decode(&ready_bytes[..3]).is_err());
        let abort_bytes = abort.encode();
        assert!(StreamAbort::decode(&abort_bytes[..1]).is_err());
        let value_bytes = value.encode();
        assert!(ValueStreamBegin::decode(&value_bytes[..7]).is_err());
        let mut trailing = value_bytes;
        trailing.push(0);
        assert!(ValueStreamBegin::decode(&trailing).is_err());
        // Bad discriminators fail, never default.
        let mut bad = bare.encode();
        let total_flag = 8 + 4 + bare.key.len();
        bad[total_flag] = 7;
        assert!(StreamBegin::decode(&bad).is_err());
    }

    #[test]
    fn redirect_rejects_absurd_endpoints() {
        assert!(matches!(
            RedirectInfo::new(
                DirectoryVersion::from_u64(1),
                TabletId::from_u64(1),
                TabletEpoch::from_u64(1),
                WorkerId::from_u64(0),
                "x".repeat(300),
                PartitionRange::Hash(HashPrefix::new(0, 0).expect("root")),
            ),
            Err(ProtocolError::Malformed { .. })
        ));
        let _ = Key::from("typed");
    }
}
