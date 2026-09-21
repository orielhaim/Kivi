//! Typed native operations, results, and failures.
//!
//! The core never sees Redis command strings: callers express intent with
//! [`Operation`], whose semantics are explicit and stable. Wrong-type access
//! fails without mutation; counter overflow fails without mutation; expired
//! objects behave as absent (a read never deletes).

use bytes::Bytes;
use kivi_codec::{CodecError, Decode, Encode, decode_byte_vec, encode_bytes};
use kivi_types::{Expiry, ManifestId, TabletId, WallTimestamp};

use crate::mutation::{ApplyOutcome, Mutation};
use crate::object::{Key, ObjectType, ObjectVersion};
use crate::txn::{TxnId, TxnWrite};

/// One typed native operation against a single key.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Operation {
    /// Fetch bytes; fails with [`WrongType`](OpError::WrongType) on counters.
    Get {
        /// Key to read.
        key: Key,
    },
    /// Store bytes, overwriting any type and clearing any expiry (SET semantics).
    Set {
        /// Key to write.
        key: Key,
        /// Value to store.
        value: Bytes,
    },
    /// Store a large byte string by chunk reference, overwriting any type
    /// and clearing any expiry — the chunked spelling of [`Set`](Self::Set).
    /// The engine stages every referenced chunk and proves pack durability
    /// *before* admitting this; the store trusts the reference exactly as
    /// it trusts an inline value. Produced by the streaming commit path and
    /// by large legacy `Set`s the engine converts at admission, never by
    /// end users directly.
    SetChunked {
        /// Key to write.
        key: Key,
        /// Manifest addressing the immutable chunk sequence.
        manifest: ManifestId,
        /// Total logical bytes across the manifest.
        logical_len: u64,
    },
    /// Store a medium byte string by Memory Fabric reference, overwriting
    /// any type and clearing any expiry — the fabric spelling of
    /// [`Set`](Self::Set). The engine stages the bytes into the fabric
    /// and proves the materialization *before* admitting this; the store
    /// trusts the reference exactly as it trusts an inline value.
    /// Produced by medium `Set`s the engine converts at admission, never
    /// by end users directly.
    SetFabric {
        /// Key to write.
        key: Key,
        /// Fabric-scoped object id assigned at staging.
        fabric_id: u64,
        /// Total logical bytes of the materialized value.
        logical_len: u64,
        /// Logical version the bytes were published at (pins reads).
        version: u64,
    },
    /// Patch a byte range of a value (partial update). Against absent or
    /// expired state the base is the empty string; past-the-end gaps
    /// zero-pad (Redis `SETRANGE` semantics). Against byte strings the
    /// untouched prefix and suffix survive; against counters this fails with
    /// [`WrongType`](OpError::WrongType) without mutation. Unlike
    /// [`Set`](Self::Set), a patch preserves the live expiry: it touches
    /// bytes, never the TTL. Large results the engine converts to chunked
    /// roots at admission (carrying the preserved expiry), so the store
    /// spells the outcome as whatever representation fits.
    SetRange {
        /// Key to patch.
        key: Key,
        /// Logical byte index the patch overwrites from.
        offset: u64,
        /// Bytes written starting at `offset`.
        patch: Bytes,
    },
    /// Remove the key (including expired-but-unreclaimed state).
    Delete {
        /// Key to remove.
        key: Key,
    },
    /// Whether a live (present and unexpired) value exists, of any type.
    Exists {
        /// Key to probe.
        key: Key,
    },
    /// Fetch a counter value; fails with `WrongType` on byte strings.
    CounterGet {
        /// Key to read.
        key: Key,
    },
    /// Add `delta` to a counter, creating it at `delta` when absent.
    /// Negative deltas decrement. Overflow fails without mutation.
    CounterAdd {
        /// Key to update.
        key: Key,
        /// Signed addend.
        delta: i64,
    },
    /// Attach an absolute expiry; missing keys report `applied: false`.
    ExpireAt {
        /// Key to expire.
        key: Key,
        /// Logical expiration timestamp.
        expires_at: WallTimestamp,
    },
    /// Remove any expiry; keys without one report `removed: false`.
    PersistExpiry {
        /// Key to persist.
        key: Key,
    },
    /// Read the expiry of a live key (`None` when the key is absent).
    GetExpiry {
        /// Key to inspect.
        key: Key,
    },
    /// Read a byte slice of a value without materializing the whole object
    /// on the caller's terms: `[offset, offset + len)` clamped to the
    /// logical length (empty when `offset` is past the end). Against absent
    /// or expired state answers `None`; against counters fails with
    /// [`WrongType`](OpError::WrongType) without mutation. The engine may
    /// satisfy this from only the required immutable chunks instead of the
    /// full payload.
    GetRange {
        /// Key to slice.
        key: Key,
        /// Logical byte index the slice starts at.
        offset: u64,
        /// Maximum bytes to return from `offset`.
        len: u64,
    },
    /// Report the logical byte length of a value (`None` when absent).
    /// Uses logical metadata only: chunked roots answer from the root
    /// without reading any chunk payload. Counters fail with
    /// [`WrongType`](OpError::WrongType) without mutation.
    BytesLength {
        /// Key to measure.
        key: Key,
    },
    /// Report the logical version of a live key (`None` when absent).
    /// Metadata only, like [`BytesLength`](Self::BytesLength): chunked
    /// roots answer from the root, counters answer like any other type.
    /// Index maintenance and OCC planning read versions without fetching
    /// (possibly huge) values.
    GetVersion {
        /// Key to inspect.
        key: Key,
    },
    /// Conditionally store bytes, atomically at the owning tablet. The
    /// condition is evaluated against live state (present and unexpired)
    /// and the write applies only when it holds; the expiry policy selects
    /// the stored expiry (`Clear` clears, `Keep` preserves the live expiry
    /// or stays immortal when absent, `ExpireAt` attaches a stamp). There
    /// is deliberately no `ReturnPrevious` in this stage: returning the
    /// previous value would force materialization of arbitrarily large
    /// chunked payloads into a bounded response.
    SetConditional {
        /// Key to write.
        key: Key,
        /// Value to store when the condition holds.
        value: Bytes,
        /// Presence condition evaluated atomically.
        condition: SetCondition,
        /// Expiry policy for the stored object.
        expiry: ExpiryPolicy,
    },
    /// Conditionally store a large byte string by chunk reference: the
    /// chunked spelling of [`SetConditional`](Self::SetConditional). The
    /// engine stages every referenced chunk and proves pack durability
    /// *before* admitting this; the store trusts the reference exactly as
    /// it trusts an inline value. Produced by large conditional stores the
    /// engine converts at admission, never by end users directly.
    SetConditionalChunked {
        /// Key to write.
        key: Key,
        /// Manifest addressing the immutable chunk sequence.
        manifest: ManifestId,
        /// Total logical bytes across the manifest.
        logical_len: u64,
        /// Presence condition evaluated atomically.
        condition: SetCondition,
        /// Expiry policy for the stored object.
        expiry: ExpiryPolicy,
    },
    /// Conditionally store a medium byte string by fabric reference: the
    /// fabric spelling of [`SetConditional`](Self::SetConditional). Same
    /// staging contract as [`SetFabric`](Self::SetFabric), produced by
    /// medium conditional stores the engine converts at admission.
    SetConditionalFabric {
        /// Key to write.
        key: Key,
        /// Fabric-scoped object id assigned at staging.
        fabric_id: u64,
        /// Total logical bytes of the materialized value.
        logical_len: u64,
        /// Logical version the bytes were published at (pins reads).
        version: u64,
        /// Presence condition evaluated atomically.
        condition: SetCondition,
        /// Expiry policy for the stored object.
        expiry: ExpiryPolicy,
    },
    /// Reserve one key for a distributed transaction (2PC prepare): OCC
    /// version validation plus durable intent recording, with no
    /// user-visible mutation yet. Driven by the transaction coordinator
    /// through the normal propose path, never by end users directly.
    TxnPrepare {
        /// Transaction identity.
        txn: TxnId,
        /// Coordinator tablet owning the decision record.
        coordinator: TabletId,
        /// Prepared write (key, kind, OCC expectation).
        write: TxnWrite,
        /// Digest over the full write set (binds this reservation to the
        /// exact transaction; a conflicting digest under one `TxnId` is
        /// rejected, never overwritten).
        digest: [u8; 32],
    },
    /// Resolve one key's transaction intent: commit applies the prepared
    /// write, abort discards it. Idempotent by `TxnId`.
    TxnFinalize {
        /// Transaction identity.
        txn: TxnId,
        /// Key whose intent resolves.
        key: Key,
        /// Whether to apply (`true`) or discard (`false`) the intent.
        commit: bool,
        /// Digest the intent was prepared for (must match the reservation;
        /// mismatches finalize nothing and report conflict).
        digest: [u8; 32],
    },
    /// Commit a same-tablet write set as one ordered atomic mutation: every
    /// write validates and applies together, or nothing does. Only drivers
    /// construct this, and only for write sets already proven single-tablet
    /// (non-empty; all keys route to one tablet); cross-tablet sets use
    /// 2PC intents instead. The key drivers route on is the first write's
    /// key — all keys share its tablet by construction.
    TxnCommitLocal {
        /// Transaction identity (idempotency key for the whole batch).
        txn: TxnId,
        /// Writes in request order (non-empty; duplicates collapse
        /// last-wins, matching the 2PC intent-overwrite rule).
        writes: Vec<TxnWrite>,
    },
    /// Add `delta` to a commutative counter, creating it at `delta` when
    /// absent. Overflow fails without mutation. The result is `Applied`:
    /// no ordinal is ever exposed, so additions commute observably and a
    /// future unordered fast path can execute them safely.
    CommutativeAdd {
        /// Key to update.
        key: Key,
        /// Signed addend.
        delta: i64,
    },
    /// Fetch a commutative counter's current sum (`None` when absent; a
    /// state query, not an ordinal).
    CommutativeGet {
        /// Key to read.
        key: Key,
    },
    /// Create a bounded counter with global `capacity` and a full local
    /// escrow share `[0, capacity]` held by `holder` (the tablet executing
    /// the create, filled by the driver after routing). Fails without
    /// mutation when the key already holds live state.
    BoundedCounterCreate {
        /// Key to create.
        key: Key,
        /// Global capacity (inclusive upper bound).
        capacity: u64,
        /// Tablet holding the initial full share.
        holder: TabletId,
    },
    /// Add `delta` to a bounded counter within locally owned escrow rights:
    /// the result must satisfy both `0 <= value <= capacity` and
    /// `share.min <= value <= share.max`, else [`BoundedExceeded`](OpError::BoundedExceeded)
    /// without mutation. Rights transfers (not global checks) move the share.
    BoundedCounterAdd {
        /// Key to update.
        key: Key,
        /// Signed addend.
        delta: i64,
    },
    /// Read a bounded counter's value and capacity (`None` when absent).
    BoundedCounterGet {
        /// Key to read.
        key: Key,
    },
    /// Narrow or widen one bounded counter's escrow share durably. Narrowing
    /// (giving rights away) is unilateral; widening (taking rights) must pair
    /// with a matching narrow in the same atomic batch/transaction so total
    /// rights are conserved — see
    /// [`verify_escrow_widths`](crate::txn::verify_escrow_widths).
    /// Requires the live value inside the new share and the new share inside
    /// `[0, capacity]`.
    EscrowTransfer {
        /// Key whose share moves.
        key: Key,
        /// New share lower bound (inclusive).
        new_min: i64,
        /// New share upper bound (inclusive).
        new_max: i64,
    },
    /// Create a semaphore with `capacity` total permits. Fails without
    /// mutation when the key already holds live state.
    SemaphoreCreate {
        /// Key to create.
        key: Key,
        /// Total capacity (outstanding never exceeds this).
        capacity: u64,
    },
    /// Acquire `qty` permits under `permit`: idempotent by permit id (retry
    /// with the same id returns the original outcome, never a second
    /// acquisition). Fails with [`SemaphoreExhausted`](OpError::SemaphoreExhausted)
    /// when `outstanding + qty > capacity`.
    SemaphoreAcquire {
        /// Key of the semaphore.
        key: Key,
        /// Permit identity (client-minted, stable across retries).
        permit: crate::PermitId,
        /// Owner/session identity acquiring.
        owner: u64,
        /// Quantity to acquire.
        qty: u64,
    },
    /// Release the permits held under `permit`. Idempotent: releasing an
    /// unknown or already-released id reports `released: false` and mints no
    /// capacity. A release can never create capacity beyond the declared
    /// maximum.
    SemaphoreRelease {
        /// Key of the semaphore.
        key: Key,
        /// Permit identity to release.
        permit: crate::PermitId,
    },
    /// Inspect a semaphore's load: outstanding permits and total capacity.
    /// Never mutates.
    SemaphoreInspect {
        /// Key of the semaphore.
        key: Key,
    },
    /// Acquire (or re-acquire after expiry) the lease for `owner` with a
    /// logical TTL. Success issues a fresh fencing token (`next_fencing`,
    /// strictly increasing) and records `expires_at = now + ttl`. A live
    /// holder blocks other owners with [`LeaseConflict`](OpError::LeaseConflict);
    /// retrying the winning acquisition (same owner, same request identity
    /// at the engine layer) returns the original grant.
    LeaseAcquire {
        /// Key of the lease.
        key: Key,
        /// Owner/session identity acquiring.
        owner: u64,
        /// Logical time-to-live in micros from `now`.
        ttl_micros: u64,
    },
    /// Extend a live holding: requires the exact current fencing token and a
    /// live, unexpired grant for `owner`. A stale token fails with
    /// [`StaleFencing`](OpError::StaleFencing) and can never revive a
    /// replaced/expired lease.
    LeaseRenew {
        /// Key of the lease.
        key: Key,
        /// Owner/session identity renewing.
        owner: u64,
        /// Fencing token of the grant being renewed.
        fencing: crate::FencingToken,
        /// Logical time-to-live in micros from `now`.
        ttl_micros: u64,
    },
    /// Release a live holding: requires the exact current fencing token for
    /// `owner`. A stale token fails with [`StaleFencing`](OpError::StaleFencing)
    /// and can never free a newer holder's lease.
    LeaseRelease {
        /// Key of the lease.
        key: Key,
        /// Owner/session identity releasing.
        owner: u64,
        /// Fencing token of the grant being released.
        fencing: crate::FencingToken,
    },
    /// Inspect the lease: current holder (if live at `now`) and the next
    /// fencing token to be issued. Never mutates.
    LeaseInspect {
        /// Key of the lease.
        key: Key,
    },
    /// Create one shard of a sharded stream. Shards are independent ordered
    /// logs: total order within the shard, no global order across shards.
    StreamCreate {
        /// Key holding this shard's log.
        key: Key,
        /// Stream identity (shared by all shards of the stream).
        stream: [u8; 16],
        /// Shard index within the stream.
        shard: u32,
    },
    /// Append one entry to a shard: routed by explicit `partition_key`
    /// (stable/deterministic shard selection by the caller), assigned the
    /// shard's next offset. Idempotent under the engine's request identity;
    /// the offset is shard-local and monotonic, never global.
    StreamAppend {
        /// Key holding this shard's log.
        key: Key,
        /// Partition key that routed this entry here (retained for audit).
        partition: Vec<u8>,
        /// Entry payload bytes (bounded by [`MAX_STREAM_ENTRY_BYTES`](crate::MAX_STREAM_ENTRY_BYTES)).
        payload: Bytes,
    },
    /// Read a shard's entries from `from_offset` (inclusive), up to
    /// `max_entries`. Never implies anything about other shards.
    StreamRead {
        /// Key holding this shard's log.
        key: Key,
        /// First offset to return (inclusive).
        from_offset: u64,
        /// Maximum entries to return.
        max_entries: u32,
    },
    /// Trim a shard's retained prefix through `through_offset` (inclusive).
    /// `next_offset` never rewinds: trimmed offsets are gone, not reusable.
    StreamTrim {
        /// Key holding this shard's log.
        key: Key,
        /// Trim retained entries with `offset <= through_offset`.
        through_offset: u64,
    },
}

/// Presence condition for [`Operation::SetConditional`]: evaluated against
/// live state (present and unexpired) atomically at the owning tablet.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SetCondition {
    /// Always store (same admission as [`Operation::Set`] plus policy).
    Always,
    /// Store only when no live value exists.
    IfAbsent,
    /// Store only when a live value exists.
    IfPresent,
}

/// Expiry policy for [`Operation::SetConditional`]: a Kivi-owned model,
/// not Redis `EX`/`PX`/`KEEPTTL` flags (the RESP adapter translates those
/// onto this).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ExpiryPolicy {
    /// Clear any expiry (same as [`Operation::Set`]).
    Clear,
    /// Preserve the live expiry, or stay immortal when creating.
    Keep,
    /// Attach this absolute logical timestamp on store.
    ExpireAt(WallTimestamp),
}

impl Operation {
    /// Returns the single key this operation touches. Every operation is
    /// single-key except [`TxnCommitLocal`](Self::TxnCommitLocal), which
    /// answers its first write's key (all its keys share one tablet by
    /// driver construction, so routing on the first is exact). Single-key
    /// soundness is what makes batch-local overlays work: preparation reads
    /// no state beyond the op key, and local commits stage every touched
    /// key explicitly.
    ///
    /// # Panics
    ///
    /// Panics on [`TxnCommitLocal`](Self::TxnCommitLocal) with an empty
    /// write list: drivers never construct one, so an empty commit here
    /// is a caller bug.
    #[must_use]
    pub fn key(&self) -> &Key {
        match self {
            Self::Get { key }
            | Self::Set { key, .. }
            | Self::SetChunked { key, .. }
            | Self::SetFabric { key, .. }
            | Self::SetRange { key, .. }
            | Self::Delete { key }
            | Self::Exists { key }
            | Self::CounterGet { key }
            | Self::CounterAdd { key, .. }
            | Self::ExpireAt { key, .. }
            | Self::PersistExpiry { key }
            | Self::GetExpiry { key }
            | Self::GetRange { key, .. }
            | Self::BytesLength { key }
            | Self::GetVersion { key }
            | Self::SetConditional { key, .. }
            | Self::SetConditionalChunked { key, .. }
            | Self::SetConditionalFabric { key, .. }
            | Self::TxnFinalize { key, .. }
            | Self::CommutativeAdd { key, .. }
            | Self::CommutativeGet { key, .. }
            | Self::BoundedCounterCreate { key, .. }
            | Self::BoundedCounterAdd { key, .. }
            | Self::BoundedCounterGet { key, .. }
            | Self::EscrowTransfer { key, .. }
            | Self::SemaphoreCreate { key, .. }
            | Self::SemaphoreAcquire { key, .. }
            | Self::SemaphoreRelease { key, .. }
            | Self::SemaphoreInspect { key, .. }
            | Self::LeaseAcquire { key, .. }
            | Self::LeaseRenew { key, .. }
            | Self::LeaseRelease { key, .. }
            | Self::LeaseInspect { key, .. }
            | Self::StreamCreate { key, .. }
            | Self::StreamAppend { key, .. }
            | Self::StreamRead { key, .. }
            | Self::StreamTrim { key, .. } => key,
            Self::TxnPrepare { write, .. } => &write.key,
            Self::TxnCommitLocal { writes, .. } => {
                &writes.first().expect("local commits are never empty").key
            }
        }
    }

    /// Whether this operation can change logical state (and therefore enters
    /// the WAL in durable mode). Reads never do, whatever their outcome.
    /// This is the semantic boundary both frontends share: the native wire
    /// decoder and the RESP adapter meet here, never in opcode bytes.
    #[must_use]
    pub const fn is_mutating(&self) -> bool {
        match self {
            Self::Set { .. }
            | Self::SetChunked { .. }
            | Self::SetFabric { .. }
            | Self::SetRange { .. }
            | Self::Delete { .. }
            | Self::CounterAdd { .. }
            | Self::ExpireAt { .. }
            | Self::PersistExpiry { .. }
            | Self::SetConditional { .. }
            | Self::SetConditionalChunked { .. }
            | Self::SetConditionalFabric { .. }
            | Self::TxnPrepare { .. }
            | Self::TxnFinalize { .. }
            | Self::TxnCommitLocal { .. }
            | Self::CommutativeAdd { .. }
            | Self::BoundedCounterCreate { .. }
            | Self::BoundedCounterAdd { .. }
            | Self::EscrowTransfer { .. }
            | Self::SemaphoreCreate { .. }
            | Self::SemaphoreAcquire { .. }
            | Self::SemaphoreRelease { .. }
            | Self::LeaseAcquire { .. }
            | Self::LeaseRenew { .. }
            | Self::LeaseRelease { .. }
            | Self::StreamCreate { .. }
            | Self::StreamAppend { .. }
            | Self::StreamTrim { .. } => true,
            Self::Get { .. }
            | Self::Exists { .. }
            | Self::CounterGet { .. }
            | Self::GetExpiry { .. }
            | Self::GetRange { .. }
            | Self::GetVersion { .. }
            | Self::BytesLength { .. }
            | Self::CommutativeGet { .. }
            | Self::BoundedCounterGet { .. }
            | Self::LeaseInspect { .. }
            | Self::SemaphoreInspect { .. }
            | Self::StreamRead { .. } => false,
        }
    }
}

/// Outcome of one prepared operation.
///
/// Exhaustive by design: every consumer in the workspace must handle every
/// outcome, so adding an operation breaks matches at compile time instead of
/// silently falling into a wildcard.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OperationResult {
    /// `Get` payload (`None` when absent, expired, or — via error — mistyped).
    Value(Option<Bytes>),
    /// `Get` hit a chunked root: the engine must resolve these bytes
    /// through the chunk lane before replying. Never crosses the wire and
    /// never persists: reads answer inline, so no WAL record, dedup entry,
    /// or checkpoint band ever names this variant. It exists so the
    /// deterministic store can name "bytes live elsewhere" without
    /// performing I/O itself.
    ChunkedValue {
        /// Manifest addressing the immutable chunk sequence.
        manifest: ManifestId,
        /// Total logical bytes across the manifest.
        logical_len: u64,
    },
    /// `Get` hit a fabric root: the engine must resolve these bytes
    /// through the Memory Fabric before replying, suspending for
    /// asynchronous promotion when the primary is off-core. Same
    /// never-wire/never-persist contract as
    /// [`ChunkedValue`](Self::ChunkedValue). The `version` pins the read:
    /// only a materialization published at this version may serve it.
    FabricValue {
        /// Fabric-scoped object id assigned at staging.
        fabric_id: u64,
        /// Total logical bytes of the materialized value.
        logical_len: u64,
        /// Logical version the bytes were published at.
        version: u64,
    },
    /// `Set` completed with the new version.
    Stored {
        /// Version after the store.
        version: crate::object::ObjectVersion,
    },
    /// `Delete` completed; `existed` names a previously live object.
    Deleted {
        /// Whether a live object was removed.
        existed: bool,
    },
    /// `Exists` answer.
    Exists(bool),
    /// `CounterGet` value (`None` when absent).
    Counter(Option<i64>),
    /// `CounterAdd` completed with the new value and version.
    CounterUpdated {
        /// Value after the addition.
        value: i64,
        /// Version after the addition.
        version: crate::object::ObjectVersion,
    },
    /// `ExpireAt` outcome.
    ExpirySet {
        /// Whether the key was live and took the expiry.
        applied: bool,
    },
    /// `PersistExpiry` outcome.
    ExpiryPersisted {
        /// Whether an expiry was actually removed.
        removed: bool,
    },
    /// `GetExpiry` answer (`None` when the key is absent).
    Expiry(Option<Expiry>),
    /// `BytesLength` answer (`None` when the key is absent).
    Length(Option<u64>),
    /// `GetVersion` answer: the live version (`None` when absent).
    Version(Option<crate::object::ObjectVersion>),
    /// `SetConditional` outcome: whether the condition held and the store
    /// applied. `applied: false` carries no version and mutated nothing.
    ConditionalSet {
        /// Whether the condition held and the value was stored.
        applied: bool,
        /// Version after the store (`None` when not applied).
        version: Option<crate::object::ObjectVersion>,
    },
    /// `TxnPrepare` outcome: the intent is durably reserved.
    TxnPrepared,
    /// A transactional prepare or conflicting write met OCC/intent
    /// contention (deterministic outcome, safe to persist and retry).
    TxnConflict,
    /// `TxnFinalize` outcome: whether an intent was applied.
    TxnFinalized {
        /// Whether a prepared write was applied (`false` for aborts and
        /// idempotent replays with no intent left).
        applied: bool,
        /// Version after the applied write (`None` when nothing applied).
        version: Option<crate::object::ObjectVersion>,
    },
    /// Same-tablet atomic commit outcome: resulting versions in request
    /// order (`None` for deletes). Shapes as `AtomicCommitted` on the wire.
    TxnLocalCommitted {
        /// Resulting versions in request order.
        versions: Vec<Option<crate::object::ObjectVersion>>,
    },
    /// `CommutativeAdd` outcome: applied, with no ordinal. The absence of a
    /// value here is the semantic point: permutations of applied additions
    /// produce identical observable histories.
    CommutativeApplied,
    /// `CommutativeGet` value (`None` when absent).
    CommutativeValue(Option<i64>),
    /// Bounded-counter write outcome with the post-apply value and version.
    BoundedUpdated {
        /// Value after the addition.
        value: i64,
        /// Version after the update.
        version: crate::object::ObjectVersion,
    },
    /// `BoundedCounterGet` answer (`None`s when absent).
    BoundedValue {
        /// Current value (`None` when absent).
        value: Option<i64>,
        /// Global capacity (`None` when absent).
        capacity: Option<u64>,
        /// Locally owned escrow share `[min, max]` (`None` when absent;
        /// read-only planning input for rights transfers).
        share: Option<(i64, i64)>,
    },
    /// `SemaphoreAcquire` outcome: the permit is live (or the identical
    /// retry was recognized).
    SemaphoreAcquired,
    /// `SemaphoreRelease` outcome.
    SemaphoreReleased {
        /// Whether a live permit was actually released (`false` for unknown
        /// or already-released ids: idempotent, mints nothing).
        released: bool,
    },
    /// Semaphore load answer: outstanding permits and total capacity.
    SemaphoreLoad {
        /// Total outstanding quantity.
        outstanding: u64,
        /// Total capacity.
        capacity: u64,
    },
    /// `LeaseAcquire` outcome: the grant with its fencing token. The token
    /// — not the expiry — is what external systems must compare.
    LeaseAcquired {
        /// Fencing token issued with this grant (strictly increasing).
        fencing: crate::FencingToken,
        /// Logical expiry of the grant.
        expires_at: WallTimestamp,
    },
    /// `LeaseRenew` outcome: the extended grant (same fencing token).
    LeaseRenewed {
        /// Fencing token of the continuing grant.
        fencing: crate::FencingToken,
        /// New logical expiry of the grant.
        expires_at: WallTimestamp,
    },
    /// `LeaseRelease` outcome.
    LeaseReleased {
        /// Whether a live holding was actually released (`false` when no
        /// live holding existed: idempotent).
        released: bool,
    },
    /// `LeaseInspect` answer: the live holder at `now` (`None` when free or
    /// expired) plus the next fencing token to be issued.
    LeaseInfo {
        /// Live holder (`None` when free or expired-but-unreclaimed).
        holder: Option<crate::LeaseHolder>,
        /// Next fencing token to be issued.
        next_fencing: crate::FencingToken,
    },
    /// `StreamCreate` outcome.
    StreamCreated,
    /// `StreamAppend` outcome: the shard-local offset assigned.
    StreamAppended {
        /// Assigned offset (shard-local, monotonic, never global).
        offset: u64,
    },
    /// `StreamRead` answer: entries in shard order plus the shard's next
    /// offset (the read cursor for the following page).
    StreamEntries {
        /// Entries in offset order.
        entries: Vec<crate::StreamEntry>,
        /// Shard's next offset (exclusive upper bound of the log).
        next_offset: u64,
    },
    /// `StreamTrim` outcome: how many retained entries were dropped.
    StreamTrimmed {
        /// Entries removed.
        removed: u64,
    },
}

/// Native operation failure. Reads and validations fail; validated writes
/// apply deterministically.
///
/// Exhaustive like [`OperationResult`]: new failure modes must be handled
/// explicitly everywhere they surface.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum OpError {
    /// The key holds a different logical type; nothing was mutated.
    #[error("wrong type: operation needs {expected}, key holds {found}")]
    WrongType {
        /// Type the operation required.
        expected: ObjectType,
        /// Type actually stored.
        found: ObjectType,
    },
    /// A counter addition overflowed `i64`; nothing was mutated.
    #[error("counter addition overflows i64")]
    CounterOverflow,
    /// A `SetRange` met a chunked root its admission plan did not expect:
    /// the root changed representation between the engine's peek and
    /// preparation. Nothing was mutated; the client retries and the retry
    /// re-plans against the new root. Unreachable on the current
    /// single-owner paths (peek and prepare share one synchronous tablet
    /// turn), kept total for future replica proposals.
    #[error("range base changed representation during admission; retry")]
    StaleRangeBase,
    /// A transactional prepare failed OCC validation (expected version
    /// moved) or a prepared intent from another transaction blocks the key.
    /// Nothing was mutated; the coordinator aborts or retries with a fresh
    /// attempt.
    #[error("transaction conflict: expected version moved or key reserved")]
    TxnConflict,
    /// A bounded-counter addition would leave locally owned escrow rights
    /// (or the global `[0, capacity]` bound). Nothing was mutated; move
    /// rights first, then retry.
    #[error("bounded counter would exceed owned escrow rights")]
    BoundedExceeded,
    /// A semaphore acquisition would exceed capacity. Nothing was mutated.
    #[error("semaphore capacity exhausted")]
    SemaphoreExhausted,
    /// A lease acquisition met a live holder owned by someone else, or a
    /// create met existing live state. Nothing was mutated.
    #[error("lease held by another live owner")]
    LeaseConflict,
    /// A lease renew/release named a fencing token that is not the current
    /// grant's. Nothing was mutated; the stale holder must re-acquire.
    #[error("stale fencing token: not the current lease grant")]
    StaleFencing,
    /// A stream append would exceed the shard's entry or payload bounds, or
    /// a fencing/lease token space is exhausted. Nothing was mutated.
    #[error("stream shard full")]
    StreamFull,
    /// The key holds no live value of the required type (a semaphore or
    /// stream shard must be created before use). Nothing was mutated.
    #[error("no live value under the key")]
    NotFound,
}

/// Canonical wire tags. Fixed forever within framing version 1.
const TAG_OP_WRONG_TYPE: u8 = 1;
const TAG_OP_OVERFLOW: u8 = 2;
/// Stale range-base tag. New tags never reuse old ones.
const TAG_OP_STALE_RANGE_BASE: u8 = 3;
/// Transaction-conflict tag. New tags never reuse old ones.
const TAG_OP_TXN_CONFLICT: u8 = 4;
/// Bounded-rights exceeded tag. New tags never reuse old ones.
const TAG_OP_BOUNDED_EXCEEDED: u8 = 5;
/// Semaphore-exhausted tag. New tags never reuse old ones.
const TAG_OP_SEMAPHORE_EXHAUSTED: u8 = 6;
/// Lease-held-by-other tag. New tags never reuse old ones.
const TAG_OP_LEASE_CONFLICT: u8 = 7;
/// Stale-fencing tag. New tags never reuse old ones.
const TAG_OP_STALE_FENCING: u8 = 8;
/// Stream-full tag. New tags never reuse old ones.
const TAG_OP_STREAM_FULL: u8 = 9;
/// Not-found tag. New tags never reuse old ones.
const TAG_OP_NOT_FOUND: u8 = 10;

impl Encode for OpError {
    fn encoded_len(&self) -> usize {
        match self {
            Self::WrongType { .. } => 1 + 1 + 1,
            Self::CounterOverflow
            | Self::StaleRangeBase
            | Self::TxnConflict
            | Self::BoundedExceeded
            | Self::SemaphoreExhausted
            | Self::LeaseConflict
            | Self::StaleFencing
            | Self::StreamFull
            | Self::NotFound => 1,
        }
    }

    fn encode(&self, out: &mut Vec<u8>) {
        match self {
            Self::WrongType { expected, found } => {
                out.push(TAG_OP_WRONG_TYPE);
                expected.encode(out);
                found.encode(out);
            }
            Self::CounterOverflow => out.push(TAG_OP_OVERFLOW),
            Self::StaleRangeBase => out.push(TAG_OP_STALE_RANGE_BASE),
            Self::TxnConflict => out.push(TAG_OP_TXN_CONFLICT),
            Self::BoundedExceeded => out.push(TAG_OP_BOUNDED_EXCEEDED),
            Self::SemaphoreExhausted => out.push(TAG_OP_SEMAPHORE_EXHAUSTED),
            Self::LeaseConflict => out.push(TAG_OP_LEASE_CONFLICT),
            Self::StaleFencing => out.push(TAG_OP_STALE_FENCING),
            Self::StreamFull => out.push(TAG_OP_STREAM_FULL),
            Self::NotFound => out.push(TAG_OP_NOT_FOUND),
        }
    }
}

impl Decode for OpError {
    fn decode(input: &[u8]) -> Result<(Self, usize), CodecError> {
        let (tag, first) = u8::decode(input)?;
        match tag {
            TAG_OP_WRONG_TYPE => {
                let (expected, second) = ObjectType::decode(&input[first..])?;
                let (found, third) = ObjectType::decode(&input[first + second..])?;
                Ok((Self::WrongType { expected, found }, first + second + third))
            }
            TAG_OP_OVERFLOW => Ok((Self::CounterOverflow, first)),
            TAG_OP_STALE_RANGE_BASE => Ok((Self::StaleRangeBase, first)),
            TAG_OP_TXN_CONFLICT => Ok((Self::TxnConflict, first)),
            TAG_OP_BOUNDED_EXCEEDED => Ok((Self::BoundedExceeded, first)),
            TAG_OP_SEMAPHORE_EXHAUSTED => Ok((Self::SemaphoreExhausted, first)),
            TAG_OP_LEASE_CONFLICT => Ok((Self::LeaseConflict, first)),
            TAG_OP_STALE_FENCING => Ok((Self::StaleFencing, first)),
            TAG_OP_STREAM_FULL => Ok((Self::StreamFull, first)),
            TAG_OP_NOT_FOUND => Ok((Self::NotFound, first)),
            other => Err(CodecError::InvalidTag {
                kind: "op-error",
                tag: other,
            }),
        }
    }
}

/// Canonical wire tags for [`OperationResult`]. Fixed forever within
/// framing version 1.
const TAG_RES_VALUE: u8 = 1;
const TAG_RES_STORED: u8 = 2;
const TAG_RES_DELETED: u8 = 3;
const TAG_RES_EXISTS: u8 = 4;
const TAG_RES_COUNTER: u8 = 5;
const TAG_RES_COUNTER_UPDATED: u8 = 6;
const TAG_RES_EXPIRY_SET: u8 = 7;
const TAG_RES_EXPIRY_PERSISTED: u8 = 8;
const TAG_RES_EXPIRY: u8 = 9;
/// Engine-internal chunked-read marker. New tag, never reused; see the
/// variant docs for why it never persists.
const TAG_RES_CHUNKED: u8 = 10;
/// Logical byte-length answer tag. New tags never reuse old ones.
const TAG_RES_LENGTH: u8 = 11;
/// Version answer tag. New tags never reuse old ones.
const TAG_RES_VERSION: u8 = 16;
/// Conditional-set outcome tag. New tags never reuse old ones.
const TAG_RES_CONDITIONAL_SET: u8 = 12;
/// Transaction-prepared outcome tag. New tags never reuse old ones.
const TAG_RES_TXN_PREPARED: u8 = 13;
/// Transaction-conflict outcome tag. New tags never reuse old ones.
const TAG_RES_TXN_CONFLICT: u8 = 14;
/// Transaction-finalized outcome tag. New tags never reuse old ones.
const TAG_RES_TXN_FINALIZED: u8 = 15;
/// Commutative-applied tag (no ordinal carried, by design).
const TAG_RES_COMMUTATIVE_APPLIED: u8 = 17;
/// Commutative-value tag.
const TAG_RES_COMMUTATIVE_VALUE: u8 = 18;
/// Bounded-updated tag.
const TAG_RES_BOUNDED_UPDATED: u8 = 19;
/// Bounded-value tag.
const TAG_RES_BOUNDED_VALUE: u8 = 20;
/// Semaphore-acquired tag.
const TAG_RES_SEMAPHORE_ACQUIRED: u8 = 21;
/// Semaphore-released tag.
const TAG_RES_SEMAPHORE_RELEASED: u8 = 22;
/// Semaphore-load tag.
const TAG_RES_SEMAPHORE_LOAD: u8 = 23;
/// Lease-acquired tag.
const TAG_RES_LEASE_ACQUIRED: u8 = 24;
/// Lease-renewed tag.
const TAG_RES_LEASE_RENEWED: u8 = 25;
/// Lease-released tag.
const TAG_RES_LEASE_RELEASED: u8 = 26;
/// Lease-info tag.
const TAG_RES_LEASE_INFO: u8 = 27;
/// Stream-created tag.
const TAG_RES_STREAM_CREATED: u8 = 28;
/// Stream-appended tag.
const TAG_RES_STREAM_APPENDED: u8 = 29;
/// Stream-entries tag.
const TAG_RES_STREAM_ENTRIES: u8 = 30;
/// Stream-trimmed tag.
const TAG_RES_STREAM_TRIMMED: u8 = 31;
/// Same-tablet atomic commit tag. New tags never reuse old ones.
const TAG_RES_TXN_LOCAL_COMMITTED: u8 = 32;
/// Engine-internal fabric-read marker. New tag, never reused; same
/// never-persists contract as [`TAG_RES_CHUNKED`].
const TAG_RES_FABRIC: u8 = 33;

impl Encode for OperationResult {
    fn encoded_len(&self) -> usize {
        match self {
            Self::Value(value) => 1 + 1 + value.as_ref().map_or(0, |bytes| 4 + bytes.len()),
            Self::ChunkedValue { .. } => 1 + 32 + 8,
            Self::FabricValue { .. } => 1 + 8 + 8 + 8,
            Self::Stored { .. } | Self::StreamAppended { .. } | Self::StreamTrimmed { .. } => 1 + 8,
            Self::Deleted { .. }
            | Self::Exists(_)
            | Self::ExpirySet { .. }
            | Self::ExpiryPersisted { .. }
            | Self::SemaphoreReleased { .. }
            | Self::LeaseReleased { .. } => 1 + 1,
            Self::Counter(value) | Self::CommutativeValue(value) => {
                1 + 1 + usize::from(value.is_some()) * 8
            }
            Self::CounterUpdated { .. }
            | Self::BoundedUpdated { .. }
            | Self::SemaphoreLoad { .. } => 1 + 8 + 8,
            Self::Expiry(value) => 1 + 1 + value.as_ref().map_or(0, Expiry::encoded_len),
            Self::Length(value) => 1 + 1 + usize::from(value.is_some()) * 8,
            Self::Version(value) => 1 + 1 + usize::from(value.is_some()) * 8,
            Self::ConditionalSet { version, .. } => 1 + 1 + 1 + version.as_ref().map_or(0, |_| 8),
            Self::TxnPrepared
            | Self::TxnConflict
            | Self::CommutativeApplied
            | Self::SemaphoreAcquired
            | Self::StreamCreated => 1,
            Self::TxnFinalized { version, .. } => 1 + 1 + 1 + version.as_ref().map_or(0, |_| 8),
            Self::BoundedValue {
                value,
                capacity,
                share,
            } => {
                1 + 1
                    + usize::from(value.is_some()) * 8
                    + 1
                    + usize::from(capacity.is_some()) * 8
                    + 1
                    + usize::from(share.is_some()) * 16
            }
            Self::LeaseAcquired { .. } | Self::LeaseRenewed { .. } => {
                1 + 8 + WallTimestamp::from_micros(0).encoded_len()
            }
            Self::LeaseInfo { holder, .. } => {
                1 + 1 + holder.map_or(0, |holder| holder.encoded_len()) + 8
            }
            Self::StreamEntries { entries, .. } => {
                1 + 4
                    + 8
                    + entries
                        .iter()
                        .map(crate::StreamEntry::encoded_len)
                        .sum::<usize>()
            }
            Self::TxnLocalCommitted { versions } => {
                1 + 2
                    + versions
                        .iter()
                        .map(|version| 1 + usize::from(version.is_some()) * 8)
                        .sum::<usize>()
            }
        }
    }

    #[allow(clippy::too_many_lines)]
    fn encode(&self, out: &mut Vec<u8>) {
        match self {
            Self::Value(value) => {
                out.push(TAG_RES_VALUE);
                match value {
                    None => out.push(0),
                    Some(bytes) => {
                        out.push(1);
                        encode_bytes(out, bytes);
                    }
                }
            }
            Self::Stored { version } => {
                out.push(TAG_RES_STORED);
                version.encode(out);
            }
            Self::Deleted { existed } => {
                out.push(TAG_RES_DELETED);
                existed.encode(out);
            }
            Self::Exists(present) => {
                out.push(TAG_RES_EXISTS);
                present.encode(out);
            }
            Self::Counter(value) => {
                out.push(TAG_RES_COUNTER);
                match value {
                    None => out.push(0),
                    Some(counter) => {
                        out.push(1);
                        out.extend_from_slice(&counter.to_le_bytes());
                    }
                }
            }
            Self::CounterUpdated { value, version } => {
                out.push(TAG_RES_COUNTER_UPDATED);
                out.extend_from_slice(&value.to_le_bytes());
                version.encode(out);
            }
            Self::ExpirySet { applied } => {
                out.push(TAG_RES_EXPIRY_SET);
                applied.encode(out);
            }
            Self::ExpiryPersisted { removed } => {
                out.push(TAG_RES_EXPIRY_PERSISTED);
                removed.encode(out);
            }
            Self::Expiry(value) => {
                out.push(TAG_RES_EXPIRY);
                match value {
                    None => out.push(0),
                    Some(expiry) => {
                        out.push(1);
                        expiry.encode(out);
                    }
                }
            }
            Self::ChunkedValue {
                manifest,
                logical_len,
            } => {
                out.push(TAG_RES_CHUNKED);
                out.extend_from_slice(manifest.as_bytes());
                logical_len.encode(out);
            }
            Self::FabricValue {
                fabric_id,
                logical_len,
                version,
            } => {
                out.push(TAG_RES_FABRIC);
                fabric_id.encode(out);
                logical_len.encode(out);
                version.encode(out);
            }
            Self::Length(value) => {
                out.push(TAG_RES_LENGTH);
                match value {
                    None => out.push(0),
                    Some(len) => {
                        out.push(1);
                        len.encode(out);
                    }
                }
            }
            Self::Version(value) => {
                out.push(TAG_RES_VERSION);
                match value {
                    None => out.push(0),
                    Some(version) => {
                        out.push(1);
                        version.encode(out);
                    }
                }
            }
            Self::ConditionalSet { applied, version } => {
                out.push(TAG_RES_CONDITIONAL_SET);
                applied.encode(out);
                match version {
                    None => out.push(0),
                    Some(version) => {
                        out.push(1);
                        version.encode(out);
                    }
                }
            }
            Self::TxnPrepared => out.push(TAG_RES_TXN_PREPARED),
            Self::TxnConflict => out.push(TAG_RES_TXN_CONFLICT),
            Self::TxnFinalized { applied, version } => {
                out.push(TAG_RES_TXN_FINALIZED);
                applied.encode(out);
                match version {
                    None => out.push(0),
                    Some(version) => {
                        out.push(1);
                        version.encode(out);
                    }
                }
            }
            Self::CommutativeApplied => out.push(TAG_RES_COMMUTATIVE_APPLIED),
            Self::CommutativeValue(value) => {
                out.push(TAG_RES_COMMUTATIVE_VALUE);
                match value {
                    None => out.push(0),
                    Some(counter) => {
                        out.push(1);
                        out.extend_from_slice(&counter.to_le_bytes());
                    }
                }
            }
            Self::BoundedUpdated { value, version } => {
                out.push(TAG_RES_BOUNDED_UPDATED);
                out.extend_from_slice(&value.to_le_bytes());
                version.encode(out);
            }
            Self::BoundedValue {
                value,
                capacity,
                share,
            } => {
                out.push(TAG_RES_BOUNDED_VALUE);
                match value {
                    None => out.push(0),
                    Some(value) => {
                        out.push(1);
                        out.extend_from_slice(&value.to_le_bytes());
                    }
                }
                match capacity {
                    None => out.push(0),
                    Some(capacity) => {
                        out.push(1);
                        capacity.encode(out);
                    }
                }
                match share {
                    None => out.push(0),
                    Some((min, max)) => {
                        out.push(1);
                        out.extend_from_slice(&min.to_le_bytes());
                        out.extend_from_slice(&max.to_le_bytes());
                    }
                }
            }
            Self::SemaphoreAcquired => out.push(TAG_RES_SEMAPHORE_ACQUIRED),
            Self::SemaphoreReleased { released } => {
                out.push(TAG_RES_SEMAPHORE_RELEASED);
                released.encode(out);
            }
            Self::SemaphoreLoad {
                outstanding,
                capacity,
            } => {
                out.push(TAG_RES_SEMAPHORE_LOAD);
                outstanding.encode(out);
                capacity.encode(out);
            }
            Self::LeaseAcquired {
                fencing,
                expires_at,
            } => {
                out.push(TAG_RES_LEASE_ACQUIRED);
                fencing.encode(out);
                expires_at.encode(out);
            }
            Self::LeaseRenewed {
                fencing,
                expires_at,
            } => {
                out.push(TAG_RES_LEASE_RENEWED);
                fencing.encode(out);
                expires_at.encode(out);
            }
            Self::LeaseReleased { released } => {
                out.push(TAG_RES_LEASE_RELEASED);
                released.encode(out);
            }
            Self::LeaseInfo {
                holder,
                next_fencing,
            } => {
                out.push(TAG_RES_LEASE_INFO);
                match holder {
                    None => out.push(0),
                    Some(holder) => {
                        out.push(1);
                        holder.encode(out);
                    }
                }
                next_fencing.encode(out);
            }
            Self::StreamCreated => out.push(TAG_RES_STREAM_CREATED),
            Self::StreamAppended { offset } => {
                out.push(TAG_RES_STREAM_APPENDED);
                offset.encode(out);
            }
            Self::StreamEntries {
                entries,
                next_offset,
            } => {
                out.push(TAG_RES_STREAM_ENTRIES);
                let count = u32::try_from(entries.len()).unwrap_or(u32::MAX);
                count.encode(out);
                for entry in entries {
                    entry.encode(out);
                }
                next_offset.encode(out);
            }
            Self::StreamTrimmed { removed } => {
                out.push(TAG_RES_STREAM_TRIMMED);
                removed.encode(out);
            }
            Self::TxnLocalCommitted { versions } => {
                out.push(TAG_RES_TXN_LOCAL_COMMITTED);
                let count = u16::try_from(versions.len()).unwrap_or(u16::MAX);
                out.extend_from_slice(&count.to_le_bytes());
                for version in versions {
                    match version {
                        None => out.push(0),
                        Some(version) => {
                            out.push(1);
                            version.encode(out);
                        }
                    }
                }
            }
        }
    }
}

impl Decode for OperationResult {
    #[allow(clippy::too_many_lines)]
    fn decode(input: &[u8]) -> Result<(Self, usize), CodecError> {
        let (tag, first) = u8::decode(input)?;
        match tag {
            TAG_RES_VALUE => {
                let (present, second) = bool::decode(&input[first..])?;
                if !present {
                    return Ok((Self::Value(None), first + second));
                }
                let (bytes, third) = decode_byte_vec(&input[first + second..])?;
                Ok((
                    Self::Value(Some(Bytes::from(bytes))),
                    first + second + third,
                ))
            }
            TAG_RES_STORED => {
                let (version, second) = ObjectVersion::decode(&input[first..])?;
                Ok((Self::Stored { version }, first + second))
            }
            TAG_RES_DELETED => {
                let (existed, second) = bool::decode(&input[first..])?;
                Ok((Self::Deleted { existed }, first + second))
            }
            TAG_RES_EXISTS => {
                let (present, second) = bool::decode(&input[first..])?;
                Ok((Self::Exists(present), first + second))
            }
            TAG_RES_COUNTER => {
                let (present, second) = bool::decode(&input[first..])?;
                if !present {
                    return Ok((Self::Counter(None), first + second));
                }
                let (raw, third) = <[u8; 8]>::decode(&input[first + second..])?;
                Ok((
                    Self::Counter(Some(i64::from_le_bytes(raw))),
                    first + second + third,
                ))
            }
            TAG_RES_COUNTER_UPDATED => {
                let (raw_value, second) = <[u8; 8]>::decode(&input[first..])?;
                let (version, third) = ObjectVersion::decode(&input[first + second..])?;
                Ok((
                    Self::CounterUpdated {
                        value: i64::from_le_bytes(raw_value),
                        version,
                    },
                    first + second + third,
                ))
            }
            TAG_RES_EXPIRY_SET => {
                let (applied, second) = bool::decode(&input[first..])?;
                Ok((Self::ExpirySet { applied }, first + second))
            }
            TAG_RES_EXPIRY_PERSISTED => {
                let (removed, second) = bool::decode(&input[first..])?;
                Ok((Self::ExpiryPersisted { removed }, first + second))
            }
            TAG_RES_EXPIRY => {
                let (present, second) = bool::decode(&input[first..])?;
                if !present {
                    return Ok((Self::Expiry(None), first + second));
                }
                let (expiry, third) = Expiry::decode(&input[first + second..])?;
                Ok((Self::Expiry(Some(expiry)), first + second + third))
            }
            TAG_RES_CHUNKED => {
                let (raw, second) = <[u8; 32]>::decode(&input[first..])?;
                let (logical_len, third) = u64::decode(&input[first + second..])?;
                Ok((
                    Self::ChunkedValue {
                        manifest: ManifestId::from_bytes(raw),
                        logical_len,
                    },
                    first + second + third,
                ))
            }
            TAG_RES_FABRIC => {
                let (fabric_id, second) = u64::decode(&input[first..])?;
                let (logical_len, third) = u64::decode(&input[first + second..])?;
                let (version, fourth) = u64::decode(&input[first + second + third..])?;
                Ok((
                    Self::FabricValue {
                        fabric_id,
                        logical_len,
                        version,
                    },
                    first + second + third + fourth,
                ))
            }
            TAG_RES_LENGTH => {
                let (present, second) = bool::decode(&input[first..])?;
                if !present {
                    return Ok((Self::Length(None), first + second));
                }
                let (len, third) = u64::decode(&input[first + second..])?;
                Ok((Self::Length(Some(len)), first + second + third))
            }
            TAG_RES_VERSION => {
                let (present, second) = bool::decode(&input[first..])?;
                if !present {
                    return Ok((Self::Version(None), first + second));
                }
                let (version, third) = ObjectVersion::decode(&input[first + second..])?;
                Ok((Self::Version(Some(version)), first + second + third))
            }
            TAG_RES_CONDITIONAL_SET => {
                let (applied, second) = bool::decode(&input[first..])?;
                let (present, third) = bool::decode(&input[first + second..])?;
                if !present {
                    return Ok((
                        Self::ConditionalSet {
                            applied,
                            version: None,
                        },
                        first + second + third,
                    ));
                }
                let (version, fourth) = ObjectVersion::decode(&input[first + second + third..])?;
                Ok((
                    Self::ConditionalSet {
                        applied,
                        version: Some(version),
                    },
                    first + second + third + fourth,
                ))
            }
            TAG_RES_TXN_PREPARED => Ok((Self::TxnPrepared, first)),
            TAG_RES_TXN_CONFLICT => Ok((Self::TxnConflict, first)),
            TAG_RES_TXN_FINALIZED => {
                let (applied, second) = bool::decode(&input[first..])?;
                let (present, third) = bool::decode(&input[first + second..])?;
                if !present {
                    return Ok((
                        Self::TxnFinalized {
                            applied,
                            version: None,
                        },
                        first + second + third,
                    ));
                }
                let (version, fourth) = ObjectVersion::decode(&input[first + second + third..])?;
                Ok((
                    Self::TxnFinalized {
                        applied,
                        version: Some(version),
                    },
                    first + second + third + fourth,
                ))
            }
            TAG_RES_COMMUTATIVE_APPLIED => Ok((Self::CommutativeApplied, first)),
            TAG_RES_COMMUTATIVE_VALUE => {
                let (present, second) = bool::decode(&input[first..])?;
                if !present {
                    return Ok((Self::CommutativeValue(None), first + second));
                }
                let (raw, third) = <[u8; 8]>::decode(&input[first + second..])?;
                Ok((
                    Self::CommutativeValue(Some(i64::from_le_bytes(raw))),
                    first + second + third,
                ))
            }
            TAG_RES_BOUNDED_UPDATED => {
                let (raw_value, second) = <[u8; 8]>::decode(&input[first..])?;
                let (version, third) = ObjectVersion::decode(&input[first + second..])?;
                Ok((
                    Self::BoundedUpdated {
                        value: i64::from_le_bytes(raw_value),
                        version,
                    },
                    first + second + third,
                ))
            }
            TAG_RES_BOUNDED_VALUE => {
                let (value_present, second) = bool::decode(&input[first..])?;
                let (value, third) = if value_present {
                    let (raw, used) = <[u8; 8]>::decode(&input[first + second..])?;
                    (Some(i64::from_le_bytes(raw)), used)
                } else {
                    (None, 0)
                };
                let (cap_present, fourth) = bool::decode(&input[first + second + third..])?;
                let (capacity, fifth) = if cap_present {
                    let (cap, used) = u64::decode(&input[first + second + third + fourth..])?;
                    (Some(cap), used)
                } else {
                    (None, 0)
                };
                let (share_present, sixth) =
                    bool::decode(&input[first + second + third + fourth + fifth..])?;
                let (share, seventh) = if share_present {
                    let (min_raw, used_a) = <[u8; 8]>::decode(
                        &input[first + second + third + fourth + fifth + sixth..],
                    )?;
                    let (max_raw, used_b) = <[u8; 8]>::decode(
                        &input[first + second + third + fourth + fifth + sixth + used_a..],
                    )?;
                    (
                        Some((i64::from_le_bytes(min_raw), i64::from_le_bytes(max_raw))),
                        used_a + used_b,
                    )
                } else {
                    (None, 0)
                };
                Ok((
                    Self::BoundedValue {
                        value,
                        capacity,
                        share,
                    },
                    first + second + third + fourth + fifth + sixth + seventh,
                ))
            }
            TAG_RES_SEMAPHORE_ACQUIRED => Ok((Self::SemaphoreAcquired, first)),
            TAG_RES_SEMAPHORE_RELEASED => {
                let (released, second) = bool::decode(&input[first..])?;
                Ok((Self::SemaphoreReleased { released }, first + second))
            }
            TAG_RES_SEMAPHORE_LOAD => {
                let (outstanding, second) = u64::decode(&input[first..])?;
                let (capacity, third) = u64::decode(&input[first + second..])?;
                Ok((
                    Self::SemaphoreLoad {
                        outstanding,
                        capacity,
                    },
                    first + second + third,
                ))
            }
            TAG_RES_LEASE_ACQUIRED => {
                let (fencing, second) = crate::FencingToken::decode(&input[first..])?;
                let (expires_at, third) = WallTimestamp::decode(&input[first + second..])?;
                Ok((
                    Self::LeaseAcquired {
                        fencing,
                        expires_at,
                    },
                    first + second + third,
                ))
            }
            TAG_RES_LEASE_RENEWED => {
                let (fencing, second) = crate::FencingToken::decode(&input[first..])?;
                let (expires_at, third) = WallTimestamp::decode(&input[first + second..])?;
                Ok((
                    Self::LeaseRenewed {
                        fencing,
                        expires_at,
                    },
                    first + second + third,
                ))
            }
            TAG_RES_LEASE_RELEASED => {
                let (released, second) = bool::decode(&input[first..])?;
                Ok((Self::LeaseReleased { released }, first + second))
            }
            TAG_RES_LEASE_INFO => {
                let (present, second) = bool::decode(&input[first..])?;
                let (holder, third) = if present {
                    let (holder, used) = crate::LeaseHolder::decode(&input[first + second..])?;
                    (Some(holder), used)
                } else {
                    (None, 0)
                };
                let (next_fencing, fourth) =
                    crate::FencingToken::decode(&input[first + second + third..])?;
                Ok((
                    Self::LeaseInfo {
                        holder,
                        next_fencing,
                    },
                    first + second + third + fourth,
                ))
            }
            TAG_RES_STREAM_CREATED => Ok((Self::StreamCreated, first)),
            TAG_RES_STREAM_APPENDED => {
                let (offset, second) = u64::decode(&input[first..])?;
                Ok((Self::StreamAppended { offset }, first + second))
            }
            TAG_RES_STREAM_ENTRIES => {
                let (count, mut at) = u32::decode(&input[first..])?;
                at += first;
                let count = usize::try_from(count).map_err(|_| CodecError::InvalidTag {
                    kind: "operation-result",
                    tag: TAG_RES_STREAM_ENTRIES,
                })?;
                if count > crate::MAX_STREAM_SHARD_ENTRIES + 1 {
                    return Err(CodecError::InvalidTag {
                        kind: "operation-result",
                        tag: TAG_RES_STREAM_ENTRIES,
                    });
                }
                let mut entries = Vec::with_capacity(count);
                for _ in 0..count {
                    let (entry, used) = crate::StreamEntry::decode(&input[at..])?;
                    at += used;
                    entries.push(entry);
                }
                let (next_offset, used) = u64::decode(&input[at..])?;
                at += used;
                Ok((
                    Self::StreamEntries {
                        entries,
                        next_offset,
                    },
                    at,
                ))
            }
            TAG_RES_STREAM_TRIMMED => {
                let (removed, second) = u64::decode(&input[first..])?;
                Ok((Self::StreamTrimmed { removed }, first + second))
            }
            TAG_RES_TXN_LOCAL_COMMITTED => {
                let (count_raw, mut at) = <[u8; 2]>::decode(&input[first..])?;
                at += first;
                let count = usize::from(u16::from_le_bytes(count_raw));
                if count > crate::txn::MAX_TXN_KEYS {
                    return Err(CodecError::InvalidTag {
                        kind: "operation-result",
                        tag: TAG_RES_TXN_LOCAL_COMMITTED,
                    });
                }
                let mut versions = Vec::with_capacity(count.min(16));
                for _ in 0..count {
                    let (present, used) = bool::decode(&input[at..])?;
                    at += used;
                    if !present {
                        versions.push(None);
                        continue;
                    }
                    let (version, used) = ObjectVersion::decode(&input[at..])?;
                    at += used;
                    versions.push(Some(version));
                }
                Ok((Self::TxnLocalCommitted { versions }, at))
            }
            other => Err(CodecError::InvalidTag {
                kind: "operation-result",
                tag: other,
            }),
        }
    }
}

/// Terminal outcome of one durable mutating request: either the completed
/// result or the terminal rejection that must be replayed to a retrying
/// client instead of re-executing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DurableOutcome {
    /// The request completed with this result.
    Completed(OperationResult),
    /// The request was terminally rejected without mutating.
    Rejected(OpError),
    /// An object version could not advance; nothing mutated.
    VersionExhausted,
}

/// Canonical wire tags. Fixed forever within framing version 1.
const TAG_OUTCOME_COMPLETED: u8 = 1;
const TAG_OUTCOME_REJECTED: u8 = 2;
const TAG_OUTCOME_VERSION_EXHAUSTED: u8 = 3;

impl Encode for DurableOutcome {
    fn encoded_len(&self) -> usize {
        match self {
            Self::Completed(result) => 1 + result.encoded_len(),
            Self::Rejected(error) => 1 + error.encoded_len(),
            Self::VersionExhausted => 1,
        }
    }

    fn encode(&self, out: &mut Vec<u8>) {
        match self {
            Self::Completed(result) => {
                out.push(TAG_OUTCOME_COMPLETED);
                result.encode(out);
            }
            Self::Rejected(error) => {
                out.push(TAG_OUTCOME_REJECTED);
                error.encode(out);
            }
            Self::VersionExhausted => out.push(TAG_OUTCOME_VERSION_EXHAUSTED),
        }
    }
}

impl Decode for DurableOutcome {
    fn decode(input: &[u8]) -> Result<(Self, usize), CodecError> {
        let (tag, first) = u8::decode(input)?;
        match tag {
            TAG_OUTCOME_COMPLETED => {
                let (result, second) = OperationResult::decode(&input[first..])?;
                Ok((Self::Completed(result), first + second))
            }
            TAG_OUTCOME_REJECTED => {
                let (error, second) = OpError::decode(&input[first..])?;
                Ok((Self::Rejected(error), first + second))
            }
            TAG_OUTCOME_VERSION_EXHAUSTED => Ok((Self::VersionExhausted, first)),
            other => Err(CodecError::InvalidTag {
                kind: "durable-outcome",
                tag: other,
            }),
        }
    }
}

/// Maps one applied mutation to its operation result. This is the single
/// canonical `(Mutation, ApplyOutcome)` mapping: the live path and WAL
/// replay verification share it, so any divergence fails loudly instead of
/// forking semantics.
///
/// `is_persist_expiry` disambiguates `SetExpiry` outcomes (`ExpirySet` for
/// `ExpireAt`, `ExpiryPersisted` for `PersistExpiry`); the WAL record and
/// dedup entry always know the originating opcode.
///
/// # Panics
///
/// Panics on impossible mutation/outcome pairings (e.g. a `Set` yielding a
/// counter outcome): preparation and application run back-to-back on the
/// same state, so disagreement is an internal logic bug, never a runtime
/// condition.
#[must_use]
pub fn outcome_for(
    mutation: &Mutation,
    outcome: &ApplyOutcome,
    is_persist_expiry: bool,
) -> OperationResult {
    if let Some(result) = outcome_for_semantic(mutation, outcome) {
        return result;
    }
    match (mutation, outcome) {
        // Inline, chunked, fabric, and spliced stores answer identically:
        // the result names the new version either way, never the
        // representation.
        (
            Mutation::PutBytes { .. }
            | Mutation::ReplaceChunkedRoot { .. }
            | Mutation::ReplaceFabricRoot { .. }
            | Mutation::SpliceBytes { .. },
            ApplyOutcome::Put { version },
        ) => OperationResult::Stored { version: *version },
        // Conditional stores answer with their applied flag: the mutation
        // only exists when the condition held, so reaching here means
        // applied with the new version.
        (
            Mutation::PutBytesWithExpiry { .. }
            | Mutation::ReplaceChunkedRootWithExpiry { .. }
            | Mutation::ReplaceFabricRootWithExpiry { .. },
            ApplyOutcome::Put { version },
        ) => OperationResult::ConditionalSet {
            applied: true,
            version: Some(*version),
        },
        (Mutation::Delete { .. }, ApplyOutcome::Deleted { existed }) => {
            OperationResult::Deleted { existed: *existed }
        }
        (Mutation::CounterAdd { .. }, ApplyOutcome::Counter { value, version }) => {
            OperationResult::CounterUpdated {
                value: *value,
                version: *version,
            }
        }
        (Mutation::SetExpiry { .. }, ApplyOutcome::Expiry { applied, .. }) => {
            if is_persist_expiry {
                OperationResult::ExpiryPersisted { removed: *applied }
            } else {
                OperationResult::ExpirySet { applied: *applied }
            }
        }
        (Mutation::TxnPrepare { .. }, ApplyOutcome::TxnPrepared) => OperationResult::TxnPrepared,
        (Mutation::TxnPrepare { .. }, ApplyOutcome::TxnConflict) => OperationResult::TxnConflict,
        (Mutation::TxnFinalize { .. }, ApplyOutcome::TxnFinalized { applied, version }) => {
            OperationResult::TxnFinalized {
                applied: *applied,
                version: *version,
            }
        }
        // A commit-finalize applies its prepared write inline: the inner
        // outcome surfaces with the finalize shape (applied + version).
        // `apply` maps these before returning, so these arms only serve
        // callers that pair outcomes manually (tests, replay tools).
        (Mutation::TxnFinalize { .. }, ApplyOutcome::Put { version })
        | (Mutation::TxnFinalize { .. }, ApplyOutcome::Counter { version, .. }) => {
            OperationResult::TxnFinalized {
                applied: true,
                version: Some(*version),
            }
        }
        (Mutation::TxnFinalize { .. }, ApplyOutcome::Deleted { .. }) => {
            OperationResult::TxnFinalized {
                applied: true,
                version: None,
            }
        }
        (mutation, outcome) => {
            panic!("prepare/apply contract violated: {mutation:?} -> {outcome:?}")
        }
    }
}

/// Maps one applied semantic mutation to its operation result (`None` when
/// the mutation is not semantic: the caller falls back to the legacy
/// mapping). Split from [`outcome_for`] so each half stays reviewable.
fn outcome_for_semantic(mutation: &Mutation, outcome: &ApplyOutcome) -> Option<OperationResult> {
    let result = match (mutation, outcome) {
        // Commutative additions converge to `Applied`: the post-apply sum
        // verifies durability prediction but never reaches the client, so
        // no history can distinguish operation order.
        (Mutation::CommutativeAdd { .. }, ApplyOutcome::Commutative { .. }) => {
            OperationResult::CommutativeApplied
        }
        // Bounded creates, share moves, and semaphore creates answer like
        // ordinary stores (new version either way, never the rights); adds
        // answer with values.
        (
            Mutation::BoundedCreate { .. }
            | Mutation::EscrowSetShare { .. }
            | Mutation::SemaphoreCreate { .. },
            ApplyOutcome::Put { version },
        ) => OperationResult::Stored { version: *version },
        (Mutation::BoundedAdd { .. }, ApplyOutcome::Bounded { value, version }) => {
            OperationResult::BoundedUpdated {
                value: *value,
                version: *version,
            }
        }
        // Semaphore acquires/releases answer id.
        (Mutation::SemaphoreAcquire { .. }, ApplyOutcome::SemaphoreAcquired) => {
            OperationResult::SemaphoreAcquired
        }
        (Mutation::SemaphoreRelease { .. }, ApplyOutcome::SemaphoreReleased { released }) => {
            OperationResult::SemaphoreReleased {
                released: *released,
            }
        }
        // Lease grants answer with the authoritative token; releases with
        // whether a live holding was freed.
        (
            Mutation::LeaseAcquire { .. },
            ApplyOutcome::LeaseGranted {
                fencing,
                expires_at,
            },
        ) => OperationResult::LeaseAcquired {
            fencing: *fencing,
            expires_at: *expires_at,
        },
        (
            Mutation::LeaseRenew { .. },
            ApplyOutcome::LeaseGranted {
                fencing,
                expires_at,
            },
        ) => OperationResult::LeaseRenewed {
            fencing: *fencing,
            expires_at: *expires_at,
        },
        (Mutation::LeaseRelease { .. }, ApplyOutcome::LeaseReleased { released }) => {
            OperationResult::LeaseReleased {
                released: *released,
            }
        }
        // Stream creates answer created; appends answer with the assigned
        // shard-local offset; trims answer with the dropped count.
        (Mutation::StreamCreate { .. }, ApplyOutcome::Put { .. }) => OperationResult::StreamCreated,
        (Mutation::StreamAppend { .. }, ApplyOutcome::StreamAppended { offset }) => {
            OperationResult::StreamAppended { offset: *offset }
        }
        (Mutation::StreamTrim { .. }, ApplyOutcome::StreamTrimmed { removed }) => {
            OperationResult::StreamTrimmed { removed: *removed }
        }
        (Mutation::TxnCommitLocal { .. }, ApplyOutcome::TxnLocalCommitted { versions }) => {
            OperationResult::TxnLocalCommitted {
                versions: versions.clone(),
            }
        }
        _ => return None,
    };
    Some(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use kivi_codec::{Decode, Encode};
    use kivi_types::WallTimestamp;

    fn round_trip<T>(value: &T) -> T
    where
        T: Encode + Decode + PartialEq + core::fmt::Debug,
    {
        let mut bytes = Vec::new();
        value.encode(&mut bytes);
        assert_eq!(bytes.len(), value.encoded_len(), "length prefix honest");
        let (back, consumed) = T::decode(&bytes).expect("decode");
        assert_eq!(consumed, bytes.len(), "exact consumption");
        assert_eq!(&back, value);
        back
    }

    #[test]
    fn operation_results_round_trip_every_variant() {
        let version = ObjectVersion::from_u64(7);
        round_trip(&OperationResult::Value(None));
        round_trip(&OperationResult::Value(Some(Bytes::from_static(b"v"))));
        round_trip(&OperationResult::ChunkedValue {
            manifest: ManifestId::from_bytes([0x22; 32]),
            logical_len: 3_000_000,
        });
        round_trip(&OperationResult::Stored { version });
        round_trip(&OperationResult::Deleted { existed: true });
        round_trip(&OperationResult::Exists(false));
        round_trip(&OperationResult::Counter(None));
        round_trip(&OperationResult::Counter(Some(-41)));
        round_trip(&OperationResult::CounterUpdated { value: 9, version });
        round_trip(&OperationResult::ExpirySet { applied: true });
        round_trip(&OperationResult::ExpiryPersisted { removed: false });
        round_trip(&OperationResult::Expiry(None));
        round_trip(&OperationResult::Expiry(Some(Expiry::at(
            WallTimestamp::from_micros(99),
        ))));
        round_trip(&OperationResult::Expiry(Some(Expiry::NEVER)));
        round_trip(&OperationResult::Length(None));
        round_trip(&OperationResult::Length(Some(41)));
        round_trip(&OperationResult::Version(None));
        round_trip(&OperationResult::Version(Some(version)));
        round_trip(&OperationResult::ConditionalSet {
            applied: false,
            version: None,
        });
        round_trip(&OperationResult::ConditionalSet {
            applied: true,
            version: Some(version),
        });
        round_trip(&OperationResult::TxnPrepared);
        round_trip(&OperationResult::TxnConflict);
        round_trip(&OperationResult::TxnFinalized {
            applied: true,
            version: Some(version),
        });
        round_trip(&OperationResult::TxnFinalized {
            applied: false,
            version: None,
        });
        round_trip(&OperationResult::TxnLocalCommitted {
            versions: vec![Some(version), None],
        });
        round_trip(&OperationResult::TxnLocalCommitted { versions: vec![] });
        round_trip(&OperationResult::CommutativeApplied);
        round_trip(&OperationResult::CommutativeValue(None));
        round_trip(&OperationResult::CommutativeValue(Some(-7)));
        round_trip(&OperationResult::BoundedUpdated { value: 3, version });
        round_trip(&OperationResult::BoundedValue {
            value: None,
            capacity: None,
            share: None,
        });
        round_trip(&OperationResult::BoundedValue {
            value: Some(3),
            capacity: Some(10),
            share: Some((0, 10)),
        });
        round_trip(&OperationResult::SemaphoreAcquired);
        round_trip(&OperationResult::SemaphoreReleased { released: true });
        round_trip(&OperationResult::SemaphoreReleased { released: false });
        round_trip(&OperationResult::SemaphoreLoad {
            outstanding: 2,
            capacity: 5,
        });
        round_trip(&OperationResult::LeaseAcquired {
            fencing: crate::FencingToken::from_u64(4),
            expires_at: WallTimestamp::from_micros(99),
        });
        round_trip(&OperationResult::LeaseRenewed {
            fencing: crate::FencingToken::from_u64(4),
            expires_at: WallTimestamp::from_micros(199),
        });
        round_trip(&OperationResult::LeaseReleased { released: true });
        round_trip(&OperationResult::LeaseInfo {
            holder: None,
            next_fencing: crate::FencingToken::from_u64(2),
        });
        round_trip(&OperationResult::LeaseInfo {
            holder: Some(crate::LeaseHolder {
                owner: 11,
                fencing: crate::FencingToken::from_u64(3),
                expires_at: WallTimestamp::from_micros(77),
            }),
            next_fencing: crate::FencingToken::from_u64(4),
        });
        round_trip(&OperationResult::StreamCreated);
        round_trip(&OperationResult::StreamAppended { offset: 41 });
        round_trip(&OperationResult::StreamEntries {
            entries: vec![crate::StreamEntry {
                offset: 0,
                partition: vec![0x70],
                payload: Bytes::from_static(b"e0"),
            }],
            next_offset: 1,
        });
        round_trip(&OperationResult::StreamTrimmed { removed: 9 });
    }

    #[test]
    fn op_errors_and_durable_outcomes_round_trip() {
        round_trip(&OpError::WrongType {
            expected: ObjectType::Bytes,
            found: ObjectType::StrictCounter,
        });
        round_trip(&OpError::CounterOverflow);
        round_trip(&OpError::TxnConflict);
        round_trip(&OpError::BoundedExceeded);
        round_trip(&OpError::SemaphoreExhausted);
        round_trip(&OpError::LeaseConflict);
        round_trip(&OpError::StaleFencing);
        round_trip(&OpError::StreamFull);
        round_trip(&OpError::NotFound);
        round_trip(&DurableOutcome::Completed(OperationResult::Stored {
            version: ObjectVersion::FIRST,
        }));
        round_trip(&DurableOutcome::Rejected(OpError::CounterOverflow));
        round_trip(&DurableOutcome::VersionExhausted);
    }

    #[test]
    fn stored_wire_bytes_are_stable() {
        // Golden bytes: tag 2 then version 7 LE. Pin the layout deliberately.
        let mut bytes = Vec::new();
        OperationResult::Stored {
            version: ObjectVersion::from_u64(7),
        }
        .encode(&mut bytes);
        assert_eq!(bytes, vec![2, 7, 0, 0, 0, 0, 0, 0, 0]);
    }

    #[test]
    fn unknown_result_tags_rejected() {
        assert!(OperationResult::decode(&[0xFF]).is_err());
        assert!(OpError::decode(&[0xFF]).is_err());
        assert!(DurableOutcome::decode(&[0xFF]).is_err());
    }

    #[test]
    fn outcome_for_covers_the_matrix() {
        let version = ObjectVersion::from_u64(3);
        let key = Key::from("k");
        assert_eq!(
            outcome_for(
                &Mutation::PutBytes {
                    key: key.clone(),
                    value: Bytes::from_static(b"v"),
                },
                &ApplyOutcome::Put { version },
                false,
            ),
            OperationResult::Stored { version }
        );
        assert_eq!(
            outcome_for(
                &Mutation::SetExpiry {
                    key: key.clone(),
                    expiry: Expiry::NEVER,
                },
                &ApplyOutcome::Expiry {
                    applied: true,
                    version: Some(version),
                },
                true,
            ),
            OperationResult::ExpiryPersisted { removed: true }
        );
        assert_eq!(
            outcome_for(
                &Mutation::SetExpiry {
                    key: key.clone(),
                    expiry: Expiry::at(WallTimestamp::from_micros(5)),
                },
                &ApplyOutcome::Expiry {
                    applied: true,
                    version: Some(version),
                },
                false,
            ),
            OperationResult::ExpirySet { applied: true }
        );
        assert_eq!(
            outcome_for(
                &Mutation::PutBytesWithExpiry {
                    key: key.clone(),
                    value: Bytes::from_static(b"v"),
                    expiry: Expiry::NEVER,
                },
                &ApplyOutcome::Put { version },
                false,
            ),
            OperationResult::ConditionalSet {
                applied: true,
                version: Some(version),
            }
        );
        assert_eq!(
            outcome_for(
                &Mutation::ReplaceChunkedRootWithExpiry {
                    key,
                    manifest: ManifestId::from_bytes([0x11; 32]),
                    logical_len: 9,
                    expiry: Expiry::NEVER,
                },
                &ApplyOutcome::Put { version },
                false,
            ),
            OperationResult::ConditionalSet {
                applied: true,
                version: Some(version),
            }
        );
    }

    #[test]
    fn conditional_sets_evaluate_atomically_with_policies() {
        use crate::store::ObjectStore;
        let now = WallTimestamp::from_micros(1_000_000);
        let mut store = ObjectStore::new();
        let key = Key::from("k");
        // IfAbsent stores on missing.
        match store
            .prepare(
                &Operation::SetConditional {
                    key: key.clone(),
                    value: Bytes::from_static(b"v1"),
                    condition: SetCondition::IfAbsent,
                    expiry: ExpiryPolicy::Clear,
                },
                now,
            )
            .expect("prepare")
        {
            crate::store::Prepared::Write(mutation) => {
                store.apply(&mutation, now).expect("apply");
            }
            crate::store::Prepared::Read(_) => panic!("IfAbsent on missing must write"),
        }
        // IfAbsent refuses when present without mutating the version.
        match store
            .prepare(
                &Operation::SetConditional {
                    key: key.clone(),
                    value: Bytes::from_static(b"v2"),
                    condition: SetCondition::IfAbsent,
                    expiry: ExpiryPolicy::Clear,
                },
                now,
            )
            .expect("prepare")
        {
            crate::store::Prepared::Read(OperationResult::ConditionalSet { applied, .. }) => {
                assert!(!applied);
            }
            _ => panic!("IfAbsent on present must not apply"),
        }
        // IfPresent stores and Keep preserves a dated expiry.
        let deadline = WallTimestamp::from_micros(9_000_000);
        match store
            .prepare(
                &Operation::ExpireAt {
                    key: key.clone(),
                    expires_at: deadline,
                },
                now,
            )
            .expect("prepare")
        {
            crate::store::Prepared::Write(mutation) => {
                store.apply(&mutation, now).expect("apply");
            }
            crate::store::Prepared::Read(_) => panic!("expire must write"),
        }
        match store
            .prepare(
                &Operation::SetConditional {
                    key: key.clone(),
                    value: Bytes::from_static(b"v3"),
                    condition: SetCondition::IfPresent,
                    expiry: ExpiryPolicy::Keep,
                },
                now,
            )
            .expect("prepare")
        {
            crate::store::Prepared::Write(mutation) => {
                store.apply(&mutation, now).expect("apply");
            }
            crate::store::Prepared::Read(_) => panic!("IfPresent on present must write"),
        }
        match store
            .prepare(&Operation::GetExpiry { key: key.clone() }, now)
            .expect("prepare")
        {
            crate::store::Prepared::Read(OperationResult::Expiry(Some(expiry))) => {
                assert_eq!(expiry, Expiry::at(deadline));
            }
            _ => panic!("Keep must preserve the deadline"),
        }
    }

    #[test]
    fn ranges_slice_and_measure_without_payload_reads() {
        use crate::store::ObjectStore;
        let now = WallTimestamp::from_micros(1_000_000);
        let mut store = ObjectStore::new();
        let key = Key::from("k");
        // Missing reads answer absent.
        match store
            .prepare(
                &Operation::GetRange {
                    key: key.clone(),
                    offset: 0,
                    len: 4,
                },
                now,
            )
            .expect("prepare")
        {
            crate::store::Prepared::Read(OperationResult::Value(None)) => {}
            _ => panic!("missing range must be None"),
        }
        match store
            .prepare(&Operation::BytesLength { key: key.clone() }, now)
            .expect("prepare")
        {
            crate::store::Prepared::Read(OperationResult::Length(None)) => {}
            _ => panic!("missing length must be None"),
        }
        match store
            .prepare(
                &Operation::Set {
                    key: key.clone(),
                    value: Bytes::from_static(b"hello"),
                },
                now,
            )
            .expect("prepare")
        {
            crate::store::Prepared::Write(mutation) => {
                store.apply(&mutation, now).expect("apply");
            }
            crate::store::Prepared::Read(_) => panic!("set must write"),
        }
        match store
            .prepare(
                &Operation::GetRange {
                    key: key.clone(),
                    offset: 1,
                    len: 3,
                },
                now,
            )
            .expect("prepare")
        {
            crate::store::Prepared::Read(OperationResult::Value(Some(bytes))) => {
                assert_eq!(&bytes[..], b"ell");
            }
            _ => panic!("range must slice"),
        }
        match store
            .prepare(
                &Operation::GetRange {
                    key: key.clone(),
                    offset: 99,
                    len: 4,
                },
                now,
            )
            .expect("prepare")
        {
            crate::store::Prepared::Read(OperationResult::Value(Some(bytes))) => {
                assert!(bytes.is_empty());
            }
            _ => panic!("past-the-end range must be empty"),
        }
        match store
            .prepare(&Operation::BytesLength { key }, now)
            .expect("prepare")
        {
            crate::store::Prepared::Read(OperationResult::Length(Some(len))) => {
                assert_eq!(len, 5);
            }
            _ => panic!("length must measure"),
        }
    }
}
