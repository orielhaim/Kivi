//! Logical objects: keys, versions, types, values, and metadata.
//!
//! A [`StoredObject`] is a small authoritative root: a logical value, a
//! monotonically advanced [`ObjectVersion`], an [`Expiry`], and a physical
//! representation tag. The representation is private and currently always
//! inline; future representations change storage
//! mechanics, never the semantics defined here.

use core::fmt;

use bytes::Bytes;
use kivi_codec::{CodecError, Decode, Encode, decode_byte_vec, encode_bytes};
use kivi_types::{Expiry, ManifestId};

/// Object key: an immutable shared byte string.
///
/// Backed by [`Bytes`] so clones are cheap and payloads can be shared with
/// the network layer later without copying. This is an in-memory handle,
/// not a durable format — future persistent layouts define their own key
/// encoding. Lookups construct a `Key` (one small copy for borrowed input);
/// a `Borrow<[u8]>` impl is deliberately absent so map hashing can never
/// silently diverge between owned and borrowed forms.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Key(Bytes);

impl Key {
    /// Wraps bytes as a key.
    #[must_use]
    pub fn new(bytes: impl Into<Bytes>) -> Self {
        Self(bytes.into())
    }

    /// Reserved system-key prefix: keys starting with `0xFF` hold Kivi
    /// internals (transaction records, index projections) beside user data
    /// in the same tablet. Scans skip them; user writes to them are rejected
    /// at the server boundary (`FoundationDB` reserves `\xff` the same way).
    pub const SYSTEM_PREFIX: u8 = 0xFF;

    /// Whether this key names reserved system state rather than user data.
    #[must_use]
    pub fn is_system(&self) -> bool {
        self.0
            .first()
            .is_some_and(|byte| *byte == Self::SYSTEM_PREFIX)
    }

    /// Returns the key bytes.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        &self.0
    }

    /// Returns the key length in bytes.
    #[must_use]
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Whether the key is empty (allowed, like Redis).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl From<Vec<u8>> for Key {
    /// Wraps owned bytes without copying.
    fn from(bytes: Vec<u8>) -> Self {
        Self(Bytes::from(bytes))
    }
}

impl From<&[u8]> for Key {
    /// Copies borrowed bytes into a shared handle.
    fn from(bytes: &[u8]) -> Self {
        Self(Bytes::copy_from_slice(bytes))
    }
}

impl From<Bytes> for Key {
    /// Wraps a shared handle without copying.
    fn from(bytes: Bytes) -> Self {
        Self(bytes)
    }
}

impl From<&str> for Key {
    /// Copies a string's bytes into a shared handle.
    fn from(text: &str) -> Self {
        Self(Bytes::copy_from_slice(text.as_bytes()))
    }
}

impl From<String> for Key {
    /// Wraps an owned string's bytes without copying.
    fn from(text: String) -> Self {
        Self(Bytes::from(text.into_bytes()))
    }
}

impl fmt::Display for Key {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", String::from_utf8_lossy(&self.0))
    }
}

impl Encode for Key {
    fn encoded_len(&self) -> usize {
        4 + self.len()
    }

    fn encode(&self, out: &mut Vec<u8>) {
        encode_bytes(out, self.as_bytes());
    }
}

impl Decode for Key {
    fn decode(input: &[u8]) -> Result<(Self, usize), CodecError> {
        let (bytes, consumed) = decode_byte_vec(input)?;
        Ok((Self::from(bytes), consumed))
    }
}

/// Logical version of one object: `1` at creation, `+1` per state-changing
/// mutation, checked (never wrapping or saturating).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ObjectVersion(u64);

impl ObjectVersion {
    /// Version assigned at object creation.
    pub const FIRST: Self = Self(1);

    /// Wraps a raw version value.
    #[must_use]
    pub const fn from_u64(value: u64) -> Self {
        Self(value)
    }

    /// Returns the raw value.
    #[must_use]
    pub const fn as_u64(self) -> u64 {
        self.0
    }

    /// Successor version for the next state-changing mutation.
    ///
    /// # Errors
    ///
    /// Returns [`VersionExhausted`] at `u64::MAX` instead of wrapping or
    /// saturating: reusing a version would alias two distinct object states.
    pub const fn next(self) -> Result<Self, VersionExhausted> {
        match self.0.checked_add(1) {
            Some(value) => Ok(Self(value)),
            None => Err(VersionExhausted),
        }
    }
}

impl From<ObjectVersion> for u64 {
    /// Returns the raw version value.
    fn from(version: ObjectVersion) -> Self {
        version.0
    }
}

impl From<u64> for ObjectVersion {
    /// Wraps a raw version value.
    fn from(value: u64) -> Self {
        ObjectVersion(value)
    }
}

impl fmt::Display for ObjectVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "v{}", self.0)
    }
}

impl Encode for ObjectVersion {
    fn encoded_len(&self) -> usize {
        8
    }

    fn encode(&self, out: &mut Vec<u8>) {
        self.0.encode(out);
    }
}

impl Decode for ObjectVersion {
    fn decode(input: &[u8]) -> Result<(Self, usize), CodecError> {
        let (raw, consumed) = u64::decode(input)?;
        Ok((Self::from_u64(raw), consumed))
    }
}

/// Reported when an object's version counter cannot advance past `u64::MAX`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, thiserror::Error)]
#[error("object version space exhausted")]
pub struct VersionExhausted;

/// Logical type of a stored object. Types are strict: operations against the
/// wrong type fail without mutation (native semantics, not Redis command
/// strings).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ObjectType {
    /// Opaque byte string.
    Bytes,
    /// Exact 64-bit counter with ordinal (non-commutative-result) semantics.
    StrictCounter,
    /// Order-free counter: additions commute in both state and observable
    /// result (`Applied`, never an ordinal).
    CommutativeCounter,
    /// Escrow-guarded counter: `escrow_min <= value <= escrow_max` within
    /// `0 <= value <= capacity`, spending only locally owned rights.
    BoundedCounter,
    /// Capacity-guarded permit set: `sum(active) <= capacity`, permits
    /// named by [`PermitId`](crate::PermitId).
    Semaphore,
    /// Fenced ownership record: monotonic [`FencingToken`](crate::FencingToken).
    Lease,
    /// One ordered shard of a [`ShardedStream`](crate::StreamShardState):
    /// total order within the shard, none across shards.
    StreamShard,
}

impl fmt::Display for ObjectType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Bytes => write!(f, "bytes"),
            Self::StrictCounter => write!(f, "strict-counter"),
            Self::CommutativeCounter => write!(f, "commutative-counter"),
            Self::BoundedCounter => write!(f, "bounded-counter"),
            Self::Semaphore => write!(f, "semaphore"),
            Self::Lease => write!(f, "lease"),
            Self::StreamShard => write!(f, "stream-shard"),
        }
    }
}

/// Canonical wire tags. Fixed forever within framing version 1.
const TAG_OBJECT_BYTES: u8 = 1;
const TAG_OBJECT_COUNTER: u8 = 2;
/// Order-free counter tag. New tags never reuse old ones.
const TAG_OBJECT_COMMUTATIVE_COUNTER: u8 = 3;
/// Escrow counter tag. New tags never reuse old ones.
const TAG_OBJECT_BOUNDED_COUNTER: u8 = 4;
/// Semaphore tag. New tags never reuse old ones.
const TAG_OBJECT_SEMAPHORE: u8 = 5;
/// Lease tag. New tags never reuse old ones.
const TAG_OBJECT_LEASE: u8 = 6;
/// Stream-shard tag. New tags never reuse old ones.
const TAG_OBJECT_STREAM_SHARD: u8 = 7;

impl Encode for ObjectType {
    fn encoded_len(&self) -> usize {
        1
    }

    fn encode(&self, out: &mut Vec<u8>) {
        out.push(match self {
            Self::Bytes => TAG_OBJECT_BYTES,
            Self::StrictCounter => TAG_OBJECT_COUNTER,
            Self::CommutativeCounter => TAG_OBJECT_COMMUTATIVE_COUNTER,
            Self::BoundedCounter => TAG_OBJECT_BOUNDED_COUNTER,
            Self::Semaphore => TAG_OBJECT_SEMAPHORE,
            Self::Lease => TAG_OBJECT_LEASE,
            Self::StreamShard => TAG_OBJECT_STREAM_SHARD,
        });
    }
}

impl Decode for ObjectType {
    fn decode(input: &[u8]) -> Result<(Self, usize), CodecError> {
        let (tag, consumed) = u8::decode(input)?;
        match tag {
            TAG_OBJECT_BYTES => Ok((Self::Bytes, consumed)),
            TAG_OBJECT_COUNTER => Ok((Self::StrictCounter, consumed)),
            TAG_OBJECT_COMMUTATIVE_COUNTER => Ok((Self::CommutativeCounter, consumed)),
            TAG_OBJECT_BOUNDED_COUNTER => Ok((Self::BoundedCounter, consumed)),
            TAG_OBJECT_SEMAPHORE => Ok((Self::Semaphore, consumed)),
            TAG_OBJECT_LEASE => Ok((Self::Lease, consumed)),
            TAG_OBJECT_STREAM_SHARD => Ok((Self::StreamShard, consumed)),
            other => Err(CodecError::InvalidTag {
                kind: "object-type",
                tag: other,
            }),
        }
    }
}

/// Logical value of a stored object.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LogicalValue {
    /// Opaque byte string, resident.
    Bytes(Bytes),
    /// Opaque byte string addressed by immutable chunks. Logically identical
    /// to [`Bytes`](Self::Bytes) of the same content: same type, same reads,
    /// same overwrites. Only the physical representation differs, and only
    /// the engine (never this layer) resolves it into bytes.
    Chunked(ChunkedRef),
    /// Opaque byte string managed by the Memory Fabric. Logically identical
    /// to [`Bytes`](Self::Bytes) of the same content: same type, same reads,
    /// same overwrites. The reference names a fabric object (worker-local
    /// physical name, meaningless across nodes, like a chunk manifest id);
    /// arena slots, `NVMe` offsets, and provider placement never appear
    /// here. Only the engine (never this layer) resolves it into bytes,
    /// possibly suspending for asynchronous promotion.
    Fabric(FabricRef),
    /// Exact counter value.
    StrictCounter(i64),
    /// Order-free counter value: the sum of all applied addends. Reads
    /// answer the current sum (a state query, not an ordinal); additions
    /// return `Applied` and commute observably.
    CommutativeCounter(i64),
    /// Escrow-guarded counter with explicit local rights.
    BoundedCounter(BoundedCounterState),
    /// Capacity-guarded permit set.
    Semaphore(SemaphoreState),
    /// Fenced ownership record.
    Lease(LeaseState),
    /// One ordered shard of a sharded stream.
    StreamShard(StreamShardState),
}

impl LogicalValue {
    /// Returns the value's logical type. Chunked and fabric bytes are
    /// bytes: type checks never distinguish representation.
    #[must_use]
    pub const fn object_type(&self) -> ObjectType {
        match self {
            Self::Bytes(_) | Self::Chunked(_) | Self::Fabric(_) => ObjectType::Bytes,
            Self::StrictCounter(_) => ObjectType::StrictCounter,
            Self::CommutativeCounter(_) => ObjectType::CommutativeCounter,
            Self::BoundedCounter(_) => ObjectType::BoundedCounter,
            Self::Semaphore(_) => ObjectType::Semaphore,
            Self::Lease(_) => ObjectType::Lease,
            Self::StreamShard(_) => ObjectType::StreamShard,
        }
    }

    /// Small canonical scan descriptor for semantic values: full state for
    /// fixed-size values (commutative/bounded counters, leases), load
    /// summaries for collections (semaphore capacity + outstanding, stream
    /// cursor + retained count). Returns `None` for bytes/counters, which
    /// project through their dedicated scan shapes instead. Bounded: permit
    /// sets and retained entries never enter scan pages.
    #[must_use]
    pub fn scan_descriptor(&self) -> Option<(ObjectType, Bytes)> {
        match self {
            Self::Bytes(_) | Self::Chunked(_) | Self::Fabric(_) | Self::StrictCounter(_) => None,
            Self::CommutativeCounter(value) => Some((
                ObjectType::CommutativeCounter,
                Bytes::copy_from_slice(&value.to_le_bytes()),
            )),
            Self::BoundedCounter(state) => {
                let mut out = Vec::with_capacity(state.encoded_len());
                state.encode(&mut out);
                Some((ObjectType::BoundedCounter, Bytes::from(out)))
            }
            Self::Semaphore(state) => {
                let mut out = Vec::with_capacity(24);
                state.capacity.encode(&mut out);
                state.outstanding().encode(&mut out);
                let count = u64::try_from(state.permits.len()).unwrap_or(u64::MAX);
                count.encode(&mut out);
                Some((ObjectType::Semaphore, Bytes::from(out)))
            }
            Self::Lease(state) => {
                let mut out = Vec::with_capacity(state.encoded_len());
                state.encode(&mut out);
                Some((ObjectType::Lease, Bytes::from(out)))
            }
            Self::StreamShard(shard) => {
                let mut out = Vec::with_capacity(16 + 4 + 8 + 8);
                out.extend_from_slice(&shard.stream);
                shard.shard.encode(&mut out);
                shard.next_offset.encode(&mut out);
                let retained = u64::try_from(shard.entries.len()).unwrap_or(u64::MAX);
                retained.encode(&mut out);
                Some((ObjectType::StreamShard, Bytes::from(out)))
            }
        }
    }

    /// Estimated logical bytes for telemetry/split policy. Bounded: permit
    /// sets and stream entries count their retained payloads.
    #[must_use]
    pub fn logical_bytes(&self) -> u64 {
        match self {
            Self::Bytes(value) => value.len() as u64,
            Self::Chunked(chunked) => chunked.logical_len,
            Self::Fabric(fabric) => fabric.logical_len,
            Self::StrictCounter(_) | Self::CommutativeCounter(_) => 8,
            Self::BoundedCounter(state) => state.encoded_len() as u64,
            Self::Semaphore(state) => state.encoded_len() as u64,
            Self::Lease(state) => state.encoded_len() as u64,
            Self::StreamShard(shard) => shard.encoded_len() as u64,
        }
    }
}

/// Reference to a chunked byte string: the manifest addressing its immutable
/// chunks plus the total logical length. Small (`Copy`): tablet roots,
/// checkpoints, and WAL records carry this instead of bulk bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ChunkedRef {
    /// Manifest addressing the immutable chunk sequence.
    pub manifest: ManifestId,
    /// Total logical bytes across the manifest's entries.
    pub logical_len: u64,
}

impl Encode for ChunkedRef {
    fn encoded_len(&self) -> usize {
        32 + 8
    }

    fn encode(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(self.manifest.as_bytes());
        self.logical_len.encode(out);
    }
}

impl Decode for ChunkedRef {
    fn decode(input: &[u8]) -> Result<(Self, usize), CodecError> {
        let (raw, first) = <[u8; 32]>::decode(input)?;
        let (logical_len, second) = u64::decode(&input[first..])?;
        Ok((
            Self {
                manifest: ManifestId::from_bytes(raw),
                logical_len,
            },
            first + second,
        ))
    }
}

/// Reference to a Memory Fabric byte string: the fabric-scoped object id
/// plus the total logical length and the logical version the bytes were
/// published at. Small (`Copy`): tablet roots, checkpoints, and WAL
/// records carry this instead of bulk bytes, exactly like [`ChunkedRef`].
///
/// The `id` is a worker-local physical name (like a manifest id without
/// content addressing): it reveals no placement and is meaningless on
/// another node. Topology transfer moves logical bytes, never this id.
/// The `version` pins reads: the engine serves the reference only when it
/// matches the root's logical version, so a delayed promotion can never
/// resurrect an older value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FabricRef {
    /// Fabric-scoped object id assigned at staging.
    pub id: u64,
    /// Total logical bytes of the materialized value.
    pub logical_len: u64,
    /// Logical version the bytes were published at.
    pub version: u64,
}

impl Encode for FabricRef {
    fn encoded_len(&self) -> usize {
        8 + 8 + 8
    }

    fn encode(&self, out: &mut Vec<u8>) {
        self.id.encode(out);
        self.logical_len.encode(out);
        self.version.encode(out);
    }
}

impl Decode for FabricRef {
    fn decode(input: &[u8]) -> Result<(Self, usize), CodecError> {
        let (id, first) = u64::decode(input)?;
        let (logical_len, second) = u64::decode(&input[first..])?;
        let (version, third) = u64::decode(&input[first + second..])?;
        Ok((
            Self {
                id,
                logical_len,
                version,
            },
            first + second + third,
        ))
    }
}

/// Escrow share: the local rights `[min, max]` this tablet may spend without
/// coordinating. `holder` names the owning tablet so a later geography
/// dimension extends this struct without changing the logical API.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EscrowShare {
    /// Tablet owning these rights.
    pub holder: kivi_types::TabletId,
    /// Minimum value spendable locally (inclusive).
    pub min: i64,
    /// Maximum value spendable locally (inclusive).
    pub max: i64,
}

impl Encode for EscrowShare {
    fn encoded_len(&self) -> usize {
        8 + 8 + 8
    }

    fn encode(&self, out: &mut Vec<u8>) {
        self.holder.as_u64().encode(out);
        out.extend_from_slice(&self.min.to_le_bytes());
        out.extend_from_slice(&self.max.to_le_bytes());
    }
}

impl Decode for EscrowShare {
    fn decode(input: &[u8]) -> Result<(Self, usize), CodecError> {
        let (holder, first) = u64::decode(input)?;
        let (min_raw, second) = <[u8; 8]>::decode(&input[first..])?;
        let (max_raw, third) = <[u8; 8]>::decode(&input[first + second..])?;
        Ok((
            Self {
                holder: kivi_types::TabletId::from_u64(holder),
                min: i64::from_le_bytes(min_raw),
                max: i64::from_le_bytes(max_raw),
            },
            first + second + third,
        ))
    }
}

/// Bounded-counter state: `0 <= value <= capacity` globally, with local
/// spending confined to `share.min <= value <= share.max`. A single-holder
/// counter owns `share = [0, capacity]`; rights transfer narrows one
/// holder's share while widening another's, conserving total rights.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BoundedCounterState {
    /// Current logical value.
    pub value: i64,
    /// Global capacity (inclusive upper bound, `<= i64::MAX`).
    pub capacity: u64,
    /// Locally owned spendable interval.
    pub share: EscrowShare,
}

impl BoundedCounterState {
    /// Structural invariant: rights are well-formed and the value is both
    /// globally and locally covered.
    #[must_use]
    pub fn invariant_holds(self) -> bool {
        let Ok(cap) = i64::try_from(self.capacity) else {
            return false;
        };
        0 <= self.share.min
            && self.share.min <= self.value
            && self.value <= self.share.max
            && self.share.max <= cap
    }
}

impl Encode for BoundedCounterState {
    fn encoded_len(&self) -> usize {
        8 + 8 + self.share.encoded_len()
    }

    fn encode(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.value.to_le_bytes());
        self.capacity.encode(out);
        self.share.encode(out);
    }
}

impl Decode for BoundedCounterState {
    fn decode(input: &[u8]) -> Result<(Self, usize), CodecError> {
        let (value_raw, first) = <[u8; 8]>::decode(input)?;
        let (capacity, second) = u64::decode(&input[first..])?;
        let (share, third) = EscrowShare::decode(&input[first + second..])?;
        Ok((
            Self {
                value: i64::from_le_bytes(value_raw),
                capacity,
                share,
            },
            first + second + third,
        ))
    }
}

/// Permit identity: client-minted 128-bit id bound to one acquisition
/// attempt. Retries reuse it (no double-acquire); releases name it (no
/// double-release minting capacity).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PermitId([u8; 16]);

impl PermitId {
    /// Wraps raw id bytes (decode path and client minting).
    #[must_use]
    pub const fn from_bytes(bytes: [u8; 16]) -> Self {
        Self(bytes)
    }

    /// Returns the raw id bytes.
    #[must_use]
    pub const fn as_bytes(self) -> [u8; 16] {
        self.0
    }

    /// Derives an id from session, sequence, and attempt (same inputs →
    /// same id, so transport retries never acquire twice).
    #[must_use]
    pub fn derive(session: u128, seq: u64, attempt: u64) -> Self {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"KIVI-PERMIT-V1");
        hasher.update(&session.to_le_bytes());
        hasher.update(&seq.to_le_bytes());
        hasher.update(&attempt.to_le_bytes());
        let hash = hasher.finalize();
        let mut id = [0u8; 16];
        id.copy_from_slice(&hash.as_bytes()[..16]);
        Self(id)
    }
}

impl Encode for PermitId {
    fn encoded_len(&self) -> usize {
        16
    }

    fn encode(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.0);
    }
}

impl Decode for PermitId {
    fn decode(input: &[u8]) -> Result<(Self, usize), CodecError> {
        let (raw, consumed) = <[u8; 16]>::decode(input)?;
        Ok((Self::from_bytes(raw), consumed))
    }
}

/// One live permit: its owner and held quantity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PermitRecord {
    /// Owner/session identity that acquired it.
    pub owner: u64,
    /// Quantity held by this permit.
    pub qty: u64,
}

impl Encode for PermitRecord {
    fn encoded_len(&self) -> usize {
        8 + 8
    }

    fn encode(&self, out: &mut Vec<u8>) {
        self.owner.encode(out);
        self.qty.encode(out);
    }
}

impl Decode for PermitRecord {
    fn decode(input: &[u8]) -> Result<(Self, usize), CodecError> {
        let (owner, first) = u64::decode(input)?;
        let (qty, second) = u64::decode(&input[first..])?;
        Ok((Self { owner, qty }, first + second))
    }
}

/// Maximum live permits tracked on one semaphore object (bounds the durable
/// root; beyond this the object is full and acquisitions fail fast).
pub const MAX_SEMAPHORE_PERMITS: usize = 4096;

/// Semaphore state: `sum(active qty) <= capacity` always.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SemaphoreState {
    /// Total capacity (permits outstanding never exceed this).
    pub capacity: u64,
    /// Live permits by id.
    pub permits: std::collections::BTreeMap<PermitId, PermitRecord>,
}

impl SemaphoreState {
    /// Total outstanding quantity across live permits (saturating; the real
    /// invariant keeps this `<= capacity`).
    #[must_use]
    pub fn outstanding(&self) -> u64 {
        self.permits
            .values()
            .fold(0u64, |sum, permit| sum.saturating_add(permit.qty))
    }

    /// Structural invariant: outstanding never exceeds capacity.
    #[must_use]
    pub fn invariant_holds(&self) -> bool {
        self.outstanding() <= self.capacity
    }
}

impl Encode for SemaphoreState {
    fn encoded_len(&self) -> usize {
        8 + 4 + self.permits.len() * (16 + 16)
    }

    fn encode(&self, out: &mut Vec<u8>) {
        self.capacity.encode(out);
        let count = u32::try_from(self.permits.len()).unwrap_or(u32::MAX);
        count.encode(out);
        for (id, permit) in &self.permits {
            id.encode(out);
            permit.encode(out);
        }
    }
}

impl Decode for SemaphoreState {
    fn decode(input: &[u8]) -> Result<(Self, usize), CodecError> {
        let (capacity, mut at) = u64::decode(input)?;
        let (count, used) = u32::decode(&input[at..])?;
        at += used;
        let count = usize::try_from(count).map_err(|_| CodecError::InvalidTag {
            kind: "semaphore-permits",
            tag: 0xFF,
        })?;
        if count > MAX_SEMAPHORE_PERMITS {
            return Err(CodecError::InvalidTag {
                kind: "semaphore-permits",
                tag: 0xFE,
            });
        }
        let mut permits = std::collections::BTreeMap::new();
        for _ in 0..count {
            let (id, first) = PermitId::decode(&input[at..])?;
            let (permit, second) = PermitRecord::decode(&input[at + first..])?;
            at += first + second;
            if permits.insert(id, permit).is_some() {
                return Err(CodecError::InvalidTag {
                    kind: "semaphore-permit-id",
                    tag: 0xFD,
                });
            }
        }
        Ok((Self { capacity, permits }, at))
    }
}

/// Monotonic fencing token: every successful ownership change advances it.
/// External systems reject `token < current` — that rejection, not time
/// expiry, is what makes an old holder powerless.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FencingToken(u64);

impl FencingToken {
    /// The token no live holder has ever held (initial state).
    pub const ZERO: Self = Self(0);

    /// Wraps a raw token value.
    #[must_use]
    pub const fn from_u64(value: u64) -> Self {
        Self(value)
    }

    /// Returns the raw value (monotonic, comparable by recipients).
    #[must_use]
    pub const fn as_u64(self) -> u64 {
        self.0
    }

    /// Successor token for the next ownership grant.
    ///
    /// # Errors
    ///
    /// Returns [`VersionExhausted`] at `u64::MAX`: tokens must never wrap,
    /// so the lease freezes instead of reissuing a stale-comparable token.
    pub const fn next(self) -> Result<Self, VersionExhausted> {
        match self.0.checked_add(1) {
            Some(value) => Ok(Self(value)),
            None => Err(VersionExhausted),
        }
    }
}

impl Encode for FencingToken {
    fn encoded_len(&self) -> usize {
        8
    }

    fn encode(&self, out: &mut Vec<u8>) {
        self.0.encode(out);
    }
}

impl Decode for FencingToken {
    fn decode(input: &[u8]) -> Result<(Self, usize), CodecError> {
        let (raw, consumed) = u64::decode(input)?;
        Ok((Self::from_u64(raw), consumed))
    }
}

/// Current lease holder: who owns the lease, under which fencing token,
/// until which logical instant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LeaseHolder {
    /// Owner/session identity holding the lease.
    pub owner: u64,
    /// Fencing token issued with this grant (strictly increasing).
    pub fencing: FencingToken,
    /// Logical expiry in micros (deterministic `WallTimestamp`, never a clock
    /// read inside state logic).
    pub expires_at: kivi_types::WallTimestamp,
}

impl Encode for LeaseHolder {
    fn encoded_len(&self) -> usize {
        8 + 8 + self.expires_at.encoded_len()
    }

    fn encode(&self, out: &mut Vec<u8>) {
        self.owner.encode(out);
        self.fencing.encode(out);
        self.expires_at.encode(out);
    }
}

impl Decode for LeaseHolder {
    fn decode(input: &[u8]) -> Result<(Self, usize), CodecError> {
        let (owner, first) = u64::decode(input)?;
        let (fencing, second) = FencingToken::decode(&input[first..])?;
        let (expires_at, third) = kivi_types::WallTimestamp::decode(&input[first + second..])?;
        Ok((
            Self {
                owner,
                fencing,
                expires_at,
            },
            first + second + third,
        ))
    }
}

/// Lease state: at most one live holder plus the next token to issue.
/// `next_fencing` only advances; a stale release/renew naming an older
/// token can never revive or free a newer holder.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LeaseState {
    /// Live holder (`None` when free or expired-but-unreclaimed).
    pub holder: Option<LeaseHolder>,
    /// Next fencing token to issue (monotonic across all grants).
    pub next_fencing: FencingToken,
}

impl LeaseState {
    /// A free lease ready for its first grant.
    #[must_use]
    pub const fn free() -> Self {
        Self {
            holder: None,
            next_fencing: FencingToken::from_u64(1),
        }
    }
}

impl Encode for LeaseState {
    fn encoded_len(&self) -> usize {
        1 + self.holder.map_or(0, |holder| holder.encoded_len()) + 8
    }

    fn encode(&self, out: &mut Vec<u8>) {
        match &self.holder {
            None => out.push(0),
            Some(holder) => {
                out.push(1);
                holder.encode(out);
            }
        }
        self.next_fencing.encode(out);
    }
}

impl Decode for LeaseState {
    fn decode(input: &[u8]) -> Result<(Self, usize), CodecError> {
        let (tag, mut at) = u8::decode(input)?;
        let holder = match tag {
            0 => None,
            1 => {
                let (holder, used) = LeaseHolder::decode(&input[at..])?;
                at += used;
                Some(holder)
            }
            other => {
                return Err(CodecError::InvalidTag {
                    kind: "lease-holder",
                    tag: other,
                });
            }
        };
        let (next_fencing, used) = FencingToken::decode(&input[at..])?;
        Ok((
            Self {
                holder,
                next_fencing,
            },
            at + used,
        ))
    }
}

/// Maximum entries retained in one stream shard object (bounds the durable
/// root; consumers trim via explicit trim, operators watch `next_offset`).
pub const MAX_STREAM_SHARD_ENTRIES: usize = 1024;
/// Maximum payload bytes per stream entry (larger payloads stay chunked
/// values referenced by key; the shard orders references, never bulk).
pub const MAX_STREAM_ENTRY_BYTES: usize = 64 << 10;

/// One ordered stream entry: its shard-local offset plus payload.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StreamEntry {
    /// Shard-local offset (assigned `next_offset`, never reused).
    pub offset: u64,
    /// Partition key that routed this entry here (retained for audit).
    pub partition: Vec<u8>,
    /// Entry payload bytes.
    pub payload: Bytes,
}

impl Encode for StreamEntry {
    fn encoded_len(&self) -> usize {
        8 + (4 + self.partition.len()) + (4 + self.payload.len())
    }

    fn encode(&self, out: &mut Vec<u8>) {
        self.offset.encode(out);
        encode_bytes(out, &self.partition);
        encode_bytes(out, &self.payload);
    }
}

impl Decode for StreamEntry {
    fn decode(input: &[u8]) -> Result<(Self, usize), CodecError> {
        let (offset, first) = u64::decode(input)?;
        let (partition, second) = decode_byte_vec(&input[first..])?;
        let (payload, third) = decode_byte_vec(&input[first + second..])?;
        Ok((
            Self {
                offset,
                partition,
                payload: Bytes::from(payload),
            },
            first + second + third,
        ))
    }
}

/// One shard of a sharded stream: `next_offset` plus the ordered entry
/// window. Order within the shard is the append order; no cross-shard
/// total order exists or is implied.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StreamShardState {
    /// Stream identity (all shards of a stream share it).
    pub stream: [u8; 16],
    /// Shard index within the stream.
    pub shard: u32,
    /// Next offset to assign (monotonic, never reused after trim).
    pub next_offset: u64,
    /// Retained entries in offset order (oldest first).
    pub entries: std::collections::VecDeque<StreamEntry>,
}

impl Encode for StreamShardState {
    fn encoded_len(&self) -> usize {
        16 + 4
            + 8
            + 4
            + self
                .entries
                .iter()
                .map(StreamEntry::encoded_len)
                .sum::<usize>()
    }

    fn encode(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.stream);
        self.shard.encode(out);
        self.next_offset.encode(out);
        let count = u32::try_from(self.entries.len()).unwrap_or(u32::MAX);
        count.encode(out);
        for entry in &self.entries {
            entry.encode(out);
        }
    }
}

impl Decode for StreamShardState {
    fn decode(input: &[u8]) -> Result<(Self, usize), CodecError> {
        let (stream, mut at) = <[u8; 16]>::decode(input)?;
        let (shard, used) = u32::decode(&input[at..])?;
        at += used;
        let (next_offset, used) = u64::decode(&input[at..])?;
        at += used;
        let (count, used) = u32::decode(&input[at..])?;
        at += used;
        let count = usize::try_from(count).map_err(|_| CodecError::InvalidTag {
            kind: "stream-entries",
            tag: 0xFF,
        })?;
        if count > MAX_STREAM_SHARD_ENTRIES {
            return Err(CodecError::InvalidTag {
                kind: "stream-entries",
                tag: 0xFE,
            });
        }
        let mut entries = std::collections::VecDeque::with_capacity(count);
        let mut previous: Option<u64> = None;
        for _ in 0..count {
            let (entry, used) = StreamEntry::decode(&input[at..])?;
            at += used;
            if entry.payload.len() > MAX_STREAM_ENTRY_BYTES {
                return Err(CodecError::InvalidTag {
                    kind: "stream-entry",
                    tag: 0xFD,
                });
            }
            if previous.is_some_and(|prev| entry.offset <= prev) {
                return Err(CodecError::InvalidTag {
                    kind: "stream-offset",
                    tag: 0xFC,
                });
            }
            previous = Some(entry.offset);
            entries.push_back(entry);
        }
        Ok((
            Self {
                stream,
                shard,
                next_offset,
                entries,
            },
            at,
        ))
    }
}

/// Physical representation tag. Private: logical semantics never branch on
/// it (matches go through [`LogicalValue`]); it exists so debugging,
/// checkpoints, and tiering can name what a root physically is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Representation {
    Inline,
    Chunked,
    Fabric,
}

/// A stored logical object: value, version, expiry, and representation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StoredObject {
    value: LogicalValue,
    version: ObjectVersion,
    expiry: Expiry,
    representation: Representation,
}

impl StoredObject {
    /// Builds an object. Only the store constructs these; callers go through
    /// typed operations and mutations. The representation derives from the
    /// value, so a fabric root can never masquerade as inline or reverse.
    pub(crate) fn new(value: LogicalValue, version: ObjectVersion, expiry: Expiry) -> Self {
        let representation = match &value {
            LogicalValue::Bytes(_)
            | LogicalValue::StrictCounter(_)
            | LogicalValue::CommutativeCounter(_)
            | LogicalValue::BoundedCounter(_)
            | LogicalValue::Semaphore(_)
            | LogicalValue::Lease(_)
            | LogicalValue::StreamShard(_) => Representation::Inline,
            LogicalValue::Chunked(_) => Representation::Chunked,
            LogicalValue::Fabric(_) => Representation::Fabric,
        };
        Self {
            value,
            version,
            expiry,
            representation,
        }
    }

    /// Rebuilds an object from checkpointed components. Checkpoint
    /// recovery is the only production caller: versions are logical
    /// history and must be restored exactly, never recomputed. Chunked
    /// roots restore as chunk references (bulk bytes stay in chunk packs);
    /// fabric roots restore as fabric references (bytes stay materialized);
    /// dependency validation proves them before serving.
    #[must_use]
    pub fn restore(value: LogicalValue, version: ObjectVersion, expiry: Expiry) -> Self {
        Self::new(value, version, expiry)
    }

    /// Returns the logical value.
    #[must_use]
    pub const fn value(&self) -> &LogicalValue {
        &self.value
    }

    /// Returns the logical version.
    #[must_use]
    pub const fn version(&self) -> ObjectVersion {
        self.version
    }

    /// Returns the logical expiry.
    #[must_use]
    pub const fn expiry(&self) -> Expiry {
        self.expiry
    }

    /// Returns the logical type.
    #[must_use]
    pub const fn object_type(&self) -> ObjectType {
        self.value.object_type()
    }

    /// Whether this root addresses immutable chunks instead of resident
    /// bytes. Diagnostics, checkpoints, and GC use this; reads resolve
    /// through the engine either way.
    #[must_use]
    pub const fn is_chunked(&self) -> bool {
        matches!(self.value, LogicalValue::Chunked(_))
    }

    /// Returns the chunk reference of a chunked root, if chunked.
    #[must_use]
    pub const fn chunk_ref(&self) -> Option<ChunkedRef> {
        match &self.value {
            LogicalValue::Chunked(chunked) => Some(*chunked),
            _ => None,
        }
    }

    /// Whether this root names a Memory Fabric materialization instead of
    /// resident bytes. Diagnostics, checkpoints, and GC use this; reads
    /// resolve through the engine either way.
    #[must_use]
    pub const fn is_fabric(&self) -> bool {
        matches!(self.value, LogicalValue::Fabric(_))
    }

    /// Returns the fabric reference of a fabric root, if fabric.
    #[must_use]
    pub const fn fabric_ref(&self) -> Option<FabricRef> {
        match &self.value {
            LogicalValue::Fabric(fabric) => Some(*fabric),
            _ => None,
        }
    }

    /// Returns the physical representation name for `EXPLAIN`-style
    /// diagnostics (`"inline"`, `"chunked"`, or `"fabric"`).
    /// Informational only.
    #[must_use]
    pub const fn representation_name(&self) -> &'static str {
        match self.representation {
            Representation::Inline => "inline",
            Representation::Chunked => "chunked",
            Representation::Fabric => "fabric",
        }
    }

    /// Whether the object is logically absent at `now` (`now >= expires_at`).
    /// Pure comparison: the caller supplies time from the clock boundary.
    #[must_use]
    pub const fn is_expired(&self, now: kivi_types::WallTimestamp) -> bool {
        self.expiry.is_expired(now)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn versions_advance_and_fail_explicitly_at_the_top() {
        assert_eq!(ObjectVersion::FIRST.as_u64(), 1);
        assert_eq!(
            ObjectVersion::FIRST.next().expect("advances"),
            ObjectVersion::from_u64(2)
        );
        assert_eq!(
            ObjectVersion::from_u64(u64::MAX).next(),
            Err(VersionExhausted)
        );
    }

    #[test]
    fn keys_wrap_bytes_cheaply() {
        let key = Key::new(b"user:123".as_slice());
        assert_eq!(key.as_bytes(), b"user:123");
        assert_eq!(key.to_string(), "user:123");
        assert_eq!(Key::from("abc"), Key::new(b"abc".as_slice()));
    }

    #[test]
    fn semantic_states_round_trip_and_hold_invariants() {
        fn round_trip<T>(value: &T) -> T
        where
            T: Encode + Decode + PartialEq + core::fmt::Debug,
        {
            let mut bytes = Vec::new();
            value.encode(&mut bytes);
            assert_eq!(bytes.len(), value.encoded_len());
            let (back, consumed) = T::decode(&bytes).expect("decode");
            assert_eq!(consumed, bytes.len());
            assert_eq!(&back, value);
            back
        }

        let bounded = BoundedCounterState {
            value: 30,
            capacity: 100,
            share: EscrowShare {
                holder: kivi_types::TabletId::from_u64(9),
                min: 0,
                max: 60,
            },
        };
        assert!(bounded.invariant_holds());
        round_trip(&bounded);
        assert!(
            !BoundedCounterState {
                value: 61,
                ..bounded
            }
            .invariant_holds()
        );

        let mut permits = std::collections::BTreeMap::new();
        permits.insert(
            PermitId::from_bytes([0xA5; 16]),
            PermitRecord { owner: 7, qty: 3 },
        );
        let semaphore = SemaphoreState {
            capacity: 10,
            permits,
        };
        assert_eq!(semaphore.outstanding(), 3);
        assert!(semaphore.invariant_holds());
        round_trip(&semaphore);

        let lease = LeaseState {
            holder: Some(LeaseHolder {
                owner: 7,
                fencing: FencingToken::from_u64(4),
                expires_at: kivi_types::WallTimestamp::from_micros(99),
            }),
            next_fencing: FencingToken::from_u64(5),
        };
        round_trip(&lease);
        round_trip(&LeaseState::free());
        assert_eq!(
            FencingToken::from_u64(4).next().expect("advances"),
            FencingToken::from_u64(5)
        );
        assert_eq!(
            FencingToken::from_u64(u64::MAX).next(),
            Err(VersionExhausted)
        );

        let mut entries = std::collections::VecDeque::new();
        entries.push_back(StreamEntry {
            offset: 0,
            partition: vec![0x70],
            payload: Bytes::from_static(b"e0"),
        });
        let shard = StreamShardState {
            stream: [0x5E; 16],
            shard: 2,
            next_offset: 1,
            entries,
        };
        round_trip(&shard);

        for object_type in [
            ObjectType::Bytes,
            ObjectType::StrictCounter,
            ObjectType::CommutativeCounter,
            ObjectType::BoundedCounter,
            ObjectType::Semaphore,
            ObjectType::Lease,
            ObjectType::StreamShard,
        ] {
            round_trip(&object_type);
        }
    }
}
