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

use kivi_state::{Key, Operation};
use kivi_tablet::{DirectoryVersion, HashPrefix, OrderedRange, PartitionRange};
use kivi_types::{
    ClusterId, NamespaceId, NodeId, NodeIncarnation, RequestIdentity, RequestSeq, TabletEpoch,
    TabletId, UnixMicros, WorkerId,
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
    /// semantic rejections are not.
    #[must_use]
    pub const fn is_retriable(self) -> bool {
        matches!(self, Self::StaleRoute | Self::NotLocal | Self::Overloaded)
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
    /// Absolute expiry micros (meaningful for `ExpireAt` only).
    pub expiry: u64,
    /// Patch offset (meaningful for `SetRange` only): the logical byte
    /// index the patch overwrites from, zero-padding past the end.
    pub offset: u64,
    /// Retry identity for mutating opcodes (`None` for reads and for
    /// clients that predate durable dedup). Durable servers require it on
    /// every mutating request; ephemeral servers ignore it.
    pub identity: Option<RequestIdentity>,
    /// Client acknowledgement watermark for the identity's session
    /// (meaningful only alongside `identity`; zero otherwise).
    pub ack_floor: RequestSeq,
}

impl Request {
    /// Translates into the project-owned typed operation. Total: every
    /// decodable request maps to exactly one operation.
    #[must_use]
    pub fn into_operation(self) -> Operation {
        let key = Key::from(self.key);
        match self.opcode {
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
        }
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
    /// Retry directly at the attached authority.
    Redirect(RedirectInfo),
    /// Machine code plus optional human diagnostic (never Rust internals).
    Diagnostic(String),
}

/// One typed native response: stable status plus body.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Response {
    /// Machine-readable outcome.
    pub status: Status,
    /// Outcome payload.
    pub body: ResponseBody,
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
            Opcode::CounterAdd => push_i64(&mut out, self.delta),
            Opcode::ExpireAt => push_u64(&mut out, self.expiry),
            Opcode::SetRange => {
                push_u64(&mut out, self.offset);
                push_blob(&mut out, self.value.as_deref().unwrap_or_default());
            }
            Opcode::Get
            | Opcode::Delete
            | Opcode::Exists
            | Opcode::CounterGet
            | Opcode::PersistExpiry
            | Opcode::GetExpiry
            | Opcode::GetStream => {}
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
            value: None,
            delta: 0,
            expiry: 0,
            offset: 0,
            identity: None,
            ack_floor: RequestSeq::from_u64(0),
        };
        match opcode {
            Opcode::Set => {
                request.value = Some(cursor.blob(CONTEXT)?.to_vec());
            }
            Opcode::CounterAdd => {
                request.delta = cursor.i64(CONTEXT)?;
            }
            Opcode::ExpireAt => {
                request.expiry = cursor.u64(CONTEXT)?;
            }
            Opcode::SetRange => {
                request.offset = cursor.u64(CONTEXT)?;
                request.value = Some(cursor.blob(CONTEXT)?.to_vec());
            }
            Opcode::Get
            | Opcode::Delete
            | Opcode::Exists
            | Opcode::CounterGet
            | Opcode::PersistExpiry
            | Opcode::GetExpiry
            | Opcode::GetStream => {}
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
        cursor.end(CONTEXT)?;
        Ok(request)
    }
}

impl Response {
    /// Encodes one response payload. The opcode travels alongside the status
    /// so decoders never guess the body shape from connection state.
    #[must_use]
    pub fn encode(&self, opcode: Opcode) -> Vec<u8> {
        let mut out = Vec::with_capacity(32);
        push_u16(&mut out, self.status.as_u16());
        push_u8(&mut out, opcode.as_u8());
        match &self.body {
            ResponseBody::Value(value) => push_blob(&mut out, value),
            ResponseBody::Stored { version } => push_u64(&mut out, *version),
            ResponseBody::Deleted { existed } => push_u8(&mut out, u8::from(*existed)),
            ResponseBody::Exists(present) => push_u8(&mut out, u8::from(*present)),
            ResponseBody::Counter(value) => push_i64(&mut out, *value),
            ResponseBody::CounterUpdated { value, version } => {
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
            ResponseBody::Redirect(info) => out.extend_from_slice(&info.encode()),
            ResponseBody::Diagnostic(message) => {
                push_u16(&mut out, u16::try_from(message.len()).unwrap_or(u16::MAX));
                out.extend_from_slice(message.as_bytes());
            }
        }
        out
    }

    /// Decodes one response payload, requiring exact consumption.
    ///
    /// # Errors
    ///
    /// Returns [`ProtocolError`] on unknown statuses/opcodes, truncation,
    /// trailing bytes, or shapes inconsistent with the status/opcode pair.
    pub fn decode(input: &[u8]) -> Result<(Self, Opcode), ProtocolError> {
        const CONTEXT: &str = "response";
        let mut cursor = Cursor::new(input);
        let status = Status::from_u16(cursor.u16(CONTEXT)?)
            .ok_or(ProtocolError::Malformed { context: CONTEXT })?;
        let opcode = Opcode::from_u8(cursor.u8(CONTEXT)?)
            .ok_or(ProtocolError::Malformed { context: CONTEXT })?;
        let body = match status {
            Status::Ok => match opcode {
                Opcode::Get => ResponseBody::Value(cursor.blob(CONTEXT)?.to_vec()),
                Opcode::Set | Opcode::SetRange => ResponseBody::Stored {
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
            },
            Status::NotFound => match opcode {
                Opcode::Get | Opcode::CounterGet | Opcode::GetExpiry | Opcode::GetStream => {
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
        cursor.end(CONTEXT)?;
        Ok((Self { status, body }, opcode))
    }
}

/// Decodes a `0/1` flag byte.
fn flag(cursor: &mut Cursor<'_>) -> Result<bool, ProtocolError> {
    match cursor.u8("flag")? {
        0 => Ok(false),
        1 => Ok(true),
        _ => Err(ProtocolError::Malformed { context: "flag" }),
    }
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
            identity: Some(RequestIdentity::new(
                kivi_types::SessionId::from_u128(0x00C0_FFEE),
                RequestSeq::from_u64(41),
            )),
            ack_floor: RequestSeq::from_u64(40),
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
        // Operation translation is total and typed.
        let op = decoded.into_operation();
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
            // GetStream executes the same read as Get; only delivery differs.
            Opcode::GetStream => assert!(matches!(op, Operation::Get { .. })),
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
    fn every_status_round_trips(#[case] status: Status) {
        assert_eq!(Status::from_u16(status.as_u16()), Some(status));
        assert_eq!(
            status.is_retriable(),
            matches!(
                status,
                Status::StaleRoute | Status::NotLocal | Status::Overloaded
            )
        );
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
        ];
        for (opcode, body) in bodies {
            let response = Response {
                status: Status::Ok,
                body,
            };
            let (decoded, back) = Response::decode(&response.encode(opcode)).expect("round trip");
            assert_eq!(back, opcode);
            assert_eq!(decoded, response);
        }
        // Redirects and diagnostics ride their own statuses.
        let redirect = Response {
            status: Status::StaleRoute,
            body: ResponseBody::Redirect(redirect()),
        };
        let (decoded, opcode) = Response::decode(&redirect.encode(Opcode::Get)).expect("redirect");
        assert_eq!(opcode, Opcode::Get);
        assert_eq!(decoded, redirect);
        let error = Response {
            status: Status::WrongType,
            body: ResponseBody::Diagnostic("needs bytes".to_owned()),
        };
        let (decoded, _) = Response::decode(&error.encode(Opcode::Get)).expect("error");
        assert_eq!(decoded, error);
        // Missing values surface as NotFound, not empty Ok bodies.
        let missing = Response {
            status: Status::NotFound,
            body: ResponseBody::Diagnostic(String::new()),
        };
        let (decoded, opcode) = Response::decode(&missing.encode(Opcode::CounterGet)).expect("404");
        assert_eq!(decoded.status, Status::NotFound);
        assert_eq!(opcode, Opcode::CounterGet);
    }

    #[test]
    fn malformed_responses_rejected() {
        // Truncated Ok body.
        let mut bytes = Response {
            status: Status::Ok,
            body: ResponseBody::Stored { version: 1 },
        }
        .encode(Opcode::Set);
        bytes.pop();
        assert!(Response::decode(&bytes).is_err());
        // Ok body inconsistent with a redirect status.
        let mut bad = Response {
            status: Status::StaleRoute,
            body: ResponseBody::Stored { version: 1 },
        }
        .encode(Opcode::Set);
        let _ = &mut bad;
        assert!(Response::decode(&bad).is_err());
        // Unknown status code.
        let mut bad = Response {
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
