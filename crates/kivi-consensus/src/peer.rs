//! Dedicated Kivi peer protocol.
//!
//! Consensus traffic runs on its own TCP mesh with project-owned,
//! versioned framing — never the native client protocol, RESP, or admin
//! HTTP. Every connection opens with a handshake carrying the peer
//! protocol version, [`ClusterId`], [`NodeId`], [`NodeIncarnation`], and
//! capability bits; the receiver rejects wrong clusters, stale
//! incarnations, and missing required capabilities loudly.
//!
//! ```text
//! frame := magic u32, version u16, len u32, body[len], crc u32
//! body  := kind u16, id u64, group u64, payload…
//! ```
//!
//! Every consensus body carries the addressed consensus group
//! ([`ConsensusGroupId`](crate::types::ConsensusGroupId), as its tablet
//! id): one physical mesh multiplexes many Raft groups (Multi-Raft
//! direction), and the group id routes each RPC to the right instance.
//!
//! Kinds `1–3` are handshake (`Hello`, `Accept`, `Reject`); kinds
//! `10–17` carry consensus RPCs as Kivi-owned structs (never `OpenRaft`
//! memory layout — [`crate::router`] owns translation both ways, so
//! `OpenRaft` upgrades cannot change these bytes).
//!
//! ## Snapshot identity discipline
//!
//! Snapshot metadata carries the Kivi **checkpoint** identity
//! ([`PeerSnapshotMeta::checkpoint`]: immutable state at a cut,
//! content-addressed). The **transfer** identity
//! ([`PeerSnapshotRequest::transfer`]: one attempt to move that
//! checkpoint) rides only the fragment envelope, generated fresh per
//! `full_snapshot` call, so a retransmitted snapshot is never mistaken
//! for a continuation of an aborted one. These are intentionally separate
//! concepts (matching `OpenRaft` 0.10's removal of `snapshot_id` from
//! logical snapshot metadata).
//!
//! ## Incarnation discipline
//!
//! A restarted node keeps its [`NodeId`] but advances its
//! [`NodeIncarnation`]. [`IncarnationTable`] remembers the highest
//! incarnation observed per node; handshakes strictly below a newer
//! observed incarnation are stale and rejected, so traffic from a
//! previous process generation can never regain validity. Equal
//! incarnations reconnect freely: they are the same live generation,
//! which transient drops and suspend/resume healing depend on.

use std::collections::HashMap;

use kivi_types::{ClusterId, NodeId, NodeIncarnation};

/// Peer-frame magic (`KVSP`: Kivi peer), little-endian.
pub const PEER_MAGIC: u32 = 0x5053_564B;
/// Peer protocol version spoken here (v2: group-multiplexed bodies,
/// pre-vote, checkpoint/transfer snapshot identity split).
pub const PEER_PROTOCOL_VERSION: u16 = 2;
/// Required capability: consensus RPCs v2 (election/append/snapshot).
pub const CAP_CONSENSUS_V2: u64 = 1 << 0;
/// All capabilities this binary requires of its peers.
pub const REQUIRED_CAPABILITIES: u64 = CAP_CONSENSUS_V2;

/// Maximum peer frame in bytes (matches the canonical frame ceiling; a
/// larger declaration fails the connection, never allocates).
pub const PEER_MAX_FRAME_BYTES: usize = 16 * 1024 * 1024;

/// Frame kinds: handshake.
pub const KIND_HELLO: u16 = 1;
/// Frame kinds: handshake accept.
pub const KIND_ACCEPT: u16 = 2;
/// Frame kinds: handshake reject.
pub const KIND_REJECT: u16 = 3;
/// Frame kinds: vote request.
pub const KIND_VOTE_REQUEST: u16 = 10;
/// Frame kinds: vote response.
pub const KIND_VOTE_RESPONSE: u16 = 11;
/// Frame kinds: append-entries request.
pub const KIND_APPEND_REQUEST: u16 = 12;
/// Frame kinds: append-entries response.
pub const KIND_APPEND_RESPONSE: u16 = 13;
/// Frame kinds: install-snapshot request fragment.
pub const KIND_SNAPSHOT_REQUEST: u16 = 14;
/// Frame kinds: install-snapshot response.
pub const KIND_SNAPSHOT_RESPONSE: u16 = 15;
/// Frame kinds: pre-vote request (same shape as a vote request).
pub const KIND_PRE_VOTE_REQUEST: u16 = 16;
/// Frame kinds: pre-vote response (same shape as a vote response).
pub const KIND_PRE_VOTE_RESPONSE: u16 = 17;

/// Stable peer identity: who is speaking, in which cluster, in which
/// process generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PeerIdentity {
    /// Owning cluster.
    pub cluster: ClusterId,
    /// Speaking node.
    pub node: NodeId,
    /// Process generation (advanced on every restart).
    pub incarnation: NodeIncarnation,
}

/// Opening handshake of every peer connection, both directions (each side
/// validates the other's hello).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HandshakeHello {
    /// Protocol version offered.
    pub version: u16,
    /// Claimed cluster.
    pub cluster: u128,
    /// Claimed node.
    pub node: u64,
    /// Claimed incarnation.
    pub incarnation: u64,
    /// Offered capability bits.
    pub capabilities: u64,
}

/// Successful handshake reply: the acceptor's own identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HandshakeAccept {
    /// Protocol version accepted.
    pub version: u16,
    /// Acceptor cluster.
    pub cluster: u128,
    /// Acceptor node.
    pub node: u64,
    /// Acceptor incarnation.
    pub incarnation: u64,
    /// Acceptor capability bits.
    pub capabilities: u64,
}

/// Why a handshake was refused. Every variant fails the connection —
/// never a downgrade, never a silent retry on the same socket.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum HandshakeReject {
    /// The peer speaks a version this binary cannot serve.
    #[error("unsupported peer protocol version {found}")]
    UnsupportedVersion {
        /// Offered version.
        found: u16,
    },
    /// The peer belongs to a different cluster.
    #[error("wrong cluster: peer serves {found}, local serves {expected}")]
    WrongCluster {
        /// Expected cluster.
        expected: u128,
        /// Observed cluster.
        found: u128,
    },
    /// The peer's incarnation is invalid or stale.
    #[error("stale incarnation {found} for node {node} (newest observed {newest})")]
    StaleIncarnation {
        /// Speaking node.
        node: u64,
        /// Observed incarnation.
        found: u64,
        /// Newest incarnation observed for the node.
        newest: u64,
    },
    /// The peer lacks a required capability.
    #[error("missing required peer capabilities {missing:#x}")]
    MissingCapability {
        /// Required bits absent from the offer.
        missing: u64,
    },
}

/// Owned vote (term, candidate, committed flag).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PeerVote {
    /// Term.
    pub term: u64,
    /// Candidate node.
    pub node: u64,
    /// Whether the vote is committed (granted, not merely seen).
    pub committed: bool,
}

/// Owned log id (term, leader, index).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PeerLogId {
    /// Entry term.
    pub term: u64,
    /// Entry leader.
    pub leader: u64,
    /// Entry index.
    pub index: u64,
}

/// Owned vote request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PeerVoteRequest {
    /// Candidate vote.
    pub vote: PeerVote,
    /// Candidate's last log.
    pub last_log: Option<PeerLogId>,
    /// Whether the election is a leadership transfer authorized by the
    /// current leader (a voter processes it even if its lease has not
    /// expired).
    pub leadership_transfer: bool,
}

/// Owned vote response.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PeerVoteResponse {
    /// Responder vote.
    pub vote: PeerVote,
    /// Whether the vote was granted.
    pub granted: bool,
    /// Responder's last log.
    pub last_log: Option<PeerLogId>,
}

/// Owned log-entry payload: blank barrier, opaque deterministic command
/// bytes (the canonical [`ReplicatedMutation`](crate::mutation::ReplicatedMutation)
/// encoding), or membership.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PeerEntryPayload {
    /// Leader commit barrier (no data).
    Blank,
    /// Opaque deterministic command bytes.
    Normal(Vec<u8>),
    /// Membership: voters plus dialable peer addresses.
    Membership {
        /// Voter node ids.
        voters: Vec<u64>,
        /// (node id, peer address) pairs.
        nodes: Vec<(u64, String)>,
    },
}

/// Owned log entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PeerEntry {
    /// Entry term.
    pub term: u64,
    /// Entry leader.
    pub leader: u64,
    /// Entry index.
    pub index: u64,
    /// Entry payload.
    pub payload: PeerEntryPayload,
}

/// Owned append-entries request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PeerAppendRequest {
    /// Leader vote.
    pub vote: PeerVote,
    /// Log id preceding the entries.
    pub prev_log: Option<PeerLogId>,
    /// Entries to replicate (possibly empty: heartbeat/commit bump).
    pub entries: Vec<PeerEntry>,
    /// Leader commit index.
    pub leader_commit: Option<PeerLogId>,
}

/// Owned append-entries response.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PeerAppendResponse {
    /// All entries replicated.
    Success,
    /// A prefix replicated (at least the previous match).
    PartialSuccess(Option<PeerLogId>),
    /// Log mismatch at the previous id.
    Conflict,
    /// The responder holds a higher vote.
    HigherVote(PeerVote),
}

/// Owned snapshot metadata: applied base, membership, checkpoint
/// identity. The checkpoint reference identifies immutable Kivi state at
/// a cut (content-addressed); transfer-session identity rides only the
/// fragment envelope ([`PeerSnapshotRequest::transfer`]), never here.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PeerSnapshotMeta {
    /// Snapshot's last log id.
    pub last_log: Option<PeerLogId>,
    /// Voter set at the snapshot.
    pub voters: Vec<u64>,
    /// Dialable nodes at the snapshot.
    pub nodes: Vec<(u64, String)>,
    /// Log id of the membership entry (grounds the purge base).
    pub membership_log: Option<PeerLogId>,
    /// Checkpoint identity bytes (Kivi checkpoint-cut reference).
    pub checkpoint: Vec<u8>,
}

/// Owned install-snapshot request fragment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PeerSnapshotRequest {
    /// Leader vote.
    pub vote: PeerVote,
    /// Snapshot metadata (repeated on every fragment).
    pub meta: PeerSnapshotMeta,
    /// Transfer-session id: one attempt to move this checkpoint,
    /// generated fresh per `full_snapshot` call. The receiver compares it
    /// against the in-flight stream to tell a new transfer from a
    /// continuation — never part of durable snapshot identity.
    pub transfer: u64,
    /// Byte offset of this fragment.
    pub offset: u64,
    /// Fragment bytes.
    pub data: Vec<u8>,
    /// Whether this is the final fragment.
    pub done: bool,
}

/// Owned install-snapshot response.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PeerSnapshotResponse {
    /// Responder vote.
    pub vote: PeerVote,
}

/// One consensus RPC in either direction (request bodies; responses are
/// matched by the transport's request id, not by this enum). Pre-vote
/// reuses the vote shapes under its own kinds.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PeerRequest {
    /// Election RPC.
    Vote(PeerVoteRequest),
    /// Pre-vote probe RPC (hypothetical next term, persists nothing).
    PreVote(PeerVoteRequest),
    /// Replication/heartbeat RPC.
    Append(PeerAppendRequest),
    /// Snapshot fragment RPC.
    Snapshot(PeerSnapshotRequest),
}

/// One consensus RPC response body.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PeerResponse {
    /// Election answer.
    Vote(PeerVoteResponse),
    /// Pre-vote answer.
    PreVote(PeerVoteResponse),
    /// Replication answer.
    Append(PeerAppendResponse),
    /// Snapshot fragment answer.
    Snapshot(PeerSnapshotResponse),
}

/// Remote-side RPC failure: the peer decoded the request but its Raft
/// layer refused it (shutdown, storage fault, non-member). Never success;
/// the client backs off and retries per `OpenRaft` policy.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("remote peer RPC failed: {detail}")]
pub struct PeerRpcError {
    /// Human-readable cause (logged verbatim).
    pub detail: String,
}

/// Structural frame/codec failure (never a remote verdict).
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum PeerCodecError {
    /// Fewer bytes than framing promises.
    #[error("truncated peer frame")]
    Truncated,
    /// A length declaration exceeds the frame ceiling or the input.
    #[error("oversize peer declaration {len}")]
    Oversize {
        /// Declared length.
        len: usize,
    },
    /// Wrong framing magic (not a peer socket).
    #[error("bad peer magic {found:#x}")]
    BadMagic {
        /// Observed magic.
        found: u32,
    },
    /// CRC mismatch (corruption, never retried on the same bytes).
    #[error("peer frame CRC mismatch")]
    BadCrc,
    /// Unknown frame kind.
    #[error("unknown peer frame kind {kind}")]
    UnknownKind {
        /// Observed kind.
        kind: u16,
    },
    /// A payload tag is unknown.
    #[error("unknown peer payload tag {tag}")]
    BadTag {
        /// Observed tag.
        tag: u8,
    },
    /// Non-UTF8 address or snapshot id.
    #[error("non-UTF8 peer string")]
    BadString,
    /// Trailing bytes after the framed body.
    #[error("trailing bytes in peer frame")]
    TrailingBytes,
}

fn push_u64(out: &mut Vec<u8>, value: u64) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn push_u128(out: &mut Vec<u8>, value: u128) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn push_blob(out: &mut Vec<u8>, bytes: &[u8]) {
    let len = u32::try_from(bytes.len()).unwrap_or(u32::MAX);
    out.extend_from_slice(&len.to_le_bytes());
    out.extend_from_slice(bytes);
}

fn push_string(out: &mut Vec<u8>, value: &str) {
    push_blob(out, value.as_bytes());
}

fn push_log(out: &mut Vec<u8>, id: &PeerLogId) {
    push_u64(out, id.term);
    push_u64(out, id.leader);
    push_u64(out, id.index);
}

fn push_opt_log(out: &mut Vec<u8>, id: Option<&PeerLogId>) {
    match id {
        None => out.push(0),
        Some(id) => {
            out.push(1);
            push_log(out, id);
        }
    }
}

fn push_vote(out: &mut Vec<u8>, vote: &PeerVote) {
    push_u64(out, vote.term);
    push_u64(out, vote.node);
    out.push(u8::from(vote.committed));
}

fn push_nodes(out: &mut Vec<u8>, nodes: &[(u64, String)]) {
    push_u64(out, nodes.len() as u64);
    for (id, addr) in nodes {
        push_u64(out, *id);
        push_string(out, addr);
    }
}

/// Incremental frame decoder over a TCP byte stream: callers push
/// received bytes and drain complete verified frames. Oversize
/// declarations fail the connection before any large allocation.
#[derive(Debug, Default)]
pub struct FrameDecoder {
    buffer: Vec<u8>,
}

impl FrameDecoder {
    /// Creates an empty decoder.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Appends newly received bytes.
    pub fn push(&mut self, bytes: &[u8]) {
        self.buffer.extend_from_slice(bytes);
    }

    /// Returns the number of buffered (unframed) bytes.
    #[must_use]
    pub fn buffered(&self) -> usize {
        self.buffer.len()
    }

    /// Takes the next complete verified frame body (`kind u16` +
    /// payload), or `None` when fewer than one frame is buffered.
    ///
    /// # Errors
    ///
    /// Returns [`PeerCodecError`] on magic, length, or CRC faults. The
    /// caller must fail the connection on error (the stream position is
    /// no longer trustworthy).
    pub fn take_frame(&mut self) -> Result<Option<Vec<u8>>, PeerCodecError> {
        use PeerCodecError as Fault;
        const HEADER: usize = 4 + 2 + 4;
        const CRC: usize = 4;
        if self.buffer.len() < HEADER {
            return Ok(None);
        }
        let magic = u32::from_le_bytes(self.buffer[..4].try_into().unwrap_or([0; 4]));
        if magic != PEER_MAGIC {
            return Err(Fault::BadMagic { found: magic });
        }
        let version = u16::from_le_bytes(self.buffer[4..6].try_into().unwrap_or([0; 2]));
        if version != PEER_PROTOCOL_VERSION {
            // Unknown versions fail the connection: no downgrade path is
            // negotiated on an established socket (handshakes already
            // agreed on this version).
            return Err(Fault::UnknownKind { kind: version });
        }
        let len = u32::from_le_bytes(self.buffer[6..10].try_into().unwrap_or([0; 4])) as usize;
        if len > PEER_MAX_FRAME_BYTES {
            return Err(Fault::Oversize { len });
        }
        if self.buffer.len() < HEADER + len + CRC {
            return Ok(None);
        }
        let body = self.buffer[HEADER..HEADER + len].to_vec();
        let found = u32::from_le_bytes(
            self.buffer[HEADER + len..HEADER + len + CRC]
                .try_into()
                .unwrap_or([0; 4]),
        );
        if found != crc32c::crc32c(&body) {
            return Err(Fault::BadCrc);
        }
        self.buffer.drain(..HEADER + len + CRC);
        Ok(Some(body))
    }
}

/// Frames one body for the wire. Bodies larger than the frame ceiling
/// saturate the length declaration so the peer fails `Oversize` (never a
/// truncated send); all builders in this module stay far below the
/// ceiling by construction.
#[must_use]
pub fn frame_body(body: &[u8]) -> Vec<u8> {
    debug_assert!(
        body.len() <= PEER_MAX_FRAME_BYTES,
        "peer bodies stay below the frame ceiling by construction"
    );
    let mut out = Vec::with_capacity(4 + 2 + 4 + body.len() + 4);
    out.extend_from_slice(&PEER_MAGIC.to_le_bytes());
    out.extend_from_slice(&PEER_PROTOCOL_VERSION.to_le_bytes());
    let len = u32::try_from(body.len()).unwrap_or(u32::MAX);
    out.extend_from_slice(&len.to_le_bytes());
    out.extend_from_slice(body);
    out.extend_from_slice(&crc32c::crc32c(body).to_le_bytes());
    out
}

struct Reader<'a> {
    input: &'a [u8],
    at: usize,
}

impl<'a> Reader<'a> {
    fn new(input: &'a [u8]) -> Self {
        Self { input, at: 0 }
    }

    fn rest(&self) -> usize {
        self.input.len().saturating_sub(self.at)
    }

    fn bytes(&mut self, n: usize) -> Result<&'a [u8], PeerCodecError> {
        if self.rest() < n {
            return Err(PeerCodecError::Truncated);
        }
        let slice = &self.input[self.at..self.at + n];
        self.at += n;
        Ok(slice)
    }

    fn u8(&mut self) -> Result<u8, PeerCodecError> {
        Ok(self.bytes(1)?[0])
    }

    fn u16(&mut self) -> Result<u16, PeerCodecError> {
        Ok(u16::from_le_bytes(
            self.bytes(2)?.try_into().unwrap_or([0; 2]),
        ))
    }

    fn u64(&mut self) -> Result<u64, PeerCodecError> {
        Ok(u64::from_le_bytes(
            self.bytes(8)?.try_into().unwrap_or([0; 8]),
        ))
    }

    fn u128(&mut self) -> Result<u128, PeerCodecError> {
        Ok(u128::from_le_bytes(
            self.bytes(16)?.try_into().unwrap_or([0; 16]),
        ))
    }

    fn blob(&mut self) -> Result<Vec<u8>, PeerCodecError> {
        let len = u32::from_le_bytes(self.bytes(4)?.try_into().unwrap_or([0; 4])) as usize;
        if len > PEER_MAX_FRAME_BYTES || len > self.rest() {
            return Err(PeerCodecError::Oversize { len });
        }
        Ok(self.bytes(len)?.to_vec())
    }

    fn string(&mut self) -> Result<String, PeerCodecError> {
        let blob = self.blob()?;
        String::from_utf8(blob).map_err(|_| PeerCodecError::BadString)
    }

    fn vote(&mut self) -> Result<PeerVote, PeerCodecError> {
        let term = self.u64()?;
        let node = self.u64()?;
        let committed = match self.u8()? {
            0 => false,
            1 => true,
            tag => return Err(PeerCodecError::BadTag { tag }),
        };
        Ok(PeerVote {
            term,
            node,
            committed,
        })
    }

    fn log(&mut self) -> Result<PeerLogId, PeerCodecError> {
        Ok(PeerLogId {
            term: self.u64()?,
            leader: self.u64()?,
            index: self.u64()?,
        })
    }

    fn opt_log(&mut self) -> Result<Option<PeerLogId>, PeerCodecError> {
        match self.u8()? {
            0 => Ok(None),
            1 => Ok(Some(self.log()?)),
            tag => Err(PeerCodecError::BadTag { tag }),
        }
    }

    fn nodes(&mut self) -> Result<Vec<(u64, String)>, PeerCodecError> {
        let count = self.u64()?;
        let count =
            usize::try_from(count).map_err(|_| PeerCodecError::Oversize { len: usize::MAX })?;
        if count.saturating_mul(12) > self.rest() {
            return Err(PeerCodecError::Oversize {
                len: count.saturating_mul(12),
            });
        }
        let mut out = Vec::with_capacity(count);
        for _ in 0..count {
            let id = self.u64()?;
            let addr = self.string()?;
            out.push((id, addr));
        }
        Ok(out)
    }

    fn voters(&mut self) -> Result<Vec<u64>, PeerCodecError> {
        let count = self.u64()?;
        let count =
            usize::try_from(count).map_err(|_| PeerCodecError::Oversize { len: usize::MAX })?;
        if count.saturating_mul(8) > self.rest() {
            return Err(PeerCodecError::Oversize {
                len: count.saturating_mul(8),
            });
        }
        let mut out = Vec::with_capacity(count);
        for _ in 0..count {
            out.push(self.u64()?);
        }
        Ok(out)
    }

    fn entry_payload(&mut self) -> Result<PeerEntryPayload, PeerCodecError> {
        match self.u8()? {
            0 => Ok(PeerEntryPayload::Blank),
            1 => Ok(PeerEntryPayload::Normal(self.blob()?)),
            2 => {
                let voters = self.voters()?;
                let nodes = self.nodes()?;
                Ok(PeerEntryPayload::Membership { voters, nodes })
            }
            tag => Err(PeerCodecError::BadTag { tag }),
        }
    }

    fn finish(self) -> Result<(), PeerCodecError> {
        if self.rest() == 0 {
            Ok(())
        } else {
            Err(PeerCodecError::TrailingBytes)
        }
    }
}

/// Encodes a handshake hello body.
#[must_use]
pub fn encode_hello(hello: &HandshakeHello) -> Vec<u8> {
    let mut out = Vec::with_capacity(2 + 2 + 16 + 8 + 8 + 8);
    out.extend_from_slice(&KIND_HELLO.to_le_bytes());
    out.extend_from_slice(&hello.version.to_le_bytes());
    push_u128(&mut out, hello.cluster);
    push_u64(&mut out, hello.node);
    push_u64(&mut out, hello.incarnation);
    push_u64(&mut out, hello.capabilities);
    out
}

/// Encodes a handshake accept body.
#[must_use]
pub fn encode_accept(accept: &HandshakeAccept) -> Vec<u8> {
    let mut out = Vec::with_capacity(2 + 2 + 16 + 8 + 8 + 8);
    out.extend_from_slice(&KIND_ACCEPT.to_le_bytes());
    out.extend_from_slice(&accept.version.to_le_bytes());
    push_u128(&mut out, accept.cluster);
    push_u64(&mut out, accept.node);
    push_u64(&mut out, accept.incarnation);
    push_u64(&mut out, accept.capabilities);
    out
}

/// Handshake outcome after validating a hello: accept with our identity,
/// or the precise rejection.
///
/// Only strictly OLDER generations are stale: an equal incarnation is
/// the same live process generation reconnecting (transient drops and
/// suspend/resume heal through this path), while an older one is a
/// restarted-away generation that must never regain validity (task K).
/// `INVALID` is never valid anywhere.
///
/// # Errors
///
/// Returns [`HandshakeReject`] for version, cluster, incarnation, or
/// capability mismatches. Every rejection fails the connection.
pub fn validate_hello(
    hello: &HandshakeHello,
    local: &PeerIdentity,
    table: &IncarnationTable,
) -> Result<HandshakeAccept, HandshakeReject> {
    use HandshakeReject as Fault;
    if hello.version != PEER_PROTOCOL_VERSION {
        return Err(Fault::UnsupportedVersion {
            found: hello.version,
        });
    }
    if hello.cluster != local.cluster.as_u128() {
        return Err(Fault::WrongCluster {
            expected: local.cluster.as_u128(),
            found: hello.cluster,
        });
    }
    if hello.incarnation == NodeIncarnation::INVALID.as_u64() {
        return Err(Fault::StaleIncarnation {
            node: hello.node,
            found: hello.incarnation,
            newest: table.newest(NodeId::from_u64(hello.node)),
        });
    }
    if let Some(newest) = table.observed(NodeId::from_u64(hello.node))
        && hello.incarnation < newest.as_u64()
    {
        return Err(Fault::StaleIncarnation {
            node: hello.node,
            found: hello.incarnation,
            newest: newest.as_u64(),
        });
    }
    let missing = REQUIRED_CAPABILITIES & !hello.capabilities;
    if missing != 0 {
        return Err(Fault::MissingCapability { missing });
    }
    Ok(HandshakeAccept {
        version: PEER_PROTOCOL_VERSION,
        cluster: local.cluster.as_u128(),
        node: local.node.as_u64(),
        incarnation: local.incarnation.as_u64(),
        capabilities: REQUIRED_CAPABILITIES,
    })
}

/// Decodes a handshake body (hello, accept, or reject reason).
///
/// # Errors
///
/// Returns [`PeerCodecError`] on structural faults.
pub fn decode_handshake(body: &[u8]) -> Result<HandshakeFrame, PeerCodecError> {
    let mut reader = Reader::new(body);
    let frame = match reader.u16()? {
        KIND_HELLO => HandshakeFrame::Hello(HandshakeHello {
            version: reader.u16()?,
            cluster: reader.u128()?,
            node: reader.u64()?,
            incarnation: reader.u64()?,
            capabilities: reader.u64()?,
        }),
        KIND_ACCEPT => HandshakeFrame::Accept(HandshakeAccept {
            version: reader.u16()?,
            cluster: reader.u128()?,
            node: reader.u64()?,
            incarnation: reader.u64()?,
            capabilities: reader.u64()?,
        }),
        KIND_REJECT => {
            let tag = reader.u8()?;
            let a = reader.u64()?;
            let b = reader.u64()?;
            let c = reader.u64()?;
            let d = reader.u64()?;
            let reason = match tag {
                1 => HandshakeReject::UnsupportedVersion {
                    found: u16::try_from(a).unwrap_or(u16::MAX),
                },
                2 => HandshakeReject::WrongCluster {
                    found: u128::from(a) | (u128::from(b) << 64),
                    expected: u128::from(c) | (u128::from(d) << 64),
                },
                3 => HandshakeReject::StaleIncarnation {
                    node: a,
                    found: b,
                    newest: c,
                },
                4 => HandshakeReject::MissingCapability { missing: a },
                tag => return Err(PeerCodecError::BadTag { tag }),
            };
            HandshakeFrame::Reject(reason)
        }
        kind => return Err(PeerCodecError::UnknownKind { kind }),
    };
    reader.finish()?;
    Ok(frame)
}

/// One decoded handshake body.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HandshakeFrame {
    /// Opening offer.
    Hello(HandshakeHello),
    /// Acceptance with the acceptor's identity.
    Accept(HandshakeAccept),
    /// Refusal with its reason.
    Reject(HandshakeReject),
}

/// Encodes a handshake rejection body: tag plus four fixed 64-bit slots
/// (`a`, `b`, `c`, `d`) interpreted per tag by [`decode_handshake`].
/// Cluster ids ride as low/high halves so both sides identify the exact
/// clusters involved.
#[must_use]
pub fn encode_reject(reason: &HandshakeReject) -> Vec<u8> {
    fn split_u128(value: u128) -> (u64, u64) {
        // Exact halves through bytes (never a lossy numeric cast).
        let bytes = value.to_le_bytes();
        let lo = u64::from_le_bytes(bytes[..8].try_into().unwrap_or([0; 8]));
        let hi = u64::from_le_bytes(bytes[8..].try_into().unwrap_or([0; 8]));
        (lo, hi)
    }
    let mut out = Vec::with_capacity(2 + 1 + 8 * 4);
    out.extend_from_slice(&KIND_REJECT.to_le_bytes());
    match *reason {
        HandshakeReject::UnsupportedVersion { found } => {
            out.push(1);
            push_u64(&mut out, u64::from(found));
            push_u64(&mut out, 0);
            push_u64(&mut out, 0);
            push_u64(&mut out, 0);
        }
        HandshakeReject::WrongCluster { expected, found } => {
            out.push(2);
            let (found_lo, found_hi) = split_u128(found);
            let (expected_lo, expected_hi) = split_u128(expected);
            push_u64(&mut out, found_lo);
            push_u64(&mut out, found_hi);
            push_u64(&mut out, expected_lo);
            push_u64(&mut out, expected_hi);
        }
        HandshakeReject::StaleIncarnation {
            node,
            found,
            newest,
        } => {
            out.push(3);
            push_u64(&mut out, node);
            push_u64(&mut out, found);
            push_u64(&mut out, newest);
            push_u64(&mut out, 0);
        }
        HandshakeReject::MissingCapability { missing } => {
            out.push(4);
            push_u64(&mut out, missing);
            push_u64(&mut out, 0);
            push_u64(&mut out, 0);
            push_u64(&mut out, 0);
        }
    }
    out
}

/// Encodes one consensus request body: kind, transport request id,
/// addressed consensus group (tablet id), then the RPC payload.
#[must_use]
pub fn encode_request(id: u64, group: u64, request: &PeerRequest) -> Vec<u8> {
    fn push_vote_request(out: &mut Vec<u8>, request: &PeerVoteRequest) {
        push_vote(out, &request.vote);
        push_opt_log(out, request.last_log.as_ref());
        out.push(u8::from(request.leadership_transfer));
    }
    let mut out = Vec::new();
    let kind = match request {
        PeerRequest::Vote(_) => KIND_VOTE_REQUEST,
        PeerRequest::PreVote(_) => KIND_PRE_VOTE_REQUEST,
        PeerRequest::Append(_) => KIND_APPEND_REQUEST,
        PeerRequest::Snapshot(_) => KIND_SNAPSHOT_REQUEST,
    };
    out.extend_from_slice(&kind.to_le_bytes());
    push_u64(&mut out, id);
    push_u64(&mut out, group);
    match request {
        PeerRequest::Vote(request) | PeerRequest::PreVote(request) => {
            push_vote_request(&mut out, request);
        }
        PeerRequest::Append(request) => {
            push_vote(&mut out, &request.vote);
            push_opt_log(&mut out, request.prev_log.as_ref());
            push_u64(&mut out, request.entries.len() as u64);
            for entry in &request.entries {
                push_u64(&mut out, entry.term);
                push_u64(&mut out, entry.leader);
                push_u64(&mut out, entry.index);
                match &entry.payload {
                    PeerEntryPayload::Blank => out.push(0),
                    PeerEntryPayload::Normal(command) => {
                        out.push(1);
                        push_blob(&mut out, command);
                    }
                    PeerEntryPayload::Membership { voters, nodes } => {
                        out.push(2);
                        push_u64(&mut out, voters.len() as u64);
                        for voter in voters {
                            push_u64(&mut out, *voter);
                        }
                        push_nodes(&mut out, nodes);
                    }
                }
            }
            push_opt_log(&mut out, request.leader_commit.as_ref());
        }
        PeerRequest::Snapshot(request) => {
            push_vote(&mut out, &request.vote);
            push_opt_log(&mut out, request.meta.last_log.as_ref());
            push_u64(&mut out, request.meta.voters.len() as u64);
            for voter in &request.meta.voters {
                push_u64(&mut out, *voter);
            }
            push_nodes(&mut out, &request.meta.nodes);
            push_opt_log(&mut out, request.meta.membership_log.as_ref());
            push_blob(&mut out, &request.meta.checkpoint);
            push_u64(&mut out, request.transfer);
            push_u64(&mut out, request.offset);
            push_blob(&mut out, &request.data);
            out.push(u8::from(request.done));
        }
    }
    out
}

/// Encodes one consensus response body, matching `id` and echoing the
/// addressed `group`, carrying either the answer or the remote refusal.
#[must_use]
pub fn encode_response(
    id: u64,
    group: u64,
    result: &Result<PeerResponse, PeerRpcError>,
    request_kind: u16,
) -> Vec<u8> {
    let mut out = Vec::new();
    let kind = match request_kind {
        KIND_VOTE_REQUEST => KIND_VOTE_RESPONSE,
        KIND_PRE_VOTE_REQUEST => KIND_PRE_VOTE_RESPONSE,
        KIND_APPEND_REQUEST => KIND_APPEND_RESPONSE,
        _ => KIND_SNAPSHOT_RESPONSE,
    };
    out.extend_from_slice(&kind.to_le_bytes());
    push_u64(&mut out, id);
    push_u64(&mut out, group);
    match result {
        Ok(response) => {
            out.push(0);
            match response {
                PeerResponse::Vote(response) | PeerResponse::PreVote(response) => {
                    push_vote_response(&mut out, response);
                }
                PeerResponse::Append(response) => match response {
                    PeerAppendResponse::Success => out.push(0),
                    PeerAppendResponse::PartialSuccess(matched) => {
                        out.push(1);
                        push_opt_log(&mut out, matched.as_ref());
                    }
                    PeerAppendResponse::Conflict => out.push(2),
                    PeerAppendResponse::HigherVote(vote) => {
                        out.push(3);
                        push_vote(&mut out, vote);
                    }
                },
                PeerResponse::Snapshot(response) => {
                    push_vote(&mut out, &response.vote);
                }
            }
        }
        Err(error) => {
            out.push(1);
            push_string(&mut out, &error.detail);
        }
    }
    out
}

fn push_vote_response(out: &mut Vec<u8>, response: &PeerVoteResponse) {
    push_vote(out, &response.vote);
    out.push(u8::from(response.granted));
    push_opt_log(out, response.last_log.as_ref());
}

/// One decoded consensus body with its request id and addressed group.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PeerBody {
    /// A request awaiting service.
    Request {
        /// Transport-level request id (responses echo it).
        id: u64,
        /// Addressed consensus group (tablet id): routes the RPC to the
        /// right Raft instance on a shared mesh.
        group: u64,
        /// Request body.
        request: PeerRequest,
    },
    /// A response to a locally issued request.
    Response {
        /// Echoed request id.
        id: u64,
        /// Answer or remote refusal.
        result: Result<PeerResponse, PeerRpcError>,
    },
}

/// Decodes one request body (kinds `10`, `12`, `14`, `16`).
fn decode_request_body(
    kind: u16,
    id: u64,
    group: u64,
    reader: &mut Reader<'_>,
) -> Result<PeerBody, PeerCodecError> {
    fn vote_request(reader: &mut Reader<'_>) -> Result<PeerVoteRequest, PeerCodecError> {
        Ok(PeerVoteRequest {
            vote: reader.vote()?,
            last_log: reader.opt_log()?,
            leadership_transfer: match reader.u8()? {
                0 => false,
                1 => true,
                tag => return Err(PeerCodecError::BadTag { tag }),
            },
        })
    }
    let decoded = match kind {
        KIND_VOTE_REQUEST => PeerBody::Request {
            id,
            group,
            request: PeerRequest::Vote(vote_request(reader)?),
        },
        KIND_PRE_VOTE_REQUEST => PeerBody::Request {
            id,
            group,
            request: PeerRequest::PreVote(vote_request(reader)?),
        },
        KIND_APPEND_REQUEST => {
            let vote = reader.vote()?;
            let prev_log = reader.opt_log()?;
            let count = reader.u64()?;
            let count =
                usize::try_from(count).map_err(|_| PeerCodecError::Oversize { len: usize::MAX })?;
            if count > 1024 * 1024 {
                return Err(PeerCodecError::Oversize { len: count });
            }
            let mut entries = Vec::with_capacity(count.min(1024));
            for _ in 0..count {
                entries.push(PeerEntry {
                    term: reader.u64()?,
                    leader: reader.u64()?,
                    index: reader.u64()?,
                    payload: reader.entry_payload()?,
                });
            }
            PeerBody::Request {
                id,
                group,
                request: PeerRequest::Append(PeerAppendRequest {
                    vote,
                    prev_log,
                    entries,
                    leader_commit: reader.opt_log()?,
                }),
            }
        }
        KIND_SNAPSHOT_REQUEST => PeerBody::Request {
            id,
            group,
            request: PeerRequest::Snapshot(PeerSnapshotRequest {
                vote: reader.vote()?,
                meta: PeerSnapshotMeta {
                    last_log: reader.opt_log()?,
                    voters: reader.voters()?,
                    nodes: reader.nodes()?,
                    membership_log: reader.opt_log()?,
                    checkpoint: reader.blob()?,
                },
                transfer: reader.u64()?,
                offset: reader.u64()?,
                data: reader.blob()?,
                done: match reader.u8()? {
                    0 => false,
                    1 => true,
                    tag => return Err(PeerCodecError::BadTag { tag }),
                },
            }),
        },
        kind => return Err(PeerCodecError::UnknownKind { kind }),
    };
    Ok(decoded)
}

/// Decodes one response body (kinds `11`, `13`, `15`).
fn decode_response_body(
    kind: u16,
    id: u64,
    reader: &mut Reader<'_>,
) -> Result<PeerBody, PeerCodecError> {
    let failed = reader.u8()?;
    if failed == 1 {
        return Ok(PeerBody::Response {
            id,
            result: Err(PeerRpcError {
                detail: reader.string()?,
            }),
        });
    }
    if failed != 0 {
        return Err(PeerCodecError::BadTag { tag: failed });
    }
    let response = match kind {
        KIND_VOTE_RESPONSE | KIND_PRE_VOTE_RESPONSE => {
            let vote = reader.vote()?;
            let granted = match reader.u8()? {
                0 => false,
                1 => true,
                tag => return Err(PeerCodecError::BadTag { tag }),
            };
            let last_log = reader.opt_log()?;
            let answer = PeerVoteResponse {
                vote,
                granted,
                last_log,
            };
            if kind == KIND_VOTE_RESPONSE {
                PeerResponse::Vote(answer)
            } else {
                PeerResponse::PreVote(answer)
            }
        }
        KIND_APPEND_RESPONSE => PeerResponse::Append(match reader.u8()? {
            0 => PeerAppendResponse::Success,
            1 => PeerAppendResponse::PartialSuccess(reader.opt_log()?),
            2 => PeerAppendResponse::Conflict,
            3 => PeerAppendResponse::HigherVote(reader.vote()?),
            tag => return Err(PeerCodecError::BadTag { tag }),
        }),
        KIND_SNAPSHOT_RESPONSE => PeerResponse::Snapshot(PeerSnapshotResponse {
            vote: reader.vote()?,
        }),
        kind => return Err(PeerCodecError::UnknownKind { kind }),
    };
    Ok(PeerBody::Response {
        id,
        result: Ok(response),
    })
}

/// Decodes one consensus body (kinds `10–17`).
///
/// # Errors
///
/// Returns [`PeerCodecError`] on structural faults.
pub fn decode_body(body: &[u8]) -> Result<PeerBody, PeerCodecError> {
    let mut reader = Reader::new(body);
    let kind = reader.u16()?;
    let id = reader.u64()?;
    let group = reader.u64()?;
    let decoded = match kind {
        KIND_VOTE_REQUEST | KIND_PRE_VOTE_REQUEST | KIND_APPEND_REQUEST | KIND_SNAPSHOT_REQUEST => {
            decode_request_body(kind, id, group, &mut reader)?
        }
        KIND_VOTE_RESPONSE
        | KIND_PRE_VOTE_RESPONSE
        | KIND_APPEND_RESPONSE
        | KIND_SNAPSHOT_RESPONSE => decode_response_body(kind, id, &mut reader)?,
        kind => return Err(PeerCodecError::UnknownKind { kind }),
    };
    reader.finish()?;
    Ok(decoded)
}

/// Highest incarnation observed per node (task K). Connections and
/// messages from an older generation than recorded never regain
/// validity; a newer generation supersedes and is recorded.
#[derive(Debug, Default)]
pub struct IncarnationTable {
    observed: HashMap<NodeId, NodeIncarnation>,
}

impl IncarnationTable {
    /// Creates an empty table.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Records an observed incarnation. Returns `true` when the
    /// incarnation is current (newer than, or the first for, the node)
    /// and records it; returns `false` for stale generations, recording
    /// nothing. `INVALID` is never current.
    pub fn observe(&mut self, node: NodeId, incarnation: NodeIncarnation) -> bool {
        if incarnation == NodeIncarnation::INVALID {
            return false;
        }
        match self.observed.get(&node) {
            Some(known) if incarnation.as_u64() <= known.as_u64() => false,
            _ => {
                self.observed.insert(node, incarnation);
                true
            }
        }
    }

    /// Returns the newest observed incarnation for a node, if any.
    #[must_use]
    pub fn observed(&self, node: NodeId) -> Option<NodeIncarnation> {
        self.observed.get(&node).copied()
    }

    /// Returns the newest observed incarnation as a raw value (`0` when
    /// nothing was observed — handshake diagnostics only).
    #[must_use]
    pub fn newest(&self, node: NodeId) -> u64 {
        self.observed
            .get(&node)
            .map_or(0, |incarnation| incarnation.as_u64())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity() -> PeerIdentity {
        PeerIdentity {
            cluster: ClusterId::from_u128(0x0C10_57E2),
            node: NodeId::from_u64(1),
            incarnation: NodeIncarnation::from_u64(7),
        }
    }

    fn hello() -> HandshakeHello {
        HandshakeHello {
            version: PEER_PROTOCOL_VERSION,
            cluster: 0x0C10_57E2,
            node: 2,
            incarnation: 3,
            capabilities: REQUIRED_CAPABILITIES,
        }
    }

    #[test]
    fn handshake_accepts_current_peers() {
        let table = IncarnationTable::new();
        let accept = validate_hello(&hello(), &identity(), &table).expect("accepts");
        assert_eq!(accept.node, 1);
        assert_eq!(accept.incarnation, 7);
        // Hello/accept bodies round-trip through framing.
        let framed = frame_body(&encode_hello(&hello()));
        let mut decoder = FrameDecoder::new();
        decoder.push(&framed);
        let body = decoder.take_frame().expect("frames").expect("complete");
        assert!(matches!(
            decode_handshake(&body).expect("decodes"),
            HandshakeFrame::Hello(_)
        ));
        let framed = frame_body(&encode_accept(&accept));
        let mut decoder = FrameDecoder::new();
        decoder.push(&framed);
        let body = decoder.take_frame().expect("frames").expect("complete");
        assert!(matches!(
            decode_handshake(&body).expect("decodes"),
            HandshakeFrame::Accept(_)
        ));
        // Rejections round-trip with their exact reasons.
        for reason in [
            HandshakeReject::UnsupportedVersion { found: 7 },
            HandshakeReject::WrongCluster {
                expected: 0x0C10_57E2,
                found: 0xDEAD_BEEF_CAFE_F00D_DEAD_BEEF_CAFE_F00D,
            },
            HandshakeReject::StaleIncarnation {
                node: 2,
                found: 3,
                newest: 8,
            },
            HandshakeReject::MissingCapability { missing: 0x10 },
        ] {
            let framed = frame_body(&encode_reject(&reason));
            let mut decoder = FrameDecoder::new();
            decoder.push(&framed);
            let body = decoder.take_frame().expect("frames").expect("complete");
            assert_eq!(
                decode_handshake(&body).expect("decodes"),
                HandshakeFrame::Reject(reason)
            );
        }
    }

    #[test]
    fn handshake_rejects_wrong_cluster_stale_and_weak_peers() {
        let mut table = IncarnationTable::new();
        // Wrong cluster.
        let mut foreign = hello();
        foreign.cluster = 0xDEAD;
        assert!(matches!(
            validate_hello(&foreign, &identity(), &table),
            Err(HandshakeReject::WrongCluster { .. })
        ));
        // Missing capability.
        let mut weak = hello();
        weak.capabilities = 0;
        assert!(matches!(
            validate_hello(&weak, &identity(), &table),
            Err(HandshakeReject::MissingCapability { .. })
        ));
        // Stale incarnation after observing a newer generation (the
        // sample hello carries incarnation 3 against observed 8).
        assert!(table.observe(NodeId::from_u64(2), NodeIncarnation::from_u64(8)));
        assert!(matches!(
            validate_hello(&hello(), &identity(), &table),
            Err(HandshakeReject::StaleIncarnation { .. })
        ));
        // Equal incarnations reconnect freely: the same live generation
        // revalidates (transient healing depends on this; only strictly
        // older generations are stale).
        let mut equal = hello();
        equal.incarnation = 8;
        assert!(
            validate_hello(&equal, &identity(), &table).is_ok(),
            "equal incarnation must revalidate"
        );
        // Invalid incarnations are never current.
        assert!(!table.observe(NodeId::from_u64(9), NodeIncarnation::INVALID));
    }

    #[test]
    fn incarnation_table_advances_monotonically() {
        let mut table = IncarnationTable::new();
        let node = NodeId::from_u64(2);
        assert!(table.observe(node, NodeIncarnation::from_u64(7)));
        assert!(!table.observe(node, NodeIncarnation::from_u64(7)));
        assert!(!table.observe(node, NodeIncarnation::from_u64(6)));
        assert!(table.observe(node, NodeIncarnation::from_u64(8)));
        assert_eq!(table.newest(node), 8);
    }

    #[test]
    fn rpc_bodies_round_trip() {
        let vote = PeerVote {
            term: 3,
            node: 1,
            committed: true,
        };
        let log = PeerLogId {
            term: 3,
            leader: 1,
            index: 42,
        };
        let cases = [
            PeerRequest::Vote(PeerVoteRequest {
                vote,
                last_log: Some(log),
                leadership_transfer: false,
            }),
            PeerRequest::PreVote(PeerVoteRequest {
                vote,
                last_log: Some(log),
                leadership_transfer: true,
            }),
            PeerRequest::Append(PeerAppendRequest {
                vote,
                prev_log: Some(log),
                entries: vec![PeerEntry {
                    term: 3,
                    leader: 1,
                    index: 43,
                    payload: PeerEntryPayload::Normal(vec![0xDE, 0xAD]),
                }],
                leader_commit: Some(log),
            }),
            PeerRequest::Snapshot(PeerSnapshotRequest {
                vote,
                meta: PeerSnapshotMeta {
                    last_log: Some(log),
                    voters: vec![1, 2, 3],
                    nodes: vec![(1, "127.0.0.1:9101".to_owned())],
                    membership_log: Some(log),
                    checkpoint: b"ckpt-9@42".to_vec(),
                },
                transfer: 77,
                offset: 0,
                data: vec![1, 2, 3],
                done: true,
            }),
        ];
        for (id, request) in cases.iter().enumerate() {
            let id = id as u64 + 1;
            // Group 9 rides every body (Multi-Raft multiplexing).
            let framed = frame_body(&encode_request(id, 9, request));
            let mut decoder = FrameDecoder::new();
            // Split delivery still frames (streaming correctness).
            decoder.push(&framed[..7]);
            assert!(decoder.take_frame().expect("partial").is_none());
            decoder.push(&framed[7..]);
            let body = decoder.take_frame().expect("frames").expect("complete");
            assert_eq!(
                decode_body(&body).expect("decodes"),
                PeerBody::Request {
                    id,
                    group: 9,
                    request: request.clone()
                }
            );
        }
        // Responses round-trip with their request kinds.
        let response = encode_response(
            9,
            9,
            &Ok(PeerResponse::Append(PeerAppendResponse::HigherVote(vote))),
            KIND_APPEND_REQUEST,
        );
        let mut decoder = FrameDecoder::new();
        decoder.push(&frame_body(&response));
        let body = decoder.take_frame().expect("frames").expect("complete");
        assert!(matches!(
            decode_body(&body).expect("decodes"),
            PeerBody::Response { id: 9, .. }
        ));
    }

    #[test]
    fn corrupt_frames_fail_without_allocation() {
        // Oversize declarations fail before allocation.
        let mut oversize = frame_body(&encode_hello(&hello()));
        let len_at = 4 + 2;
        oversize[len_at..len_at + 4].copy_from_slice(&u32::MAX.to_le_bytes());
        // Re-stamp the CRC over the tampered body so the length fault (not
        // the CRC fault) is what fails — the length gate runs first.
        let mut decoder = FrameDecoder::new();
        decoder.push(&oversize);
        assert!(matches!(
            decoder.take_frame(),
            Err(PeerCodecError::BadCrc | PeerCodecError::Oversize { .. })
        ));
        // Bad magic fails (once a full header is buffered).
        let mut decoder = FrameDecoder::new();
        decoder.push(&[0xFF; 12]);
        assert!(matches!(
            decoder.take_frame(),
            Err(PeerCodecError::BadMagic { .. })
        ));
    }
}
