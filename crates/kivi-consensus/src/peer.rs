//! Kivi peer message vocabulary over QUIC/H3.
//!
//! Peer traffic runs on the QUIC/H3 mesh ([`crate::transport`]) — never the
//! native client protocol, RESP, or admin HTTP. HTTP/3 provides
//! multiplexing, framing, flow control, and stream lifecycle; this module
//! owns only Kivi semantics:
//!
//! * peer identity ([`PeerIdentity`]: cluster, node, incarnation) carried
//!   in authenticated H3 headers and validated per request (wrong cluster,
//!   stale incarnation, and missing capabilities refuse loudly);
//! * Raft RPC payloads as Kivi-owned binary structs (never `OpenRaft`
//!   memory layout — [`crate::router`] owns translation both ways, so
//!   `OpenRaft` upgrades cannot change these bytes);
//! * immutable sidecar identities (`ChunkId`/`ManifestId` over content,
//!   never pack offsets).
//!
//! Every RPC addresses its consensus group
//! ([`ConsensusGroupId`](crate::types::ConsensusGroupId), as its tablet
//! id) in the H3 path: one mesh multiplexes many Raft groups (Multi-Raft
//! direction), and the group id routes each RPC to the right instance.
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

/// Required capability: consensus RPCs (election/append/snapshot).
pub const CAP_CONSENSUS_V2: u64 = 1 << 0;
/// Required capability: immutable sidecar bulk transfer (manifest/chunk).
pub const CAP_SIDECAR_V1: u64 = 1 << 1;
/// All capabilities this binary requires of its peers.
pub const REQUIRED_CAPABILITIES: u64 = CAP_CONSENSUS_V2 | CAP_SIDECAR_V1;
/// Maximum bulk payload bytes per frame (chunk + framing, well under the
/// 16 MiB peer ceiling; chunk bodies never exceed 8 MiB by construction).
pub const MAX_BULK_FRAME_BYTES: usize = 8 * 1024 * 1024;

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

/// Why a peer request was refused at the authentication layer (wrong
/// cluster, stale incarnation, missing capability, suspended link).
/// Refusals fail only the request (mapped to retry-safe transport
/// errors), never the connection — H3 streams are cheap.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum AuthReject {
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

/// Validated Kivi peer identity for one request: the claimed headers, with
/// the TLS certificate already bound to a configured peer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ValidatedPeer {
    /// Claimed node.
    pub node: NodeId,
    /// Claimed incarnation.
    pub incarnation: NodeIncarnation,
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

/// Owned immutable manifest request (bulk lane).
///
/// Requests the canonical manifest bytes for `manifest` by content
/// identity. The responder serves from its local sidecar store; physical
/// pack offsets never cross the network.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PeerManifestRequest {
    /// Requested manifest id.
    pub manifest: [u8; 32],
}

/// Owned immutable manifest response (bulk lane).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PeerManifestResponse {
    /// Echoed manifest id.
    pub manifest: [u8; 32],
    /// Canonical manifest bytes (verified by the requester).
    pub canonical: Vec<u8>,
}

/// Owned immutable chunk request (bulk lane).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PeerChunkRequest {
    /// Requested chunk id.
    pub chunk: [u8; 32],
}

/// Owned immutable chunk response (bulk lane).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PeerChunkResponse {
    /// Echoed chunk id.
    pub chunk: [u8; 32],
    /// Logical chunk bytes (verified by the requester).
    pub bytes: Vec<u8>,
}

/// Owned sidecar preflight request (bulk lane).
///
/// Names one immutable root by content identity plus its logical length.
/// The responder ensures the manifest and every referenced chunk are
/// locally durable and verified before replying.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PeerPrepareRequest {
    /// Root manifest id.
    pub manifest: [u8; 32],
    /// Logical value length in bytes (sizes the acquisition).
    pub logical_len: u64,
}

/// Owned sidecar preflight response (bulk lane).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PeerPreparedResponse {
    /// Echoed manifest id (durable locally).
    pub manifest: [u8; 32],
}

/// Framing version of [`PeerLeaseBatch`] bodies. Version 2 adds the
/// grantor committed index to renewals (floor input closing the
/// establishment gap); version 1 decoders refuse it loudly instead of
/// misreading positions.
pub const LEASE_BATCH_VERSION: u16 = 2;

/// Maximum lease-protocol messages per batch (a tick emits at most one
/// message per live pairing; the cap only bounds a corrupt peer).
pub const LEASE_BATCH_CAP: usize = 256;

/// Maximum responders in one decoded roster (voter sets are small; the cap
/// only bounds a corrupt peer, never a real roster).
pub const LEASE_ROSTER_CAP: usize = 4096;

/// One lease-protocol batch: the outbound traffic of one engine step,
/// in emission order. Requests carry live traffic; responses piggyback
/// the replies the receiver's engine step produced (every message
/// typically answers at most once, so piggybacking halves the packet
/// count; second-order messages ride the next tick, which converges in a
/// fixed small number of rounds when healthy).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PeerLeaseBatch {
    /// Lease messages, in emission order.
    pub messages: Vec<kivi_types::LeaseMessage>,
}

/// Lazy-ALR sync request: a follower (or the leader for its own batch)
/// asks the leader to order a boundary after batch formation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PeerAlrSyncRequest {
    /// Batch fence identity (tablet, batch, requester).
    pub fence_tablet: u64,
    /// Batch sequence within the tablet.
    pub batch: u64,
    /// Replica that formed the batch.
    pub requester: u64,
    /// Former applied commit position at batch formation (same-log
    /// coordinates; `0` before the first apply).
    pub formation_applied: u64,
}

/// Lazy-ALR sync answer: the ordered boundary to wait for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PeerAlrSyncResponse {
    /// Raft log index whose local apply covers the batch: either the
    /// appended fence or a subsuming write ordered after formation.
    pub boundary: u64,
    /// Whether a concurrent write subsumed the extra fence entry.
    pub subsumed: bool,
}

/// Responder-coverage poll: the leader asks one active responder whether
/// it applied through `commit` and can serve from it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PeerCoveragePoll {
    /// Committed position the write needs covered (inclusive).
    pub commit: u64,
}

/// Responder-coverage report: the answering replica's applied pointer
/// plus whether it can serve the logical contract from it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PeerCoverageReport {
    /// Reporting replica.
    pub responder: u64,
    /// Process lifetime that applied the state.
    pub incarnation: u64,
    /// Applied commit position (inclusive).
    pub applied: u64,
    /// Whether the replica can serve from `applied` (chunked roots
    /// resolve; no latched storage fault).
    pub healthy: bool,
}

/// One peer RPC. H3 request streams carry these without any further
/// transport envelope: the route selects the family, the body is the
/// payload codec below. Pre-vote reuses the vote shape under its own
/// route. Manifest/chunk sidecar RPCs (and sidecar preflight) ride the
/// bulk H3 connection so bulk transfer never starves critical traffic.
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
    /// Immutable manifest fetch (bulk lane).
    Manifest(PeerManifestRequest),
    /// Immutable chunk fetch (bulk lane).
    Chunk(PeerChunkRequest),
    /// Sidecar preflight: "prepare immutable root M" (bulk lane). The
    /// follower acquires, verifies, and durably installs the manifest plus
    /// all referenced chunks through the same acquisition primitives as
    /// the append gate, replying only once durable. Preflight mutates no
    /// logical state and creates no Raft decision; it is purely an
    /// optimization moving sidecar movement before the tiny root proposal.
    Prepare(PeerPrepareRequest),
    /// Roster-lease traffic batch (control lane): Guard/Renew/Revoke and
    /// their replies for one tablet group. Served on the owner thread
    /// through the lease engine, never through Raft itself.
    Lease(PeerLeaseBatch),
    /// Lazy-ALR sync request (control lane): order a read boundary after
    /// batch formation. Served on the leader's owner thread only;
    /// followers answer refusal.
    AlrSync(PeerAlrSyncRequest),
    /// Responder-coverage poll (control lane): "have you applied through
    /// this commit and can you serve from it". Served on the owner thread
    /// from the state machine's applied pointer.
    CoveragePoll(PeerCoveragePoll),
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
    /// Immutable manifest answer (bulk lane).
    Manifest(PeerManifestResponse),
    /// Immutable chunk answer (bulk lane).
    Chunk(PeerChunkResponse),
    /// Sidecar preflight answer (bulk lane): the root is durable locally.
    Prepared(PeerPreparedResponse),
    /// Roster-lease traffic answer (control lane): the replies the
    /// receiver's engine step produced for the requested messages.
    Lease(PeerLeaseBatch),
    /// Lazy-ALR sync answer (control lane): the ordered boundary.
    AlrSync(PeerAlrSyncResponse),
    /// Responder-coverage answer (control lane): applied pointer plus
    /// servability.
    Coverage(PeerCoverageReport),
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

/// Structural payload-codec failure (never a remote verdict). QUIC/TLS
/// already guarantee transport integrity; these faults mean a corrupt or
/// version-skewed peer body.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum PeerCodecError {
    /// Fewer bytes than the payload promises.
    #[error("truncated peer payload")]
    Truncated,
    /// A length declaration exceeds the body cap or the input.
    #[error("oversize peer declaration {len}")]
    Oversize {
        /// Declared length.
        len: usize,
    },
    /// Unknown route suffix.
    #[error("unknown peer route {route}")]
    UnknownRoute {
        /// Observed route suffix.
        route: String,
    },
    /// A payload tag is unknown.
    #[error("unknown peer payload tag {tag}")]
    BadTag {
        /// Observed tag.
        tag: u8,
    },
    /// Non-UTF8 address string.
    #[error("non-UTF8 peer string")]
    BadString,
    /// Trailing bytes after the payload body.
    #[error("trailing bytes in peer payload")]
    TrailingBytes,
    /// A hex identity in the route is malformed.
    #[error("malformed peer identity hex: {detail}")]
    BadHex {
        /// Human-readable cause.
        detail: String,
    },
    /// A decoded roster identity is not fully valid (bad authority, term,
    /// or generation half). The engine would refuse it; the codec refuses
    /// it first so no invalid roster ever enters engine state.
    #[error("invalid roster identity in lease payload")]
    InvalidRoster,
    /// A framing version this binary cannot speak.
    #[error("unsupported peer payload version {found}")]
    UnsupportedVersion {
        /// Observed version.
        found: u16,
    },
}

fn push_u64(out: &mut Vec<u8>, value: u64) {
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

pub(crate) struct Reader<'a> {
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

    fn u64(&mut self) -> Result<u64, PeerCodecError> {
        Ok(u64::from_le_bytes(
            self.bytes(8)?.try_into().unwrap_or([0; 8]),
        ))
    }

    fn blob(&mut self) -> Result<Vec<u8>, PeerCodecError> {
        self.bulk_blob(crate::transport::MAX_RAFT_BODY_BYTES)
    }

    fn string(&mut self) -> Result<String, PeerCodecError> {
        let blob = self.blob()?;
        String::from_utf8(blob).map_err(|_| PeerCodecError::BadString)
    }

    fn bulk_blob(&mut self, max: usize) -> Result<Vec<u8>, PeerCodecError> {
        let len = u32::from_le_bytes(self.bytes(4)?.try_into().unwrap_or([0; 4])) as usize;
        if len > max || len > self.rest() {
            return Err(PeerCodecError::Oversize { len });
        }
        Ok(self.bytes(len)?.to_vec())
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

    fn id32(&mut self) -> Result<[u8; 32], PeerCodecError> {
        Ok(self.bytes(32)?.try_into().unwrap_or([0; 32]))
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

/// Validates authenticated peer headers for one request: cluster match,
/// live incarnation, required capabilities.
///
/// Only strictly OLDER generations are stale: an equal incarnation is the
/// same live process generation reconnecting (transient drops and
/// suspend/resume heal through this path), while an older one is a
/// restarted-away generation that must never regain validity. `INVALID`
/// is never valid anywhere.
///
/// # Errors
///
/// Returns [`AuthReject`] for cluster, incarnation, or capability
/// mismatches. Every rejection fails only the request.
pub fn validate_peer_headers(
    cluster: u128,
    node: u64,
    incarnation: u64,
    capabilities: u64,
    local: &PeerIdentity,
    table: &IncarnationTable,
) -> Result<ValidatedPeer, AuthReject> {
    use AuthReject as Fault;
    if cluster != local.cluster.as_u128() {
        return Err(Fault::WrongCluster {
            expected: local.cluster.as_u128(),
            found: cluster,
        });
    }
    let node_id = NodeId::from_u64(node);
    if incarnation == NodeIncarnation::INVALID.as_u64() {
        return Err(Fault::StaleIncarnation {
            node,
            found: incarnation,
            newest: table.newest(node_id),
        });
    }
    if let Some(newest) = table.observed(node_id)
        && incarnation < newest.as_u64()
    {
        return Err(Fault::StaleIncarnation {
            node,
            found: incarnation,
            newest: newest.as_u64(),
        });
    }
    let missing = REQUIRED_CAPABILITIES & !capabilities;
    if missing != 0 {
        return Err(Fault::MissingCapability { missing });
    }
    Ok(ValidatedPeer {
        node: node_id,
        incarnation: NodeIncarnation::from_u64(incarnation),
    })
}

/// Renders 32 identity bytes as lowercase hex for immutable paths.
#[must_use]
pub fn id32_hex(id: &[u8; 32]) -> String {
    let mut out = String::with_capacity(64);
    for byte in id {
        use std::fmt::Write as _;
        let _ = write!(out, "{byte:02x}");
    }
    out
}

/// Parses lowercase (or upper) hex back into 32 identity bytes.
///
/// # Errors
///
/// Returns [`PeerCodecError::BadHex`] on length or digit faults.
pub fn parse_id32_hex(hex: &str) -> Result<[u8; 32], PeerCodecError> {
    let bad = |detail: String| PeerCodecError::BadHex { detail };
    if hex.len() != 64 {
        return Err(bad(format!("expected 64 hex chars, got {}", hex.len())));
    }
    let mut out = [0u8; 32];
    for (i, byte) in out.iter_mut().enumerate() {
        let pair = hex
            .get(2 * i..2 * i + 2)
            .ok_or_else(|| bad("truncated hex".to_owned()))?;
        *byte =
            u8::from_str_radix(pair, 16).map_err(|_| bad(format!("invalid hex pair {pair:?}")))?;
    }
    Ok(out)
}

/// H3 route suffixes. Raft routes live under `/_kivi/raft/{tablet}/`;
/// immutable routes under `/_kivi/immutable/`.
pub mod route {
    /// Election RPC route suffix.
    pub const VOTE: &str = "vote";
    /// Pre-vote probe route suffix.
    pub const PREVOTE: &str = "prevote";
    /// Replication/heartbeat route suffix.
    pub const APPEND: &str = "append";
    /// Snapshot-fragment route suffix.
    pub const SNAPSHOT_FRAG: &str = "snapshot-frag";
    /// Sidecar-preflight route suffix (bulk lane, per-group Raft route).
    pub const PREPARE: &str = "prepare";
    /// Roster-lease batch route suffix (control lane, per-group route).
    pub const LEASE: &str = "lease";
    /// Lazy-ALR sync route suffix (control lane, per-group route).
    pub const ALR_SYNC: &str = "alr-sync";
    /// Responder-coverage poll route suffix (control lane, per-group route).
    pub const COVERAGE: &str = "coverage";
}

/// Renders the H3 path for one request under `group`'s tablet. Manifest
/// and chunk identities ride the path as lowercase hex; every other
/// family carries its payload codec as the body.
#[must_use]
pub fn h3_request_path(group_tablet: u64, request: &PeerRequest) -> String {
    match request {
        PeerRequest::Vote(_) => format!("/_kivi/raft/{group_tablet}/{}", route::VOTE),
        PeerRequest::PreVote(_) => format!("/_kivi/raft/{group_tablet}/{}", route::PREVOTE),
        PeerRequest::Append(_) => format!("/_kivi/raft/{group_tablet}/{}", route::APPEND),
        PeerRequest::Snapshot(_) => {
            format!("/_kivi/raft/{group_tablet}/{}", route::SNAPSHOT_FRAG)
        }
        PeerRequest::Prepare(_) => {
            format!("/_kivi/raft/{group_tablet}/{}", route::PREPARE)
        }
        PeerRequest::Lease(_) => {
            format!("/_kivi/raft/{group_tablet}/{}", route::LEASE)
        }
        PeerRequest::AlrSync(_) => {
            format!("/_kivi/raft/{group_tablet}/{}", route::ALR_SYNC)
        }
        PeerRequest::CoveragePoll(_) => {
            format!("/_kivi/raft/{group_tablet}/{}", route::COVERAGE)
        }
        PeerRequest::Manifest(request) => {
            format!("/_kivi/immutable/manifest/{}", id32_hex(&request.manifest))
        }
        PeerRequest::Chunk(request) => {
            format!("/_kivi/immutable/chunk/{}", id32_hex(&request.chunk))
        }
    }
}

/// Media type for a request family (transport metadata only, never
/// durable state).
#[must_use]
pub const fn h3_media_type(request: &PeerRequest) -> &'static str {
    match request {
        PeerRequest::Vote(_)
        | PeerRequest::PreVote(_)
        | PeerRequest::Append(_)
        | PeerRequest::Snapshot(_) => "application/vnd.kivi.raft",
        PeerRequest::Prepare(_) => "application/vnd.kivi.prepare",
        PeerRequest::Lease(_) | PeerRequest::AlrSync(_) | PeerRequest::CoveragePoll(_) => {
            "application/vnd.kivi.read"
        }
        PeerRequest::Manifest(_) => "application/vnd.kivi.manifest",
        PeerRequest::Chunk(_) => "application/vnd.kivi.chunk",
    }
}

/// HTTP method for a request family (transport metadata only, never
/// durable state). Identity-in-path bulk fetches are empty-body GETs;
/// every other family — Raft RPCs and the sidecar-preflight POST, which
/// carries its manifest identity plus logical length as a small body — is
/// a POST. A GET with a body is never emitted: intermediaries may drop
/// the body, which would corrupt preflight into an undecodable request.
#[must_use]
pub fn h3_method(request: &PeerRequest) -> http::Method {
    match request {
        PeerRequest::Manifest(_) | PeerRequest::Chunk(_) => http::Method::GET,
        PeerRequest::Vote(_)
        | PeerRequest::PreVote(_)
        | PeerRequest::Append(_)
        | PeerRequest::Snapshot(_)
        | PeerRequest::Prepare(_)
        | PeerRequest::Lease(_)
        | PeerRequest::AlrSync(_)
        | PeerRequest::CoveragePoll(_) => http::Method::POST,
    }
}

/// Encodes one request as its H3 `(path, body)`. Manifest/chunk requests
/// have empty bodies (identity is the path); every other family encodes
/// its payload codec.
///
/// Returns the path plus body bytes.
#[must_use]
pub fn encode_h3_request(request: &PeerRequest, group_tablet: u64) -> (String, Vec<u8>) {
    fn push_vote_request(out: &mut Vec<u8>, request: &PeerVoteRequest) {
        push_vote(out, &request.vote);
        push_opt_log(out, request.last_log.as_ref());
        out.push(u8::from(request.leadership_transfer));
    }
    let path = h3_request_path(group_tablet, request);
    let mut out = Vec::new();
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
        PeerRequest::Prepare(request) => {
            out.extend_from_slice(&request.manifest);
            push_u64(&mut out, request.logical_len);
        }
        PeerRequest::Lease(batch) => {
            out.extend_from_slice(&LEASE_BATCH_VERSION.to_le_bytes());
            push_u64(&mut out, batch.messages.len() as u64);
            for message in &batch.messages {
                encode_lease_message(&mut out, message);
            }
        }
        PeerRequest::AlrSync(request) => {
            push_u64(&mut out, request.fence_tablet);
            push_u64(&mut out, request.batch);
            push_u64(&mut out, request.requester);
            push_u64(&mut out, request.formation_applied);
        }
        PeerRequest::CoveragePoll(request) => {
            push_u64(&mut out, request.commit);
        }
        PeerRequest::Manifest(_) | PeerRequest::Chunk(_) => {}
    }
    (path, out)
}

/// Decoded H3 request: addressed tablet plus the RPC.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodedH3Request {
    /// Addressed tablet id.
    pub tablet: u64,
    /// Request body.
    pub request: PeerRequest,
}

/// Decodes one H3 request path plus body.
///
/// # Errors
///
/// Returns [`PeerCodecError`] on unknown routes, malformed identities, or
/// structural payload faults.
pub fn decode_h3_request(path: &str, body: &[u8]) -> Result<DecodedH3Request, PeerCodecError> {
    let mut reader = Reader::new(body);
    let (tablet, request) = if let Some(rest) = path.strip_prefix("/_kivi/raft/") {
        let (tablet, suffix) =
            rest.split_once('/')
                .ok_or_else(|| PeerCodecError::UnknownRoute {
                    route: path.to_owned(),
                })?;
        let tablet: u64 = tablet.parse().map_err(|_| PeerCodecError::UnknownRoute {
            route: path.to_owned(),
        })?;
        let request = match suffix {
            route::VOTE => PeerRequest::Vote(decode_vote_request(&mut reader)?),
            route::PREVOTE => PeerRequest::PreVote(decode_vote_request(&mut reader)?),
            route::APPEND => PeerRequest::Append(decode_append_request(&mut reader)?),
            route::SNAPSHOT_FRAG => PeerRequest::Snapshot(decode_snapshot_request(&mut reader)?),
            route::PREPARE => PeerRequest::Prepare(decode_prepare_request(&mut reader)?),
            route::LEASE => PeerRequest::Lease(decode_lease_batch(&mut reader)?),
            route::ALR_SYNC => PeerRequest::AlrSync(PeerAlrSyncRequest {
                fence_tablet: reader.u64()?,
                batch: reader.u64()?,
                requester: reader.u64()?,
                formation_applied: reader.u64()?,
            }),
            route::COVERAGE => PeerRequest::CoveragePoll(PeerCoveragePoll {
                commit: reader.u64()?,
            }),
            _ => {
                return Err(PeerCodecError::UnknownRoute {
                    route: path.to_owned(),
                });
            }
        };
        reader.finish()?;
        (tablet, request)
    } else if let Some(hex) = path.strip_prefix("/_kivi/immutable/manifest/") {
        reader.finish()?;
        (
            0,
            PeerRequest::Manifest(PeerManifestRequest {
                manifest: parse_id32_hex(hex)?,
            }),
        )
    } else if let Some(hex) = path.strip_prefix("/_kivi/immutable/chunk/") {
        reader.finish()?;
        (
            0,
            PeerRequest::Chunk(PeerChunkRequest {
                chunk: parse_id32_hex(hex)?,
            }),
        )
    } else {
        return Err(PeerCodecError::UnknownRoute {
            route: path.to_owned(),
        });
    };
    Ok(DecodedH3Request { tablet, request })
}

/// Encodes one response body for the family the request selected.
/// Manifest/chunk responses are the raw bytes (identity already rode the
/// request path); Raft responses use the payload codec.
#[must_use]
pub fn encode_h3_response(response: &PeerResponse) -> Vec<u8> {
    let mut out = Vec::new();
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
        PeerResponse::Prepared(response) => {
            out.extend_from_slice(&response.manifest);
        }
        PeerResponse::Lease(batch) => {
            out.extend_from_slice(&LEASE_BATCH_VERSION.to_le_bytes());
            push_u64(&mut out, batch.messages.len() as u64);
            for message in &batch.messages {
                encode_lease_message(&mut out, message);
            }
        }
        PeerResponse::AlrSync(response) => {
            push_u64(&mut out, response.boundary);
            out.push(u8::from(response.subsumed));
        }
        PeerResponse::Coverage(response) => {
            push_u64(&mut out, response.responder);
            push_u64(&mut out, response.incarnation);
            push_u64(&mut out, response.applied);
            out.push(u8::from(response.healthy));
        }
        PeerResponse::Manifest(response) => {
            out.extend_from_slice(&response.canonical);
        }
        PeerResponse::Chunk(response) => {
            out.extend_from_slice(&response.bytes);
        }
    }
    out
}

/// Decodes one response body for the family the request selected.
///
/// # Errors
///
/// Returns [`PeerCodecError`] on structural faults or family mismatch.
pub fn decode_h3_response(
    request: &PeerRequest,
    body: &[u8],
) -> Result<PeerResponse, PeerCodecError> {
    let mut reader = Reader::new(body);
    let response = match request {
        PeerRequest::Vote(_) => PeerResponse::Vote(decode_vote_response(&mut reader)?),
        PeerRequest::PreVote(_) => PeerResponse::PreVote(decode_vote_response(&mut reader)?),
        PeerRequest::Append(_) => PeerResponse::Append(match reader.u8()? {
            0 => PeerAppendResponse::Success,
            1 => PeerAppendResponse::PartialSuccess(reader.opt_log()?),
            2 => PeerAppendResponse::Conflict,
            3 => PeerAppendResponse::HigherVote(reader.vote()?),
            tag => return Err(PeerCodecError::BadTag { tag }),
        }),
        PeerRequest::Snapshot(_) => PeerResponse::Snapshot(PeerSnapshotResponse {
            vote: reader.vote()?,
        }),
        PeerRequest::Prepare(_) => {
            let manifest = reader.id32()?;
            PeerResponse::Prepared(PeerPreparedResponse { manifest })
        }
        PeerRequest::Lease(_) => PeerResponse::Lease(decode_lease_batch(&mut reader)?),
        PeerRequest::AlrSync(_) => PeerResponse::AlrSync(PeerAlrSyncResponse {
            boundary: reader.u64()?,
            subsumed: match reader.u8()? {
                0 => false,
                1 => true,
                tag => return Err(PeerCodecError::BadTag { tag }),
            },
        }),
        PeerRequest::CoveragePoll(_) => PeerResponse::Coverage(PeerCoverageReport {
            responder: reader.u64()?,
            incarnation: reader.u64()?,
            applied: reader.u64()?,
            healthy: match reader.u8()? {
                0 => false,
                1 => true,
                tag => return Err(PeerCodecError::BadTag { tag }),
            },
        }),
        PeerRequest::Manifest(request) => PeerResponse::Manifest(PeerManifestResponse {
            manifest: request.manifest,
            canonical: body.to_vec(),
        }),
        PeerRequest::Chunk(request) => PeerResponse::Chunk(PeerChunkResponse {
            chunk: request.chunk,
            bytes: body.to_vec(),
        }),
    };
    // Manifest/chunk/prepare-ack bodies are fixed-shape and must consume
    // exactly (prepare acks echo 32 manifest bytes); only manifest/chunk
    // bulk payloads are raw variable bytes.
    if !matches!(request, PeerRequest::Manifest(_) | PeerRequest::Chunk(_)) {
        reader.finish()?;
    }
    Ok(response)
}

fn decode_vote_request(reader: &mut Reader<'_>) -> Result<PeerVoteRequest, PeerCodecError> {
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

fn decode_vote_response(reader: &mut Reader<'_>) -> Result<PeerVoteResponse, PeerCodecError> {
    let vote = reader.vote()?;
    let granted = match reader.u8()? {
        0 => false,
        1 => true,
        tag => return Err(PeerCodecError::BadTag { tag }),
    };
    let last_log = reader.opt_log()?;
    Ok(PeerVoteResponse {
        vote,
        granted,
        last_log,
    })
}

fn decode_append_request(reader: &mut Reader<'_>) -> Result<PeerAppendRequest, PeerCodecError> {
    let vote = reader.vote()?;
    let prev_log = reader.opt_log()?;
    let count = reader.u64()?;
    let count = usize::try_from(count).map_err(|_| PeerCodecError::Oversize { len: usize::MAX })?;
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
    Ok(PeerAppendRequest {
        vote,
        prev_log,
        entries,
        leader_commit: reader.opt_log()?,
    })
}

fn decode_prepare_request(reader: &mut Reader<'_>) -> Result<PeerPrepareRequest, PeerCodecError> {
    Ok(PeerPrepareRequest {
        manifest: reader.id32()?,
        logical_len: reader.u64()?,
    })
}

/// Decodes one lease batch body (request or response direction).
///
/// # Errors
///
/// Returns [`PeerCodecError`] on version mismatch, truncation, oversize
/// declarations, undecodable messages, or trailing bytes.
fn decode_lease_batch(reader: &mut Reader<'_>) -> Result<PeerLeaseBatch, PeerCodecError> {
    let version = u16::from_le_bytes(reader.bytes(2)?.try_into().unwrap_or([0; 2]));
    if version != LEASE_BATCH_VERSION {
        return Err(PeerCodecError::UnsupportedVersion { found: version });
    }
    let count = reader.u64()?;
    let count = usize::try_from(count).map_err(|_| PeerCodecError::Oversize { len: usize::MAX })?;
    if count > LEASE_BATCH_CAP {
        return Err(PeerCodecError::Oversize { len: count });
    }
    let mut messages = Vec::with_capacity(count.min(16));
    for _ in 0..count {
        messages.push(decode_lease_message(reader)?);
    }
    Ok(PeerLeaseBatch { messages })
}

/// Encodes one lease-protocol message: tag `u8` plus fixed `u64` fields
/// (roster content carries a count-prefixed responder vec, capped on
/// decode). Structural only — attempt identity, incarnation binding, and
/// roster ordering are engine checks, never codec checks.
pub fn encode_lease_message(out: &mut Vec<u8>, message: &kivi_types::LeaseMessage) {
    use kivi_types::LeaseMessage as M;
    fn push_roster_id(out: &mut Vec<u8>, id: kivi_types::RosterId) {
        push_u64(out, id.authority.tablet().as_u64());
        push_u64(out, id.authority.epoch().as_u64());
        push_u64(out, id.authority.guard().as_u64());
        push_u64(out, id.term.as_u64());
        push_u64(out, id.generation.as_u64());
    }
    fn push_attempt(out: &mut Vec<u8>, attempt: kivi_types::LeaseAttemptId) {
        push_u64(out, attempt.grantor_incarnation.as_u64());
        push_u64(out, attempt.seq);
    }
    match message {
        M::Guard(guard) => {
            out.push(0);
            push_roster_id(out, guard.roster.id);
            push_u64(out, guard.roster.leader.as_u64());
            push_u64(out, guard.roster.responders.len() as u64);
            for responder in &guard.roster.responders {
                push_u64(out, responder.as_u64());
            }
            push_u64(out, guard.thresh_accepted.as_u64());
            push_attempt(out, guard.attempt);
            push_u64(out, guard.seq);
            push_u64(out, guard.grantor.as_u64());
        }
        M::GuardReply(reply) => {
            out.push(1);
            push_roster_id(out, reply.roster_id);
            push_attempt(out, reply.attempt);
            push_u64(out, reply.seq);
            push_u64(out, reply.grantee.as_u64());
            push_u64(out, reply.grantor.as_u64());
        }
        M::Renew(renew) => {
            out.push(2);
            push_roster_id(out, renew.roster_id);
            push_attempt(out, renew.attempt);
            push_u64(out, renew.seq);
            push_u64(out, renew.committed.as_u64());
            push_u64(out, renew.grantor.as_u64());
        }
        M::RenewReply(reply) => {
            out.push(3);
            push_roster_id(out, reply.roster_id);
            push_attempt(out, reply.attempt);
            push_u64(out, reply.seq);
            push_u64(out, reply.grantee.as_u64());
            push_u64(out, reply.grantor.as_u64());
        }
        M::Revoke(revoke) => {
            out.push(4);
            push_roster_id(out, revoke.roster_id);
            push_attempt(out, revoke.attempt);
            push_u64(out, revoke.seq);
            push_u64(out, revoke.grantor.as_u64());
        }
        M::RevokeReply(reply) => {
            out.push(5);
            push_roster_id(out, reply.roster_id);
            push_attempt(out, reply.attempt);
            push_u64(out, reply.seq);
            push_u64(out, reply.grantee.as_u64());
            push_u64(out, reply.grantor.as_u64());
        }
    }
}

/// Decodes one lease-protocol message.
///
/// # Errors
///
/// Returns [`PeerCodecError`] on truncation, unknown tags, oversize
/// responder sets, or trailing structure. Unknown roster identities are
/// accepted structurally — the engine (which owns authority, term, and
/// generation validity) refuses what it must.
pub(crate) fn decode_lease_message(
    reader: &mut Reader<'_>,
) -> Result<kivi_types::LeaseMessage, PeerCodecError> {
    use kivi_types::{
        CommitPosition, LeaseAttemptId, LeaseGuard, LeaseGuardReply, LeaseMessage as M, LeaseRenew,
        LeaseRenewReply, LeaseRevoke, LeaseRevokeReply, RosterId, RosterTerm,
    };
    fn read_roster_id(reader: &mut Reader<'_>) -> Result<RosterId, PeerCodecError> {
        use kivi_types::{TabletAuthority, TabletEpoch, TabletId, WriteGuardGeneration};
        Ok(RosterId::new(
            TabletAuthority::new(
                TabletId::from_u64(reader.u64()?),
                TabletEpoch::from_u64(reader.u64()?),
                WriteGuardGeneration::from_u64(reader.u64()?),
            ),
            RosterTerm::from_u64(reader.u64()?),
            kivi_types::RosterGeneration::from_u64(reader.u64()?),
        ))
    }
    fn read_attempt(reader: &mut Reader<'_>) -> Result<LeaseAttemptId, PeerCodecError> {
        Ok(LeaseAttemptId::new(
            kivi_types::NodeIncarnation::from_u64(reader.u64()?),
            reader.u64()?,
        ))
    }
    Ok(match reader.u8()? {
        0 => {
            let id = read_roster_id(reader)?;
            let leader = NodeId::from_u64(reader.u64()?);
            let count = reader.u64()?;
            let count =
                usize::try_from(count).map_err(|_| PeerCodecError::Oversize { len: usize::MAX })?;
            if count > LEASE_ROSTER_CAP {
                return Err(PeerCodecError::Oversize { len: count });
            }
            let mut responders = Vec::with_capacity(count.min(64));
            for _ in 0..count {
                responders.push(NodeId::from_u64(reader.u64()?));
            }
            let roster = kivi_types::Roster::new(id, leader, responders)
                .map_err(|_| PeerCodecError::InvalidRoster)?;
            M::Guard(LeaseGuard {
                roster,
                thresh_accepted: CommitPosition::from_u64(reader.u64()?),
                attempt: read_attempt(reader)?,
                seq: reader.u64()?,
                grantor: NodeId::from_u64(reader.u64()?),
            })
        }
        1 => M::GuardReply(LeaseGuardReply {
            roster_id: read_roster_id(reader)?,
            attempt: read_attempt(reader)?,
            seq: reader.u64()?,
            grantee: NodeId::from_u64(reader.u64()?),
            grantor: NodeId::from_u64(reader.u64()?),
        }),
        2 => M::Renew(LeaseRenew {
            roster_id: read_roster_id(reader)?,
            attempt: read_attempt(reader)?,
            seq: reader.u64()?,
            committed: kivi_types::CommitPosition::from_u64(reader.u64()?),
            grantor: NodeId::from_u64(reader.u64()?),
        }),
        3 => M::RenewReply(LeaseRenewReply {
            roster_id: read_roster_id(reader)?,
            attempt: read_attempt(reader)?,
            seq: reader.u64()?,
            grantee: NodeId::from_u64(reader.u64()?),
            grantor: NodeId::from_u64(reader.u64()?),
        }),
        4 => M::Revoke(LeaseRevoke {
            roster_id: read_roster_id(reader)?,
            attempt: read_attempt(reader)?,
            seq: reader.u64()?,
            grantor: NodeId::from_u64(reader.u64()?),
        }),
        5 => M::RevokeReply(LeaseRevokeReply {
            roster_id: read_roster_id(reader)?,
            attempt: read_attempt(reader)?,
            seq: reader.u64()?,
            grantee: NodeId::from_u64(reader.u64()?),
            grantor: NodeId::from_u64(reader.u64()?),
        }),
        tag => return Err(PeerCodecError::BadTag { tag }),
    })
}

fn decode_snapshot_request(reader: &mut Reader<'_>) -> Result<PeerSnapshotRequest, PeerCodecError> {
    Ok(PeerSnapshotRequest {
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
    })
}

fn push_vote_response(out: &mut Vec<u8>, response: &PeerVoteResponse) {
    push_vote(out, &response.vote);
    out.push(u8::from(response.granted));
    push_opt_log(out, response.last_log.as_ref());
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

    /// Sample authenticated headers: (cluster, node, incarnation, caps).
    fn headers() -> (u128, u64, u64, u64) {
        (0x0C10_57E2, 2, 3, REQUIRED_CAPABILITIES)
    }

    #[test]
    fn headers_accept_current_peers() {
        let table = IncarnationTable::new();
        let (cluster, node, incarnation, caps) = headers();
        let validated =
            validate_peer_headers(cluster, node, incarnation, caps, &identity(), &table)
                .expect("accepts");
        assert_eq!(validated.node, NodeId::from_u64(2));
        assert_eq!(validated.incarnation, NodeIncarnation::from_u64(3));
        // Rejections carry their exact reasons.
        for (cluster, node, incarnation, caps) in [
            (
                0xDEAD_BEEF_CAFE_F00D_DEAD_BEEF_CAFE_F00D,
                2,
                3,
                REQUIRED_CAPABILITIES,
            ),
            (0x0C10_57E2, 2, 3, 0),
        ] {
            assert!(
                validate_peer_headers(cluster, node, incarnation, caps, &identity(), &table)
                    .is_err()
            );
        }
    }

    #[test]
    fn headers_reject_wrong_cluster_stale_and_weak_peers() {
        let mut table = IncarnationTable::new();
        let (cluster, node, incarnation, caps) = headers();
        // Wrong cluster.
        assert!(matches!(
            validate_peer_headers(0xDEAD, node, incarnation, caps, &identity(), &table),
            Err(AuthReject::WrongCluster { .. })
        ));
        // Missing capability.
        assert!(matches!(
            validate_peer_headers(cluster, node, incarnation, 0, &identity(), &table),
            Err(AuthReject::MissingCapability { .. })
        ));
        // Stale incarnation after observing a newer generation.
        assert!(table.observe(NodeId::from_u64(2), NodeIncarnation::from_u64(8)));
        assert!(matches!(
            validate_peer_headers(cluster, node, incarnation, caps, &identity(), &table),
            Err(AuthReject::StaleIncarnation { .. })
        ));
        // Equal incarnations reconnect freely: the same live generation
        // revalidates (transient healing depends on this; only strictly
        // older generations are stale).
        assert!(
            validate_peer_headers(cluster, node, 8, caps, &identity(), &table).is_ok(),
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
    #[allow(clippy::too_many_lines)]
    fn h3_payloads_round_trip() {
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
            PeerRequest::Manifest(PeerManifestRequest {
                manifest: [0x11; 32],
            }),
            PeerRequest::Chunk(PeerChunkRequest { chunk: [0x22; 32] }),
            PeerRequest::Prepare(PeerPrepareRequest {
                manifest: [0x33; 32],
                logical_len: 1234,
            }),
            PeerRequest::Lease(PeerLeaseBatch {
                messages: vec![lease_guard_fixture(), lease_renew_fixture()],
            }),
            PeerRequest::AlrSync(PeerAlrSyncRequest {
                fence_tablet: 9,
                batch: 17,
                requester: 2,
                formation_applied: 41,
            }),
            PeerRequest::CoveragePoll(PeerCoveragePoll { commit: 42 }),
        ];
        for request in &cases {
            // Tablet 9 rides every Raft path (Multi-Raft multiplexing).
            let (path, body) = encode_h3_request(request, 9);
            let decoded = decode_h3_request(&path, &body).expect("decodes");
            match request {
                PeerRequest::Manifest(_) | PeerRequest::Chunk(_) => {
                    assert_eq!(decoded.tablet, 0, "sidecars are group-independent");
                    assert_eq!(
                        h3_method(request),
                        http::Method::GET,
                        "fetches are empty-body GETs"
                    );
                    assert!(body.is_empty(), "GETs carry no body");
                }
                _ => {
                    assert_eq!(decoded.tablet, 9);
                    assert_eq!(
                        h3_method(request),
                        http::Method::POST,
                        "bodied families are POSTs"
                    );
                }
            }
            assert_eq!(&decoded.request, request);
            // Truncated bodies fail the decode (backoff), never apply.
            // Command-level validation (undecodable `ConsensusCommand`)
            // lives in the router codec, tested there.
            if body.len() > 4 {
                assert!(decode_h3_request(&path, &body[..body.len() - 1]).is_err());
            }
        }
        // Responses round-trip under the requesting family.
        let cases = [
            (
                &cases[2],
                PeerResponse::Append(PeerAppendResponse::HigherVote(vote)),
            ),
            (
                &cases[0],
                PeerResponse::Vote(PeerVoteResponse {
                    vote,
                    granted: true,
                    last_log: Some(log),
                }),
            ),
            (
                &cases[7],
                PeerResponse::Lease(PeerLeaseBatch {
                    messages: vec![lease_guard_reply_fixture()],
                }),
            ),
            (
                &cases[8],
                PeerResponse::AlrSync(PeerAlrSyncResponse {
                    boundary: 43,
                    subsumed: true,
                }),
            ),
            (
                &cases[9],
                PeerResponse::Coverage(PeerCoverageReport {
                    responder: 2,
                    incarnation: 7,
                    applied: 42,
                    healthy: true,
                }),
            ),
        ];
        for (request, response) in cases {
            let body = encode_h3_response(&response);
            assert_eq!(
                decode_h3_response(request, &body).expect("decodes"),
                response
            );
        }
        // Raw sidecar bodies echo byte-for-byte.
        let manifest_request = PeerRequest::Manifest(PeerManifestRequest {
            manifest: [0x11; 32],
        });
        let manifest = PeerResponse::Manifest(PeerManifestResponse {
            manifest: [0x11; 32],
            canonical: vec![7; 78],
        });
        assert_eq!(
            decode_h3_response(&manifest_request, &encode_h3_response(&manifest)).expect("decodes"),
            manifest
        );
        // Preflight acks echo the prepared manifest identity.
        let prepare_request = PeerRequest::Prepare(PeerPrepareRequest {
            manifest: [0x33; 32],
            logical_len: 1234,
        });
        let prepared = PeerResponse::Prepared(PeerPreparedResponse {
            manifest: [0x33; 32],
        });
        assert_eq!(
            decode_h3_response(&prepare_request, &encode_h3_response(&prepared)).expect("decodes"),
            prepared
        );
        // Unknown routes refuse loudly.
        assert!(matches!(
            decode_h3_request("/_kivi/nope", &[]),
            Err(PeerCodecError::UnknownRoute { .. })
        ));
        assert!(matches!(
            decode_h3_request("/_kivi/immutable/chunk/zzz", &[]),
            Err(PeerCodecError::BadHex { .. })
        ));
        // Identity hex round-trips (paths embed lowercase hex).
        assert_eq!(id32_hex(&[0xABu8; 32]), "ab".repeat(32));
        assert_eq!(
            parse_id32_hex(&id32_hex(&[0xCD; 32])).expect("parses"),
            [0xCD; 32]
        );
    }

    fn lease_roster_fixture() -> kivi_types::Roster {
        use kivi_types::{
            RosterGeneration, RosterId, RosterTerm, TabletAuthority, TabletEpoch, TabletId,
            WriteGuardGeneration,
        };
        let id = RosterId::new(
            TabletAuthority::new(
                TabletId::from_u64(9),
                TabletEpoch::from_u64(3),
                WriteGuardGeneration::from_u64(7),
            ),
            RosterTerm::from_u64(5),
            RosterGeneration::INITIAL,
        );
        kivi_types::Roster::new(
            id,
            NodeId::from_u64(1),
            vec![NodeId::from_u64(2), NodeId::from_u64(3)],
        )
        .expect("valid fixture roster")
    }

    fn lease_attempt_fixture() -> kivi_types::LeaseAttemptId {
        kivi_types::LeaseAttemptId::new(NodeIncarnation::from_u64(4), 11)
    }

    fn lease_guard_fixture() -> kivi_types::LeaseMessage {
        kivi_types::LeaseMessage::Guard(kivi_types::LeaseGuard {
            roster: lease_roster_fixture(),
            thresh_accepted: kivi_types::CommitPosition::from_u64(41),
            attempt: lease_attempt_fixture(),
            seq: 11,
            grantor: NodeId::from_u64(1),
        })
    }

    fn lease_guard_reply_fixture() -> kivi_types::LeaseMessage {
        kivi_types::LeaseMessage::GuardReply(kivi_types::LeaseGuardReply {
            roster_id: lease_roster_fixture().id,
            attempt: lease_attempt_fixture(),
            seq: 11,
            grantee: NodeId::from_u64(2),
            grantor: NodeId::from_u64(1),
        })
    }

    fn lease_renew_fixture() -> kivi_types::LeaseMessage {
        kivi_types::LeaseMessage::Renew(kivi_types::LeaseRenew {
            roster_id: lease_roster_fixture().id,
            attempt: lease_attempt_fixture(),
            seq: 12,
            committed: kivi_types::CommitPosition::from_u64(40),
            grantor: NodeId::from_u64(1),
        })
    }

    #[test]
    fn lease_messages_round_trip_through_batch_codec() {
        for message in [
            lease_guard_fixture(),
            lease_guard_reply_fixture(),
            lease_renew_fixture(),
            kivi_types::LeaseMessage::RenewReply(kivi_types::LeaseRenewReply {
                roster_id: lease_roster_fixture().id,
                attempt: lease_attempt_fixture(),
                seq: 12,
                grantee: NodeId::from_u64(2),
                grantor: NodeId::from_u64(1),
            }),
            kivi_types::LeaseMessage::Revoke(kivi_types::LeaseRevoke {
                roster_id: lease_roster_fixture().id,
                attempt: lease_attempt_fixture(),
                seq: 13,
                grantor: NodeId::from_u64(1),
            }),
            kivi_types::LeaseMessage::RevokeReply(kivi_types::LeaseRevokeReply {
                roster_id: lease_roster_fixture().id,
                attempt: lease_attempt_fixture(),
                seq: 13,
                grantee: NodeId::from_u64(2),
                grantor: NodeId::from_u64(1),
            }),
        ] {
            let mut out = Vec::new();
            encode_lease_message(&mut out, &message);
            let mut reader = Reader::new(&out);
            assert_eq!(decode_lease_message(&mut reader).expect("decodes"), message);
            reader.finish().expect("exact");
            // Unknown tags refuse loudly (never misread as a live message).
            let mut tagged = out.clone();
            tagged[0] = 0x7F;
            assert!(matches!(
                decode_lease_message(&mut Reader::new(&tagged)),
                Err(PeerCodecError::BadTag { .. })
            ));
        }
        // An invalid roster identity (zero term) refuses at the codec, so
        // it never enters engine state.
        let mut out = Vec::new();
        encode_lease_message(&mut out, &lease_guard_fixture());
        // Corrupt the term half (bytes 2+8*3..2+8*4) to zero.
        out[26..34].fill(0);
        assert!(matches!(
            decode_lease_message(&mut Reader::new(&out)),
            Err(PeerCodecError::InvalidRoster)
        ));
    }

    #[test]
    fn lease_batch_rejects_version_and_oversize() {
        let batch = PeerLeaseBatch {
            messages: vec![lease_guard_fixture()],
        };
        let request = PeerRequest::Lease(batch);
        let (path, mut body) = encode_h3_request(&request, 9);
        assert_eq!(
            decode_h3_request(&path, &body).expect("decodes").request,
            request
        );
        // Version bump refuses loudly.
        body[0] = body[0].wrapping_add(1);
        assert!(matches!(
            decode_h3_request(&path, &body),
            Err(PeerCodecError::UnsupportedVersion { .. })
        ));
        // Declared responder floods refuse without allocating.
        let mut flood = Vec::new();
        flood.extend_from_slice(&LEASE_BATCH_VERSION.to_le_bytes());
        flood.extend_from_slice(&1u64.to_le_bytes());
        flood.push(0); // Guard tag
        // Valid roster id halves.
        for half in [9u64, 3, 7, 5, 1] {
            flood.extend_from_slice(&half.to_le_bytes());
        }
        flood.extend_from_slice(&1u64.to_le_bytes()); // leader
        flood.extend_from_slice(&u64::MAX.to_le_bytes()); // responder count
        let flood_request = PeerRequest::Lease(PeerLeaseBatch {
            messages: Vec::new(),
        });
        let flood_path = h3_request_path(9, &flood_request);
        assert!(matches!(
            decode_h3_request(&flood_path, &flood),
            Err(PeerCodecError::Oversize { .. })
        ));
    }
}
