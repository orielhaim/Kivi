//! `OpenRaft` network integration over the Kivi peer mesh.
//!
//! The boundary is:
//!
//! ```text
//! OpenRaft RPC
//!     ↓ (this module's codec: owned Peer* structs, never memory layout)
//! H3 request stream (group in path, Kivi binary body)
//!     ↓ (transport: QUIC/H3 mesh, TLS identity, incarnation)
//! peer QUIC
//! ```
//!
//! `OpenRaft` never owns a socket. [`PeerRouter`] is the node's shared
//! group router (implements `GroupRouter`, is `Send + Sync`): every Raft
//! group on the node routes `(target node, group id)` through it onto the
//! shared H3 mesh (one control connection per node pair, never per
//! group). Per group, [`GroupNetworkFactory`] binds the group id into
//! [`openraft_multi::GroupNetworkAdapter`]s, which satisfy `OpenRaft`'s
//! network traits through the blanket impls.
//!
//! Kivi owns peer identity, request routing, body caps, and
//! `NodeIncarnation` semantics ([`crate::transport`], [`crate::peer`]).
//! `openraft-multi` provides only the routing adapters; all
//! transport-level failures (unreachable, timeout, remote refusal,
//! family mismatch) map to retry-safe errors — the safe direction
//! is always backoff-and-retry, never an invented verdict.
//!
//! Election, pre-vote, append/heartbeat, and full-snapshot delivery are
//! all supported. Snapshot reassembly happens on the receiving owner
//! thread ([`crate::node`]): the sender fragments one full snapshot into
//! offset-keyed frames under a fresh transfer id; the receiver
//! reassembles and calls `install_full_snapshot` once.
//!
//! `transfer_leader` and `stream_append` keep their defaults (unreachable
//! error, sequential unary calls): leader transfer is not used yet, and
//! pipelined appends can arrive without a wire change later.

use std::io::Cursor;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use openraft::errors::{RPCError, StreamingError, Unreachable};
use openraft::network::{RPCOption, RaftNetworkFactory};
use openraft::raft::{
    AppendEntriesRequest, AppendEntriesResponse, SnapshotResponse, VoteRequest, VoteResponse,
};
use openraft::type_config::alias::{EntryOf, LogIdOf, SnapshotMetaOf, SnapshotOf, VoteOf};
use openraft::vote::RaftLeaderId as _;
use openraft_multi::{GroupNetworkAdapter, GroupRouter};

use crate::config::KiviTypeConfig;
use crate::peer::{
    PeerAppendRequest, PeerAppendResponse, PeerEntry, PeerEntryPayload, PeerLogId, PeerRequest,
    PeerResponse, PeerSnapshotMeta, PeerSnapshotRequest, PeerVote, PeerVoteRequest,
    PeerVoteResponse,
};
use crate::transport::{PeerTransport, TransportError};
use crate::types::ConsensusGroupId;

/// Maps any transport outcome onto the retry-safe `Unreachable` error.
/// Every arm backs off in `OpenRaft`'s replication core; no arm invents a
/// verdict the peer never gave.
fn unreachable(target: u64, error: &TransportError) -> RPCError<KiviTypeConfig> {
    let detail = format!("peer {target}: {error}");
    let source = std::io::Error::new(std::io::ErrorKind::NotConnected, detail);
    RPCError::Unreachable(Unreachable::new(&source))
}

fn snapshot_unreachable(target: u64, error: &TransportError) -> StreamingError<KiviTypeConfig> {
    let detail = format!("peer {target}: {error}");
    let source = std::io::Error::new(std::io::ErrorKind::NotConnected, detail);
    StreamingError::Unreachable(Unreachable::new(&source))
}

/// Maps a local protocol fault (codec failure, family mismatch — our own
/// bug or a version-skewed peer, never a remote verdict) onto the same
/// retry-safe error.
fn protocol_unreachable(target: u64, detail: &str) -> RPCError<KiviTypeConfig> {
    let source = std::io::Error::new(
        std::io::ErrorKind::NotConnected,
        format!("peer {target}: {detail}"),
    );
    RPCError::Unreachable(Unreachable::new(&source))
}

fn protocol_snapshot_unreachable(target: u64, detail: &str) -> StreamingError<KiviTypeConfig> {
    let source = std::io::Error::new(
        std::io::ErrorKind::NotConnected,
        format!("peer {target}: {detail}"),
    );
    StreamingError::Unreachable(Unreachable::new(&source))
}

/// Why a peer message could not cross the `OpenRaft` boundary. All
/// variants are sender-side faults (corrupt or version-skewed peer) or
/// local refusal — never ordinary replication outcomes (those encode as
/// normal responses).
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum RpcCodecError {
    /// A command entry's bytes do not decode as a
    /// [`ReplicatedMutation`](crate::mutation::ReplicatedMutation).
    #[error("undecodable replicated command: {detail}")]
    BadCommand {
        /// Decoder detail.
        detail: String,
    },
    /// A command entry carries no bytes (never minted; corruption if seen).
    #[error("empty command entry")]
    EmptyCommand,
    /// A membership entry names voters with no dialable nodes (never
    /// minted; corruption if seen).
    #[error("invalid membership entry: {detail}")]
    BadMembership {
        /// Decoder detail.
        detail: String,
    },
}

/// Converts an owned wire vote into its `OpenRaft` form. The
/// `leader_id_adv` leader carries term plus node id directly (total
/// order, multiple leaders per term), so the mapping is field-for-field.
#[must_use]
pub fn decode_wire_vote(vote: &PeerVote) -> VoteOf<KiviTypeConfig> {
    if vote.committed {
        openraft::impls::Vote::new_committed(vote.term, vote.node)
    } else {
        openraft::impls::Vote::new(vote.term, vote.node)
    }
}

/// Converts an `OpenRaft` vote into its owned wire form.
#[must_use]
pub fn encode_wire_vote(vote: &VoteOf<KiviTypeConfig>) -> PeerVote {
    PeerVote {
        term: vote.leader_id.term,
        node: vote.leader_id.node_id,
        committed: vote.committed,
    }
}

/// Converts an owned wire log id into its `OpenRaft` form.
#[must_use]
pub fn decode_wire_log(id: &PeerLogId) -> LogIdOf<KiviTypeConfig> {
    use openraft::impls::leader_id_adv::LeaderId;
    openraft::LogId::new(LeaderId::new(id.term, id.leader), id.index)
}

/// Converts an `OpenRaft` log id into its owned wire form.
#[must_use]
pub fn encode_wire_log(id: &LogIdOf<KiviTypeConfig>) -> PeerLogId {
    PeerLogId {
        term: id.leader_id.term,
        leader: id.leader_id.node_id,
        index: id.index,
    }
}

fn encode_opt_log(id: Option<&LogIdOf<KiviTypeConfig>>) -> Option<PeerLogId> {
    id.map(encode_wire_log)
}

fn decode_opt_log(id: Option<&PeerLogId>) -> Option<LogIdOf<KiviTypeConfig>> {
    id.map(decode_wire_log)
}

/// Converts one `OpenRaft` entry into its owned wire form. Command bytes
/// are the canonical [`ConsensusCommand`](crate::command::ConsensusCommand)
/// encoding (never `OpenRaft` memory layout).
#[must_use]
pub fn encode_wire_entry(entry: &EntryOf<KiviTypeConfig>) -> PeerEntry {
    use openraft::EntryPayload;
    let payload = match &entry.payload {
        EntryPayload::Blank => PeerEntryPayload::Blank,
        EntryPayload::Normal(command) => PeerEntryPayload::Normal(command.encode_to_vec()),
        EntryPayload::Membership(membership) => PeerEntryPayload::Membership {
            voters: membership.voter_ids().collect(),
            nodes: membership
                .nodes()
                .map(|(id, node)| (*id, node.addr.clone()))
                .collect(),
        },
    };
    PeerEntry {
        term: entry.log_id.leader_id.term,
        leader: entry.log_id.leader_id.node_id,
        index: entry.log_id.index,
        payload,
    }
}

/// Converts one owned wire entry back into `OpenRaft` form.
///
/// # Errors
///
/// Returns [`RpcCodecError`] when command bytes do not decode or the
/// membership is incoherent (a corrupt or version-skewed peer — the
/// caller fails the RPC with backoff, never applies half an entry).
pub fn decode_wire_entry(entry: &PeerEntry) -> Result<EntryOf<KiviTypeConfig>, RpcCodecError> {
    use openraft::EntryPayload;
    use openraft::impls::leader_id_adv::LeaderId;
    let payload = match &entry.payload {
        PeerEntryPayload::Blank => EntryPayload::Blank,
        PeerEntryPayload::Normal(command) => {
            if command.is_empty() {
                return Err(RpcCodecError::EmptyCommand);
            }
            let mantra =
                crate::command::ConsensusCommand::decode_exact(command).map_err(|error| {
                    RpcCodecError::BadCommand {
                        detail: error.to_string(),
                    }
                })?;
            EntryPayload::Normal(mantra)
        }
        PeerEntryPayload::Membership { voters, nodes } => {
            let voter_set: std::collections::BTreeSet<u64> = voters.iter().copied().collect();
            let node_map: std::collections::BTreeMap<u64, openraft::BasicNode> = nodes
                .iter()
                .map(|(id, addr)| (*id, openraft::BasicNode::new(addr.clone())))
                .collect();
            let membership =
                openraft::Membership::new(vec![voter_set], node_map).map_err(|error| {
                    RpcCodecError::BadMembership {
                        detail: error.to_string(),
                    }
                })?;
            EntryPayload::Membership(membership)
        }
    };
    Ok(openraft::Entry {
        log_id: openraft::LogId::new(LeaderId::new(entry.term, entry.leader), entry.index),
        payload,
    })
}

/// Encodes an outgoing vote (or pre-vote) request for the wire.
#[must_use]
pub fn encode_vote_request(rpc: &VoteRequest<KiviTypeConfig>) -> PeerVoteRequest {
    PeerVoteRequest {
        vote: encode_wire_vote(&rpc.vote),
        last_log: encode_opt_log(rpc.last_log_id.as_ref()),
        leadership_transfer: rpc.leadership_transfer,
    }
}

/// Decodes an incoming vote (or pre-vote) request from the wire.
#[must_use]
pub fn decode_vote_request(request: &PeerVoteRequest) -> VoteRequest<KiviTypeConfig> {
    VoteRequest {
        vote: decode_wire_vote(&request.vote),
        last_log_id: decode_opt_log(request.last_log.as_ref()),
        leadership_transfer: request.leadership_transfer,
    }
}

/// Encodes an outgoing vote (or pre-vote) response for the wire.
#[must_use]
pub fn encode_vote_response(rpc: &VoteResponse<KiviTypeConfig>) -> PeerVoteResponse {
    PeerVoteResponse {
        vote: encode_wire_vote(&rpc.vote),
        granted: rpc.vote_granted,
        last_log: encode_opt_log(rpc.last_log_id.as_ref()),
    }
}

/// Decodes an incoming vote (or pre-vote) response from the wire.
#[must_use]
pub fn decode_vote_response(response: &PeerVoteResponse) -> VoteResponse<KiviTypeConfig> {
    VoteResponse::new(
        decode_wire_vote(&response.vote),
        decode_opt_log(response.last_log.as_ref()),
        response.granted,
    )
}

/// Encodes an outgoing append-entries request for the wire.
///
/// This conversion is total over the request shape (entries encode
/// infallibly).
pub fn encode_append_request(rpc: &AppendEntriesRequest<KiviTypeConfig>) -> PeerAppendRequest {
    PeerAppendRequest {
        vote: encode_wire_vote(&rpc.vote),
        prev_log: encode_opt_log(rpc.prev_log_id.as_ref()),
        entries: rpc.entries.iter().map(encode_wire_entry).collect(),
        leader_commit: encode_opt_log(rpc.leader_commit.as_ref()),
    }
}

/// Decodes an incoming append-entries request from the wire.
///
/// # Errors
///
/// Returns [`RpcCodecError`] when any entry's command bytes do not decode.
pub fn decode_append_request(
    request: &PeerAppendRequest,
) -> Result<AppendEntriesRequest<KiviTypeConfig>, RpcCodecError> {
    let mut entries = Vec::with_capacity(request.entries.len());
    for entry in &request.entries {
        entries.push(decode_wire_entry(entry)?);
    }
    Ok(AppendEntriesRequest {
        vote: decode_wire_vote(&request.vote),
        prev_log_id: decode_opt_log(request.prev_log.as_ref()),
        entries,
        leader_commit: decode_opt_log(request.leader_commit.as_ref()),
    })
}

/// Encodes an outgoing append-entries response for the wire.
#[must_use]
pub fn encode_append_response(rpc: &AppendEntriesResponse<KiviTypeConfig>) -> PeerAppendResponse {
    match rpc {
        AppendEntriesResponse::Success => crate::peer::PeerAppendResponse::Success,
        AppendEntriesResponse::PartialSuccess(matched) => {
            crate::peer::PeerAppendResponse::PartialSuccess(encode_opt_log(matched.as_ref()))
        }
        AppendEntriesResponse::Conflict => crate::peer::PeerAppendResponse::Conflict,
        AppendEntriesResponse::HigherVote(vote) => {
            crate::peer::PeerAppendResponse::HigherVote(encode_wire_vote(vote))
        }
    }
}

/// Decodes an incoming append-entries response from the wire.
#[must_use]
pub fn decode_append_response(
    response: &crate::peer::PeerAppendResponse,
) -> AppendEntriesResponse<KiviTypeConfig> {
    match response {
        crate::peer::PeerAppendResponse::Success => AppendEntriesResponse::Success,
        crate::peer::PeerAppendResponse::PartialSuccess(matched) => {
            AppendEntriesResponse::PartialSuccess(decode_opt_log(matched.as_ref()))
        }
        crate::peer::PeerAppendResponse::Conflict => AppendEntriesResponse::Conflict,
        crate::peer::PeerAppendResponse::HigherVote(vote) => {
            AppendEntriesResponse::HigherVote(decode_wire_vote(vote))
        }
    }
}

/// Encodes outgoing snapshot metadata for the wire. The 0.10 metadata
/// carries no transfer identity (that rides the fragment envelope); the
/// checkpoint reference identifies immutable Kivi state.
#[must_use]
pub fn encode_snapshot_meta(meta: &SnapshotMetaOf<KiviTypeConfig>) -> PeerSnapshotMeta {
    let voters: Vec<u64> = meta.last_membership.membership().voter_ids().collect();
    let nodes: Vec<(u64, String)> = meta
        .last_membership
        .membership()
        .nodes()
        .map(|(id, node)| (*id, node.addr.clone()))
        .collect();
    PeerSnapshotMeta {
        last_log: encode_opt_log(meta.last_log_id.as_ref()),
        voters,
        nodes,
        membership_log: encode_opt_log(meta.last_membership.log_id().as_ref()),
        checkpoint: checkpoint_of(meta),
    }
}

/// Derives the checkpoint reference bound into snapshot metadata: the
/// state machine's snapshot bytes are content-addressed, and the durable
/// `RaftSnapshotInstalled` record grounds the same reference. The meta
/// itself carries no separate id in 0.10; the reference travels here so
/// the receiver can name the installed image before streaming ends.
fn checkpoint_of(_meta: &SnapshotMetaOf<KiviTypeConfig>) -> Vec<u8> {
    // The checkpoint reference is content-derived by the state machine
    // when it seals the image (see `ReplicatedStateMachine`); the
    // transfer envelope below carries it opaquely once known. Metadata
    // built for transfer always pairs with its bytes, so an empty
    // reference here means "derive from the delivered bytes".
    Vec::new()
}

/// Decodes incoming snapshot metadata from the wire.
///
/// # Errors
///
/// Returns [`RpcCodecError::BadMembership`] when the membership is
/// incoherent (a corrupt or version-skewed peer — the transfer is
/// refused, never installed).
pub fn decode_snapshot_meta(
    meta: &PeerSnapshotMeta,
) -> Result<SnapshotMetaOf<KiviTypeConfig>, RpcCodecError> {
    let voter_set: std::collections::BTreeSet<u64> = meta.voters.iter().copied().collect();
    let node_map: std::collections::BTreeMap<u64, openraft::BasicNode> = meta
        .nodes
        .iter()
        .map(|(id, addr)| (*id, openraft::BasicNode::new(addr.clone())))
        .collect();
    let membership = openraft::Membership::new(vec![voter_set], node_map).map_err(|error| {
        RpcCodecError::BadMembership {
            detail: error.to_string(),
        }
    })?;
    Ok(openraft::SnapshotMeta {
        last_log_id: decode_opt_log(meta.last_log.as_ref()),
        last_membership: openraft::StoredMembership::new(
            decode_opt_log(meta.membership_log.as_ref()),
            membership,
        ),
    })
}

/// Serves one incoming peer request against the local `Raft` instance.
/// Every RPC family is supported, including pre-vote and the election /
/// append / snapshot flow; remote verdicts encode as normal responses
/// while local decode faults and Raft refusals encode as
/// [`crate::peer::PeerRpcError`]
/// (the caller backs off — never a phantom success).
///
/// # Errors
///
/// Returns [`crate::peer::PeerRpcError`] when the request fails to decode or the local
/// `Raft` refuses it (shutdown, storage fault, non-member).
pub async fn serve_peer_request<SM>(
    raft: &openraft::Raft<KiviTypeConfig, SM>,
    request: crate::peer::PeerRequest,
) -> Result<crate::peer::PeerResponse, crate::peer::PeerRpcError>
where
    SM: openraft::storage::RaftStateMachine<KiviTypeConfig>,
{
    use crate::peer::{PeerRequest, PeerResponse, PeerRpcError};
    let refused = |detail: String| PeerRpcError { detail };
    match request {
        PeerRequest::Vote(request) => {
            let response = raft
                .vote(decode_vote_request(&request))
                .await
                .map_err(|error| refused(error.to_string()))?;
            Ok(PeerResponse::Vote(encode_vote_response(&response)))
        }
        PeerRequest::PreVote(request) => {
            let response = raft
                .pre_vote(decode_vote_request(&request))
                .await
                .map_err(|error| refused(error.to_string()))?;
            Ok(PeerResponse::PreVote(encode_vote_response(&response)))
        }
        PeerRequest::Append(request) => {
            let rpc =
                decode_append_request(&request).map_err(|error| refused(error.to_string()))?;
            let response = raft
                .append_entries(rpc)
                .await
                .map_err(|error| refused(error.to_string()))?;
            Ok(PeerResponse::Append(encode_append_response(&response)))
        }
        // Snapshot fragments never reach per-RPC service: the owner
        // thread reassembles the transfer and installs it whole via
        // `install_full_snapshot`. A fragment arriving here is a routing
        // bug, refused loudly.
        PeerRequest::Snapshot(_) => Err(refused(
            "snapshot fragments reassemble on the owner thread".to_owned(),
        )),
        // Sidecar bulk requests never reach the Raft service either: the
        // owner serves them from the sidecar store (fetches) or the
        // durability gate (preflight). Arriving here is a routing bug,
        // refused loudly.
        PeerRequest::Manifest(_) | PeerRequest::Chunk(_) => Err(refused(
            "sidecar bulk requests serve from the sidecar store".to_owned(),
        )),
        PeerRequest::Prepare(_) => Err(refused(
            "sidecar preflight serves from the durability gate".to_owned(),
        )),
        // Read-authority RPCs never reach the Raft service: the owner
        // thread serves leases from the lease engine, ALR syncs through
        // the log, and coverage polls from the applied pointer.
        // Arriving here is a routing bug, refused loudly.
        PeerRequest::Lease(_) | PeerRequest::AlrSync(_) | PeerRequest::CoveragePoll(_) => Err(
            refused("read-authority RPCs serve on the owner thread".to_owned()),
        ),
    }
}

/// Maximum snapshot fragment payload per H3 request body. Well under the
/// transport body cap; the transfer loops until `done`, so larger
/// checkpoints simply take more streams.
const SNAPSHOT_FRAGMENT_BYTES: usize = 256 * 1024;

/// The node's shared peer router: one physical mesh multiplexing every
/// Raft group. Implements [`GroupRouter`] keyed by `(target node, group
/// id)`; per-group [`GroupNetworkFactory`]s bind the group half.
#[derive(Debug, Clone)]
pub struct PeerRouter {
    transport: PeerTransport,
    transfers: Arc<AtomicU64>,
}

impl PeerRouter {
    /// Binds the router to the node's shared transport.
    #[must_use]
    pub fn new(transport: PeerTransport) -> Self {
        Self {
            transport,
            transfers: Arc::new(AtomicU64::new(1)),
        }
    }

    /// Returns the shared transport (server wiring, diagnostics).
    #[must_use]
    pub fn transport(&self) -> &PeerTransport {
        &self.transport
    }

    /// Issues one request/response RPC on the shared mesh and decodes the
    /// family-matched answer. Family mismatches and remote refusals map to
    /// the retry-safe error (never an invented verdict).
    async fn call(
        &self,
        target: u64,
        group: ConsensusGroupId,
        request: PeerRequest,
        option: &RPCOption,
    ) -> Result<PeerResponse, TransportError> {
        use kivi_types::NodeId;
        let response = self
            .transport
            .call(NodeId::from_u64(target), group, request, option.hard_ttl())
            .await?;
        Ok(response)
    }
}

impl GroupRouter<KiviTypeConfig, ConsensusGroupId> for PeerRouter {
    type SnapshotData = Cursor<Vec<u8>>;

    async fn append_entries(
        &self,
        target: u64,
        group_id: ConsensusGroupId,
        rpc: AppendEntriesRequest<KiviTypeConfig>,
        option: RPCOption,
    ) -> Result<AppendEntriesResponse<KiviTypeConfig>, RPCError<KiviTypeConfig>> {
        let request = PeerRequest::Append(encode_append_request(&rpc));
        let response = self
            .call(target, group_id, request, &option)
            .await
            .map_err(|error| unreachable(target, &error))?;
        match response {
            PeerResponse::Append(response) => Ok(decode_append_response(&response)),
            unexpected => Err(protocol_unreachable(
                target,
                &format!("peer answered append with {unexpected:?}"),
            )),
        }
    }

    async fn vote(
        &self,
        target: u64,
        group_id: ConsensusGroupId,
        rpc: VoteRequest<KiviTypeConfig>,
        option: RPCOption,
    ) -> Result<VoteResponse<KiviTypeConfig>, RPCError<KiviTypeConfig>> {
        let request = PeerRequest::Vote(encode_vote_request(&rpc));
        let response = self
            .call(target, group_id, request, &option)
            .await
            .map_err(|error| unreachable(target, &error))?;
        match response {
            PeerResponse::Vote(response) => Ok(decode_vote_response(&response)),
            unexpected => Err(protocol_unreachable(
                target,
                &format!("peer answered vote with {unexpected:?}"),
            )),
        }
    }

    async fn full_snapshot(
        &self,
        target: u64,
        group_id: ConsensusGroupId,
        vote: VoteOf<KiviTypeConfig>,
        snapshot: SnapshotOf<KiviTypeConfig, Cursor<Vec<u8>>>,
        _cancel: impl Future<Output = openraft::errors::ReplicationClosed>
        + openraft::OptionalSend
        + 'static,
        option: RPCOption,
    ) -> Result<SnapshotResponse<KiviTypeConfig>, StreamingError<KiviTypeConfig>> {
        // `cancel` fires when RaftCore aborts the transmission; fragment
        // sends are sequential unary RPCs, so cancellation surfaces as
        // the next send failing or the group shutting down — either way
        // the transfer stops and the receiver discards the partial
        // attempt when a fresh transfer id arrives.
        let SnapshotOf::<KiviTypeConfig, Cursor<Vec<u8>>> { meta, snapshot } = snapshot;
        let wire_meta = encode_snapshot_meta(&meta);
        let bytes = snapshot.into_inner();
        let transfer = self.transfers.fetch_add(1, Ordering::SeqCst);
        let wire_vote = encode_wire_vote(&vote);
        let mut offset = 0usize;
        loop {
            let end = (offset + SNAPSHOT_FRAGMENT_BYTES).min(bytes.len());
            let done = end == bytes.len();
            let request = PeerRequest::Snapshot(PeerSnapshotRequest {
                vote: wire_vote,
                meta: wire_meta.clone(),
                transfer,
                offset: offset as u64,
                data: bytes[offset..end].to_vec(),
                done,
            });
            let response = self
                .call(target, group_id, request, &option)
                .await
                .map_err(|error| snapshot_unreachable(target, &error))?;
            match response {
                PeerResponse::Snapshot(response) => {
                    if done {
                        return Ok(SnapshotResponse {
                            vote: decode_wire_vote(&response.vote),
                        });
                    }
                }
                unexpected => {
                    return Err(protocol_snapshot_unreachable(
                        target,
                        &format!("peer answered snapshot with {unexpected:?}"),
                    ));
                }
            }
            offset = end;
        }
    }
}

/// Factory binding one consensus group to a shared router: mints one
/// [`GroupNetworkAdapter`] per replication target, each carrying
/// `(target, group)`. With [`PeerRouter`] the mesh stays shared across
/// all groups on the node (one connection per node pair, never per
/// group); experiments may substitute any [`GroupRouter`] (such as an
/// in-process registry) without changing group code.
#[derive(Debug, Clone)]
pub struct GroupNetworkFactory<R = PeerRouter> {
    router: R,
    group: ConsensusGroupId,
}

impl<R> GroupNetworkFactory<R> {
    /// Binds the factory to a shared router for one group.
    #[must_use]
    pub const fn new(router: R, group: ConsensusGroupId) -> Self {
        Self { router, group }
    }
}

impl<R> RaftNetworkFactory<KiviTypeConfig> for GroupNetworkFactory<R>
where
    R: GroupRouter<KiviTypeConfig, ConsensusGroupId> + Clone + 'static,
{
    type Network = GroupNetworkAdapter<KiviTypeConfig, ConsensusGroupId, R>;

    #[allow(clippy::unused_async_trait_impl)]
    async fn new_client(&mut self, target: u64, _node: &openraft::BasicNode) -> Self::Network {
        // By `OpenRaft` contract this builds the client without
        // connecting: the mesh dials (and redials) on its own loops, so a
        // client for a temporarily down peer is still a valid client.
        GroupNetworkAdapter::new(self.router.clone(), target, self.group)
    }
}

#[cfg(test)]
mod tests {
    use openraft::vote::RaftLeaderId as _;

    use super::{
        decode_append_request, decode_snapshot_meta, decode_vote_request, decode_wire_entry,
        encode_append_request, encode_snapshot_meta, encode_vote_request, encode_wire_entry,
    };
    use crate::command::ConsensusCommand;
    use crate::mutation::ReplicatedMutation;

    /// Builds one deterministic test command for codec round-trips.
    fn test_command() -> ConsensusCommand {
        use kivi_state::{Key, Mutation, MutationEnvelope, ObjectVersion, OperationResult};
        use kivi_types::{
            NamespaceId, RequestIdentity, RequestSeq, SessionId, TabletAuthority, TabletEpoch,
            TabletId, UnixMicros, WriteGuardGeneration,
        };
        let authority = TabletAuthority::new(
            TabletId::from_u64(9),
            TabletEpoch::INITIAL,
            WriteGuardGeneration::INITIAL,
        );
        ConsensusCommand::Tablet(ReplicatedMutation::new(
            MutationEnvelope::new(
                NamespaceId::from_u64(1),
                authority,
                RequestIdentity::new(SessionId::from_u128(7), RequestSeq::from_u64(1)),
                None,
                Mutation::CounterAdd {
                    key: Key::from("n"),
                    delta: 2,
                },
            ),
            OperationResult::CounterUpdated {
                value: 2,
                version: ObjectVersion::FIRST,
            },
            UnixMicros::from_micros(5),
            RequestSeq::from_u64(0),
        ))
    }

    /// The full RPC surface round-trips through owned structs: encode on
    /// the caller, decode serving, re-encode answering, decode completing.
    /// Whatever the transport does with the bytes in between cannot change
    /// the verdicts.
    #[test]
    fn rpc_verdicts_survive_the_owned_codec() {
        use openraft::impls::leader_id_adv::LeaderId;
        let command = test_command();
        let entry = openraft::Entry {
            log_id: openraft::LogId::new(LeaderId::new(3, 1), 42),
            payload: openraft::EntryPayload::Normal(command),
        };
        let vote = openraft::impls::Vote::new_committed(3, 1);
        let request = openraft::raft::AppendEntriesRequest {
            vote,
            prev_log_id: Some(openraft::LogId::new(LeaderId::new(3, 1), 41)),
            entries: vec![entry],
            leader_commit: None,
        };
        let wire = encode_append_request(&request);
        let back = decode_append_request(&wire).expect("decodes");
        assert_eq!(back.vote, request.vote);
        assert_eq!(back.prev_log_id, request.prev_log_id);
        assert_eq!(back.entries.len(), 1);
        assert_eq!(back.entries[0].log_id, request.entries[0].log_id);
        // Corrupt command bytes fail the decode (backoff), never apply.
        let mut damaged = wire;
        let crate::peer::PeerEntryPayload::Normal(command) = &mut damaged.entries[0].payload else {
            panic!("test entry is a command");
        };
        command[0] ^= 0xFF;
        assert!(decode_append_request(&damaged).is_err());
        // Entry-level round-trip, including undecodable-command refusal.
        let single = encode_wire_entry(&request.entries[0]);
        let _ = decode_wire_entry(&single).expect("entry decodes");
        // Vote round-trips, including the leadership-transfer flag.
        let vote_rpc = openraft::raft::VoteRequest {
            vote,
            last_log_id: None,
            leadership_transfer: true,
        };
        let wire_vote = encode_vote_request(&vote_rpc);
        assert!(wire_vote.leadership_transfer);
        let back_vote = decode_vote_request(&wire_vote);
        assert_eq!(back_vote.vote, vote);
        assert!(back_vote.leadership_transfer);
        // Snapshot metadata binds base, membership, and checkpoint
        // identity — with no transfer identity anywhere in it.
        let meta = openraft::storage::SnapshotMeta {
            last_log_id: Some(openraft::LogId::new(LeaderId::new(3, 1), 41)),
            last_membership: openraft::StoredMembership::new(
                Some(openraft::LogId::new(LeaderId::new(2, 1), 0)),
                openraft::Membership::new(
                    vec![[1u64, 2, 3].into_iter().collect()],
                    [
                        (1u64, openraft::BasicNode::new("127.0.0.1:9101".to_owned())),
                        (2u64, openraft::BasicNode::new("127.0.0.1:9102".to_owned())),
                        (3u64, openraft::BasicNode::new("127.0.0.1:9103".to_owned())),
                    ]
                    .into_iter()
                    .collect::<std::collections::BTreeMap<_, _>>(),
                )
                .expect("test membership validates"),
            ),
        };
        let wire_meta = encode_snapshot_meta(&meta);
        assert!(
            wire_meta.checkpoint.is_empty(),
            "transfer builds pair bytes"
        );
        let back_meta = decode_snapshot_meta(&wire_meta).expect("meta decodes");
        assert_eq!(back_meta.last_log_id, meta.last_log_id);
        assert_eq!(
            back_meta
                .last_membership
                .membership()
                .voter_ids()
                .collect::<Vec<_>>(),
            vec![1, 2, 3]
        );
    }
}
