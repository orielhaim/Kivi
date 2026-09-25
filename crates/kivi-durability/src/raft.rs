//! Physical consensus records for the replicated tablet.
//!
//! In replicated mode the Raft log — not the tablet commit chain — is the
//! authoritative mutation history (no double-logging: one ordered truth).
//! These versioned, Kivi-owned record types persist exactly what the
//! `OpenRaft` storage contract needs, and nothing of `OpenRaft`'s Rust
//! memory representation:
//!
//! ```text
//! kind 10 v1 RAFT_VOTE:    namespace u64, group u64, term u64,
//!                          candidate u64, committed u8
//! kind 11 v1 RAFT_ENTRY:   namespace u64, group u64, index u64, term u64,
//!                          leader u64, payload_tag u8, payload…
//!     tag 0 BLANK:         (no further bytes)
//!     tag 1 NORMAL:        u32 len, command bytes (opaque deterministic
//!                          `ReplicatedMutation` encoding, owned by the
//!                          consensus adapter — never JSON, never `OpenRaft`
//!                          memory layout)
//!     tag 2 MEMBERSHIP:    u32 voters, u64 ids…, u32 nodes,
//!                          (u64 id, u32 addr len, addr bytes)…
//! kind 12 v1 RAFT_TRUNCATE: namespace u64, group u64, from_index u64
//!     (logical conflicting-suffix removal, inclusive: recovery replays
//!     the marker instead of truncating the shared physical file, so one
//!     physical WAL can hold many future groups safely)
//! kind 13 v1 RAFT_PURGE:   namespace u64, group u64, term u64,
//!                          leader u64, through_index u64
//!     (logical prefix removal, inclusive: distinct from physical segment
//!     deletion, which stays a separate conservative proof. The purged
//!     entry's term/leader ride along so recovery rebuilds an exact
//!     `last_purged_log_id` without guessing)
//! kind 14 v1 RAFT_COMMITTED: namespace u64, group u64, term u64,
//!                           leader u64, index u64
//!     (the `save_committed` pointer; absent means "re-derive at startup")
//! ```
//!
//! Every record identifies its consensus group (backed by [`TabletId`])
//! plus the term/leader identity the corresponding `OpenRaft` operation
//! requires. Framing (`kind u16, version u16, length u32`) is shared with
//! the tablet records in [`crate::wal`]; unknown kinds/versions fail
//! recovery loudly there, never skip.

use kivi_types::{NamespaceId, NodeId, TabletId};

use crate::error::DurabilityError;
use crate::wal::{RecordFault, WAL_MAX_BODY_BYTES};

/// `RAFT_VOTE` record kind (see module docs).
pub const RECORD_KIND_RAFT_VOTE: u16 = 10;
/// `RAFT_ENTRY` record kind (see module docs).
pub const RECORD_KIND_RAFT_ENTRY: u16 = 11;
/// `RAFT_TRUNCATE` record kind (see module docs).
pub const RECORD_KIND_RAFT_TRUNCATE: u16 = 12;
/// `RAFT_PURGE` record kind (see module docs).
pub const RECORD_KIND_RAFT_PURGE: u16 = 13;
/// `RAFT_COMMITTED` record kind (see module docs).
pub const RECORD_KIND_RAFT_COMMITTED: u16 = 14;

/// Entry payload tag: leader blank (commit barrier, no data).
pub const RAFT_PAYLOAD_BLANK: u8 = 0;
/// Entry payload tag: opaque deterministic command bytes.
pub const RAFT_PAYLOAD_NORMAL: u8 = 1;
/// Entry payload tag: membership configuration.
pub const RAFT_PAYLOAD_MEMBERSHIP: u8 = 2;

/// Returns whether a record kind tag belongs to the consensus family
/// (kinds 10–14). The segment scanner dispatches on this before version
/// checks, so unknown consensus versions fail exactly like unknown tablet
/// versions: loudly, never skipped.
#[must_use]
pub const fn is_raft_kind(kind: u16) -> bool {
    matches!(
        kind,
        RECORD_KIND_RAFT_VOTE
            | RECORD_KIND_RAFT_ENTRY
            | RECORD_KIND_RAFT_TRUNCATE
            | RECORD_KIND_RAFT_PURGE
            | RECORD_KIND_RAFT_COMMITTED
    )
}

/// Durably recorded vote: the term granted plus who it was granted to.
/// Mirrors the `OpenRaft` durability contract (persist before returning)
/// without persisting its memory layout.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RaftVote {
    /// Owning namespace.
    pub namespace: NamespaceId,
    /// Consensus group (tablet) this vote belongs to.
    pub group: TabletId,
    /// Granted term.
    pub term: u64,
    /// Candidate node the vote was granted to.
    pub candidate: NodeId,
    /// Whether the vote is committed (granted, not merely seen).
    pub committed: bool,
}

/// One replicated log entry: authoritative history in replicated mode.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RaftEntry {
    /// Owning namespace.
    pub namespace: NamespaceId,
    /// Consensus group (tablet) this entry belongs to.
    pub group: TabletId,
    /// Per-group log index (consecutive — no holes). `OpenRaft` 0.10
    /// numbers its first entry 0, so 0 is a legal index here.
    pub index: u64,
    /// Term that produced the entry.
    pub term: u64,
    /// Leader that produced the entry.
    pub leader: NodeId,
    /// Entry payload (blank barrier, command bytes, or membership).
    pub payload: RaftEntryPayload,
}

/// Payload of one [`RaftEntry`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RaftEntryPayload {
    /// Leader commit barrier (no data).
    Blank,
    /// Opaque deterministic command bytes (Kivi-owned encoding).
    Normal(Vec<u8>),
    /// Membership configuration.
    Membership(RaftMembership),
}

/// Durably recorded membership: voters plus their dialable nodes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RaftMembership {
    /// Voter node ids.
    pub voters: Vec<u64>,
    /// (node id, dialable address) pairs.
    pub nodes: Vec<(u64, String)>,
}

/// Logical conflicting-suffix removal from `from_index` (inclusive).
/// Recovery rebuilds the group history by applying the marker — the shared
/// physical file is never `ftruncate`d for one group.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RaftTruncate {
    /// Owning namespace.
    pub namespace: NamespaceId,
    /// Consensus group (tablet) truncated.
    pub group: TabletId,
    /// First index removed (inclusive).
    pub from_index: u64,
}

/// Logical prefix removal through `through_index` (inclusive). Distinct
/// from physical segment deletion.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RaftPurge {
    /// Owning namespace.
    pub namespace: NamespaceId,
    /// Consensus group (tablet) purged.
    pub group: TabletId,
    /// Purged entry's term (rebuilds `last_purged_log_id` exactly).
    pub term: u64,
    /// Purged entry's leader (rebuilds `last_purged_log_id` exactly).
    pub leader: NodeId,
    /// Last index removed (inclusive).
    pub through_index: u64,
}

/// Durably recorded commit pointer (`save_committed`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RaftCommitted {
    /// Owning namespace.
    pub namespace: NamespaceId,
    /// Consensus group (tablet) committed.
    pub group: TabletId,
    /// Committed entry's term.
    pub term: u64,
    /// Committed entry's leader.
    pub leader: NodeId,
    /// Committed entry's index.
    pub index: u64,
}

/// One physical consensus record.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RaftRecord {
    /// Granted vote.
    Vote(RaftVote),
    /// Replicated log entry.
    Entry(RaftEntry),
    /// Logical suffix truncation marker.
    Truncate(RaftTruncate),
    /// Logical prefix purge marker.
    Purge(RaftPurge),
    /// Commit pointer.
    Committed(RaftCommitted),
}

impl RaftRecord {
    /// The consensus group (tablet) this record belongs to.
    #[must_use]
    pub const fn group(&self) -> TabletId {
        match self {
            Self::Vote(record) => record.group,
            Self::Entry(record) => record.group,
            Self::Truncate(record) => record.group,
            Self::Purge(record) => record.group,
            Self::Committed(record) => record.group,
        }
    }

    /// The owning namespace.
    #[must_use]
    pub const fn namespace(&self) -> NamespaceId {
        match self {
            Self::Vote(record) => record.namespace,
            Self::Entry(record) => record.namespace,
            Self::Truncate(record) => record.namespace,
            Self::Purge(record) => record.namespace,
            Self::Committed(record) => record.namespace,
        }
    }

    /// The record kind tag for framing.
    #[must_use]
    pub const fn kind(&self) -> u16 {
        match self {
            Self::Vote(_) => RECORD_KIND_RAFT_VOTE,
            Self::Entry(_) => RECORD_KIND_RAFT_ENTRY,
            Self::Truncate(_) => RECORD_KIND_RAFT_TRUNCATE,
            Self::Purge(_) => RECORD_KIND_RAFT_PURGE,
            Self::Committed(_) => RECORD_KIND_RAFT_COMMITTED,
        }
    }
}

fn push_u64(out: &mut Vec<u8>, value: u64) {
    out.extend_from_slice(&value.to_le_bytes());
}

/// Length-prefixed blob with a checked length (mirrors `push_wal_blob`:
/// oversize fails the append, never records a truncation).
fn push_blob(out: &mut Vec<u8>, bytes: &[u8]) -> Result<(), DurabilityError> {
    let len = u32::try_from(bytes.len()).map_err(|_| DurabilityError::InvalidConfig {
        reason: "raft record blob exceeds u32 length",
    })?;
    out.extend_from_slice(&len.to_le_bytes());
    out.extend_from_slice(bytes);
    Ok(())
}

fn take_u64(input: &[u8]) -> Result<(u64, &[u8]), RecordFault> {
    if input.len() < 8 {
        return Err(RecordFault::Truncated {
            context: "raft record word",
        });
    }
    Ok((
        u64::from_le_bytes(input[..8].try_into().unwrap_or([0; 8])),
        &input[8..],
    ))
}

fn take_u8(input: &[u8]) -> Result<(u8, &[u8]), RecordFault> {
    if input.is_empty() {
        return Err(RecordFault::Truncated {
            context: "raft record tag",
        });
    }
    Ok((input[0], &input[1..]))
}

fn take_blob(input: &[u8], max: usize) -> Result<(&[u8], &[u8]), RecordFault> {
    if input.len() < 4 {
        return Err(RecordFault::Truncated {
            context: "raft record blob",
        });
    }
    let len = u32::from_le_bytes(input[..4].try_into().unwrap_or([0; 4])) as usize;
    if len > max || input.len() < 4 + len {
        return Err(RecordFault::Oversized { len });
    }
    Ok((&input[4..4 + len], &input[4 + len..]))
}

/// Encodes one consensus record body (framing adds kind/version/length).
pub(crate) fn encode_raft_body(
    record: &RaftRecord,
    out: &mut Vec<u8>,
) -> Result<(), DurabilityError> {
    match record {
        RaftRecord::Vote(vote) => {
            push_u64(out, vote.namespace.as_u64());
            push_u64(out, vote.group.as_u64());
            push_u64(out, vote.term);
            push_u64(out, vote.candidate.as_u64());
            out.push(u8::from(vote.committed));
        }
        RaftRecord::Entry(entry) => {
            push_u64(out, entry.namespace.as_u64());
            push_u64(out, entry.group.as_u64());
            push_u64(out, entry.index);
            push_u64(out, entry.term);
            push_u64(out, entry.leader.as_u64());
            match &entry.payload {
                RaftEntryPayload::Blank => out.push(RAFT_PAYLOAD_BLANK),
                RaftEntryPayload::Normal(command) => {
                    out.push(RAFT_PAYLOAD_NORMAL);
                    push_blob(out, command)?;
                }
                RaftEntryPayload::Membership(membership) => {
                    out.push(RAFT_PAYLOAD_MEMBERSHIP);
                    push_u64(out, membership.voters.len() as u64);
                    for voter in &membership.voters {
                        push_u64(out, *voter);
                    }
                    push_u64(out, membership.nodes.len() as u64);
                    for (id, addr) in &membership.nodes {
                        push_u64(out, *id);
                        push_blob(out, addr.as_bytes())?;
                    }
                }
            }
        }
        RaftRecord::Truncate(marker) => {
            push_u64(out, marker.namespace.as_u64());
            push_u64(out, marker.group.as_u64());
            push_u64(out, marker.from_index);
        }
        RaftRecord::Purge(marker) => {
            push_u64(out, marker.namespace.as_u64());
            push_u64(out, marker.group.as_u64());
            push_u64(out, marker.term);
            push_u64(out, marker.leader.as_u64());
            push_u64(out, marker.through_index);
        }
        RaftRecord::Committed(pointer) => {
            push_u64(out, pointer.namespace.as_u64());
            push_u64(out, pointer.group.as_u64());
            push_u64(out, pointer.term);
            push_u64(out, pointer.leader.as_u64());
            push_u64(out, pointer.index);
        }
    }
    Ok(())
}

/// Decodes one consensus record body, reporting position-free faults the
/// segment scanner attaches to its lane/segment (same contract as the
/// tablet path). Complete-but-wrong bytes are corruption (loud); only
/// physical shortfall at the batch level counts as a torn tail, which the
/// scanner already excluded.
pub(crate) fn decode_raft_body(
    kind: u16,
    version: u16,
    body: &[u8],
) -> Result<RaftRecord, RecordFault> {
    if version != crate::wal::RECORD_VERSION_1 {
        return Err(RecordFault::UnsupportedVersion);
    }
    match kind {
        RECORD_KIND_RAFT_VOTE => decode_vote(body),
        RECORD_KIND_RAFT_ENTRY => decode_entry(body),
        RECORD_KIND_RAFT_TRUNCATE => decode_truncate(body),
        RECORD_KIND_RAFT_PURGE => decode_purge(body),
        RECORD_KIND_RAFT_COMMITTED => decode_committed(body),
        other => Err(RecordFault::Unknown {
            kind: other,
            version: crate::wal::RECORD_VERSION_1,
        }),
    }
}

fn decode_vote(body: &[u8]) -> Result<RaftRecord, RecordFault> {
    let (namespace, rest) = take_u64(body)?;
    let (group, rest) = take_u64(rest)?;
    let (term, rest) = take_u64(rest)?;
    let (candidate, rest) = take_u64(rest)?;
    let (committed, rest) = take_u8(rest)?;
    if !rest.is_empty() || committed > 1 {
        return Err(RecordFault::Corrupt("malformed raft vote record"));
    }
    Ok(RaftRecord::Vote(RaftVote {
        namespace: NamespaceId::from_u64(namespace),
        group: TabletId::from_u64(group),
        term,
        candidate: NodeId::from_u64(candidate),
        committed: committed == 1,
    }))
}

fn decode_entry(body: &[u8]) -> Result<RaftRecord, RecordFault> {
    let (namespace, rest) = take_u64(body)?;
    let (group, rest) = take_u64(rest)?;
    let (index, rest) = take_u64(rest)?;
    let (term, rest) = take_u64(rest)?;
    let (leader, rest) = take_u64(rest)?;
    let (tag, rest) = take_u8(rest)?;
    let payload = match tag {
        RAFT_PAYLOAD_BLANK => {
            if !rest.is_empty() {
                return Err(RecordFault::Corrupt("trailing bytes in blank entry"));
            }
            RaftEntryPayload::Blank
        }
        RAFT_PAYLOAD_NORMAL => {
            let (command, rest) = take_blob(rest, WAL_MAX_BODY_BYTES)?;
            if !rest.is_empty() {
                return Err(RecordFault::Corrupt("trailing bytes in command entry"));
            }
            if command.is_empty() {
                // Empty commands are never minted; their presence means
                // corruption, not an old version.
                return Err(RecordFault::Corrupt("empty command entry"));
            }
            RaftEntryPayload::Normal(command.to_vec())
        }
        RAFT_PAYLOAD_MEMBERSHIP => {
            let (voters, rest) = take_counted(rest, 8)?;
            let (nodes, rest) = take_nodes(rest)?;
            if !rest.is_empty() {
                return Err(RecordFault::Corrupt("trailing bytes in membership entry"));
            }
            RaftEntryPayload::Membership(RaftMembership { voters, nodes })
        }
        _ => return Err(RecordFault::Corrupt("unknown raft entry payload tag")),
    };
    // Note: index 0 is legal — `OpenRaft` 0.10 numbers its first entry
    // 0 — so no nonzero check here (unlike truncate/purge markers,
    // whose zero values are never minted).
    Ok(RaftRecord::Entry(RaftEntry {
        namespace: NamespaceId::from_u64(namespace),
        group: TabletId::from_u64(group),
        index,
        term,
        leader: NodeId::from_u64(leader),
        payload,
    }))
}

fn decode_truncate(body: &[u8]) -> Result<RaftRecord, RecordFault> {
    let (namespace, rest) = take_u64(body)?;
    let (group, rest) = take_u64(rest)?;
    let (from_index, rest) = take_u64(rest)?;
    if !rest.is_empty() {
        return Err(RecordFault::Corrupt("malformed raft truncate record"));
    }
    Ok(RaftRecord::Truncate(RaftTruncate {
        namespace: NamespaceId::from_u64(namespace),
        group: TabletId::from_u64(group),
        from_index,
    }))
}

fn decode_purge(body: &[u8]) -> Result<RaftRecord, RecordFault> {
    let (namespace, rest) = take_u64(body)?;
    let (group, rest) = take_u64(rest)?;
    let (term, rest) = take_u64(rest)?;
    let (leader, rest) = take_u64(rest)?;
    let (through_index, rest) = take_u64(rest)?;
    if !rest.is_empty() {
        return Err(RecordFault::Corrupt("malformed raft purge record"));
    }
    Ok(RaftRecord::Purge(RaftPurge {
        namespace: NamespaceId::from_u64(namespace),
        group: TabletId::from_u64(group),
        term,
        leader: NodeId::from_u64(leader),
        through_index,
    }))
}

fn decode_committed(body: &[u8]) -> Result<RaftRecord, RecordFault> {
    let (namespace, rest) = take_u64(body)?;
    let (group, rest) = take_u64(rest)?;
    let (term, rest) = take_u64(rest)?;
    let (leader, rest) = take_u64(rest)?;
    let (index, rest) = take_u64(rest)?;
    if !rest.is_empty() {
        return Err(RecordFault::Corrupt("malformed raft committed record"));
    }
    Ok(RaftRecord::Committed(RaftCommitted {
        namespace: NamespaceId::from_u64(namespace),
        group: TabletId::from_u64(group),
        term,
        leader: NodeId::from_u64(leader),
        index,
    }))
}

/// Reads a u64-counted list of fixed 8-byte words, bounding allocation by
/// the remaining input (each element costs 8 bytes on the wire).
fn take_counted(input: &[u8], stride: usize) -> Result<(Vec<u64>, &[u8]), RecordFault> {
    let (count, mut rest) = take_u64(input)?;
    let count = usize::try_from(count).map_err(|_| RecordFault::Oversized { len: usize::MAX })?;
    if count.saturating_mul(stride) > rest.len() {
        return Err(RecordFault::Oversized {
            len: count.saturating_mul(stride),
        });
    }
    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        let (word, tail) = take_u64(rest)?;
        out.push(word);
        rest = tail;
    }
    Ok((out, rest))
}

/// Dialable node addresses decoded from one membership entry.
type NodeAddrs = Vec<(u64, String)>;

/// Reads a u64-counted list of (id, address blob) pairs.
fn take_nodes(input: &[u8]) -> Result<(NodeAddrs, &[u8]), RecordFault> {
    let (count, mut rest) = take_u64(input)?;
    let count = usize::try_from(count).map_err(|_| RecordFault::Oversized { len: usize::MAX })?;
    // Each pair costs at least 12 bytes on the wire (id + empty blob
    // framing); bounding by the remaining input keeps allocation honest.
    if count.saturating_mul(12) > rest.len() {
        return Err(RecordFault::Oversized {
            len: count.saturating_mul(12),
        });
    }
    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        let (id, tail) = take_u64(rest)?;
        let (addr, tail) = take_blob(tail, 1024 * 1024)?;
        let addr = core::str::from_utf8(addr)
            .map_err(|_| RecordFault::Corrupt("non-utf8 node address in raft membership"))?;
        out.push((id, addr.to_owned()));
        rest = tail;
    }
    Ok((out, rest))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn group() -> TabletId {
        TabletId::from_u64(9)
    }

    fn namespace() -> NamespaceId {
        NamespaceId::from_u64(1)
    }

    fn round_trip(record: &RaftRecord) {
        let mut body = Vec::new();
        encode_raft_body(record, &mut body).expect("encodes");
        // Faults carry no lane/segment (the scanner attaches them); only
        // successful decodes reach this assertion.
        let back =
            decode_raft_body(record.kind(), crate::wal::RECORD_VERSION_1, &body).expect("decodes");
        assert_eq!(&back, record);
    }

    #[test]
    fn all_kinds_round_trip() {
        round_trip(&RaftRecord::Vote(RaftVote {
            namespace: namespace(),
            group: group(),
            term: 7,
            candidate: NodeId::from_u64(2),
            committed: true,
        }));
        round_trip(&RaftRecord::Entry(RaftEntry {
            namespace: namespace(),
            group: group(),
            index: 41,
            term: 7,
            leader: NodeId::from_u64(2),
            payload: RaftEntryPayload::Blank,
        }));
        round_trip(&RaftRecord::Entry(RaftEntry {
            namespace: namespace(),
            group: group(),
            index: 42,
            term: 7,
            leader: NodeId::from_u64(2),
            payload: RaftEntryPayload::Normal(vec![0xDE, 0xAD, 0xBE, 0xEF]),
        }));
        round_trip(&RaftRecord::Entry(RaftEntry {
            namespace: namespace(),
            group: group(),
            index: 43,
            term: 7,
            leader: NodeId::from_u64(2),
            payload: RaftEntryPayload::Membership(RaftMembership {
                voters: vec![1, 2, 3],
                nodes: vec![
                    (1, "127.0.0.1:9000".to_owned()),
                    (2, "127.0.0.1:9001".to_owned()),
                    (3, "127.0.0.1:9002".to_owned()),
                ],
            }),
        }));
        round_trip(&RaftRecord::Truncate(RaftTruncate {
            namespace: namespace(),
            group: group(),
            from_index: 44,
        }));
        round_trip(&RaftRecord::Purge(RaftPurge {
            namespace: namespace(),
            group: group(),
            term: 6,
            leader: NodeId::from_u64(2),
            through_index: 40,
        }));
        round_trip(&RaftRecord::Committed(RaftCommitted {
            namespace: namespace(),
            group: group(),
            term: 7,
            leader: NodeId::from_u64(2),
            index: 42,
        }));
    }

    #[test]
    fn vote_bytes_are_stable() {
        // Golden bytes: independent oracle for RAFT_VOTE v1. Any intentional
        // format change must update this literal deliberately.
        let record = RaftRecord::Vote(RaftVote {
            namespace: NamespaceId::from_u64(1),
            group: TabletId::from_u64(9),
            term: 7,
            candidate: NodeId::from_u64(2),
            committed: true,
        });
        let mut body = Vec::new();
        encode_raft_body(&record, &mut body).expect("encodes");
        let mut golden = Vec::new();
        for word in [1u64, 9, 7, 2] {
            golden.extend_from_slice(&word.to_le_bytes());
        }
        golden.push(1);
        assert_eq!(body, golden);
        assert_eq!(body.len(), 33);
    }

    #[test]
    fn corrupt_bodies_fail_loudly() {
        let record = RaftRecord::Truncate(RaftTruncate {
            namespace: namespace(),
            group: group(),
            from_index: 44,
        });
        let mut body = Vec::new();
        encode_raft_body(&record, &mut body).expect("encodes");
        // Short input fails (truncation, not a wrong value).
        assert!(
            decode_raft_body(record.kind(), crate::wal::RECORD_VERSION_1, &body[..10]).is_err()
        );
        // Trailing garbage fails.
        body.push(0xFF);
        assert!(decode_raft_body(record.kind(), crate::wal::RECORD_VERSION_1, &body).is_err());
        // Wrong version fails.
        let mut versioned = Vec::new();
        encode_raft_body(&record, &mut versioned).expect("encodes");
        assert!(decode_raft_body(record.kind(), 2, &versioned).is_err());
        // Zero markers are legal (0-based log: purge through the first
        // entry); trailing garbage after one still fails.
        let zero = RaftRecord::Purge(RaftPurge {
            namespace: namespace(),
            group: group(),
            term: 6,
            leader: NodeId::from_u64(2),
            through_index: 0,
        });
        let mut encoded = Vec::new();
        encode_raft_body(&zero, &mut encoded).expect("encodes");
        let back = decode_raft_body(zero.kind(), crate::wal::RECORD_VERSION_1, &encoded)
            .expect("zero purge decodes");
        assert_eq!(&back, &zero);
        // Unknown tag fails at the scanner with its kind and version.
        let fault = decode_raft_body(99, crate::wal::RECORD_VERSION_1, &body).expect_err("unknown");
        assert!(matches!(
            fault,
            crate::wal::RecordFault::Unknown { kind: 99, .. }
        ));
    }
}
