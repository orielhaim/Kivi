//! Segmented per-lane write-ahead log: formats, appends, recovery.
//!
//! One logical lane per worker; no global WAL mutex. A record carries its
//! full tablet identity (tablet, epoch, guard) plus its tablet commit
//! position, so recovery never assumes a tablet always lived in one lane —
//! future tablet movement across workers keeps old lanes recoverable.
//!
//! ## Segment file layout (`wal/lane-XXXX/<seq:020>.wal`)
//!
//! ```text
//! segment header (56 bytes, fixed):
//!   0   4  magic "KVWL"
//!   4   2  major = 1
//!   6   2  minor = 0
//!   8  16  cluster (u128)
//!   24  8  node (u64)
//!   32  8  incarnation (u64, the writer's own)
//!   40  2  lane (u16)
//!   42  8  segment sequence (u64, from 1, contiguous per lane)
//!   50  2  reserved (0)
//!   52  4  CRC32C of bytes [0..52]
//! then batches back to back.
//! ```
//!
//! ## Batch framing (one physical durability unit)
//!
//! ```text
//! batch header (40 bytes, fixed):
//!   0   4  magic "KVWB"
//!   4   2  major = 1
//!   6   2  minor = 0
//!   8   8  batch sequence (u64, per lane from 1, contiguous)
//!   16  4  body length in bytes
//!   20  2  record count (≥ 1)
//!   22  2  flags (0)
//!   24  2  lane (u16, must match the directory)
//!   26  8  segment sequence (u64, must match the file)
//!   34  2  reserved (0)
//!   36  4  CRC32C of bytes [0..36]
//! body: `record_count` records (see below)
//! batch footer (20 bytes, fixed):
//!   0   4  magic "KVWE"
//!   4   8  batch sequence (must match the header)
//!   12  4  CRC32C of the body bytes
//!   16  4  CRC32C of bytes [0..16]
//! ```
//!
//! ## Logical records (inside the body)
//!
//! ```text
//! kind u16, version u16, length u32, body[length]
//! kind 1 v1 MUTATION:
//!   namespace u64, tablet u64, epoch u64, guard u64, commit u64, now u64,
//!   opcode u8, has_identity u8, [session u128, seq u64, ack u64],
//!   mutation_len u32, mutation[..], outcome_len u32, outcome[..]
//!   (outcome is the OperationResult the apply must reproduce; `now` is the
//!   commit timestamp replay re-applies under, so expiry visibility matches
//!   commit time exactly)
//! kind 2 v1 OUTCOME (terminal result with no mutation):
//!   namespace u64, tablet u64, epoch u64, guard u64, commit u64, now u64,
//!   opcode u8, has_identity u8, [session u128, seq u64, ack u64],
//!   outcome_len u32, outcome[..]
//!   (outcome is the DurableOutcome to replay to retries)
//! ```
//!
//! ## Recovery rules
//!
//! * Structural validation (magic, versions, identity, CRCs, framing) runs
//!   before any record is trusted.
//! * Only a physically truncated final batch of the lane's last segment is
//!   a torn tail: truncate the file to the last complete valid boundary and
//!   continue. A complete-but-wrong final batch is corruption, not a tear.
//! * Corruption anywhere else (middle segments, complete bad batches,
//!   duplicate/skipped batch sequences, segment gaps, unknown record
//!   kinds/versions, oversize declarations) fails recovery loudly —
//!   history is never silently truncated.
//! * Segments rotate at the configured target size; old segments are never
//!   deleted in this stage (compaction belongs to the checkpoint fabric).

use std::fs::{self, File};
use std::io::{self, Write};
use std::path::{Path, PathBuf};

use kivi_codec::integrity::crc32c_checksum;
use kivi_codec::{CodecError, Decode, Encode};
use kivi_state::{DurableOutcome, Mutation, OperationResult};
use kivi_types::{
    ClusterId, MutationIdentity, NamespaceId, NodeId, NodeIncarnation, RequestIdentity, RequestSeq,
    TabletEpoch, TabletId, WriteGuardGeneration,
};
use zerocopy::byteorder::{LittleEndian, U16, U32, U64, U128};
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

use crate::error::{DurabilityError, RecoveryError};
use crate::fs::{create_dir_all_sync, sync_file};
use crate::provider::{
    CommitProof, DurabilityLevel, DurabilityProvider, LaneStats, PersistIntent, StorageHealth,
};

/// Magic word: ASCII `"KVWL"` read as a little-endian `u32`.
pub const WAL_MAGIC_SEGMENT: u32 = 0x4C57_564B;
/// Magic word: ASCII `"KVWB"` read as a little-endian `u32`.
pub const WAL_MAGIC_BATCH: u32 = 0x4257_564B;
/// Magic word: ASCII `"KVWE"` read as a little-endian `u32`.
pub const WAL_MAGIC_FOOTER: u32 = 0x4557_564B;
/// Current WAL major version.
pub const WAL_MAJOR: u16 = 1;
/// Current WAL minor version.
pub const WAL_MINOR: u16 = 0;
/// Encoded segment header length in bytes.
pub const SEGMENT_HEADER_LEN: usize = 56;
/// Encoded batch header length in bytes.
pub const BATCH_HEADER_LEN: usize = 40;
/// Encoded batch footer length in bytes.
pub const BATCH_FOOTER_LEN: usize = 20;
/// Allocation guard for one batch body (larger declarations fail before
/// allocating). Individual values are already bounded by the protocol
/// frame ceiling; this guards the framing arithmetic itself.
pub const WAL_MAX_BODY_BYTES: usize = 256 * 1024 * 1024;
/// Allocation guard for records per batch (baseline batches hold few).
pub const WAL_MAX_RECORDS_PER_BATCH: usize = 4096;
/// Production segment rotation target (RFC starting point).
pub const DEFAULT_SEGMENT_TARGET_BYTES: u64 = 256 * 1024 * 1024;
/// Record kinds. Fixed forever within format version 1.
pub const RECORD_KIND_MUTATION: u16 = 1;
/// Terminal-outcome record kind (see [`WalRecord::Outcome`]).
pub const RECORD_KIND_OUTCOME: u16 = 2;
/// Record format version for every kind minted in this stage.
pub const RECORD_VERSION_1: u16 = 1;

// Only `unsafe` in this module: zerocopy's derive-generated trait impls.
// Every use goes through the safe API; hand-written `unsafe` stays banned.
#[allow(unsafe_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout)]
#[repr(C)]
struct SegmentHeaderWire {
    magic: U32<LittleEndian>,
    major: U16<LittleEndian>,
    minor: U16<LittleEndian>,
    cluster: U128<LittleEndian>,
    node: U64<LittleEndian>,
    incarnation: U64<LittleEndian>,
    lane: U16<LittleEndian>,
    segment_seq: U64<LittleEndian>,
    reserved: U16<LittleEndian>,
    header_crc: U32<LittleEndian>,
}

const _: () = assert!(core::mem::size_of::<SegmentHeaderWire>() == SEGMENT_HEADER_LEN);

#[allow(unsafe_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout)]
#[repr(C)]
struct BatchHeaderWire {
    magic: U32<LittleEndian>,
    major: U16<LittleEndian>,
    minor: U16<LittleEndian>,
    batch_seq: U64<LittleEndian>,
    body_len: U32<LittleEndian>,
    record_count: U16<LittleEndian>,
    flags: U16<LittleEndian>,
    lane: U16<LittleEndian>,
    segment_seq: U64<LittleEndian>,
    reserved: U16<LittleEndian>,
    header_crc: U32<LittleEndian>,
}

const _: () = assert!(core::mem::size_of::<BatchHeaderWire>() == BATCH_HEADER_LEN);

#[allow(unsafe_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout)]
#[repr(C)]
struct BatchFooterWire {
    magic: U32<LittleEndian>,
    batch_seq: U64<LittleEndian>,
    body_crc: U32<LittleEndian>,
    footer_crc: U32<LittleEndian>,
}

const _: () = assert!(core::mem::size_of::<BatchFooterWire>() == BATCH_FOOTER_LEN);

/// Identity the WAL stamps into every segment it writes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LaneIdentity {
    /// Owning cluster (misdirected-file detection).
    pub cluster: ClusterId,
    /// Owning node.
    pub node: NodeId,
    /// Incarnation of the writing process (forensics + future fencing).
    pub incarnation: NodeIncarnation,
}

/// One accepted mutation with everything replay needs: routing identity,
/// fencing, ordering, retry identity, the Mutation IR, and the outcome
/// `apply` must reproduce.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MutationRecord {
    /// Target namespace.
    pub namespace: NamespaceId,
    /// Owning tablet.
    pub tablet: TabletId,
    /// Tablet epoch at accept time.
    pub epoch: TabletEpoch,
    /// Write-guard generation at accept time.
    pub guard: WriteGuardGeneration,
    /// This tablet's commit position (exactly once per accepted request).
    pub commit: kivi_types::CommitPosition,
    /// Commit timestamp (micros): replay re-applies under this `now` so
    /// expiry visibility matches commit time exactly.
    pub now: kivi_types::UnixMicros,
    /// Originating opcode discriminant (response shaping + replay mapping).
    pub opcode: u8,
    /// Retry identity with acknowledgement floor, if the request had one.
    pub identity: Option<MutationIdentity>,
    /// The deterministic mutation to re-apply.
    pub mutation: Mutation,
    /// The outcome re-application must reproduce.
    pub expected: OperationResult,
}

/// One terminal outcome with no mutation (counter rejection, expiry no-op,
/// version exhaustion): replay installs it for retries without executing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OutcomeRecord {
    /// Target namespace.
    pub namespace: NamespaceId,
    /// Owning tablet.
    pub tablet: TabletId,
    /// Tablet epoch at accept time.
    pub epoch: TabletEpoch,
    /// Write-guard generation at accept time.
    pub guard: WriteGuardGeneration,
    /// This tablet's commit position (consumed like any completion).
    pub commit: kivi_types::CommitPosition,
    /// Commit timestamp (micros, forensics + replay uniformity).
    pub now: kivi_types::UnixMicros,
    /// Originating opcode discriminant.
    pub opcode: u8,
    /// Retry identity with acknowledgement floor, if the request had one.
    pub identity: Option<MutationIdentity>,
    /// The terminal outcome to replay to retries.
    pub outcome: DurableOutcome,
}

/// One logical WAL record.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WalRecord {
    /// Accepted mutation plus its expected outcome.
    Mutation(MutationRecord),
    /// Terminal outcome with no mutation.
    Outcome(OutcomeRecord),
}

impl WalRecord {
    /// The tablet this record belongs to.
    #[must_use]
    pub fn tablet(&self) -> TabletId {
        match self {
            Self::Mutation(record) => record.tablet,
            Self::Outcome(record) => record.tablet,
        }
    }

    /// This tablet's commit position.
    #[must_use]
    pub fn commit(&self) -> kivi_types::CommitPosition {
        match self {
            Self::Mutation(record) => record.commit,
            Self::Outcome(record) => record.commit,
        }
    }

    /// The commit timestamp replay re-applies under.
    #[must_use]
    pub fn now(&self) -> kivi_types::UnixMicros {
        match self {
            Self::Mutation(record) => record.now,
            Self::Outcome(record) => record.now,
        }
    }

    /// The target namespace.
    #[must_use]
    pub fn namespace(&self) -> NamespaceId {
        match self {
            Self::Mutation(record) => record.namespace,
            Self::Outcome(record) => record.namespace,
        }
    }

    /// The tablet epoch at accept time.
    #[must_use]
    pub fn epoch(&self) -> TabletEpoch {
        match self {
            Self::Mutation(record) => record.epoch,
            Self::Outcome(record) => record.epoch,
        }
    }

    /// The write-guard generation at accept time.
    #[must_use]
    pub fn guard(&self) -> WriteGuardGeneration {
        match self {
            Self::Mutation(record) => record.guard,
            Self::Outcome(record) => record.guard,
        }
    }

    /// The retry identity, if the request carried one.
    #[must_use]
    pub fn identity(&self) -> Option<MutationIdentity> {
        match self {
            Self::Mutation(record) => record.identity,
            Self::Outcome(record) => record.identity,
        }
    }
}

/// One recovered record with its physical position.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecoveredRecord {
    /// Per-lane batch sequence the record arrived in.
    pub batch_seq: u64,
    /// The logical record.
    pub record: WalRecord,
}

/// Recovery result for one lane: validated records plus repair accounting.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LaneRecovery {
    /// The recovered lane.
    pub lane: u16,
    /// Validated records in file order.
    pub records: Vec<RecoveredRecord>,
    /// Segment files scanned.
    pub segments_scanned: u64,
    /// Bytes truncated from a torn tail (0 when the tail was clean).
    pub truncated_bytes: u64,
}

/// Whole-engine recovery summary for operators: totals across lanes plus
/// how many tablets replayed history.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RecoverySummary {
    /// Logical records replayed into tablets.
    pub records_replayed: u64,
    /// Segment files scanned across all lanes.
    pub segments_scanned: u64,
    /// Bytes truncated from torn tails across all lanes.
    pub tail_truncated_bytes: u64,
    /// Tablets that replayed at least... every tablet present in the
    /// startup directory is validated; this counts tablets that applied
    /// at least one record.
    pub tablets_recovered: u64,
}

fn encode_identity_prefix(out: &mut Vec<u8>, identity: Option<&MutationIdentity>) {
    match identity {
        None => {
            out.push(0);
        }
        Some(identity) => {
            out.push(1);
            out.extend_from_slice(&identity.client.session().as_u128().to_le_bytes());
            out.extend_from_slice(&identity.client.seq().as_u64().to_le_bytes());
            out.extend_from_slice(&identity.ack_floor.as_u64().to_le_bytes());
        }
    }
}

fn decode_identity_prefix(input: &[u8]) -> Result<(Option<MutationIdentity>, usize), CodecError> {
    let (flag, mut at) = u8::decode(input)?;
    match flag {
        0 => Ok((None, at)),
        1 => {
            let (session, used) = u128::decode(&input[at..])?;
            at += used;
            let (seq, used) = u64::decode(&input[at..])?;
            at += used;
            let (ack, used) = u64::decode(&input[at..])?;
            at += used;
            Ok((
                Some(MutationIdentity::new(
                    RequestIdentity::new(
                        kivi_types::SessionId::from_u128(session),
                        RequestSeq::from_u64(seq),
                    ),
                    RequestSeq::from_u64(ack),
                )),
                at,
            ))
        }
        _ => Err(CodecError::InvalidTag {
            kind: "wal-identity",
            tag: flag,
        }),
    }
}

/// Length-prefixed blob with a checked length (u32 range enforced rather
/// than truncated by cast).
fn push_wal_blob(out: &mut Vec<u8>, bytes: &[u8]) -> Result<(), DurabilityError> {
    let len = u32::try_from(bytes.len()).map_err(|_| DurabilityError::InvalidConfig {
        reason: "WAL record blob exceeds u32 length",
    })?;
    out.extend_from_slice(&len.to_le_bytes());
    out.extend_from_slice(bytes);
    Ok(())
}

fn encode_record_body(record: &WalRecord, out: &mut Vec<u8>) -> Result<(), DurabilityError> {
    match record {
        WalRecord::Mutation(entry) => {
            out.extend_from_slice(&entry.namespace.as_u64().to_le_bytes());
            out.extend_from_slice(&entry.tablet.as_u64().to_le_bytes());
            out.extend_from_slice(&entry.epoch.as_u64().to_le_bytes());
            out.extend_from_slice(&entry.guard.as_u64().to_le_bytes());
            out.extend_from_slice(&entry.commit.as_u64().to_le_bytes());
            out.extend_from_slice(&entry.now.as_micros().to_le_bytes());
            out.push(entry.opcode);
            encode_identity_prefix(out, entry.identity.as_ref());
            let mut mutation = Vec::new();
            entry.mutation.encode(&mut mutation);
            push_wal_blob(out, &mutation)?;
            let mut expected = Vec::new();
            entry.expected.encode(&mut expected);
            push_wal_blob(out, &expected)?;
        }
        WalRecord::Outcome(entry) => {
            out.extend_from_slice(&entry.namespace.as_u64().to_le_bytes());
            out.extend_from_slice(&entry.tablet.as_u64().to_le_bytes());
            out.extend_from_slice(&entry.epoch.as_u64().to_le_bytes());
            out.extend_from_slice(&entry.guard.as_u64().to_le_bytes());
            out.extend_from_slice(&entry.commit.as_u64().to_le_bytes());
            out.extend_from_slice(&entry.now.as_micros().to_le_bytes());
            out.push(entry.opcode);
            encode_identity_prefix(out, entry.identity.as_ref());
            let mut outcome = Vec::new();
            entry.outcome.encode(&mut outcome);
            push_wal_blob(out, &outcome)?;
        }
    }
    Ok(())
}

/// Encodes one batch body: framed records with backpatched lengths.
/// Validates count and total size before the caller allocates the frame.
fn encode_batch_body(records: &[WalRecord]) -> Result<Vec<u8>, DurabilityError> {
    if records.is_empty() || records.len() > WAL_MAX_RECORDS_PER_BATCH {
        return Err(DurabilityError::InvalidConfig {
            reason: "WAL batch must hold 1..=4096 records",
        });
    }
    let mut body = Vec::new();
    for record in records {
        let kind = match record {
            WalRecord::Mutation(_) => RECORD_KIND_MUTATION,
            WalRecord::Outcome(_) => RECORD_KIND_OUTCOME,
        };
        let start = body.len();
        body.extend_from_slice(&kind.to_le_bytes());
        body.extend_from_slice(&RECORD_VERSION_1.to_le_bytes());
        body.extend_from_slice(&0u32.to_le_bytes()); // length backpatch
        encode_record_body(record, &mut body)?;
        let len =
            u32::try_from(body.len() - start - 8).map_err(|_| DurabilityError::InvalidConfig {
                reason: "WAL record exceeds u32 length",
            })?;
        body[start + 4..start + 8].copy_from_slice(&len.to_le_bytes());
    }
    if body.len() > WAL_MAX_BODY_BYTES {
        return Err(DurabilityError::InvalidConfig {
            reason: "WAL batch body exceeds allocation guard",
        });
    }
    Ok(body)
}

/// Structural record-decode failure without file position: the segment
/// scanner attaches lane and segment before surfacing it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RecordFault {
    /// Complete bytes that fail validation.
    Corrupt(&'static str),
    /// Unknown state-changing kind or version (never skipped).
    Unknown {
        /// Observed kind tag.
        kind: u16,
        /// Observed version.
        version: u16,
    },
    /// A declared length exceeds the allocation guard.
    Oversized {
        /// Declared length.
        len: usize,
    },
    /// Truncated input where the batch framing promised bytes.
    Truncated {
        /// What was being decoded (static context).
        context: &'static str,
    },
    /// Unsupported record format version.
    UnsupportedVersion,
}

impl RecordFault {
    /// Attaches the file position, producing the public error.
    fn at(self, lane: u16, segment: u64) -> RecoveryError {
        match self {
            Self::Corrupt(reason) => RecoveryError::CorruptSegment {
                lane,
                segment,
                reason,
            },
            Self::Unknown { kind, version } => RecoveryError::UnknownRecord { kind, version },
            Self::Oversized { len } => RecoveryError::Oversized {
                len,
                max: WAL_MAX_BODY_BYTES,
            },
            Self::Truncated { context } => RecoveryError::Truncated { context },
            Self::UnsupportedVersion => RecoveryError::UnsupportedVersion {
                major: WAL_MAJOR,
                minor: WAL_MINOR,
                context: "wal record",
            },
        }
    }
}

/// Fixed-size record prefix: routing identity, fencing, ordering, and the
/// commit timestamp, all raw little-endian words.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct RecordPrefix {
    namespace: u64,
    tablet: u64,
    epoch: u64,
    guard: u64,
    commit: u64,
    now: u64,
    opcode: u8,
}

fn decode_fixed_prefix(input: &[u8]) -> Result<(RecordPrefix, usize), RecordFault> {
    let need = 8 * 6 + 1;
    if input.len() < need {
        return Err(RecordFault::Truncated {
            context: "wal record prefix",
        });
    }
    Ok((
        RecordPrefix {
            namespace: u64::from_le_bytes(input[0..8].try_into().unwrap_or([0; 8])),
            tablet: u64::from_le_bytes(input[8..16].try_into().unwrap_or([0; 8])),
            epoch: u64::from_le_bytes(input[16..24].try_into().unwrap_or([0; 8])),
            guard: u64::from_le_bytes(input[24..32].try_into().unwrap_or([0; 8])),
            commit: u64::from_le_bytes(input[32..40].try_into().unwrap_or([0; 8])),
            now: u64::from_le_bytes(input[40..48].try_into().unwrap_or([0; 8])),
            opcode: input[48],
        },
        need,
    ))
}

fn decode_blob_prefix(input: &[u8]) -> Result<(&[u8], usize), RecordFault> {
    if input.len() < 4 {
        return Err(RecordFault::Truncated {
            context: "wal record blob",
        });
    }
    let len = u32::from_le_bytes(input[0..4].try_into().unwrap_or([0; 4])) as usize;
    if len > WAL_MAX_BODY_BYTES || input.len() < 4 + len {
        return Err(RecordFault::Oversized { len });
    }
    Ok((&input[4..4 + len], 4 + len))
}

fn decode_record_body(kind: u16, version: u16, body: &[u8]) -> Result<WalRecord, RecordFault> {
    if version != RECORD_VERSION_1 {
        return Err(RecordFault::UnsupportedVersion);
    }
    // Codec failures inside a present body are corruption (the batch
    // framing already proved the bytes arrived); only physical shortfall
    // at the batch level counts as a torn tail.
    let identity_at = |input: &[u8], what: &'static str| {
        decode_identity_prefix(input).map_err(|_| RecordFault::Corrupt(what))
    };
    match kind {
        RECORD_KIND_MUTATION => {
            let (prefix, mut at) = decode_fixed_prefix(body)?;
            let (identity, used) = identity_at(&body[at..], "undecodable mutation identity")?;
            at += used;
            let (mutation_bytes, used) = decode_blob_prefix(&body[at..])?;
            at += used;
            let (expected_bytes, used) = decode_blob_prefix(&body[at..])?;
            at += used;
            if at != body.len() {
                return Err(RecordFault::Corrupt("trailing bytes in mutation record"));
            }
            let (mutation, consumed) = Mutation::decode(mutation_bytes)
                .map_err(|_| RecordFault::Corrupt("undecodable mutation"))?;
            if consumed != mutation_bytes.len() {
                return Err(RecordFault::Corrupt("trailing bytes in mutation IR"));
            }
            let (expected, consumed) = OperationResult::decode(expected_bytes)
                .map_err(|_| RecordFault::Corrupt("undecodable expected outcome"))?;
            if consumed != expected_bytes.len() {
                return Err(RecordFault::Corrupt("trailing bytes in expected outcome"));
            }
            Ok(WalRecord::Mutation(MutationRecord {
                namespace: NamespaceId::from_u64(prefix.namespace),
                tablet: TabletId::from_u64(prefix.tablet),
                epoch: TabletEpoch::from_u64(prefix.epoch),
                guard: WriteGuardGeneration::from_u64(prefix.guard),
                commit: kivi_types::CommitPosition::from_u64(prefix.commit),
                now: kivi_types::UnixMicros::from_micros(prefix.now),
                opcode: prefix.opcode,
                identity,
                mutation,
                expected,
            }))
        }
        RECORD_KIND_OUTCOME => {
            let (prefix, mut at) = decode_fixed_prefix(body)?;
            let (identity, used) = identity_at(&body[at..], "undecodable outcome identity")?;
            at += used;
            let (outcome_bytes, used) = decode_blob_prefix(&body[at..])?;
            at += used;
            if at != body.len() {
                return Err(RecordFault::Corrupt("trailing bytes in outcome record"));
            }
            let (outcome, consumed) = DurableOutcome::decode(outcome_bytes)
                .map_err(|_| RecordFault::Corrupt("undecodable outcome"))?;
            if consumed != outcome_bytes.len() {
                return Err(RecordFault::Corrupt("trailing bytes in outcome"));
            }
            Ok(WalRecord::Outcome(OutcomeRecord {
                namespace: NamespaceId::from_u64(prefix.namespace),
                tablet: TabletId::from_u64(prefix.tablet),
                epoch: TabletEpoch::from_u64(prefix.epoch),
                guard: WriteGuardGeneration::from_u64(prefix.guard),
                commit: kivi_types::CommitPosition::from_u64(prefix.commit),
                now: kivi_types::UnixMicros::from_micros(prefix.now),
                opcode: prefix.opcode,
                identity,
                outcome,
            }))
        }
        other => Err(RecordFault::Unknown {
            kind: other,
            version,
        }),
    }
}

/// Formats a lane directory name (`lane-0007`).
#[must_use]
pub fn lane_dir_name(lane: u16) -> String {
    format!("lane-{lane:04}")
}

/// Formats a segment file name (`00000000000000000007.wal`).
#[must_use]
pub fn segment_file_name(segment_seq: u64) -> String {
    format!("{segment_seq:020}.wal")
}

/// Parses a segment file name back to its sequence, if well-formed.
fn parse_segment_name(name: &str) -> Option<u64> {
    name.strip_suffix(".wal")?.parse::<u64>().ok()
}

/// One lane's full admin snapshot: identity, position, health, counters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WorkerLaneStats {
    /// The lane index.
    pub lane: u16,
    /// Highest segment sequence present.
    pub active_segment: u64,
    /// Segment files present.
    pub segments: u64,
    /// Current coarse health.
    pub health: StorageHealth,
    /// Cumulative counters.
    pub stats: LaneStats,
}

/// One worker's WAL lane: an open segment file plus rotation and recovery.
///
/// All file operations are synchronous `std::fs` calls made from the owning
/// worker thread with no `.await` in between, so concurrent connection
/// tasks on the same thread cannot interleave a batch. Blocking an
/// executor thread on `fsync` is the honest baseline cost this stage
/// measures (see the durability benchmarks); an `io_uring` backend can
/// replace the mechanism later behind [`DurabilityProvider`].
#[derive(Debug)]
pub struct LocalWalLane {
    dir: PathBuf,
    lane: u16,
    identity: LaneIdentity,
    target_bytes: u64,
    file: Option<File>,
    segment_seq: u64,
    segment_bytes: u64,
    segments: u64,
    next_batch_seq: u64,
    stats: LaneStats,
    health: StorageHealth,
}

impl LocalWalLane {
    /// Opens (creating if needed) one lane directory and discovers its
    /// segments. Recovery itself runs separately via [`recover`](Self::recover)
    /// so callers observe the repair accounting before serving.
    ///
    /// # Errors
    ///
    /// Returns [`DurabilityError`] on filesystem failures or a zero segment
    /// target.
    pub fn open(
        wal_dir: &Path,
        lane: u16,
        identity: LaneIdentity,
        target_bytes: u64,
    ) -> Result<Self, DurabilityError> {
        if target_bytes == 0 {
            return Err(DurabilityError::InvalidConfig {
                reason: "WAL segment target must be nonzero",
            });
        }
        let dir = wal_dir.join(lane_dir_name(lane));
        let dir_sync = create_dir_all_sync(&dir)
            .map_err(|error| DurabilityError::io("create WAL lane directory", &dir, &error))?;
        tracing::debug!(lane, dir = %dir.display(), ?dir_sync, "WAL lane directory ready");
        let mut lane_state = Self {
            dir,
            lane,
            identity,
            target_bytes,
            file: None,
            segment_seq: 0,
            segment_bytes: 0,
            segments: 0,
            next_batch_seq: 1,
            stats: LaneStats::default(),
            health: StorageHealth::Healthy,
        };
        lane_state.discover()?;
        Ok(lane_state)
    }

    /// Returns the lane index.
    #[must_use]
    pub fn lane(&self) -> u16 {
        self.lane
    }

    /// Returns the active (highest) segment sequence (0 before the first
    /// append).
    #[must_use]
    pub fn active_segment(&self) -> u64 {
        self.segment_seq
    }

    /// Returns the number of segment files discovered or created.
    #[must_use]
    pub fn segment_count(&self) -> u64 {
        self.segments
    }

    /// Returns the full admin snapshot for this lane.
    #[must_use]
    pub fn lane_stats(&self) -> WorkerLaneStats {
        WorkerLaneStats {
            lane: self.lane,
            active_segment: self.segment_seq,
            segments: self.segments,
            health: self.health,
            stats: self.stats,
        }
    }

    /// Returns the next batch sequence that will be assigned.
    #[must_use]
    pub fn next_batch_seq(&self) -> u64 {
        self.next_batch_seq
    }

    /// Discovers existing segments: counts them for rotation and admin.
    /// Full validation happens in [`recover`](Self::recover), which also
    /// re-derives the exact batch sequence.
    fn discover(&mut self) -> Result<(), DurabilityError> {
        let seqs = segment_seqs(&self.dir, self.lane)?;
        self.segments = seqs.len() as u64;
        self.next_batch_seq = 1;
        Ok(())
    }

    /// Scans every segment, validates structure, truncates a torn tail on
    /// the last segment only, and returns the validated records in file
    /// order. Must run before the lane serves appends or replays.
    ///
    /// # Errors
    ///
    /// Returns [`RecoveryError`] on any corruption outside a torn final
    /// tail (see the module docs for the exact rules).
    pub fn recover(&mut self) -> Result<LaneRecovery, RecoveryError> {
        let mut seqs =
            segment_seqs(&self.dir, self.lane).map_err(|_| RecoveryError::CorruptSegment {
                lane: self.lane,
                segment: 0,
                reason: "unreadable lane directory",
            })?;
        seqs.sort_unstable();
        for (offset, found) in (1u64..).zip(seqs.iter()) {
            if *found != offset {
                return Err(RecoveryError::SegmentGap {
                    lane: self.lane,
                    expected: offset,
                    found: *found,
                });
            }
        }
        let mut records = Vec::new();
        let mut truncated_bytes = 0u64;
        let mut expected_batch = 1u64;
        let last = seqs.last().copied();
        for seq in &seqs {
            let is_last = Some(*seq) == last;
            let path = self.dir.join(segment_file_name(*seq));
            let bytes = read_segment_capped(&path, self.target_bytes).map_err(|_| {
                RecoveryError::CorruptSegment {
                    lane: self.lane,
                    segment: *seq,
                    reason: "unreadable segment file",
                }
            })?;
            let scanned =
                self.scan_segment(*seq, &bytes, is_last, &mut expected_batch, &mut records)?;
            if scanned.truncated > 0 {
                let file = File::options().write(true).open(&path).map_err(|_| {
                    RecoveryError::CorruptSegment {
                        lane: self.lane,
                        segment: *seq,
                        reason: "torn tail truncation failed to open",
                    }
                })?;
                file.set_len(scanned.kept)
                    .map_err(|_| RecoveryError::CorruptSegment {
                        lane: self.lane,
                        segment: *seq,
                        reason: "torn tail truncation failed",
                    })?;
                sync_file(&file).map_err(|_| RecoveryError::CorruptSegment {
                    lane: self.lane,
                    segment: *seq,
                    reason: "torn tail sync failed",
                })?;
                truncated_bytes += scanned.truncated;
            }
        }
        self.segments = seqs.len() as u64;
        self.segment_seq = last.unwrap_or(0);
        self.next_batch_seq = expected_batch;
        tracing::info!(
            lane = self.lane,
            segments = seqs.len(),
            batches = expected_batch.saturating_sub(1),
            records = records.len(),
            truncated_bytes,
            "WAL lane recovered",
        );
        Ok(LaneRecovery {
            lane: self.lane,
            records,
            segments_scanned: seqs.len() as u64,
            truncated_bytes,
        })
    }

    /// Appends one batch and syncs it before returning: the caller may ack
    /// the client as soon as this returns `Ok`. Rotates the segment first
    /// when the batch would overflow the target (never mid-batch).
    fn append_records(&mut self, records: &[WalRecord]) -> Result<CommitProof, DurabilityError> {
        if self.health == StorageHealth::Failed {
            return Err(DurabilityError::Io {
                op: "append to failed WAL lane",
                message: "lane is terminally failed; restart to recover".to_owned(),
                code: None,
            });
        }
        let body = encode_batch_body(records)?;
        let batch_len = (BATCH_HEADER_LEN + body.len() + BATCH_FOOTER_LEN) as u64;
        if self.file.is_none()
            || (self.segment_bytes > 0 && self.segment_bytes + batch_len > self.target_bytes)
        {
            self.rotate()?;
        }
        let batch_seq = self.next_batch_seq;
        let body_len = u32::try_from(body.len()).map_err(|_| DurabilityError::InvalidConfig {
            reason: "WAL batch body exceeds u32 length",
        })?;
        let record_count =
            u16::try_from(records.len()).map_err(|_| DurabilityError::InvalidConfig {
                reason: "WAL batch exceeds u16 record count",
            })?;
        let header = BatchHeaderWire {
            magic: U32::new(WAL_MAGIC_BATCH),
            major: U16::new(WAL_MAJOR),
            minor: U16::new(WAL_MINOR),
            batch_seq: U64::new(batch_seq),
            body_len: U32::new(body_len),
            record_count: U16::new(record_count),
            flags: U16::new(0),
            lane: U16::new(self.lane),
            segment_seq: U64::new(self.segment_seq),
            reserved: U16::new(0),
            header_crc: U32::new(0),
        };
        let mut header_bytes = [0u8; BATCH_HEADER_LEN];
        header_bytes.copy_from_slice(header.as_bytes());
        let header_crc = crc32c_checksum(&header_bytes[..BATCH_HEADER_LEN - 4]);
        header_bytes[BATCH_HEADER_LEN - 4..].copy_from_slice(&header_crc.to_le_bytes());
        let footer = BatchFooterWire {
            magic: U32::new(WAL_MAGIC_FOOTER),
            batch_seq: U64::new(batch_seq),
            body_crc: U32::new(crc32c_checksum(&body)),
            footer_crc: U32::new(0),
        };
        let mut footer_bytes = [0u8; BATCH_FOOTER_LEN];
        footer_bytes.copy_from_slice(footer.as_bytes());
        let footer_crc = crc32c_checksum(&footer_bytes[..BATCH_FOOTER_LEN - 4]);
        footer_bytes[BATCH_FOOTER_LEN - 4..].copy_from_slice(&footer_crc.to_le_bytes());
        let file = self.file.as_mut().expect("lane file open after rotate");
        let path = self.dir.join(segment_file_name(self.segment_seq));
        let write_outcome = (|| -> io::Result<()> {
            file.write_all(&header_bytes)?;
            file.write_all(&body)?;
            file.write_all(&footer_bytes)?;
            crate::fs::sync_file(file)?;
            Ok(())
        })();
        if let Err(error) = write_outcome {
            // Health follows the failure kind: full disks go read-only
            // (explicit, recoverable), anything else fails the lane
            // terminally. Either way the typed error below carries the
            // real OS failure — never a placeholder.
            if error.kind() == io::ErrorKind::StorageFull {
                self.health = StorageHealth::ReadOnly;
                tracing::warn!(lane = self.lane, path = %path.display(), "WAL lane read-only: storage full");
            } else {
                self.health = StorageHealth::Failed;
                tracing::error!(lane = self.lane, path = %path.display(), %error, "WAL lane failed");
            }
            return Err(DurabilityError::io("append WAL batch", &path, &error));
        }
        self.segment_bytes += batch_len;
        self.next_batch_seq += 1;
        self.stats.batches += 1;
        self.stats.records += records.len() as u64;
        self.stats.bytes += batch_len;
        self.stats.fsyncs += 1;
        if self.health == StorageHealth::ReadOnly {
            self.health = StorageHealth::Healthy;
            tracing::info!(lane = self.lane, "WAL lane writable again");
        }
        Ok(CommitProof {
            batch_seq,
            bytes_durable: batch_len,
            fsyncs: 1,
        })
    }

    /// Seals the current segment (if any) and opens the next sequence:
    /// create, header, sync file, sync directory, switch.
    fn rotate(&mut self) -> Result<(), DurabilityError> {
        // Drop the current handle first: the new segment must be fully
        // published before it becomes current.
        self.file = None;
        let seq = self.segment_seq + 1;
        let path = self.dir.join(segment_file_name(seq));
        let header = SegmentHeaderWire {
            magic: U32::new(WAL_MAGIC_SEGMENT),
            major: U16::new(WAL_MAJOR),
            minor: U16::new(WAL_MINOR),
            cluster: U128::new(self.identity.cluster.as_u128()),
            node: U64::new(self.identity.node.as_u64()),
            incarnation: U64::new(self.identity.incarnation.as_u64()),
            lane: U16::new(self.lane),
            segment_seq: U64::new(seq),
            reserved: U16::new(0),
            header_crc: U32::new(0),
        };
        let mut header_bytes = [0u8; SEGMENT_HEADER_LEN];
        header_bytes.copy_from_slice(header.as_bytes());
        let header_crc = crc32c_checksum(&header_bytes[..SEGMENT_HEADER_LEN - 4]);
        header_bytes[SEGMENT_HEADER_LEN - 4..].copy_from_slice(&header_crc.to_le_bytes());
        let mut file = File::create_new(&path)
            .map_err(|error| DurabilityError::io("create WAL segment", &path, &error))?;
        if let Err(error) = (|| -> io::Result<()> {
            file.write_all(&header_bytes)?;
            crate::fs::sync_file(&file)?;
            Ok(())
        })() {
            return Err(DurabilityError::io(
                "publish WAL segment header",
                &path,
                &error,
            ));
        }
        let _ = crate::fs::sync_dir(&self.dir);
        self.file = Some(file);
        self.segment_seq = seq;
        self.segment_bytes = SEGMENT_HEADER_LEN as u64;
        self.segments += 1;
        tracing::debug!(lane = self.lane, segment = seq, "WAL segment rotated");
        Ok(())
    }

    /// Opens the active segment file for appending (validates its header
    /// matches this lane first). Called lazily on first append so pure
    /// recovery runs never create files.
    fn ensure_open(&mut self) -> Result<(), DurabilityError> {
        if self.file.is_some() {
            return Ok(());
        }
        if self.segment_seq == 0 {
            return self.rotate();
        }
        let path = self.dir.join(segment_file_name(self.segment_seq));
        let file = File::options()
            .read(true)
            .append(true)
            .open(&path)
            .map_err(|error| DurabilityError::io("reopen WAL segment", &path, &error))?;
        let len = file
            .metadata()
            .map_err(|error| DurabilityError::io("stat WAL segment", &path, &error))?
            .len();
        self.segment_bytes = len;
        self.file = Some(file);
        Ok(())
    }

    /// Scans one segment's bytes: validates the header, then every batch.
    /// Returns how many bytes to keep plus how many were truncated (nonzero
    /// only for a torn tail on the last segment).
    #[allow(clippy::too_many_lines)]
    fn scan_segment(
        &self,
        seq: u64,
        bytes: &[u8],
        is_last: bool,
        expected_batch: &mut u64,
        records: &mut Vec<RecoveredRecord>,
    ) -> Result<ScannedSegment, RecoveryError> {
        let fail = |reason: &'static str| RecoveryError::CorruptSegment {
            lane: self.lane,
            segment: seq,
            reason,
        };
        if bytes.len() < SEGMENT_HEADER_LEN {
            // A short final segment is a torn tail (even a torn header):
            // keep nothing from this file. Anywhere else it is corruption.
            if !is_last {
                return Err(fail("short segment header in non-final segment"));
            }
            return Ok(ScannedSegment {
                kept: 0,
                truncated: bytes.len() as u64,
            });
        }
        let header = SegmentHeaderWire::read_from_bytes(&bytes[..SEGMENT_HEADER_LEN])
            .map_err(|_| fail("unreadable segment header"))?;
        if header.magic.get() != WAL_MAGIC_SEGMENT {
            return Err(fail("bad segment magic"));
        }
        if header.major.get() != WAL_MAJOR || header.minor.get() != WAL_MINOR {
            return Err(RecoveryError::UnsupportedVersion {
                major: header.major.get(),
                minor: header.minor.get(),
                context: "wal segment",
            });
        }
        if header.cluster.get() != self.identity.cluster.as_u128()
            || header.node.get() != self.identity.node.as_u64()
            || header.lane.get() != self.lane
            || header.segment_seq.get() != seq
        {
            return Err(RecoveryError::Misdirected {
                reason: "segment identity disagrees with its file and lane",
            });
        }
        if NodeIncarnation::from_u64(header.incarnation.get()) != self.identity.incarnation
            && header.incarnation.get() > self.identity.incarnation.as_u64()
        {
            return Err(RecoveryError::Misdirected {
                reason: "segment incarnation is from the future",
            });
        }
        let stored = header.header_crc.get();
        if crc32c_checksum(&bytes[..SEGMENT_HEADER_LEN - 4]) != stored {
            return Err(fail("segment header checksum mismatch"));
        }
        let mut offset = SEGMENT_HEADER_LEN;
        while offset < bytes.len() {
            let batch_start = offset;
            let remaining = bytes.len() - offset;
            if remaining < BATCH_HEADER_LEN {
                // Physically truncated batch header: torn tail or corruption.
                if !is_last {
                    return Err(fail("truncated batch header in non-final segment"));
                }
                return Ok(ScannedSegment {
                    kept: batch_start as u64,
                    truncated: remaining as u64,
                });
            }
            let batch = BatchHeaderWire::read_from_bytes(&bytes[offset..offset + BATCH_HEADER_LEN])
                .map_err(|_| fail("unreadable batch header"))?;
            if batch.magic.get() != WAL_MAGIC_BATCH {
                // Complete bytes, wrong magic: corruption, never a tear —
                // even on the last segment.
                return Err(fail("bad batch magic"));
            }
            if batch.major.get() != WAL_MAJOR || batch.minor.get() != WAL_MINOR {
                return Err(RecoveryError::UnsupportedVersion {
                    major: batch.major.get(),
                    minor: batch.minor.get(),
                    context: "wal batch",
                });
            }
            if batch.flags.get() != 0 {
                return Err(fail("nonzero batch flags"));
            }
            if batch.lane.get() != self.lane || batch.segment_seq.get() != seq {
                return Err(RecoveryError::Misdirected {
                    reason: "batch lane/segment disagrees with its file",
                });
            }
            if batch.batch_seq.get() != *expected_batch {
                return Err(fail("batch sequence gap or duplicate"));
            }
            let stored_header_crc = batch.header_crc.get();
            if crc32c_checksum(&bytes[offset..offset + BATCH_HEADER_LEN - 4]) != stored_header_crc {
                return Err(fail("batch header checksum mismatch"));
            }
            let body_len = batch.body_len.get() as usize;
            let count = batch.record_count.get() as usize;
            if body_len > WAL_MAX_BODY_BYTES {
                return Err(RecoveryError::Oversized {
                    len: body_len,
                    max: WAL_MAX_BODY_BYTES,
                });
            }
            if count == 0 || count > WAL_MAX_RECORDS_PER_BATCH {
                return Err(fail("bad batch record count"));
            }
            let need = BATCH_HEADER_LEN + body_len + BATCH_FOOTER_LEN;
            if remaining < need {
                if !is_last {
                    return Err(fail("truncated batch in non-final segment"));
                }
                return Ok(ScannedSegment {
                    kept: batch_start as u64,
                    truncated: remaining as u64,
                });
            }
            let body = &bytes[offset + BATCH_HEADER_LEN..offset + BATCH_HEADER_LEN + body_len];
            let footer = BatchFooterWire::read_from_bytes(
                &bytes[offset + BATCH_HEADER_LEN + body_len..offset + need],
            )
            .map_err(|_| fail("unreadable batch footer"))?;
            if footer.magic.get() != WAL_MAGIC_FOOTER
                || footer.batch_seq.get() != batch.batch_seq.get()
            {
                return Err(fail("bad batch footer"));
            }
            if crc32c_checksum(body) != footer.body_crc.get() {
                return Err(fail("batch body checksum mismatch"));
            }
            let footer_stored = footer.footer_crc.get();
            let footer_bytes = &bytes[offset + BATCH_HEADER_LEN + body_len..offset + need];
            if crc32c_checksum(&footer_bytes[..BATCH_FOOTER_LEN - 4]) != footer_stored {
                return Err(fail("batch footer checksum mismatch"));
            }
            let mut at = 0usize;
            for _ in 0..count {
                if body.len() < at + 8 {
                    return Err(fail("truncated record framing"));
                }
                let kind = u16::from_le_bytes(body[at..at + 2].try_into().unwrap_or([0; 2]));
                let version = u16::from_le_bytes(body[at + 2..at + 4].try_into().unwrap_or([0; 2]));
                let len =
                    u32::from_le_bytes(body[at + 4..at + 8].try_into().unwrap_or([0; 4])) as usize;
                if len > WAL_MAX_BODY_BYTES || body.len() < at + 8 + len {
                    return Err(RecoveryError::Oversized {
                        len,
                        max: WAL_MAX_BODY_BYTES,
                    });
                }
                let record = decode_record_body(kind, version, &body[at + 8..at + 8 + len])
                    .map_err(|fault| fault.at(self.lane, seq))?;
                records.push(RecoveredRecord {
                    batch_seq: batch.batch_seq.get(),
                    record,
                });
                at += 8 + len;
            }
            if at != body_len {
                return Err(fail("trailing bytes in batch body"));
            }
            *expected_batch += 1;
            offset += need;
        }
        Ok(ScannedSegment {
            kept: offset as u64,
            truncated: 0,
        })
    }
}

/// Scan outcome for one segment: bytes to keep and bytes truncated.
struct ScannedSegment {
    /// File length to keep (may equal the full length: clean).
    kept: u64,
    /// Bytes discarded as torn tail (0 when clean).
    truncated: u64,
}

impl DurabilityProvider for LocalWalLane {
    fn append_batch(&mut self, intent: &PersistIntent<'_>) -> Result<CommitProof, DurabilityError> {
        match intent.level {
            DurabilityLevel::Sync => {}
        }
        self.ensure_open()?;
        self.append_records(intent.records)
    }

    fn health(&self) -> StorageHealth {
        self.health
    }

    fn lane(&self) -> u16 {
        self.lane
    }

    fn stats(&self) -> LaneStats {
        self.stats
    }
}

/// Lists a lane's segment sequences, rejecting stray files loudly.
fn segment_seqs(dir: &Path, _lane: u16) -> Result<Vec<u64>, DurabilityError> {
    let entries =
        fs::read_dir(dir).map_err(|error| DurabilityError::io("list WAL lane", dir, &error))?;
    let mut seqs = Vec::new();
    for entry in entries {
        let entry = entry.map_err(|error| DurabilityError::io("list WAL lane", dir, &error))?;
        let name = entry.file_name().to_string_lossy().into_owned();
        match parse_segment_name(&name) {
            Some(seq) => seqs.push(seq),
            None => {
                return Err(DurabilityError::InvalidConfig {
                    reason: "WAL lane holds an unexpected file (refusing to guess)",
                });
            }
        }
    }
    Ok(seqs)
}

/// Reads one segment file with an allocation cap derived from the rotation
/// target (honest files are bounded by it plus one oversized batch).
fn read_segment_capped(path: &Path, target_bytes: u64) -> Result<Vec<u8>, DurabilityError> {
    let len = fs::metadata(path)
        .map_err(|error| DurabilityError::io("stat WAL segment", path, &error))?
        .len();
    let cap = target_bytes
        .saturating_mul(4)
        .saturating_add(WAL_MAX_BODY_BYTES as u64 + 1024 * 1024)
        .max(64 * 1024 * 1024);
    if len > cap {
        return Err(DurabilityError::InvalidConfig {
            reason: "WAL segment exceeds read cap",
        });
    }
    fs::read(path).map_err(|error| DurabilityError::io("read WAL segment", path, &error))
}

#[cfg(test)]
mod tests {
    use super::*;
    use kivi_state::{DurableOutcome, Key, Mutation, OperationResult};
    use kivi_types::CommitPosition;

    const CLUSTER: u128 = 0xC10C;
    const NODE: u64 = 7;

    fn test_identity() -> LaneIdentity {
        LaneIdentity {
            cluster: ClusterId::from_u128(CLUSTER),
            node: NodeId::from_u64(NODE),
            incarnation: NodeIncarnation::INITIAL,
        }
    }

    fn test_mutation(tablet: u64, commit: u64, key: &str) -> WalRecord {
        WalRecord::Mutation(MutationRecord {
            namespace: NamespaceId::from_u64(1),
            tablet: TabletId::from_u64(tablet),
            epoch: TabletEpoch::from_u64(1),
            guard: WriteGuardGeneration::from_u64(1),
            commit: CommitPosition::from_u64(commit),
            now: kivi_types::UnixMicros::from_micros(1_000_000),
            opcode: 2,
            identity: None,
            mutation: Mutation::PutBytes {
                key: Key::from(key),
                value: bytes_for(key),
            },
            expected: OperationResult::Stored {
                version: kivi_state::ObjectVersion::from_u64(commit),
            },
        })
    }

    fn bytes_for(key: &str) -> bytes::Bytes {
        bytes::Bytes::copy_from_slice(format!("value-for-{key}").as_bytes())
    }

    fn test_outcome(tablet: u64, commit: u64) -> WalRecord {
        WalRecord::Outcome(OutcomeRecord {
            namespace: NamespaceId::from_u64(1),
            tablet: TabletId::from_u64(tablet),
            epoch: TabletEpoch::from_u64(1),
            guard: WriteGuardGeneration::from_u64(1),
            commit: CommitPosition::from_u64(commit),
            now: kivi_types::UnixMicros::from_micros(1_000_000),
            opcode: 6,
            identity: None,
            outcome: DurableOutcome::Completed(OperationResult::CounterUpdated {
                value: 41,
                version: kivi_state::ObjectVersion::from_u64(commit),
            }),
        })
    }

    fn open_lane(dir: &Path, target: u64) -> LocalWalLane {
        LocalWalLane::open(dir, 3, test_identity(), target).expect("lane opens")
    }

    #[test]
    fn append_then_recover_round_trips() {
        let scratch = tempfile::tempdir().expect("scratch");
        let mut lane = open_lane(scratch.path(), 1024 * 1024);
        let records = vec![
            test_mutation(9, 1, "a"),
            test_outcome(9, 2),
            test_mutation(10, 1, "b"),
        ];
        for record in &records {
            lane.append_batch(&PersistIntent {
                records: std::slice::from_ref(record),
                level: DurabilityLevel::Sync,
            })
            .expect("append");
        }
        assert_eq!(lane.stats().batches, 3);
        assert_eq!(lane.stats().records, 3);
        assert_eq!(lane.health(), StorageHealth::Healthy);
        let recovery = lane.recover().expect("recover");
        assert_eq!(recovery.segments_scanned, 1);
        assert_eq!(recovery.truncated_bytes, 0);
        let replayed: Vec<WalRecord> = recovery
            .records
            .into_iter()
            .map(|entry| entry.record)
            .collect();
        assert_eq!(replayed, records);
        // Batch sequences are contiguous from 1 across the lane.
        assert_eq!(lane.next_batch_seq(), 4);
    }

    #[test]
    fn rotation_spans_segments_and_recovery_crosses_them() {
        let scratch = tempfile::tempdir().expect("scratch");
        // Tiny target forces a new segment nearly every batch.
        let mut lane = open_lane(scratch.path(), 256);
        for index in 0..8u64 {
            let record = test_mutation(9, index + 1, &format!("k{index}"));
            lane.append_batch(&PersistIntent {
                records: std::slice::from_ref(&record),
                level: DurabilityLevel::Sync,
            })
            .expect("append");
        }
        assert!(
            lane.segment_count() >= 3,
            "rotated, got {}",
            lane.segment_count()
        );
        let recovery = lane.recover().expect("recover across segments");
        assert_eq!(recovery.records.len(), 8);
        assert_eq!(recovery.truncated_bytes, 0);
        for (position, entry) in recovery.records.iter().enumerate() {
            assert_eq!(entry.batch_seq, position as u64 + 1);
            assert_eq!(entry.record.commit().as_u64(), position as u64 + 1);
        }
        // Old segments are never deleted in this stage.
        assert!(lane.segment_count() >= 3);
    }

    #[test]
    fn torn_tail_at_every_boundary_truncates_safely() {
        let scratch = tempfile::tempdir().expect("scratch");
        let dir = scratch.path().join(lane_dir_name(3));
        let mut lane = open_lane(scratch.path(), 1024 * 1024);
        for index in 0..3u64 {
            let record = test_mutation(9, index + 1, &format!("k{index}"));
            lane.append_batch(&PersistIntent {
                records: std::slice::from_ref(&record),
                level: DurabilityLevel::Sync,
            })
            .expect("append");
        }
        drop(lane);
        let path = dir.join(segment_file_name(1));
        let full = fs::read(&path).expect("segment bytes");
        // Interesting cut points: inside the segment header, batch headers,
        // record bodies, footers, and exactly on batch boundaries.
        let mut cuts = vec![0, 1, 55, SEGMENT_HEADER_LEN, SEGMENT_HEADER_LEN + 1];
        let mut offset = SEGMENT_HEADER_LEN;
        for _ in 0..3 {
            let body_len =
                u32::from_le_bytes(full[offset + 16..offset + 20].try_into().unwrap()) as usize;
            let batch_len = BATCH_HEADER_LEN + body_len + BATCH_FOOTER_LEN;
            cuts.push(offset + 1);
            cuts.push(offset + BATCH_HEADER_LEN - 1);
            cuts.push(offset + BATCH_HEADER_LEN + 1);
            cuts.push(offset + BATCH_HEADER_LEN + body_len / 2);
            cuts.push(offset + batch_len - BATCH_FOOTER_LEN / 2);
            cuts.push(offset + batch_len);
            offset += batch_len;
        }
        assert_eq!(offset, full.len(), "test walks whole file");
        for cut in cuts {
            let truncated = full[..cut.min(full.len())].to_vec();
            fs::write(&path, &truncated).expect("truncate");
            let mut lane = open_lane(scratch.path(), 1024 * 1024);
            let recovery = lane.recover().expect("torn tail always recovers");
            // Every surviving record ends at or before the cut, and the
            // truncation accounting equals (bytes present) - (bytes kept),
            // where a cut exactly on a boundary keeps everything (clean).
            let mut kept = if cut >= SEGMENT_HEADER_LEN {
                SEGMENT_HEADER_LEN
            } else {
                0
            };
            for entry in &recovery.records {
                let end = batch_end(&full, entry.batch_seq);
                assert!(
                    end <= cut,
                    "batch {} ends at {end}, cut at {cut}",
                    entry.batch_seq
                );
                kept = kept.max(end);
            }
            assert_eq!(
                recovery.truncated_bytes,
                u64::try_from(truncated.len().saturating_sub(kept)).expect("test file fits u64"),
                "truncation accounting exact at cut {cut}"
            );
        }
        // Restore the full file: clean recovery reports zero truncation.
        fs::write(&path, &full).expect("restore");
        let mut lane = open_lane(scratch.path(), 1024 * 1024);
        let recovery = lane.recover().expect("clean");
        assert_eq!(recovery.records.len(), 3);
        assert_eq!(recovery.truncated_bytes, 0);
    }

    /// End offset (exclusive) of the `n`th batch (1-based) in `file`.
    fn batch_end(file: &[u8], n: u64) -> usize {
        let mut offset = SEGMENT_HEADER_LEN;
        for _ in 1..=n {
            let body_len =
                u32::from_le_bytes(file[offset + 16..offset + 20].try_into().unwrap()) as usize;
            offset += BATCH_HEADER_LEN + body_len + BATCH_FOOTER_LEN;
        }
        offset
    }

    #[test]
    fn every_single_bit_flip_fails_recovery() {
        let scratch = tempfile::tempdir().expect("scratch");
        let dir = scratch.path().join(lane_dir_name(3));
        let mut lane = open_lane(scratch.path(), 1024 * 1024);
        for index in 0..2u64 {
            let record = test_mutation(9, index + 1, &format!("k{index}"));
            lane.append_batch(&PersistIntent {
                records: std::slice::from_ref(&record),
                level: DurabilityLevel::Sync,
            })
            .expect("append");
        }
        drop(lane);
        let path = dir.join(segment_file_name(1));
        let full = fs::read(&path).expect("segment bytes");
        // Every byte is CRC-covered or value-checked: any single-bit flip
        // must fail loudly, never recover wrong data.
        for index in 0..full.len() {
            let mut damaged = full.clone();
            damaged[index] ^= 0x01;
            fs::write(&path, &damaged).expect("damage");
            let mut lane = open_lane(scratch.path(), 1024 * 1024);
            assert!(
                lane.recover().is_err(),
                "bit flip at byte {index} went undetected"
            );
        }
        fs::write(&path, &full).expect("restore");
    }

    #[test]
    fn corrupt_middle_batch_fails_even_on_last_segment() {
        let scratch = tempfile::tempdir().expect("scratch");
        let dir = scratch.path().join(lane_dir_name(3));
        let mut lane = open_lane(scratch.path(), 1024 * 1024);
        for index in 0..3u64 {
            let record = test_mutation(9, index + 1, &format!("k{index}"));
            lane.append_batch(&PersistIntent {
                records: std::slice::from_ref(&record),
                level: DurabilityLevel::Sync,
            })
            .expect("append");
        }
        drop(lane);
        let path = dir.join(segment_file_name(1));
        let mut damaged = fs::read(&path).expect("segment bytes");
        // Corrupt a body byte of the FIRST batch (complete bytes, bad CRC):
        // committed history must fail, never truncate.
        let first_body = SEGMENT_HEADER_LEN + BATCH_HEADER_LEN + 5;
        damaged[first_body] ^= 0xFF;
        fs::write(&path, &damaged).expect("damage");
        let mut lane = open_lane(scratch.path(), 1024 * 1024);
        let err = lane.recover().expect_err("middle corruption fails");
        assert!(
            matches!(err, RecoveryError::CorruptSegment { .. }),
            "fail loudly, got {err:?}"
        );
    }

    #[test]
    fn foreign_identity_is_misdirected() {
        let scratch = tempfile::tempdir().expect("scratch");
        let mut lane = open_lane(scratch.path(), 1024 * 1024);
        let record = test_mutation(9, 1, "a");
        lane.append_batch(&PersistIntent {
            records: std::slice::from_ref(&record),
            level: DurabilityLevel::Sync,
        })
        .expect("append");
        drop(lane);
        let foreign = LaneIdentity {
            cluster: ClusterId::from_u128(0xDEAD),
            node: NodeId::from_u64(NODE),
            incarnation: NodeIncarnation::INITIAL,
        };
        let mut lane = LocalWalLane::open(scratch.path(), 3, foreign, 1024 * 1024).expect("open");
        let err = lane.recover().expect_err("foreign cluster fails");
        assert!(
            matches!(err, RecoveryError::Misdirected { .. }),
            "got {err:?}"
        );
    }

    #[test]
    fn segment_gap_fails() {
        let scratch = tempfile::tempdir().expect("scratch");
        let dir = scratch.path().join(lane_dir_name(3));
        let mut lane = open_lane(scratch.path(), 200);
        for index in 0..6u64 {
            let record = test_mutation(9, index + 1, &format!("k{index}"));
            lane.append_batch(&PersistIntent {
                records: std::slice::from_ref(&record),
                level: DurabilityLevel::Sync,
            })
            .expect("append");
        }
        assert!(lane.segment_count() >= 3);
        drop(lane);
        fs::remove_file(dir.join(segment_file_name(2))).expect("remove middle");
        let mut lane = open_lane(scratch.path(), 200);
        let err = lane.recover().expect_err("gap fails");
        assert!(
            matches!(err, RecoveryError::SegmentGap { expected: 2, .. }),
            "got {err:?}"
        );
    }

    #[test]
    fn duplicate_batch_sequence_fails() {
        let scratch = tempfile::tempdir().expect("scratch");
        let dir = scratch.path().join(lane_dir_name(3));
        let mut lane = open_lane(scratch.path(), 1024 * 1024);
        for index in 0..2u64 {
            let record = test_mutation(9, index + 1, &format!("k{index}"));
            lane.append_batch(&PersistIntent {
                records: std::slice::from_ref(&record),
                level: DurabilityLevel::Sync,
            })
            .expect("append");
        }
        drop(lane);
        let path = dir.join(segment_file_name(1));
        let full = fs::read(&path).expect("segment bytes");
        // Splice a copy of batch 1 right after itself: complete bytes,
        // duplicate sequence — corruption, not a tear.
        let first_len = batch_end(&full, 1) - SEGMENT_HEADER_LEN;
        let mut doubled = Vec::new();
        doubled.extend_from_slice(&full[..batch_end(&full, 1)]);
        doubled.extend_from_slice(&full[SEGMENT_HEADER_LEN..SEGMENT_HEADER_LEN + first_len]);
        doubled.extend_from_slice(&full[batch_end(&full, 1)..]);
        fs::write(&path, &doubled).expect("splice");
        let mut lane = open_lane(scratch.path(), 1024 * 1024);
        assert!(lane.recover().is_err(), "duplicate sequence fails");
    }

    #[test]
    fn oversized_body_declaration_fails_before_allocating() {
        let scratch = tempfile::tempdir().expect("scratch");
        let dir = scratch.path().join(lane_dir_name(3));
        let mut lane = open_lane(scratch.path(), 1024 * 1024);
        let record = test_mutation(9, 1, "a");
        lane.append_batch(&PersistIntent {
            records: std::slice::from_ref(&record),
            level: DurabilityLevel::Sync,
        })
        .expect("append");
        drop(lane);
        let path = dir.join(segment_file_name(1));
        let mut damaged = fs::read(&path).expect("segment bytes");
        // Rewrite the batch's body length to 1 GiB without adding payload,
        // repairing the header CRC so the lie reaches the guard.
        let header_at = SEGMENT_HEADER_LEN;
        damaged[header_at + 16..header_at + 20].copy_from_slice(&(1u32 << 30).to_le_bytes());
        let crc = kivi_codec::integrity::crc32c_checksum(&damaged[header_at..header_at + 36]);
        damaged[header_at + 36..header_at + 40].copy_from_slice(&crc.to_le_bytes());
        fs::write(&path, &damaged).expect("damage");
        let mut lane = open_lane(scratch.path(), 1024 * 1024);
        let err = lane.recover().expect_err("absurd length fails");
        assert!(
            matches!(err, RecoveryError::Oversized { .. }),
            "allocation guard fires, got {err:?}"
        );
    }
}
