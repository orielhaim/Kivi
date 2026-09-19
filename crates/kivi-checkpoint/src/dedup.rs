//! Dedup-component format: the session state a checkpoint must carry so
//! safe retry survives WAL reclamation, plus prepared transaction intents
//! (deterministic state that must survive restart exactly like sessions).
//!
//! The component persists exactly the logically necessary state per
//! session: the acknowledged floor plus retained outcomes above it, in
//! deterministic sorted order; then prepared intents in key order. It is a
//! separate immutable artifact (not mixed into object bands) because dedup
//! and intent state have different growth and lifecycle characteristics
//! than objects — and they must scale without disturbing band reuse.
//!
//! ```text
//! header (32 bytes, fixed):
//!   0   4  magic "KVCD"
//!   4   2  major = 3
//!   6   2  minor = 0
//!   8   8  tablet (u64)
//!   16  8  cut commit position (u64)
//!   24  4  session count (u32)
//!   28  4  CRC32C of bytes [0..28]
//! body: sessions sorted by session id:
//!   session u128, floor u64, entry_count u32,
//!   entries sorted by seq: seq u64, commit u64, opcode u8,
//!     outcome_len u32, outcome[..] (DurableOutcome codec)
//! then intents sorted by key:
//!   intent_count u32,
//!   txn16, coordinator u64, key (Key codec), expect (TxnExpect codec),
//!     write (TxnWriteKind codec), digest32, observed flag u8 (+ version u64),
//!     prepared_at u64
//! footer (40 bytes): body CRC32C, BLAKE3 of header[0..28] + body
//! (content identity covers artifact identity), footer CRC32C
//! (same uniform footer as bands)
//! ```
//!
//! Recovery restores floors verbatim and reinstalls retained outcomes, so
//! a retry of an above-floor identity hits exactly as before the
//! checkpoint, while a below-floor identity expires exactly as before.
//! Intents restore verbatim: unresolved transactions are never discarded
//! on restart.
//!
//! Format v1 (sessions only, no intents) and v2 (intents without
//! write-set digests) are rejected: breaking prototype formats is
//! allowed, and silently dropping intents — or their confused-deputy
//! binding — would violate the durability contract.

use kivi_codec::integrity::crc32c_checksum;
use kivi_codec::{Decode as _, Encode as _};
use kivi_types::{CommitPosition, RequestSeq, SessionId, TabletId};

use crate::artifact::ArtifactHash;
use crate::band::band_take;
use crate::error::CheckpointError;

/// Magic word: ASCII `"KVCD"` read as a little-endian `u32`.
pub const DEDUP_MAGIC: u32 = 0x4443_564B;
/// Current dedup-component major version (v3 binds intents to their
/// write-set digests; v1/v2 are rejected loudly).
pub const DEDUP_MAJOR: u16 = 3;
/// Current dedup-component minor version.
pub const DEDUP_MINOR: u16 = 0;
/// Encoded header length in bytes.
pub const DEDUP_HEADER_LEN: usize = 32;
/// Encoded footer length in bytes (uniform with bands).
pub const DEDUP_FOOTER_LEN: usize = 40;

/// One session's checkpointed dedup state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SessionCheckpoint {
    /// Client session.
    pub session: SessionId,
    /// Acknowledged floor (retries at or below this expire).
    pub floor: RequestSeq,
    /// Retained outcomes above the floor, sorted by sequence.
    pub outcomes: Vec<OutcomeCheckpoint>,
}

/// One retained outcome above the floor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OutcomeCheckpoint {
    /// Request sequence.
    pub seq: RequestSeq,
    /// Commit position that installed it.
    pub commit: CommitPosition,
    /// Originating opcode discriminant.
    pub opcode: u8,
    /// The recorded outcome.
    pub outcome: kivi_state::DurableOutcome,
}

/// A built dedup component: content identity plus file bytes.
#[derive(Debug, Clone)]
pub struct BuiltDedup {
    /// BLAKE3 of the body (filename and manifest reference).
    pub hash: ArtifactHash,
    /// Complete file bytes.
    pub bytes: Vec<u8>,
    /// Sessions encoded.
    pub sessions: u32,
    /// Outcomes encoded.
    pub outcomes: u64,
}

/// A verified loaded dedup component.
#[derive(Debug, Clone)]
pub struct LoadedDedup {
    /// Tablet the component belongs to.
    pub tablet: TabletId,
    /// Checkpoint cut taken at.
    pub cut: u64,
    /// Sessions sorted by session id.
    pub sessions: Vec<SessionCheckpoint>,
    /// Prepared intents sorted by key.
    pub intents: Vec<kivi_state::TxnIntent>,
}

/// Builds one immutable dedup component from unsorted session snapshots
/// plus prepared intents.
///
/// # Errors
///
/// Returns [`CheckpointError`] on duplicate sessions/sequences or
/// oversized encodings.
pub fn build_dedup(
    tablet: TabletId,
    cut: u64,
    mut sessions: Vec<SessionCheckpoint>,
    mut intents: Vec<kivi_state::TxnIntent>,
) -> Result<BuiltDedup, CheckpointError> {
    sessions.sort_by_key(|session| session.session.as_u128());
    for pair in sessions.windows(2) {
        if pair[0].session == pair[1].session {
            return Err(CheckpointError::Format {
                detail: "duplicate session in dedup component".to_owned(),
            });
        }
    }
    let mut body = Vec::new();
    let mut outcome_total = 0u64;
    for session in &mut sessions {
        session.outcomes.sort_by_key(|outcome| outcome.seq.as_u64());
        let mut previous_seq: Option<u64> = None;
        for outcome in &session.outcomes {
            if previous_seq.is_some_and(|previous| previous >= outcome.seq.as_u64()) {
                return Err(CheckpointError::Format {
                    detail: "duplicate sequence in dedup session".to_owned(),
                });
            }
            // Retained outcomes must sit strictly above the floor: anything
            // else is a builder bug that would corrupt retry semantics.
            if outcome.seq.as_u64() <= session.floor.as_u64() {
                return Err(CheckpointError::Format {
                    detail: "dedup outcome at or below the session floor".to_owned(),
                });
            }
            previous_seq = Some(outcome.seq.as_u64());
        }
        body.extend_from_slice(&session.session.as_u128().to_le_bytes());
        body.extend_from_slice(&session.floor.as_u64().to_le_bytes());
        let count = u32::try_from(session.outcomes.len()).map_err(|_| CheckpointError::Format {
            detail: "dedup session exceeds u32 outcomes".to_owned(),
        })?;
        body.extend_from_slice(&count.to_le_bytes());
        for outcome in &session.outcomes {
            body.extend_from_slice(&outcome.seq.as_u64().to_le_bytes());
            body.extend_from_slice(&outcome.commit.as_u64().to_le_bytes());
            body.push(outcome.opcode);
            let mut encoded = Vec::new();
            outcome.outcome.encode(&mut encoded);
            let len = u32::try_from(encoded.len()).map_err(|_| CheckpointError::Format {
                detail: "dedup outcome exceeds u32 length".to_owned(),
            })?;
            body.extend_from_slice(&len.to_le_bytes());
            body.extend_from_slice(&encoded);
            outcome_total += 1;
        }
    }
    let session_total = u32::try_from(sessions.len()).map_err(|_| CheckpointError::Format {
        detail: "dedup component exceeds u32 sessions".to_owned(),
    })?;
    encode_intents(&mut body, &mut intents)?;
    let mut header = [0u8; DEDUP_HEADER_LEN];
    header[0..4].copy_from_slice(&DEDUP_MAGIC.to_le_bytes());
    header[4..6].copy_from_slice(&DEDUP_MAJOR.to_le_bytes());
    header[6..8].copy_from_slice(&DEDUP_MINOR.to_le_bytes());
    header[8..16].copy_from_slice(&tablet.as_u64().to_le_bytes());
    header[16..24].copy_from_slice(&cut.to_le_bytes());
    header[24..28].copy_from_slice(&session_total.to_le_bytes());
    let header_crc = crc32c_checksum(&header[..28]);
    header[28..32].copy_from_slice(&header_crc.to_le_bytes());
    let body_crc = crc32c_checksum(&body);
    let content_hash = crate::artifact::ArtifactHash::of_chunks(&[&header[..28], &body]);
    let mut footer = [0u8; DEDUP_FOOTER_LEN];
    footer[0..4].copy_from_slice(&body_crc.to_le_bytes());
    footer[4..36].copy_from_slice(content_hash.as_bytes());
    let footer_crc = crc32c_checksum(&footer[..36]);
    footer[36..40].copy_from_slice(&footer_crc.to_le_bytes());
    let mut bytes = Vec::with_capacity(DEDUP_HEADER_LEN + body.len() + DEDUP_FOOTER_LEN);
    bytes.extend_from_slice(&header);
    bytes.extend_from_slice(&body);
    bytes.extend_from_slice(&footer);
    Ok(BuiltDedup {
        hash: content_hash,
        bytes,
        sessions: session_total,
        outcomes: outcome_total,
    })
}

/// Encodes the intent section: count plus key-ordered intents (duplicate
/// keys rejected — one intent per key by construction).
fn encode_intents(
    body: &mut Vec<u8>,
    intents: &mut [kivi_state::TxnIntent],
) -> Result<(), CheckpointError> {
    intents.sort_by(|left, right| left.key.as_bytes().cmp(right.key.as_bytes()));
    for pair in intents.windows(2) {
        if pair[0].key == pair[1].key {
            return Err(CheckpointError::Format {
                detail: "duplicate intent key in dedup component".to_owned(),
            });
        }
    }
    let count = u32::try_from(intents.len()).map_err(|_| CheckpointError::Format {
        detail: "dedup component exceeds u32 intents".to_owned(),
    })?;
    body.extend_from_slice(&count.to_le_bytes());
    for intent in intents.iter() {
        body.extend_from_slice(&intent.id.as_bytes());
        body.extend_from_slice(&intent.coordinator.as_u64().to_le_bytes());
        intent.key.encode(body);
        intent.write.expect.encode(body);
        intent.write.kind.encode(body);
        body.extend_from_slice(&intent.digest);
        match intent.observed {
            None => body.push(0),
            Some(version) => {
                body.push(1);
                version.encode(body);
            }
        }
        body.extend_from_slice(&intent.prepared_at.to_le_bytes());
    }
    Ok(())
}

/// Takes `n` bytes at the cursor (intent-section walker).
fn intent_take<'a>(input: &'a [u8], at: &mut usize, n: usize) -> Result<&'a [u8], CheckpointError> {
    if input.len() < *at + n {
        return Err(CheckpointError::Corrupt {
            detail: "dedup intent section truncated".to_owned(),
        });
    }
    let slice = &input[*at..*at + n];
    *at += n;
    Ok(slice)
}

/// Decodes the intent section at the cursor (exactly `count` entries,
/// key-ordered, then end of section).
fn decode_intents(cursor: &[u8]) -> Result<(Vec<kivi_state::TxnIntent>, usize), CheckpointError> {
    let corrupt = |detail: String| CheckpointError::Corrupt { detail };
    let mut at = 0usize;
    let count = u32::from_le_bytes(
        intent_take(cursor, &mut at, 4)?
            .try_into()
            .map_err(|_| corrupt("dedup intent count unreadable".to_owned()))?,
    ) as usize;
    if count > 1_000_000 {
        return Err(corrupt("dedup intent count absurd".to_owned()));
    }
    let mut intents = Vec::with_capacity(count.min(1024));
    let mut previous: Option<Vec<u8>> = None;
    for _ in 0..count {
        let raw = intent_take(cursor, &mut at, 16)?;
        let id = kivi_state::TxnId::from_bytes(raw.try_into().unwrap_or([0; 16]));
        let coordinator = kivi_types::TabletId::from_u64(u64::from_le_bytes(
            intent_take(cursor, &mut at, 8)?
                .try_into()
                .map_err(|_| corrupt("dedup intent coordinator unreadable".to_owned()))?,
        ));
        let (key, used) = kivi_state::Key::decode(&cursor[at..])
            .map_err(|_| corrupt("dedup intent key undecodable".to_owned()))?;
        at += used;
        let (expect, used) = kivi_state::TxnExpect::decode(&cursor[at..])
            .map_err(|_| corrupt("dedup intent expectation undecodable".to_owned()))?;
        at += used;
        let (kind, used) = kivi_state::TxnWriteKind::decode(&cursor[at..])
            .map_err(|_| corrupt("dedup intent write undecodable".to_owned()))?;
        at += used;
        let digest: [u8; 32] = intent_take(cursor, &mut at, 32)?
            .try_into()
            .map_err(|_| corrupt("dedup intent digest unreadable".to_owned()))?;
        let observed = match intent_take(cursor, &mut at, 1)?[0] {
            0 => None,
            1 => {
                let (version, used) = kivi_state::ObjectVersion::decode(&cursor[at..])
                    .map_err(|_| corrupt("dedup intent version undecodable".to_owned()))?;
                at += used;
                Some(version)
            }
            _ => {
                return Err(corrupt("dedup intent observed tag invalid".to_owned()));
            }
        };
        let prepared_at = u64::from_le_bytes(
            intent_take(cursor, &mut at, 8)?
                .try_into()
                .map_err(|_| corrupt("dedup intent timestamp unreadable".to_owned()))?,
        );
        if previous
            .as_ref()
            .is_some_and(|prev| prev >= &key.as_bytes().to_vec())
        {
            return Err(corrupt(
                "dedup intents out of order or duplicated".to_owned(),
            ));
        }
        previous = Some(key.as_bytes().to_vec());
        intents.push(kivi_state::TxnIntent {
            id,
            coordinator,
            key: key.clone(),
            write: kivi_state::TxnWrite { key, kind, expect },
            digest,
            observed,
            prepared_at,
        });
    }
    Ok((intents, at))
}
/// header CRC and provenance, body CRC and content hash, session/sequence
/// ordering, floor invariants, and outcome decoding.
///
/// # Errors
///
/// Returns [`CheckpointError`] on any integrity or format violation.
// Long linear field-by-field decoder (like the WAL scanner).
#[allow(clippy::too_many_lines)]
pub fn load_dedup(
    expected_tablet: TabletId,
    expected_hash: ArtifactHash,
    bytes: &[u8],
) -> Result<LoadedDedup, CheckpointError> {
    let corrupt = |detail: String| CheckpointError::Corrupt { detail };
    let format = |detail: String| CheckpointError::Format { detail };
    if bytes.len() < DEDUP_HEADER_LEN + DEDUP_FOOTER_LEN {
        return Err(corrupt(format!(
            "dedup file shorter than header + footer: {} bytes",
            bytes.len()
        )));
    }
    let (header, rest) = bytes.split_at(DEDUP_HEADER_LEN);
    let (body, footer) = rest.split_at(rest.len() - DEDUP_FOOTER_LEN);
    let u32_at = |range: std::ops::Range<usize>, context: &'static str| {
        header
            .get(range)
            .and_then(|slice| slice.try_into().ok())
            .map(u32::from_le_bytes)
            .ok_or_else(|| corrupt(format!("dedup {context} unreadable")))
    };
    if u32_at(0..4, "magic")? != DEDUP_MAGIC {
        return Err(format("dedup magic mismatch".to_owned()));
    }
    if crc32c_checksum(&header[..28]) != u32_at(28..32, "header CRC")? {
        return Err(corrupt("dedup header CRC mismatch".to_owned()));
    }
    let major = u16::from_le_bytes(
        header[4..6]
            .try_into()
            .map_err(|_| corrupt("dedup version unreadable".to_owned()))?,
    );
    let minor = u16::from_le_bytes(
        header[6..8]
            .try_into()
            .map_err(|_| corrupt("dedup version unreadable".to_owned()))?,
    );
    if major != DEDUP_MAJOR || minor != DEDUP_MINOR {
        return Err(CheckpointError::Unsupported {
            detail: format!("dedup version {major}.{minor}"),
        });
    }
    let tablet = TabletId::from_u64(u64::from_le_bytes(
        header[8..16]
            .try_into()
            .map_err(|_| corrupt("dedup tablet unreadable".to_owned()))?,
    ));
    if tablet != expected_tablet {
        return Err(format(format!(
            "dedup tablet {tablet} does not match descriptor"
        )));
    }
    let cut = u64::from_le_bytes(
        header[16..24]
            .try_into()
            .map_err(|_| corrupt("dedup cut unreadable".to_owned()))?,
    );
    let session_count = u32_at(24..28, "session count")? as usize;
    if u32::from_le_bytes(
        footer[0..4]
            .try_into()
            .map_err(|_| corrupt("dedup footer unreadable".to_owned()))?,
    ) != crc32c_checksum(body)
    {
        return Err(corrupt("dedup body CRC mismatch".to_owned()));
    }
    let content_hash: [u8; 32] = footer[4..36]
        .try_into()
        .map_err(|_| corrupt("dedup content hash unreadable".to_owned()))?;
    if crate::artifact::ArtifactHash::of_chunks(&[&header[..28], body]).as_bytes() != &content_hash
    {
        return Err(corrupt("dedup BLAKE3 content hash mismatch".to_owned()));
    }
    if ArtifactHash::from_bytes(content_hash) != expected_hash {
        return Err(corrupt(
            "dedup content hash does not match its manifest reference".to_owned(),
        ));
    }
    if crc32c_checksum(&footer[..36])
        != u32::from_le_bytes(
            footer[36..40]
                .try_into()
                .map_err(|_| corrupt("dedup footer CRC unreadable".to_owned()))?,
        )
    {
        return Err(corrupt("dedup footer CRC mismatch".to_owned()));
    }
    let mut sessions = Vec::with_capacity(session_count.min(1_000_000));
    let mut cursor = body;
    let mut previous_session: Option<u128> = None;
    for _ in 0..session_count {
        if cursor.is_empty() {
            return Err(corrupt(
                "dedup session count exceeds encoded sessions".to_owned(),
            ));
        }
        let (session, consumed) = decode_session(cursor)
            .map_err(|error| corrupt(format!("dedup session undecodable: {error:?}")))?;
        if previous_session.is_some_and(|previous| previous >= session.session.as_u128()) {
            return Err(corrupt(
                "dedup sessions out of order or duplicated".to_owned(),
            ));
        }
        previous_session = Some(session.session.as_u128());
        sessions.push(session);
        cursor = &cursor[consumed..];
    }
    let (intents, used) = decode_intents(cursor)?;
    cursor = &cursor[used..];
    if !cursor.is_empty() {
        return Err(corrupt("dedup body has trailing bytes".to_owned()));
    }
    Ok(LoadedDedup {
        tablet,
        cut,
        sessions,
        intents,
    })
}

fn decode_session(input: &[u8]) -> Result<(SessionCheckpoint, usize), kivi_codec::CodecError> {
    use kivi_codec::CodecError;
    let mut at = 0usize;
    let session = SessionId::from_u128(u128::from_le_bytes(band_take(input, &mut at)?));
    let floor = RequestSeq::from_u64(u64::from_le_bytes(band_take(input, &mut at)?));
    let count = u32::from_le_bytes(band_take(input, &mut at)?) as usize;
    let mut outcomes = Vec::with_capacity(count.min(1_000_000));
    let mut previous_seq: Option<u64> = None;
    for _ in 0..count {
        let seq = RequestSeq::from_u64(u64::from_le_bytes(band_take(input, &mut at)?));
        if previous_seq.is_some_and(|previous| previous >= seq.as_u64()) {
            return Err(CodecError::InvalidTag {
                kind: "dedup sequence order",
                tag: 0,
            });
        }
        if seq.as_u64() <= floor.as_u64() {
            return Err(CodecError::InvalidTag {
                kind: "dedup outcome at or below floor",
                tag: 0,
            });
        }
        previous_seq = Some(seq.as_u64());
        let commit =
            kivi_types::CommitPosition::from_u64(u64::from_le_bytes(band_take(input, &mut at)?));
        let opcode: [u8; 1] = band_take(input, &mut at)?;
        let len = u32::from_le_bytes(band_take(input, &mut at)?) as usize;
        if input.len() < at + len {
            return Err(CodecError::Truncated {
                expected: at + len,
                available: input.len(),
            });
        }
        let outcome = kivi_state::DurableOutcome::decode_exact(&input[at..at + len])?;
        at += len;
        outcomes.push(OutcomeCheckpoint {
            seq,
            commit,
            opcode: opcode[0],
            outcome,
        });
    }
    Ok((
        SessionCheckpoint {
            session,
            floor,
            outcomes,
        },
        at,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use kivi_state::{ObjectVersion, OperationResult};
    use kivi_types::RequestSeq;

    fn session(id: u128, floor: u64, seqs: &[u64]) -> SessionCheckpoint {
        SessionCheckpoint {
            session: SessionId::from_u128(id),
            floor: RequestSeq::from_u64(floor),
            outcomes: seqs
                .iter()
                .map(|seq| OutcomeCheckpoint {
                    seq: RequestSeq::from_u64(*seq),
                    commit: CommitPosition::from_u64(*seq),
                    opcode: 6,
                    outcome: kivi_state::DurableOutcome::Completed(
                        OperationResult::CounterUpdated {
                            value: i64::try_from(*seq).expect("test seq fits i64"),
                            version: ObjectVersion::FIRST,
                        },
                    ),
                })
                .collect(),
        }
    }

    #[test]
    fn dedup_round_trips_sorted() {
        let built = build_dedup(
            TabletId::from_u64(1),
            12,
            vec![session(9, 4, &[5, 7]), session(3, 0, &[1])],
            Vec::new(),
        )
        .expect("builds");
        assert_eq!(built.sessions, 2);
        assert_eq!(built.outcomes, 3);
        let loaded = load_dedup(TabletId::from_u64(1), built.hash, &built.bytes).expect("loads");
        assert_eq!(loaded.cut, 12);
        // Sorted by session regardless of input order.
        assert_eq!(loaded.sessions[0].session, SessionId::from_u128(3));
        assert_eq!(loaded.sessions[1].outcomes.len(), 2);
        assert!(loaded.intents.is_empty());
    }

    #[test]
    fn empty_dedup_is_valid() {
        let built = build_dedup(TabletId::from_u64(1), 1, Vec::new(), Vec::new()).expect("builds");
        let loaded = load_dedup(TabletId::from_u64(1), built.hash, &built.bytes).expect("loads");
        assert!(loaded.sessions.is_empty());
        assert!(loaded.intents.is_empty());
    }

    #[test]
    fn outcomes_at_or_below_floor_are_rejected() {
        assert!(
            build_dedup(
                TabletId::from_u64(1),
                1,
                vec![session(1, 5, &[5])],
                Vec::new()
            )
            .is_err(),
            "outcome at the floor is not retained state"
        );
        assert!(
            build_dedup(
                TabletId::from_u64(1),
                1,
                vec![session(1, 9, &[5])],
                Vec::new()
            )
            .is_err(),
            "outcome below the floor is not retained state"
        );
    }

    #[test]
    fn corruption_is_detected() {
        let built = build_dedup(
            TabletId::from_u64(1),
            1,
            vec![session(1, 0, &[1, 2])],
            Vec::new(),
        )
        .expect("builds");
        let mut damaged = built.bytes.clone();
        damaged[DEDUP_HEADER_LEN + 3] ^= 0xFF;
        assert!(load_dedup(TabletId::from_u64(1), built.hash, &damaged).is_err());
    }

    #[test]
    fn intents_round_trip_sorted_and_restore_verbatim() {
        use kivi_state::{Key, TxnExpect, TxnId, TxnIntent, TxnWrite, TxnWriteKind};
        let intent = |key: &str| TxnIntent {
            id: TxnId::derive(7, 1, 0),
            coordinator: TabletId::from_u64(9),
            key: Key::from(key),
            write: TxnWrite {
                key: Key::from(key),
                kind: TxnWriteKind::Put(bytes::Bytes::from_static(b"v")),
                expect: TxnExpect::Any,
            },
            digest: [0xD1; 32],
            observed: None,
            prepared_at: 99,
        };
        let built = build_dedup(
            TabletId::from_u64(1),
            12,
            Vec::new(),
            vec![intent("b"), intent("a")],
        )
        .expect("builds");
        let loaded = load_dedup(TabletId::from_u64(1), built.hash, &built.bytes).expect("loads");
        assert_eq!(loaded.intents.len(), 2);
        // Sorted by key regardless of input order.
        assert_eq!(loaded.intents[0].key, Key::from("a"));
        assert_eq!(loaded.intents[1].key, Key::from("b"));
        assert_eq!(loaded.intents[0].prepared_at, 99);
    }
}
