//! Installed-checkpoint catalog: CURRENT pointers and the WAL floor.
//!
//! Recovery never scans directories for "the newest checkpoint".
//! Instead each tablet owns one explicit durable pointer record
//! (`CURRENT`), and the WAL reclamation floor lives in one explicit
//! durable record (`WAL_FLOOR`). Both are fixed-size versioned records
//! published through crash-safe atomic writes.
//!
//! ```text
//! CURRENT (81 bytes, fixed):
//!   0   4  magic "KVCC"
//!   4   2  major = 1
//!   6   2  minor = 0
//!   8   8  tablet (u64)
//!   16  8  cut commit position (u64)
//!   24  32 manifest content hash
//!   56  1  has_prev (0/1)
//!   57  32 previous manifest content hash (zero when absent)
//!   89  8  created wall micros, signed Unix time (forensics/age only)
//!   97  4  CRC32C of bytes [0..97]
//!
//! WAL_FLOOR (variable):
//!   0   4  magic "KVCF"
//!   4   2  major = 1
//!   6   2  minor = 0
//!   8   4  lane entry count (u32)
//!   12 ..  per lane (20 bytes): lane u16, reserved u16,
//!          first_segment u64, first_batch u64
//!   ..      next_batch u64 (per lane, after first_batch)
//!   ..  4  CRC32C of all preceding bytes
//! ```
//!
//! Lane entry (28 bytes): `lane` u16, reserved u16, `first_segment` u64,
//! `first_batch` u64, `next_batch` u64.
//!
//! Publication order (spec V) is orchestrated by the publisher: bands,
//! then dedup, then manifest, then `CURRENT` - each step synced before the
//! next becomes visible. `WAL_FLOOR` advances only after the checkpoints
//! that justify it are installed.

use kivi_durability::LaneFloor;
use kivi_types::TabletId;

use crate::artifact::ArtifactHash;
use crate::error::CheckpointError;

/// Magic word: ASCII `"KVCC"` read as a little-endian `u32`.
pub const CURRENT_MAGIC: u32 = 0x4343_564B;
/// Current pointer major version.
pub const CURRENT_MAJOR: u16 = 1;
/// Current pointer minor version.
pub const CURRENT_MINOR: u16 = 0;
/// Encoded CURRENT length in bytes.
pub const CURRENT_LEN: usize = 101;

/// Magic word: ASCII `"KVCF"` read as a little-endian `u32`.
pub const WAL_FLOOR_MAGIC: u32 = 0x4643_564B;
/// Current floor major version.
pub const WAL_FLOOR_MAJOR: u16 = 1;
/// Current floor minor version.
pub const WAL_FLOOR_MINOR: u16 = 0;
/// Encoded lane-floor entry length in bytes.
pub const LANE_FLOOR_ENTRY_LEN: usize = 28;

/// One tablet's installed-checkpoint pointer (current + previous).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CurrentRecord {
    /// Tablet pointed at.
    pub tablet: TabletId,
    /// Installed cut.
    pub cut: u64,
    /// Installed manifest hash.
    pub manifest: ArtifactHash,
    /// Previous manifest hash (retention chain), if any.
    pub previous: Option<ArtifactHash>,
    /// Wall time at publication (age reporting only).
    pub created_wall_micros: kivi_types::WallTimestamp,
}

/// Encodes one CURRENT record (fixed size, checksummed).
#[must_use]
pub fn encode_current(record: &CurrentRecord) -> Vec<u8> {
    use kivi_codec::integrity::crc32c_checksum;
    let mut out = [0u8; CURRENT_LEN];
    out[0..4].copy_from_slice(&CURRENT_MAGIC.to_le_bytes());
    out[4..6].copy_from_slice(&CURRENT_MAJOR.to_le_bytes());
    out[6..8].copy_from_slice(&CURRENT_MINOR.to_le_bytes());
    out[8..16].copy_from_slice(&record.tablet.as_u64().to_le_bytes());
    out[16..24].copy_from_slice(&record.cut.to_le_bytes());
    out[24..56].copy_from_slice(record.manifest.as_bytes());
    match record.previous {
        None => {
            out[56] = 0;
        }
        Some(hash) => {
            out[56] = 1;
            out[57..89].copy_from_slice(hash.as_bytes());
        }
    }
    out[89..97].copy_from_slice(&record.created_wall_micros.as_micros().to_le_bytes());
    let crc = crc32c_checksum(&out[..97]);
    out[97..101].copy_from_slice(&crc.to_le_bytes());
    out.to_vec()
}

/// Decodes and verifies one CURRENT record.
///
/// # Errors
///
/// Returns [`CheckpointError`] on any integrity or format violation.
/// A missing file is [`CheckpointError::Absent`] (fresh tablet), never a
/// zeroed guess.
pub fn decode_current(tablet: TabletId, bytes: &[u8]) -> Result<CurrentRecord, CheckpointError> {
    use kivi_codec::integrity::crc32c_checksum;
    let corrupt = |detail: String| CheckpointError::Corrupt { detail };
    let format = |detail: String| CheckpointError::Format { detail };
    if bytes.len() != CURRENT_LEN {
        return Err(corrupt(format!(
            "CURRENT length {} is not {CURRENT_LEN}",
            bytes.len()
        )));
    }
    if u32::from_le_bytes(
        bytes[0..4]
            .try_into()
            .map_err(|_| corrupt("CURRENT magic unreadable".to_owned()))?,
    ) != CURRENT_MAGIC
    {
        return Err(format("CURRENT magic mismatch".to_owned()));
    }
    let major = u16::from_le_bytes(
        bytes[4..6]
            .try_into()
            .map_err(|_| corrupt("CURRENT version unreadable".to_owned()))?,
    );
    let minor = u16::from_le_bytes(
        bytes[6..8]
            .try_into()
            .map_err(|_| corrupt("CURRENT version unreadable".to_owned()))?,
    );
    if major != CURRENT_MAJOR || minor != CURRENT_MINOR {
        return Err(CheckpointError::Unsupported {
            detail: format!("CURRENT version {major}.{minor}"),
        });
    }
    if crc32c_checksum(&bytes[..97])
        != u32::from_le_bytes(
            bytes[97..101]
                .try_into()
                .map_err(|_| corrupt("CURRENT CRC unreadable".to_owned()))?,
        )
    {
        return Err(corrupt("CURRENT CRC mismatch".to_owned()));
    }
    let stored_tablet = TabletId::from_u64(u64::from_le_bytes(
        bytes[8..16]
            .try_into()
            .map_err(|_| corrupt("CURRENT tablet unreadable".to_owned()))?,
    ));
    if stored_tablet != tablet {
        return Err(format(format!(
            "CURRENT names tablet {stored_tablet}, expected {tablet}"
        )));
    }
    let cut = u64::from_le_bytes(
        bytes[16..24]
            .try_into()
            .map_err(|_| corrupt("CURRENT cut unreadable".to_owned()))?,
    );
    let manifest: [u8; 32] = bytes[24..56]
        .try_into()
        .map_err(|_| corrupt("CURRENT manifest hash unreadable".to_owned()))?;
    let previous = match bytes[56] {
        0 => {
            if bytes[57..89] != [0u8; 32] {
                return Err(format(
                    "CURRENT previous hash present without flag".to_owned(),
                ));
            }
            None
        }
        1 => {
            let hash: [u8; 32] = bytes[57..89]
                .try_into()
                .map_err(|_| corrupt("CURRENT previous hash unreadable".to_owned()))?;
            Some(ArtifactHash::from_bytes(hash))
        }
        flag => {
            return Err(format(format!(
                "CURRENT previous flag {flag} is not 0 or 1"
            )));
        }
    };
    let created_wall_micros = kivi_types::WallTimestamp::from_micros(i64::from_le_bytes(
        bytes[89..97]
            .try_into()
            .map_err(|_| corrupt("CURRENT timestamp unreadable".to_owned()))?,
    ));
    Ok(CurrentRecord {
        tablet,
        cut,
        manifest: ArtifactHash::from_bytes(manifest),
        previous,
        created_wall_micros,
    })
}

/// Encodes the durable WAL-floor record (sorted by lane for determinism).
///
/// # Errors
///
/// Returns [`CheckpointError`] when the lane count overflows `u32`.
pub fn encode_wal_floor(mut floors: Vec<LaneFloor>) -> Result<Vec<u8>, CheckpointError> {
    use kivi_codec::integrity::crc32c_checksum;
    floors.sort_by_key(|floor| floor.lane);
    let mut out = Vec::with_capacity(12 + floors.len() * LANE_FLOOR_ENTRY_LEN + 4);
    out.extend_from_slice(&WAL_FLOOR_MAGIC.to_le_bytes());
    out.extend_from_slice(&WAL_FLOOR_MAJOR.to_le_bytes());
    out.extend_from_slice(&WAL_FLOOR_MINOR.to_le_bytes());
    out.extend_from_slice(
        &u32::try_from(floors.len())
            .map_err(|_| CheckpointError::Format {
                detail: "WAL floor exceeds u32 lanes".to_owned(),
            })?
            .to_le_bytes(),
    );
    for floor in &floors {
        out.extend_from_slice(&floor.lane.to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes());
        out.extend_from_slice(&floor.first_segment.to_le_bytes());
        out.extend_from_slice(&floor.first_batch.to_le_bytes());
        out.extend_from_slice(&floor.next_batch.to_le_bytes());
    }
    let crc = crc32c_checksum(&out);
    out.extend_from_slice(&crc.to_le_bytes());
    Ok(out)
}

/// Decodes and verifies the WAL-floor record. Each entry must carry
/// nonzero positions (a zero floor is not a floor — it is a missing
/// record, and recovery must replay from genesis instead of skipping).
///
/// # Errors
///
/// Returns [`CheckpointError`] on any integrity or format violation.
pub fn decode_wal_floor(bytes: &[u8]) -> Result<Vec<LaneFloor>, CheckpointError> {
    use kivi_codec::integrity::crc32c_checksum;
    let corrupt = |detail: String| CheckpointError::Corrupt { detail };
    let format = |detail: String| CheckpointError::Format { detail };
    if bytes.len() < 16 {
        return Err(corrupt(format!(
            "WAL_FLOOR shorter than header: {} bytes",
            bytes.len()
        )));
    }
    if u32::from_le_bytes(
        bytes[0..4]
            .try_into()
            .map_err(|_| corrupt("WAL_FLOOR magic unreadable".to_owned()))?,
    ) != WAL_FLOOR_MAGIC
    {
        return Err(format("WAL_FLOOR magic mismatch".to_owned()));
    }
    let major = u16::from_le_bytes(
        bytes[4..6]
            .try_into()
            .map_err(|_| corrupt("WAL_FLOOR version unreadable".to_owned()))?,
    );
    let minor = u16::from_le_bytes(
        bytes[6..8]
            .try_into()
            .map_err(|_| corrupt("WAL_FLOOR version unreadable".to_owned()))?,
    );
    if major != WAL_FLOOR_MAJOR || minor != WAL_FLOOR_MINOR {
        return Err(CheckpointError::Unsupported {
            detail: format!("WAL_FLOOR version {major}.{minor}"),
        });
    }
    let count = u32::from_le_bytes(
        bytes[8..12]
            .try_into()
            .map_err(|_| corrupt("WAL_FLOOR count unreadable".to_owned()))?,
    ) as usize;
    if bytes.len() != 12 + count * LANE_FLOOR_ENTRY_LEN + 4 {
        return Err(corrupt(format!(
            "WAL_FLOOR length {} disagrees with {count} entries",
            bytes.len()
        )));
    }
    if crc32c_checksum(&bytes[..bytes.len() - 4])
        != u32::from_le_bytes(
            bytes[bytes.len() - 4..]
                .try_into()
                .map_err(|_| corrupt("WAL_FLOOR CRC unreadable".to_owned()))?,
        )
    {
        return Err(corrupt("WAL_FLOOR CRC mismatch".to_owned()));
    }
    let mut floors = Vec::with_capacity(count);
    let mut previous_lane: Option<u16> = None;
    for index in 0..count {
        let base = 12 + index * LANE_FLOOR_ENTRY_LEN;
        let entry = &bytes[base..base + LANE_FLOOR_ENTRY_LEN];
        let lane = u16::from_le_bytes(
            entry[0..2]
                .try_into()
                .map_err(|_| corrupt("floor lane unreadable".to_owned()))?,
        );
        if entry[2..4] != [0u8; 2] {
            return Err(format("floor entry reserved bytes nonzero".to_owned()));
        }
        if previous_lane.is_some_and(|previous| previous >= lane) {
            return Err(corrupt(
                "WAL_FLOOR lanes out of order or duplicated".to_owned(),
            ));
        }
        previous_lane = Some(lane);
        let first_segment = u64::from_le_bytes(
            entry[4..12]
                .try_into()
                .map_err(|_| corrupt("floor segment unreadable".to_owned()))?,
        );
        let first_batch = u64::from_le_bytes(
            entry[12..20]
                .try_into()
                .map_err(|_| corrupt("floor batch unreadable".to_owned()))?,
        );
        let next_batch = u64::from_le_bytes(
            entry[20..28]
                .try_into()
                .map_err(|_| corrupt("floor watermark unreadable".to_owned()))?,
        );
        if first_segment == 0 || first_batch == 0 || next_batch == 0 {
            return Err(format("WAL_FLOOR carries zero positions".to_owned()));
        }
        if next_batch < first_batch {
            return Err(corrupt("WAL_FLOOR watermark precedes its floor".to_owned()));
        }
        floors.push(LaneFloor {
            lane,
            first_segment,
            first_batch,
            next_batch,
        });
    }
    Ok(floors)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn current() -> CurrentRecord {
        CurrentRecord {
            tablet: TabletId::from_u64(1),
            cut: 42,
            manifest: ArtifactHash::of(b"manifest"),
            previous: Some(ArtifactHash::of(b"previous")),
            created_wall_micros: kivi_types::WallTimestamp::from_micros(1_000),
        }
    }

    #[test]
    fn current_round_trips() {
        let loaded =
            decode_current(TabletId::from_u64(1), &encode_current(&current())).expect("loads");
        assert_eq!(loaded, current());
        let plain = CurrentRecord {
            previous: None,
            ..current()
        };
        let loaded = decode_current(TabletId::from_u64(1), &encode_current(&plain)).expect("loads");
        assert_eq!(loaded.previous, None);
    }

    #[test]
    fn current_rejects_wrong_tablet_and_damage() {
        let bytes = encode_current(&current());
        assert!(decode_current(TabletId::from_u64(2), &bytes).is_err());
        let mut damaged = bytes.clone();
        damaged[20] ^= 0xFF;
        assert!(decode_current(TabletId::from_u64(1), &damaged).is_err());
        assert!(decode_current(TabletId::from_u64(1), &bytes[..10]).is_err());
    }

    #[test]
    fn wal_floor_round_trips_sorted() {
        let floors = vec![
            LaneFloor {
                lane: 3,
                first_segment: 9,
                first_batch: 100,
                next_batch: 150,
            },
            LaneFloor {
                lane: 1,
                first_segment: 4,
                first_batch: 40,
                next_batch: 60,
            },
        ];
        let loaded = decode_wal_floor(&encode_wal_floor(floors).expect("encodes")).expect("loads");
        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded[0].lane, 1);
        assert_eq!(loaded[1].lane, 3);
        assert!(decode_wal_floor(&[]).is_err());
    }

    #[test]
    fn wal_floor_rejects_zero_positions() {
        let encoded = encode_wal_floor(vec![LaneFloor {
            lane: 0,
            first_segment: 0,
            first_batch: 1,
            next_batch: 1,
        }])
        .expect("encodes");
        assert!(decode_wal_floor(&encoded).is_err(), "zero is not a floor");
    }
}
