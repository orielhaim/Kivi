//! Durable frame header (RFC §53).
//!
//! Every persistent structure is wrapped as:
//!
//! ```text
//! offset  size  field
//! 0       4     magic (`KIVI_MAGIC`, ASCII "KIVI" little-endian)
//! 4       2     major version (breaking-change counter)
//! 6       2     minor version (accepted only when known)
//! 8       4     reserved flags (must be zero in v1.0)
//! 12      4     payload length in bytes (<= `MAX_FRAME_BYTES`)
//! 16      4     CRC32C of the payload bytes
//! 20      4     CRC32C of header bytes [0..20]
//! total   24    `HEADER_LEN`
//! ```
//!
//! Validation order is deliberate: length is range-checked before any
//! slicing (no corrupt-length allocation), the magic/version/flags gate
//! understanding, then the header CRC, then the payload CRC.

use zerocopy::byteorder::{LittleEndian, U16, U32};
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

use crate::MAX_FRAME_BYTES;
use crate::error::CodecError;
use crate::integrity::crc32c_checksum;

/// Magic word: ASCII `"KIVI"` read as a little-endian `u32`.
pub const KIVI_MAGIC: u32 = 0x4956_494B;

/// Current major wire version.
pub const CURRENT_MAJOR: u16 = 1;
/// Current minor wire version.
pub const CURRENT_MINOR: u16 = 0;

/// Encoded header length in bytes.
pub const HEADER_LEN: usize = 24;

/// Bytes covered by the header CRC (everything but the trailing checksum).
const CRC_PREFIX_LEN: usize = 20;

/// The 24-byte durable header exactly as laid out on the wire.
///
/// `repr(C)` over 1-aligned endian-aware fields, so the Rust layout IS the
/// wire layout by construction (compile-time size-asserted below); the CRC
/// rules, version gates, and bounds all stay project-owned in
/// [`FrameHeader::seal`] and [`FrameHeader::open`].
// Only `unsafe` in this crate: zerocopy's derive-generated trait impls.
// Every use goes through zerocopy's safe API; hand-written `unsafe` stays
// banned by the workspace lint.
#[allow(unsafe_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout)]
#[repr(C)]
struct WireHeader {
    magic: U32<LittleEndian>,
    major: U16<LittleEndian>,
    minor: U16<LittleEndian>,
    flags: U32<LittleEndian>,
    payload_len: U32<LittleEndian>,
    payload_crc: U32<LittleEndian>,
    header_crc: U32<LittleEndian>,
}

const _: () = assert!(core::mem::size_of::<WireHeader>() == HEADER_LEN);
const _: () = assert!(core::mem::size_of::<WireHeader>() - 4 == CRC_PREFIX_LEN);

/// Validated durable frame header (payload itself travels alongside).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FrameHeader {
    major: u16,
    minor: u16,
    flags: u32,
    payload_len: u32,
    payload_crc: u32,
}

impl FrameHeader {
    /// Returns the major version.
    #[must_use]
    pub const fn major(self) -> u16 {
        self.major
    }

    /// Returns the minor version.
    #[must_use]
    pub const fn minor(self) -> u16 {
        self.minor
    }

    /// Returns the reserved flags word.
    #[must_use]
    pub const fn flags(self) -> u32 {
        self.flags
    }

    /// Returns the payload length in bytes.
    #[must_use]
    pub const fn payload_len(self) -> u32 {
        self.payload_len
    }

    /// Returns the CRC32C stored for the payload.
    #[must_use]
    pub const fn payload_crc(self) -> u32 {
        self.payload_crc
    }

    /// Seals `payload` into a complete frame: header plus payload bytes.
    ///
    /// # Errors
    ///
    /// Returns [`CodecError::TooLarge`] if the payload exceeds
    /// [`MAX_FRAME_BYTES`].
    pub fn seal(payload: &[u8]) -> Result<Vec<u8>, CodecError> {
        if payload.len() > MAX_FRAME_BYTES {
            return Err(CodecError::TooLarge {
                len: payload.len(),
                max: MAX_FRAME_BYTES,
            });
        }
        let payload_len = u32::try_from(payload.len()).map_err(|_| CodecError::TooLarge {
            len: payload.len(),
            max: MAX_FRAME_BYTES,
        })?;
        let mut wire = WireHeader {
            magic: U32::new(KIVI_MAGIC),
            major: U16::new(CURRENT_MAJOR),
            minor: U16::new(CURRENT_MINOR),
            flags: U32::new(0),
            payload_len: U32::new(payload_len),
            payload_crc: U32::new(crc32c_checksum(payload)),
            header_crc: U32::new(0),
        };
        let header_crc = crc32c_checksum(&wire.as_bytes()[..CRC_PREFIX_LEN]);
        wire.header_crc = U32::new(header_crc);
        let mut out = Vec::with_capacity(HEADER_LEN + payload.len());
        out.extend_from_slice(wire.as_bytes());
        out.extend_from_slice(payload);
        Ok(out)
    }

    /// Opens one frame from the prefix of `input`, validating everything.
    ///
    /// Returns the header, the borrowed payload slice, and total bytes
    /// consumed (so frames can be pipelined back to back).
    ///
    /// # Errors
    ///
    /// Returns [`CodecError`] on truncation, bad magic, unknown version,
    /// set reserved flags, oversize length, or either checksum mismatch.
    /// Never panics on untrusted input.
    pub fn open(input: &[u8]) -> Result<(Self, &[u8], usize), CodecError> {
        if input.len() < HEADER_LEN {
            return Err(CodecError::Truncated {
                expected: HEADER_LEN,
                available: input.len(),
            });
        }
        let wire = WireHeader::read_from_bytes(&input[..HEADER_LEN]).map_err(|_| {
            CodecError::Malformed {
                context: "durable frame header",
            }
        })?;
        if wire.magic.get() != KIVI_MAGIC {
            return Err(CodecError::InvalidMagic {
                found: wire.magic.get(),
            });
        }
        let (major, minor) = (wire.major.get(), wire.minor.get());
        if major != CURRENT_MAJOR || minor != CURRENT_MINOR {
            return Err(CodecError::UnsupportedVersion { major, minor });
        }
        let flags = wire.flags.get();
        if flags != 0 {
            return Err(CodecError::NonZeroReservedFlags { flags });
        }
        let payload_len = wire.payload_len.get();
        let payload_len_usize = usize::try_from(payload_len).unwrap_or(MAX_FRAME_BYTES);
        if payload_len_usize > MAX_FRAME_BYTES {
            return Err(CodecError::TooLarge {
                len: payload_len_usize,
                max: MAX_FRAME_BYTES,
            });
        }
        let stored_payload_crc = wire.payload_crc.get();
        let stored_header_crc = wire.header_crc.get();
        let computed_header_crc = crc32c_checksum(&input[..CRC_PREFIX_LEN]);
        if computed_header_crc != stored_header_crc {
            return Err(CodecError::HeaderChecksumMismatch {
                expected: stored_header_crc,
                computed: computed_header_crc,
            });
        }
        let total = HEADER_LEN
            .checked_add(payload_len_usize)
            .ok_or(CodecError::TooLarge {
                len: payload_len_usize,
                max: MAX_FRAME_BYTES,
            })?;
        if input.len() < total {
            return Err(CodecError::Truncated {
                expected: total,
                available: input.len(),
            });
        }
        let payload = &input[HEADER_LEN..total];
        let computed_payload_crc = crc32c_checksum(payload);
        if computed_payload_crc != stored_payload_crc {
            return Err(CodecError::PayloadChecksumMismatch {
                expected: stored_payload_crc,
                computed: computed_payload_crc,
            });
        }
        let header = Self {
            major,
            minor,
            flags,
            payload_len,
            payload_crc: stored_payload_crc,
        };
        Ok((header, payload, total))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn magic_is_ascii_kivi_little_endian() {
        assert_eq!(u32::from_le_bytes(*b"KIVI"), KIVI_MAGIC);
    }

    #[test]
    fn header_wire_bytes_are_stable() {
        // Golden layout: magic, versions, zero flags, length, payload CRC,
        // header CRC over the first 20 bytes. Any intentional wire change
        // must update this reconstruction deliberately.
        let payload = b"golden";
        let frame = FrameHeader::seal(payload).expect("seal");
        assert_eq!(&frame[0..4], b"KIVI");
        assert_eq!(&frame[4..6], &1u16.to_le_bytes());
        assert_eq!(&frame[6..8], &0u16.to_le_bytes());
        assert_eq!(&frame[8..12], &0u32.to_le_bytes());
        assert_eq!(
            &frame[12..16],
            &u32::try_from(payload.len())
                .expect("test payload fits")
                .to_le_bytes()
        );
        assert_eq!(&frame[16..20], &crc32c_checksum(payload).to_le_bytes());
        assert_eq!(&frame[20..24], &crc32c_checksum(&frame[..20]).to_le_bytes());
        assert_eq!(&frame[24..], payload);
        let (header, body, consumed) = FrameHeader::open(&frame).expect("open");
        assert_eq!(header.payload_len() as usize, payload.len());
        assert_eq!(body, payload);
        assert_eq!(consumed, frame.len());
    }

    #[test]
    fn seal_then_open_round_trip() {
        let payload = b"hello tablet";
        let frame = FrameHeader::seal(payload).expect("seal");
        assert_eq!(frame.len(), HEADER_LEN + payload.len());
        let (header, body, consumed) = FrameHeader::open(&frame).expect("open");
        assert_eq!(
            header.payload_len(),
            u32::try_from(payload.len()).expect("test payload fits")
        );
        assert_eq!(header.major(), CURRENT_MAJOR);
        assert_eq!(header.flags(), 0);
        assert_eq!(body, payload);
        assert_eq!(consumed, frame.len());
    }

    #[test]
    fn empty_payload_frames_work() {
        let frame = FrameHeader::seal(&[]).expect("seal empty");
        let (_, body, consumed) = FrameHeader::open(&frame).expect("open empty");
        assert!(body.is_empty());
        assert_eq!(consumed, HEADER_LEN);
    }

    #[test]
    fn single_bit_corruption_is_detected() {
        let frame = FrameHeader::seal(b"twelve bytes").expect("seal");
        for index in [0, 7, 21, 25, frame.len() - 1] {
            let mut corrupt = frame.clone();
            corrupt[index] ^= 0x01;
            assert!(
                FrameHeader::open(&corrupt).is_err(),
                "corruption at byte {index} went undetected"
            );
        }
    }

    #[test]
    fn truncated_wrong_magic_and_oversize_are_rejected() {
        assert!(matches!(
            FrameHeader::open(&[0u8; 10]),
            Err(CodecError::Truncated { .. })
        ));
        let mut bad_magic = FrameHeader::seal(b"x").expect("seal");
        bad_magic[0] = 0x00;
        assert!(matches!(
            FrameHeader::open(&bad_magic),
            Err(CodecError::HeaderChecksumMismatch { .. } | CodecError::InvalidMagic { .. })
        ));
        assert!(matches!(
            FrameHeader::seal(&vec![0u8; MAX_FRAME_BYTES + 1]),
            Err(CodecError::TooLarge { .. })
        ));
    }

    #[test]
    fn reserved_flags_are_rejected() {
        let mut frame = FrameHeader::seal(b"flagged").expect("seal");
        frame[8] = 0x01; // set lowest flag bit, then repair the header CRC
        let repaired = crc32c_checksum(&frame[..20]);
        frame[20..24].copy_from_slice(&repaired.to_le_bytes());
        assert!(matches!(
            FrameHeader::open(&frame),
            Err(CodecError::NonZeroReservedFlags { flags: 1 })
        ));
    }

    #[test]
    fn pipelined_frames_open_back_to_back() {
        let first = FrameHeader::seal(b"one").expect("seal 1");
        let second = FrameHeader::seal(b"two!").expect("seal 2");
        let combined = [first.clone(), second.clone()].concat();
        let (_, body1, used1) = FrameHeader::open(&combined).expect("open 1");
        assert_eq!(body1, b"one");
        let (_, body2, used2) = FrameHeader::open(&combined[used1..]).expect("open 2");
        assert_eq!(body2, b"two!");
        assert_eq!(used1 + used2, combined.len());
    }
}
