//! Strict versioned framing: 24-byte header plus payload.
//!
//! Layout (all integers little-endian, never Rust layout-dependent):
//!
//! ```text
//! offset  size  field
//! 0       4     magic ("KVNP")
//! 4       2     protocol major
//! 6       2     protocol minor
//! 8       1     frame kind
//! 9       1     flags (reserved, must be zero in v1)
//! 10      4     payload length in bytes
//! 14      8     request id (0 when the kind carries none)
//! 22      2     reserved (must be zero in v1)
//! total   24    FRAME_HEADER_LEN
//! ```
//!
//! [`FrameReader`] incrementally parses frames from arbitrary byte chunks:
//! split headers, coalesced frames, and truncated tails are all normal
//! inputs, never panics. Oversized declarations are rejected before their
//! payload fully arrives (the connection-level input bound caps how much
//! junk can accumulate first).

use bytes::Bytes;
use zerocopy::byteorder::{LittleEndian, U16, U32, U64};
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

/// Magic word: ASCII `"KVNP"` (Kivi native protocol) as little-endian `u32`.
/// Distinct from the storage frame magic by construction.
pub const PROTOCOL_MAGIC: u32 = 0x504E_564B;

/// Current protocol major version (breaking-change counter).
pub const PROTOCOL_MAJOR: u16 = 1;
/// Current protocol minor version (accepted only when known).
pub const PROTOCOL_MINOR: u16 = 0;

/// Encoded frame header length in bytes.
pub const FRAME_HEADER_LEN: usize = 24;

/// Default maximum: 64 MiB per non-stream frame (RFC starting point).
/// Servers and tests may configure a smaller bound; larger requires a
/// protocol version bump, never a silent constant change.
pub const DEFAULT_MAX_FRAME: usize = 64 * 1024 * 1024;

/// Frame kinds. Fixed discriminants on the wire, never Rust layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FrameKind {
    /// Client handshake opener.
    ClientHello = 1,
    /// Server handshake answer.
    ServerHello = 2,
    /// One typed operation invocation.
    Request = 3,
    /// One typed operation outcome.
    Response = 4,
    /// Keepalive probe (empty payload).
    Ping = 5,
    /// Keepalive answer (empty payload).
    Pong = 6,
}

impl FrameKind {
    /// Decodes a kind byte.
    #[must_use]
    pub const fn from_u8(value: u8) -> Option<Self> {
        match value {
            1 => Some(Self::ClientHello),
            2 => Some(Self::ServerHello),
            3 => Some(Self::Request),
            4 => Some(Self::Response),
            5 => Some(Self::Ping),
            6 => Some(Self::Pong),
            _ => None,
        }
    }

    /// Encodes the kind discriminant.
    #[must_use]
    pub const fn as_u8(self) -> u8 {
        self as u8
    }
}

impl core::fmt::Display for FrameKind {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::ClientHello => write!(f, "client-hello"),
            Self::ServerHello => write!(f, "server-hello"),
            Self::Request => write!(f, "request"),
            Self::Response => write!(f, "response"),
            Self::Ping => write!(f, "ping"),
            Self::Pong => write!(f, "pong"),
        }
    }
}

bitflags::bitflags! {
    /// Frame header flags byte. No flags are assigned in v1: every set bit is
    /// a reserved-bits violation today, while unknown bits stay observable for
    /// forward-compatible negotiation tomorrow.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct FrameFlags: u8 {
    }
}

/// The 24-byte frame header exactly as laid out on the wire.
///
/// `repr(C)` over 1-aligned endian-aware fields, so the Rust layout IS the
/// wire layout by construction (compile-time size-asserted below): field
/// order, widths, and little-endianness are all visible in the type, and
/// zerocopy moves bytes without hand-rolled slicing. Kivi still owns every
/// other rule — magic, versions, tags, reserved-bit rejection, and bounds
/// all validate in [`FrameHeader::decode`], never in the layout.
// Only `unsafe` in this crate: zerocopy's derive-generated trait impls.
// Every use goes through zerocopy's safe API (`read_from_bytes`,
// `as_bytes`); hand-written `unsafe` stays banned by the workspace lint.
#[allow(unsafe_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout)]
#[repr(C)]
struct WireHeader {
    magic: U32<LittleEndian>,
    major: U16<LittleEndian>,
    minor: U16<LittleEndian>,
    kind: u8,
    flags: u8,
    payload_len: U32<LittleEndian>,
    request_id: U64<LittleEndian>,
    reserved: U16<LittleEndian>,
}

const _: () = assert!(core::mem::size_of::<WireHeader>() == FRAME_HEADER_LEN);

/// The 24-byte frame header as validated domain values.
///
/// Parsing goes through `WireHeader` (byte movement) plus explicit rule
/// checks (magic, version, kind, reserved bits); encoding is the reverse.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FrameHeader {
    /// Magic word (`PROTOCOL_MAGIC`).
    pub magic: u32,
    /// Protocol major version.
    pub major: u16,
    /// Protocol minor version.
    pub minor: u16,
    /// Frame kind discriminant.
    pub kind: u8,
    /// Flags (no bits assigned: must be empty in v1).
    pub flags: FrameFlags,
    /// Payload length in bytes (not including this header).
    pub payload_len: u32,
    /// Request id echoed between request and response.
    pub request_id: u64,
    /// Reserved (must be zero).
    pub reserved: u16,
}

impl FrameHeader {
    /// Builds a header for an outgoing frame.
    #[must_use]
    pub fn new(kind: FrameKind, request_id: u64, payload_len: usize) -> Self {
        Self {
            magic: PROTOCOL_MAGIC,
            major: PROTOCOL_MAJOR,
            minor: PROTOCOL_MINOR,
            kind: kind.as_u8(),
            flags: FrameFlags::empty(),
            payload_len: u32::try_from(payload_len).unwrap_or(u32::MAX),
            request_id,
            reserved: 0,
        }
    }

    /// Encodes the header to its 24 canonical bytes.
    #[must_use]
    pub fn encode(&self) -> [u8; FRAME_HEADER_LEN] {
        let wire = WireHeader {
            magic: U32::new(self.magic),
            major: U16::new(self.major),
            minor: U16::new(self.minor),
            kind: self.kind,
            flags: self.flags.bits(),
            payload_len: U32::new(self.payload_len),
            request_id: U64::new(self.request_id),
            reserved: U16::new(self.reserved),
        };
        let mut out = [0u8; FRAME_HEADER_LEN];
        out.copy_from_slice(wire.as_bytes());
        out
    }

    /// Parses and structurally validates a header slice (at least 24 bytes).
    ///
    /// # Errors
    ///
    /// Returns [`ProtocolError`] on short input, bad magic, unknown
    /// versions/kinds, or set reserved bits.
    pub fn decode(input: &[u8]) -> Result<Self, ProtocolError> {
        if input.len() < FRAME_HEADER_LEN {
            return Err(ProtocolError::Truncated);
        }
        let wire = WireHeader::read_from_bytes(&input[..FRAME_HEADER_LEN]).map_err(|_| {
            ProtocolError::Malformed {
                context: "frame header",
            }
        })?;
        let header = Self {
            magic: wire.magic.get(),
            major: wire.major.get(),
            minor: wire.minor.get(),
            kind: wire.kind,
            flags: FrameFlags::from_bits_retain(wire.flags),
            payload_len: wire.payload_len.get(),
            request_id: wire.request_id.get(),
            reserved: wire.reserved.get(),
        };
        if header.magic != PROTOCOL_MAGIC {
            return Err(ProtocolError::BadMagic {
                found: header.magic,
            });
        }
        if header.major != PROTOCOL_MAJOR || header.minor != PROTOCOL_MINOR {
            return Err(ProtocolError::UnsupportedVersion {
                major: header.major,
                minor: header.minor,
            });
        }
        if FrameKind::from_u8(header.kind).is_none() {
            return Err(ProtocolError::UnknownKind { kind: header.kind });
        }
        if !header.flags.is_empty() || header.reserved != 0 {
            return Err(ProtocolError::ReservedFlags {
                flags: header.flags.bits(),
                reserved: header.reserved,
            });
        }
        Ok(header)
    }
}

/// One parsed frame: kind, request id, and payload bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Frame {
    /// Frame kind.
    pub kind: FrameKind,
    /// Echoed request id (`0` for hellos and keepalives).
    pub request_id: u64,
    /// Payload bytes (length already validated against the bound).
    pub payload: Bytes,
}

/// Framing failure: malformed input closes the connection safely, never panics.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum ProtocolError {
    /// Magic word mismatch (not a Kivi native peer, or desynchronized stream).
    #[error("bad protocol magic {found:#010X}")]
    BadMagic {
        /// Observed word.
        found: u32,
    },
    /// Major/minor version this build cannot speak.
    #[error("unsupported protocol version {major}.{minor}")]
    UnsupportedVersion {
        /// Observed major.
        major: u16,
        /// Observed minor.
        minor: u16,
    },
    /// Unknown frame kind discriminant.
    #[error("unknown frame kind {kind:#04X}")]
    UnknownKind {
        /// Observed discriminant.
        kind: u8,
    },
    /// Reserved flags or padding carried nonzero bits.
    #[error("reserved bits set (flags {flags:#04X}, reserved {reserved:#06X})")]
    ReservedFlags {
        /// Observed flags byte.
        flags: u8,
        /// Observed reserved word.
        reserved: u16,
    },
    /// Declared payload exceeds the configured maximum (rejected before the
    /// full payload arrives or allocates).
    #[error("frame of {len} bytes exceeds maximum {max}")]
    Oversized {
        /// Declared length.
        len: usize,
        /// Configured maximum.
        max: usize,
    },
    /// Structurally invalid payload for its kind/opcode.
    #[error("malformed {context}")]
    Malformed {
        /// What was being decoded (static context, never client bytes).
        context: &'static str,
    },
    /// Unknown operation opcode.
    #[error("unknown opcode {opcode:#04X}")]
    UnknownOpcode {
        /// Observed opcode.
        opcode: u8,
    },
    /// Connection closed (or buffer ended) mid-frame.
    #[error("truncated frame")]
    Truncated,
}

/// Encodes one frame (header plus payload) to freshly allocated bytes.
#[must_use]
pub fn encode_frame(kind: FrameKind, request_id: u64, payload: &[u8]) -> Vec<u8> {
    let header = FrameHeader::new(kind, request_id, payload.len());
    let mut out = Vec::with_capacity(FRAME_HEADER_LEN + payload.len());
    out.extend_from_slice(&header.encode());
    out.extend_from_slice(payload);
    out
}

/// Incremental frame parser over arbitrary byte chunks.
///
/// Codec state only — no I/O, no timers, no allocation beyond the output
/// frames and the pending tail buffer (whose growth the connection bounds).
#[derive(Debug)]
pub struct FrameReader {
    buf: Vec<u8>,
    max_frame: usize,
}

impl FrameReader {
    /// Creates a parser rejecting payloads above `max_frame` bytes.
    #[must_use]
    pub fn new(max_frame: usize) -> Self {
        Self {
            buf: Vec::new(),
            max_frame,
        }
    }

    /// Returns the configured maximum payload in bytes.
    #[must_use]
    pub const fn max_frame(&self) -> usize {
        self.max_frame
    }

    /// Returns pending (incomplete-frame) buffered bytes.
    #[must_use]
    pub fn buffered(&self) -> usize {
        self.buf.len()
    }

    /// Feeds newly read bytes, returning every frame completed by them
    /// (usually zero or one; coalesced reads yield more).
    ///
    /// # Errors
    ///
    /// Returns [`ProtocolError`] on bad magic, unknown versions/kinds, set
    /// reserved bits, or oversized declarations. The offending bytes stay
    /// buffered; callers close the connection on any error.
    pub fn push(&mut self, bytes: &[u8]) -> Result<Vec<Frame>, ProtocolError> {
        self.buf.extend_from_slice(bytes);
        let mut frames = Vec::new();
        loop {
            if self.buf.len() < FRAME_HEADER_LEN {
                return Ok(frames);
            }
            let (kind, request_id, payload_len) = self.parse_header()?;
            if self.buf.len() < FRAME_HEADER_LEN + payload_len {
                return Ok(frames);
            }
            self.buf.drain(..FRAME_HEADER_LEN);
            let payload = Bytes::copy_from_slice(&self.buf[..payload_len]);
            self.buf.drain(..payload_len);
            frames.push(Frame {
                kind,
                request_id,
                payload,
            });
        }
    }

    /// Validates the stream ended cleanly: leftover bytes mean truncation.
    ///
    /// # Errors
    ///
    /// Returns [`ProtocolError::Truncated`] when bytes remain buffered.
    pub fn end(&self) -> Result<(), ProtocolError> {
        if self.buf.is_empty() {
            Ok(())
        } else {
            Err(ProtocolError::Truncated)
        }
    }

    /// Parses and validates the header at the buffer front, returning the
    /// kind, request id, and payload length.
    fn parse_header(&self) -> Result<(FrameKind, u64, usize), ProtocolError> {
        let header = FrameHeader::decode(&self.buf[..FRAME_HEADER_LEN])?;
        let kind = FrameKind::from_u8(header.kind)
            .ok_or(ProtocolError::UnknownKind { kind: header.kind })?;
        let payload_len = usize::try_from(header.payload_len).unwrap_or(usize::MAX);
        if payload_len > self.max_frame {
            return Err(ProtocolError::Oversized {
                len: payload_len,
                max: self.max_frame,
            });
        }
        Ok((kind, header.request_id, payload_len))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use rstest::rstest;

    #[test]
    fn magic_is_ascii_kvnp_little_endian() {
        assert_eq!(u32::from_le_bytes(*b"KVNP"), PROTOCOL_MAGIC);
        assert_ne!(PROTOCOL_MAGIC, 0x4956_494B, "distinct from storage magic");
        assert_eq!(core::mem::size_of::<FrameHeader>(), FRAME_HEADER_LEN);
    }

    #[test]
    fn header_wire_bytes_are_stable() {
        // Golden bytes: independent oracle for the layout documented above.
        // Any intentional wire change must update this literal deliberately.
        let golden: [u8; FRAME_HEADER_LEN] = [
            0x4B, 0x56, 0x4E, 0x50, // "KVNP" magic
            0x01, 0x00, // major 1
            0x00, 0x00, // minor 0
            0x03, // kind: request
            0x00, // flags: none assigned
            0x0B, 0x00, 0x00, 0x00, // payload length 11
            0x08, 0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01, // request id
            0x00, 0x00, // reserved
        ];
        let header = FrameHeader::new(FrameKind::Request, 0x0102_0304_0506_0708, 11);
        assert_eq!(header.encode(), golden);
        let back = FrameHeader::decode(&golden).expect("golden parses");
        assert_eq!(back, header);
    }

    #[test]
    fn flags_round_trip_through_retained_bits() {
        // Unknown flag bits are retained (observable) but rejected: the
        // wire preserves what validation refuses.
        let mut bytes = encode_frame(FrameKind::Ping, 0, &[]);
        bytes[9] = 0x80;
        let mut reader = FrameReader::new(1024);
        assert!(matches!(
            reader.push(&bytes),
            Err(ProtocolError::ReservedFlags { flags: 0x80, .. })
        ));
    }

    #[test]
    fn header_round_trips() {
        let header = FrameHeader::new(FrameKind::Request, 0x0102_0304_0506_0708, 11);
        let bytes = header.encode();
        assert_eq!(bytes.len(), FRAME_HEADER_LEN);
        let back = FrameHeader::decode(&bytes).expect("parse");
        assert_eq!(back, header);
        assert_eq!(back.magic, PROTOCOL_MAGIC);
        assert_eq!(back.request_id, 0x0102_0304_0506_0708);
        assert_eq!(back.payload_len, 11);
    }

    #[test]
    fn split_header_reassembles() {
        let bytes = encode_frame(FrameKind::Ping, 0, &[]);
        let mut reader = FrameReader::new(1024);
        assert!(reader.push(&bytes[..10]).expect("partial").is_empty());
        assert!(reader.push(&bytes[10..20]).expect("partial").is_empty());
        let frames = reader.push(&bytes[20..]).expect("complete");
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].kind, FrameKind::Ping);
        assert_eq!(reader.end(), Ok(()));
    }

    #[test]
    fn coalesced_frames_all_parse() {
        let mut blob = encode_frame(FrameKind::Ping, 1, &[]);
        blob.extend_from_slice(&encode_frame(FrameKind::Pong, 2, b"xy"));
        blob.extend_from_slice(&encode_frame(FrameKind::Ping, 3, &[]));
        let mut reader = FrameReader::new(1024);
        let frames = reader.push(&blob).expect("parse");
        assert_eq!(frames.len(), 3);
        assert_eq!(frames[0].request_id, 1);
        assert_eq!(frames[1].payload.as_ref(), b"xy");
        assert_eq!(frames[2].request_id, 3);
    }

    #[test]
    fn oversized_declared_length_fails_before_payload_arrives() {
        let mut bytes = encode_frame(FrameKind::Request, 9, b"tiny");
        // Rewrite the length to a huge value without sending the payload.
        bytes[10..14].copy_from_slice(&(1u32 << 26).to_le_bytes());
        // The oversized declaration must fail immediately, without waiting
        // for 64 MiB that will never arrive.
        let mut reader = FrameReader::new(1024);
        assert!(matches!(
            reader.push(&bytes),
            Err(ProtocolError::Oversized { .. })
        ));
    }

    #[rstest]
    #[case(FrameKind::ClientHello)]
    #[case(FrameKind::ServerHello)]
    #[case(FrameKind::Request)]
    #[case(FrameKind::Response)]
    #[case(FrameKind::Ping)]
    #[case(FrameKind::Pong)]
    fn kinds_round_trip(#[case] kind: FrameKind) {
        assert_eq!(FrameKind::from_u8(kind.as_u8()), Some(kind));
    }

    #[test]
    fn bad_magic_bad_version_bad_flags_rejected() {
        let good = encode_frame(FrameKind::Ping, 0, &[]);
        let mut reader = FrameReader::new(1024);
        let mut bad = good.clone();
        bad[0] ^= 0xFF;
        assert!(matches!(
            reader.push(&bad),
            Err(ProtocolError::BadMagic { .. })
        ));
        let mut reader = FrameReader::new(1024);
        let mut bad = good.clone();
        bad[4] = 0xFF;
        assert!(matches!(
            reader.push(&bad),
            Err(ProtocolError::UnsupportedVersion { .. })
        ));
        let mut reader = FrameReader::new(1024);
        let mut bad = good.clone();
        bad[9] = 0x01;
        assert!(matches!(
            reader.push(&bad),
            Err(ProtocolError::ReservedFlags { .. })
        ));
        let mut reader = FrameReader::new(1024);
        let mut bad = good;
        bad[8] = 0x7F;
        assert!(matches!(
            reader.push(&bad),
            Err(ProtocolError::UnknownKind { .. })
        ));
    }

    #[test]
    fn trailing_bytes_at_close_are_truncation() {
        let mut reader = FrameReader::new(1024);
        reader.push(&[1, 2, 3]).expect("buffered");
        assert_eq!(reader.end(), Err(ProtocolError::Truncated));
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(512))]

        /// The byte-level parser never panics on arbitrary input, however
        /// adversarial the chunking.
        #[test]
        fn decoder_never_panics(bytes in prop::collection::vec(any::<u8>(), 0..300)) {
            let mut reader = FrameReader::new(256);
            for chunk in bytes.chunks(7) {
                let _ = reader.push(chunk);
            }
            let _ = reader.end();
        }

        /// Structured frames survive split-at-every-position reassembly.
        #[test]
        fn split_anywhere_reassembles(
            payload in prop::collection::vec(any::<u8>(), 0..64),
            at in 0usize..100,
        ) {
            let bytes = encode_frame(FrameKind::Request, 77, &payload);
            let at = at.min(bytes.len());
            let mut reader = FrameReader::new(4096);
            let first = reader.push(&bytes[..at]).expect("valid prefix parses");
            let rest = reader.push(&bytes[at..]).expect("valid suffix parses");
            let mut all = first;
            all.extend(rest);
            assert_eq!(all.len(), 1, "exactly one frame across the split");
            assert_eq!(all[0].request_id, 77);
            assert_eq!(all[0].payload.as_ref(), payload.as_slice());
        }
    }
}
