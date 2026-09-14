//! Kivi native binary protocol: wire semantics, nothing else.
//!
//! This crate owns bytes-on-the-wire: framing, handshake, requests,
//! responses, and the error taxonomy. It translates to and from the
//! project-owned domain types (`Operation`, `PartitionRange`, tablet
//! identities) but never duplicates their semantics and never touches
//! storage, tablets, or scheduling.
//!
//! Format principles (shared with the storage codec family, independently
//! versioned): explicit little-endian integers, length-prefixed blobs,
//! versioned tags, magic-gated frames, CRC-free (TCP checksums plus
//! application-level integrity later — frames fail closed on shape, not
//! content hashes). No serde, bincode, rkyv, or JSON anywhere near this
//! wire.

pub mod frame;
pub mod message;

pub use frame::{
    DEFAULT_MAX_FRAME, FRAME_HEADER_LEN, Frame, FrameFlags, FrameHeader, FrameKind, FrameReader,
    PROTOCOL_MAGIC, PROTOCOL_MAJOR, PROTOCOL_MINOR, ProtocolError, encode_frame,
};
pub use message::{
    COND_ALWAYS, COND_IF_ABSENT, COND_IF_PRESENT, Capabilities, ClientHello, EXPIRY_AT,
    EXPIRY_CLEAR, EXPIRY_KEEP, MAX_STREAM_UPLOAD_BYTES, Opcode, RedirectInfo, Request, RequestId,
    Response, ResponseBody, RouteHint, ServerHello, Status, StreamAbort, StreamBegin, StreamReady,
    ValueStreamBegin, operation_opcode,
};
