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
    BATCH_BOUNDED_ADD, BATCH_COMMUTATIVE_ADD, BATCH_COUNTER_ADD, BATCH_DELETE, BATCH_ESCROW_SHARE,
    BATCH_EXPECT_ABSENT, BATCH_EXPECT_ANY, BATCH_EXPECT_VERSION, BATCH_PUT, BATCH_PUT_CHUNKED,
    BatchWrite, COND_ALWAYS, COND_IF_ABSENT, COND_IF_PRESENT, Capabilities, ClientHello, EXPIRY_AT,
    EXPIRY_CLEAR, EXPIRY_KEEP, MAX_STREAM_UPLOAD_BYTES, Opcode, RedirectInfo, Request, RequestId,
    Response, ResponseBody, RouteHint, SCAN_ANY, SCAN_FORWARD, SCAN_KEYS_AND_VALUES,
    SCAN_KEYS_ONLY, SCAN_LATEST_PER_TABLET, SCAN_REVERSE, SCAN_VALUE_CHUNKED, SCAN_VALUE_COUNTER,
    SCAN_VALUE_INLINE, SCAN_VALUE_NONE, SCAN_VALUE_OVERSIZE, SCAN_VALUE_SEMANTIC, ScanEntryBody,
    ScanValueBody, ServerHello, Status, StreamAbort, StreamBegin, StreamEntryBody, StreamReady,
    ValueStreamBegin, mutation_opcode, operation_opcode,
};
