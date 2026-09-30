//! Project-owned canonical binary codec for Kivi.
//!
//! All durable and wire formats are defined here, explicitly, in little-endian
//! byte order. No long-lived canonical format is derived from `serde`,
//! `bincode`, `rkyv`, or any other generic framework: those
//! may serve admin JSON, config, tests, and debug output, but never
//! `MutationIR`, the WAL, checkpoint images, or tablet metadata.
//!
//! Layout rules:
//!
//! * Every persistent structure carries its own magic, version, and CRC32C
//!   header; the field order of each is the wire contract.
//! * Immutable artifacts additionally carry a BLAKE3-256 content hash (see
//!   [`integrity::chunk_id`]).
//! * Integrity roles: CRC32C (Castagnoli) for cheap
//!   accidental-corruption detection, BLAKE3-256 for content identity.
//!
//! Evolution: a major version bump means breaking change -
//! old decoders reject it. Unknown reserved flags are rejected (fail closed)
//! until their meaning is specified.

pub mod error;
pub mod ids;
pub mod integrity;
pub mod traits;

pub use error::CodecError;
pub use integrity::{blake3_256, chunk_id, crc32c_checksum};
pub use traits::{Decode, Encode, decode_byte_slice, decode_byte_vec, encode_bytes};

/// Maximum accepted length of any length-prefixed blob or frame payload, in
/// bytes (16 MiB).
///
/// This comfortably covers the largest benchmarked values (1 MiB)
/// plus manifest/envelope overhead, while bounding worst-case allocation from
/// a corrupt length field. It is a local admission limit, not a wire version:
/// raising it later is negotiated through the cluster capability floor
/// and never silently changes stored bytes.
pub const MAX_FRAME_BYTES: usize = 16 * 1024 * 1024;
