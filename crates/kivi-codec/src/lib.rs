//! Project-owned canonical binary codec for Kivi.
//!
//! All durable and wire formats are defined here, explicitly, in little-endian
//! byte order. No long-lived canonical format is derived from `serde`,
//! `bincode`, `rkyv`, or any other generic framework (RFC §53, §224): those
//! may serve admin JSON, config, tests, and debug output, but never
//! `MutationIR`, the WAL, checkpoint images, or tablet metadata.
//!
//! Layout rules (RFC §53):
//!
//! * Every persistent structure starts with a [`header::FrameHeader`]
//!   carrying magic, major/minor version, feature flags, length, and CRCs.
//! * Canonical immutable artifacts additionally carry a BLAKE3-256 content
//!   hash (see [`integrity::chunk_id`]).
//! * Integrity roles (RFC §54): CRC32C (Castagnoli) for cheap
//!   accidental-corruption detection, BLAKE3-256 for content identity.
//!   Algorithm IDs are stored on the wire ([`algorithm`]); no algorithm is
//!   assumed permanent.
//!
//! Evolution (RFC §226, §228): a major version bump means breaking change —
//! old decoders reject it. Old and new representations may coexist; migrations
//! are lazy and incremental, never stop-the-world. Unknown reserved flags are
//! rejected (fail closed) until their meaning is specified.

pub mod algorithm;
pub mod error;
pub mod header;
pub mod ids;
pub mod integrity;
pub mod traits;

pub use algorithm::{ChecksumAlgorithmId, HashAlgorithmId};
pub use error::CodecError;
pub use header::{CURRENT_MAJOR, CURRENT_MINOR, FrameHeader, HEADER_LEN, KIVI_MAGIC};
pub use integrity::{blake3_256, chunk_id, crc32c_checksum};
pub use traits::{Decode, Encode, decode_byte_slice, decode_byte_vec, encode_bytes};

/// Maximum accepted length of any length-prefixed blob or frame payload, in
/// bytes (16 MiB).
///
/// This comfortably covers the largest benchmarked values (1 MiB, RFC §237)
/// plus manifest/envelope overhead, while bounding worst-case allocation from
/// a corrupt length field. It is a local admission limit, not a wire version:
/// raising it later is negotiated through the cluster capability floor
/// (RFC §226) and never silently changes stored bytes.
pub const MAX_FRAME_BYTES: usize = 16 * 1024 * 1024;
