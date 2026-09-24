//! Immutable chunk fabric: content-addressed chunks in append-only packs.
//!
//! Large byte values become small authoritative roots plus immutable bulk
//! information:
//!
//! ```text
//! Logical Bytes Object
//!         │
//!         ▼
//!     ObjectRoot
//!         │
//!         ├── Inline (≤ threshold, resident)
//!         │
//!         └── Chunked { manifest, logical_len }
//!               │
//!               ▼
//!         ChunkManifest (canonical, content-addressed)
//!          │   │   │
//!          ▼   ▼   ▼
//!        Chunk Chunk Chunk  (1 MiB logical pieces, content-addressed)
//! ```
//!
//! ## Ordering rule (the invariant everything else serves)
//!
//! ```text
//! stage chunks + manifest
//!        ↓
//! sync the pack tail (one barrier for the whole batch)
//!        ↓
//! prove durability (DurableChunk / DurableManifest tokens)
//!        ↓
//! commit the small root mutation through the normal WAL/group commit
//!        ↓
//! apply root, reply
//! ```
//!
//! A crash after pack durability leaves harmless orphans; a crash after
//! root commit always finds its chunks. Orphans are reclaimed by mark GC,
//! never by reference counting.
//!
//! ## Crate boundaries
//!
//! This crate owns chunking, manifests, pack bytes, the derived index,
//! staging, verification, and GC planning. It never touches tablets,
//! routing, `MutationIR` semantics, or the network: the engine resolves
//! physical bytes and commits logical roots.

pub mod codec;
pub mod error;
pub mod gc;
pub mod manifest;
pub mod pack;
pub mod policy;
pub mod redundancy;
pub mod store;

pub use codec::ChunkCodecId;
pub use error::ChunkError;
pub use gc::{GcInputs, GcReport, plan as plan_gc};
pub use manifest::{ChunkEntry, ChunkManifest, build_manifest, splice_entries, verify_manifest};
pub use pack::{RecordId, ScannedRecord, pack_file_name};
pub use policy::{Chunking, DEFAULT_CHUNK_SIZE, DEFAULT_INLINE_THRESHOLD, MAX_CHUNKS_PER_MANIFEST};
pub use redundancy::{
    chunk_asset, manifest_asset, protect_manifest_staged, protect_staged, read_manifest_via_fabric,
    read_via_fabric,
};
pub use store::{
    ChunkRecovery, ChunkStats, ChunkStore, DurableChunk, DurableManifest, StageOutcome,
};
