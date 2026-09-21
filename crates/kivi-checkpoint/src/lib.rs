//! Online immutable checkpoints: formats, capture, publication, recovery.
//!
//! This crate owns every checkpoint semantic: the physical band layout,
//! the immutable band and manifest formats, crash-safe publication, the
//! installed-checkpoint catalog, loading and validation, and WAL
//! reclamation rules. The engine drives it (capture views, background
//! builds, recovery restores) but never defines its formats.
//!
//! Design invariants (see the stage specification):
//!
//! * A checkpoint is cut at a durable logical position only — never
//!   speculative or pending state.
//! * Bands are immutable and content-addressed (BLAKE3); manifests
//!   reference band hashes, enabling incremental reuse.
//! * Publication is crash-safe (temp + sync + atomic rename + dir sync);
//!   an interrupted publication is invisible, never partial.
//! * Recovery loads `CURRENT` + bands and replays only the WAL tail after
//!   the cut; reappearing deleted segments are harmless by cut semantics.
//! * WAL reclamation only removes sealed segments fully covered by the
//!   oldest retained recovery point (current + previous checkpoints).

pub mod artifact;
pub mod band;
pub mod builder;
pub mod catalog;
pub mod dedup;
pub mod error;
pub mod layout;
pub mod load;
pub mod manifest;
pub mod publish;
pub mod reclaim;
pub mod snapshot;

pub use artifact::{ArtifactHash, ArtifactKind};
pub use band::{BandRecord, BuiltBand, CompressionCodecId, LoadedBand, build_band, load_band};
pub use builder::{
    BuildStats, BuiltBandRef, BuiltTablet, CheckpointIdentity, CheckpointPolicy, build_tablet,
};
pub use catalog::{
    CurrentRecord, decode_current, decode_wal_floor, encode_current, encode_wal_floor,
};
pub use dedup::{
    BuiltDedup, LoadedDedup, OutcomeCheckpoint, SessionCheckpoint, build_dedup, load_dedup,
};
pub use error::CheckpointError;
pub use layout::{BANDS_PER_TABLET, BandDirty, BandLayout, BandLayoutId};
pub use load::{InstalledCheckpoint, load_installed};
pub use manifest::{
    BandDescriptor, BuiltManifest, DedupRef, LoadedManifest, build_manifest, load_manifest,
};
pub use publish::{PublishStats, publish_tablet, read_current, read_wal_floor, write_wal_floor};
pub use reclaim::{LaneReclaim, plan_reclaim};
pub use snapshot::TabletSnapshot;
