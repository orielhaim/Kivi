//! Exact logical tablet and directory model for Kivi (RFC §6–§9, §190–§198).
//!
//! This crate owns the ownership truth: which tablet is the single writable
//! authority for every routable point, and how that authority moves. It is
//! deliberately free of runtime, networking, storage, consensus, clocks, and
//! hardware concerns — those systems will consume this model, never redefine
//! it.
//!
//! Core rules:
//!
//! * One active authoritative tablet owns any routable point; a snapshot that
//!   would route one point to two writable tablets is rejected, never built.
//! * Authority moves forward only: `Allocated → Inactive → Active → Fenced →`
//!   [`Tombstone`](TabletState::Tombstone), plus `Inactive → Tombstone` for
//!   aborted builds. A fenced or tombstoned authority never becomes writable
//!   again, and fencing generations never wrap or saturate.
//! * Snapshots are immutable values. Every transition takes `&self` and
//!   returns a new [`DirectorySnapshot`] with a bumped [`DirectoryVersion`];
//!   a future RCU/`ArcSwap` publication layer can publish these snapshots
//!   without changing any semantics defined here.
//!
//! Lookup is deterministic for a given snapshot: routing depends only on the
//! snapshot's contents, never on timing, caches, or execution context.

pub mod directory;
pub mod namespace;
pub mod range;
pub mod tablet;

pub use directory::{DirectoryError, DirectorySnapshot, DirectoryVersion, MultiDirectory};
pub use namespace::{NamespaceDescriptor, NamespaceLayout};
pub use range::{HashPrefix, OrderedRange, PartitionKind, PartitionRange, RangeError};
pub use tablet::{Redirect, TabletDescriptor, TabletState};
