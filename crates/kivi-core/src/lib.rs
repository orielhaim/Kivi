//! Deterministic core interfaces for Kivi.
//!
//! Production effects drive hardware; simulation replays the same logic
//! against virtual network, storage, randomness, scheduling, and fault
//! injection. To keep that possible, correctness logic must never call the
//! outside world directly - no `thread_rng`, no OS entropy, no sockets, no
//! files. It takes abstracted inputs (see [`clock`], [`rng`]) instead.
//!
//! This crate is the deterministic kernel vocabulary: wall [`clock`] reads at
//! production boundaries, [`rng`] sources, generic [`fault`] decisions, and
//! scheduler [`schedule`] identifiers. Domain logic (mutations, envelopes,
//! tablets) lives in the crates above and builds on these exact interfaces.

pub mod clock;
pub mod fault;
pub mod rng;
pub mod schedule;

pub use clock::{SystemClock, WallClock, WallError, wall_now_or_max};
pub use fault::{EventView, FaultDecision, FaultPolicy};
pub use rng::RandomSource;
pub use schedule::{EventId, EventIdExhausted};
