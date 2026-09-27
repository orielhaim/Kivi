//! Deterministic core interfaces for Kivi (RFC §176, §177, §251).
//!
//! Production effects drive hardware; simulation replays the same logic
//! against virtual time, network, storage, randomness, scheduling, and fault
//! injection. To keep that possible, correctness logic must never call the
//! outside world directly — no `Instant::now()`, no thread-local RNG, no
//! sockets, no files. It takes abstracted inputs (see [`Clock`], [`rng`])
//! instead.
//!
//! This crate is the deterministic kernel vocabulary: a virtual [`clock`],
//! [`rng`] sources, generic [`fault`] decisions, and scheduler [`schedule`]
//! identifiers. Domain logic (mutations, envelopes, tablets) lives in the
//! crates above and builds on these exact interfaces.

pub mod clock;
pub mod fault;
pub mod rng;
pub mod schedule;

pub use clock::{
    Clock, ManualClock, ManualWallClock, SystemClock, WallClock, WallError, wall_now_or_max,
};
pub use fault::{EventView, FaultDecision, FaultPolicy};
pub use rng::RandomSource;
pub use schedule::{EventId, EventIdExhausted};
