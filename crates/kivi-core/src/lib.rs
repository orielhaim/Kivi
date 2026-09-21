//! Deterministic core interfaces for Kivi (RFC §176, §177, §251).
//!
//! Correctness logic follows one shape:
//!
//! ```text
//! Event
//!  ↓
//! State Machine
//!  ↓
//! Effects
//! ```
//!
//! Production effects drive hardware; simulation replays the same state
//! machines against virtual time, network, storage, randomness, scheduling,
//! and failure injection. To keep that possible, correctness logic must never
//! call the outside world directly — no `Instant::now()`, no thread-local
//! RNG, no sockets, no files. It receives one event plus abstracted inputs
//! (see [`Clock`]), mutates owned state, and emits [`Effects`] describing
//! everything else. The machine contract itself is [`StateMachine`].
//!
//! This crate is the deterministic kernel vocabulary: the [`machine`]
//! event/effect model, a virtual [`clock`], [`rng`] sources, generic
//! [`fault`] decisions, and scheduler [`schedule`] identifiers. Domain logic
//! (mutations, envelopes, tablets) lives in the crates above and builds on
//! these exact interfaces.

pub mod clock;
pub mod fault;
pub mod machine;
pub mod rng;
pub mod schedule;

pub use clock::{
    Clock, ManualClock, ManualWallClock, SystemClock, WallClock, WallError, wall_now_or_max,
};
pub use fault::{EventView, FaultDecision, FaultPolicy};
pub use machine::{EffectOverflow, EffectSink, Effects, StateMachine};
pub use rng::RandomSource;
pub use schedule::{EventId, EventIdExhausted};
