//! Production single-node Kivi engine: live tablets, worker threads, local
//! routing, and the embedded client API.
//!
//! Ownership model (no locks on the data path): each worker owns its OS
//! thread, its [`LiveTablet`]s, and their object stores. Requests travel
//! over bounded channels; routing snapshots publish through RCU-style atomic
//! swaps, never a global mutex.
//!
//! The embedded [`LocalEngine`]/[`LocalClient`] path is the local API —
//! future network connections will live directly on workers and call tablet
//! execution without this cross-thread hop. The two paths share tablet logic
//! but are not optimized as one.

pub mod affinity;
pub mod clock;
pub mod engine;
pub mod net;
pub mod routing;
pub mod tablet;
pub mod worker;

pub use affinity::{AffinityError, AffinityMode, available_cores, pin_current_thread};
pub use clock::SystemClock;
pub use engine::{
    AdminHandle, DurabilityMode, DurableConfig, EngineConfig, EngineDurability, EngineError,
    LocalClient, LocalEngine, ShutdownReport,
};
pub use net::{ConnLimits, EngineNetwork, NetConfig, NetStartError, TurnBudget};
pub use routing::{Placement, RoutingSnapshot};
pub use tablet::{
    DedupEntry, DurablePrepared, LiveTablet, SessionDedup, TabletError, TabletMetrics,
};
pub use worker::{LaneAccess, WorkerControl, WorkerDurability, WorkerMetrics};
