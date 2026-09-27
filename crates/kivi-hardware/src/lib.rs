//! Physical machine discovery, capability snapshot, and worker placement.
//!
//! This crate is Kivi's only window onto the machine. It owns:
//!
//! * [`topology`] — a project-owned model of logical processors, physical
//!   cores, SMT siblings, packages, memory nodes, cache instances, processor
//!   classes and I/O device locality.
//! * [`capability`] — one snapshot of every optional accelerator, each with a
//!   reason when it is absent.
//! * [`placement`] — how Kivi's existing owner threads map onto that topology.
//! * [`cpuset`] — cross-platform CPU sets and thread binding.
//! * [`pages`] — huge-page policy for long-lived arenas.
//! * [`numa`] — arenas with a memory policy applied before first touch, and
//!   the kernel's own page census to prove where the pages went.
//! * [`io`] — direct I/O and aligned buffers.
//! * [`dax`] — CXL and pmem memory regions.
//! * [`telemetry`] — hardware observation backends and the one translation
//!   layer that keeps them from being double counted.
//!
//! ## The invariant
//!
//! No accelerator in this crate can change what Kivi believes. NUMA placement,
//! SIMD width, PMU counters, huge pages, direct I/O, CXL memory, RDMA, DSA and
//! SPDK all change the cost of an operation and none of them change its
//! result. A machine with none of them runs the same exact-state semantics,
//! more slowly. Every backend here therefore reports absence as a value rather
//! than as an error, and every caller has a path that works without it.
//!
//! ## What does not live here
//!
//! Database policy. No module in this crate knows what a tablet, a write
//! barrier, a fragment or a materialization is. The crate describes the
//! machine; `kivi-memory` and `kivi-control` decide what to do about it.
//!
//! ## Unsafe
//!
//! Hand-written `unsafe` exists only where a platform has no safe interface:
//! processor binding, thread topology, huge-page advice, direct I/O and
//! device-control codes. Each site carries a safety comment naming its
//! invariant, and the workspace's `unsafe_code = "deny"` is not weakened: every
//! allowance is a scoped, grep-able `#[allow]` on one item.

#![doc = include_str!("../README.md")]

mod capability;
mod cpuset;
pub mod device;
mod placement;
pub mod topology;

pub mod accelerator;
pub mod dax;
pub mod io;
pub mod kernel;
pub mod numa;
pub mod pages;
pub mod telemetry;
pub mod transport;

pub use capability::{
    Capabilities, DamonParams, DirectIoSupport, HugePageSupport, IoUringSupport, IsaFeature,
    IsaSupport, Mechanism, MemoryRegion, PmuCounter, PmuSupport, Support, Unavailable, snapshot,
};
pub use cpuset::{BindError, Binding, CpuSet, bind_current_thread, current_cpu, parse_cpu_list};
pub use numa::{Arena, HUGE_PAGE_BYTES, NodePolicy, PagePlacement, PlacementOutcome, page_size};
pub use placement::{
    LatencySensitivity, MemoryIntensity, Placement, PlacementDiagnostics, PlacementPlan,
    PlacementRole, PlacementShortfall, PlacementStrategy, WorkerSlot, plan,
};
pub use topology::{
    CacheKind, CacheLevel, CoreKey, CpuClass, CpuId, DeviceClass, HugePagePools, IoDevice,
    LogicalCpu, MemoryNode, MemoryTier, NumaNodeId, PackageId, PhysicalCore, Topology,
    TopologySource,
};
