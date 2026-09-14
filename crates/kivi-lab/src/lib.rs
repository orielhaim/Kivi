//! `kivi-lab`: Kivi integration testing, compatibility/conformance,
//! process orchestration, and benchmarking.
//!
//! This crate is explicitly **non-production**. Nothing under
//! `crates/kivi-*` may depend on it (enforced by the dependency graph:
//! it only ever points at production crates, never the reverse).
//!
//! ```text
//!                 kivi-lab
//!                /        \
//!               ▼          ▼
//!        Kivi production   redis-rs / reference Redis
//! ```
//!
//! Contents:
//!
//! * [`process`] — one reusable race-free server harness (spawn, endpoint
//!   discovery, kill/restart, admin queries).
//! * [`workload`] — target-neutral workload model (mix, key distribution,
//!   value sizes, TTL behavior). Knows nothing about chunk ids, manifests,
//!   worker routing, or connection structs.
//! * [`targets`] — [`targets::BenchTarget`] adapters (`KiviNativeTarget`,
//!   generic `RespTarget`). One RESP implementation serves Kivi RESP,
//!   Redis, Valkey, and Dragonfly alike.
//! * [`resp_client`] — minimal raw RESP2 client over blocking TCP, shared
//!   by benchmarks and conformance so comparisons isolate server
//!   differences, not client libraries.
//! * [`runner`] — concurrency, pipelining, warmup, timing, and histogram
//!   methodology, identical for every target.
//! * [`metrics`] — machine-readable benchmark schema (JSON tooling output).
//! * [`conformance`] — differential-testing helpers: key-prefix isolation,
//!   normalization, and reference-Redis configuration inspection.

pub mod cluster;
pub mod conformance;
pub mod metrics;
pub mod process;
pub mod resp_client;
pub mod runner;
pub mod targets;
pub mod workload;
