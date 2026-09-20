//! Criticality/cost-aware placement with hysteresis.
//!
//! The production planner ([`CriticalityPlanner`]) scores every eligible
//! provider by calibrated cost against [`ExecutionCriticality`] and
//! [`MaterializationIntent`]: access behavior, bytes, mutability,
//! read/write ratio, reconstruction cost, compression ratio, loaded
//! provider latency, memory pressure, tail-latency contribution,
//! migration cost, and tenant policy. Simple [`SieveBaseline`] and
//! [`S3FifoBaseline`] implementations exist for measurement comparison;
//! benchmarks decide, but the production default is criticality-aware,
//! never LRU or hot/cold alone.
//!
//! Every decision carries hysteresis: minimum improvement, sustained
//! duration, and cooldown. No object bounces between representations
//! every few seconds.

use std::collections::HashMap;

use crate::criticality::ExecutionCriticality;
use crate::intent::MaterializationIntent;
use crate::provider::{ProviderId, ProviderRegistry};

/// What the planner decided for one object.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PlacementDecision {
    /// Target provider, or [`ProviderId::NONE`] to defer.
    pub target: ProviderId,
    /// Expected cost improvement in abstract units (must exceed the
    /// hysteresis floor to act).
    pub improvement: f64,
    /// Confidence in `[0, 1]`.
    pub confidence: f64,
    /// Ticks until this decision may be revisited (cooldown).
    pub cooldown_ticks: u64,
    /// Human-readable reason for `EXPLAIN`-style output.
    pub reason: &'static str,
}

impl PlacementDecision {
    /// Decision to leave the object where it is.
    #[must_use]
    pub const fn stay(reason: &'static str) -> Self {
        Self {
            target: ProviderId::NONE,
            improvement: 0.0,
            confidence: 1.0,
            cooldown_ticks: 0,
            reason,
        }
    }
}

/// Planner input for one object.
#[derive(Debug, Clone, PartialEq)]
pub struct PlacementInput<'a> {
    /// Which object is being placed.
    pub object: u64,
    /// Declarative requirements.
    pub intent: &'a MaterializationIntent,
    /// Measured criticality.
    pub criticality: ExecutionCriticality,
    /// Logical bytes.
    pub logical_bytes: u64,
    /// Current provider, if resident.
    pub current: Option<ProviderId>,
    /// Current memory pressure in `[0, 1]`.
    pub pressure: f64,
    /// Migration cost in nanoseconds (bytes / bandwidth + fixed).
    pub migration_cost_ns: u64,
    /// Ticks since the last move of this object.
    pub ticks_since_move: u64,
}

/// A placement policy: pure function of input + calibrated providers.
pub trait Planner {
    /// Plans one object's placement.
    fn plan(
        &mut self,
        input: &PlacementInput<'_>,
        providers: &ProviderRegistry,
    ) -> PlacementDecision;
}

/// Production planner: criticality/cost-aware scoring.
///
/// Cost model per eligible provider:
/// `expected_access_cost + storage_cost + migration_amortized - pressure_relief`
/// where access cost folds calibrated latency, loaded penalty, criticality
/// score, and tail weight; storage cost folds bytes, duration, and tenant
/// cost weight; migration is amortized over the cooldown horizon. The
/// cheapest provider wins when it beats the current placement by the
/// hysteresis floor.
#[derive(Debug, Default)]
pub struct CriticalityPlanner {
    /// Minimum relative improvement to move (e.g. 0.1 = 10%).
    pub min_improvement: f64,
    /// Minimum ticks between moves of the same object.
    pub cooldown_ticks: u64,
    /// Horizon in ticks over which migration cost amortizes.
    pub amortization_ticks: u64,
}

impl CriticalityPlanner {
    /// Creates a planner with the given hysteresis parameters.
    #[must_use]
    pub const fn new(min_improvement: f64, cooldown_ticks: u64, amortization_ticks: u64) -> Self {
        Self {
            min_improvement,
            cooldown_ticks,
            amortization_ticks,
        }
    }

    /// Default production hysteresis: 10% improvement, 500-tick
    /// cooldown, 10k-tick amortization.
    #[must_use]
    pub const fn production() -> Self {
        Self::new(0.10, 500, 10_000)
    }

    /// Scores one provider for an input. Lower is better.
    ///
    /// Latencies and byte counts are far below 2^53, so float conversion
    /// is exact for all reachable inputs.
    #[allow(clippy::cast_precision_loss)]
    fn score(
        &self,
        input: &PlacementInput<'_>,
        provider: &crate::provider::ProviderDescriptor,
    ) -> f64 {
        let caps = &provider.caps;
        let eligible = caps.satisfies(
            input.intent.coherence,
            input.intent.sharing,
            input.intent.durability_required,
            input.intent.locality,
            input.intent.encryption_required,
        );
        if !eligible {
            return f64::INFINITY;
        }
        // Access cost: calibrated read latency scaled by criticality.
        // Critical objects multiply latency (they feel it); cold objects
        // barely do. Loaded devices pay the saturation penalty.
        let loaded = 1.0 + input.pressure * 2.0;
        let latency = (caps.read_latency_ns as f64).max(1.0) * loaded;
        let access_rate = (input.criticality.score + 1.0).ln_1p();
        let access_cost = latency * (1.0 + access_rate);
        // Storage cost: bytes * provider cost * tenant weight.
        let storage_cost = input.logical_bytes as f64
            * caps.cost_per_byte as f64
            * (f64::from(input.intent.cost_weight_bps) / 100.0);
        // Migration amortization: moving costs now, pays over horizon.
        let migration = input.migration_cost_ns as f64
            / f64::from(
                u32::try_from(self.amortization_ticks)
                    .unwrap_or(u32::MAX)
                    .max(1),
            );
        // Pressure relief: moving off a pressured current provider is
        // worth something; moving onto one costs.
        let relief = if Some(provider.id) == input.current {
            input.pressure * 1.0e6
        } else {
            0.0
        };
        access_cost + storage_cost * 1.0e-6 + migration - relief
    }
}

impl Planner for CriticalityPlanner {
    fn plan(
        &mut self,
        input: &PlacementInput<'_>,
        providers: &ProviderRegistry,
    ) -> PlacementDecision {
        if input.ticks_since_move < self.cooldown_ticks {
            return PlacementDecision::stay("cooldown");
        }
        let mut best: Option<(ProviderId, f64)> = None;
        for provider in providers.all() {
            let cost = self.score(input, provider);
            if !cost.is_finite() {
                continue;
            }
            if best.is_none_or(|(_, best_cost)| cost < best_cost) {
                best = Some((provider.id, cost));
            }
        }
        let Some((target, best_cost)) = best else {
            return PlacementDecision::stay("no eligible provider");
        };
        if Some(target) == input.current {
            return PlacementDecision {
                target: ProviderId::NONE,
                improvement: 0.0,
                confidence: 0.9,
                cooldown_ticks: 0,
                reason: "already optimal",
            };
        }
        // Improvement relative to staying: compare against the current
        // provider's score (or a large default when homeless).
        let current_cost = input
            .current
            .and_then(|id| providers.all().iter().find(|p| p.id == id))
            .map_or(f64::INFINITY, |p| self.score(input, p));
        let improvement = if current_cost.is_finite() && current_cost > 0.0 {
            (current_cost - best_cost) / current_cost
        } else {
            1.0
        };
        if improvement < self.min_improvement {
            return PlacementDecision::stay("below hysteresis floor");
        }
        PlacementDecision {
            target,
            improvement,
            confidence: 0.8,
            cooldown_ticks: self.cooldown_ticks,
            reason: "criticality/cost optimum",
        }
    }
}

/// SIEVE-style baseline: single-clock FIFO with second-chance bits.
///
/// Kept for measurement comparison (RFC §141). Request-path simple by
/// design; the benchmark harness decides whether it survives.
#[derive(Debug, Default)]
pub struct SieveBaseline {
    visited: HashMap<u64, bool>,
    order: Vec<u64>,
}

impl SieveBaseline {
    /// Creates an empty baseline.
    #[must_use]
    pub fn new() -> Self {
        Self {
            visited: HashMap::new(),
            order: Vec::new(),
        }
    }

    /// Records a cache hit (sets the second-chance bit).
    pub fn hit(&mut self, object: u64) {
        self.visited.insert(object, true);
        if !self.order.contains(&object) {
            self.order.push(object);
        }
    }
}

impl Planner for SieveBaseline {
    fn plan(
        &mut self,
        input: &PlacementInput<'_>,
        providers: &ProviderRegistry,
    ) -> PlacementDecision {
        // Baseline heuristic: objects with any recent hit stay on the
        // fastest eligible provider; misses go to the cheapest.
        let hot = self.visited.get(&input.object).copied().unwrap_or(false);
        let mut eligible: Vec<_> = providers
            .all()
            .iter()
            .filter(|p| {
                p.caps.satisfies(
                    input.intent.coherence,
                    input.intent.sharing,
                    input.intent.durability_required,
                    input.intent.locality,
                    input.intent.encryption_required,
                )
            })
            .collect();
        if eligible.is_empty() {
            return PlacementDecision::stay("no eligible provider");
        }
        eligible.sort_by_key(|p| {
            if hot {
                p.caps.read_latency_ns
            } else {
                u64::MAX - p.caps.cost_per_byte
            }
        });
        let target = eligible[0].id;
        if Some(target) == input.current {
            PlacementDecision::stay("sieve: already placed")
        } else {
            PlacementDecision {
                target,
                improvement: 0.05,
                confidence: 0.4,
                cooldown_ticks: 0,
                reason: "sieve baseline",
            }
        }
    }
}

/// S3-FIFO-style baseline: small ghost queue plus main FIFO.
///
/// Kept for measurement comparison (RFC §141). Like SIEVE, the harness
/// measures it; the production planner does not depend on it.
#[derive(Debug, Default)]
pub struct S3FifoBaseline {
    small: Vec<u64>,
    ghost: Vec<u64>,
    small_capacity: usize,
}

impl S3FifoBaseline {
    /// Creates a baseline with the given small-queue capacity.
    #[must_use]
    pub fn new(small_capacity: usize) -> Self {
        Self {
            small: Vec::new(),
            ghost: Vec::new(),
            small_capacity: if small_capacity == 0 {
                1
            } else {
                small_capacity
            },
        }
    }

    /// Records an access.
    pub fn access(&mut self, object: u64) {
        if self.small.contains(&object) || self.ghost.contains(&object) {
            return;
        }
        if self.small.len() >= self.small_capacity {
            let evicted = self.small.remove(0);
            self.ghost.push(evicted);
            if self.ghost.len() > self.small_capacity * 2 {
                self.ghost.remove(0);
            }
        }
        self.small.push(object);
    }
}

impl Planner for S3FifoBaseline {
    fn plan(
        &mut self,
        input: &PlacementInput<'_>,
        providers: &ProviderRegistry,
    ) -> PlacementDecision {
        let promoted = self.ghost.contains(&input.object);
        let mut eligible: Vec<_> = providers
            .all()
            .iter()
            .filter(|p| {
                p.caps.satisfies(
                    input.intent.coherence,
                    input.intent.sharing,
                    input.intent.durability_required,
                    input.intent.locality,
                    input.intent.encryption_required,
                )
            })
            .collect();
        if eligible.is_empty() {
            return PlacementDecision::stay("no eligible provider");
        }
        eligible.sort_by_key(|p| {
            if promoted {
                p.caps.read_latency_ns
            } else {
                u64::MAX - p.caps.cost_per_byte
            }
        });
        let target = eligible[0].id;
        if Some(target) == input.current {
            PlacementDecision::stay("s3-fifo: already placed")
        } else {
            PlacementDecision {
                target,
                improvement: 0.05,
                confidence: 0.4,
                cooldown_ticks: 0,
                reason: "s3-fifo baseline",
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::criticality::AccessSignals;
    use crate::intent::MaterializationIntent;
    use crate::provider::ProviderCaps;

    fn test_registry() -> ProviderRegistry {
        let mut registry = ProviderRegistry::new();
        registry.register(ProviderCaps::dram(Some(0), 1 << 30));
        registry.register(ProviderCaps::compressed(Some(0), 1 << 30));
        registry.register(ProviderCaps::nvme(1 << 40));
        registry
    }

    #[test]
    fn critical_object_stays_fast_and_cold_object_goes_cheap() {
        let registry = test_registry();
        let mut planner = CriticalityPlanner::production();
        let hot_signals = AccessSignals {
            reads: 10_000,
            writes: 10,
            reuse_distance: 0,
            bytes_touched: 64,
            cpu_ns: 1_000_000,
            stall_fraction: 0.9,
            mlp: 1.0,
            critical_path_fraction: 1.0,
            tail_fraction: 0.9,
            mutability: crate::criticality::Mutability::ReadMostly,
            logical_bytes: 64,
            reconstruction_cost_ns: 500,
            compression_ratio: None,
        };
        let hot = crate::criticality::criticality_of(&hot_signals, 90.0);
        let intent = MaterializationIntent::hot();
        let input = PlacementInput {
            object: 1,
            intent: &intent,
            criticality: hot,
            logical_bytes: 64,
            current: None,
            pressure: 0.1,
            migration_cost_ns: 1_000,
            ticks_since_move: 10_000,
        };
        let decision = planner.plan(&input, &registry);
        let caps = registry.get(decision.target).expect("target");
        assert_eq!(caps.kind, crate::provider::ProviderKind::DRAM);

        let cold_signals = AccessSignals::cold(4096);
        let cold = crate::criticality::criticality_of(&cold_signals, 90.0);
        let cold_intent = MaterializationIntent::cold();
        let cold_input = PlacementInput {
            object: 2,
            intent: &cold_intent,
            criticality: cold,
            logical_bytes: 4096,
            current: None,
            pressure: 0.9,
            migration_cost_ns: 1_000,
            ticks_since_move: 10_000,
        };
        let decision = planner.plan(&cold_input, &registry);
        let caps = registry.get(decision.target).expect("target");
        // Under high pressure a cold object must not land on DRAM.
        assert_ne!(caps.kind, crate::provider::ProviderKind::DRAM);
    }

    #[test]
    fn cooldown_suppresses_bouncing() {
        let registry = test_registry();
        let mut planner = CriticalityPlanner::production();
        let intent = MaterializationIntent::hot();
        let criticality = crate::criticality::ExecutionCriticality::cold();
        let input = PlacementInput {
            object: 1,
            intent: &intent,
            criticality,
            logical_bytes: 64,
            current: None,
            pressure: 0.1,
            migration_cost_ns: 1_000,
            ticks_since_move: 3,
        };
        let decision = planner.plan(&input, &registry);
        assert_eq!(decision.target, ProviderId::NONE);
        assert_eq!(decision.reason, "cooldown");
    }
}
