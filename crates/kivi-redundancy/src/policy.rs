//! Deterministic Phase 12 self-healing policy and redundancy preference adapter.

use core::time::Duration;

use kivi_observation::{
    DegradedAssetsSignal, LatencySummary, ObservationScope, ObservationSnapshot, QueueingSignal,
    RepairDebt, ScrubDebt, TypedSignal,
};
use kivi_types::Ticks;

use crate::dist::DistributedFabric;
use crate::healing::MaintenanceBudgets;
use crate::scheme::{fit_to_domains, plan_baseline};
use crate::{RedundancyError, RedundancyIntent, SchemeParams};

const SCORE_SCALE: u64 = 10_000;
const CRITICAL_SCORE: u64 = 8_000;
const PPM_SCALE: u64 = 1_000_000;

/// The largest asset that is treated as small for scheme preference.
pub const HOT_SMALL_MAX_BYTES: u64 = 64 * 1024;

/// Validation failures for the self-healing policy adapter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum PolicyError {
    /// The configured ceiling cannot make maintenance progress.
    #[error("self-healing policy ceiling is invalid")]
    InvalidCeiling,
    /// The configured mandatory floor is invalid.
    #[error("self-healing policy floor is invalid")]
    InvalidFloor,
    /// A mandatory floor is greater than its corresponding ceiling.
    #[error("self-healing policy floor exceeds its ceiling")]
    FloorExceedsCeiling,
    /// A threshold or bounded step is zero.
    #[error("self-healing policy field {field} must be nonzero")]
    InvalidField {
        /// Invalid field name.
        field: &'static str,
    },
    /// Hysteresis is not a strict fraction.
    #[error("self-healing policy hysteresis must be below one million")]
    InvalidHysteresis,
    /// The budgets currently installed on a fabric are invalid.
    #[error("current maintenance budgets are invalid")]
    InvalidCurrentBudgets,
    /// The policy failed to produce a valid budget.
    #[error("self-healing policy produced invalid maintenance budgets")]
    InvalidOutcome,
}

/// Typed observations consumed by the self-healing policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PolicyObservation {
    /// Typed repair debt carried by the observation window.
    pub repair_debt: RepairDebt,
    /// Typed scrub debt carried by the observation window.
    pub scrub_debt: ScrubDebt,
    /// Typed queue pressure and admission evidence.
    pub queueing: QueueingSignal,
    /// Typed bounded foreground latency summary.
    pub foreground_latency: LatencySummary,
    /// Typed degraded-asset count and age.
    pub degraded_assets: DegradedAssetsSignal,
}

/// Compatibility name for callers that describe the policy input as an input.
pub type PolicyInput = PolicyObservation;

impl PolicyObservation {
    /// Builds an observation from typed kivi-observation values.
    #[must_use]
    pub const fn new(
        repair_debt: RepairDebt,
        scrub_debt: ScrubDebt,
        queueing: QueueingSignal,
        foreground_latency: LatencySummary,
        degraded_assets: DegradedAssetsSignal,
    ) -> Self {
        Self {
            repair_debt,
            scrub_debt,
            queueing,
            foreground_latency,
            degraded_assets,
        }
    }

    /// Builds a zero-valued observation for a window without optional signals.
    #[must_use]
    pub const fn empty() -> Self {
        Self {
            repair_debt: RepairDebt {
                assets: 0,
                bytes: 0,
                oldest_age: Duration::ZERO,
            },
            scrub_debt: ScrubDebt {
                bytes: 0,
                oldest_age: Duration::ZERO,
            },
            queueing: QueueingSignal {
                depth: 0,
                wait_p50_ns: 0,
                wait_p99_ns: 0,
                wait_p999_ns: 0,
                admission_rejections: 0,
            },
            foreground_latency: LatencySummary {
                sample_count: 0,
                p50_ns: 0,
                p99_ns: 0,
                p999_ns: 0,
                max_ns: 0,
            },
            degraded_assets: DegradedAssetsSignal {
                assets: 0,
                oldest_age: Duration::ZERO,
            },
        }
    }

    /// Extracts policy input from a closed sequence of typed signals.
    #[must_use]
    pub fn from_signals(signals: &[TypedSignal]) -> Self {
        let mut observation = Self::empty();
        for signal in signals {
            match signal {
                TypedSignal::MaintenanceDebt(signal) => {
                    observation.repair_debt = signal.repair;
                    observation.scrub_debt = signal.scrub;
                }
                TypedSignal::Queueing(signal) => {
                    observation.queueing.depth = observation.queueing.depth.max(signal.depth);
                    observation.queueing.wait_p50_ns =
                        observation.queueing.wait_p50_ns.max(signal.wait_p50_ns);
                    observation.queueing.wait_p99_ns =
                        observation.queueing.wait_p99_ns.max(signal.wait_p99_ns);
                    observation.queueing.wait_p999_ns =
                        observation.queueing.wait_p999_ns.max(signal.wait_p999_ns);
                    observation.queueing.admission_rejections = observation
                        .queueing
                        .admission_rejections
                        .saturating_add(signal.admission_rejections);
                }
                TypedSignal::Latency(signal) => {
                    observation.foreground_latency.sample_count = observation
                        .foreground_latency
                        .sample_count
                        .saturating_add(1);
                    observation.foreground_latency.p50_ns = observation
                        .foreground_latency
                        .p50_ns
                        .max(signal.duration_ns);
                    observation.foreground_latency.p99_ns = observation
                        .foreground_latency
                        .p99_ns
                        .max(signal.duration_ns);
                    observation.foreground_latency.p999_ns = observation
                        .foreground_latency
                        .p999_ns
                        .max(signal.duration_ns);
                    observation.foreground_latency.max_ns = observation
                        .foreground_latency
                        .max_ns
                        .max(signal.duration_ns);
                }
                TypedSignal::DegradedAssets(signal) => {
                    observation.degraded_assets = *signal;
                }
                _ => {}
            }
        }
        observation
    }

    /// Extracts policy input from the most authoritative snapshot scope.
    ///
    /// A cluster scope is preferred; otherwise the last deterministically
    /// ordered scope is used. Missing optional signals become typed zeroes.
    #[must_use]
    pub fn from_snapshot(snapshot: &ObservationSnapshot) -> Option<Self> {
        let scope = snapshot
            .scopes
            .iter()
            .find(|scope| matches!(scope.scope, ObservationScope::Cluster(_)))
            .or_else(|| snapshot.scopes.last())?;
        let maintenance = scope.maintenance_debt;
        let repair_debt = maintenance.map_or_else(
            || RepairDebt {
                assets: 0,
                bytes: 0,
                oldest_age: Duration::ZERO,
            },
            |maintenance| RepairDebt {
                assets: maintenance.repair.assets,
                bytes: maintenance.repair.bytes,
                oldest_age: maintenance.repair.oldest_age,
            },
        );
        let scrub_debt = maintenance.map_or_else(
            || ScrubDebt {
                bytes: 0,
                oldest_age: Duration::ZERO,
            },
            |maintenance| ScrubDebt {
                bytes: maintenance.scrub.bytes,
                oldest_age: maintenance.scrub.oldest_age,
            },
        );
        let queueing = scope.queueing.map_or_else(
            || QueueingSignal {
                depth: 0,
                wait_p50_ns: 0,
                wait_p99_ns: 0,
                wait_p999_ns: 0,
                admission_rejections: 0,
            },
            |queue| QueueingSignal {
                depth: queue.max_depth,
                wait_p50_ns: finite_nonnegative_u64(queue.mean_wait_p50_ns),
                wait_p99_ns: finite_nonnegative_u64(queue.mean_wait_p99_ns),
                wait_p999_ns: finite_nonnegative_u64(queue.mean_wait_p999_ns),
                admission_rejections: queue.admission_rejections,
            },
        );
        let degraded = scope.degraded_assets.map_or_else(
            || DegradedAssetsSignal {
                assets: 0,
                oldest_age: Duration::ZERO,
            },
            |degraded| DegradedAssetsSignal {
                assets: degraded.assets,
                oldest_age: degraded.oldest_age,
            },
        );
        Some(Self::new(
            repair_debt,
            scrub_debt,
            queueing,
            scope.latency.unwrap_or(LatencySummary {
                sample_count: 0,
                p50_ns: 0,
                p99_ns: 0,
                p999_ns: 0,
                max_ns: 0,
            }),
            degraded,
        ))
    }

    /// Returns the typed foreground p99 used by the policy.
    #[must_use]
    pub const fn foreground_p99_ns(&self) -> u64 {
        self.foreground_latency.p99_ns
    }
}

/// Persistent state used to make budget changes replayable and hysteretic.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PolicyState {
    /// Last budget selected by the adapter.
    pub budgets: MaintenanceBudgets,
    /// Time at which the selected budget last changed.
    pub last_change_at: Option<Ticks>,
    /// Last urgency score, or `None` before the first decision.
    pub last_urgency: Option<u32>,
    /// Whether the previous decision suppressed foreground-sensitive work.
    pub last_foreground_suppressed: bool,
}

impl PolicyState {
    /// Creates state for a known current budget.
    #[must_use]
    pub const fn new(budgets: MaintenanceBudgets) -> Self {
        Self {
            budgets,
            last_change_at: None,
            last_urgency: None,
            last_foreground_suppressed: false,
        }
    }
}

impl Default for PolicyState {
    fn default() -> Self {
        Self::new(MaintenanceBudgets::conservative())
    }
}

/// Why the policy selected or retained a budget.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PolicyDecisionReason {
    /// The first decision established the initial bounded target.
    Initial,
    /// A safety floor or critical risk caused an immediate change.
    Safety,
    /// A cooldown or hysteresis guard retained the prior budget.
    Held,
    /// A configured bounded step moved toward the target.
    BoundedStep,
    /// The target was applied without a guard.
    Applied,
    /// The target and current budget were identical.
    Unchanged,
}

/// Complete, replayable explanation of one policy decision.
#[allow(clippy::struct_excessive_bools)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PolicyDecisionTrace {
    /// Decision time supplied by the caller.
    pub observed_at: Ticks,
    /// Repair debt assets observed.
    pub repair_debt_assets: u64,
    /// Repair debt bytes observed.
    pub repair_debt_bytes: u64,
    /// Age of the oldest repair debt item.
    pub repair_debt_age: Duration,
    /// Scrub debt bytes observed.
    pub scrub_debt_bytes: u64,
    /// Age of the oldest scrub debt item.
    pub scrub_debt_age: Duration,
    /// Degraded assets observed.
    pub degraded_assets: u64,
    /// Age of the oldest degraded asset.
    pub degraded_age: Duration,
    /// Foreground p99 latency observed.
    pub foreground_p99_ns: u64,
    /// Queue depth observed.
    pub queue_depth: u64,
    /// Queue p99 wait observed.
    pub queue_wait_p99_ns: u64,
    /// Admission rejections observed.
    pub admission_rejections: u64,
    /// Normalized repair risk score.
    pub repair_risk: u32,
    /// Normalized degradation risk score.
    pub degradation_risk: u32,
    /// Normalized overall urgency score.
    pub urgency: u32,
    /// Whether foreground-sensitive work was suppressed.
    pub foreground_suppressed: bool,
    /// Budget before the decision.
    pub previous: MaintenanceBudgets,
    /// Target after floors and ceilings were applied.
    pub target: MaintenanceBudgets,
    /// Budget after bounded steps and guards.
    pub budgets: MaintenanceBudgets,
    /// Whether the selected budget changed.
    pub changed: bool,
    /// Whether cooldown retained the prior budget.
    pub cooldown_held: bool,
    /// Whether hysteresis retained the prior urgency regime.
    pub hysteresis_held: bool,
    /// Whether a ceiling clamp was required.
    pub clamped_to_ceiling: bool,
    /// Whether a mandatory floor had to be enforced.
    pub floor_enforced: bool,
    /// Whether state was rebuilt because the fabric budget differed.
    pub state_reinitialized: bool,
    /// Stable reason category.
    pub reason: PolicyDecisionReason,
}

/// Result of applying a policy observation to current budgets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PolicyOutcome {
    /// Budget before the decision.
    pub previous: MaintenanceBudgets,
    /// Validated budget selected by the policy.
    pub budgets: MaintenanceBudgets,
    /// Full decision trace.
    pub trace: PolicyDecisionTrace,
    /// State to persist for the next replay position.
    pub state: PolicyState,
}

impl PolicyOutcome {
    /// Returns whether the authoritative budget needs replacement.
    #[must_use]
    pub fn changed(&self) -> bool {
        self.previous != self.budgets
    }
}

/// Bounded self-healing policy configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SelfHealingPolicy {
    /// Existing absolute budget ceilings.
    pub ceiling: MaintenanceBudgets,
    /// Minimum scrub assets per maintenance window.
    pub min_scrub_assets: usize,
    /// Minimum scrub bytes per maintenance window.
    pub min_scrub_bytes: u64,
    /// Minimum repair assets per maintenance window.
    pub min_repair_assets: usize,
    /// Minimum repair bytes per maintenance window.
    pub min_repair_bytes: u64,
    /// Minimum concurrent reconstruction slots.
    pub min_concurrent_reconstructions: usize,
    /// Minimum concurrent transfer slots.
    pub min_concurrent_transfers: usize,
    /// Minimum concurrent scrub slots.
    pub min_concurrent_scrubs: usize,
    /// Minimum queued maintenance units.
    pub min_queued_work: usize,
    /// Minimum reconstruction slots reserved for critical repair.
    pub min_critical_reserve: usize,
    /// Foreground p99 latency suppression threshold.
    pub foreground_p99_threshold_ns: u64,
    /// Queue depth suppression threshold.
    pub queue_depth_threshold: u64,
    /// Queue p99 wait suppression threshold.
    pub queue_wait_p99_threshold_ns: u64,
    /// Admission rejection suppression threshold.
    pub admission_rejection_threshold: u64,
    /// Repair asset count normalization threshold.
    pub repair_assets_threshold: u64,
    /// Repair byte normalization threshold.
    pub repair_bytes_threshold: u64,
    /// Repair age normalization threshold.
    pub repair_age_threshold: Duration,
    /// Scrub byte normalization threshold.
    pub scrub_bytes_threshold: u64,
    /// Scrub age normalization threshold.
    pub scrub_age_threshold: Duration,
    /// Degraded asset count normalization threshold.
    pub degraded_assets_threshold: u64,
    /// Degraded age normalization threshold.
    pub degradation_age_threshold: Duration,
    /// Hysteresis width in parts per million.
    pub hysteresis_ppm: u32,
    /// Minimum time between non-safety budget changes.
    pub cooldown: Duration,
    /// Maximum asset-count change per decision.
    pub max_step_assets: usize,
    /// Maximum byte change per decision.
    pub max_step_bytes: u64,
    /// Maximum concurrency change per decision.
    pub max_step_concurrency: usize,
    /// Whether foreground pressure suppresses discretionary work.
    pub suppress_foreground_work: bool,
}

impl SelfHealingPolicy {
    /// Conservative adaptive policy with wide deterministic ceilings.
    #[must_use]
    pub const fn conservative() -> Self {
        Self {
            ceiling: MaintenanceBudgets::test_wide(),
            min_scrub_assets: 1,
            min_scrub_bytes: 4 * 1024 * 1024,
            min_repair_assets: 1,
            min_repair_bytes: 4 * 1024 * 1024,
            min_concurrent_reconstructions: 1,
            min_concurrent_transfers: 1,
            min_concurrent_scrubs: 1,
            min_queued_work: 1,
            min_critical_reserve: 1,
            foreground_p99_threshold_ns: 10_000_000,
            queue_depth_threshold: 1024,
            queue_wait_p99_threshold_ns: 5_000_000,
            admission_rejection_threshold: 1,
            repair_assets_threshold: 32,
            repair_bytes_threshold: 64 * 1024 * 1024,
            repair_age_threshold: Duration::from_secs(300),
            scrub_bytes_threshold: 1024 * 1024 * 1024,
            scrub_age_threshold: Duration::from_hours(7 * 24),
            degraded_assets_threshold: 8,
            degradation_age_threshold: Duration::from_mins(15),
            hysteresis_ppm: 1_000,
            cooldown: Duration::from_secs(30),
            max_step_assets: 4,
            max_step_bytes: 32 * 1024 * 1024,
            max_step_concurrency: 2,
            suppress_foreground_work: true,
        }
    }

    /// Creates a conservative policy bounded by an existing ceiling.
    ///
    /// # Errors
    ///
    /// Returns [`PolicyError`] when a mandatory floor does not fit the ceiling.
    pub fn with_ceiling(ceiling: MaintenanceBudgets) -> Result<Self, PolicyError> {
        let mut policy = Self::conservative();
        policy.ceiling = ceiling;
        policy.validate()?;
        Ok(policy)
    }

    /// Returns a copy bounded by another ceiling.
    ///
    /// # Errors
    ///
    /// Returns [`PolicyError`] when a mandatory floor does not fit the ceiling.
    pub fn bounded_by(mut self, ceiling: &MaintenanceBudgets) -> Result<Self, PolicyError> {
        self.ceiling = *ceiling;
        self.validate()?;
        Ok(self)
    }

    /// Returns the mandatory floor as a validated budget.
    #[must_use]
    pub const fn floor_budgets(&self) -> MaintenanceBudgets {
        MaintenanceBudgets {
            max_scrub_assets: self.min_scrub_assets,
            max_scrub_bytes: self.min_scrub_bytes,
            max_repair_assets: self.min_repair_assets,
            max_repair_bytes: self.min_repair_bytes,
            max_concurrent_reconstructions: self.min_concurrent_reconstructions,
            max_concurrent_transfers: self.min_concurrent_transfers,
            max_concurrent_scrubs: self.min_concurrent_scrubs,
            max_queued_work: self.min_queued_work,
            critical_reserve: self.min_critical_reserve,
        }
    }

    /// Validates floors, ceilings, thresholds, and bounded steps.
    ///
    /// # Errors
    ///
    /// Returns [`PolicyError`] when any bound cannot produce safe progress.
    pub fn validate(&self) -> Result<(), PolicyError> {
        if !self.ceiling.validate() {
            return Err(PolicyError::InvalidCeiling);
        }
        let floor = self.floor_budgets();
        if !floor.validate() {
            return Err(PolicyError::InvalidFloor);
        }
        if floor.max_scrub_assets > self.ceiling.max_scrub_assets
            || floor.max_scrub_bytes > self.ceiling.max_scrub_bytes
            || floor.max_repair_assets > self.ceiling.max_repair_assets
            || floor.max_repair_bytes > self.ceiling.max_repair_bytes
            || floor.max_concurrent_reconstructions > self.ceiling.max_concurrent_reconstructions
            || floor.max_concurrent_transfers > self.ceiling.max_concurrent_transfers
            || floor.max_concurrent_scrubs > self.ceiling.max_concurrent_scrubs
            || floor.max_queued_work > self.ceiling.max_queued_work
            || floor.critical_reserve > self.ceiling.critical_reserve
        {
            return Err(PolicyError::FloorExceedsCeiling);
        }
        if self.foreground_p99_threshold_ns == 0
            || self.queue_depth_threshold == 0
            || self.queue_wait_p99_threshold_ns == 0
            || self.admission_rejection_threshold == 0
            || self.repair_assets_threshold == 0
            || self.repair_bytes_threshold == 0
            || self.repair_age_threshold.is_zero()
            || self.scrub_bytes_threshold == 0
            || self.scrub_age_threshold.is_zero()
            || self.degraded_assets_threshold == 0
            || self.degradation_age_threshold.is_zero()
            || self.max_step_assets == 0
            || self.max_step_bytes == 0
            || self.max_step_concurrency == 0
        {
            return Err(PolicyError::InvalidField {
                field: "threshold or step",
            });
        }
        if self.hysteresis_ppm >= u32::try_from(PPM_SCALE).unwrap_or(u32::MAX) {
            return Err(PolicyError::InvalidHysteresis);
        }
        Ok(())
    }

    /// Returns whether the policy passes validation.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.validate().is_ok()
    }

    /// Computes one stateless decision.
    ///
    /// # Errors
    ///
    /// Returns [`PolicyError`] when the policy or current budget is invalid.
    pub fn decide(
        &self,
        observation: &PolicyObservation,
        current: MaintenanceBudgets,
        now: Ticks,
    ) -> Result<PolicyOutcome, PolicyError> {
        self.decide_with_state(observation, current, now, PolicyState::new(current))
    }

    /// Computes one stateless evaluation.
    ///
    /// # Errors
    ///
    /// Returns [`PolicyError`] when the policy or current budget is invalid.
    pub fn evaluate(
        &self,
        observation: &PolicyObservation,
        current: MaintenanceBudgets,
        now: Ticks,
    ) -> Result<PolicyOutcome, PolicyError> {
        self.decide(observation, current, now)
    }

    /// Computes one decision using persisted hysteresis and cooldown state.
    ///
    /// # Errors
    ///
    /// Returns [`PolicyError`] when the policy, current budget, or result is invalid.
    pub fn update(
        &self,
        observation: &PolicyObservation,
        current: MaintenanceBudgets,
        now: Ticks,
        state: PolicyState,
    ) -> Result<PolicyOutcome, PolicyError> {
        self.decide_with_state(observation, current, now, state)
    }

    /// Computes one decision using persisted hysteresis and cooldown state.
    ///
    /// # Errors
    ///
    /// Returns [`PolicyError`] when the policy, current budget, or result is invalid.
    #[allow(clippy::too_many_lines)]
    pub fn decide_with_state(
        &self,
        observation: &PolicyObservation,
        current: MaintenanceBudgets,
        now: Ticks,
        state: PolicyState,
    ) -> Result<PolicyOutcome, PolicyError> {
        self.validate()?;
        if !current.validate() {
            return Err(PolicyError::InvalidCurrentBudgets);
        }

        let state_reinitialized = state.budgets != current;
        let state = if state_reinitialized {
            PolicyState::new(current)
        } else {
            state
        };
        let floor = self.floor_budgets();
        let ceiling = self.ceiling;
        let raw_foreground = self.foreground_pressure(observation);
        let foreground_score = self.foreground_score(observation);
        let foreground_suppressed = self.stabilize_suppression(
            raw_foreground,
            foreground_score,
            state.last_foreground_suppressed,
        );
        let repair_risk = self.repair_risk(observation);
        let degradation_risk = self.degradation_risk(observation);
        let scrub_risk = self.scrub_risk(observation);
        let raw_urgency =
            weighted_urgency(repair_risk, degradation_risk, scrub_risk, foreground_score);
        let critical = (observation.repair_debt.assets > 0 && repair_risk >= CRITICAL_SCORE)
            || (observation.degraded_assets.assets > 0 && degradation_risk >= CRITICAL_SCORE);
        let hysteresis_held = !critical
            && state.last_urgency.is_some_and(|last| {
                u64::from(last).abs_diff(u64::from(raw_urgency)) < u64::from(self.hysteresis_ppm)
            });
        let urgency = if hysteresis_held {
            u64::from(state.last_urgency.unwrap_or(raw_urgency))
        } else {
            u64::from(raw_urgency)
        };
        let repair_score = if critical {
            SCORE_SCALE
        } else if foreground_suppressed {
            0
        } else {
            urgency
        };
        let scrub_score = if foreground_suppressed { 0 } else { scrub_risk };
        let concurrency_score = if critical {
            SCORE_SCALE
        } else if foreground_suppressed {
            0
        } else {
            urgency
        };
        let reserve_score = if critical {
            SCORE_SCALE
        } else if foreground_suppressed {
            0
        } else {
            urgency
        };
        let raw_target = MaintenanceBudgets {
            max_scrub_assets: interpolate_usize(
                floor.max_scrub_assets,
                ceiling.max_scrub_assets,
                scrub_score,
            ),
            max_scrub_bytes: interpolate_u64(
                floor.max_scrub_bytes,
                ceiling.max_scrub_bytes,
                scrub_score,
            ),
            max_repair_assets: interpolate_usize(
                floor.max_repair_assets,
                ceiling.max_repair_assets,
                repair_score,
            ),
            max_repair_bytes: interpolate_u64(
                floor.max_repair_bytes,
                ceiling.max_repair_bytes,
                repair_score,
            ),
            max_concurrent_reconstructions: interpolate_usize(
                floor.max_concurrent_reconstructions,
                ceiling.max_concurrent_reconstructions,
                concurrency_score,
            ),
            max_concurrent_transfers: interpolate_usize(
                floor.max_concurrent_transfers,
                ceiling.max_concurrent_transfers,
                concurrency_score,
            ),
            max_concurrent_scrubs: interpolate_usize(
                floor.max_concurrent_scrubs,
                ceiling.max_concurrent_scrubs,
                if foreground_suppressed { 0 } else { scrub_risk },
            ),
            max_queued_work: interpolate_usize(
                floor.max_queued_work,
                ceiling.max_queued_work,
                if critical { SCORE_SCALE } else { urgency },
            ),
            critical_reserve: interpolate_usize(
                floor.critical_reserve,
                ceiling.critical_reserve,
                reserve_score,
            ),
        };
        let target = bound_budget(raw_target, &floor, &ceiling);
        let floor_enforced = below_floor(current, &floor);
        let clamped_to_ceiling =
            above_ceiling(current, &ceiling) || above_ceiling(raw_target, &ceiling);
        let mut budgets = if above_ceiling(current, &ceiling) {
            target
        } else {
            step_budget(
                current,
                target,
                &floor,
                self.max_step_assets,
                self.max_step_bytes,
                self.max_step_concurrency,
            )
        };
        let safety_override = floor_enforced
            || target.critical_reserve > current.critical_reserve
            || (critical && target.max_repair_assets > current.max_repair_assets);
        let cooldown_held = budgets != current
            && !safety_override
            && state
                .last_change_at
                .is_some_and(|last| now.saturating_since(last) < self.cooldown);
        if cooldown_held {
            budgets = current;
        }
        let bounded_step = budgets != current && budgets != target;
        let changed = budgets != current;
        let reason = if cooldown_held || hysteresis_held {
            PolicyDecisionReason::Held
        } else if floor_enforced || critical {
            PolicyDecisionReason::Safety
        } else if bounded_step {
            PolicyDecisionReason::BoundedStep
        } else if changed {
            PolicyDecisionReason::Applied
        } else if state_reinitialized {
            PolicyDecisionReason::Initial
        } else {
            PolicyDecisionReason::Unchanged
        };
        let trace = PolicyDecisionTrace {
            observed_at: now,
            repair_debt_assets: observation.repair_debt.assets,
            repair_debt_bytes: observation.repair_debt.bytes,
            repair_debt_age: observation.repair_debt.oldest_age,
            scrub_debt_bytes: observation.scrub_debt.bytes,
            scrub_debt_age: observation.scrub_debt.oldest_age,
            degraded_assets: observation.degraded_assets.assets,
            degraded_age: observation.degraded_assets.oldest_age,
            foreground_p99_ns: observation.foreground_p99_ns(),
            queue_depth: observation.queueing.depth,
            queue_wait_p99_ns: observation.queueing.wait_p99_ns,
            admission_rejections: observation.queueing.admission_rejections,
            repair_risk: u32::try_from(repair_risk).unwrap_or(u32::MAX),
            degradation_risk: u32::try_from(degradation_risk).unwrap_or(u32::MAX),
            urgency: u32::try_from(urgency).unwrap_or(u32::MAX),
            foreground_suppressed,
            previous: current,
            target,
            budgets,
            changed,
            cooldown_held,
            hysteresis_held,
            clamped_to_ceiling,
            floor_enforced,
            state_reinitialized,
            reason,
        };
        let next_state = PolicyState {
            budgets,
            last_change_at: if changed {
                Some(now)
            } else {
                state.last_change_at
            },
            last_urgency: Some(u32::try_from(urgency).unwrap_or(u32::MAX)),
            last_foreground_suppressed: foreground_suppressed,
        };
        if !budgets.validate() {
            return Err(PolicyError::InvalidOutcome);
        }
        Ok(PolicyOutcome {
            previous: current,
            budgets,
            trace,
            state: next_state,
        })
    }

    fn foreground_pressure(&self, observation: &PolicyObservation) -> bool {
        self.suppress_foreground_work
            && (observation.foreground_p99_ns() >= self.foreground_p99_threshold_ns
                || observation.queueing.depth >= self.queue_depth_threshold
                || observation.queueing.wait_p99_ns >= self.queue_wait_p99_threshold_ns
                || observation.queueing.admission_rejections >= self.admission_rejection_threshold)
    }

    fn stabilize_suppression(&self, raw: bool, score: u64, previous: bool) -> bool {
        if !self.suppress_foreground_work {
            return false;
        }
        if !previous {
            return raw;
        }
        let release = SCORE_SCALE.saturating_sub(u64::from(self.hysteresis_ppm));
        raw || score >= release
    }

    fn foreground_score(&self, observation: &PolicyObservation) -> u64 {
        max4(
            ratio_ppm(
                observation.foreground_p99_ns(),
                self.foreground_p99_threshold_ns,
            ),
            ratio_ppm(observation.queueing.depth, self.queue_depth_threshold),
            ratio_ppm(
                observation.queueing.wait_p99_ns,
                self.queue_wait_p99_threshold_ns,
            ),
            ratio_ppm(
                observation.queueing.admission_rejections,
                self.admission_rejection_threshold,
            ),
        )
    }

    fn repair_risk(&self, observation: &PolicyObservation) -> u64 {
        max3(
            ratio_ppm(observation.repair_debt.assets, self.repair_assets_threshold),
            ratio_ppm(observation.repair_debt.bytes, self.repair_bytes_threshold),
            duration_ratio_ppm(
                observation.repair_debt.oldest_age,
                self.repair_age_threshold,
            ),
        )
    }

    fn degradation_risk(&self, observation: &PolicyObservation) -> u64 {
        max3(
            ratio_ppm(
                observation.degraded_assets.assets,
                self.degraded_assets_threshold,
            ),
            duration_ratio_ppm(
                observation.repair_debt.oldest_age,
                self.repair_age_threshold,
            ),
            duration_ratio_ppm(
                observation.degraded_assets.oldest_age,
                self.degradation_age_threshold,
            ),
        )
    }

    fn scrub_risk(&self, observation: &PolicyObservation) -> u64 {
        max2(
            ratio_ppm(observation.scrub_debt.bytes, self.scrub_bytes_threshold),
            duration_ratio_ppm(observation.scrub_debt.oldest_age, self.scrub_age_threshold),
        )
    }
}

impl Default for SelfHealingPolicy {
    fn default() -> Self {
        Self::conservative()
    }
}

/// Stateful adapter that persists deterministic policy state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SelfHealingPolicyAdapter {
    policy: SelfHealingPolicy,
    state: PolicyState,
}

impl SelfHealingPolicyAdapter {
    /// Creates an adapter around a known current budget.
    ///
    /// # Errors
    ///
    /// Returns [`PolicyError`] when the policy or current budget is invalid.
    pub fn new(
        policy: SelfHealingPolicy,
        current: MaintenanceBudgets,
    ) -> Result<Self, PolicyError> {
        policy.validate()?;
        if !current.validate() {
            return Err(PolicyError::InvalidCurrentBudgets);
        }
        Ok(Self {
            policy,
            state: PolicyState::new(current),
        })
    }

    /// Creates an adapter from the budget currently installed on a fabric.
    ///
    /// # Errors
    ///
    /// Returns [`PolicyError`] when the policy or current budget is invalid.
    pub fn for_fabric(
        policy: SelfHealingPolicy,
        fabric: &DistributedFabric,
    ) -> Result<Self, PolicyError> {
        Self::new(policy, fabric.maintenance_budgets())
    }

    /// Returns the immutable policy configuration.
    #[must_use]
    pub const fn policy(&self) -> &SelfHealingPolicy {
        &self.policy
    }

    /// Returns the state to persist after the last decision.
    #[must_use]
    pub const fn state(&self) -> PolicyState {
        self.state
    }

    /// Resets adapter state to a known current budget.
    ///
    /// # Errors
    ///
    /// Returns [`PolicyError`] when the current budget is invalid.
    pub fn reset(&mut self, current: MaintenanceBudgets) -> Result<(), PolicyError> {
        if !current.validate() {
            return Err(PolicyError::InvalidCurrentBudgets);
        }
        self.state = PolicyState::new(current);
        Ok(())
    }

    /// Computes and persists a decision without applying it to a fabric.
    ///
    /// # Errors
    ///
    /// Returns [`PolicyError`] when the policy or current budget is invalid.
    pub fn decide(
        &mut self,
        observation: &PolicyObservation,
        now: Ticks,
    ) -> Result<PolicyOutcome, PolicyError> {
        let outcome =
            self.policy
                .decide_with_state(observation, self.state.budgets, now, self.state)?;
        self.state = outcome.state;
        Ok(outcome)
    }

    /// Applies a decision through `DistributedFabric`'s authoritative setter.
    ///
    /// # Errors
    ///
    /// Returns [`PolicyError`] when the policy or current budget is invalid.
    pub fn apply(
        &mut self,
        fabric: &mut DistributedFabric,
        observation: &PolicyObservation,
        now: Ticks,
    ) -> Result<PolicyOutcome, PolicyError> {
        let current = fabric.maintenance_budgets();
        let outcome = self
            .policy
            .decide_with_state(observation, current, now, self.state)?;
        let _ = fabric.set_maintenance_budgets(outcome.budgets);
        self.state = outcome.state;
        Ok(outcome)
    }
}

/// Applies a policy through the distributed fabric's authoritative budget setter.
///
/// # Errors
///
/// Returns [`PolicyError`] when the policy or current budget is invalid.
pub fn apply_policy(
    fabric: &mut DistributedFabric,
    adapter: &mut SelfHealingPolicyAdapter,
    observation: &PolicyObservation,
    now: Ticks,
) -> Result<PolicyOutcome, PolicyError> {
    adapter.apply(fabric, observation, now)
}

/// Alias for [`apply_policy`].
///
/// # Errors
///
/// Returns [`PolicyError`] when the policy or current budget is invalid.
pub fn apply_maintenance_policy(
    fabric: &mut DistributedFabric,
    adapter: &mut SelfHealingPolicyAdapter,
    observation: &PolicyObservation,
    now: Ticks,
) -> Result<PolicyOutcome, PolicyError> {
    apply_policy(fabric, adapter, observation, now)
}

/// Alias for [`apply_policy`] emphasizing the update operation.
///
/// # Errors
///
/// Returns [`PolicyError`] when the policy or current budget is invalid.
pub fn update_maintenance_budgets(
    fabric: &mut DistributedFabric,
    adapter: &mut SelfHealingPolicyAdapter,
    observation: &PolicyObservation,
    now: Ticks,
) -> Result<PolicyOutcome, PolicyError> {
    apply_policy(fabric, adapter, observation, now)
}

impl DistributedFabric {
    /// Applies a self-healing policy through the existing budget setter.
    ///
    /// # Errors
    ///
    /// Returns [`PolicyError`] when the policy or current budget is invalid.
    pub fn apply_self_healing_policy(
        &mut self,
        adapter: &mut SelfHealingPolicyAdapter,
        observation: &PolicyObservation,
        now: Ticks,
    ) -> Result<PolicyOutcome, PolicyError> {
        apply_policy(self, adapter, observation, now)
    }
}

/// Temperature class used to prefer a redundancy scheme family.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AssetTemperature {
    /// The asset is expected to be read infrequently.
    Cold,
    /// The asset is expected to be read frequently.
    Hot,
}

impl AssetTemperature {
    /// Returns whether the asset is hot.
    #[must_use]
    pub const fn is_hot(self) -> bool {
        matches!(self, Self::Hot)
    }
}

/// Why a scheme preference was selected or held.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SchemePreferenceReason {
    /// The asset is small or hot and replication was preferred.
    HotSmall,
    /// The asset is large and cold and erasure coding was preferred.
    ColdLarge,
    /// The current scheme already satisfies the preference.
    Current,
    /// A transition was held by the configured cooldown.
    Cooldown,
    /// The current scheme violated the intent and was replaced immediately.
    SafetyRequirement,
}

/// Result of one anti-thrashing scheme preference decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SchemePreferenceDecision {
    /// Current scheme before the decision.
    pub current: Option<SchemeParams>,
    /// Scheme preferred by the current temperature and intent.
    pub preferred: SchemeParams,
    /// Scheme to install; equal to current when held.
    pub target: SchemeParams,
    /// Whether a transition should be started.
    pub transition: bool,
    /// Whether cooldown held a safe family change.
    pub cooldown_held: bool,
    /// Stable reason category.
    pub reason: SchemePreferenceReason,
}

/// Stateful scheme preference with transition cooldown.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SchemePreference {
    current: Option<SchemeParams>,
    last_transition: Option<Ticks>,
    cooldown: Duration,
}

impl SchemePreference {
    /// Creates a preference with no installed scheme.
    #[must_use]
    pub const fn new(cooldown: Duration) -> Self {
        Self {
            current: None,
            last_transition: None,
            cooldown,
        }
    }

    /// Creates a preference around an already installed scheme.
    #[must_use]
    pub const fn with_current(current: SchemeParams, cooldown: Duration) -> Self {
        Self {
            current: Some(current),
            last_transition: None,
            cooldown,
        }
    }

    /// Resumes persisted preference state after a restart.
    #[must_use]
    pub const fn resume(
        current: Option<SchemeParams>,
        last_transition: Option<Ticks>,
        cooldown: Duration,
    ) -> Self {
        Self {
            current,
            last_transition,
            cooldown,
        }
    }

    /// Returns the currently retained scheme.
    #[must_use]
    pub const fn current(&self) -> Option<SchemeParams> {
        self.current
    }

    /// Returns the time of the last accepted transition.
    #[must_use]
    pub const fn last_transition(&self) -> Option<Ticks> {
        self.last_transition
    }

    /// Returns the configured transition cooldown.
    #[must_use]
    pub const fn cooldown(&self) -> Duration {
        self.cooldown
    }

    /// Chooses a safe target and records accepted family transitions.
    ///
    /// # Errors
    ///
    /// Returns [`RedundancyError`] when the intent or selected scheme is invalid.
    pub fn decide(
        &mut self,
        intent: &RedundancyIntent,
        logical_len: u64,
        temperature: AssetTemperature,
        independent_domains: usize,
        now: Ticks,
    ) -> Result<SchemePreferenceDecision, RedundancyError> {
        let preferred =
            prefer_redundancy_scheme(intent, logical_len, temperature, independent_domains)?;
        let current = self.current;
        let Some(current) = current else {
            self.current = Some(preferred);
            self.last_transition = Some(now);
            return Ok(SchemePreferenceDecision {
                current: None,
                preferred,
                target: preferred,
                transition: true,
                cooldown_held: false,
                reason: reason_for(temperature),
            });
        };
        if !scheme_satisfies_intent(current, intent) {
            self.current = Some(preferred);
            self.last_transition = Some(now);
            return Ok(SchemePreferenceDecision {
                current: Some(current),
                preferred,
                target: preferred,
                transition: preferred != current,
                cooldown_held: false,
                reason: SchemePreferenceReason::SafetyRequirement,
            });
        }
        if current.scheme() == preferred.scheme() {
            return Ok(SchemePreferenceDecision {
                current: Some(current),
                preferred,
                target: current,
                transition: false,
                cooldown_held: false,
                reason: SchemePreferenceReason::Current,
            });
        }
        if self
            .last_transition
            .is_some_and(|last| now.saturating_since(last) < self.cooldown)
        {
            return Ok(SchemePreferenceDecision {
                current: Some(current),
                preferred,
                target: current,
                transition: false,
                cooldown_held: true,
                reason: SchemePreferenceReason::Cooldown,
            });
        }
        self.current = Some(preferred);
        self.last_transition = Some(now);
        Ok(SchemePreferenceDecision {
            current: Some(current),
            preferred,
            target: preferred,
            transition: preferred != current,
            cooldown_held: false,
            reason: reason_for(temperature),
        })
    }
}

impl Default for SchemePreference {
    fn default() -> Self {
        Self::new(Duration::from_secs(60))
    }
}

/// Selects replication for hot or small assets and coding for cold large assets.
///
/// The selected parameters are fitted to the independent-domain count and
/// checked against the intent before they are returned.
///
/// # Errors
///
/// Returns [`RedundancyError`] when the intent is invalid or the available
/// domains cannot satisfy its tolerance and minimum-available requirements.
pub fn prefer_redundancy_scheme(
    intent: &RedundancyIntent,
    logical_len: u64,
    temperature: AssetTemperature,
    independent_domains: usize,
) -> Result<SchemeParams, RedundancyError> {
    intent.validate()?;
    let mut preference_intent = intent.clone();
    preference_intent.repair.prefer_direct_reads =
        temperature.is_hot() || logical_len <= HOT_SMALL_MAX_BYTES;
    let baseline = plan_baseline(&preference_intent, logical_len)?;
    let target = fit_to_domains(baseline, logical_len, independent_domains);
    target.validate()?;
    if target.tolerance() < u32::from(intent.tolerance)
        || target.required_pieces() < intent.min_available
    {
        return Err(RedundancyError::invalid_params(format!(
            "preferred scheme cannot satisfy tolerance {} and minimum available {} with {independent_domains} domains",
            intent.tolerance, intent.min_available
        )));
    }
    Ok(target)
}

/// Convenience helper using a boolean hotness value.
///
/// # Errors
///
/// Returns [`RedundancyError`] when the preference cannot satisfy the intent.
pub fn preferred_scheme(
    intent: &RedundancyIntent,
    logical_len: u64,
    hot: bool,
    independent_domains: usize,
) -> Result<SchemeParams, RedundancyError> {
    let temperature = if hot {
        AssetTemperature::Hot
    } else {
        AssetTemperature::Cold
    };
    prefer_redundancy_scheme(intent, logical_len, temperature, independent_domains)
}

/// Returns whether a scheme meets the intent's semantic safety contract.
#[must_use]
pub fn scheme_satisfies_intent(params: SchemeParams, intent: &RedundancyIntent) -> bool {
    params.validate().is_ok()
        && params.tolerance() >= u32::from(intent.tolerance)
        && params.required_pieces() >= intent.min_available
}

fn reason_for(temperature: AssetTemperature) -> SchemePreferenceReason {
    match temperature {
        AssetTemperature::Hot => SchemePreferenceReason::HotSmall,
        AssetTemperature::Cold => SchemePreferenceReason::ColdLarge,
    }
}

fn weighted_urgency(repair: u64, degradation: u64, scrub: u64, foreground: u64) -> u32 {
    let score = repair
        .saturating_mul(45)
        .saturating_add(degradation.saturating_mul(35))
        .saturating_add(scrub.saturating_mul(15))
        .saturating_add(foreground.saturating_mul(5));
    u32::try_from(score / 100).unwrap_or(u32::MAX)
}

fn ratio_ppm(value: u64, threshold: u64) -> u64 {
    if threshold == 0 {
        return if value == 0 { 0 } else { SCORE_SCALE };
    }
    let numerator = u128::from(value).saturating_mul(u128::from(SCORE_SCALE));
    let denominator = u128::from(threshold);
    let ratio = numerator / denominator;
    u64::try_from(ratio.min(u128::from(SCORE_SCALE))).unwrap_or(SCORE_SCALE)
}

fn duration_ratio_ppm(value: Duration, threshold: Duration) -> u64 {
    let value = u64::try_from(value.as_micros()).unwrap_or(u64::MAX);
    let threshold = u64::try_from(threshold.as_micros()).unwrap_or(u64::MAX);
    ratio_ppm(value, threshold)
}

fn max2(left: u64, right: u64) -> u64 {
    left.max(right)
}

fn max3(first: u64, second: u64, third: u64) -> u64 {
    max2(max2(first, second), third)
}

fn max4(first: u64, second: u64, third: u64, fourth: u64) -> u64 {
    max2(max3(first, second, third), fourth)
}

fn interpolate_u64(floor: u64, ceiling: u64, score: u64) -> u64 {
    let span = u128::from(ceiling.saturating_sub(floor));
    let scaled = span.saturating_mul(u128::from(score)) / u128::from(SCORE_SCALE);
    floor.saturating_add(u64::try_from(scaled).unwrap_or(u64::MAX))
}

fn interpolate_usize(floor: usize, ceiling: usize, score: u64) -> usize {
    let span = u128::try_from(ceiling.saturating_sub(floor)).unwrap_or(u128::MAX);
    let scaled = span.saturating_mul(u128::from(score)) / u128::from(SCORE_SCALE);
    usize::try_from(scaled).map_or(usize::MAX, |value| floor.saturating_add(value))
}

fn bound_budget(
    value: MaintenanceBudgets,
    floor: &MaintenanceBudgets,
    ceiling: &MaintenanceBudgets,
) -> MaintenanceBudgets {
    MaintenanceBudgets {
        max_scrub_assets: value
            .max_scrub_assets
            .clamp(floor.max_scrub_assets, ceiling.max_scrub_assets),
        max_scrub_bytes: value
            .max_scrub_bytes
            .clamp(floor.max_scrub_bytes, ceiling.max_scrub_bytes),
        max_repair_assets: value
            .max_repair_assets
            .clamp(floor.max_repair_assets, ceiling.max_repair_assets),
        max_repair_bytes: value
            .max_repair_bytes
            .clamp(floor.max_repair_bytes, ceiling.max_repair_bytes),
        max_concurrent_reconstructions: value.max_concurrent_reconstructions.clamp(
            floor.max_concurrent_reconstructions,
            ceiling.max_concurrent_reconstructions,
        ),
        max_concurrent_transfers: value.max_concurrent_transfers.clamp(
            floor.max_concurrent_transfers,
            ceiling.max_concurrent_transfers,
        ),
        max_concurrent_scrubs: value
            .max_concurrent_scrubs
            .clamp(floor.max_concurrent_scrubs, ceiling.max_concurrent_scrubs),
        max_queued_work: value
            .max_queued_work
            .clamp(floor.max_queued_work, ceiling.max_queued_work),
        critical_reserve: value
            .critical_reserve
            .clamp(floor.critical_reserve, ceiling.critical_reserve),
    }
}

fn step_budget(
    current: MaintenanceBudgets,
    target: MaintenanceBudgets,
    floor: &MaintenanceBudgets,
    max_step_assets: usize,
    max_step_bytes: u64,
    max_step_concurrency: usize,
) -> MaintenanceBudgets {
    MaintenanceBudgets {
        max_scrub_assets: step_usize(
            current.max_scrub_assets,
            target.max_scrub_assets,
            floor.max_scrub_assets,
            max_step_assets,
        ),
        max_scrub_bytes: step_u64(
            current.max_scrub_bytes,
            target.max_scrub_bytes,
            floor.max_scrub_bytes,
            max_step_bytes,
        ),
        max_repair_assets: step_usize(
            current.max_repair_assets,
            target.max_repair_assets,
            floor.max_repair_assets,
            max_step_assets,
        ),
        max_repair_bytes: step_u64(
            current.max_repair_bytes,
            target.max_repair_bytes,
            floor.max_repair_bytes,
            max_step_bytes,
        ),
        max_concurrent_reconstructions: step_usize(
            current.max_concurrent_reconstructions,
            target.max_concurrent_reconstructions,
            floor.max_concurrent_reconstructions,
            max_step_concurrency,
        ),
        max_concurrent_transfers: step_usize(
            current.max_concurrent_transfers,
            target.max_concurrent_transfers,
            floor.max_concurrent_transfers,
            max_step_concurrency,
        ),
        max_concurrent_scrubs: step_usize(
            current.max_concurrent_scrubs,
            target.max_concurrent_scrubs,
            floor.max_concurrent_scrubs,
            max_step_concurrency,
        ),
        max_queued_work: step_usize(
            current.max_queued_work,
            target.max_queued_work,
            floor.max_queued_work,
            max_step_assets,
        ),
        critical_reserve: step_usize(
            current.critical_reserve,
            target.critical_reserve,
            floor.critical_reserve,
            max_step_concurrency,
        ),
    }
}

fn step_usize(current: usize, target: usize, floor: usize, step: usize) -> usize {
    if current < floor {
        floor
    } else {
        move_toward_usize(current, target, step)
    }
}

fn step_u64(current: u64, target: u64, floor: u64, step: u64) -> u64 {
    if current < floor {
        floor
    } else {
        move_toward_u64(current, target, step)
    }
}

fn move_toward_usize(current: usize, target: usize, step: usize) -> usize {
    if current < target {
        current.saturating_add(step.min(target - current))
    } else {
        current.saturating_sub(step.min(current - target))
    }
}

fn move_toward_u64(current: u64, target: u64, step: u64) -> u64 {
    if current < target {
        current.saturating_add(step.min(target - current))
    } else {
        current.saturating_sub(step.min(current - target))
    }
}

fn below_floor(value: MaintenanceBudgets, floor: &MaintenanceBudgets) -> bool {
    value.max_scrub_assets < floor.max_scrub_assets
        || value.max_scrub_bytes < floor.max_scrub_bytes
        || value.max_repair_assets < floor.max_repair_assets
        || value.max_repair_bytes < floor.max_repair_bytes
        || value.max_concurrent_reconstructions < floor.max_concurrent_reconstructions
        || value.max_concurrent_transfers < floor.max_concurrent_transfers
        || value.max_concurrent_scrubs < floor.max_concurrent_scrubs
        || value.max_queued_work < floor.max_queued_work
        || value.critical_reserve < floor.critical_reserve
}

fn above_ceiling(value: MaintenanceBudgets, ceiling: &MaintenanceBudgets) -> bool {
    value.max_scrub_assets > ceiling.max_scrub_assets
        || value.max_scrub_bytes > ceiling.max_scrub_bytes
        || value.max_repair_assets > ceiling.max_repair_assets
        || value.max_repair_bytes > ceiling.max_repair_bytes
        || value.max_concurrent_reconstructions > ceiling.max_concurrent_reconstructions
        || value.max_concurrent_transfers > ceiling.max_concurrent_transfers
        || value.max_concurrent_scrubs > ceiling.max_concurrent_scrubs
        || value.max_queued_work > ceiling.max_queued_work
        || value.critical_reserve > ceiling.critical_reserve
}

#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss
)]
fn finite_nonnegative_u64(value: f64) -> u64 {
    if value.is_finite() && value > 0.0 {
        value as u64
    } else {
        0
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;

    fn observation(
        repair_assets: u64,
        repair_age: Duration,
        foreground_p99_ns: u64,
        queue_depth: u64,
    ) -> PolicyObservation {
        PolicyObservation::new(
            RepairDebt {
                assets: repair_assets,
                bytes: repair_assets.saturating_mul(1024 * 1024),
                oldest_age: repair_age,
            },
            ScrubDebt {
                bytes: 0,
                oldest_age: Duration::ZERO,
            },
            QueueingSignal {
                depth: queue_depth,
                wait_p50_ns: 0,
                wait_p99_ns: queue_depth,
                wait_p999_ns: queue_depth,
                admission_rejections: 0,
            },
            LatencySummary {
                sample_count: 1,
                p50_ns: foreground_p99_ns,
                p99_ns: foreground_p99_ns,
                p999_ns: foreground_p99_ns,
                max_ns: foreground_p99_ns,
            },
            DegradedAssetsSignal {
                assets: 0,
                oldest_age: Duration::ZERO,
            },
        )
    }

    fn policy() -> SelfHealingPolicy {
        let mut policy = SelfHealingPolicy::conservative();
        policy.min_critical_reserve = 3;
        policy.min_concurrent_reconstructions = 3;
        policy.min_repair_assets = 2;
        policy.min_repair_bytes = 8 * 1024 * 1024;
        policy.cooldown = Duration::from_secs(60);
        policy
    }

    #[test]
    fn budgets_are_clamped_to_the_configured_ceiling() {
        let policy = SelfHealingPolicy::with_ceiling(MaintenanceBudgets::conservative())
            .expect("bounded policy");
        let outcome = policy
            .decide(
                &PolicyObservation::empty(),
                MaintenanceBudgets::test_wide(),
                Ticks::from_micros(1),
            )
            .expect("decision");
        assert!(outcome.budgets.max_scrub_assets <= policy.ceiling.max_scrub_assets);
        assert!(outcome.budgets.max_repair_assets <= policy.ceiling.max_repair_assets);
        assert!(outcome.budgets.max_repair_bytes <= policy.ceiling.max_repair_bytes);
        assert!(outcome.trace.clamped_to_ceiling);
    }

    #[test]
    fn critical_repair_reserve_is_preserved() {
        let policy = policy();
        let current = MaintenanceBudgets::test_wide();
        let outcome = policy
            .decide(
                &observation(100, Duration::from_secs(3600), 0, 0),
                current,
                Ticks::from_micros(1),
            )
            .expect("decision");
        assert!(outcome.budgets.critical_reserve >= policy.min_critical_reserve);
        assert!(outcome.budgets.max_concurrent_reconstructions >= outcome.budgets.critical_reserve);
        assert!(outcome.budgets.max_repair_assets >= policy.min_repair_assets);
        assert!(outcome.budgets.max_repair_bytes >= policy.min_repair_bytes);
        assert!(outcome.budgets.validate());
    }

    #[test]
    fn foreground_suppression_reduces_discretionary_work() {
        let policy = policy();
        let current = MaintenanceBudgets::test_wide();
        let outcome = policy
            .decide(
                &observation(0, Duration::ZERO, 100_000_000, 10_000),
                current,
                Ticks::from_micros(1),
            )
            .expect("decision");
        assert!(outcome.trace.foreground_suppressed);
        assert!(outcome.budgets.max_scrub_assets < current.max_scrub_assets);
        assert!(outcome.budgets.max_scrub_bytes < current.max_scrub_bytes);
        assert!(outcome.budgets.max_concurrent_scrubs < current.max_concurrent_scrubs);
        assert!(outcome.budgets.critical_reserve >= policy.min_critical_reserve);
    }

    #[test]
    fn scheme_preference_preserves_intent_and_transition_cooldown() {
        let intent = RedundancyIntent::survive_two();
        let cold = prefer_redundancy_scheme(&intent, 4 * 1024 * 1024, AssetTemperature::Cold, 12)
            .expect("cold preference");
        let hot = preferred_scheme(&intent, 1024, true, 12).expect("hot preference");
        assert!(matches!(cold, SchemeParams::ReedSolomon(_)));
        assert!(matches!(hot, SchemeParams::Replication(_)));
        assert!(scheme_satisfies_intent(cold, &intent));
        assert!(scheme_satisfies_intent(hot, &intent));

        let mut preference = SchemePreference::new(Duration::from_secs(60));
        let first = preference
            .decide(
                &intent,
                4 * 1024 * 1024,
                AssetTemperature::Cold,
                12,
                Ticks::from_micros(1),
            )
            .expect("initial preference");
        assert!(first.transition);
        let held = preference
            .decide(
                &intent,
                1024,
                AssetTemperature::Hot,
                12,
                Ticks::from_micros(2),
            )
            .expect("held preference");
        assert!(held.cooldown_held);
        assert_eq!(held.target, cold);
        let changed = preference
            .decide(
                &intent,
                1024,
                AssetTemperature::Hot,
                12,
                Ticks::from_micros(60_000_001),
            )
            .expect("transition preference");
        assert!(changed.transition);
        assert_eq!(changed.target, hot);
    }

    #[test]
    fn deterministic_replay_returns_identical_trace() {
        let policy = policy();
        let current = MaintenanceBudgets::conservative();
        let input = observation(8, Duration::from_secs(120), 0, 0);
        let first = policy
            .decide(&input, current, Ticks::from_micros(100))
            .expect("first decision");
        let second = policy
            .decide(&input, current, Ticks::from_micros(100))
            .expect("replayed decision");
        assert_eq!(first, second);
    }

    #[test]
    fn apply_policy_uses_the_authoritative_setter() {
        let transport = Arc::new(crate::transport::LoopbackTransport::new());
        let catalog = Arc::new(crate::catalog::MemCatalog::new());
        let current = MaintenanceBudgets::test_wide();
        let mut fabric = DistributedFabric::with_maintenance_budgets(
            Arc::clone(&transport) as Arc<dyn crate::transport::FragmentTransport>,
            Arc::clone(&catalog) as Arc<dyn crate::catalog::LayoutCatalog>,
            Arc::clone(&catalog) as Arc<dyn crate::catalog::CatalogPublish>,
            Arc::new(crate::metrics::FabricMetrics::new()),
            crate::dist::DistConfig::conservative(),
            current,
        )
        .expect("fabric");
        let mut adapter =
            SelfHealingPolicyAdapter::new(SelfHealingPolicy::default(), current).expect("adapter");
        let outcome = apply_policy(
            &mut fabric,
            &mut adapter,
            &observation(0, Duration::ZERO, 100_000_000, 10_000),
            Ticks::from_micros(1),
        )
        .expect("apply");
        assert_eq!(fabric.maintenance_budgets(), outcome.budgets);
    }

    #[test]
    fn cooldown_prevents_alternating_pressure_from_thrashing() {
        let policy = policy();
        let current = MaintenanceBudgets::test_wide();
        let mut adapter = SelfHealingPolicyAdapter::new(policy, current).expect("adapter");
        let high = observation(0, Duration::ZERO, 100_000_000, 10_000);
        let low = observation(0, Duration::ZERO, 0, 0);
        let first = adapter
            .decide(&high, Ticks::from_micros(1))
            .expect("high pressure");
        let second = adapter
            .decide(&low, Ticks::from_micros(2))
            .expect("low pressure");
        let third = adapter
            .decide(&high, Ticks::from_micros(3))
            .expect("high pressure again");
        assert!(first.changed());
        assert!(!second.changed());
        assert!(!third.changed());
        assert_eq!(second.budgets, third.budgets);
    }
}
