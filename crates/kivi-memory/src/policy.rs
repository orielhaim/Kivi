//! Bounded control policy for memory placement and movement.

use kivi_observation::UnitInterval;

use crate::movement::MovementBudget;

/// Default target fraction of arena capacity that may remain resident.
pub const DEFAULT_DRAM_RESIDENCY_TARGET: f64 = 0.70;
/// Default compression aggressiveness.
pub const DEFAULT_COMPRESSION_AGGRESSIVENESS: f64 = 1.0;
/// Default pressure at which an `NVMe` demotion is allowed.
pub const DEFAULT_NVME_DEMOTION_PRESSURE: f64 = 0.75;
/// Default normalized criticality required for background promotion.
pub const DEFAULT_PROMOTION_THRESHOLD: f64 = 0.0;
/// Default criticality multiplier.
pub const DEFAULT_CRITICALITY_WEIGHTING: f64 = 1.0;

/// A bounded set of controls for memory placement decisions.
///
/// Fractions are represented by [`UnitInterval`], so policy values cannot
/// escape the closed interval `[0, 1]`. The movement budget is reset when
/// applied to a fabric and is additionally bounded by that fabric's
/// configured resource ceiling.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MemoryControlPolicy {
    /// Preferred upper bound for resident DRAM capacity.
    pub dram_residency_target: UnitInterval,
    /// Fraction of eligible cold objects that should be compressed.
    pub compression_aggressiveness: UnitInterval,
    /// Pressure at which a demotion to `NVMe` is allowed.
    pub nvme_demotion_pressure: UnitInterval,
    /// Per-tick movement limits.
    pub movement_budget: MovementBudget,
    /// Normalized criticality required for background promotion.
    pub promotion_threshold: UnitInterval,
    /// Multiplier applied to criticality scores during planning.
    pub criticality_weighting: UnitInterval,
}

/// Validation failure for a memory control policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum MemoryControlPolicyError {
    /// A policy fraction was outside the closed unit interval.
    #[error("memory control policy {field} must be in [0, 1]")]
    InvalidFraction {
        /// Invalid field name.
        field: &'static str,
    },
    /// A movement budget contained invalid accounting or limits.
    #[error("memory control policy movement budget is invalid: {field}")]
    InvalidMovementBudget {
        /// Invalid budget field name.
        field: &'static str,
    },
}

impl MemoryControlPolicy {
    /// Creates a policy from fractions when every value is valid.
    #[must_use]
    pub fn new(
        dram_residency_target: f64,
        compression_aggressiveness: f64,
        nvme_demotion_pressure: f64,
        movement_budget: MovementBudget,
        promotion_threshold: f64,
        criticality_weighting: f64,
    ) -> Option<Self> {
        Self::try_new(
            dram_residency_target,
            compression_aggressiveness,
            nvme_demotion_pressure,
            movement_budget,
            promotion_threshold,
            criticality_weighting,
        )
        .ok()
    }

    /// Creates a policy from fractions and reports validation failures.
    ///
    /// # Errors
    ///
    /// Returns [`MemoryControlPolicyError`] when a fraction or movement
    /// budget is outside its valid domain.
    pub fn try_new(
        dram_residency_target: f64,
        compression_aggressiveness: f64,
        nvme_demotion_pressure: f64,
        movement_budget: MovementBudget,
        promotion_threshold: f64,
        criticality_weighting: f64,
    ) -> Result<Self, MemoryControlPolicyError> {
        Self::from_typed(
            fraction(dram_residency_target, "dram_residency_target")?,
            fraction(compression_aggressiveness, "compression_aggressiveness")?,
            fraction(nvme_demotion_pressure, "nvme_demotion_pressure")?,
            movement_budget,
            fraction(promotion_threshold, "promotion_threshold")?,
            fraction(criticality_weighting, "criticality_weighting")?,
        )
    }

    /// Creates a policy from already bounded fractions when valid.
    #[must_use]
    pub fn new_typed(
        dram_residency_target: UnitInterval,
        compression_aggressiveness: UnitInterval,
        nvme_demotion_pressure: UnitInterval,
        movement_budget: MovementBudget,
        promotion_threshold: UnitInterval,
        criticality_weighting: UnitInterval,
    ) -> Option<Self> {
        Self::from_typed(
            dram_residency_target,
            compression_aggressiveness,
            nvme_demotion_pressure,
            movement_budget,
            promotion_threshold,
            criticality_weighting,
        )
        .ok()
    }

    /// Creates a policy from already bounded fractions.
    ///
    /// # Errors
    ///
    /// Returns [`MemoryControlPolicyError`] when the movement budget is
    /// invalid.
    pub fn from_typed(
        dram_residency_target: UnitInterval,
        compression_aggressiveness: UnitInterval,
        nvme_demotion_pressure: UnitInterval,
        movement_budget: MovementBudget,
        promotion_threshold: UnitInterval,
        criticality_weighting: UnitInterval,
    ) -> Result<Self, MemoryControlPolicyError> {
        let policy = Self {
            dram_residency_target,
            compression_aggressiveness,
            nvme_demotion_pressure,
            movement_budget,
            promotion_threshold,
            criticality_weighting,
        };
        policy.validate()?;
        Ok(policy)
    }

    /// Validates the movement accounting carried by this policy.
    ///
    /// # Errors
    ///
    /// Returns [`MemoryControlPolicyError::InvalidMovementBudget`] when a
    /// used counter exceeds its corresponding limit.
    pub fn validate(&self) -> Result<(), MemoryControlPolicyError> {
        self.movement_budget
            .validate()
            .map_err(|field| MemoryControlPolicyError::InvalidMovementBudget { field })
    }

    /// Returns whether the policy passes validation.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.validate().is_ok()
    }

    /// Returns the policy with a fresh movement budget bounded by `ceiling`.
    #[must_use]
    pub fn bounded_by(mut self, ceiling: &MovementBudget) -> Self {
        self.movement_budget = self.movement_budget.bounded_by(ceiling);
        self
    }

    /// Returns a copy with a fresh replacement movement budget.
    ///
    /// # Errors
    ///
    /// Returns [`MemoryControlPolicyError`] when `movement_budget` is
    /// invalid.
    pub fn with_movement_budget(
        mut self,
        movement_budget: MovementBudget,
    ) -> Result<Self, MemoryControlPolicyError> {
        self.movement_budget = movement_budget.reset_usage();
        self.validate()?;
        Ok(self)
    }

    /// Returns the DRAM residency target.
    #[must_use]
    pub const fn dram_residency_target(&self) -> UnitInterval {
        self.dram_residency_target
    }

    /// Returns the compression aggressiveness.
    #[must_use]
    pub const fn compression_aggressiveness(&self) -> UnitInterval {
        self.compression_aggressiveness
    }

    /// Returns the `NVMe` demotion pressure.
    #[must_use]
    pub const fn nvme_demotion_pressure(&self) -> UnitInterval {
        self.nvme_demotion_pressure
    }

    /// Returns the movement budget.
    #[must_use]
    pub const fn movement_budget(&self) -> MovementBudget {
        self.movement_budget
    }

    /// Returns the promotion threshold.
    #[must_use]
    pub const fn promotion_threshold(&self) -> UnitInterval {
        self.promotion_threshold
    }

    /// Returns the criticality weighting.
    #[must_use]
    pub const fn criticality_weighting(&self) -> UnitInterval {
        self.criticality_weighting
    }

    pub(crate) fn planning_pressure(&self, actual: f64) -> f64 {
        let actual = if actual.is_finite() {
            actual.clamp(0.0, 1.0)
        } else {
            1.0
        };
        let target = self.dram_residency_target.as_f64();
        let demotion = self.nvme_demotion_pressure.as_f64();
        let adjustment = f64::midpoint(actual - target, actual - demotion);
        (actual + adjustment).clamp(0.0, 1.0)
    }

    pub(crate) fn weighted_score(&self, score: f64) -> f64 {
        if score.is_finite() {
            (score.max(0.0) * self.criticality_weighting.as_f64()).max(0.0)
        } else {
            f64::INFINITY
        }
    }

    pub(crate) fn allows_demotion(&self, score: f64, actual_pressure: f64) -> bool {
        self.planning_pressure(actual_pressure) >= self.nvme_demotion_pressure.as_f64()
            || normalized_score(score) <= 0.25
    }

    pub(crate) fn allows_promotion(&self, score: f64, actual_pressure: f64) -> bool {
        normalized_score(score) >= self.promotion_threshold.as_f64()
            || self.planning_pressure(actual_pressure) >= self.dram_residency_target.as_f64()
    }

    pub(crate) fn allows_compression(&self, score: f64, actual_pressure: f64) -> bool {
        let aggressiveness = self.compression_aggressiveness.as_f64();
        aggressiveness >= 1.0
            || self.planning_pressure(actual_pressure) >= 1.0 - aggressiveness
            || (aggressiveness > 0.0 && normalized_score(score) <= 1.0 - aggressiveness)
    }
}

impl Default for MemoryControlPolicy {
    fn default() -> Self {
        Self {
            dram_residency_target: UnitInterval::new(DEFAULT_DRAM_RESIDENCY_TARGET)
                .unwrap_or(UnitInterval::ZERO),
            compression_aggressiveness: UnitInterval::new(DEFAULT_COMPRESSION_AGGRESSIVENESS)
                .unwrap_or(UnitInterval::ZERO),
            nvme_demotion_pressure: UnitInterval::new(DEFAULT_NVME_DEMOTION_PRESSURE)
                .unwrap_or(UnitInterval::ZERO),
            movement_budget: MovementBudget::production(),
            promotion_threshold: UnitInterval::new(DEFAULT_PROMOTION_THRESHOLD)
                .unwrap_or(UnitInterval::ZERO),
            criticality_weighting: UnitInterval::new(DEFAULT_CRITICALITY_WEIGHTING)
                .unwrap_or(UnitInterval::ZERO),
        }
    }
}

fn fraction(value: f64, field: &'static str) -> Result<UnitInterval, MemoryControlPolicyError> {
    UnitInterval::new(value).ok_or(MemoryControlPolicyError::InvalidFraction { field })
}

fn normalized_score(score: f64) -> f64 {
    if !score.is_finite() {
        return 1.0;
    }
    let score = score.max(0.0);
    score / (1.0 + score)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_policy_is_valid_and_bounded() {
        let policy = MemoryControlPolicy::default();
        assert!(policy.validate().is_ok());
        assert!(policy.movement_budget.max_bytes_per_tick > 0);
        assert!(policy.planning_pressure(0.0).is_finite());
        assert!(policy.planning_pressure(1.0).is_finite());
    }

    #[test]
    fn invalid_fraction_and_budget_are_rejected() {
        let budget = MovementBudget::new(1, 1, 1, 1);
        assert!(MemoryControlPolicy::new(1.1, 0.5, 0.5, budget, 0.5, 1.0).is_none());
        let mut invalid = budget;
        invalid.used_ops = 2;
        assert!(MemoryControlPolicy::new(0.5, 0.5, 0.5, invalid, 0.5, 1.0).is_none());
    }

    #[test]
    fn policy_controls_change_decisions_without_nondeterminism() {
        let budget = MovementBudget::new(10, 1, 10, 1);
        let conservative =
            MemoryControlPolicy::new(0.9, 0.0, 1.0, budget, 1.0, 0.25).expect("valid policy");
        let permissive =
            MemoryControlPolicy::new(0.1, 1.0, 0.0, budget, 0.0, 1.0).expect("valid policy");
        assert!(!conservative.allows_compression(2.0, 0.2));
        assert!(permissive.allows_compression(2.0, 0.2));
        assert!(!conservative.allows_demotion(2.0, 0.8));
        assert!(permissive.allows_demotion(2.0, 0.8));
        assert!(!conservative.allows_promotion(2.0, 0.2));
        assert!(permissive.allows_promotion(2.0, 0.2));
    }

    #[test]
    fn policy_changes_are_deterministic() {
        let budget = MovementBudget::new(10, 1, 10, 1);
        let first =
            MemoryControlPolicy::new(0.5, 0.5, 0.5, budget, 0.5, 0.5).expect("valid policy");
        let second =
            MemoryControlPolicy::new(0.5, 0.5, 0.5, budget, 0.5, 0.5).expect("valid policy");
        assert_eq!(first, second);
        assert_eq!(
            first.planning_pressure(0.4).to_bits(),
            second.planning_pressure(0.4).to_bits()
        );
    }
}
