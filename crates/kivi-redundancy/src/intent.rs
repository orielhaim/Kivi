//! Declarative redundancy requirements: what protection, not how.
//!
//! [`RedundancyIntent`] expresses properties — required independent failure
//! tolerance, minimum available pieces, locality, allowed domains,
//! storage-cost preference, repair-cost constraints — while specific layouts
//! (`RS(8,3)`, replica counts, shard widths) remain planner decisions. The
//! initial planner is simple and deterministic, but the API leaves room for
//! later adaptive control (opaque policy version + hints travel with the
//! intent without changing its safety semantics).

/// How much extra storage the planner may spend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum StorageCost {
    /// Minimize physical bytes (prefer wide erasure coding).
    Minimize,
    /// Balanced default.
    Balanced,
    /// Spend storage to minimize repair work and read latency.
    PreferSpeed,
}

impl StorageCost {
    /// Returns the default preference.
    #[must_use]
    pub const fn default() -> Self {
        Self::Balanced
    }
}

/// Repair-cost/latency constraints attached to an intent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RepairConstraints {
    /// Maximum bytes the planner may read to rebuild one fragment.
    pub max_repair_read_bytes: u64,
    /// Whether degraded reads must avoid decoding on the hot path when a
    /// full copy exists elsewhere (prefers replication for hot assets).
    pub prefer_direct_reads: bool,
}

impl RepairConstraints {
    /// Balanced defaults: 8 MiB per-fragment repair reads, no hot-path bias.
    #[must_use]
    pub const fn balanced() -> Self {
        Self {
            max_repair_read_bytes: 8 * 1024 * 1024,
            prefer_direct_reads: false,
        }
    }

    /// Hot-asset defaults: prefer layouts with direct reads.
    #[must_use]
    pub const fn hot() -> Self {
        Self {
            max_repair_read_bytes: 8 * 1024 * 1024,
            prefer_direct_reads: true,
        }
    }
}

/// Locality requirements for fragment placement.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LocalityRequirement {
    /// No locality constraint beyond failure independence.
    None,
    /// Fragments must span racks when the topology names them.
    RackSpanning,
    /// Fragments must span zones when the topology names them.
    ZoneSpanning,
}

impl LocalityRequirement {
    /// Returns the default (no extra locality).
    #[must_use]
    pub const fn default() -> Self {
        Self::None
    }
}

/// Declarative protection requirement for one asset (or asset class).
///
/// The contract is semantic: *survive `tolerance` independent failures with
/// at least `min_available` pieces retrievable*. The planner maps that onto
/// replication or erasure coding; callers never name `RS(8,3)`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RedundancyIntent {
    /// Required independent failure tolerance (0 = best-effort single copy).
    pub tolerance: u8,
    /// Minimum pieces that must stay retrievable (defaults to 1).
    pub min_available: u32,
    /// Locality requirements beyond independence.
    pub locality: LocalityRequirement,
    /// Allowed failure-domain labels; empty means any domain qualifies.
    pub allowed_domains: Vec<String>,
    /// Storage-cost preference (planner hint, never safety).
    pub storage_cost: StorageCost,
    /// Repair-cost/latency constraints.
    pub repair: RepairConstraints,
    /// Opaque policy version for future adaptive control. `0` is the
    /// deterministic baseline planner; later controllers increment it
    /// without changing the safety fields above.
    pub policy_version: u32,
}

impl RedundancyIntent {
    /// Maximum tolerance the baseline planner accepts (bounded repair).
    pub const MAX_TOLERANCE: u8 = 8;

    /// Builds the weakest intent: one copy, no tolerance.
    #[must_use]
    pub const fn best_effort() -> Self {
        Self {
            tolerance: 0,
            min_available: 1,
            locality: LocalityRequirement::None,
            allowed_domains: Vec::new(),
            storage_cost: StorageCost::Balanced,
            repair: RepairConstraints {
                max_repair_read_bytes: 8 * 1024 * 1024,
                prefer_direct_reads: false,
            },
            policy_version: 0,
        }
    }

    /// Survive one independent failure (minimum durable default).
    #[must_use]
    pub fn survive_one() -> Self {
        Self {
            tolerance: 1,
            ..Self::best_effort()
        }
    }

    /// Survive two independent failures (durable default for critical data).
    #[must_use]
    pub fn survive_two() -> Self {
        Self {
            tolerance: 2,
            ..Self::best_effort()
        }
    }

    /// Validates hard safety invariants.
    ///
    /// # Errors
    ///
    /// Returns [`crate::RedundancyError::InvalidIntent`] when tolerance
    /// exceeds the bounded maximum or `min_available` is zero.
    pub fn validate(&self) -> Result<(), crate::RedundancyError> {
        if self.min_available == 0 {
            return Err(crate::RedundancyError::invalid_intent(
                "min_available must be nonzero",
            ));
        }
        if self.tolerance > Self::MAX_TOLERANCE {
            return Err(crate::RedundancyError::invalid_intent(format!(
                "tolerance {} exceeds maximum {}",
                self.tolerance,
                Self::MAX_TOLERANCE
            )));
        }
        Ok(())
    }

    /// Whether this intent requires more than one independent piece.
    #[must_use]
    pub fn requires_redundancy(&self) -> bool {
        self.tolerance > 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn intents_validate_bounds() {
        RedundancyIntent::survive_two().validate().expect("valid");
        let mut bad = RedundancyIntent::survive_two();
        bad.min_available = 0;
        assert!(bad.validate().is_err());
        let mut wild = RedundancyIntent::survive_two();
        wild.tolerance = 9;
        assert!(wild.validate().is_err());
    }

    #[test]
    fn semantic_contract_hides_layout() {
        let intent = RedundancyIntent::survive_two();
        // The intent names tolerance, never "RS(8,3)": debug form carries
        // no codec widths.
        let text = format!("{intent:?}");
        assert!(text.contains("tolerance"));
        assert!(!text.contains("RS("));
    }
}
