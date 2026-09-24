//! Redundancy schemes: one abstraction for replication and erasure coding.
//!
//! [`SchemeParams`] is the planner's layout decision behind the declarative
//! [`crate::RedundancyIntent`]. Replication and Reed-Solomon share every
//! surrounding abstraction — asset identity, fragment identity, placement,
//! publication, reconstruction, repair, observability — proving the fabric
//! protects *information*, not "the erasure-code module".
//!
//! Bounds are hard and versioned: replication `1..=MAX_COPIES` copies,
//! Reed-Solomon `1..=MAX_DATA` data by `1..=MAX_PARITY` parity with
//! explicit versioned fragment sizes. Wider asks fail closed.

/// Replication parameters: `copies` physical copies on independent nodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ReplicationParams {
    /// Physical copies (`1..=MAX_COPIES`).
    pub copies: u8,
}

impl ReplicationParams {
    /// Maximum copies the baseline supports.
    pub const MAX_COPIES: u8 = 8;

    /// Validates bounds.
    ///
    /// # Errors
    ///
    /// Returns [`crate::RedundancyError::InvalidParams`] outside `1..=MAX`.
    pub const fn validate(self) -> Result<(), crate::RedundancyError> {
        if self.copies == 0 || self.copies > Self::MAX_COPIES {
            return Err(crate::RedundancyError::InvalidParams {
                detail: String::new(),
            });
        }
        Ok(())
    }
}

/// Reed-Solomon layout version 1: systematic Vandermonde over `GF(2^8)` with
/// explicit padded shard size and 64-byte alignment.
///
/// `fragment_len` is the exact padded shard bytes every fragment carries
/// (data shards hold `fragment_len` bytes with zero padding past the asset
/// end; parity shards are exactly `fragment_len`). Alignment is part of the
/// version: `V1` pads shard sizes up to a multiple of 64 so layouts stay
/// compatible across `reed-solomon-simd` versions and SIMD kernels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RsParams {
    /// Data shards (`1..=MAX_DATA`).
    pub data: u8,
    /// Parity shards (`1..=MAX_PARITY`).
    pub parity: u8,
    /// Padded shard bytes (explicit, versioned, `> 0`).
    pub fragment_len: u32,
}

impl RsParams {
    /// Layout version minted here.
    pub const VERSION: u16 = 1;
    /// Maximum data shards.
    pub const MAX_DATA: u8 = 32;
    /// Maximum parity shards.
    pub const MAX_PARITY: u8 = 8;
    /// Maximum total shards.
    pub const MAX_TOTAL: u8 = 40;
    /// Maximum padded shard bytes (8 MiB).
    pub const MAX_FRAGMENT_LEN: u32 = 8 * 1024 * 1024;
    /// Alignment of padded shard sizes (64 bytes, cross-version compat).
    pub const ALIGNMENT: u32 = 64;
    /// Maximum logical asset bytes protected by one coded layout (64 MiB,
    /// matching the legacy value ceiling; larger assets use replication or
    /// are split by the caller before protection).
    pub const MAX_ASSET_BYTES: u64 = 64 * 1024 * 1024;

    /// Pads `shard` up to the versioned alignment.
    #[must_use]
    pub const fn pad_len(shard: u32) -> u32 {
        let rem = shard % Self::ALIGNMENT;
        if rem == 0 {
            shard
        } else {
            shard + (Self::ALIGNMENT - rem)
        }
    }

    /// Derives padded shard size for `logical_len` under `data` shards.
    #[must_use]
    pub fn shard_for_len(logical_len: u64, data: u8) -> u32 {
        let data_u64 = u64::from(data.max(1));
        let raw = logical_len.div_ceil(data_u64);
        let raw32 = u32::try_from(raw).unwrap_or(Self::MAX_FRAGMENT_LEN);
        Self::pad_len(raw32.max(2))
    }

    /// Validates widths and shard size.
    ///
    /// # Errors
    ///
    /// Returns [`crate::RedundancyError::InvalidParams`] outside bounds.
    pub fn validate(self) -> Result<(), crate::RedundancyError> {
        if self.data == 0 || self.data > Self::MAX_DATA {
            return Err(crate::RedundancyError::invalid_params(format!(
                "rs data {} outside 1..={}",
                self.data,
                Self::MAX_DATA
            )));
        }
        if self.parity == 0 || self.parity > Self::MAX_PARITY {
            return Err(crate::RedundancyError::invalid_params(format!(
                "rs parity {} outside 1..={}",
                self.parity,
                Self::MAX_PARITY
            )));
        }
        if self.data.saturating_add(self.parity) > Self::MAX_TOTAL {
            return Err(crate::RedundancyError::invalid_params(format!(
                "rs total {} exceeds {}",
                self.data.saturating_add(self.parity),
                Self::MAX_TOTAL
            )));
        }
        if self.fragment_len == 0 || self.fragment_len > Self::MAX_FRAGMENT_LEN {
            return Err(crate::RedundancyError::invalid_params(format!(
                "rs fragment_len {} outside 1..={}",
                self.fragment_len,
                Self::MAX_FRAGMENT_LEN
            )));
        }
        if !self.fragment_len.is_multiple_of(Self::ALIGNMENT) {
            return Err(crate::RedundancyError::invalid_params(format!(
                "rs fragment_len {} is not {}-aligned",
                self.fragment_len,
                Self::ALIGNMENT
            )));
        }
        Ok(())
    }

    /// Total fragments (`data + parity`).
    #[must_use]
    pub fn total(self) -> u8 {
        self.data.saturating_add(self.parity)
    }

    /// Failures survivable (`parity`).
    #[must_use]
    pub fn tolerance(self) -> u8 {
        self.parity
    }
}

/// Which family a layout belongs to (durable discriminant).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SchemeId {
    /// Ordinary replication.
    Replication,
    /// Reed-Solomon-style erasure coding (`RS_V1`).
    ReedSolomon,
}

impl SchemeId {
    /// Wraps a raw discriminant.
    ///
    /// # Errors
    ///
    /// Returns [`crate::RedundancyError::BadLayout`] for unknown values.
    pub const fn from_u8(value: u8) -> Result<Self, crate::RedundancyError> {
        match value {
            0 => Ok(Self::Replication),
            1 => Ok(Self::ReedSolomon),
            _ => Err(crate::RedundancyError::BadLayout {
                detail: String::new(),
            }),
        }
    }

    /// Returns the frozen discriminant.
    #[must_use]
    pub const fn as_u8(self) -> u8 {
        match self {
            Self::Replication => 0,
            Self::ReedSolomon => 1,
        }
    }

    /// Human-readable name.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Replication => "replication",
            Self::ReedSolomon => "reed-solomon",
        }
    }
}

/// One planner layout decision: scheme plus bounded parameters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SchemeParams {
    /// `N` physical copies.
    Replication(ReplicationParams),
    /// Systematic `RS(data, parity)` with explicit shard size.
    ReedSolomon(RsParams),
}

impl SchemeParams {
    /// Returns the scheme family.
    #[must_use]
    pub fn scheme(self) -> SchemeId {
        match self {
            Self::Replication(_) => SchemeId::Replication,
            Self::ReedSolomon(_) => SchemeId::ReedSolomon,
        }
    }

    /// Validates bounded limits.
    ///
    /// # Errors
    ///
    /// Returns [`crate::RedundancyError::InvalidParams`] outside bounds.
    pub fn validate(self) -> Result<(), crate::RedundancyError> {
        match self {
            Self::Replication(params) => {
                if params.copies == 0 || params.copies > ReplicationParams::MAX_COPIES {
                    return Err(crate::RedundancyError::invalid_params(format!(
                        "replication copies {} outside 1..={}",
                        params.copies,
                        ReplicationParams::MAX_COPIES
                    )));
                }
                Ok(())
            }
            Self::ReedSolomon(params) => params.validate(),
        }
    }

    /// Total physical fragments.
    #[must_use]
    pub fn total_fragments(self) -> u32 {
        match self {
            Self::Replication(params) => u32::from(params.copies),
            Self::ReedSolomon(params) => u32::from(params.data) + u32::from(params.parity),
        }
    }

    /// Failures survivable under ideal placement.
    #[must_use]
    pub fn tolerance(self) -> u32 {
        match self {
            Self::Replication(params) => u32::from(params.copies.saturating_sub(1)),
            Self::ReedSolomon(params) => u32::from(params.parity),
        }
    }

    /// Fragments required to reconstruct.
    #[must_use]
    pub fn required_pieces(self) -> u32 {
        match self {
            Self::Replication(_) => 1,
            Self::ReedSolomon(params) => u32::from(params.data),
        }
    }

    /// Storage amplification (`physical / logical`) as numerator/denominator.
    #[must_use]
    pub fn amplification(self) -> (u64, u64) {
        match self {
            Self::Replication(params) => (u64::from(params.copies), 1),
            Self::ReedSolomon(params) => (
                u64::from(params.data) + u64::from(params.parity),
                u64::from(params.data),
            ),
        }
    }
}

/// Provider capabilities (what a scheme offers, not its parameters).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SchemeCapabilities {
    /// Scheme family.
    pub scheme: SchemeId,
    /// Maximum-data-symbol code (MDS: any `required` of `total` suffice).
    pub mds: bool,
    /// Ordinary reads can use one piece without decoding.
    pub systematic: bool,
    /// Exact `k`-of-`n` reconstruction threshold.
    pub exact_threshold: bool,
}

impl SchemeCapabilities {
    /// Capabilities of replication.
    #[must_use]
    pub const fn replication() -> Self {
        Self {
            scheme: SchemeId::Replication,
            mds: true,
            systematic: true,
            exact_threshold: true,
        }
    }

    /// Capabilities of the Reed-Solomon baseline.
    #[must_use]
    pub const fn reed_solomon() -> Self {
        Self {
            scheme: SchemeId::ReedSolomon,
            mds: true,
            systematic: true,
            exact_threshold: true,
        }
    }
}

/// Fits an intent-derived layout to the cluster's independent failure
/// domains.
///
/// The MDS condition bounds a layout by the domains that actually exist:
/// with `D` domains, at most `m` fragments may share one domain, so the data
/// width can never exceed `m * (D - 1)` (for `RS(8, 1)` on three domains
/// that is `2`, i.e. `RS(2, 1)`), and replication can never hold more copies
/// than there are domains. Baseline intents describe a *goal* ("survive one
/// failure"), not a fixed width, so the planner sizes them to the cluster;
/// explicit widths (transitions) are never silently rewritten and still fail
/// closed when unsatisfiable.
#[must_use]
pub fn fit_to_domains(params: SchemeParams, logical_len: u64, domains: usize) -> SchemeParams {
    let domains = domains.max(1);
    match params {
        SchemeParams::Replication(rule) => {
            #[allow(clippy::cast_possible_truncation)]
            let copies = usize::from(rule.copies).min(domains).max(1);
            #[allow(clippy::cast_possible_truncation)]
            SchemeParams::Replication(ReplicationParams {
                copies: copies as u8,
            })
        }
        SchemeParams::ReedSolomon(rule) => {
            let capacity = usize::from(rule.parity).saturating_mul(domains.saturating_sub(1));
            #[allow(clippy::cast_possible_truncation)]
            let data = usize::from(rule.data).min(capacity).max(1);
            #[allow(clippy::cast_possible_truncation)]
            let data = data.min(usize::from(RsParams::MAX_DATA)) as u8;
            if data == rule.data {
                return SchemeParams::ReedSolomon(rule);
            }
            SchemeParams::ReedSolomon(RsParams {
                data,
                parity: rule.parity,
                fragment_len: RsParams::shard_for_len(logical_len, data),
            })
        }
    }
}

/// Chooses a baseline layout for an intent and asset size.
///
/// Deterministic and simple: small assets or hot direct-read intents use
/// replication (`tolerance + 1` copies); larger cold assets use `RS` with a
/// narrow width that meets the tolerance (`data = 8` baseline, wider when
/// storage-cost minimization demands it). Specific widths are planner
/// decisions — the intent never names them.
///
/// # Errors
///
/// Returns [`crate::RedundancyError`] on invalid intents or oversize assets.
pub fn plan_baseline(
    intent: &crate::RedundancyIntent,
    logical_len: u64,
) -> Result<SchemeParams, crate::RedundancyError> {
    // Replication threshold: assets under 64 KiB, hot direct-read intents,
    // or tolerance 1 with balanced cost stay replicated (cheap, fast reads).
    const REPLICATION_THRESHOLD: u64 = 64 * 1024;
    intent.validate()?;
    if intent.tolerance == 0 {
        return Ok(SchemeParams::Replication(ReplicationParams { copies: 1 }));
    }
    if logical_len <= REPLICATION_THRESHOLD || intent.repair.prefer_direct_reads {
        let copies = intent
            .tolerance
            .saturating_add(1)
            .min(ReplicationParams::MAX_COPIES);
        return Ok(SchemeParams::Replication(ReplicationParams { copies }));
    }
    // Erasure-coded baseline: data width 8 (or fewer for tiny assets so
    // shards stay meaningful), parity = tolerance.
    if u64::from(intent.tolerance) > u64::from(RsParams::MAX_PARITY) {
        return Err(crate::RedundancyError::invalid_params(format!(
            "tolerance {} exceeds RS parity cap {}",
            intent.tolerance,
            RsParams::MAX_PARITY
        )));
    }
    if logical_len > RsParams::MAX_ASSET_BYTES {
        // Oversize assets stay replicated rather than blowing memory bounds.
        let copies = intent
            .tolerance
            .saturating_add(1)
            .min(ReplicationParams::MAX_COPIES);
        return Ok(SchemeParams::Replication(ReplicationParams { copies }));
    }
    let data: u8 = if logical_len <= 256 * 1024 {
        4
    } else if intent.storage_cost == crate::StorageCost::Minimize {
        16
    } else {
        8
    };
    let fragment_len = RsParams::shard_for_len(logical_len, data);
    let params = RsParams {
        data,
        parity: intent.tolerance,
        fragment_len,
    };
    params.validate()?;
    Ok(SchemeParams::ReedSolomon(params))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{RedundancyIntent, StorageCost};

    #[test]
    fn baseline_prefers_replication_for_small_or_hot() {
        let mut intent = RedundancyIntent::survive_two();
        let small = plan_baseline(&intent, 1024).expect("plans");
        assert!(matches!(small, SchemeParams::Replication(_)));
        intent.repair.prefer_direct_reads = true;
        let hot = plan_baseline(&intent, 4 * 1024 * 1024).expect("plans");
        assert!(matches!(hot, SchemeParams::Replication(_)));
    }

    #[test]
    fn baseline_codes_large_cold_assets() {
        let intent = RedundancyIntent::survive_two();
        let coded = plan_baseline(&intent, 4 * 1024 * 1024).expect("plans");
        match coded {
            SchemeParams::ReedSolomon(params) => {
                assert_eq!(params.parity, 2);
                assert_eq!(params.fragment_len % RsParams::ALIGNMENT, 0);
            }
            SchemeParams::Replication(_) => panic!("expected coding"),
        }
        let mut lean = RedundancyIntent::survive_two();
        lean.storage_cost = StorageCost::Minimize;
        match plan_baseline(&lean, 4 * 1024 * 1024).expect("plans") {
            SchemeParams::ReedSolomon(params) => assert_eq!(params.data, 16),
            SchemeParams::Replication(_) => panic!("expected coding"),
        }
    }

    #[test]
    fn fitting_narrows_widths_to_the_available_domains() {
        let wide = SchemeParams::ReedSolomon(RsParams {
            data: 8,
            parity: 1,
            fragment_len: RsParams::shard_for_len(512 * 1024, 8),
        });
        match fit_to_domains(wide, 512 * 1024, 3) {
            SchemeParams::ReedSolomon(params) => {
                assert_eq!(params.data, 2);
                assert_eq!(params.parity, 1);
                assert_eq!(params.fragment_len, RsParams::shard_for_len(512 * 1024, 2));
            }
            SchemeParams::Replication(_) => panic!("expected coding"),
        }
        match fit_to_domains(wide, 512 * 1024, 12) {
            SchemeParams::ReedSolomon(params) => assert_eq!(params.data, 8),
            SchemeParams::Replication(_) => panic!("expected coding"),
        }
        let copies = SchemeParams::Replication(ReplicationParams { copies: 3 });
        match fit_to_domains(copies, 1024, 2) {
            SchemeParams::Replication(params) => assert_eq!(params.copies, 2),
            SchemeParams::ReedSolomon(_) => panic!("expected replication"),
        }
    }

    #[test]
    fn bounds_fail_closed() {
        assert!(
            RsParams {
                data: 0,
                parity: 1,
                fragment_len: 64
            }
            .validate()
            .is_err()
        );
        assert!(
            RsParams {
                data: 33,
                parity: 1,
                fragment_len: 64
            }
            .validate()
            .is_err()
        );
        assert!(
            RsParams {
                data: 8,
                parity: 9,
                fragment_len: 64
            }
            .validate()
            .is_err()
        );
        assert!(
            RsParams {
                data: 8,
                parity: 2,
                fragment_len: 100
            }
            .validate()
            .is_err()
        );
    }
}
