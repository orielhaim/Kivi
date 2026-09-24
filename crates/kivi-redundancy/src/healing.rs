//! Phase 11 self-healing controller model.
//!
//! These values are pure inputs and outputs for a distributed controller. They
//! do not read clocks, choose nodes, perform I/O, or retry work. Every ranking,
//! aggregation, and scheduling transition is deterministic so controllers can
//! reproduce a decision after a restart.

use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::time::Duration;

use kivi_types::{NodeId, Ticks};

use crate::{AssetId, SchemeId, SchemeParams};

const REPAIR_PRIORITY_UNRECOVERABLE: u64 = 1_000_000_000_000;
const REPAIR_PRIORITY_NO_TOLERANCE: u64 = 800_000_000_000;
const REPAIR_PRIORITY_SCHEME_VIOLATION: u64 = 600_000_000_000;
const REPAIR_PRIORITY_FRAGMENT_DEFICIT: u64 = 1_000_000_000;
const REPAIR_PRIORITY_TOLERANCE_DEFICIT: u64 = 500_000_000;
const REPAIR_PRIORITY_UNAVAILABLE_NODE: u64 = 10_000_000;
const REPAIR_PRIORITY_DRAINING_NODE: u64 = 2_000_000;
const REPAIR_PRIORITY_CORRUPT_FRAGMENT: u64 = 20_000_000;
const REPAIR_PRIORITY_UNAVAILABLE_FRAGMENT: u64 = 5_000_000;
const REPAIR_PRIORITY_CONCENTRATION: u64 = 1_000_000;
const REPAIR_PRIORITY_PLACEMENT: u64 = 2_000_000;
const REPAIR_PRIORITY_LAYOUT_DRIFT: u64 = 250_000;
const REPAIR_PRIORITY_AGE_CAP_MICROS: u64 = 86_400_000_000;
const REPAIR_PRIORITY_DEGRADATION_MICROS: u64 = 10_000;
const REPAIR_PRIORITY_VERIFICATION_MICROS: u64 = 2_000;

/// Amount of fragment data inspected by a scrub pass.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ScrubMode {
    /// Inspect layout metadata and fragment presence only.
    #[default]
    Metadata,
    /// Read and verify every fragment covered by the pass.
    Full,
}

impl ScrubMode {
    /// Whether the mode reads fragment payloads.
    #[must_use]
    pub const fn reads_payloads(self) -> bool {
        matches!(self, Self::Full)
    }

    /// Stable operator-facing name.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Metadata => "metadata",
            Self::Full => "full",
        }
    }
}

/// Scheme-derived requirements used when evaluating an asset.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SchemeRequirements {
    /// Scheme family used by the published layout.
    pub scheme: SchemeId,
    /// Total physical fragments expected by the scheme.
    pub total_fragments: u32,
    /// Fragments needed to reconstruct the asset.
    pub required_fragments: u32,
    /// Independent failures the scheme is intended to survive.
    pub independent_tolerance: u32,
}

impl SchemeRequirements {
    /// Builds requirements directly from bounded scheme parameters.
    #[must_use]
    pub fn from_params(params: SchemeParams) -> Self {
        Self {
            scheme: params.scheme(),
            total_fragments: params.total_fragments(),
            required_fragments: params.required_pieces(),
            independent_tolerance: params.tolerance(),
        }
    }

    /// Builds requirements without decoding a [`SchemeParams`] value.
    #[must_use]
    pub const fn new(
        scheme: SchemeId,
        total_fragments: u32,
        required_fragments: u32,
        independent_tolerance: u32,
    ) -> Self {
        Self {
            scheme,
            total_fragments,
            required_fragments,
            independent_tolerance,
        }
    }

    /// Whether the observed fragment and independence counts satisfy the
    /// scheme contract.
    #[must_use]
    pub const fn is_satisfied_by(
        self,
        usable_fragments: u32,
        independent_failure_domains: u32,
    ) -> bool {
        usable_fragments >= self.required_fragments
            && independent_failure_domains.saturating_sub(self.required_fragments)
                >= self.independent_tolerance
    }
}

/// Explicit limits shared by scrub, repair, reconstruction, and transfer
/// work.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MaintenanceBudgets {
    /// Maximum assets inspected by one scrub batch.
    pub max_scrub_assets: usize,
    /// Maximum bytes inspected by one scrub batch.
    pub max_scrub_bytes: u64,
    /// Maximum assets repaired by one maintenance batch.
    pub max_repair_assets: usize,
    /// Maximum bytes reconstructed or transferred by one repair batch.
    pub max_repair_bytes: u64,
    /// Maximum simultaneous reconstruction operations.
    pub max_concurrent_reconstructions: usize,
    /// Maximum simultaneous fragment transfers.
    pub max_concurrent_transfers: usize,
    /// Maximum simultaneous scrub operations.
    pub max_concurrent_scrubs: usize,
    /// Maximum queued maintenance units across all classes.
    pub max_queued_work: usize,
    /// Reconstruction slots held for critical repairs.
    pub critical_reserve: usize,
}

impl MaintenanceBudgets {
    /// Conservative defaults for a foreground-sensitive production cluster.
    #[must_use]
    pub const fn conservative() -> Self {
        Self {
            max_scrub_assets: 16,
            max_scrub_bytes: 64 * 1024 * 1024,
            max_repair_assets: 8,
            max_repair_bytes: 128 * 1024 * 1024,
            max_concurrent_reconstructions: 2,
            max_concurrent_transfers: 4,
            max_concurrent_scrubs: 1,
            max_queued_work: 1024,
            critical_reserve: 1,
        }
    }

    /// Large finite limits for deterministic tests and benchmarks.
    #[must_use]
    pub const fn test_wide() -> Self {
        Self {
            max_scrub_assets: 4096,
            max_scrub_bytes: 1 << 40,
            max_repair_assets: 4096,
            max_repair_bytes: 1 << 40,
            max_concurrent_reconstructions: 64,
            max_concurrent_transfers: 64,
            max_concurrent_scrubs: 16,
            max_queued_work: 65_536,
            critical_reserve: 8,
        }
    }

    /// Checks that work can make progress and the critical reserve fits the
    /// reconstruction limit.
    #[must_use]
    pub const fn validate(self) -> bool {
        self.max_scrub_assets > 0
            && self.max_scrub_bytes > 0
            && self.max_repair_assets > 0
            && self.max_repair_bytes > 0
            && self.max_concurrent_reconstructions > 0
            && self.max_concurrent_transfers > 0
            && self.max_concurrent_scrubs > 0
            && self.max_queued_work > 0
            && self.critical_reserve <= self.max_concurrent_reconstructions
    }
}

impl Default for MaintenanceBudgets {
    fn default() -> Self {
        Self::conservative()
    }
}

/// Risk facts for one asset and its installed fragments.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RepairRisk {
    /// Independent failures still survivable after observed placement.
    pub remaining_independent_tolerance: u32,
    /// Fragments currently readable and verified.
    pub usable_fragments: u32,
    /// Fragments required by the installed scheme for reconstruction.
    pub required_fragments: u32,
    /// Total fragments in the installed layout.
    pub total_fragments: u32,
    /// Scheme contract that the risk is evaluated against.
    pub scheme_requirements: SchemeRequirements,
    /// Largest number of usable fragments sharing one failure domain.
    pub failure_domain_concentration: u32,
    /// Number of distinct failure domains represented by usable fragments.
    pub independent_failure_domains: u32,
    /// Nodes currently unavailable to the controller.
    pub unavailable_nodes: Vec<NodeId>,
    /// Nodes being drained by the controller.
    pub draining_nodes: Vec<NodeId>,
    /// Fragments known to contain corrupt bytes.
    pub corrupt_fragments: u32,
    /// Fragments unavailable because a holder or path is unavailable.
    pub unavailable_fragments: u32,
    /// Logical bytes represented by the asset.
    pub asset_bytes: u64,
    /// Bytes that a full reconstruction would read or write.
    pub reconstruction_bytes: u64,
    /// Time since the asset first became degraded.
    pub degradation_age: Duration,
    /// Time since the last successful verification.
    pub verification_age: Duration,
    /// Number of desired-versus-installed placement violations.
    pub placement_violations: u32,
}

impl RepairRisk {
    /// Scheme family used by the asset.
    #[must_use]
    pub const fn scheme(&self) -> SchemeId {
        self.scheme_requirements.scheme
    }

    /// Whether enough verified fragments exist for reconstruction now.
    #[must_use]
    pub const fn is_reconstructable(&self) -> bool {
        self.usable_fragments >= self.required_fragments
    }

    /// Whether the current verified set cannot reconstruct the asset.
    #[must_use]
    pub const fn is_unrecoverable(&self) -> bool {
        !self.is_reconstructable()
    }

    /// Whether both reconstruction and the scheme's independent contract hold.
    #[must_use]
    pub const fn meets_scheme_requirements(&self) -> bool {
        self.scheme_requirements
            .is_satisfied_by(self.usable_fragments, self.independent_failure_domains)
            && self.remaining_independent_tolerance
                >= self.scheme_requirements.independent_tolerance
    }

    /// Whether repair should be treated as critical work.
    #[must_use]
    pub const fn is_critical(&self) -> bool {
        !self.is_reconstructable() || self.remaining_independent_tolerance == 0
    }

    /// Calculates a deterministic, saturating maintenance priority.
    ///
    /// Unrecoverable risk dominates, followed by exhausted tolerance and
    /// scheme violations. Fragment deficits, corruption, unavailable nodes,
    /// placement drift, age, and byte volume then add bounded, integral
    /// weights. The calculation uses no floating point and does not depend on
    /// node-list order.
    #[must_use]
    pub fn priority(&self) -> u64 {
        let mut score: u64 = if self.is_unrecoverable() {
            REPAIR_PRIORITY_UNRECOVERABLE
        } else if self.remaining_independent_tolerance == 0 {
            REPAIR_PRIORITY_NO_TOLERANCE
        } else if !self.meets_scheme_requirements() {
            REPAIR_PRIORITY_SCHEME_VIOLATION
        } else {
            0
        };

        let fragment_deficit = self
            .required_fragments
            .saturating_sub(self.usable_fragments);
        score = score.saturating_add(
            u64::from(fragment_deficit).saturating_mul(REPAIR_PRIORITY_FRAGMENT_DEFICIT),
        );

        let tolerance_deficit = self
            .scheme_requirements
            .independent_tolerance
            .saturating_sub(self.remaining_independent_tolerance);
        score = score.saturating_add(
            u64::from(tolerance_deficit).saturating_mul(REPAIR_PRIORITY_TOLERANCE_DEFICIT),
        );

        let unavailable_nodes = count_nodes(&self.unavailable_nodes);
        let draining_nodes = count_nodes(&self.draining_nodes);
        score = score
            .saturating_add(unavailable_nodes.saturating_mul(REPAIR_PRIORITY_UNAVAILABLE_NODE));
        score = score.saturating_add(draining_nodes.saturating_mul(REPAIR_PRIORITY_DRAINING_NODE));
        score = score.saturating_add(
            u64::from(self.corrupt_fragments).saturating_mul(REPAIR_PRIORITY_CORRUPT_FRAGMENT),
        );
        score = score.saturating_add(
            u64::from(self.unavailable_fragments)
                .saturating_mul(REPAIR_PRIORITY_UNAVAILABLE_FRAGMENT),
        );

        let concentration_excess = self
            .failure_domain_concentration
            .saturating_sub(self.independent_failure_domains.max(1));
        score = score.saturating_add(
            u64::from(concentration_excess).saturating_mul(REPAIR_PRIORITY_CONCENTRATION),
        );
        score = score.saturating_add(
            u64::from(self.placement_violations).saturating_mul(REPAIR_PRIORITY_PLACEMENT),
        );
        let layout_drift = self
            .total_fragments
            .abs_diff(self.scheme_requirements.total_fragments);
        score = score
            .saturating_add(u64::from(layout_drift).saturating_mul(REPAIR_PRIORITY_LAYOUT_DRIFT));

        let degradation_age =
            duration_micros(self.degradation_age).min(REPAIR_PRIORITY_AGE_CAP_MICROS);
        let verification_age =
            duration_micros(self.verification_age).min(REPAIR_PRIORITY_AGE_CAP_MICROS);
        score = score.saturating_add(
            (degradation_age / 1_000).saturating_mul(REPAIR_PRIORITY_DEGRADATION_MICROS),
        );
        score = score.saturating_add(
            (verification_age / 1_000).saturating_mul(REPAIR_PRIORITY_VERIFICATION_MICROS),
        );
        score = score.saturating_add(byte_priority(self.asset_bytes));
        score = score.saturating_add(byte_priority(self.reconstruction_bytes));
        score
    }

    /// Alias for [`priority`](Self::priority) for controller integrations that
    /// name the value explicitly as a score.
    #[must_use]
    pub fn priority_score(&self) -> u64 {
        self.priority()
    }
}

fn count_nodes(nodes: &[NodeId]) -> u64 {
    u64::try_from(nodes.len()).unwrap_or(u64::MAX)
}

fn duration_micros(value: Duration) -> u64 {
    u64::try_from(value.as_micros()).unwrap_or(u64::MAX)
}

fn byte_priority(bytes: u64) -> u64 {
    const MIB: u64 = 1024 * 1024;
    const SIXTY_FOUR_MIB: u64 = 64 * MIB;
    const GIB: u64 = 1024 * MIB;
    const SECOND_START: u64 = MIB + 1;
    const THIRD_START: u64 = SIXTY_FOUR_MIB + 1;
    match bytes {
        0 => 0,
        1..=MIB => 1_000,
        SECOND_START..=SIXTY_FOUR_MIB => 4_000,
        THIRD_START..=GIB => 8_000,
        _ => 16_000,
    }
}

/// Additive debt counters used by each aggregate dimension.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DebtCounters {
    /// Number of debt units represented by the counters.
    pub assets: u64,
    /// Logical or repair bytes represented by the counters.
    pub bytes: u64,
    /// Fragments represented by the counters.
    pub fragments: u64,
    /// Debt units classified as critical.
    pub critical_assets: u64,
}

impl DebtCounters {
    /// Builds a counter set.
    #[must_use]
    pub const fn new(assets: u64, bytes: u64, fragments: u64, critical_assets: u64) -> Self {
        Self {
            assets,
            bytes,
            fragments,
            critical_assets,
        }
    }

    /// Adds another counter set without wrapping.
    pub fn add(&mut self, other: Self) {
        self.assets = self.assets.saturating_add(other.assets);
        self.bytes = self.bytes.saturating_add(other.bytes);
        self.fragments = self.fragments.saturating_add(other.fragments);
        self.critical_assets = self.critical_assets.saturating_add(other.critical_assets);
    }

    /// Subtracts another counter set with saturation at zero.
    pub fn subtract(&mut self, other: Self) {
        self.assets = self.assets.saturating_sub(other.assets);
        self.bytes = self.bytes.saturating_sub(other.bytes);
        self.fragments = self.fragments.saturating_sub(other.fragments);
        self.critical_assets = self.critical_assets.saturating_sub(other.critical_assets);
    }

    /// Whether every counter is zero.
    #[must_use]
    pub const fn is_zero(self) -> bool {
        self.assets == 0 && self.bytes == 0 && self.fragments == 0 && self.critical_assets == 0
    }
}

/// Deterministic ordering key for a scheme debt breakdown.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SchemeKey(SchemeId);

impl SchemeKey {
    /// Wraps a scheme identifier for use as a deterministic map key.
    #[must_use]
    pub const fn new(scheme: SchemeId) -> Self {
        Self(scheme)
    }

    /// Returns the wrapped scheme identifier.
    #[must_use]
    pub const fn scheme(self) -> SchemeId {
        self.0
    }
}

impl Ord for SchemeKey {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.as_u8().cmp(&other.0.as_u8())
    }
}

impl PartialOrd for SchemeKey {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// One additive contribution to repair debt and its placement dimensions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RepairDebtEntry {
    /// Holder whose debt is represented.
    pub node: NodeId,
    /// Independence key used for the holder's failure domain.
    pub failure_domain: String,
    /// Scheme used by the affected asset.
    pub scheme: SchemeId,
    /// Counters contributed to every aggregate.
    pub counters: DebtCounters,
}

impl RepairDebtEntry {
    /// Builds a debt contribution.
    #[must_use]
    pub fn new(
        node: NodeId,
        failure_domain: String,
        scheme: SchemeId,
        counters: DebtCounters,
    ) -> Self {
        Self {
            node,
            failure_domain,
            scheme,
            counters,
        }
    }
}

/// Cluster-wide repair debt with deterministic per-dimension views.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RepairDebt {
    /// Total debt units in the cluster aggregate.
    pub total_assets: u64,
    /// Total repair bytes in the cluster aggregate.
    pub total_bytes: u64,
    /// Total affected fragments in the cluster aggregate.
    pub total_fragments: u64,
    /// Total critical debt units in the cluster aggregate.
    pub total_critical_assets: u64,
    /// Debt grouped by current holder.
    pub by_node: BTreeMap<NodeId, DebtCounters>,
    /// Debt grouped by independence key.
    pub by_failure_domain: BTreeMap<String, DebtCounters>,
    /// Debt grouped by redundancy scheme.
    pub by_scheme: BTreeMap<SchemeKey, DebtCounters>,
}

impl RepairDebt {
    /// Adds one contribution to all four aggregates.
    pub fn add(&mut self, entry: &RepairDebtEntry) {
        self.total_assets = self.total_assets.saturating_add(entry.counters.assets);
        self.total_bytes = self.total_bytes.saturating_add(entry.counters.bytes);
        self.total_fragments = self
            .total_fragments
            .saturating_add(entry.counters.fragments);
        self.total_critical_assets = self
            .total_critical_assets
            .saturating_add(entry.counters.critical_assets);
        add_map(&mut self.by_node, entry.node, entry.counters);
        add_map(
            &mut self.by_failure_domain,
            entry.failure_domain.clone(),
            entry.counters,
        );
        add_map(
            &mut self.by_scheme,
            SchemeKey::new(entry.scheme),
            entry.counters,
        );
    }

    /// Removes one contribution with saturation at zero.
    pub fn remove(&mut self, entry: &RepairDebtEntry) {
        self.total_assets = self.total_assets.saturating_sub(entry.counters.assets);
        self.total_bytes = self.total_bytes.saturating_sub(entry.counters.bytes);
        self.total_fragments = self
            .total_fragments
            .saturating_sub(entry.counters.fragments);
        self.total_critical_assets = self
            .total_critical_assets
            .saturating_sub(entry.counters.critical_assets);
        subtract_map(&mut self.by_node, &entry.node, entry.counters);
        subtract_map(
            &mut self.by_failure_domain,
            &entry.failure_domain,
            entry.counters,
        );
        subtract_map(
            &mut self.by_scheme,
            &SchemeKey::new(entry.scheme),
            entry.counters,
        );
    }

    /// Merges two complete debt aggregates deterministically.
    #[must_use]
    pub fn merge(mut self, other: Self) -> Self {
        self.total_assets = self.total_assets.saturating_add(other.total_assets);
        self.total_bytes = self.total_bytes.saturating_add(other.total_bytes);
        self.total_fragments = self.total_fragments.saturating_add(other.total_fragments);
        self.total_critical_assets = self
            .total_critical_assets
            .saturating_add(other.total_critical_assets);
        for (key, counters) in other.by_node {
            add_map(&mut self.by_node, key, counters);
        }
        for (key, counters) in other.by_failure_domain {
            add_map(&mut self.by_failure_domain, key, counters);
        }
        for (key, counters) in other.by_scheme {
            add_map(&mut self.by_scheme, key, counters);
        }
        self
    }

    /// Whether the aggregate and all breakdowns are empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.total_assets == 0
            && self.total_bytes == 0
            && self.total_fragments == 0
            && self.total_critical_assets == 0
            && self.by_node.is_empty()
            && self.by_failure_domain.is_empty()
            && self.by_scheme.is_empty()
    }

    /// Returns the debt attached to one node, or zero when absent.
    #[must_use]
    pub fn for_node(&self, node: NodeId) -> DebtCounters {
        self.by_node.get(&node).copied().unwrap_or_default()
    }

    /// Returns the debt attached to one failure-domain key, or zero.
    #[must_use]
    pub fn for_failure_domain(&self, failure_domain: &str) -> DebtCounters {
        self.by_failure_domain
            .get(failure_domain)
            .copied()
            .unwrap_or_default()
    }

    /// Returns the debt attached to one scheme, or zero.
    #[must_use]
    pub fn for_scheme(&self, scheme: SchemeId) -> DebtCounters {
        self.by_scheme
            .get(&SchemeKey::new(scheme))
            .copied()
            .unwrap_or_default()
    }
}

fn add_map<K: Ord>(map: &mut BTreeMap<K, DebtCounters>, key: K, counters: DebtCounters) {
    if counters.is_zero() {
        return;
    }
    map.entry(key).or_default().add(counters);
}

fn subtract_map<K: Ord>(map: &mut BTreeMap<K, DebtCounters>, key: &K, counters: DebtCounters) {
    let remove = if let Some(current) = map.get_mut(key) {
        current.subtract(counters);
        current.is_zero()
    } else {
        false
    };
    if remove {
        map.remove(key);
    }
}

/// One asset in the controller's stable scrub catalog.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ScrubTarget {
    /// Immutable asset identity.
    pub asset: AssetId,
    /// Number of bytes to inspect for the asset.
    pub bytes: u64,
}

impl ScrubTarget {
    /// Builds one catalog entry.
    #[must_use]
    pub const fn new(asset: AssetId, bytes: u64) -> Self {
        Self { asset, bytes }
    }
}

/// One contiguous byte range selected for scrubbing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ScrubRange {
    /// Asset being inspected.
    pub asset: AssetId,
    /// Starting byte offset within the asset.
    pub offset: u64,
    /// Number of bytes selected.
    pub bytes: u64,
}

/// Restart-safe position in a scrub pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScrubCursor {
    /// Completed full-catalog cycle number.
    pub cycle: u64,
    /// Next asset to inspect, or `None` after the catalog is exhausted.
    pub next_asset: Option<AssetId>,
    /// Byte offset within `next_asset`.
    pub byte_offset: u64,
    /// Mode used for the current cycle.
    pub mode: ScrubMode,
    /// Whether the current cycle reached the end of the catalog.
    pub cycle_complete: bool,
}

impl ScrubCursor {
    /// Starts a cursor at the beginning of a catalog.
    #[must_use]
    pub const fn new(mode: ScrubMode) -> Self {
        Self {
            cycle: 0,
            next_asset: None,
            byte_offset: 0,
            mode,
            cycle_complete: false,
        }
    }

    /// Whether the current cycle has reached its end.
    #[must_use]
    pub const fn is_complete(self) -> bool {
        self.cycle_complete
    }

    /// Reconciles the cursor with the current catalog.
    ///
    /// The catalog is ordered by [`AssetId`]. A missing cursor asset resumes
    /// at the first later asset, so removing an already-visited entry cannot
    /// make a restart skip the rest of the catalog.
    pub fn normalize(&mut self, targets: &[ScrubTarget]) {
        let ordered = ordered_targets(targets);
        if ordered.is_empty() {
            self.next_asset = None;
            self.byte_offset = 0;
            self.cycle_complete = true;
            return;
        }
        if self.cycle_complete {
            self.cycle = self.cycle.saturating_add(1);
            self.next_asset = Some(ordered[0].asset);
            self.byte_offset = 0;
            self.cycle_complete = false;
        } else if self.next_asset.is_none() {
            self.next_asset = Some(ordered[0].asset);
            self.byte_offset = 0;
        }
        if let Some(asset) = self.next_asset {
            match ordered.binary_search_by_key(&asset, |target| target.asset) {
                Ok(index) => {
                    self.byte_offset = self.byte_offset.min(ordered[index].bytes);
                }
                Err(index) if index < ordered.len() => {
                    self.next_asset = Some(ordered[index].asset);
                    self.byte_offset = 0;
                }
                Err(_) => {
                    self.next_asset = None;
                    self.byte_offset = 0;
                    self.cycle_complete = true;
                }
            }
        }
    }

    /// Alias for [`normalize`](Self::normalize) that emphasizes bounds
    /// checking to persistence adapters.
    pub fn clamp(&mut self, targets: &[ScrubTarget]) {
        self.normalize(targets);
    }

    /// Returns the cursor's position in the stable catalog, if active.
    #[must_use]
    pub fn position(&self, targets: &[ScrubTarget]) -> Option<usize> {
        if self.cycle_complete {
            return None;
        }
        let ordered = ordered_targets(targets);
        let asset = self.next_asset?;
        ordered
            .binary_search_by_key(&asset, |target| target.asset)
            .ok()
    }
}

impl Default for ScrubCursor {
    fn default() -> Self {
        Self::new(ScrubMode::default())
    }
}

/// A bounded scrub work batch and the cursor to persist after it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScrubBatch {
    /// Mode used to select the ranges.
    pub mode: ScrubMode,
    /// Selected ranges in stable asset order.
    pub ranges: Vec<ScrubRange>,
    /// Total bytes selected.
    pub bytes: u64,
    /// Cursor to persist after the batch completes.
    pub next_cursor: ScrubCursor,
    /// Whether this batch completed the current catalog cycle.
    pub cycle_complete: bool,
}

impl ScrubBatch {
    /// Number of asset ranges in the batch.
    #[must_use]
    pub fn asset_count(&self) -> usize {
        self.ranges.len()
    }

    /// Whether the batch contains no work.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.ranges.is_empty()
    }
}

/// Bounded, deterministic metadata/full scrub schedule.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScrubSchedule {
    /// Minimum interval between metadata passes.
    pub metadata_interval: Duration,
    /// Minimum interval between full payload passes.
    pub full_interval: Duration,
    /// Deterministic scheduling spread added to a due time.
    pub jitter: Duration,
    /// Maximum asset ranges per batch.
    pub max_assets: usize,
    /// Maximum bytes per batch.
    pub max_bytes: u64,
    /// Persisted position in the catalog.
    pub cursor: ScrubCursor,
}

impl ScrubSchedule {
    /// Conservative schedule defaults.
    #[must_use]
    pub const fn conservative() -> Self {
        Self {
            metadata_interval: Duration::from_hours(24),
            full_interval: Duration::from_hours(168),
            jitter: Duration::from_hours(1),
            max_assets: 16,
            max_bytes: 64 * 1024 * 1024,
            cursor: ScrubCursor::new(ScrubMode::Metadata),
        }
    }

    /// Large finite schedule limits for deterministic tests and benchmarks.
    #[must_use]
    pub const fn test_wide() -> Self {
        Self {
            metadata_interval: Duration::from_hours(24),
            full_interval: Duration::from_hours(168),
            jitter: Duration::ZERO,
            max_assets: 4096,
            max_bytes: 1 << 40,
            cursor: ScrubCursor::new(ScrubMode::Metadata),
        }
    }

    /// Checks schedule bounds and ensures full scans are no more frequent than
    /// metadata scans.
    #[must_use]
    pub fn validate(self) -> bool {
        !self.metadata_interval.is_zero()
            && !self.full_interval.is_zero()
            && self.full_interval >= self.metadata_interval
            && self.jitter <= self.full_interval
            && self.max_assets > 0
            && self.max_bytes > 0
    }

    /// Reconstructs a schedule around a persisted cursor.
    #[must_use]
    pub fn resume(&self, cursor: ScrubCursor) -> Self {
        Self {
            metadata_interval: self.metadata_interval,
            full_interval: self.full_interval,
            jitter: self.jitter,
            max_assets: self.max_assets,
            max_bytes: self.max_bytes,
            cursor,
        }
    }

    /// Returns the current persisted cursor.
    #[must_use]
    pub const fn cursor(&self) -> ScrubCursor {
        self.cursor
    }

    /// Computes the deterministic jitter for a cursor.
    #[must_use]
    pub fn jitter_for(&self, cursor: &ScrubCursor) -> Duration {
        let span = self.jitter.as_micros();
        if span == 0 {
            return Duration::ZERO;
        }
        let mut state = 0xCBF2_9CE4_8422_2325_u64;
        for byte in cursor.next_asset.map_or([0; 32], |asset| asset.hash) {
            state = mix(state, u64::from(byte));
        }
        state = mix(state, cursor.cycle);
        state = mix(state, cursor.byte_offset);
        state = mix(
            state,
            match cursor.mode {
                ScrubMode::Metadata => 0,
                ScrubMode::Full => 1,
            },
        );
        let offset = u128::from(state) % span;
        Duration::from_micros(u64::try_from(offset).unwrap_or(u64::MAX))
    }

    /// Returns the next due time for a mode after `last`.
    #[must_use]
    pub fn next_due(&self, last: Ticks, mode: ScrubMode, cursor: &ScrubCursor) -> Ticks {
        let interval = match mode {
            ScrubMode::Metadata => self.metadata_interval,
            ScrubMode::Full => self.full_interval,
        };
        last.advance_by(interval.saturating_add(self.jitter_for(cursor)))
    }

    /// Selects the highest-priority due scrub mode, or `None` when neither
    /// interval has elapsed.
    #[must_use]
    pub fn due_mode(
        &self,
        now: Ticks,
        last_metadata: Option<Ticks>,
        last_full: Option<Ticks>,
        cursor: &ScrubCursor,
    ) -> Option<ScrubMode> {
        if last_full.is_none_or(|last| now >= self.next_due(last, ScrubMode::Full, cursor)) {
            Some(ScrubMode::Full)
        } else if last_metadata
            .is_none_or(|last| now >= self.next_due(last, ScrubMode::Metadata, cursor))
        {
            Some(ScrubMode::Metadata)
        } else {
            None
        }
    }

    /// Selects the next bounded batch and advances the persisted cursor.
    pub fn next_batch(&mut self, targets: &[ScrubTarget]) -> ScrubBatch {
        let ordered = ordered_targets(targets);
        let mut cursor = self.cursor;
        cursor.normalize(&ordered);
        let mut ranges = Vec::new();
        let mut bytes = 0_u64;
        let mut index = cursor.position(&ordered).unwrap_or(0);

        while index < ordered.len() && ranges.len() < self.max_assets && bytes < self.max_bytes {
            let target = ordered[index];
            if target.bytes == 0 {
                index += 1;
                if index < ordered.len() {
                    cursor.next_asset = Some(ordered[index].asset);
                    cursor.byte_offset = 0;
                } else {
                    cursor.next_asset = None;
                    cursor.byte_offset = 0;
                    cursor.cycle_complete = true;
                }
                continue;
            }

            let offset = cursor.byte_offset.min(target.bytes);
            let remaining = target.bytes.saturating_sub(offset);
            let available = self.max_bytes.saturating_sub(bytes);
            let selected = remaining.min(available);
            if selected == 0 {
                break;
            }
            ranges.push(ScrubRange {
                asset: target.asset,
                offset,
                bytes: selected,
            });
            bytes = bytes.saturating_add(selected);
            let next_offset = offset.saturating_add(selected);
            if next_offset >= target.bytes {
                index += 1;
                if index < ordered.len() {
                    cursor.next_asset = Some(ordered[index].asset);
                    cursor.byte_offset = 0;
                } else {
                    cursor.next_asset = None;
                    cursor.byte_offset = 0;
                    cursor.cycle_complete = true;
                }
            } else {
                cursor.next_asset = Some(target.asset);
                cursor.byte_offset = next_offset;
                break;
            }
        }

        self.cursor = cursor;
        ScrubBatch {
            mode: self.cursor.mode,
            ranges,
            bytes,
            next_cursor: self.cursor,
            cycle_complete: self.cursor.cycle_complete,
        }
    }
}

impl Default for ScrubSchedule {
    fn default() -> Self {
        Self::conservative()
    }
}

fn ordered_targets(targets: &[ScrubTarget]) -> Vec<ScrubTarget> {
    let mut ordered = targets.to_vec();
    ordered.sort_unstable_by_key(|target| (target.asset, target.bytes));
    ordered.dedup_by_key(|target| target.asset);
    ordered
}

fn mix(mut state: u64, word: u64) -> u64 {
    state ^= word;
    state = state.wrapping_mul(0x1000_0000_01B3);
    state ^= state >> 29;
    state = state.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    state
}

/// Counts completed maintenance work for an observability surface.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MaintenanceReport {
    /// Start of the maintenance window.
    pub started_at: Ticks,
    /// End of the maintenance window.
    pub finished_at: Ticks,
    /// Scrub mode used during the window.
    pub scrub_mode: ScrubMode,
    /// Assets inspected by scrubbing.
    pub scrub_assets: u64,
    /// Bytes read by scrubbing.
    pub scrub_bytes: u64,
    /// Assets repaired.
    pub repair_assets: u64,
    /// Bytes reconstructed or transferred by repair.
    pub repair_bytes: u64,
    /// Completed reconstruction operations.
    pub reconstructions: u64,
    /// Completed fragment transfers.
    pub transfers: u64,
    /// Corruption observations.
    pub corruption_detected: u64,
    /// Failed repair or reconstruction operations.
    pub repair_failures: u64,
    /// Stale completions fenced by the controller.
    pub stale_results: u64,
    /// Debt remaining after the window.
    pub debt: RepairDebt,
}

impl MaintenanceReport {
    /// Builds an empty report for a completed time window.
    #[must_use]
    pub const fn new(started_at: Ticks, finished_at: Ticks, scrub_mode: ScrubMode) -> Self {
        Self {
            started_at,
            finished_at,
            scrub_mode,
            scrub_assets: 0,
            scrub_bytes: 0,
            repair_assets: 0,
            repair_bytes: 0,
            reconstructions: 0,
            transfers: 0,
            corruption_detected: 0,
            repair_failures: 0,
            stale_results: 0,
            debt: RepairDebt::new_const(),
        }
    }

    /// Adds scrub work to the report.
    pub fn record_scrub(&mut self, assets: u64, bytes: u64) {
        self.scrub_assets = self.scrub_assets.saturating_add(assets);
        self.scrub_bytes = self.scrub_bytes.saturating_add(bytes);
    }

    /// Adds repair work to the report.
    pub fn record_repair(&mut self, assets: u64, bytes: u64) {
        self.repair_assets = self.repair_assets.saturating_add(assets);
        self.repair_bytes = self.repair_bytes.saturating_add(bytes);
    }

    /// Adds one completed reconstruction.
    pub fn record_reconstruction(&mut self) {
        self.reconstructions = self.reconstructions.saturating_add(1);
    }

    /// Adds one completed fragment transfer.
    pub fn record_transfer(&mut self) {
        self.transfers = self.transfers.saturating_add(1);
    }

    /// Adds one corruption observation.
    pub fn record_corruption(&mut self) {
        self.corruption_detected = self.corruption_detected.saturating_add(1);
    }

    /// Adds one failed maintenance operation.
    pub fn record_failure(&mut self) {
        self.repair_failures = self.repair_failures.saturating_add(1);
    }

    /// Adds one stale completion rejected by fencing.
    pub fn record_stale_result(&mut self) {
        self.stale_results = self.stale_results.saturating_add(1);
    }

    /// Returns the non-negative duration between the report boundaries.
    #[must_use]
    pub const fn elapsed(&self) -> Duration {
        self.finished_at.saturating_since(self.started_at)
    }
}

/// Point-in-time state exposed by a distributed maintenance controller.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ControllerSnapshot {
    /// Time at which the state was observed.
    pub observed_at: Ticks,
    /// Active resource limits.
    pub budgets: MaintenanceBudgets,
    /// Scrub schedule and its persisted cursor.
    pub schedule: ScrubSchedule,
    /// Current aggregate debt.
    pub debt: RepairDebt,
    /// Work waiting in the shared queue.
    pub queued_work: usize,
    /// Reconstructions currently executing.
    pub in_flight_reconstructions: usize,
    /// Transfers currently executing.
    pub in_flight_transfers: usize,
    /// Scrub operations currently executing.
    pub in_flight_scrubs: usize,
    /// Most recent completed maintenance window.
    pub last_report: Option<MaintenanceReport>,
    /// Most recent repair or scrub failure, retained for operator diagnosis.
    pub last_error: Option<String>,
}

impl ControllerSnapshot {
    /// Builds an idle snapshot with a supplied observation time.
    #[must_use]
    pub const fn new(
        observed_at: Ticks,
        budgets: MaintenanceBudgets,
        schedule: ScrubSchedule,
        debt: RepairDebt,
    ) -> Self {
        Self {
            observed_at,
            budgets,
            schedule,
            debt,
            queued_work: 0,
            in_flight_reconstructions: 0,
            in_flight_transfers: 0,
            in_flight_scrubs: 0,
            last_report: None,
            last_error: None,
        }
    }

    /// Returns the persisted scrub cursor without exposing schedule internals.
    #[must_use]
    pub const fn cursor(&self) -> ScrubCursor {
        self.schedule.cursor
    }

    /// Replaces the most recent report.
    pub fn set_last_report(&mut self, report: MaintenanceReport) {
        self.last_report = Some(report);
    }
}

impl RepairDebt {
    const fn new_const() -> Self {
        Self {
            total_assets: 0,
            total_bytes: 0,
            total_fragments: 0,
            total_critical_assets: 0,
            by_node: BTreeMap::new(),
            by_failure_domain: BTreeMap::new(),
            by_scheme: BTreeMap::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::AssetKind;

    fn asset(value: u8) -> AssetId {
        AssetId::new(AssetKind::Chunk, 0, [value; 32]).expect("asset")
    }

    fn risk(usable: u32, remaining: u32, corrupt: u32) -> RepairRisk {
        RepairRisk {
            remaining_independent_tolerance: remaining,
            usable_fragments: usable,
            required_fragments: 3,
            total_fragments: 5,
            scheme_requirements: SchemeRequirements::new(SchemeId::ReedSolomon, 5, 3, 2),
            failure_domain_concentration: usable,
            independent_failure_domains: usable,
            unavailable_nodes: Vec::new(),
            draining_nodes: Vec::new(),
            corrupt_fragments: corrupt,
            unavailable_fragments: 0,
            asset_bytes: 1024,
            reconstruction_bytes: 2048,
            degradation_age: Duration::ZERO,
            verification_age: Duration::ZERO,
            placement_violations: 0,
        }
    }

    #[test]
    fn priority_is_deterministic_and_ordered_by_risk() {
        let mut critical = risk(1, 0, 0);
        critical.corrupt_fragments = 2;
        let mut same_risk = critical.clone();
        same_risk.unavailable_nodes = vec![NodeId::from_u64(9), NodeId::from_u64(2)];
        same_risk.draining_nodes = vec![NodeId::from_u64(8), NodeId::from_u64(1)];
        let mut reordered = same_risk.clone();
        reordered.unavailable_nodes.reverse();
        reordered.draining_nodes.reverse();
        assert_eq!(same_risk.priority(), reordered.priority());

        let corruption = risk(3, 1, 1);
        assert!(critical.priority() > corruption.priority());
        let healthy = risk(5, 2, 0);
        assert!(corruption.priority() > healthy.priority());
    }

    #[test]
    fn debt_aggregates_removes_and_merges_deterministically() {
        let first = RepairDebtEntry::new(
            NodeId::from_u64(1),
            "rack-a".to_owned(),
            SchemeId::ReedSolomon,
            DebtCounters::new(2, 100, 4, 1),
        );
        let second = RepairDebtEntry::new(
            NodeId::from_u64(2),
            "rack-b".to_owned(),
            SchemeId::Replication,
            DebtCounters::new(3, 200, 3, 0),
        );
        let mut left = RepairDebt::default();
        left.add(&first);
        left.add(&second);
        assert_eq!(left.total_assets, 5);
        assert_eq!(left.total_bytes, 300);
        assert_eq!(left.total_fragments, 7);
        assert_eq!(left.for_node(NodeId::from_u64(1)).assets, 2);
        assert_eq!(left.for_failure_domain("rack-b").assets, 3);
        assert_eq!(left.for_scheme(SchemeId::Replication).bytes, 200);

        left.remove(&first);
        assert_eq!(left.total_assets, 3);
        assert!(left.for_node(NodeId::from_u64(1)).is_zero());

        let merged = RepairDebt::default().merge(left).merge(RepairDebt {
            total_assets: 1,
            total_bytes: 7,
            total_fragments: 1,
            total_critical_assets: 1,
            by_node: BTreeMap::from([(NodeId::from_u64(9), DebtCounters::new(1, 7, 1, 1))]),
            by_failure_domain: BTreeMap::from([(
                "rack-c".to_owned(),
                DebtCounters::new(1, 7, 1, 1),
            )]),
            by_scheme: BTreeMap::from([(
                SchemeKey::new(SchemeId::ReedSolomon),
                DebtCounters::new(1, 7, 1, 1),
            )]),
        });
        assert_eq!(merged.total_assets, 4);
        assert_eq!(merged.for_node(NodeId::from_u64(9)).critical_assets, 1);
    }

    #[test]
    fn cursor_is_bounded_and_resumes_after_reconstruction() {
        let targets = vec![
            ScrubTarget::new(asset(3), 8),
            ScrubTarget::new(asset(1), 6),
            ScrubTarget::new(asset(2), 5),
        ];
        let mut schedule = ScrubSchedule::test_wide();
        schedule.max_assets = 2;
        schedule.max_bytes = 9;
        let first = schedule.next_batch(&targets);
        assert_eq!(first.bytes, 9);
        assert!(first.ranges.len() <= 2);
        assert!(first.ranges.iter().all(|range| range.bytes <= 9));

        let mut resumed = schedule.resume(first.next_cursor);
        let mut uninterrupted = schedule;
        uninterrupted.cursor = first.next_cursor;
        assert_eq!(
            resumed.next_batch(&targets),
            uninterrupted.next_batch(&targets)
        );

        let mut cursor = ScrubCursor::new(ScrubMode::Full);
        cursor.next_asset = Some(asset(2));
        cursor.byte_offset = 99;
        cursor.clamp(&targets);
        assert_eq!(cursor.byte_offset, 5);
        assert!(cursor.position(&targets).is_some());
    }
}
