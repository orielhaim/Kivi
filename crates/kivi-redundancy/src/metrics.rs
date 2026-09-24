//! Observability: why an asset is considered protected or degraded.
//!
//! [`FabricMetrics`] snapshots everything an operator needs: assets by
//! scheme, logical vs physical bytes, storage amplification,
//! healthy/degraded/critical/unrecoverable counts, fragments by
//! node/failure domain, pending repairs, repair bytes, encode/decode
//! activity, corruption detections, reconstruction failures, stale work
//! rejected, and transition state.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

use kivi_types::NodeId;

/// Health classes the repair engine reports per asset.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AssetHealth {
    /// All fragments healthy on independent domains.
    Healthy,
    /// Degraded but reconstructable now.
    Degraded,
    /// One failure from unrecoverable (tolerance exhausted).
    Critical,
    /// Not reconstructable now.
    Unrecoverable,
}

impl AssetHealth {
    /// Human-readable name.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Healthy => "healthy",
            Self::Degraded => "degraded",
            Self::Critical => "critical",
            Self::Unrecoverable => "unrecoverable",
        }
    }
}

/// Live counters for the fabric (atomics for lock-free admin reads).
#[derive(Debug, Default)]
pub struct FabricMetrics {
    /// Assets protected by replication.
    pub replicated_assets: AtomicU64,
    /// Assets protected by erasure coding.
    pub coded_assets: AtomicU64,
    /// Sum of logical bytes protected.
    pub logical_bytes: AtomicU64,
    /// Sum of physical fragment bytes stored.
    pub physical_bytes: AtomicU64,
    /// Healthy assets.
    pub healthy: AtomicU64,
    /// Degraded-but-recoverable assets.
    pub degraded: AtomicU64,
    /// Critically degraded assets.
    pub critical: AtomicU64,
    /// Unrecoverable assets.
    pub unrecoverable: AtomicU64,
    /// Fragments pending repair.
    pub pending_repairs: AtomicU64,
    /// Bytes queued for repair.
    pub repair_bytes_pending: AtomicU64,
    /// Bytes repaired (cumulative).
    pub repair_bytes_done: AtomicU64,
    /// Encode operations completed.
    pub encodes: AtomicU64,
    /// Decode/reconstruct operations completed.
    pub decodes: AtomicU64,
    /// Corruption detections (checksum mismatches, lies rejected).
    pub corruption_detected: AtomicU64,
    /// Reconstruction failures (unrecoverable reads).
    pub reconstruction_failures: AtomicU64,
    /// Stale-generation completions fenced and rejected.
    pub stale_rejected: AtomicU64,
    /// Scheme transitions completed.
    pub transitions: AtomicU64,
    /// Transitions that held extra redundancy during build→verify→publish.
    pub transitions_with_overlap: AtomicU64,
    /// Fragment bytes sent to remote holders (distributed uploads + repairs).
    pub remote_bytes_sent: AtomicU64,
    /// Fragment bytes received from remote holders (distributed reads).
    pub remote_bytes_received: AtomicU64,
    /// Fragment bytes stored on this node's own holder.
    pub local_bytes_written: AtomicU64,
    /// Fragment bytes read from this node's own holder.
    pub local_bytes_read: AtomicU64,
    /// Remote fragment fetches attempted.
    pub fetch_count: AtomicU64,
    /// Remote fragment fetches that failed (per-attempt, before alternates).
    pub fetch_error_count: AtomicU64,
    /// Sum of fetch latencies in microseconds (mean = sum / count).
    pub fetch_latency_us_sum: AtomicU64,
    /// Maximum observed fetch latency in microseconds.
    pub fetch_latency_us_max: AtomicU64,
    /// Distributed reads that succeeded with missing/corrupt pieces.
    pub degraded_reads: AtomicU64,
    /// Orphan fragments observed (staged without an accepted layout).
    pub orphans_seen: AtomicU64,
    /// Orphan fragments swept.
    pub orphans_swept: AtomicU64,
    /// RPCs refused on incarnation mismatch (refreshed and retried).
    pub incarnation_refusals: AtomicU64,
    /// Assessments where installed fragments diverged from desired placement.
    pub desired_vs_actual_mismatches: AtomicU64,
    /// Fragments re-installed by distributed repair (cumulative).
    pub repair_fragments_rebuilt: AtomicU64,
    /// Scrub passes started.
    pub scrub_runs: AtomicU64,
    /// Assets inspected by scrub passes.
    pub scrub_assets: AtomicU64,
    /// Bytes inspected by scrub passes.
    pub scrub_bytes: AtomicU64,
    /// Full-payload scrub passes started.
    pub full_scrubs: AtomicU64,
    /// Age of the last scrub pass in milliseconds.
    pub last_scrub_age_ms: AtomicU64,
    /// Age of the last successful verification in milliseconds.
    pub last_verification_age_ms: AtomicU64,
    /// Missing published fragments observed by maintenance.
    pub missing_fragments: AtomicU64,
    /// Logical bytes currently degraded.
    pub degraded_bytes: AtomicU64,
    /// Physical bytes missing from published layouts.
    pub missing_physical_bytes: AtomicU64,
    /// Milliseconds assets have spent degraded.
    pub degraded_time_ms: AtomicU64,
    /// Estimated bytes required to reconstruct queued repairs.
    pub estimated_reconstruction_bytes: AtomicU64,
    /// Sum of exposed failure-domain multiplicity.
    pub failure_domain_exposure: AtomicU64,
    /// Repair attempts started.
    pub repair_attempts: AtomicU64,
    /// Repair attempts that failed.
    pub repair_failures: AtomicU64,
    /// Repair retries scheduled.
    pub repair_retries: AtomicU64,
    /// Sum of repair durations in microseconds.
    pub repair_duration_us_sum: AtomicU64,
    /// Maximum repair duration in microseconds.
    pub repair_duration_us_max: AtomicU64,
    /// Number of source fragments selected by maintenance.
    pub repair_sources: AtomicU64,
    /// Number of target fragments selected by maintenance.
    pub repair_targets: AtomicU64,
    /// Fetches that used more than one source candidate.
    pub multi_source_fetches: AtomicU64,
    /// Hedged source reads started.
    pub multi_source_hedges: AtomicU64,
    /// Source reads rejected or failed during maintenance.
    pub source_failures: AtomicU64,
    /// Bytes recovered by completed repairs.
    pub recovered_bytes: AtomicU64,
    /// Placement violations observed by maintenance.
    pub placement_violations: AtomicU64,
    /// Orphan candidates retained by age or authority ambiguity.
    pub orphans_retained: AtomicU64,
    /// Bytes reclaimed by orphan lifecycle work.
    pub reclaimed_bytes: AtomicU64,
    /// Maintenance ticks throttled by a budget.
    pub maintenance_throttled: AtomicU64,
    /// Maintenance ticks suppressed by foreground pressure.
    pub foreground_pressure_suppressed: AtomicU64,
    /// Scrub operations currently executing.
    pub scrub_inflight: AtomicU64,
    /// Repair operations currently executing.
    pub repair_inflight: AtomicU64,
}

impl FabricMetrics {
    /// Creates zeroed metrics.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Snapshots counters for admin surfaces.
    #[must_use]
    pub fn snapshot(&self) -> FabricMetricsSnapshot {
        FabricMetricsSnapshot {
            replicated_assets: self.replicated_assets.load(Ordering::Relaxed),
            coded_assets: self.coded_assets.load(Ordering::Relaxed),
            logical_bytes: self.logical_bytes.load(Ordering::Relaxed),
            physical_bytes: self.physical_bytes.load(Ordering::Relaxed),
            healthy: self.healthy.load(Ordering::Relaxed),
            degraded: self.degraded.load(Ordering::Relaxed),
            critical: self.critical.load(Ordering::Relaxed),
            unrecoverable: self.unrecoverable.load(Ordering::Relaxed),
            pending_repairs: self.pending_repairs.load(Ordering::Relaxed),
            repair_bytes_pending: self.repair_bytes_pending.load(Ordering::Relaxed),
            repair_bytes_done: self.repair_bytes_done.load(Ordering::Relaxed),
            encodes: self.encodes.load(Ordering::Relaxed),
            decodes: self.decodes.load(Ordering::Relaxed),
            corruption_detected: self.corruption_detected.load(Ordering::Relaxed),
            reconstruction_failures: self.reconstruction_failures.load(Ordering::Relaxed),
            stale_rejected: self.stale_rejected.load(Ordering::Relaxed),
            transitions: self.transitions.load(Ordering::Relaxed),
            transitions_with_overlap: self.transitions_with_overlap.load(Ordering::Relaxed),
            remote_bytes_sent: self.remote_bytes_sent.load(Ordering::Relaxed),
            remote_bytes_received: self.remote_bytes_received.load(Ordering::Relaxed),
            local_bytes_written: self.local_bytes_written.load(Ordering::Relaxed),
            local_bytes_read: self.local_bytes_read.load(Ordering::Relaxed),
            fetch_count: self.fetch_count.load(Ordering::Relaxed),
            fetch_error_count: self.fetch_error_count.load(Ordering::Relaxed),
            fetch_latency_us_sum: self.fetch_latency_us_sum.load(Ordering::Relaxed),
            fetch_latency_us_max: self.fetch_latency_us_max.load(Ordering::Relaxed),
            degraded_reads: self.degraded_reads.load(Ordering::Relaxed),
            orphans_seen: self.orphans_seen.load(Ordering::Relaxed),
            orphans_swept: self.orphans_swept.load(Ordering::Relaxed),
            incarnation_refusals: self.incarnation_refusals.load(Ordering::Relaxed),
            desired_vs_actual_mismatches: self.desired_vs_actual_mismatches.load(Ordering::Relaxed),
            repair_fragments_rebuilt: self.repair_fragments_rebuilt.load(Ordering::Relaxed),
            scrub_runs: self.scrub_runs.load(Ordering::Relaxed),
            scrub_assets: self.scrub_assets.load(Ordering::Relaxed),
            scrub_bytes: self.scrub_bytes.load(Ordering::Relaxed),
            full_scrubs: self.full_scrubs.load(Ordering::Relaxed),
            last_scrub_age_ms: self.last_scrub_age_ms.load(Ordering::Relaxed),
            last_verification_age_ms: self.last_verification_age_ms.load(Ordering::Relaxed),
            missing_fragments: self.missing_fragments.load(Ordering::Relaxed),
            degraded_bytes: self.degraded_bytes.load(Ordering::Relaxed),
            missing_physical_bytes: self.missing_physical_bytes.load(Ordering::Relaxed),
            degraded_time_ms: self.degraded_time_ms.load(Ordering::Relaxed),
            estimated_reconstruction_bytes: self
                .estimated_reconstruction_bytes
                .load(Ordering::Relaxed),
            failure_domain_exposure: self.failure_domain_exposure.load(Ordering::Relaxed),
            repair_attempts: self.repair_attempts.load(Ordering::Relaxed),
            repair_failures: self.repair_failures.load(Ordering::Relaxed),
            repair_retries: self.repair_retries.load(Ordering::Relaxed),
            repair_duration_us_sum: self.repair_duration_us_sum.load(Ordering::Relaxed),
            repair_duration_us_max: self.repair_duration_us_max.load(Ordering::Relaxed),
            repair_sources: self.repair_sources.load(Ordering::Relaxed),
            repair_targets: self.repair_targets.load(Ordering::Relaxed),
            multi_source_fetches: self.multi_source_fetches.load(Ordering::Relaxed),
            multi_source_hedges: self.multi_source_hedges.load(Ordering::Relaxed),
            source_failures: self.source_failures.load(Ordering::Relaxed),
            recovered_bytes: self.recovered_bytes.load(Ordering::Relaxed),
            placement_violations: self.placement_violations.load(Ordering::Relaxed),
            orphans_retained: self.orphans_retained.load(Ordering::Relaxed),
            reclaimed_bytes: self.reclaimed_bytes.load(Ordering::Relaxed),
            maintenance_throttled: self.maintenance_throttled.load(Ordering::Relaxed),
            foreground_pressure_suppressed: self
                .foreground_pressure_suppressed
                .load(Ordering::Relaxed),
            scrub_inflight: self.scrub_inflight.load(Ordering::Relaxed),
            repair_inflight: self.repair_inflight.load(Ordering::Relaxed),
        }
    }

    /// Storage amplification (`physical / logical`, `None` when empty).
    // Amplification is approximate; precision loss past 2^53 is irrelevant.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn amplification(&self) -> Option<f64> {
        let logical = self.logical_bytes.load(Ordering::Relaxed);
        let physical = self.physical_bytes.load(Ordering::Relaxed);
        if logical == 0 {
            None
        } else {
            Some(physical as f64 / logical as f64)
        }
    }
}

/// Point-in-time metrics for admin/control planes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FabricMetricsSnapshot {
    /// Assets by replication.
    pub replicated_assets: u64,
    /// Assets by erasure coding.
    pub coded_assets: u64,
    /// Logical bytes.
    pub logical_bytes: u64,
    /// Physical bytes.
    pub physical_bytes: u64,
    /// Healthy assets.
    pub healthy: u64,
    /// Degraded assets.
    pub degraded: u64,
    /// Critical assets.
    pub critical: u64,
    /// Unrecoverable assets.
    pub unrecoverable: u64,
    /// Pending repairs.
    pub pending_repairs: u64,
    /// Repair bytes pending.
    pub repair_bytes_pending: u64,
    /// Repair bytes done.
    pub repair_bytes_done: u64,
    /// Encodes completed.
    pub encodes: u64,
    /// Decodes completed.
    pub decodes: u64,
    /// Corruption detections.
    pub corruption_detected: u64,
    /// Reconstruction failures.
    pub reconstruction_failures: u64,
    /// Stale work rejected.
    pub stale_rejected: u64,
    /// Transitions completed.
    pub transitions: u64,
    /// Transitions with overlap.
    pub transitions_with_overlap: u64,
    /// Remote bytes sent.
    pub remote_bytes_sent: u64,
    /// Remote bytes received.
    pub remote_bytes_received: u64,
    /// Bytes stored on this node's own holder.
    pub local_bytes_written: u64,
    /// Bytes read from this node's own holder.
    pub local_bytes_read: u64,
    /// Remote fetches attempted.
    pub fetch_count: u64,
    /// Remote fetches failed (per-attempt).
    pub fetch_error_count: u64,
    /// Fetch latency sum in microseconds.
    pub fetch_latency_us_sum: u64,
    /// Fetch latency maximum in microseconds.
    pub fetch_latency_us_max: u64,
    /// Degraded reads served.
    pub degraded_reads: u64,
    /// Orphans observed.
    pub orphans_seen: u64,
    /// Orphans swept.
    pub orphans_swept: u64,
    /// Incarnation refusals seen.
    pub incarnation_refusals: u64,
    /// Desired-vs-actual placement mismatches.
    pub desired_vs_actual_mismatches: u64,
    /// Fragments rebuilt by distributed repair.
    pub repair_fragments_rebuilt: u64,
    /// Scrub passes started.
    pub scrub_runs: u64,
    /// Assets inspected by scrub passes.
    pub scrub_assets: u64,
    /// Bytes inspected by scrub passes.
    pub scrub_bytes: u64,
    /// Full-payload scrub passes started.
    pub full_scrubs: u64,
    /// Age of the last scrub pass in milliseconds.
    pub last_scrub_age_ms: u64,
    /// Age of the last successful verification in milliseconds.
    pub last_verification_age_ms: u64,
    /// Missing published fragments observed by maintenance.
    pub missing_fragments: u64,
    /// Logical bytes currently degraded.
    pub degraded_bytes: u64,
    /// Physical bytes missing from published layouts.
    pub missing_physical_bytes: u64,
    /// Milliseconds assets have spent degraded.
    pub degraded_time_ms: u64,
    /// Estimated bytes required to reconstruct queued repairs.
    pub estimated_reconstruction_bytes: u64,
    /// Sum of exposed failure-domain multiplicity.
    pub failure_domain_exposure: u64,
    /// Repair attempts started.
    pub repair_attempts: u64,
    /// Repair attempts that failed.
    pub repair_failures: u64,
    /// Repair retries scheduled.
    pub repair_retries: u64,
    /// Sum of repair durations in microseconds.
    pub repair_duration_us_sum: u64,
    /// Maximum repair duration in microseconds.
    pub repair_duration_us_max: u64,
    /// Number of source fragments selected by maintenance.
    pub repair_sources: u64,
    /// Number of target fragments selected by maintenance.
    pub repair_targets: u64,
    /// Fetches that used more than one source candidate.
    pub multi_source_fetches: u64,
    /// Hedged source reads started.
    pub multi_source_hedges: u64,
    /// Source reads rejected or failed during maintenance.
    pub source_failures: u64,
    /// Bytes recovered by completed repairs.
    pub recovered_bytes: u64,
    /// Placement violations observed by maintenance.
    pub placement_violations: u64,
    /// Orphan candidates retained by age or authority ambiguity.
    pub orphans_retained: u64,
    /// Bytes reclaimed by orphan lifecycle work.
    pub reclaimed_bytes: u64,
    /// Maintenance ticks throttled by a budget.
    pub maintenance_throttled: u64,
    /// Maintenance ticks suppressed by foreground pressure.
    pub foreground_pressure_suppressed: u64,
    /// Scrub operations currently executing.
    pub scrub_inflight: u64,
    /// Repair operations currently executing.
    pub repair_inflight: u64,
}

impl FabricMetricsSnapshot {
    /// Storage amplification, if any data is protected.
    // Amplification is approximate; precision loss past 2^53 is irrelevant.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn amplification(self) -> Option<f64> {
        if self.logical_bytes == 0 {
            None
        } else {
            Some(self.physical_bytes as f64 / self.logical_bytes as f64)
        }
    }
}

/// Fragments per node for capacity balancing views.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FragmentCensus {
    /// Fragments per node id.
    pub by_node: HashMap<NodeId, u64>,
    /// Fragments per independence key (failure-domain view).
    pub by_domain: HashMap<String, u64>,
    /// Stored fragment bytes per node id (bounded by cluster size).
    pub by_node_bytes: HashMap<NodeId, u64>,
}

impl FragmentCensus {
    /// Creates an empty census.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Records one fragment.
    pub fn record(&mut self, node: NodeId, domain_key: String) {
        *self.by_node.entry(node).or_insert(0) += 1;
        *self.by_domain.entry(domain_key).or_insert(0) += 1;
    }

    /// Records one fragment's stored bytes against its holder.
    pub fn record_bytes(&mut self, node: NodeId, bytes: u64) {
        *self.by_node_bytes.entry(node).or_insert(0) += bytes;
    }
}
