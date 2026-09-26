//! Kivi campaign orchestration and reporting.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::Context as _;
use bytes::Bytes;
use clap::ValueEnum;
use kivi_client::{ClientConfig, NativeClient};
use kivi_core::{SystemClock, wall_now_or_max};
use kivi_state::Key;
use kivi_types::{NamespaceId, WallTimestamp};
use serde::Serialize;
use tempfile::TempDir;

use crate::conformance::{self, RedisConfig};
use crate::process::{Server, ServerMode, release_server_binary_path, try_admin_get_endpoint};
use crate::resp_client::{Reply, RespClient};
use crate::runner::{self, RunStats, RunnerConfig};
use crate::targets::{
    KiviNativeTarget, RespTarget, TargetStats, check_workload_reply, encode_workload_command,
};
use crate::workload::{
    KeySpace, PayloadCache, RANGE_WINDOW, Workload, WorkloadOp, counter_key_name, thread_key_names,
    workload_op_for, workload_supports_resp,
};

/// Schema identifier for campaign JSON and Markdown reports.
pub const CAMPAIGN_SCHEMA: &str = "kivi-lab-campaign/2";

static CAMPAIGN_IDS: AtomicU64 = AtomicU64::new(0);

/// Bounded campaign presets.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, ValueEnum, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum CampaignPreset {
    /// Small local smoke campaign.
    #[default]
    Smoke,
    /// Broader local campaign with repeated trials and curves.
    Standard,
    /// Large, explicitly operator-selected campaign.
    Full,
}

/// Automatically managed Kivi durability profile.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, ValueEnum, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum KiviMode {
    /// In-memory baseline.
    Ephemeral,
    /// WAL-backed temporary data directory.
    #[default]
    Durable,
}

impl CampaignPreset {
    fn profile(self) -> CampaignProfile {
        match self {
            Self::Smoke => CampaignProfile {
                trials: 1,
                duration: Duration::from_millis(250),
                warmup: 20,
                base_value_len: 1024,
                value_sizes: vec![16],
                base_key_space: KeyPoint::uniform256(),
                key_distributions: vec![KeyPoint::hot()],
                named_profiles: vec![NamedProfile::new("cache-read", Workload::Cache)],
                native_only_profiles: Vec::new(),
                base_threads: 2,
                base_pipeline: 1,
                thread_counts: vec![1, 2],
                pipeline_depths: vec![1, 4],
                curve_workload: Workload::Cache,
                large_values: Vec::new(),
            },
            Self::Standard => CampaignProfile {
                trials: 2,
                duration: Duration::from_millis(750),
                warmup: 100,
                base_value_len: 1024,
                value_sizes: vec![16, 1024, 64 * 1024],
                base_key_space: KeyPoint::uniform256(),
                key_distributions: vec![KeyPoint::small(), KeyPoint::uniform256()],
                named_profiles: vec![
                    NamedProfile::new("cache-read", Workload::Cache),
                    NamedProfile::new("read-heavy", Workload::ReadHeavy),
                    NamedProfile::new("balanced", Workload::Balanced),
                ],
                native_only_profiles: vec![
                    NamedProfile::new("counter", Workload::Counter),
                    NamedProfile::new("mixed", Workload::Mixed),
                ],
                base_threads: 4,
                base_pipeline: 1,
                thread_counts: vec![1, 2, 4, 8],
                pipeline_depths: vec![1, 4, 16],
                curve_workload: Workload::Balanced,
                large_values: vec![1024 * 1024, 4 * 1024 * 1024],
            },
            Self::Full => CampaignProfile {
                trials: 3,
                duration: Duration::from_secs(2),
                warmup: 500,
                base_value_len: 1024,
                value_sizes: vec![
                    16,
                    1024,
                    64 * 1024,
                    256 * 1024,
                    1024 * 1024,
                    4 * 1024 * 1024,
                ],
                base_key_space: KeyPoint::uniform256(),
                key_distributions: vec![
                    KeyPoint::hot(),
                    KeyPoint::small(),
                    KeyPoint::uniform256(),
                    KeyPoint::uniform4k(),
                    KeyPoint::large(),
                ],
                named_profiles: vec![
                    NamedProfile::new("get", Workload::Get),
                    NamedProfile::new("set", Workload::Set),
                    NamedProfile::new("delete", Workload::Delete),
                    NamedProfile::new("exists", Workload::Exists),
                    NamedProfile::new("range", Workload::Range),
                    NamedProfile::new("cache-read", Workload::Cache),
                    NamedProfile::new("read-heavy", Workload::ReadHeavy),
                    NamedProfile::new("balanced", Workload::Balanced),
                    NamedProfile::new("compat", Workload::Compat),
                ],
                native_only_profiles: vec![
                    NamedProfile::new("counter", Workload::Counter),
                    NamedProfile::new("mixed", Workload::Mixed),
                ],
                base_threads: 8,
                base_pipeline: 1,
                thread_counts: vec![1, 2, 4, 8, 16, 32, 64],
                pipeline_depths: vec![1, 4, 16, 64, 256],
                curve_workload: Workload::Balanced,
                large_values: vec![1024 * 1024, 4 * 1024 * 1024, 64 * 1024 * 1024],
            },
        }
    }
}

/// Configuration for a campaign run.
#[derive(Debug, Clone)]
pub struct CampaignOptions {
    /// Preset supplying bounded defaults.
    pub preset: CampaignPreset,
    /// Explicit Redis URL, or the environment/default discovery path.
    pub redis_url: Option<String>,
    /// External Kivi native endpoint. If absent, a RESP endpoint is spawned or supplied separately.
    pub kivi_endpoint: Option<String>,
    /// External Kivi RESP endpoint.
    pub kivi_resp_endpoint: Option<String>,
    /// Optional Kivi admin endpoint for metadata collection.
    pub kivi_admin_endpoint: Option<String>,
    /// Output directory. The default is timestamped under `target/kivi-lab-campaign`.
    pub output_dir: Option<PathBuf>,
    /// Override the preset trial count.
    pub trials: Option<usize>,
    /// Override duration-based runs. Fixed counts take precedence when `ops_per_trial` is set.
    pub duration: Option<Duration>,
    /// Fixed measured operations per thread for campaign points.
    pub ops_per_trial: Option<usize>,
    /// Override concurrency points.
    pub thread_counts: Option<Vec<usize>>,
    /// Override pipeline depths.
    pub pipeline_depths: Option<Vec<usize>>,
    /// Override base value length.
    pub value_len: Option<usize>,
    /// Override the base workload list.
    pub workload: Option<Workload>,
    /// Enable, disable, or inherit the preset's large-value experiment.
    pub include_large: Option<bool>,
    /// Shared key-space override; `None` uses the preset's key count.
    pub key_space: Option<KeySpace>,
    /// Durability mode for an automatically spawned Kivi server.
    pub kivi_mode: KiviMode,
    /// Native workers used when Kivi is spawned.
    pub workers: usize,
    /// Native tablets used when Kivi is spawned.
    pub tablets: usize,
    /// Namespace used by native Kivi targets.
    pub namespace: u64,
    /// Override warmup operations per thread.
    pub warmup: Option<usize>,
}

impl Default for CampaignOptions {
    fn default() -> Self {
        Self {
            preset: CampaignPreset::Smoke,
            redis_url: None,
            kivi_endpoint: None,
            kivi_resp_endpoint: None,
            kivi_admin_endpoint: None,
            output_dir: None,
            trials: None,
            duration: None,
            ops_per_trial: None,
            thread_counts: None,
            pipeline_depths: None,
            value_len: None,
            workload: None,
            include_large: None,
            key_space: None,
            kivi_mode: KiviMode::Durable,
            workers: 2,
            tablets: 1,
            namespace: 1,
            warmup: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
struct KeyPoint {
    label: String,
    space: KeySpace,
}

impl KeyPoint {
    fn hot() -> Self {
        Self {
            label: "hot".to_owned(),
            space: KeySpace::Hot,
        }
    }

    fn small() -> Self {
        Self {
            label: "small".to_owned(),
            space: KeySpace::UniformGlobal(4),
        }
    }

    fn uniform256() -> Self {
        Self {
            label: "uniform256".to_owned(),
            space: KeySpace::UniformGlobal(256),
        }
    }

    fn uniform4k() -> Self {
        Self {
            label: "uniform4k".to_owned(),
            space: KeySpace::UniformGlobal(4096),
        }
    }

    fn large() -> Self {
        Self {
            label: "large".to_owned(),
            space: KeySpace::UniformGlobal(16_384),
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct NamedProfile {
    name: &'static str,
    workload: Workload,
}

impl NamedProfile {
    const fn new(name: &'static str, workload: Workload) -> Self {
        Self { name, workload }
    }
}

#[derive(Debug, Clone)]
struct CampaignProfile {
    trials: usize,
    duration: Duration,
    warmup: usize,
    base_value_len: usize,
    value_sizes: Vec<usize>,
    base_key_space: KeyPoint,
    key_distributions: Vec<KeyPoint>,
    named_profiles: Vec<NamedProfile>,
    native_only_profiles: Vec<NamedProfile>,
    base_threads: usize,
    base_pipeline: usize,
    thread_counts: Vec<usize>,
    pipeline_depths: Vec<usize>,
    curve_workload: Workload,
    large_values: Vec<usize>,
}

/// A separately labeled target class in the campaign.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum TargetClass {
    /// Kivi's RESP frontend against Redis' RESP frontend.
    KiviRespVsRedisResp,
    /// Kivi's native frontend against Redis' RESP frontend.
    KiviNativeVsRedisResp,
    /// Kivi's native frontend against Kivi's RESP frontend.
    KiviNativeVsKiviResp,
}

impl TargetClass {
    /// Human-readable class label used in reports.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::KiviRespVsRedisResp => "Kivi RESP vs Redis RESP",
            Self::KiviNativeVsRedisResp => "Kivi native vs Redis RESP",
            Self::KiviNativeVsKiviResp => "Kivi native vs Kivi RESP",
        }
    }

    fn slug(self) -> &'static str {
        match self {
            Self::KiviRespVsRedisResp => "kivi-resp-redis-resp",
            Self::KiviNativeVsRedisResp => "kivi-native-redis-resp",
            Self::KiviNativeVsKiviResp => "kivi-native-kivi-resp",
        }
    }
}

/// Classifies a paired result without combining target classes.
#[must_use]
#[allow(clippy::too_many_arguments, clippy::cast_precision_loss)]
pub fn classify_comparison(
    left_label: &str,
    right_label: &str,
    left_ops_per_sec: f64,
    right_ops_per_sec: f64,
    left_p95_ns: u64,
    right_p95_ns: u64,
    left_bytes_per_op: Option<f64>,
    right_bytes_per_op: Option<f64>,
) -> String {
    if !left_ops_per_sec.is_finite()
        || !right_ops_per_sec.is_finite()
        || left_ops_per_sec <= 0.0
        || right_ops_per_sec <= 0.0
    {
        return "invalid".to_owned();
    }
    let throughput_delta = (left_ops_per_sec - right_ops_per_sec) / right_ops_per_sec;
    if throughput_delta.abs() <= 0.05 {
        let latency_delta = if right_p95_ns == 0 {
            0.0
        } else {
            (left_p95_ns as f64 - right_p95_ns as f64) / right_p95_ns as f64
        };
        if latency_delta.abs() <= 0.05 {
            return match (left_bytes_per_op, right_bytes_per_op) {
                (Some(left), Some(right)) if right > 0.0 && (left - right).abs() / right > 0.05 => {
                    if left < right {
                        format!("bandwidth:{left_label}")
                    } else {
                        format!("bandwidth:{right_label}")
                    }
                }
                _ => "tied".to_owned(),
            };
        }
        return if left_p95_ns < right_p95_ns {
            format!("latency:{left_label}")
        } else {
            format!("latency:{right_label}")
        };
    }
    if throughput_delta > 0.0 {
        format!("throughput:{left_label}")
    } else {
        format!("throughput:{right_label}")
    }
}

fn scaling_classification(
    left_label: &str,
    right_label: &str,
    left_ops_per_sec_per_thread: f64,
    right_ops_per_sec_per_thread: f64,
) -> Option<String> {
    if !left_ops_per_sec_per_thread.is_finite()
        || !right_ops_per_sec_per_thread.is_finite()
        || left_ops_per_sec_per_thread <= 0.0
        || right_ops_per_sec_per_thread <= 0.0
    {
        return None;
    }
    let delta =
        (left_ops_per_sec_per_thread - right_ops_per_sec_per_thread) / right_ops_per_sec_per_thread;
    if delta.abs() <= 0.05 {
        None
    } else if delta > 0.0 {
        Some(format!("scaling:{left_label}"))
    } else {
        Some(format!("scaling:{right_label}"))
    }
}

const NOISE_THRESHOLD: f64 = 0.05;

fn comparison_verdict(
    left_label: &str,
    right_label: &str,
    left_ops_per_sec: f64,
    right_ops_per_sec: f64,
) -> VerdictReport {
    if !left_ops_per_sec.is_finite()
        || !right_ops_per_sec.is_finite()
        || left_ops_per_sec <= 0.0
        || right_ops_per_sec <= 0.0
    {
        return VerdictReport {
            outcome: "invalid".to_owned(),
            winner: None,
            metric: "throughput".to_owned(),
            relative_delta: 0.0,
            noise_threshold: NOISE_THRESHOLD,
        };
    }
    let relative_delta = (left_ops_per_sec - right_ops_per_sec) / right_ops_per_sec;
    if relative_delta.abs() <= NOISE_THRESHOLD {
        return VerdictReport {
            outcome: if relative_delta == 0.0 {
                "tie".to_owned()
            } else {
                "noise".to_owned()
            },
            winner: None,
            metric: "throughput".to_owned(),
            relative_delta,
            noise_threshold: NOISE_THRESHOLD,
        };
    }
    let left_wins = relative_delta > 0.0;
    VerdictReport {
        outcome: if left_wins {
            "left-win".to_owned()
        } else {
            "right-win".to_owned()
        },
        winner: Some(if left_wins { left_label } else { right_label }.to_owned()),
        metric: "throughput".to_owned(),
        relative_delta,
        noise_threshold: NOISE_THRESHOLD,
    }
}

#[derive(Debug, Serialize)]
struct TrialReport {
    trial: usize,
    target: String,
    target_class: String,
    prefix: String,
    dimension: String,
    profile: String,
    point_label: String,
    value_label: String,
    key_distribution: String,
    workload: String,
    threads: usize,
    pipeline: usize,
    value_len: usize,
    ops_per_sec: f64,
    completed: u64,
    errors: u64,
    warmup_errors: u64,
    expected_completed: Option<u64>,
    p50_ns: u64,
    p95_ns: u64,
    p99_ns: u64,
    p999_ns: u64,
    max_ns: u64,
    batch_p50_ns: u64,
    batch_p95_ns: u64,
    batch_p99_ns: u64,
    batch_p999_ns: u64,
    batch_max_ns: u64,
    bytes_out: Option<u64>,
    bytes_in: Option<u64>,
    bytes_per_sec: Option<f64>,
    payload_bytes_per_op: f64,
    effective_payload_bytes_per_sec: f64,
    target_stats: Option<TargetStats>,
    elapsed_secs: f64,
    duration_secs: f64,
    latency_unit: &'static str,
    valid: bool,
    sample_check: bool,
    sample_checks: usize,
}

#[derive(Debug, Serialize)]
struct AggregateReport {
    trials: usize,
    valid: bool,
    median_ops_per_sec: f64,
    min_ops_per_sec: f64,
    max_ops_per_sec: f64,
    spread_ops_per_sec: f64,
    median_ops_per_sec_per_thread: f64,
    median_p50_ns: u64,
    median_p95_ns: u64,
    median_p99_ns: u64,
    median_p999_ns: u64,
    spread_p50_ns: f64,
    spread_p95_ns: f64,
    spread_p99_ns: f64,
    spread_p999_ns: f64,
    median_bytes_out: Option<u64>,
    median_bytes_in: Option<u64>,
    median_bytes_per_sec: Option<f64>,
    median_effective_payload_bytes_per_sec: f64,
}

#[derive(Debug, Serialize)]
struct RatioReport {
    throughput_ratio: Option<f64>,
    p50_ratio: Option<f64>,
    p95_ratio: Option<f64>,
    p99_ratio: Option<f64>,
    p999_ratio: Option<f64>,
    bytes_out_ratio: Option<f64>,
    bytes_per_sec_ratio: Option<f64>,
    effective_payload_ratio: Option<f64>,
    scaling_ratio: Option<f64>,
    throughput_difference: Option<f64>,
}

#[derive(Debug, Serialize)]
struct VerdictReport {
    outcome: String,
    winner: Option<String>,
    metric: String,
    relative_delta: f64,
    noise_threshold: f64,
}

#[derive(Debug, Serialize)]
struct ComparisonReport {
    class: String,
    dimension: String,
    profile: String,
    point_label: String,
    value_label: String,
    key_distribution: String,
    saturation: String,
    validation: ValidationReport,
    verdict: VerdictReport,
    left_target: String,
    right_target: String,
    left_endpoint_type: String,
    right_endpoint_type: String,
    left_endpoint: String,
    right_endpoint: String,
    workload: String,
    threads: usize,
    pipeline: usize,
    value_len: usize,
    curve: bool,
    valid: bool,
    classification: String,
    left_aggregate: AggregateReport,
    right_aggregate: AggregateReport,
    ratios: RatioReport,
    trials: Vec<TrialReport>,
    caveats: Vec<String>,
    failures: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
struct ValidationReport {
    class: String,
    point_label: String,
    dimension: String,
    profile: String,
    left_target: String,
    right_target: String,
    valid: bool,
    left_checks: usize,
    right_checks: usize,
    detail: String,
    reproducer: Option<String>,
}

#[derive(Debug, Serialize)]
struct LargeValueReport {
    class: String,
    operation: String,
    value_len: usize,
    valid: bool,
    comparison: ComparisonReport,
}

#[derive(Debug, Serialize)]
struct NativeOnlyReport {
    target: String,
    endpoint: String,
    workload: String,
    threads: usize,
    pipeline: usize,
    value_len: usize,
    key_distribution: String,
    valid: bool,
    aggregate: AggregateReport,
    trials: Vec<TrialReport>,
    validation: ValidationReport,
    failures: Vec<String>,
}

#[derive(Debug, Serialize)]
struct MetadataReport {
    system: BTreeMap<String, String>,
    build: BTreeMap<String, String>,
    kivi: BTreeMap<String, String>,
    process: BTreeMap<String, String>,
    lifecycle: BTreeMap<String, String>,
    redis_url: String,
    redis_url_source: String,
    redis: RedisConfig,
}

#[derive(Debug, Serialize)]
struct ProfileReport {
    trials: usize,
    duration_secs: f64,
    fixed_ops_per_thread: Option<usize>,
    warmup: usize,
    base_value_len: usize,
    value_sizes: Vec<String>,
    base_key_distribution: String,
    key_distributions: Vec<String>,
    named_profiles: Vec<String>,
    native_only_profiles: Vec<String>,
    thread_counts: Vec<usize>,
    pipeline_depths: Vec<usize>,
    large_values: Vec<usize>,
    kivi_mode: KiviMode,
}

#[derive(Debug, Serialize)]
#[allow(clippy::struct_excessive_bools)]
struct CoverageReport {
    expected_comparison_points: usize,
    actual_comparison_points: usize,
    expected_trials_per_point: usize,
    expected_native_only_points: usize,
    actual_native_only_points: usize,
    expected_large_value_experiments: usize,
    actual_large_value_experiments: usize,
    comparison_trials_complete: bool,
    native_only_trials_complete: bool,
    large_value_trials_complete: bool,
    complete: bool,
    preset_complete: bool,
}

#[derive(Debug, Serialize)]
struct CampaignDocument {
    schema: &'static str,
    generated_at_unix_ms: u128,
    preset: CampaignPreset,
    profile: ProfileReport,
    coverage: CoverageReport,
    metadata: MetadataReport,
    correctness: Vec<ValidationReport>,
    model_validation: Vec<ValidationReport>,
    comparisons: Vec<ComparisonReport>,
    value_size_points: Vec<ComparisonReport>,
    key_distribution_points: Vec<ComparisonReport>,
    concurrency_points: Vec<ComparisonReport>,
    pipeline_points: Vec<ComparisonReport>,
    large_value_experiments: Vec<LargeValueReport>,
    native_only: Vec<NativeOnlyReport>,
    failures: Vec<String>,
    valid: bool,
    output_dir: PathBuf,
    raw_json: PathBuf,
    markdown: PathBuf,
    markdown_report: String,
}

/// Completed campaign report with raw and Markdown artifacts.
#[derive(Debug, Serialize)]
#[serde(transparent)]
pub struct CampaignReport {
    document: CampaignDocument,
}

impl CampaignReport {
    /// Whether every requested point passed correctness and execution checks.
    #[must_use]
    pub fn valid(&self) -> bool {
        self.document.valid
    }

    /// Failure messages collected during the campaign.
    #[must_use]
    pub fn failures(&self) -> &[String] {
        &self.document.failures
    }

    /// Directory containing the generated artifacts.
    #[must_use]
    pub fn output_dir(&self) -> &Path {
        &self.document.output_dir
    }

    /// Path to the raw JSON document.
    #[must_use]
    pub fn raw_json_path(&self) -> &Path {
        &self.document.raw_json
    }

    /// Path to the Markdown report.
    #[must_use]
    pub fn markdown_path(&self) -> &Path {
        &self.document.markdown
    }

    /// Markdown report text.
    #[must_use]
    pub fn markdown(&self) -> &str {
        &self.document.markdown_report
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TargetKind {
    KiviNative,
    KiviResp,
    RedisResp,
}

impl TargetKind {
    const fn label(self) -> &'static str {
        match self {
            Self::KiviNative => "kivi-native",
            Self::KiviResp => "kivi-resp",
            Self::RedisResp => "redis-resp",
        }
    }
}

#[derive(Debug, Clone)]
struct TargetSpec {
    kind: TargetKind,
    label: &'static str,
    endpoint: String,
}

#[derive(Debug, Clone)]
struct TargetPair {
    class: TargetClass,
    left: TargetSpec,
    right: TargetSpec,
}

#[derive(Debug, Clone, PartialEq)]
struct RunSpec {
    dimension: String,
    profile: String,
    value_label: String,
    key_label: String,
    workload: Workload,
    threads: usize,
    pipeline: usize,
    value_len: usize,
    key_space: KeySpace,
    duration: Duration,
    ops: Option<usize>,
    warmup: usize,
    trials: usize,
    curve: bool,
}

#[derive(Debug, Clone)]
struct SpecSet {
    named: Vec<RunSpec>,
    native_only: Vec<RunSpec>,
    value_sizes: Vec<RunSpec>,
    key_distributions: Vec<RunSpec>,
    concurrency: Vec<RunSpec>,
    pipelines: Vec<RunSpec>,
}

struct KiviLaunch {
    native: Option<String>,
    resp: Option<String>,
    admin: Option<String>,
    server: Option<Server>,
    temp: Option<TempDir>,
    metadata: BTreeMap<String, String>,
}

impl KiviLaunch {
    fn process_diagnostics(&mut self) -> BTreeMap<String, String> {
        self.server
            .as_mut()
            .map_or_else(BTreeMap::new, Server::diagnostics)
    }

    fn admin_diagnostics(&self) -> BTreeMap<String, String> {
        let mut values = BTreeMap::new();
        for path in [
            "/ready",
            "/v1/node",
            "/v1/workers",
            "/v1/tablets",
            "/v1/durability",
            "/v1/redis",
        ] {
            let result = if let Some(server) = &self.server {
                server.try_admin_get(path)
            } else if let Some(endpoint) = &self.admin {
                try_admin_get_endpoint(endpoint, path)
            } else {
                continue;
            };
            match result {
                Ok((status, body)) => {
                    values.insert(format!("{path}.status"), status.to_string());
                    values.insert(format!("{path}.body"), bounded_metadata(&body));
                }
                Err(error) => {
                    values.insert(format!("{path}.error"), bounded_metadata(&error));
                }
            }
        }
        values
    }

    fn temp_path(&self) -> Option<&Path> {
        self.temp.as_ref().map(TempDir::path)
    }
}

/// Runs a campaign and writes `raw.json` and `report.md`.
///
/// The returned report is inspectable even when a requested point fails;
/// callers should reject it unless [`CampaignReport::valid`] is true.
///
/// # Errors
///
/// Returns an error for invalid options, endpoint setup, metadata collection,
/// or report artifact I/O.
#[allow(clippy::too_many_lines, clippy::needless_pass_by_value)]
pub fn run_campaign(options: CampaignOptions) -> anyhow::Result<CampaignReport> {
    validate_options(&options)?;
    let generated_at_unix_ms = unix_millis();
    let stamp = timestamp_stamp(generated_at_unix_ms);
    let output_dir = options
        .output_dir
        .clone()
        .unwrap_or_else(|| PathBuf::from("target/kivi-lab-campaign").join(stamp));
    if output_dir.exists() {
        let mut entries = fs::read_dir(&output_dir).with_context(|| {
            format!("reading campaign output directory {}", output_dir.display())
        })?;
        if entries.next().transpose()?.is_some() {
            anyhow::bail!(
                "campaign output directory is not empty: {}",
                output_dir.display()
            );
        }
    } else {
        fs::create_dir_all(&output_dir).with_context(|| {
            format!(
                "creating campaign output directory {}",
                output_dir.display()
            )
        })?;
    }
    let raw_json = output_dir.join("raw.json");
    let markdown_path = output_dir.join("report.md");

    let redis_url = conformance::redis_url(options.redis_url.as_deref());
    let redis_url_source = if options.redis_url.is_some() {
        "explicit"
    } else if std::env::var(conformance::REDIS_URL_ENV).is_ok() {
        "environment"
    } else {
        "default"
    };
    let redis_report_url = redact_endpoint(&redis_url);
    let mut redis_client = RespClient::connect(&redis_url)
        .with_context(|| format!("connecting to Redis at {redis_report_url}"))?;
    let redis = conformance::redis_config(&mut redis_client).context("reading Redis metadata")?;
    eprintln!("campaign: Redis metadata collected");
    let mut launch = resolve_launch(&options)?;
    eprintln!("campaign: Kivi endpoint ready");
    let pairs = build_pairs(&launch, &redis_url);
    if pairs.is_empty() {
        anyhow::bail!("no comparable Kivi and Redis endpoints are available");
    }
    let sequence = CAMPAIGN_IDS.fetch_add(1, Ordering::Relaxed);
    let run_id = u64::try_from(generated_at_unix_ms)
        .unwrap_or(u64::MAX)
        .wrapping_add(sequence);
    let profile = effective_profile(&options);
    let specs = build_specs(&options, &profile);
    if specs.named.is_empty() {
        anyhow::bail!("campaign has no RESP-supported workload points");
    }
    let mut failures = Vec::new();
    let mut correctness = Vec::new();
    let mut comparisons = Vec::new();
    let mut value_size_points = Vec::new();
    let mut key_distribution_points = Vec::new();
    let mut concurrency_points = Vec::new();
    let mut pipeline_points = Vec::new();

    for pair in &pairs {
        eprintln!("campaign: validating {}", pair.class.label());
        let validation = validate_pair(pair, run_id, options.namespace);
        correctness.push(validation.clone());
        if !validation.valid {
            failures.push(format!("{}: {}", pair.class.label(), validation.detail));
            continue;
        }
        for spec in &specs.named {
            eprintln!("campaign: {} {}", pair.class.label(), spec.dimension);
            let report = execute_pair(pair, spec, run_id, options.namespace);
            if !report.valid {
                failures.extend(
                    report
                        .failures
                        .iter()
                        .map(|failure| format!("{}: {failure}", pair.class.label())),
                );
            }
            comparisons.push(report);
        }
        for spec in &specs.value_sizes {
            eprintln!("campaign: {} value-size", pair.class.label());
            let report = execute_pair(pair, spec, run_id, options.namespace);
            if !report.valid {
                failures.extend(
                    report
                        .failures
                        .iter()
                        .map(|failure| format!("{} value-size: {failure}", pair.class.label())),
                );
            }
            value_size_points.push(report);
        }
        for spec in &specs.key_distributions {
            eprintln!("campaign: {} key-distribution", pair.class.label());
            let report = execute_pair(pair, spec, run_id, options.namespace);
            if !report.valid {
                failures.extend(
                    report.failures.iter().map(|failure| {
                        format!("{} key-distribution: {failure}", pair.class.label())
                    }),
                );
            }
            key_distribution_points.push(report);
        }
        for spec in &specs.concurrency {
            eprintln!("campaign: {} concurrency", pair.class.label());
            let report = execute_pair(pair, spec, run_id, options.namespace);
            if !report.valid {
                failures.extend(
                    report
                        .failures
                        .iter()
                        .map(|failure| format!("{} concurrency: {failure}", pair.class.label())),
                );
            }
            concurrency_points.push(report);
        }
        for spec in &specs.pipelines {
            eprintln!("campaign: {} pipeline", pair.class.label());
            let report = execute_pair(pair, spec, run_id, options.namespace);
            if !report.valid {
                failures.extend(
                    report
                        .failures
                        .iter()
                        .map(|failure| format!("{} pipeline: {failure}", pair.class.label())),
                );
            }
            pipeline_points.push(report);
        }
    }

    let mut native_only = Vec::new();
    if let Some(endpoint) = &launch.native {
        let target = TargetSpec {
            kind: TargetKind::KiviNative,
            label: "Kivi native",
            endpoint: endpoint.clone(),
        };
        for spec in &specs.native_only {
            eprintln!("campaign: Kivi native-only {}", spec.profile);
            let report = execute_native_only(&target, spec, run_id, options.namespace);
            if !report.valid {
                failures.extend(
                    report
                        .failures
                        .iter()
                        .map(|failure| format!("Kivi native-only {}: {failure}", spec.profile)),
                );
            }
            native_only.push(report);
        }
    }

    let large_values = if options.include_large == Some(false) {
        Vec::new()
    } else {
        profile.large_values.clone()
    };
    let mut large_value_experiments = Vec::new();
    for pair in &pairs {
        for value_len in &large_values {
            for (operation, workload) in [("set", Workload::Set), ("get", Workload::Get)] {
                let spec = RunSpec {
                    dimension: "large-value".to_owned(),
                    profile: operation.to_owned(),
                    value_label: value_label(*value_len),
                    key_label: "small".to_owned(),
                    workload,
                    threads: 1,
                    pipeline: 1,
                    value_len: *value_len,
                    key_space: KeySpace::UniformGlobal(1),
                    duration: Duration::from_secs(1),
                    ops: Some(3),
                    warmup: 0,
                    trials: 1,
                    curve: false,
                };
                let comparison = execute_pair(pair, &spec, run_id, options.namespace);
                if !comparison.valid {
                    failures.extend(comparison.failures.iter().map(|failure| {
                        format!(
                            "{} large {} {} bytes: {failure}",
                            pair.class.label(),
                            operation,
                            value_len
                        )
                    }));
                }
                large_value_experiments.push(LargeValueReport {
                    class: pair.class.label().to_owned(),
                    operation: operation.to_owned(),
                    value_len: *value_len,
                    valid: comparison.valid,
                    comparison,
                });
            }
        }
    }

    let mut model_validation = Vec::new();
    for report in comparisons
        .iter()
        .chain(&value_size_points)
        .chain(&key_distribution_points)
        .chain(&concurrency_points)
        .chain(&pipeline_points)
    {
        model_validation.push(report.validation.clone());
    }
    model_validation.extend(
        large_value_experiments
            .iter()
            .map(|entry| entry.comparison.validation.clone()),
    );
    model_validation.extend(native_only.iter().map(|entry| entry.validation.clone()));
    let mut kivi_metadata = launch.metadata.clone();
    if let Some(admin) = &launch.admin {
        kivi_metadata
            .entry("admin_endpoint".to_owned())
            .or_insert_with(|| admin.clone());
    }
    kivi_metadata.extend(launch.admin_diagnostics());
    let mut process = launch.process_diagnostics();
    let process_pid = process.get("pid").cloned();
    if let Some(alive) = process.remove("alive") {
        process.insert("alive_before_drop".to_owned(), alive);
    }
    let temp_path = launch.temp_path().map(Path::to_path_buf);
    let managed_server = launch.server.is_some();
    let has_native_endpoint = launch.native.is_some();
    drop(launch);
    if let Some(pid) = process_pid {
        process.insert(
            "alive_after_drop".to_owned(),
            process_is_alive(&pid).to_string(),
        );
    } else {
        process.insert("alive_after_drop".to_owned(), "not-applicable".to_owned());
    }
    let mut lifecycle = BTreeMap::new();
    lifecycle.insert("managed_server".to_owned(), managed_server.to_string());
    lifecycle.insert("shutdown_mode".to_owned(), "kill".to_owned());
    if let Some(path) = &temp_path {
        let removed = !path.exists();
        lifecycle.insert("temp_dir".to_owned(), path.display().to_string());
        lifecycle.insert("temp_dir_removed".to_owned(), removed.to_string());
        if !removed {
            failures.push(format!(
                "temporary Kivi data directory was not removed: {}",
                path.display()
            ));
        }
    } else {
        lifecycle.insert("temp_dir_removed".to_owned(), "not-applicable".to_owned());
    }
    let metadata = MetadataReport {
        system: system_metadata(),
        build: build_metadata(),
        kivi: kivi_metadata,
        process,
        lifecycle,
        redis_url: redis_report_url,
        redis_url_source: redis_url_source.to_owned(),
        redis,
    };
    let comparison_reports = comparisons
        .iter()
        .chain(&value_size_points)
        .chain(&key_distribution_points)
        .chain(&concurrency_points)
        .chain(&pipeline_points)
        .collect::<Vec<_>>();
    let expected_comparison_points = pairs.len()
        * (specs.named.len()
            + specs.value_sizes.len()
            + specs.key_distributions.len()
            + specs.concurrency.len()
            + specs.pipelines.len());
    let actual_comparison_points = comparison_reports.len();
    let expected_native_only_points = if has_native_endpoint {
        specs.native_only.len()
    } else {
        0
    };
    let expected_large_value_experiments = large_values.len() * pairs.len() * 2;
    let comparison_trials_complete = comparison_reports
        .iter()
        .all(|entry| entry.trials.len() == profile.trials.saturating_mul(2));
    let native_only_trials_complete = native_only
        .iter()
        .all(|entry| entry.trials.len() == profile.trials);
    let large_value_trials_complete = large_value_experiments
        .iter()
        .all(|entry| entry.comparison.trials.len() == 2);
    let coverage = CoverageReport {
        expected_comparison_points,
        actual_comparison_points,
        expected_trials_per_point: profile.trials,
        expected_native_only_points,
        actual_native_only_points: native_only.len(),
        expected_large_value_experiments,
        actual_large_value_experiments: large_value_experiments.len(),
        comparison_trials_complete,
        native_only_trials_complete,
        large_value_trials_complete,
        complete: actual_comparison_points == expected_comparison_points
            && native_only.len() == expected_native_only_points
            && large_value_experiments.len() == expected_large_value_experiments
            && comparison_trials_complete
            && native_only_trials_complete
            && large_value_trials_complete,
        preset_complete: actual_comparison_points == expected_comparison_points
            && native_only.len() == expected_native_only_points
            && large_value_experiments.len() == expected_large_value_experiments
            && comparison_trials_complete
            && native_only_trials_complete
            && large_value_trials_complete
            && options.trials.is_none()
            && options.duration.is_none()
            && options.ops_per_trial.is_none()
            && options.warmup.is_none()
            && options.thread_counts.is_none()
            && options.pipeline_depths.is_none()
            && options.value_len.is_none()
            && options.workload.is_none()
            && options.key_space.is_none()
            && options.include_large != Some(false),
    };
    let profile_report = ProfileReport {
        trials: profile.trials,
        duration_secs: profile.duration.as_secs_f64(),
        fixed_ops_per_thread: options.ops_per_trial,
        warmup: profile.warmup,
        base_value_len: profile.base_value_len,
        value_sizes: profile
            .value_sizes
            .iter()
            .copied()
            .map(value_label)
            .collect(),
        base_key_distribution: profile.base_key_space.label.clone(),
        key_distributions: profile
            .key_distributions
            .iter()
            .map(|point| point.label.clone())
            .collect(),
        named_profiles: profile
            .named_profiles
            .iter()
            .map(|profile| profile.name.to_owned())
            .collect(),
        native_only_profiles: profile
            .native_only_profiles
            .iter()
            .map(|profile| profile.name.to_owned())
            .collect(),
        thread_counts: profile.thread_counts.clone(),
        pipeline_depths: profile.pipeline_depths.clone(),
        large_values,
        kivi_mode: options.kivi_mode,
    };
    let valid = failures.is_empty()
        && coverage.complete
        && !comparisons.is_empty()
        && !value_size_points.is_empty()
        && !key_distribution_points.is_empty()
        && !concurrency_points.is_empty()
        && !pipeline_points.is_empty()
        && correctness.iter().all(|entry| entry.valid)
        && model_validation.iter().all(|entry| entry.valid)
        && large_value_experiments.iter().all(|entry| entry.valid)
        && native_only.iter().all(|entry| entry.valid);
    let mut document = CampaignDocument {
        schema: CAMPAIGN_SCHEMA,
        generated_at_unix_ms,
        preset: options.preset,
        profile: profile_report,
        coverage,
        metadata,
        correctness,
        model_validation,
        comparisons,
        value_size_points,
        key_distribution_points,
        concurrency_points,
        pipeline_points,
        large_value_experiments,
        native_only,
        failures,
        valid,
        output_dir: output_dir.clone(),
        raw_json: raw_json.clone(),
        markdown: markdown_path.clone(),
        markdown_report: String::new(),
    };
    document.markdown_report = render_markdown(&document);
    let serialized = serde_json::to_string_pretty(&document)?;
    let raw_tmp = output_dir.join("raw.json.tmp");
    let markdown_tmp = output_dir.join("report.md.tmp");
    fs::write(&raw_tmp, format!("{serialized}\n"))
        .with_context(|| format!("writing {}", raw_tmp.display()))?;
    fs::write(&markdown_tmp, format!("{}\n", document.markdown_report))
        .with_context(|| format!("writing {}", markdown_tmp.display()))?;
    fs::rename(&raw_tmp, &raw_json)
        .with_context(|| format!("publishing {}", raw_json.display()))?;
    fs::rename(&markdown_tmp, &markdown_path)
        .with_context(|| format!("publishing {}", markdown_path.display()))?;
    Ok(CampaignReport { document })
}

fn validate_options(options: &CampaignOptions) -> anyhow::Result<()> {
    if options.workers == 0 || options.tablets == 0 {
        anyhow::bail!("workers and tablets must be nonzero");
    }
    if options.trials.is_some_and(|value| value == 0) {
        anyhow::bail!("trials must be nonzero");
    }
    if options.duration.is_some_and(|duration| duration.is_zero()) {
        anyhow::bail!("duration must be nonzero");
    }
    if options.ops_per_trial.is_some_and(|value| value == 0) {
        anyhow::bail!("ops per trial must be nonzero");
    }
    if let Some(values) = &options.thread_counts
        && (values.is_empty() || values.contains(&0))
    {
        anyhow::bail!("thread counts must be nonzero and nonempty");
    }
    if let Some(values) = &options.pipeline_depths
        && (values.is_empty() || values.contains(&0))
    {
        anyhow::bail!("pipeline depths must be nonzero and nonempty");
    }
    if options.value_len.is_some_and(|value| value == 0) {
        anyhow::bail!("value length must be nonzero");
    }
    if matches!(
        options.key_space,
        Some(KeySpace::Uniform(0) | KeySpace::UniformGlobal(0))
    ) {
        anyhow::bail!("uniform key-space count must be nonzero");
    }
    Ok(())
}

fn resolve_launch(options: &CampaignOptions) -> anyhow::Result<KiviLaunch> {
    if options.kivi_endpoint.is_none() && options.kivi_resp_endpoint.is_none() {
        let temp = tempfile::tempdir().context("creating temporary Kivi data directory")?;
        let binary = release_server_binary_path()?;
        let mode = match options.kivi_mode {
            KiviMode::Ephemeral => ServerMode::Ephemeral,
            KiviMode::Durable => ServerMode::DataDir(temp.path().to_owned()),
        };
        let server =
            Server::spawn_with_binary(&binary, mode, options.workers, options.tablets, &[], true)
                .context("spawning kivi-server with RESP support")?;
        let native = server.endpoint();
        let resp = server
            .resp_endpoint()
            .context("spawned Kivi has no RESP endpoint")?;
        let mut metadata = BTreeMap::new();
        metadata.insert(
            "mode".to_owned(),
            match options.kivi_mode {
                KiviMode::Ephemeral => "spawned-temporary-ephemeral".to_owned(),
                KiviMode::Durable => "spawned-temporary-durable".to_owned(),
            },
        );
        metadata.insert("data_dir".to_owned(), temp.path().display().to_string());
        metadata.insert("native_endpoint".to_owned(), redact_endpoint(&native));
        metadata.insert("resp_endpoint".to_owned(), redact_endpoint(&resp));
        let admin = server.admin_endpoint();
        metadata.insert("admin_endpoint".to_owned(), admin.clone());
        metadata.insert("pid".to_owned(), server.pid().to_string());
        metadata.insert("binary".to_owned(), binary.display().to_string());
        metadata.insert(
            "binary_profile".to_owned(),
            if binary.to_string_lossy().contains("release") {
                "release".to_owned()
            } else {
                build_profile().to_owned()
            },
        );
        metadata.insert("workers".to_owned(), options.workers.to_string());
        metadata.insert("tablets".to_owned(), options.tablets.to_string());
        metadata.insert("namespace".to_owned(), options.namespace.to_string());
        return Ok(KiviLaunch {
            native: Some(native),
            resp: Some(resp),
            admin: Some(admin),
            server: Some(server),
            temp: Some(temp),
            metadata,
        });
    }
    let mut metadata = BTreeMap::new();
    metadata.insert("mode".to_owned(), "external".to_owned());
    if let Some(endpoint) = &options.kivi_endpoint {
        metadata.insert("native_endpoint".to_owned(), redact_endpoint(endpoint));
    }
    if let Some(endpoint) = &options.kivi_resp_endpoint {
        metadata.insert("resp_endpoint".to_owned(), redact_endpoint(endpoint));
    }
    if let Some(endpoint) = &options.kivi_admin_endpoint {
        metadata.insert("admin_endpoint".to_owned(), redact_endpoint(endpoint));
    }
    Ok(KiviLaunch {
        native: options.kivi_endpoint.clone(),
        resp: options.kivi_resp_endpoint.clone(),
        admin: options.kivi_admin_endpoint.clone(),
        server: None,
        temp: None,
        metadata,
    })
}

fn build_pairs(launch: &KiviLaunch, redis_url: &str) -> Vec<TargetPair> {
    let redis = TargetSpec {
        kind: TargetKind::RedisResp,
        label: "Redis RESP",
        endpoint: redis_url.to_owned(),
    };
    let mut pairs = Vec::new();
    if let Some(endpoint) = &launch.resp {
        pairs.push(TargetPair {
            class: TargetClass::KiviRespVsRedisResp,
            left: TargetSpec {
                kind: TargetKind::KiviResp,
                label: "Kivi RESP",
                endpoint: endpoint.clone(),
            },
            right: redis.clone(),
        });
    }
    if let Some(endpoint) = &launch.native {
        pairs.push(TargetPair {
            class: TargetClass::KiviNativeVsRedisResp,
            left: TargetSpec {
                kind: TargetKind::KiviNative,
                label: "Kivi native",
                endpoint: endpoint.clone(),
            },
            right: redis.clone(),
        });
        if let Some(resp_endpoint) = &launch.resp {
            pairs.push(TargetPair {
                class: TargetClass::KiviNativeVsKiviResp,
                left: TargetSpec {
                    kind: TargetKind::KiviNative,
                    label: "Kivi native",
                    endpoint: endpoint.clone(),
                },
                right: TargetSpec {
                    kind: TargetKind::KiviResp,
                    label: "Kivi RESP",
                    endpoint: resp_endpoint.clone(),
                },
            });
        }
    }
    pairs
}

fn effective_profile(options: &CampaignOptions) -> CampaignProfile {
    let mut profile = options.preset.profile();
    if let Some(trials) = options.trials {
        profile.trials = trials;
    }
    if let Some(duration) = options.duration {
        profile.duration = duration;
    }
    if let Some(warmup) = options.warmup {
        profile.warmup = warmup;
    }
    if let Some(value_len) = options.value_len {
        profile.base_value_len = value_len;
        profile.value_sizes = vec![value_len];
    }
    if let Some(workload) = options.workload {
        profile.named_profiles = vec![NamedProfile::new(profile_name(workload), workload)];
        profile.curve_workload = workload;
    }
    if let Some(threads) = &options.thread_counts {
        profile.thread_counts.clone_from(threads);
        profile.base_threads = threads[0];
    }
    if let Some(pipelines) = &options.pipeline_depths {
        profile.pipeline_depths.clone_from(pipelines);
        profile.base_pipeline = pipelines[0];
    }
    if let Some(space) = options.key_space {
        profile.base_key_space = KeyPoint {
            label: format!("custom:{}", key_space_label(space)),
            space,
        };
        profile.key_distributions = vec![profile.base_key_space.clone()];
    }
    profile
}

#[allow(clippy::too_many_lines)]
fn build_specs(options: &CampaignOptions, profile: &CampaignProfile) -> SpecSet {
    let make_spec = |dimension: &str,
                     name: &str,
                     value_len: usize,
                     key_point: &KeyPoint,
                     workload: Workload,
                     threads: usize,
                     pipeline: usize,
                     curve: bool| RunSpec {
        dimension: dimension.to_owned(),
        profile: name.to_owned(),
        value_label: value_label(value_len),
        key_label: key_point.label.clone(),
        workload,
        threads,
        pipeline,
        value_len,
        key_space: key_point.space,
        duration: profile.duration,
        ops: options.ops_per_trial,
        warmup: options.warmup.unwrap_or(profile.warmup),
        trials: profile.trials,
        curve,
    };
    let named = profile
        .named_profiles
        .iter()
        .filter(|profile| workload_supports_resp(profile.workload))
        .map(|named| {
            make_spec(
                "profile",
                named.name,
                profile.base_value_len,
                &profile.base_key_space,
                named.workload,
                profile.base_threads,
                profile.base_pipeline,
                false,
            )
        })
        .collect();
    let native_only = profile
        .native_only_profiles
        .iter()
        .map(|named| {
            make_spec(
                "native-only",
                named.name,
                profile.base_value_len,
                &profile.base_key_space,
                named.workload,
                profile.base_threads,
                profile.base_pipeline,
                false,
            )
        })
        .collect();
    let value_sizes = profile
        .value_sizes
        .iter()
        .map(|value_len| {
            make_spec(
                "value-size",
                "balanced-sweep",
                *value_len,
                &profile.base_key_space,
                profile.curve_workload,
                profile.base_threads,
                profile.base_pipeline,
                true,
            )
        })
        .collect();
    let key_distributions = profile
        .key_distributions
        .iter()
        .map(|key_point| {
            make_spec(
                "key-distribution",
                "balanced-sweep",
                profile.base_value_len,
                key_point,
                profile.curve_workload,
                profile.base_threads,
                profile.base_pipeline,
                true,
            )
        })
        .collect();
    let concurrency = profile
        .thread_counts
        .iter()
        .map(|threads| {
            make_spec(
                "concurrency",
                "balanced-sweep",
                profile.base_value_len,
                &profile.base_key_space,
                profile.curve_workload,
                *threads,
                profile.base_pipeline,
                true,
            )
        })
        .collect();
    let pipelines = profile
        .pipeline_depths
        .iter()
        .map(|pipeline| {
            make_spec(
                "pipeline",
                "balanced-sweep",
                profile.base_value_len,
                &profile.base_key_space,
                profile.curve_workload,
                profile.base_threads,
                *pipeline,
                true,
            )
        })
        .collect();
    SpecSet {
        named,
        native_only,
        value_sizes,
        key_distributions,
        concurrency,
        pipelines,
    }
}

fn execute_native_only(
    target: &TargetSpec,
    spec: &RunSpec,
    run_id: u64,
    namespace: u64,
) -> NativeOnlyReport {
    let prefix = format!(
        "kivi-lab:campaign:{}:{run_id}:kivi-native-only:{}:model:",
        std::process::id(),
        spec.profile.replace('-', "_")
    );
    let validation_result = validate_model_endpoint(target, spec, &prefix, namespace);
    let (valid, detail, reproducer, checks) = match &validation_result {
        Ok(checks) => (
            true,
            "deterministic model/state checks passed".to_owned(),
            None,
            *checks,
        ),
        Err(error) => (false, error.clone(), reproducer_from_error(error), 0),
    };
    let mut validation = ValidationReport {
        class: "Kivi native semantic".to_owned(),
        point_label: point_label(spec),
        dimension: spec.dimension.clone(),
        profile: spec.profile.clone(),
        left_target: target.label.to_owned(),
        right_target: "native model".to_owned(),
        valid,
        left_checks: checks,
        right_checks: checks,
        detail,
        reproducer: reproducer.clone(),
    };
    let mut failures = Vec::new();
    if !valid {
        failures.push(format!(
            "validation failed: {}{}",
            validation.detail,
            reproducer
                .as_ref()
                .map_or_else(String::new, |value| format!("; {value}"))
        ));
    }
    let mut trials = Vec::new();
    if valid {
        for trial in 0..spec.trials {
            match run_trial(
                target,
                "Kivi native semantic",
                "kivi-native-only",
                spec,
                run_id,
                trial,
                namespace,
            ) {
                Ok(report) => trials.push(report),
                Err(error) => {
                    validation.valid = false;
                    failures.push(format!("trial {trial}: {error}"));
                }
            }
        }
    }
    let aggregate = aggregate_reports(&trials);
    let valid = valid
        && validation.valid
        && failures.is_empty()
        && trials.len() == spec.trials
        && trials.iter().all(|trial| trial.valid);
    NativeOnlyReport {
        target: target.label.to_owned(),
        endpoint: redact_endpoint(&target.endpoint),
        workload: workload_label(spec.workload),
        threads: spec.threads,
        pipeline: spec.pipeline,
        value_len: spec.value_len,
        key_distribution: spec.key_label.clone(),
        valid,
        aggregate,
        trials,
        validation,
        failures,
    }
}

#[allow(clippy::too_many_lines)]
fn execute_pair(
    pair: &TargetPair,
    spec: &RunSpec,
    run_id: u64,
    namespace: u64,
) -> ComparisonReport {
    let mut validation = validate_spec_pair(pair, spec, run_id, namespace);
    let mut left_trials = Vec::new();
    let mut right_trials = Vec::new();
    let mut failures = Vec::new();
    if !validation.valid {
        failures.push(format!(
            "validation failed: {}{}",
            validation.detail,
            validation
                .reproducer
                .as_ref()
                .map_or_else(String::new, |reproducer| format!("; {reproducer}"))
        ));
    }
    for trial in 0..approved_trial_count(&validation, spec) {
        let order = if trial % 2 == 0 {
            [&pair.left, &pair.right]
        } else {
            [&pair.right, &pair.left]
        };
        for target in order {
            match run_trial(
                target,
                pair.class.label(),
                pair.class.slug(),
                spec,
                run_id,
                trial,
                namespace,
            ) {
                Ok(report) => {
                    if target.label == pair.left.label {
                        left_trials.push(report);
                    } else {
                        right_trials.push(report);
                    }
                }
                Err(error) => {
                    let error = error.to_string();
                    validation.valid = false;
                    validation.detail = format!("{}; post-run: {error}", validation.detail);
                    if validation.reproducer.is_none() {
                        validation.reproducer = reproducer_from_error(&error);
                    }
                    failures.push(format!("trial {trial} {}: {error}", target.label));
                }
            }
        }
    }
    let left_aggregate = aggregate_reports(&left_trials);
    let right_aggregate = aggregate_reports(&right_trials);
    let bytes_out_ratio = match (
        left_aggregate.median_bytes_out,
        right_aggregate.median_bytes_out,
    ) {
        (Some(left), Some(right)) => ratio_u64(left, right),
        _ => None,
    };
    let ratios = RatioReport {
        throughput_ratio: ratio(
            left_aggregate.median_ops_per_sec,
            right_aggregate.median_ops_per_sec,
        ),
        p50_ratio: ratio_u64(left_aggregate.median_p50_ns, right_aggregate.median_p50_ns),
        p95_ratio: ratio_u64(left_aggregate.median_p95_ns, right_aggregate.median_p95_ns),
        p99_ratio: ratio_u64(left_aggregate.median_p99_ns, right_aggregate.median_p99_ns),
        p999_ratio: ratio_u64(
            left_aggregate.median_p999_ns,
            right_aggregate.median_p999_ns,
        ),
        bytes_out_ratio,
        bytes_per_sec_ratio: ratio_optional(
            left_aggregate.median_bytes_per_sec,
            right_aggregate.median_bytes_per_sec,
        ),
        effective_payload_ratio: ratio(
            left_aggregate.median_effective_payload_bytes_per_sec,
            right_aggregate.median_effective_payload_bytes_per_sec,
        ),
        scaling_ratio: ratio(
            left_aggregate.median_ops_per_sec_per_thread,
            right_aggregate.median_ops_per_sec_per_thread,
        ),
        throughput_difference: (left_aggregate.median_ops_per_sec
            - right_aggregate.median_ops_per_sec)
            .is_finite()
            .then_some(left_aggregate.median_ops_per_sec - right_aggregate.median_ops_per_sec),
    };
    let left_bytes = left_trials.iter().find_map(bytes_per_op);
    let right_bytes = right_trials.iter().find_map(bytes_per_op);
    let base_classification = classify_comparison(
        pair.left.label,
        pair.right.label,
        left_aggregate.median_ops_per_sec,
        right_aggregate.median_ops_per_sec,
        left_aggregate.median_p95_ns,
        right_aggregate.median_p95_ns,
        left_bytes,
        right_bytes,
    );
    let classification = if base_classification == "tied" {
        scaling_classification(
            pair.left.label,
            pair.right.label,
            left_aggregate.median_ops_per_sec_per_thread,
            right_aggregate.median_ops_per_sec_per_thread,
        )
        .unwrap_or(base_classification)
    } else {
        base_classification
    };
    let mut verdict = comparison_verdict(
        pair.left.label,
        pair.right.label,
        left_aggregate.median_ops_per_sec,
        right_aggregate.median_ops_per_sec,
    );
    let mut caveats = Vec::new();
    if spec.ops.is_none() {
        caveats.push("duration mode: completed operation counts may differ".to_owned());
        caveats.push(
            "duration mode: sampled post-run state checks are diagnostic, not an exact concurrent replay"
                .to_owned(),
        );
    }
    if spec.pipeline > 1 {
        caveats.push(
            "pipeline p50/p95/p99 values divide batch latency by configured depth; use batch_* values for round-trip latency"
                .to_owned(),
        );
    }
    caveats.push(
        "effective payload throughput uses configured logical payload sizes, not observed response bytes"
            .to_owned(),
    );
    if spec.value_len >= 64 * 1024 * 1024 {
        caveats.push("64 MiB payload may exceed a frontend frame limit".to_owned());
    }
    if left_aggregate.median_bytes_out.is_none() || right_aggregate.median_bytes_out.is_none() {
        caveats.push("transport byte accounting is unavailable for one side".to_owned());
    }
    if matches!(pair.left.kind, TargetKind::KiviNative)
        || matches!(pair.right.kind, TargetKind::KiviNative)
    {
        caveats.push(
            "frontend implementations do not expose identical transport byte counters".to_owned(),
        );
    }
    if left_trials.iter().any(|trial| trial.errors > 0)
        || right_trials.iter().any(|trial| trial.errors > 0)
    {
        caveats.push("one or more measured operations returned errors".to_owned());
    }
    let valid = validation.valid
        && failures.is_empty()
        && left_trials.len() == spec.trials
        && right_trials.len() == spec.trials
        && left_trials.iter().all(|trial| trial.valid)
        && right_trials.iter().all(|trial| trial.valid);
    if !valid && failures.is_empty() {
        failures.push("one or more measured operations or sample checks failed".to_owned());
    }
    let classification = if valid {
        classification
    } else {
        "invalid".clone_into(&mut verdict.outcome);
        verdict.winner = None;
        "invalid".to_owned()
    };
    let mut trials = left_trials;
    trials.extend(right_trials);
    ComparisonReport {
        class: pair.class.label().to_owned(),
        dimension: spec.dimension.clone(),
        profile: spec.profile.clone(),
        point_label: point_label(spec),
        value_label: spec.value_label.clone(),
        key_distribution: spec.key_label.clone(),
        saturation: saturation_label(spec),
        validation,
        verdict,
        left_target: pair.left.label.to_owned(),
        right_target: pair.right.label.to_owned(),
        left_endpoint_type: pair.left.kind.label().to_owned(),
        right_endpoint_type: pair.right.kind.label().to_owned(),
        left_endpoint: redact_endpoint(&pair.left.endpoint),
        right_endpoint: redact_endpoint(&pair.right.endpoint),
        workload: workload_label(spec.workload),
        threads: spec.threads,
        pipeline: spec.pipeline,
        value_len: spec.value_len,
        curve: spec.curve,
        valid,
        classification,
        left_aggregate,
        right_aggregate,
        ratios,
        trials,
        caveats,
        failures,
    }
}

fn approved_trial_count(validation: &ValidationReport, spec: &RunSpec) -> usize {
    if validation.valid { spec.trials } else { 0 }
}

#[allow(
    clippy::too_many_lines,
    clippy::cast_precision_loss,
    clippy::similar_names
)]
fn run_trial(
    target: &TargetSpec,
    class_label: &str,
    class_slug: &str,
    spec: &RunSpec,
    run_id: u64,
    trial: usize,
    namespace: u64,
) -> anyhow::Result<TrialReport> {
    let prefix = format!(
        "kivi-lab:campaign:{}:{run_id}:{class_slug}:{}:{}:v{}:k{}:t{}:p{}:trial{trial}:",
        std::process::id(),
        spec.dimension.replace('-', "_"),
        spec.profile.replace('-', "_"),
        spec.value_len,
        spec.key_label.replace('-', "_"),
        spec.threads,
        spec.pipeline
    );
    let probe = format!("{prefix}probe");
    let probe_payload = PayloadCache::new(128);
    let state_payload = PayloadCache::new(spec.value_len);
    let result: anyhow::Result<(RunStats, usize)> = (|| match target.kind {
        TargetKind::KiviNative => {
            let client = native_client(&target.endpoint, namespace)?;
            let mut seeder = KiviNativeTarget::connect(&target.endpoint, namespace, 256)?;
            runner::seed(
                &mut seeder,
                spec.key_space,
                spec.threads,
                spec.workload,
                spec.value_len,
                &prefix,
            )?;
            client.set(
                &Key::from(probe.as_str()),
                Bytes::copy_from_slice(probe_payload.as_bytes()),
            )?;
            let stats = run_measurement(spec, target, namespace, &prefix)?;
            check_native_probe(&client, &probe, probe_payload.as_bytes())
                .map_err(anyhow::Error::msg)?;
            let sample_checks = verify_measured_state(
                target,
                spec,
                &prefix,
                namespace,
                &stats,
                state_payload.as_bytes(),
            )
            .map_err(anyhow::Error::msg)?;
            Ok((stats, sample_checks))
        }
        TargetKind::KiviResp | TargetKind::RedisResp => {
            let mut seeder = RespTarget::lazy(&target.endpoint);
            runner::seed(
                &mut seeder,
                spec.key_space,
                spec.threads,
                spec.workload,
                spec.value_len,
                &prefix,
            )?;
            let mut client = RespClient::connect(&target.endpoint)?;
            set_resp_probe(&mut client, &probe, probe_payload.as_bytes())
                .map_err(anyhow::Error::msg)?;
            let stats = run_measurement(spec, target, namespace, &prefix)?;
            check_resp_probe(&mut client, &probe, probe_payload.as_bytes())
                .map_err(anyhow::Error::msg)?;
            let sample_checks = verify_measured_state(
                target,
                spec,
                &prefix,
                namespace,
                &stats,
                state_payload.as_bytes(),
            )
            .map_err(anyhow::Error::msg)?;
            Ok((stats, sample_checks))
        }
    })();
    let cleanup: anyhow::Result<()> = match target.kind {
        TargetKind::KiviNative => {
            let client = native_client(&target.endpoint, namespace)?;
            cleanup_native_owned(&client, spec, &prefix, &probe).map_err(anyhow::Error::msg)
        }
        TargetKind::KiviResp | TargetKind::RedisResp => {
            let mut client = RespClient::connect(&target.endpoint)?;
            cleanup_resp_owned(&mut client, spec, &prefix, &probe).map_err(anyhow::Error::msg)
        }
    };
    let (stats, sample_checks) = match (result, cleanup) {
        (Ok(result), Ok(())) => result,
        (Err(error), Ok(())) | (Ok(_), Err(error)) => return Err(error),
        (Err(error), Err(cleanup_error)) => {
            return Err(anyhow::anyhow!("{error}; cleanup failed: {cleanup_error}"));
        }
    };
    let (bytes_out, bytes_in) = stats
        .bytes
        .map_or((None, None), |(out, input)| (Some(out), Some(input)));
    let expected_completed = spec
        .ops
        .map(|ops| u64::try_from(ops.saturating_mul(spec.threads)).unwrap_or(u64::MAX));
    let batch_p50_ns = stats.histogram.value_at_quantile(0.50);
    let batch_p95_ns = stats.histogram.value_at_quantile(0.95);
    let batch_p99_value = stats.histogram.value_at_quantile(0.99);
    let batch_p999_value = stats.histogram.value_at_quantile(0.999);
    let batch_max_ns = stats.histogram.max();
    let p50_normalized = normalize_latency(batch_p50_ns, spec.pipeline);
    let p95_normalized = normalize_latency(batch_p95_ns, spec.pipeline);
    let p99_normalized = normalize_latency(batch_p99_value, spec.pipeline);
    let p999_normalized = normalize_latency(batch_p999_value, spec.pipeline);
    let max_normalized = normalize_latency(batch_max_ns, spec.pipeline);
    let payload_bytes_per_op = payload_bytes_per_op(spec.workload, spec.value_len);
    let bytes_per_sec = stats
        .bytes
        .map(|(out, input)| (out.saturating_add(input)) as f64 / stats.secs.max(f64::MIN_POSITIVE));
    let effective_payload_bytes_per_sec =
        stats.completed as f64 * payload_bytes_per_op / stats.secs.max(f64::MIN_POSITIVE);
    Ok(TrialReport {
        trial,
        target: target.label.to_owned(),
        target_class: class_label.to_owned(),
        prefix: prefix.clone(),
        dimension: spec.dimension.clone(),
        profile: spec.profile.clone(),
        point_label: point_label(spec),
        value_label: spec.value_label.clone(),
        key_distribution: spec.key_label.clone(),
        workload: workload_label(spec.workload),
        threads: spec.threads,
        pipeline: spec.pipeline,
        value_len: spec.value_len,
        ops_per_sec: completed_rate(stats.completed, stats.secs),
        completed: stats.completed,
        errors: stats.errors,
        warmup_errors: stats.warmup_errors,
        expected_completed,
        p50_ns: p50_normalized,
        p95_ns: p95_normalized,
        p99_ns: p99_normalized,
        p999_ns: p999_normalized,
        max_ns: max_normalized,
        batch_p50_ns,
        batch_p95_ns,
        batch_p99_ns: batch_p99_value,
        batch_p999_ns: batch_p999_value,
        batch_max_ns,
        bytes_out,
        bytes_in,
        bytes_per_sec,
        payload_bytes_per_op,
        effective_payload_bytes_per_sec,
        target_stats: stats.target_stats,
        elapsed_secs: stats.secs,
        duration_secs: spec.duration.as_secs_f64(),
        latency_unit: if spec.pipeline == 1 {
            "ns/op"
        } else {
            "ns/batch-equivalent"
        },
        valid: stats.errors == 0
            && stats.warmup_errors == 0
            && stats.completed > 0
            && sample_checks > 0
            && expected_completed.is_none_or(|expected| stats.completed == expected),
        sample_check: sample_checks > 0,
        sample_checks,
    })
}

fn measured_model(
    spec: &RunSpec,
    prefix: &str,
    payload: &[u8],
    completed_per_thread: &[u64],
) -> ModelState {
    let mut state = ModelState::default();
    for thread in 0..spec.threads {
        for op in crate::workload::seed_ops_for(
            spec.key_space,
            thread,
            spec.workload,
            spec.value_len,
            prefix,
        ) {
            if let WorkloadOp::CounterAdd(key) = op {
                state.counters.insert(key, 0);
            } else {
                model_update(&op, &mut state, payload);
            }
        }
    }
    for thread in 0..spec.threads {
        let keys = thread_key_names(spec.key_space, thread, prefix);
        let counter = counter_key_name(spec.key_space, thread, prefix);
        for index in 0..spec.warmup {
            let op = workload_op_for(spec.workload, spec.value_len, index, &keys, &counter);
            model_update(&op, &mut state, payload);
        }
        let completed = completed_per_thread
            .get(thread)
            .copied()
            .unwrap_or_default();
        let completed = usize::try_from(completed).unwrap_or(usize::MAX);
        let (start, count) = if spec.ops.is_some() {
            (0, completed)
        } else {
            let replay_cap = if spec.value_len >= 1024 * 1024 {
                8
            } else if spec.value_len >= 64 * 1024 {
                64
            } else {
                512
            };
            let count = completed.min(replay_cap);
            (completed.saturating_sub(count), count)
        };
        for index in start..start.saturating_add(count) {
            let op = workload_op_for(spec.workload, spec.value_len, index, &keys, &counter);
            model_update(&op, &mut state, payload);
        }
    }
    state
}

fn measured_keys(spec: &RunSpec, prefix: &str) -> BTreeSet<String> {
    let mut keys = BTreeSet::new();
    for thread in 0..spec.threads {
        keys.extend(thread_key_names(spec.key_space, thread, prefix));
        if matches!(spec.workload, Workload::Counter | Workload::Mixed) {
            keys.insert(counter_key_name(spec.key_space, thread, prefix));
        }
    }
    keys
}

fn sampled_keys(keys: &BTreeSet<String>, value_len: usize) -> Vec<String> {
    const MAX_KEYS: usize = 64;
    const MAX_BYTES: usize = 4 * 1024 * 1024;
    let max_keys = value_len
        .max(1)
        .saturating_add(1)
        .min(MAX_BYTES / value_len.max(1));
    let max_keys = max_keys.clamp(1, MAX_KEYS);
    if keys.len() <= max_keys {
        return keys.iter().cloned().collect();
    }
    let head = max_keys.saturating_add(1) / 2;
    let tail = max_keys - head;
    let mut selected = keys.iter().take(head).cloned().collect::<Vec<_>>();
    if tail > 0 {
        selected.extend(keys.iter().skip(keys.len() - tail).cloned());
    }
    selected
}

fn verify_measured_state(
    target: &TargetSpec,
    spec: &RunSpec,
    prefix: &str,
    namespace: u64,
    run_stats: &RunStats,
    payload: &[u8],
) -> Result<usize, String> {
    let state = measured_model(spec, prefix, payload, &run_stats.completed_per_thread);
    let keys = measured_keys(spec, prefix);
    let selected = sampled_keys(&keys, spec.value_len);
    let lenient = spec.ops.is_none()
        || (matches!(spec.key_space, KeySpace::Hot | KeySpace::UniformGlobal(_))
            && matches!(spec.workload, Workload::Balanced | Workload::Mixed));
    match target.kind {
        TargetKind::KiviNative => {
            let client =
                native_client(&target.endpoint, namespace).map_err(|error| error.to_string())?;
            verify_native_keys(&client, &selected, &state, payload, lenient)
        }
        TargetKind::KiviResp | TargetKind::RedisResp => {
            let mut client =
                RespClient::connect(&target.endpoint).map_err(|error| error.to_string())?;
            verify_resp_keys(&mut client, &selected, &state, payload, lenient)
        }
    }
}

fn verify_native_keys(
    client: &NativeClient,
    keys: &[String],
    state: &ModelState,
    payload: &[u8],
    lenient: bool,
) -> Result<usize, String> {
    let mut checks = 0usize;
    for key in keys {
        let key_ref = Key::from(key.as_str());
        if let Some(expected_counter) = state.counters.get(key) {
            let actual = client
                .counter_get(&key_ref)
                .map_err(|error| error.to_string())?
                .ok_or_else(|| format!("counter {key} disappeared"))?;
            if actual != *expected_counter && !lenient {
                return Err(format!(
                    "reproducer=COUNTER_GET {key}; expected={expected_counter}; actual={actual}"
                ));
            }
            checks += 1;
            continue;
        }
        let expected = state.values.get(key);
        let actual = client.get(&key_ref).map_err(|error| error.to_string())?;
        verify_value(key, expected, actual.as_deref(), payload, lenient)?;
        let range = client
            .get_range(&key_ref, 0, RANGE_WINDOW)
            .map_err(|error| error.to_string())?;
        verify_range(
            key,
            expected,
            actual.as_deref(),
            range.as_deref(),
            RANGE_WINDOW,
            lenient,
        )?;
        if state.expiries.contains(key) && !lenient {
            let expiry = client
                .get_expiry(&key_ref)
                .map_err(|error| error.to_string())?;
            if expiry.is_none() {
                return Err(format!(
                    "reproducer=TTL {key}; expected=expiring; actual=missing"
                ));
            }
        }
        checks += 1;
    }
    Ok(checks)
}

fn verify_resp_keys(
    client: &mut RespClient,
    keys: &[String],
    state: &ModelState,
    payload: &[u8],
    lenient: bool,
) -> Result<usize, String> {
    let mut checks = 0usize;
    for key in keys {
        if state.counters.contains_key(key) {
            return Err(format!("counter key {key} cannot be represented by RESP"));
        }
        let get_argv: Vec<&[u8]> = vec![b"GET", key.as_bytes()];
        let get = client
            .round_trip(&get_argv)
            .map_err(|error| error.to_string())?;
        let actual = match get {
            Reply::Bulk(Some(value)) => Some(value),
            Reply::Bulk(None) => None,
            other => return Err(format!("reproducer=GET {key}; unexpected={other:?}")),
        };
        let expected = state.values.get(key);
        verify_value(key, expected, actual.as_deref(), payload, lenient)?;
        let end = (RANGE_WINDOW - 1).to_string();
        let range_argv: Vec<&[u8]> = vec![b"GETRANGE", key.as_bytes(), b"0", end.as_bytes()];
        let range = client
            .round_trip(&range_argv)
            .map_err(|error| error.to_string())?;
        let range_bytes = match range {
            Reply::Bulk(Some(value)) => Some(value),
            other => return Err(format!("reproducer=GETRANGE {key}; unexpected={other:?}")),
        };
        verify_range(
            key,
            expected,
            actual.as_deref(),
            range_bytes.as_deref(),
            RANGE_WINDOW,
            lenient,
        )?;
        if state.expiries.contains(key) && !lenient {
            let ttl_argv: Vec<&[u8]> = vec![b"TTL", key.as_bytes()];
            let ttl = client
                .round_trip(&ttl_argv)
                .map_err(|error| error.to_string())?;
            match ttl {
                Reply::Integer(value) if value >= 0 => {}
                other => {
                    return Err(format!(
                        "reproducer=TTL {key}; expected=expiring; actual={other:?}"
                    ));
                }
            }
        }
        checks += 1;
    }
    Ok(checks)
}

fn verify_value(
    key: &str,
    expected: Option<&Vec<u8>>,
    actual: Option<&[u8]>,
    payload: &[u8],
    lenient: bool,
) -> Result<(), String> {
    let valid = match (expected, actual) {
        (Some(expected), Some(actual)) if expected.as_slice() == actual => true,
        (Some(_), Some(actual)) if lenient && valid_lenient_value(actual, payload) => true,
        (Some(_), None) if lenient => true,
        (None, None) => true,
        (None, Some(actual)) if lenient => valid_lenient_value(actual, payload),
        _ => false,
    };
    if valid {
        Ok(())
    } else {
        Err(format!(
            "reproducer=GET {key}; expected={}; actual={}",
            expected.map_or_else(|| "nil".to_owned(), |value| format!("{}B", value.len())),
            actual.map_or_else(|| "nil".to_owned(), |value| format!("{}B", value.len()))
        ))
    }
}

fn verify_range(
    key: &str,
    expected: Option<&Vec<u8>>,
    observed_value: Option<&[u8]>,
    actual: Option<&[u8]>,
    window: u64,
    lenient: bool,
) -> Result<(), String> {
    let observed_range = observed_value.map_or_else(Vec::new, |value| {
        value[..value
            .len()
            .min(usize::try_from(window).unwrap_or(value.len()))]
            .to_vec()
    });
    let actual_bytes = actual.unwrap_or_default();
    if observed_range == actual_bytes {
        return Ok(());
    }
    if lenient {
        return Err(format!(
            "reproducer=GETRANGE {key}; expected_len={} actual_len={}",
            observed_range.len(),
            actual_bytes.len()
        ));
    }
    let expected_bytes = expected.map_or_else(Vec::new, |value| {
        value[..value
            .len()
            .min(usize::try_from(window).unwrap_or(value.len()))]
            .to_vec()
    });
    if expected_bytes == actual_bytes {
        Ok(())
    } else {
        Err(format!(
            "reproducer=GETRANGE {key}; expected_len={} actual_len={}",
            expected_bytes.len(),
            actual_bytes.len()
        ))
    }
}

fn valid_lenient_value(actual: &[u8], payload: &[u8]) -> bool {
    if actual == payload {
        return true;
    }
    let mut patched = payload.to_vec();
    if patched.is_empty() {
        patched.push(0);
    }
    patched[0] = b'v';
    actual == patched || actual == b"v"
}

#[allow(clippy::cast_precision_loss)]
fn completed_rate(completed: u64, seconds: f64) -> f64 {
    completed as f64 / seconds.max(f64::MIN_POSITIVE)
}

fn normalize_latency(batch_ns: u64, pipeline: usize) -> u64 {
    batch_ns / u64::try_from(pipeline.max(1)).unwrap_or(u64::MAX)
}

#[allow(clippy::cast_precision_loss)]
fn payload_bytes_per_op(workload: Workload, value_len: usize) -> f64 {
    let keys = vec!["model-key".to_owned()];
    let counter = "model-counter";
    let count = 100usize;
    let total = (0..count)
        .map(|index| {
            let op = workload_op_for(workload, value_len, index, &keys, counter);
            match op {
                WorkloadOp::Get(_) => value_len,
                WorkloadOp::Set { len, .. } => len,
                WorkloadOp::GetRange(_) => {
                    value_len.min(usize::try_from(RANGE_WINDOW).unwrap_or(value_len))
                }
                WorkloadOp::SetRange(_) => 1,
                WorkloadOp::CounterAdd(_)
                | WorkloadOp::Delete(_)
                | WorkloadOp::Exists(_)
                | WorkloadOp::Expire(_)
                | WorkloadOp::Ttl(_) => 0,
            }
        })
        .fold(0u64, |total, value| {
            total.saturating_add(u64::try_from(value).unwrap_or(u64::MAX))
        });
    total as f64 / count as f64
}

fn run_measurement(
    spec: &RunSpec,
    target: &TargetSpec,
    namespace: u64,
    prefix: &str,
) -> anyhow::Result<RunStats> {
    let config = RunnerConfig {
        workload: spec.workload,
        space: spec.key_space,
        value_len: spec.value_len,
        threads: spec.threads,
        ops_per_thread: spec.ops.unwrap_or(1),
        warmup_per_thread: spec.warmup,
        pipeline: spec.pipeline,
        prefix: prefix.to_owned(),
    };
    match target.kind {
        TargetKind::KiviNative => {
            if spec.ops.is_some() {
                runner::run(config, || {
                    KiviNativeTarget::connect(&target.endpoint, namespace, 256)
                })
            } else {
                runner::run_for_duration(config, spec.duration, || {
                    KiviNativeTarget::connect(&target.endpoint, namespace, 256)
                })
            }
        }
        TargetKind::KiviResp | TargetKind::RedisResp => {
            if spec.ops.is_some() {
                runner::run(config, || Ok(RespTarget::lazy(&target.endpoint)))
            } else {
                runner::run_for_duration(config, spec.duration, || {
                    Ok(RespTarget::lazy(&target.endpoint))
                })
            }
        }
    }
}

fn validate_pair(pair: &TargetPair, run_id: u64, namespace: u64) -> ValidationReport {
    let prefix = format!(
        "kivi-lab:campaign:{}:{run_id}:{}:validation:",
        std::process::id(),
        pair.class.slug()
    );
    let left = validate_target(&pair.left, &prefix, namespace);
    let right = validate_target(&pair.right, &prefix, namespace);
    let valid = left.is_ok() && right.is_ok();
    let left_error = left.as_ref().err().cloned();
    let right_error = right.as_ref().err().cloned();
    let detail = match (left_error, right_error) {
        (Some(left), Some(right)) => format!("{left}; {right}"),
        (Some(error), None) | (None, Some(error)) => error,
        (None, None) => "strict model and state checks passed".to_owned(),
    };
    ValidationReport {
        class: pair.class.label().to_owned(),
        point_label: "protocol-capability".to_owned(),
        dimension: "protocol".to_owned(),
        profile: "capability".to_owned(),
        left_target: pair.left.label.to_owned(),
        right_target: pair.right.label.to_owned(),
        valid,
        left_checks: left.as_ref().map_or(0, |checks| *checks),
        right_checks: right.as_ref().map_or(0, |checks| *checks),
        detail,
        reproducer: None,
    }
}

fn validate_target(target: &TargetSpec, prefix: &str, namespace: u64) -> Result<usize, String> {
    match target.kind {
        TargetKind::KiviNative => validate_native_endpoint(&target.endpoint, prefix, namespace),
        TargetKind::KiviResp | TargetKind::RedisResp => {
            validate_resp_endpoint(&target.endpoint, prefix)
        }
    }
}

#[derive(Debug, Default)]
struct ModelState {
    values: BTreeMap<String, Vec<u8>>,
    counters: BTreeMap<String, i64>,
    expiries: BTreeSet<String>,
}

fn validate_spec_pair(
    pair: &TargetPair,
    spec: &RunSpec,
    run_id: u64,
    namespace: u64,
) -> ValidationReport {
    let prefix = format!(
        "kivi-lab:campaign:{}:{run_id}:{}:{}:{}:v{}:k{}:t{}:p{}:model:",
        std::process::id(),
        pair.class.slug(),
        spec.dimension.replace('-', "_"),
        spec.profile.replace('-', "_"),
        spec.value_len,
        spec.key_label.replace('-', "_"),
        spec.threads,
        spec.pipeline
    );
    let left = validate_model_endpoint(&pair.left, spec, &prefix, namespace);
    let right = validate_model_endpoint(&pair.right, spec, &prefix, namespace);
    let valid = left.is_ok() && right.is_ok();
    let left_error = left.as_ref().err().cloned();
    let right_error = right.as_ref().err().cloned();
    let (detail, reproducer) = match (left_error, right_error) {
        (Some(left), Some(right)) => (
            format!("{left}; {right}"),
            first_reproducer(&[&left, &right]),
        ),
        (Some(error), None) | (None, Some(error)) => {
            let reproducer = reproducer_from_error(&error);
            (error, reproducer)
        }
        (None, None) => ("deterministic model/state checks passed".to_owned(), None),
    };
    ValidationReport {
        class: pair.class.label().to_owned(),
        point_label: point_label(spec),
        dimension: spec.dimension.clone(),
        profile: spec.profile.clone(),
        left_target: pair.left.label.to_owned(),
        right_target: pair.right.label.to_owned(),
        valid,
        left_checks: left.as_ref().map_or(0, |checks| *checks),
        right_checks: right.as_ref().map_or(0, |checks| *checks),
        detail,
        reproducer,
    }
}

fn first_reproducer(errors: &[&str]) -> Option<String> {
    errors.iter().find_map(|error| reproducer_from_error(error))
}

fn reproducer_from_error(error: &str) -> Option<String> {
    error.split_once("reproducer=").map(|(_, value)| {
        value
            .split("; expected=")
            .next()
            .unwrap_or(value)
            .trim()
            .to_owned()
    })
}

enum ModelClient {
    Native(NativeClient),
    Resp(RespClient),
}

#[allow(clippy::too_many_lines)]
fn validate_model_endpoint(
    target: &TargetSpec,
    spec: &RunSpec,
    prefix: &str,
    namespace: u64,
) -> Result<usize, String> {
    let value_len = spec.value_len.clamp(1, 4096);
    let key_space = match spec.key_space {
        KeySpace::Hot => KeySpace::Hot,
        KeySpace::Uniform(count) => KeySpace::Uniform(count.min(4)),
        KeySpace::UniformGlobal(count) => KeySpace::UniformGlobal(count.min(4)),
    };
    let keys = thread_key_names(key_space, 0, prefix);
    let counter = counter_key_name(key_space, 0, prefix);
    let payload = PayloadCache::new(value_len);
    let mut model_client = match target.kind {
        TargetKind::KiviNative => ModelClient::Native(
            native_client(&target.endpoint, namespace).map_err(|error| error.to_string())?,
        ),
        TargetKind::KiviResp | TargetKind::RedisResp => ModelClient::Resp(
            RespClient::connect(&target.endpoint).map_err(|error| error.to_string())?,
        ),
    };
    let mut state = ModelState::default();
    let result = (|| {
        let mut checks = 0usize;
        for key in &keys {
            let op = WorkloadOp::Set {
                key: key.clone(),
                len: value_len,
            };
            validate_model_op(&mut model_client, &op, &mut state, payload.as_bytes())?;
            checks += 1;
        }
        if matches!(target.kind, TargetKind::KiviNative) {
            match &mut model_client {
                ModelClient::Native(client) => {
                    client
                        .counter_add(&Key::from(counter.as_str()), 0)
                        .map_err(|error| error.to_string())?;
                    state.counters.insert(counter.clone(), 0);
                }
                ModelClient::Resp(_) => {
                    return Err("native model client was not constructed".to_owned());
                }
            }
            checks += 1;
        } else if matches!(spec.workload, Workload::Counter | Workload::Mixed) {
            return Err("counter workload is not supported by RESP".to_owned());
        }
        for index in 0..32usize {
            let op = workload_op_for(spec.workload, value_len, index, &keys, &counter);
            validate_model_op(&mut model_client, &op, &mut state, payload.as_bytes())?;
            checks += 1;
        }
        let key = keys
            .first()
            .ok_or_else(|| "model keyspace is empty".to_owned())?;
        let mut sequence = vec![
            WorkloadOp::Set {
                key: key.clone(),
                len: value_len,
            },
            WorkloadOp::Get(key.clone()),
            WorkloadOp::Exists(key.clone()),
            WorkloadOp::GetRange(key.clone()),
            WorkloadOp::SetRange(key.clone()),
            WorkloadOp::Expire(key.clone()),
            WorkloadOp::Ttl(key.clone()),
            WorkloadOp::Delete(key.clone()),
            WorkloadOp::Get(key.clone()),
            WorkloadOp::GetRange(key.clone()),
        ];
        if matches!(target.kind, TargetKind::KiviNative) {
            sequence.push(WorkloadOp::CounterAdd(counter.clone()));
        }
        for op in sequence {
            validate_model_op(&mut model_client, &op, &mut state, payload.as_bytes())?;
            checks += 1;
        }
        Ok::<usize, String>(checks)
    })();
    let cleanup = match target.kind {
        TargetKind::KiviNative => {
            let client =
                native_client(&target.endpoint, namespace).map_err(|error| error.to_string())?;
            let mut keys = keys.clone();
            if matches!(target.kind, TargetKind::KiviNative)
                || matches!(spec.workload, Workload::Counter | Workload::Mixed)
            {
                keys.push(counter.clone());
            }
            keys.into_iter().try_for_each(|key| {
                client
                    .delete(&Key::from(key.as_str()))
                    .map(|_| ())
                    .map_err(|error| error.to_string())
            })
        }
        TargetKind::KiviResp | TargetKind::RedisResp => {
            let mut client =
                RespClient::connect(&target.endpoint).map_err(|error| error.to_string())?;
            let mut keys = keys.clone();
            if matches!(target.kind, TargetKind::KiviNative)
                || matches!(spec.workload, Workload::Counter | Workload::Mixed)
            {
                keys.push(counter.clone());
            }
            keys.into_iter()
                .try_for_each(|key| delete_resp_key(&mut client, &key))
        }
    };
    match (result, cleanup) {
        (Ok(checks), Ok(())) => Ok(checks),
        (Err(error), Ok(())) | (Ok(_), Err(error)) => Err(error),
        (Err(error), Err(cleanup_error)) => Err(format!("{error}; cleanup: {cleanup_error}")),
    }
}

fn validate_model_op(
    client: &mut ModelClient,
    op: &WorkloadOp,
    state: &mut ModelState,
    payload: &[u8],
) -> Result<(), String> {
    if matches!(op, WorkloadOp::CounterAdd(_)) && matches!(client, ModelClient::Resp(_)) {
        return Err("counter validation is native-only and is not a RESP point".to_owned());
    }
    let expected = model_expected(op, state, payload, matches!(client, ModelClient::Native(_)));
    let actual = match client {
        ModelClient::Native(client) => validate_native_model_op(client, op, state, payload)?,
        ModelClient::Resp(client) => validate_resp_model_op(client, op, payload)?,
    };
    if actual != expected {
        return Err(model_failure(op, &expected, &actual));
    }
    model_update(op, state, payload);
    Ok(())
}

fn validate_resp_model_op(
    client: &mut RespClient,
    op: &WorkloadOp,
    payload: &[u8],
) -> Result<String, String> {
    let command = encode_workload_command(op, payload);
    let argv: Vec<&[u8]> = command.iter().map(Vec::as_slice).collect();
    let reply = client
        .round_trip(&argv)
        .map_err(|error| error.to_string())?;
    check_workload_reply(&reply, op)?;
    let actual = match op {
        WorkloadOp::Get(_) | WorkloadOp::GetRange(_) => match reply {
            Reply::Bulk(Some(value)) => format!("bulk:{}", short_bytes(&value)),
            Reply::Bulk(None) => "nil".to_owned(),
            other => format!("{other:?}"),
        },
        WorkloadOp::Set { .. } | WorkloadOp::CounterAdd(_) => "status:ok".to_owned(),
        WorkloadOp::Delete(_)
        | WorkloadOp::Exists(_)
        | WorkloadOp::Expire(_)
        | WorkloadOp::Ttl(_)
        | WorkloadOp::SetRange(_) => match reply {
            Reply::Integer(value) => {
                if matches!(op, WorkloadOp::Ttl(_)) {
                    match value {
                        -2 => "integer:missing".to_owned(),
                        -1 => "integer:immortal".to_owned(),
                        value if value >= 0 => "integer:expiring".to_owned(),
                        _ => format!("integer:{value}"),
                    }
                } else {
                    format!("integer:{value}")
                }
            }
            other => format!("{other:?}"),
        },
    };
    Ok(actual)
}

fn validate_native_model_op(
    client: &NativeClient,
    op: &WorkloadOp,
    state: &ModelState,
    payload: &[u8],
) -> Result<String, String> {
    let actual = match op {
        WorkloadOp::Get(key) => client
            .get(&Key::from(key.as_str()))
            .map_err(|error| error.to_string())
            .map(|value| {
                value.map_or_else(
                    || "nil".to_owned(),
                    |bytes| format!("bulk:{}", short_bytes(&bytes)),
                )
            })?,
        WorkloadOp::GetRange(key) => client
            .get_range(&Key::from(key.as_str()), 0, RANGE_WINDOW)
            .map_err(|error| error.to_string())
            .map(|value| {
                value.map_or_else(
                    || "nil".to_owned(),
                    |bytes| format!("bulk:{}", short_bytes(&bytes)),
                )
            })?,
        WorkloadOp::Exists(key) => client
            .exists(&Key::from(key.as_str()))
            .map_err(|error| error.to_string())
            .map(|value| format!("exists:{value}"))?,
        WorkloadOp::Set { key, len } => {
            let length = (*len).min(payload.len());
            client
                .set(
                    &Key::from(key.as_str()),
                    Bytes::copy_from_slice(&payload[..length]),
                )
                .map_err(|error| error.to_string())?;
            "status:ok".to_owned()
        }
        WorkloadOp::CounterAdd(key) => client
            .counter_add(&Key::from(key.as_str()), 1)
            .map(|value| format!("counter:{value}"))
            .map_err(|error| error.to_string())?,

        WorkloadOp::SetRange(key) => {
            client
                .set_range(&Key::from(key.as_str()), 0, Bytes::from_static(b"v"))
                .map_err(|error| error.to_string())?;
            let length = client
                .bytes_length(&Key::from(key.as_str()))
                .map_err(|error| error.to_string())?;
            format!("length:{length:?}")
        }
        WorkloadOp::Expire(key) => client
            .expire_at(&Key::from(key.as_str()), campaign_expiry())
            .map_err(|error| error.to_string())
            .map(|value| format!("exists:{value}"))?,
        WorkloadOp::Ttl(key) => {
            let present = state.values.contains_key(key);
            let expiry = client
                .get_expiry(&Key::from(key.as_str()))
                .map_err(|error| error.to_string())?;
            if !present {
                "ttl:missing".to_owned()
            } else if expiry.and_then(kivi_types::Expiry::as_stamp).is_some() {
                "ttl:expiring".to_owned()
            } else {
                "ttl:immortal".to_owned()
            }
        }
        WorkloadOp::Delete(key) => client
            .delete(&Key::from(key.as_str()))
            .map_err(|error| error.to_string())
            .map(|value| format!("exists:{value}"))?,
    };
    Ok(actual)
}

fn model_expected(op: &WorkloadOp, state: &ModelState, _payload: &[u8], native: bool) -> String {
    match op {
        WorkloadOp::Get(key) => state.values.get(key).map_or_else(
            || "nil".to_owned(),
            |value| format!("bulk:{}", short_bytes(value)),
        ),
        WorkloadOp::GetRange(key) => state.values.get(key).map_or_else(
            || {
                if native {
                    "nil".to_owned()
                } else {
                    "bulk:\"\"".to_owned()
                }
            },
            |value| {
                let length = value
                    .len()
                    .min(usize::try_from(RANGE_WINDOW).unwrap_or(value.len()));
                format!("bulk:{}", short_bytes(&value[..length]))
            },
        ),
        WorkloadOp::Set { .. } => "status:ok".to_owned(),
        WorkloadOp::SetRange(key) => {
            let length = state.values.get(key).map_or(0, Vec::len).max(1);
            if native {
                format!("length:Some({length})")
            } else {
                format!("integer:{length}")
            }
        }
        WorkloadOp::Expire(key) | WorkloadOp::Delete(key) | WorkloadOp::Exists(key) => {
            let present = state.values.contains_key(key) || state.counters.contains_key(key);
            if native {
                format!("exists:{present}")
            } else {
                format!("integer:{}", i64::from(present))
            }
        }
        WorkloadOp::Ttl(key) => {
            if !state.values.contains_key(key) {
                if native {
                    "ttl:missing".to_owned()
                } else {
                    "integer:missing".to_owned()
                }
            } else if state.expiries.contains(key) {
                if native {
                    "ttl:expiring".to_owned()
                } else {
                    "integer:expiring".to_owned()
                }
            } else if native {
                "ttl:immortal".to_owned()
            } else {
                "integer:immortal".to_owned()
            }
        }
        WorkloadOp::CounterAdd(key) => {
            let value = state
                .counters
                .get(key)
                .copied()
                .unwrap_or(0)
                .saturating_add(1);
            if native {
                format!("counter:{value}")
            } else {
                "unsupported:counter".to_owned()
            }
        }
    }
}

fn model_update(op: &WorkloadOp, state: &mut ModelState, payload: &[u8]) {
    match op {
        WorkloadOp::Set { key, len } => {
            let length = (*len).min(payload.len());
            state.values.insert(key.clone(), payload[..length].to_vec());
            state.counters.remove(key);
            state.expiries.remove(key);
        }
        WorkloadOp::SetRange(key) => {
            let mut value = state.values.remove(key).unwrap_or_default();
            if value.is_empty() {
                value.push(0);
            }
            value[0] = b'v';
            state.values.insert(key.clone(), value);
            state.counters.remove(key);
        }
        WorkloadOp::Delete(key) => {
            state.values.remove(key);
            state.counters.remove(key);
            state.expiries.remove(key);
        }
        WorkloadOp::Expire(key) => {
            if state.values.contains_key(key) {
                state.expiries.insert(key.clone());
            }
        }
        WorkloadOp::CounterAdd(key) => {
            let value = state.counters.entry(key.clone()).or_default();
            *value = value.saturating_add(1);
        }
        WorkloadOp::Get(_)
        | WorkloadOp::Exists(_)
        | WorkloadOp::Ttl(_)
        | WorkloadOp::GetRange(_) => {}
    }
}

fn model_failure(op: &WorkloadOp, expected: &str, actual: &str) -> String {
    let key = match op {
        WorkloadOp::Get(key)
        | WorkloadOp::Exists(key)
        | WorkloadOp::Delete(key)
        | WorkloadOp::Expire(key)
        | WorkloadOp::Ttl(key)
        | WorkloadOp::SetRange(key)
        | WorkloadOp::GetRange(key)
        | WorkloadOp::Set { key, .. }
        | WorkloadOp::CounterAdd(key) => key.as_str(),
    };
    let command = match op {
        WorkloadOp::Get(_) => "GET",
        WorkloadOp::Set { len, .. } => {
            return format!(
                "reproducer=SET {key} <deterministic:{len}>; expected={expected}; actual={actual}"
            );
        }
        WorkloadOp::Delete(_) => "DEL",
        WorkloadOp::Exists(_) => "EXISTS",
        WorkloadOp::Expire(_) => "EXPIRE",
        WorkloadOp::Ttl(_) => "TTL",
        WorkloadOp::SetRange(_) => "SETRANGE",
        WorkloadOp::GetRange(_) => "GETRANGE",
        WorkloadOp::CounterAdd(_) => "INCRBY",
    };
    format!("reproducer={command} {key}; expected={expected}; actual={actual}")
}

fn short_bytes(bytes: &[u8]) -> String {
    let text = String::from_utf8_lossy(bytes);
    if text.len() <= 96 {
        format!("{text:?}")
    } else {
        let end = text
            .char_indices()
            .map(|(index, _)| index)
            .take_while(|index| *index <= 96)
            .last()
            .unwrap_or(0);
        format!("{:?}...", &text[..end])
    }
}

fn validate_resp_endpoint(endpoint: &str, prefix: &str) -> Result<usize, String> {
    let mut client = RespClient::connect(endpoint).map_err(|error| error.to_string())?;
    let key = format!("{prefix}model");
    let result = validate_resp_body(&mut client, &key);
    let cleanup = delete_resp_key(&mut client, &key);
    match (result, cleanup) {
        (Ok(checks), Ok(())) => Ok(checks),
        (Err(error), Ok(())) | (Ok(_), Err(error)) => Err(error),
        (Err(error), Err(cleanup_error)) => Err(format!("{error}; cleanup: {cleanup_error}")),
    }
}

fn validate_resp_body(client: &mut RespClient, key: &str) -> Result<usize, String> {
    let payload = PayloadCache::new(128);
    let range_len = usize::try_from(RANGE_WINDOW).unwrap_or(usize::MAX);
    let mut checks = 0usize;
    let set = WorkloadOp::Set {
        key: key.to_owned(),
        len: payload.len(),
    };
    let reply = resp_command(client, &set, &[b"SET", key.as_bytes(), payload.as_bytes()])?;
    check_workload_reply(&reply, &set).map_err(|error| error.clone())?;
    checks += 1;
    let get = WorkloadOp::Get(key.to_owned());
    let reply = resp_command(client, &get, &[b"GET", key.as_bytes()])?;
    check_workload_reply(&reply, &get).map_err(|error| error.clone())?;
    expect_bulk(&reply, payload.as_bytes())?;
    checks += 1;
    let exists = WorkloadOp::Exists(key.to_owned());
    let reply = resp_command(client, &exists, &[b"EXISTS", key.as_bytes()])?;
    check_workload_reply(&reply, &exists).map_err(|error| error.clone())?;
    expect_integer(&reply, 1)?;
    checks += 1;
    let range = WorkloadOp::GetRange(key.to_owned());
    let end = (RANGE_WINDOW - 1).to_string();
    let reply = resp_command(
        client,
        &range,
        &[b"GETRANGE", key.as_bytes(), b"0", end.as_bytes()],
    )?;
    check_workload_reply(&reply, &range).map_err(|error| error.clone())?;
    expect_bulk(&reply, &payload.as_bytes()[..range_len])?;
    checks += 1;
    let set_range = WorkloadOp::SetRange(key.to_owned());
    let reply = resp_command(
        client,
        &set_range,
        &[b"SETRANGE", key.as_bytes(), b"0", b"v"],
    )?;
    check_workload_reply(&reply, &set_range).map_err(|error| error.clone())?;
    expect_integer(
        &reply,
        i64::try_from(payload.len()).map_err(|_| "payload length exceeds i64".to_owned())?,
    )?;
    checks += 1;
    let reply = resp_command(
        client,
        &range,
        &[b"GETRANGE", key.as_bytes(), b"0", end.as_bytes()],
    )?;
    let mut expected = payload.as_bytes()[..range_len].to_vec();
    expected[0] = b'v';
    expect_bulk(&reply, &expected)?;
    checks += 1;
    let expire = WorkloadOp::Expire(key.to_owned());
    let reply = resp_command(client, &expire, &[b"EXPIRE", key.as_bytes(), b"60"])?;
    check_workload_reply(&reply, &expire).map_err(|error| error.clone())?;
    expect_integer(&reply, 1)?;
    checks += 1;
    let ttl = WorkloadOp::Ttl(key.to_owned());
    let reply = resp_command(client, &ttl, &[b"TTL", key.as_bytes()])?;
    check_workload_reply(&reply, &ttl).map_err(|error| error.clone())?;
    match reply {
        Reply::Integer(value) if value > 0 => {}
        other => return Err(format!("TTL expected positive integer, got {other:?}")),
    }
    checks += 1;
    let delete = WorkloadOp::Delete(key.to_owned());
    let reply = resp_command(client, &delete, &[b"DEL", key.as_bytes()])?;
    check_workload_reply(&reply, &delete).map_err(|error| error.clone())?;
    expect_integer(&reply, 1)?;
    checks += 1;
    let reply = resp_command(client, &exists, &[b"EXISTS", key.as_bytes()])?;
    expect_integer(&reply, 0)?;
    checks += 1;
    let reply = resp_command(client, &get, &[b"GET", key.as_bytes()])?;
    expect_nil(&reply)?;
    checks += 1;
    let reply = resp_command(
        client,
        &range,
        &[b"GETRANGE", key.as_bytes(), b"0", end.as_bytes()],
    )?;
    expect_bulk(&reply, &[])?;
    Ok(checks)
}

fn validate_native_endpoint(endpoint: &str, prefix: &str, namespace: u64) -> Result<usize, String> {
    let client = native_client(endpoint, namespace).map_err(|error| error.to_string())?;
    let key = Key::from(format!("{prefix}model"));
    let payload = PayloadCache::new(128);
    let result = validate_native_body(&client, &key, payload.as_bytes());
    let cleanup = client.delete(&key).map_err(|error| error.to_string());
    match (result, cleanup) {
        (Ok(checks), Ok(_)) => Ok(checks),
        (Err(error), Ok(_)) | (Ok(_), Err(error)) => Err(error),
        (Err(error), Err(cleanup_error)) => Err(format!("{error}; cleanup: {cleanup_error}")),
    }
}

fn validate_native_body(client: &NativeClient, key: &Key, payload: &[u8]) -> Result<usize, String> {
    let range_len = usize::try_from(RANGE_WINDOW).unwrap_or(usize::MAX);
    let mut checks = 0usize;
    client
        .set(key, Bytes::copy_from_slice(payload))
        .map_err(|error| error.to_string())?;
    checks += 1;
    let value = client
        .get(key)
        .map_err(|error| error.to_string())?
        .ok_or_else(|| "native validation GET returned absent".to_owned())?;
    if value.as_ref() != payload {
        return Err("native validation GET returned different bytes".to_owned());
    }
    checks += 1;
    if !client.exists(key).map_err(|error| error.to_string())? {
        return Err("native validation EXISTS returned false".to_owned());
    }
    checks += 1;
    let range = client
        .get_range(key, 0, RANGE_WINDOW)
        .map_err(|error| error.to_string())?
        .ok_or_else(|| "native validation GETRANGE returned absent".to_owned())?;
    if range.as_ref() != &payload[..range_len] {
        return Err("native validation GETRANGE returned different bytes".to_owned());
    }
    checks += 1;
    client
        .set_range(key, 0, Bytes::from_static(b"v"))
        .map_err(|error| error.to_string())?;
    checks += 1;
    let range = client
        .get_range(key, 0, RANGE_WINDOW)
        .map_err(|error| error.to_string())?
        .ok_or_else(|| "native validation GETRANGE after patch returned absent".to_owned())?;
    let mut expected = payload[..range_len].to_vec();
    expected[0] = b'v';
    if range.as_ref() != expected {
        return Err("native validation GETRANGE after patch returned different bytes".to_owned());
    }
    checks += 1;
    if !client
        .expire_at(key, campaign_expiry())
        .map_err(|error| error.to_string())?
    {
        return Err("native validation EXPIRE returned false".to_owned());
    }
    checks += 1;
    if client
        .get_expiry(key)
        .map_err(|error| error.to_string())?
        .is_none()
    {
        return Err("native validation TTL returned absent".to_owned());
    }
    checks += 1;
    if !client.delete(key).map_err(|error| error.to_string())? {
        return Err("native validation DELETE returned false".to_owned());
    }
    checks += 1;
    if client.exists(key).map_err(|error| error.to_string())? {
        return Err("native validation EXISTS after DELETE returned true".to_owned());
    }
    checks += 1;
    if client
        .get(key)
        .map_err(|error| error.to_string())?
        .is_some()
    {
        return Err("native validation GET after DELETE returned bytes".to_owned());
    }
    checks += 1;
    if client
        .get_range(key, 0, RANGE_WINDOW)
        .map_err(|error| error.to_string())?
        .is_some()
    {
        return Err("native validation GETRANGE after DELETE returned bytes".to_owned());
    }
    Ok(checks)
}

fn resp_command(
    client: &mut RespClient,
    _op: &WorkloadOp,
    argv: &[&[u8]],
) -> Result<Reply, String> {
    client.round_trip(argv).map_err(|error| error.to_string())
}

fn expect_nil(reply: &Reply) -> Result<(), String> {
    match reply {
        Reply::Bulk(None) => Ok(()),
        other => Err(format!("expected nil bulk, got {other:?}")),
    }
}

fn expect_bulk(reply: &Reply, expected: &[u8]) -> Result<(), String> {
    match reply {
        Reply::Bulk(Some(value)) if value == expected => Ok(()),
        other => Err(format!("expected bulk {expected:?}, got {other:?}")),
    }
}

fn expect_integer(reply: &Reply, expected: i64) -> Result<(), String> {
    match reply {
        Reply::Integer(value) if *value == expected => Ok(()),
        other => Err(format!("expected integer {expected}, got {other:?}")),
    }
}

fn delete_resp_key(client: &mut RespClient, key: &str) -> Result<(), String> {
    match client.round_trip(&[b"DEL", key.as_bytes()]) {
        Ok(Reply::Integer(_)) => {}
        Ok(other) => return Err(format!("cleanup DEL returned {other:?}")),
        Err(error) => return Err(error.to_string()),
    }
    match client.round_trip(&[b"EXISTS", key.as_bytes()]) {
        Ok(Reply::Integer(0)) => Ok(()),
        Ok(other) => Err(format!("cleanup EXISTS returned {other:?}")),
        Err(error) => Err(error.to_string()),
    }
}

fn set_resp_probe(client: &mut RespClient, key: &str, payload: &[u8]) -> Result<(), String> {
    match client.round_trip(&[b"SET", key.as_bytes(), payload]) {
        Ok(Reply::Simple(value)) if value.eq_ignore_ascii_case(b"OK") => Ok(()),
        Ok(other) => Err(format!("probe SET returned {other:?}")),
        Err(error) => Err(error.to_string()),
    }
}

fn check_resp_probe(client: &mut RespClient, key: &str, expected: &[u8]) -> Result<(), String> {
    let get = client
        .round_trip(&[b"GET", key.as_bytes()])
        .map_err(|error| error.to_string())?;
    expect_bulk(&get, expected)?;
    let exists = client
        .round_trip(&[b"EXISTS", key.as_bytes()])
        .map_err(|error| error.to_string())?;
    expect_integer(&exists, 1)
}

fn check_native_probe(client: &NativeClient, key: &str, expected: &[u8]) -> Result<(), String> {
    let value = client
        .get(&Key::from(key))
        .map_err(|error| error.to_string())?
        .ok_or_else(|| "probe GET returned absent".to_owned())?;
    if value.as_ref() != expected {
        return Err("probe GET returned different bytes".to_owned());
    }
    if !client
        .exists(&Key::from(key))
        .map_err(|error| error.to_string())?
    {
        return Err("probe EXISTS returned false".to_owned());
    }
    Ok(())
}

fn cleanup_resp_owned(
    client: &mut RespClient,
    spec: &RunSpec,
    prefix: &str,
    probe: &str,
) -> Result<(), String> {
    let mut keys = owned_keys(spec, prefix);
    keys.insert(probe.to_owned());
    for key in keys {
        delete_resp_key(client, &key)?;
    }
    Ok(())
}

fn cleanup_native_owned(
    client: &NativeClient,
    spec: &RunSpec,
    prefix: &str,
    probe: &str,
) -> Result<(), String> {
    let mut keys = owned_keys(spec, prefix);
    keys.insert(probe.to_owned());
    for key in keys {
        let key = Key::from(key.as_str());
        client.delete(&key).map_err(|error| error.to_string())?;
        if client.exists(&key).map_err(|error| error.to_string())? {
            return Err(format!("cleanup key still exists: {key}"));
        }
    }
    Ok(())
}

fn owned_keys(spec: &RunSpec, prefix: &str) -> BTreeSet<String> {
    let mut keys = BTreeSet::new();
    for thread in 0..spec.threads {
        keys.extend(thread_key_names(spec.key_space, thread, prefix));
        if matches!(spec.workload, Workload::Counter | Workload::Mixed) {
            keys.insert(counter_key_name(spec.key_space, thread, prefix));
        }
    }
    keys
}

fn native_client(endpoint: &str, namespace: u64) -> anyhow::Result<NativeClient> {
    NativeClient::new(ClientConfig {
        seeds: vec![endpoint.to_owned()],
        namespace: NamespaceId::from_u64(namespace),
        ..ClientConfig::default()
    })
    .context("native Kivi client setup failed")
}

fn campaign_expiry() -> WallTimestamp {
    wall_now_or_max(&SystemClock)
        .checked_add(jiff::SignedDuration::from_secs(60))
        .unwrap_or(WallTimestamp::MAX)
}

fn profile_name(workload: Workload) -> &'static str {
    match workload {
        Workload::Get => "get",
        Workload::Set => "set",
        Workload::Counter => "counter",
        Workload::Mixed => "mixed",
        Workload::Cache => "cache-read",
        Workload::ReadHeavy => "read-heavy",
        Workload::Balanced => "balanced",
        Workload::Delete => "delete",
        Workload::Exists => "exists",
        Workload::Compat => "compat",
        Workload::Range => "range",
    }
}

fn value_label(value_len: usize) -> String {
    const KIB: usize = 1024;
    const MIB: usize = 1024 * 1024;
    if value_len == 16 {
        "16B".to_owned()
    } else if value_len == KIB {
        "1KiB".to_owned()
    } else if value_len == 64 * KIB {
        "64KiB".to_owned()
    } else if value_len == 256 * KIB {
        "256KiB".to_owned()
    } else if value_len == MIB {
        "1MiB".to_owned()
    } else if value_len == 4 * MIB {
        "4MiB".to_owned()
    } else if value_len == 64 * MIB {
        "64MiB".to_owned()
    } else {
        format!("{value_len}B")
    }
}

fn point_label(spec: &RunSpec) -> String {
    format!(
        "{} profile={} value={} keys={} threads={} pipeline={}",
        spec.dimension, spec.profile, spec.value_label, spec.key_label, spec.threads, spec.pipeline
    )
}

fn saturation_label(spec: &RunSpec) -> String {
    if spec.curve {
        match spec.dimension.as_str() {
            "pipeline" => format!("pipeline-depth={}", spec.pipeline),
            "concurrency" => format!("concurrency={}", spec.threads),
            "value-size" => format!("value={}", spec.value_label),
            "key-distribution" => format!("keys={}", spec.key_label),
            _ => format!("dimension={}", spec.dimension),
        }
    } else {
        "not-applicable".to_owned()
    }
}

fn workload_label(workload: Workload) -> String {
    format!("{workload:?}")
}

fn key_space_label(space: KeySpace) -> String {
    match space {
        KeySpace::Hot => "hot".to_owned(),
        KeySpace::Uniform(count) => format!("uniform:{count}"),
        KeySpace::UniformGlobal(count) => format!("uniform-global:{count}"),
    }
}

#[allow(clippy::cast_precision_loss)]
fn bytes_per_op(report: &TrialReport) -> Option<f64> {
    report
        .bytes_out
        .zip(report.bytes_in)
        .map(|(out, input)| (out + input) as f64 / report.completed.max(1) as f64)
}

#[allow(clippy::cast_precision_loss)]
fn aggregate_reports(reports: &[TrialReport]) -> AggregateReport {
    let ops: Vec<f64> = reports.iter().map(|report| report.ops_per_sec).collect();
    let ops_per_thread: Vec<f64> = reports
        .iter()
        .map(|report| report.ops_per_sec / report.threads.max(1) as f64)
        .collect();
    let p50: Vec<u64> = reports.iter().map(|report| report.p50_ns).collect();
    let p95: Vec<u64> = reports.iter().map(|report| report.p95_ns).collect();
    let p99: Vec<u64> = reports.iter().map(|report| report.p99_ns).collect();
    let p999: Vec<u64> = reports.iter().map(|report| report.p999_ns).collect();
    let bytes_out: Vec<u64> = reports
        .iter()
        .filter_map(|report| report.bytes_out)
        .collect();
    let bytes_in: Vec<u64> = reports
        .iter()
        .filter_map(|report| report.bytes_in)
        .collect();
    let bytes_per_sec: Vec<f64> = reports
        .iter()
        .filter_map(|report| report.bytes_per_sec)
        .collect();
    let effective_payload_bytes_per_sec: Vec<f64> = reports
        .iter()
        .map(|report| report.effective_payload_bytes_per_sec)
        .collect();
    let median_ops = median_f64(&ops);
    AggregateReport {
        trials: reports.len(),
        valid: !reports.is_empty() && reports.iter().all(|report| report.valid),
        median_ops_per_sec: median_ops,
        min_ops_per_sec: ops.iter().copied().reduce(f64::min).unwrap_or_default(),
        max_ops_per_sec: ops.iter().copied().reduce(f64::max).unwrap_or_default(),
        spread_ops_per_sec: spread(&ops),
        median_ops_per_sec_per_thread: median_f64(&ops_per_thread),
        median_p50_ns: median_u64(&p50),
        median_p95_ns: median_u64(&p95),
        median_p99_ns: median_u64(&p99),
        median_p999_ns: median_u64(&p999),
        spread_p50_ns: spread_u64(&p50),
        spread_p95_ns: spread_u64(&p95),
        spread_p99_ns: spread_u64(&p99),
        spread_p999_ns: spread_u64(&p999),
        median_bytes_out: median_u64_opt(&bytes_out),
        median_bytes_in: median_u64_opt(&bytes_in),
        median_bytes_per_sec: median_f64_opt(&bytes_per_sec),
        median_effective_payload_bytes_per_sec: median_f64(&effective_payload_bytes_per_sec),
    }
}

fn median_f64(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    if sorted.is_empty() {
        return 0.0;
    }
    let middle = sorted.len() / 2;
    if sorted.len().is_multiple_of(2) {
        f64::midpoint(sorted[middle - 1], sorted[middle])
    } else {
        sorted[middle]
    }
}

fn median_u64(values: &[u64]) -> u64 {
    median_u64_opt(values).unwrap_or_default()
}

fn median_u64_opt(values: &[u64]) -> Option<u64> {
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    if sorted.is_empty() {
        return None;
    }
    let middle = sorted.len() / 2;
    Some(if sorted.len().is_multiple_of(2) {
        sorted[middle - 1]
            .saturating_add(sorted[middle])
            .saturating_div(2)
    } else {
        sorted[middle]
    })
}

fn spread(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let min = values.iter().copied().reduce(f64::min).unwrap_or_default();
    let max = values.iter().copied().reduce(f64::max).unwrap_or_default();
    let median = median_f64(values);
    if median > 0.0 {
        (max - min) / median
    } else {
        0.0
    }
}

#[allow(clippy::cast_precision_loss)]
fn spread_u64(values: &[u64]) -> f64 {
    let Some(max) = values.iter().copied().max() else {
        return 0.0;
    };
    let min = values.iter().copied().min().unwrap_or_default();
    (max - min) as f64
}

fn median_f64_opt(values: &[f64]) -> Option<f64> {
    (!values.is_empty()).then(|| median_f64(values))
}

fn ratio(left: f64, right: f64) -> Option<f64> {
    (right > 0.0 && left.is_finite() && right.is_finite()).then_some(left / right)
}

fn ratio_optional(left: Option<f64>, right: Option<f64>) -> Option<f64> {
    left.zip(right).and_then(|(left, right)| ratio(left, right))
}

#[allow(clippy::cast_precision_loss)]
fn ratio_u64(left: u64, right: u64) -> Option<f64> {
    (right > 0).then_some(left as f64 / right as f64)
}

fn unix_millis() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_millis())
}

fn timestamp_stamp(unix_millis: u128) -> String {
    let seconds = unix_millis / 1000;
    let millis = unix_millis % 1000;
    format!("run-{seconds}-{millis:03}-{}", std::process::id())
}

fn build_profile() -> &'static str {
    if cfg!(debug_assertions) {
        "debug"
    } else {
        "release"
    }
}

fn build_metadata() -> BTreeMap<String, String> {
    let mut values = BTreeMap::new();
    values.insert(
        "package_version".to_owned(),
        env!("CARGO_PKG_VERSION").to_owned(),
    );
    values.insert("profile".to_owned(), build_profile().to_owned());
    values.insert("target".to_owned(), env!("CARGO_PKG_NAME").to_owned());
    if let Some(output) = command_text("git", &["rev-parse", "HEAD"]) {
        values.insert("git_commit".to_owned(), output);
    }
    if let Some(output) = command_text("git", &["rev-parse", "--abbrev-ref", "HEAD"]) {
        values.insert("git_branch".to_owned(), output);
    }
    if let Some(output) = command_text("git", &["rev-parse", "--show-toplevel"]) {
        values.insert("worktree_root".to_owned(), output);
    }
    if let Some(output) = command_text("git", &["status", "--porcelain"]) {
        values.insert(
            "worktree_dirty".to_owned(),
            (!output.is_empty()).to_string(),
        );
    }
    if let Some(output) = command_text("rustc", &["--version"]) {
        values.insert("rustc".to_owned(), output);
    }
    values
}

fn system_metadata() -> BTreeMap<String, String> {
    let logical_cores = std::thread::available_parallelism()
        .map_or_else(|_| "unknown".to_owned(), |value| value.get().to_string());
    let mut values = BTreeMap::new();
    values.insert("os".to_owned(), std::env::consts::OS.to_owned());
    values.insert("os_version".to_owned(), os_version());
    values.insert("arch".to_owned(), std::env::consts::ARCH.to_owned());
    values.insert("cpu_logical_cores".to_owned(), logical_cores.clone());
    values.insert("available_parallelism".to_owned(), logical_cores);
    values.insert("cpu_model".to_owned(), cpu_model());
    values.insert("memory_total_bytes".to_owned(), memory_total_bytes());
    values.insert("process_id".to_owned(), std::process::id().to_string());
    values.insert(
        "payload_seed".to_owned(),
        crate::workload::PAYLOAD_SEED.to_string(),
    );
    values
}

fn os_version() -> String {
    if cfg!(target_os = "linux")
        && let Ok(info) = fs::read_to_string("/etc/os-release")
        && let Some(version) = info.lines().find_map(|line| {
            line.strip_prefix("PRETTY_NAME=")
                .map(|value| value.trim_matches('"').to_owned())
        })
    {
        return version;
    }
    if cfg!(target_os = "macos")
        && let Some(version) = command_text("sw_vers", &["-productVersion"])
    {
        return version;
    }
    if cfg!(target_os = "windows")
        && let Some(version) = command_text(
            "powershell",
            &[
                "-NoProfile",
                "-Command",
                "[Environment]::OSVersion.VersionString",
            ],
        )
    {
        return version;
    }
    "unknown".to_owned()
}

fn cpu_model() -> String {
    if cfg!(target_os = "linux")
        && let Ok(info) = fs::read_to_string("/proc/cpuinfo")
        && let Some(model) = info.lines().find_map(|line| {
            line.strip_prefix("model name")
                .and_then(|value| value.split_once(':'))
                .map(|(_, value)| value.trim().to_owned())
        })
    {
        return model;
    }
    if cfg!(target_os = "windows")
        && let Ok(model) = std::env::var("PROCESSOR_IDENTIFIER")
        && !model.is_empty()
    {
        return model;
    }
    if cfg!(target_os = "macos")
        && let Some(model) = command_text("sysctl", &["-n", "machdep.cpu.brand_string"])
    {
        return model;
    }
    "unknown".to_owned()
}

fn memory_total_bytes() -> String {
    if cfg!(target_os = "linux")
        && let Ok(info) = fs::read_to_string("/proc/meminfo")
        && let Some(value) = info.lines().find_map(|line| {
            line.strip_prefix("MemTotal:")
                .and_then(|value| value.split_whitespace().next())
                .and_then(|value| value.parse::<u64>().ok())
        })
    {
        return value.saturating_mul(1024).to_string();
    }
    if cfg!(target_os = "macos")
        && let Some(value) = command_text("sysctl", &["-n", "hw.memsize"])
    {
        return value;
    }
    if cfg!(target_os = "windows") {
        let script = "(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory";
        if let Some(value) = command_text("powershell", &["-NoProfile", "-Command", script]) {
            return value;
        }
    }
    "unknown".to_owned()
}

fn bounded_metadata(value: &str) -> String {
    const LIMIT: usize = 16 * 1024;
    if value.len() <= LIMIT {
        return value.to_owned();
    }
    String::from_utf8_lossy(&value.as_bytes()[..LIMIT]).into_owned()
}

fn redact_endpoint(endpoint: &str) -> String {
    let authority_start = endpoint.find("://").map_or(0, |index| index + 3);
    let authority_end = endpoint[authority_start..]
        .find('/')
        .map_or(endpoint.len(), |offset| authority_start + offset);
    let authority = &endpoint[authority_start..authority_end];
    let Some(at) = authority.rfind('@') else {
        return endpoint.to_owned();
    };
    let userinfo = &authority[..at];
    let replacement = userinfo
        .split_once(':')
        .map_or_else(|| "***".to_owned(), |(user, _)| format!("{user}:***"));
    format!(
        "{}{}{}{}",
        &endpoint[..authority_start],
        replacement,
        &authority[at..],
        &endpoint[authority_end..]
    )
}

fn command_text(program: &str, args: &[&str]) -> Option<String> {
    let output = Command::new(program).args(args).output().ok()?;
    if !output.status.success() {
        return None;
    }
    Some(String::from_utf8_lossy(&output.stdout).trim().to_owned())
}

fn process_is_alive(pid: &str) -> bool {
    if cfg!(target_os = "windows") {
        Command::new("tasklist")
            .args(["/FI", &format!("PID eq {pid}")])
            .output()
            .is_ok_and(|output| {
                output.status.success() && String::from_utf8_lossy(&output.stdout).contains(pid)
            })
    } else {
        Command::new("kill")
            .args(["-0", pid])
            .output()
            .is_ok_and(|output| output.status.success())
    }
}

#[allow(clippy::too_many_lines)]
fn render_markdown(document: &CampaignDocument) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "# Kivi campaign report\n");
    let _ = writeln!(out, "- schema: `{}`", document.schema);
    let _ = writeln!(out, "- preset: `{:?}`", document.preset);
    let _ = writeln!(out, "- generated: `{}`", document.generated_at_unix_ms);
    let _ = writeln!(out, "- valid: `{}`", document.valid);
    let _ = writeln!(out, "- Redis: `{}`", document.metadata.redis_url);
    let _ = writeln!(out, "- Kivi mode: `{:?}`", document.profile.kivi_mode);
    let _ = writeln!(out, "- output: `{}`\n", document.output_dir.display());
    let _ = writeln!(out, "## Coverage\n");
    let _ = writeln!(
        out,
        "- effective points: {}/{}; native-only: {}/{}; large-value: {}/{}; comparison trials: {}; native-only trials: {}; large-value trials: {}",
        document.coverage.actual_comparison_points,
        document.coverage.expected_comparison_points,
        document.coverage.actual_native_only_points,
        document.coverage.expected_native_only_points,
        document.coverage.actual_large_value_experiments,
        document.coverage.expected_large_value_experiments,
        document.coverage.comparison_trials_complete,
        document.coverage.native_only_trials_complete,
        document.coverage.large_value_trials_complete,
    );
    let _ = writeln!(
        out,
        "- complete: `{}`; preset-complete: `{}`\n",
        document.coverage.complete, document.coverage.preset_complete
    );
    let _ = writeln!(out, "## Correctness\n");
    for entry in &document.correctness {
        let _ = writeln!(
            out,
            "- **{}**: {} ({} / {} checks)",
            entry.class, entry.detail, entry.left_checks, entry.right_checks
        );
    }
    let _ = writeln!(out, "\n## Model validation\n");
    for entry in &document.model_validation {
        let _ = writeln!(
            out,
            "- **{} / {}**: valid={} checks={}/{} detail={} reproducer={}",
            entry.class,
            entry.point_label,
            entry.valid,
            entry.left_checks,
            entry.right_checks,
            entry.detail,
            entry.reproducer.as_deref().unwrap_or("none")
        );
    }
    let _ = writeln!(out, "\n## Comparisons\n");
    for entry in &document.comparisons {
        render_comparison(&mut out, entry);
    }
    let _ = writeln!(out, "## Value-size points\n");
    for entry in &document.value_size_points {
        render_comparison(&mut out, entry);
    }
    let _ = writeln!(out, "## Key-distribution points\n");
    for entry in &document.key_distribution_points {
        render_comparison(&mut out, entry);
    }
    let _ = writeln!(out, "## Concurrency points\n");
    for entry in &document.concurrency_points {
        render_comparison(&mut out, entry);
    }
    let _ = writeln!(out, "## Pipeline points\n");
    for entry in &document.pipeline_points {
        render_comparison(&mut out, entry);
    }
    let _ = writeln!(out, "## Large-value experiment\n");
    for entry in &document.large_value_experiments {
        let _ = writeln!(
            out,
            "### {} / {} / {} bytes\n",
            entry.class, entry.operation, entry.value_len
        );
        render_comparison(&mut out, &entry.comparison);
    }
    let _ = writeln!(out, "## Kivi native-only semantics\n");
    for entry in &document.native_only {
        let _ = writeln!(
            out,
            "- {} {}: valid={} ops/sec={:.2} p50={} p95={} p99={} p99.9={} checks={}",
            entry.workload,
            entry.key_distribution,
            entry.valid,
            entry.aggregate.median_ops_per_sec,
            entry.aggregate.median_p50_ns,
            entry.aggregate.median_p95_ns,
            entry.aggregate.median_p99_ns,
            entry.aggregate.median_p999_ns,
            entry.validation.left_checks
        );
        for failure in &entry.failures {
            let _ = writeln!(out, "  - failure: {failure}");
        }
    }
    let _ = writeln!(out, "## Reproducibility metadata\n");
    for (key, value) in &document.metadata.system {
        let _ = writeln!(out, "- system.{key}: `{value}`");
    }
    for (key, value) in &document.metadata.build {
        let _ = writeln!(out, "- build.{key}: `{value}`");
    }
    for (key, value) in &document.metadata.kivi {
        let _ = writeln!(out, "- kivi.{key}: `{value}`");
    }
    for (key, value) in &document.metadata.process {
        let _ = writeln!(out, "- process.{key}: `{value}`");
    }
    for (key, value) in &document.metadata.lifecycle {
        let _ = writeln!(out, "- lifecycle.{key}: `{value}`");
    }
    let _ = writeln!(
        out,
        "- redis.url_source: `{}`",
        document.metadata.redis_url_source
    );
    let _ = writeln!(
        out,
        "- redis.version: `{}`",
        document.metadata.redis.version
    );
    let _ = writeln!(
        out,
        "- redis.persistence: appendonly={} appendfsync={} save={} maxmemory={} maxmemory-policy={} io-threads={}",
        document.metadata.redis.appendonly,
        document.metadata.redis.appendfsync,
        document.metadata.redis.save,
        document.metadata.redis.maxmemory,
        document.metadata.redis.maxmemory_policy,
        document.metadata.redis.io_threads,
    );
    let _ = writeln!(
        out,
        "- redis.server: mode={} process_id={} process_threads={} os={} arch_bits={} config_file={} executable={} tcp_port={}",
        document.metadata.redis.mode,
        document.metadata.redis.process_id,
        document.metadata.redis.process_threads,
        document.metadata.redis.os,
        document.metadata.redis.arch_bits,
        document.metadata.redis.config_file,
        document.metadata.redis.executable,
        document.metadata.redis.tcp_port,
    );
    if !document.failures.is_empty() {
        let _ = writeln!(out, "\n## Failures\n");
        for failure in &document.failures {
            let _ = writeln!(out, "- {failure}");
        }
    }
    out
}

#[allow(clippy::too_many_lines)]
fn render_comparison(out: &mut String, entry: &ComparisonReport) {
    let _ = writeln!(out, "### {} — {}\n", entry.class, entry.workload);
    let _ = writeln!(
        out,
        "- point: dimension={} profile={} value={} keys={} threads={} pipeline={} saturation={} valid={}",
        entry.dimension,
        entry.profile,
        entry.value_label,
        entry.key_distribution,
        entry.threads,
        entry.pipeline,
        entry.saturation,
        entry.valid
    );
    let _ = writeln!(
        out,
        "- endpoints: {}[{}]={} | {}[{}]={}",
        entry.left_target,
        entry.left_endpoint_type,
        entry.left_endpoint,
        entry.right_target,
        entry.right_endpoint_type,
        entry.right_endpoint
    );
    let _ = writeln!(
        out,
        "- verdict: outcome={} winner={} metric={} delta={:.3} noise_threshold={:.3}",
        entry.verdict.outcome,
        entry.verdict.winner.as_deref().unwrap_or("none"),
        entry.verdict.metric,
        entry.verdict.relative_delta,
        entry.verdict.noise_threshold
    );
    let _ = writeln!(
        out,
        "- validation: valid={} checks={}/{} detail={} reproducer={}",
        entry.validation.valid,
        entry.validation.left_checks,
        entry.validation.right_checks,
        entry.validation.detail,
        entry.validation.reproducer.as_deref().unwrap_or("none")
    );
    let _ = writeln!(out, "- classification: `{}`", entry.classification);
    let _ = writeln!(
        out,
        "- {} median: {:.2} ops/sec, p50/p95/p99/p99.9={} / {} / {} / {} ns/op, spreads={}/{}/{}/{} ns, payload={:.2} B/op, payload throughput={:.2} B/sec, transport={}",
        entry.left_target,
        entry.left_aggregate.median_ops_per_sec,
        entry.left_aggregate.median_p50_ns,
        entry.left_aggregate.median_p95_ns,
        entry.left_aggregate.median_p99_ns,
        entry.left_aggregate.median_p999_ns,
        entry.left_aggregate.spread_p50_ns,
        entry.left_aggregate.spread_p95_ns,
        entry.left_aggregate.spread_p99_ns,
        entry.left_aggregate.spread_p999_ns,
        entry
            .trials
            .iter()
            .find(|trial| trial.target == entry.left_target)
            .map_or(0.0, |trial| trial.payload_bytes_per_op),
        entry.left_aggregate.median_effective_payload_bytes_per_sec,
        entry.left_aggregate.median_bytes_per_sec.map_or_else(
            || "unavailable".to_owned(),
            |value| format!("{value:.2} B/sec")
        )
    );
    let _ = writeln!(
        out,
        "- {} median: {:.2} ops/sec, p50/p95/p99/p99.9={} / {} / {} / {} ns/op, spreads={}/{}/{}/{} ns, payload={:.2} B/op, payload throughput={:.2} B/sec, transport={}",
        entry.right_target,
        entry.right_aggregate.median_ops_per_sec,
        entry.right_aggregate.median_p50_ns,
        entry.right_aggregate.median_p95_ns,
        entry.right_aggregate.median_p99_ns,
        entry.right_aggregate.median_p999_ns,
        entry.right_aggregate.spread_p50_ns,
        entry.right_aggregate.spread_p95_ns,
        entry.right_aggregate.spread_p99_ns,
        entry.right_aggregate.spread_p999_ns,
        entry
            .trials
            .iter()
            .find(|trial| trial.target == entry.right_target)
            .map_or(0.0, |trial| trial.payload_bytes_per_op),
        entry.right_aggregate.median_effective_payload_bytes_per_sec,
        entry.right_aggregate.median_bytes_per_sec.map_or_else(
            || "unavailable".to_owned(),
            |value| format!("{value:.2} B/sec")
        )
    );
    if let Some(value) = entry.ratios.throughput_ratio {
        let _ = writeln!(out, "- left/right throughput ratio: {value:.3}");
    }
    if let Some(value) = entry.ratios.p50_ratio {
        let _ = writeln!(out, "- left/right p50 ratio: {value:.3}");
    }
    if let Some(value) = entry.ratios.p95_ratio {
        let _ = writeln!(out, "- left/right p95 ratio: {value:.3}");
    }
    if let Some(value) = entry.ratios.p99_ratio {
        let _ = writeln!(out, "- left/right p99 ratio: {value:.3}");
    }
    if let Some(value) = entry.ratios.p999_ratio {
        let _ = writeln!(out, "- left/right p99.9 ratio: {value:.3}");
    }
    if let Some(value) = entry.ratios.effective_payload_ratio {
        let _ = writeln!(out, "- left/right effective payload ratio: {value:.3}");
    }
    if let Some(value) = entry.ratios.bytes_per_sec_ratio {
        let _ = writeln!(out, "- left/right transport bytes/sec ratio: {value:.3}");
    }
    if let Some(value) = entry.ratios.scaling_ratio {
        let _ = writeln!(out, "- left/right per-thread scaling ratio: {value:.3}");
    }
    for trial in &entry.trials {
        let _ = writeln!(
            out,
            "- trial {} {}: completed={} expected={} errors={} warmup_errors={} ops/sec={:.2} p50={} p95={} p99={} p99.9={} max={} ns/op (batch p95={} ns) bytes_out={} bytes_in={} transport/sec={} payload={:.2} B/op payload/sec={:.2} elapsed={:.3}s duration={:.3}s target_stats={}",
            trial.trial,
            trial.target,
            trial.completed,
            trial
                .expected_completed
                .map_or_else(|| "-".to_owned(), |value| value.to_string()),
            trial.errors,
            trial.warmup_errors,
            trial.ops_per_sec,
            trial.p50_ns,
            trial.p95_ns,
            trial.p99_ns,
            trial.p999_ns,
            trial.max_ns,
            trial.batch_p95_ns,
            trial
                .bytes_out
                .map_or_else(|| "-".to_owned(), |value| value.to_string()),
            trial
                .bytes_in
                .map_or_else(|| "-".to_owned(), |value| value.to_string()),
            trial
                .bytes_per_sec
                .map_or_else(|| "unavailable".to_owned(), |value| format!("{value:.2}")),
            trial.payload_bytes_per_op,
            trial.effective_payload_bytes_per_sec,
            trial.elapsed_secs,
            trial.duration_secs,
            trial
                .target_stats
                .map_or_else(|| "unavailable".to_owned(), |stats| format!("{stats:?}")),
        );
    }
    for caveat in &entry.caveats {
        let _ = writeln!(out, "- caveat: {caveat}");
    }
    for failure in &entry.failures {
        let _ = writeln!(out, "- failure: {failure}");
    }
    let _ = writeln!(out);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resp_client::Reply;
    use crate::workload::WorkloadOp;

    #[test]
    fn presets_are_bounded_and_distinct() {
        let smoke = CampaignPreset::Smoke.profile();
        let standard = CampaignPreset::Standard.profile();
        let full = CampaignPreset::Full.profile();
        assert!(smoke.duration < standard.duration);
        assert!(standard.duration < full.duration);
        assert!(smoke.thread_counts.len() < full.thread_counts.len());
        assert!(full.large_values.contains(&(64 * 1024 * 1024)));
    }

    #[test]
    fn classification_keeps_target_labels() {
        assert_eq!(
            classify_comparison("left", "right", 100.0, 100.0, 10, 10, None, None),
            "tied"
        );
        assert_eq!(
            classify_comparison("left", "right", 120.0, 100.0, 10, 10, None, None),
            "throughput:left"
        );
        assert_eq!(
            classify_comparison("left", "right", 100.0, 100.0, 8, 10, None, None),
            "latency:left"
        );
        assert_eq!(
            classify_comparison(
                "left",
                "right",
                100.0,
                100.0,
                10,
                10,
                Some(10.0),
                Some(20.0),
            ),
            "bandwidth:left"
        );
        assert_eq!(
            scaling_classification("left", "right", 120.0, 100.0),
            Some("scaling:left".to_owned())
        );
    }

    #[test]
    fn strict_reply_shapes_reject_wrong_frames() {
        let op = WorkloadOp::Set {
            key: "k".to_owned(),
            len: 1,
        };
        assert!(check_workload_reply(&Reply::Simple(b"OK".to_vec()), &op).is_ok());
        assert!(check_workload_reply(&Reply::Integer(1), &op).is_err());
        assert!(check_workload_reply(&Reply::Bulk(None), &WorkloadOp::Get("k".to_owned())).is_ok());
    }

    #[test]
    fn key_space_labels_are_stable() {
        assert_eq!(key_space_label(KeySpace::Hot), "hot");
        assert_eq!(key_space_label(KeySpace::Uniform(4)), "uniform:4");
    }

    #[test]
    fn endpoint_redaction_preserves_host_and_hides_password() {
        assert_eq!(
            redact_endpoint("redis://user:secret@localhost:6379/0"),
            "redis://user:***@localhost:6379/0"
        );
        assert_eq!(
            redact_endpoint("redis://localhost:6379"),
            "redis://localhost:6379"
        );
        assert_eq!(
            redact_endpoint("user:secret@localhost:6379"),
            "user:***@localhost:6379"
        );
        assert_eq!(
            redact_endpoint("token@localhost:6379"),
            "***@localhost:6379"
        );
    }

    #[test]
    fn full_preset_covers_the_required_matrix() {
        let profile = CampaignPreset::Full.profile();
        assert_eq!(
            profile.value_sizes,
            vec![
                16,
                1024,
                64 * 1024,
                256 * 1024,
                1024 * 1024,
                4 * 1024 * 1024
            ]
        );
        assert_eq!(
            profile
                .key_distributions
                .iter()
                .map(|point| point.label.as_str())
                .collect::<Vec<_>>(),
            vec!["hot", "small", "uniform256", "uniform4k", "large"]
        );
        assert_eq!(profile.thread_counts, vec![1, 2, 4, 8, 16, 32, 64]);
        assert_eq!(profile.pipeline_depths, vec![1, 4, 16, 64, 256]);
        assert!(
            profile
                .named_profiles
                .iter()
                .any(|profile| profile.name == "balanced")
        );
        assert_eq!(value_label(64 * 1024 * 1024), "64MiB");
    }

    #[test]
    fn spec_sections_keep_independent_dimensions() {
        let profile = CampaignPreset::Full.profile();
        let specs = build_specs(&CampaignOptions::default(), &profile);
        assert_eq!(specs.value_sizes.len(), 6);
        assert_eq!(specs.key_distributions.len(), 5);
        assert_eq!(specs.concurrency.len(), 7);
        assert_eq!(specs.pipelines.len(), 5);
        assert!(specs.named.iter().all(|spec| !spec.curve));
        assert!(specs.value_sizes.iter().all(|spec| spec.curve));
        assert!(specs.key_distributions.iter().all(|spec| spec.curve));
        assert!(specs.concurrency.iter().all(|spec| spec.curve));
        assert!(specs.pipelines.iter().all(|spec| spec.curve));
    }

    #[test]
    fn model_state_tracks_operation_transitions() {
        let mut state = ModelState::default();
        let payload = b"abc";
        let key = "k".to_owned();
        let counter = "c".to_owned();
        let operations = [
            WorkloadOp::Set {
                key: key.clone(),
                len: payload.len(),
            },
            WorkloadOp::Get(key.clone()),
            WorkloadOp::Exists(key.clone()),
            WorkloadOp::GetRange(key.clone()),
            WorkloadOp::SetRange(key.clone()),
            WorkloadOp::Expire(key.clone()),
            WorkloadOp::Ttl(key.clone()),
            WorkloadOp::CounterAdd(counter.clone()),
            WorkloadOp::Delete(key.clone()),
        ];
        for operation in &operations {
            model_update(operation, &mut state, payload);
        }
        assert_eq!(state.values.get(&key), None);
        assert!(!state.expiries.contains(&key));
        assert_eq!(state.counters.get(&counter), Some(&1));
        assert_eq!(
            model_expected(
                &WorkloadOp::CounterAdd(counter.clone()),
                &state,
                payload,
                true
            ),
            "counter:2"
        );
    }

    #[test]
    fn mixed_model_sequence_updates_byte_and_counter_state() {
        let keys = vec!["k0".to_owned(), "k1".to_owned()];
        let mut state = ModelState::default();
        for index in 0..20 {
            let operation = workload_op_for(Workload::Mixed, 8, index, &keys, "counter");
            model_update(&operation, &mut state, &[7; 8]);
        }
        assert_eq!(state.counters.get("counter"), Some(&5));
        assert!(state.values.values().all(|value| value.len() <= 8));
    }

    #[test]
    fn normalized_latency_and_payload_metrics_are_explicit() {
        assert_eq!(normalize_latency(1_000, 4), 250);
        assert_eq!(normalize_latency(1_000, 0), 1_000);
        assert!((payload_bytes_per_op(Workload::Get, 32) - 32.0).abs() < f64::EPSILON);
        assert!(payload_bytes_per_op(Workload::Balanced, 32) > 0.0);
    }

    #[test]
    fn verdict_reports_tie_noise_and_winner() {
        assert_eq!(
            comparison_verdict("left", "right", 100.0, 100.0).outcome,
            "tie"
        );
        assert_eq!(
            comparison_verdict("left", "right", 102.0, 100.0).outcome,
            "noise"
        );
        let winner = comparison_verdict("left", "right", 120.0, 100.0);
        assert_eq!(winner.outcome, "left-win");
        assert_eq!(winner.winner.as_deref(), Some("left"));
    }

    #[test]
    fn reproducer_is_minimal_and_machine_readable() {
        let error = "reproducer=GET k:1; expected=bulk:\"v\"; actual=nil";
        assert_eq!(reproducer_from_error(error).as_deref(), Some("GET k:1"));
        assert_eq!(first_reproducer(&[error]).as_deref(), Some("GET k:1"));
    }
}
