//! Phase 12 production controller runtime for the single-node engine.
//!
//! The runtime owns bounded observation windows, the deterministic baseline,
//! optional learned shadowing, and the narrow actuator that applies memory
//! policy through the engine's worker control channel. It never enters the
//! request path or consensus state.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use kivi_control::{
    Action, ActionCost, ActionOutcome, ActionScope, ActuationError, Actuator, ActuatorReceipt,
    AdaptiveConfig, AdaptiveController, AdaptiveMode, BaselineConfig, BaselineController,
    ControllerFrame, ControllerSource, LearnedController, MemoryBudget, RepairBudget,
    SafetyContext, ScrubBudget,
};
use kivi_engine::AdminHandle;
use kivi_observation::{
    CheckpointDebt, ComputeSignal, Confidence, ConsensusLagSignal, DegradedAssetsSignal,
    ExecutionCriticalitySignal, HotKeyId, HotKeySignal, IoSignal, MaintenanceDebtSignal,
    MemoryPressureSignal, MemorySignal, NetworkBytes, ObservationFabric, ObservationLimits,
    ObservationMetadata, ObservationPeriod, ObservationSample, ObservationScope,
    ObservationSequence, ObservationSource, ObservationStamp, ObservationWindowId, QueueingSignal,
    RedundancySignal, RepairDebt, RequestSignal, ScrubDebt, StorageBytes, TopologyImbalanceSignal,
    UnitInterval,
};
use kivi_types::{ClusterId, NamespaceId, TabletId, Ticks, WorkerId};

/// Configuration for the production adaptive runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AdaptiveRuntimeConfig {
    /// Control interval.
    pub interval: Duration,
    /// Whether generated actions are shadow-only.
    pub shadow: bool,
    /// Maximum retained observation scopes.
    pub max_scopes: usize,
    /// Maximum retained typed samples per scope.
    pub max_samples_per_scope: usize,
    /// Maximum retained latency samples per scope.
    pub max_latency_samples: usize,
    /// Maximum retained hot keys per scope.
    pub hot_key_capacity: usize,
}

impl Default for AdaptiveRuntimeConfig {
    fn default() -> Self {
        Self {
            interval: Duration::from_millis(500),
            shadow: false,
            max_scopes: 256,
            max_samples_per_scope: 32,
            max_latency_samples: 16,
            hot_key_capacity: 64,
        }
    }
}

/// Runtime counters and current controller state for the admin plane.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdaptiveRuntimeSnapshot {
    /// Current execution mode.
    pub mode: &'static str,
    /// Active controller source.
    pub active_controller: &'static str,
    /// Number of completed observation/control intervals.
    pub intervals: u64,
    /// Latest accepted observation sequence.
    pub observation_sequence: Option<u64>,
    /// Latest observation window.
    pub observation_window: u64,
    /// Number of retained observation scopes.
    pub observation_scopes: usize,
    /// Freshness classification of the latest target evidence.
    pub observation_freshness: &'static str,
    /// Baseline proposed count.
    pub proposed: u64,
    /// Baseline accepted count.
    pub accepted: u64,
    /// Baseline rejected count.
    pub rejected: u64,
    /// Baseline suppressed count.
    pub suppressed: u64,
    /// Baseline completed count.
    pub completed: u64,
    /// Baseline failed count.
    pub failed: u64,
    /// Number of active ledger actions.
    pub active_actions: usize,
    /// Number of retained decision traces.
    pub traces: usize,
    /// Last bounded collection or actuation error.
    pub last_error: Option<String>,
}

#[derive(Debug)]
struct RuntimeState {
    observations: ObservationFabric,
    controller: AdaptiveController,
    sequence: ObservationSequence,
    window: ObservationWindowId,
    ticks: Ticks,
    period: Duration,
    previous_worker_ops: BTreeMap<u64, u64>,
    previous_fabric: BTreeMap<u64, kivi_engine::FabricStatsSnapshot>,
    previous_commit: BTreeMap<u64, kivi_engine::CommitMetricsSnapshot>,
    current_policy: kivi_memory::MemoryControlPolicy,
    current_memory_budget_ppm: u32,
    last_snapshot: Option<kivi_observation::ObservationSnapshot>,
    intervals: u64,
    last_error: Option<String>,
}

struct RuntimeInputs {
    workers: Vec<(WorkerId, kivi_engine::WorkerMetrics)>,
    fabrics: Vec<(WorkerId, kivi_engine::FabricWorkerReport)>,
    commits: Vec<(WorkerId, Option<kivi_engine::CommitMetricsSnapshot>)>,
    checkpoint: kivi_engine::CheckpointAdminState,
    redundancy: Option<kivi_engine::RedundancyAdminSnapshot>,
    memory: Vec<(WorkerId, Vec<(u64, kivi_memory::AccessSignals)>)>,
}

/// Asynchronous controller runtime shared by the engine and admin plane.
#[derive(Debug)]
pub struct AdaptiveRuntime {
    engine: AdminHandle,
    cluster: ClusterId,
    namespace: NamespaceId,
    config: AdaptiveRuntimeConfig,
    state: Arc<Mutex<RuntimeState>>,
}

impl AdaptiveRuntime {
    /// Creates a runtime with bounded observation and action history.
    #[must_use]
    pub fn new(
        engine: AdminHandle,
        cluster: ClusterId,
        namespace: NamespaceId,
        config: AdaptiveRuntimeConfig,
    ) -> Self {
        let max_samples_per_scope = config.max_samples_per_scope.max(8);
        let config = AdaptiveRuntimeConfig {
            interval: if config.interval.is_zero() {
                Duration::from_millis(1)
            } else {
                config.interval
            },
            max_scopes: config.max_scopes.max(1),
            max_samples_per_scope,
            max_latency_samples: config.max_latency_samples.max(1).min(max_samples_per_scope),
            hot_key_capacity: config.hot_key_capacity.max(1),
            ..config
        };
        let limits = ObservationLimits::new(
            config.max_scopes,
            config.max_samples_per_scope,
            config.max_latency_samples,
            config.hot_key_capacity,
            Duration::from_secs(5),
        )
        .unwrap_or_else(|| {
            ObservationLimits::new(256, 32, 16, 64, Duration::from_secs(5))
                .expect("fallback observation limits are valid")
        });
        let mode = if config.shadow {
            AdaptiveMode::Shadow
        } else {
            AdaptiveMode::Active
        };
        let controller = AdaptiveController::new(
            AdaptiveConfig {
                mode,
                ..AdaptiveConfig::default()
            },
            BaselineController::new(BaselineConfig::default()),
            Some(LearnedController::default()),
        );
        Self {
            engine,
            cluster,
            namespace,
            config,
            state: Arc::new(Mutex::new(RuntimeState {
                observations: ObservationFabric::new(limits),
                controller,
                sequence: ObservationSequence::FIRST,
                window: ObservationWindowId::FIRST,
                ticks: Ticks::from_micros(1),
                period: config.interval,
                previous_worker_ops: BTreeMap::new(),
                previous_fabric: BTreeMap::new(),
                previous_commit: BTreeMap::new(),
                current_policy: kivi_memory::MemoryControlPolicy::default(),
                current_memory_budget_ppm: 700_000,
                last_snapshot: None,
                intervals: 0,
                last_error: None,
            })),
        }
    }

    /// Returns the configured interval.
    #[must_use]
    pub const fn interval(&self) -> Duration {
        self.config.interval
    }

    /// Returns whether the runtime is in shadow mode.
    #[must_use]
    pub const fn is_shadow(&self) -> bool {
        self.config.shadow
    }

    /// Runs until `shutdown` becomes true.
    pub fn spawn(
        self: Arc<Self>,
        mut shutdown: tokio::sync::watch::Receiver<bool>,
    ) -> tokio::task::JoinHandle<()> {
        let runtime = Arc::clone(&self);
        tokio::spawn(async move {
            let mut ticker = tokio::time::interval(runtime.config.interval);
            ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
            loop {
                tokio::select! {
                    _ = ticker.tick() => runtime.clone().tick().await,
                    changed = shutdown.changed() => {
                        if changed.is_err() || *shutdown.borrow() {
                            break;
                        }
                    }
                }
            }
        })
    }

    /// Runs one observation, decision, and bounded actuation interval.
    pub async fn tick(self: Arc<Self>) {
        let this = Arc::clone(&self);
        let _ = tokio::task::spawn_blocking(move || this.tick_blocking()).await;
    }

    /// Returns an operator snapshot.
    #[must_use]
    pub fn snapshot(&self) -> AdaptiveRuntimeSnapshot {
        let state = self.state.lock().expect("adaptive state mutex");
        let target = state.last_snapshot.as_ref().and_then(|snapshot| {
            snapshot
                .scopes
                .iter()
                .find(|scope| matches!(scope.scope, ObservationScope::Tablet(_)))
                .or_else(|| snapshot.scopes.first())
        });
        let freshness = target.map_or("missing", |scope| match scope.evidence.freshness.status {
            kivi_observation::FreshnessStatus::Current => "current",
            kivi_observation::FreshnessStatus::Stale => "stale",
            kivi_observation::FreshnessStatus::Future => "future",
        });
        let baseline = state
            .controller
            .stats()
            .get(&ControllerSource::Baseline)
            .copied()
            .unwrap_or_default();
        AdaptiveRuntimeSnapshot {
            mode: if self.is_shadow() {
                "shadow"
            } else {
                match state.controller.mode() {
                    AdaptiveMode::Active => "active",
                    AdaptiveMode::Shadow => "shadow",
                }
            },
            active_controller: "baseline",
            intervals: state.intervals,
            observation_sequence: state
                .last_snapshot
                .as_ref()
                .and_then(|snapshot| snapshot.last_sequence)
                .map(ObservationSequence::as_u64),
            observation_window: state.window.as_u64(),
            observation_scopes: state.observations.scope_count(),
            observation_freshness: freshness,
            proposed: baseline.proposed,
            accepted: baseline.accepted,
            rejected: baseline.rejected,
            suppressed: baseline.suppressed,
            completed: baseline.completed,
            failed: baseline.failed,
            active_actions: state.controller.ledger().active_records().count(),
            traces: state.controller.traces().count(),
            last_error: state.last_error.clone(),
        }
    }

    fn tick_blocking(self: Arc<Self>) {
        let inputs = self.collect_inputs();
        let mut state = self.state.lock().expect("adaptive state mutex");
        state.last_error = None;
        state.observations.begin_window();
        let mut samples = Vec::new();
        self.build_samples(&mut state, &inputs, &mut samples);
        for sample in &samples {
            if let Err(error) = state.observations.collect(sample) {
                state.last_error = Some(error.to_string());
                break;
            }
        }
        let snapshot = state.observations.snapshot(state.ticks);
        let target = snapshot
            .scopes
            .iter()
            .find(|scope| matches!(scope.scope, ObservationScope::Tablet(_)))
            .or_else(|| snapshot.scopes.first());
        let Some(target) = target else {
            state.last_error = Some("no typed observation scope".to_owned());
            return;
        };
        let tablet = match target.scope {
            ObservationScope::Tablet(tablet) => tablet,
            _ => TabletId::from_u64(1),
        };
        let context = self.safety_context(&state, tablet, target);
        let frame = ControllerFrame::new(
            snapshot.clone(),
            ActionScope::new(self.cluster, self.namespace),
            snapshot.last_sequence.unwrap_or(ObservationSequence::FIRST),
            state.window,
            1,
            state.ticks,
        );
        let decision = state.controller.decide(&frame, &context);
        let mut actuator = MemoryActuator {
            engine: self.engine.clone(),
            policy: state.current_policy,
            budget_ppm: state.current_memory_budget_ppm,
        };
        for record in decision.active_actions {
            let observation = frame.action_observation();
            if let Err(error) = state
                .controller
                .execute(&mut actuator, record.id, observation)
            {
                state.last_error = Some(error.to_string());
            }
        }
        state.current_policy = actuator.policy;
        state.current_memory_budget_ppm = actuator.budget_ppm;
        state.last_snapshot = Some(snapshot);
        state.intervals = state.intervals.saturating_add(1);
        state.ticks = state.ticks.advance_by(state.period);
        if let Ok(next) = state.window.next() {
            state.window = next;
        }
    }

    fn collect_inputs(&self) -> RuntimeInputs {
        let engine = self.engine.clone();
        let workers = engine.worker_metrics_snapshot();
        let fabrics = engine.fabric_stats_snapshot();
        let commits = engine.commit_metrics_snapshot();
        let checkpoint = engine.checkpoint_state();
        let redundancy = engine.redundancy_snapshot();
        let memory = engine.memory_signals_snapshot();
        RuntimeInputs {
            workers,
            fabrics,
            commits,
            checkpoint,
            redundancy,
            memory,
        }
    }

    #[allow(clippy::too_many_lines)]
    fn build_samples(
        &self,
        state: &mut RuntimeState,
        inputs: &RuntimeInputs,
        samples: &mut Vec<ObservationSample>,
    ) {
        let period = ObservationPeriod::new(state.period)
            .unwrap_or_else(|| ObservationPeriod::new(Duration::from_millis(1)).expect("period"));
        let metadata = ObservationMetadata::new(
            state.ticks,
            ObservationSource::RuntimeCounter,
            Confidence::High,
        );
        let mut workers = inputs.workers.clone();
        workers.sort_by_key(|(worker, _)| worker.as_u64());
        let mut fabrics = inputs.fabrics.clone();
        fabrics.sort_by_key(|(worker, _)| worker.as_u64());
        let mut commits = inputs.commits.clone();
        commits.sort_by_key(|(worker, _)| worker.as_u64());
        let mut memory = inputs.memory.clone();
        memory.sort_by_key(|(worker, _)| worker.as_u64());
        let mut total_ops = 0u64;
        let mut total_resident = 0u64;
        let mut total_logical = 0u64;
        let mut max_pressure = 0.0f64;
        let mut max_queue = 0u64;
        let mut max_latency = 0u64;
        let hot_key_limit = state
            .observations
            .limits()
            .max_samples_per_scope()
            .saturating_sub(7)
            .min(32);
        for (worker, metrics) in &workers {
            let total = metrics.channel_ops.saturating_add(metrics.direct_ops);
            let previous = state
                .previous_worker_ops
                .get(&worker.as_u64())
                .copied()
                .unwrap_or(total);
            let delta = total.saturating_sub(previous);
            state.previous_worker_ops.insert(worker.as_u64(), total);
            let scope = ObservationScope::Worker(*worker);
            let commit = commits
                .iter()
                .find(|(id, _)| id == worker)
                .and_then(|(_, snapshot)| snapshot.clone())
                .unwrap_or_default();
            let fabric = fabrics
                .iter()
                .find(|(id, _)| id == worker)
                .map(|(_, report)| report.fabric)
                .unwrap_or_default();
            let pressure = fabric_stats_pressure(&fabrics, *worker);
            let barrier = duration_ns(commit.barrier_latency_avg);
            self.push_sample(state, samples, scope, metadata, period, TypedRequest(delta));
            self.push_sample(
                state,
                samples,
                scope,
                metadata,
                period,
                TypedQueueing(QueueingSignal {
                    depth: commit.queue_depth as u64,
                    wait_p50_ns: duration_ns(commit.oldest_wait_avg),
                    wait_p99_ns: duration_ns(commit.oldest_wait_max),
                    wait_p999_ns: duration_ns(commit.oldest_wait_max),
                    admission_rejections: 0,
                }),
            );
            self.push_sample(
                state,
                samples,
                scope,
                metadata,
                period,
                TypedCompute(ComputeSignal {
                    cpu_time_ns: barrier.saturating_mul(delta),
                    service_time_ns: barrier.saturating_mul(delta),
                }),
            );
            let resident = fabric_resident_bytes(fabric);
            self.push_sample(
                state,
                samples,
                scope,
                metadata,
                period,
                TypedMemory(MemorySignal {
                    resident_bytes: resident,
                    logical_bytes: resident,
                }),
            );
            self.push_sample(
                state,
                samples,
                scope,
                metadata,
                period,
                TypedMemoryPressure(MemoryPressureSignal {
                    pressure: UnitInterval::new(pressure).unwrap_or(UnitInterval::ZERO),
                    reclaim_attempts: 0,
                    reclaim_failures: 0,
                }),
            );
            let migration_delta = fabric.migration_bytes.saturating_sub(
                state
                    .previous_fabric
                    .get(&worker.as_u64())
                    .map_or(0, |old| old.migration_bytes),
            );
            self.push_sample(
                state,
                samples,
                scope,
                metadata,
                period,
                TypedIo(IoSignal {
                    network: NetworkBytes::default(),
                    storage: StorageBytes {
                        read: 0,
                        written: migration_delta,
                        synced: 0,
                    },
                }),
            );
            let signals = memory
                .iter()
                .find(|(id, _)| id == worker)
                .map_or(&[][..], |(_, signals)| signals.as_slice());
            let criticality = aggregate_criticality(signals);
            self.push_sample(
                state,
                samples,
                scope,
                metadata,
                period,
                TypedCriticality(criticality),
            );
            for (object, access) in signals.iter().take(hot_key_limit) {
                self.push_sample(
                    state,
                    samples,
                    scope,
                    metadata,
                    period,
                    TypedHotKey(hot_key_signal(*object, access)),
                );
            }
            total_ops = total_ops.saturating_add(delta);
            total_resident = total_resident.saturating_add(resident);
            total_logical = total_logical.saturating_add(resident);
            max_pressure = max_pressure.max(pressure);
            max_queue = max_queue.max(commit.queue_depth as u64);
            max_latency = max_latency.max(barrier);
            state.previous_fabric.insert(worker.as_u64(), fabric);
            if let Some(commit) = commits
                .iter()
                .find(|(id, _)| id == worker)
                .and_then(|(_, snapshot)| snapshot.clone())
            {
                state.previous_commit.insert(worker.as_u64(), commit);
            }
        }
        for (worker, report) in &fabrics {
            for (tablet, footprint) in &report.tablets {
                let resident = footprint
                    .inline_bytes
                    .saturating_add(footprint.dram_bytes)
                    .saturating_add(footprint.compressed_bytes)
                    .saturating_add(footprint.nvme_bytes);
                let scope = ObservationScope::Tablet(*tablet);
                self.push_sample(
                    state,
                    samples,
                    scope,
                    metadata,
                    period,
                    TypedMemory(MemorySignal {
                        resident_bytes: resident,
                        logical_bytes: resident,
                    }),
                );
                self.push_sample(
                    state,
                    samples,
                    scope,
                    metadata,
                    period,
                    TypedMemoryPressure(MemoryPressureSignal {
                        pressure: UnitInterval::new(report.arena_pressure)
                            .unwrap_or(UnitInterval::ZERO),
                        reclaim_attempts: 0,
                        reclaim_failures: 0,
                    }),
                );
                self.push_sample(
                    state,
                    samples,
                    scope,
                    metadata,
                    period,
                    TypedCompute(ComputeSignal {
                        cpu_time_ns: 0,
                        service_time_ns: 0,
                    }),
                );
                self.push_sample(
                    state,
                    samples,
                    scope,
                    metadata,
                    period,
                    TypedCriticality(ExecutionCriticalitySignal {
                        accesses: 0,
                        serial_accesses: 0,
                        critical_path_accesses: 0,
                        tail_accesses: 0,
                        stall_time_ns: 0,
                        access_latency_ns: 1,
                        logical_bytes: resident.max(1),
                    }),
                );
            }
            let _ = worker;
        }
        let redundancy_metrics = inputs.redundancy.as_ref().map(|snapshot| snapshot.metrics);
        let repair_debt = redundancy_metrics.map_or(
            RepairDebt {
                assets: 0,
                bytes: 0,
                oldest_age: Duration::ZERO,
            },
            |metrics| RepairDebt {
                assets: metrics.pending_repairs,
                bytes: metrics.repair_bytes_pending,
                oldest_age: Duration::from_millis(metrics.degraded_time_ms),
            },
        );
        let scrub_debt = redundancy_metrics.map_or(
            ScrubDebt {
                bytes: 0,
                oldest_age: Duration::ZERO,
            },
            |metrics| ScrubDebt {
                bytes: metrics.scrub_bytes,
                oldest_age: Duration::from_millis(metrics.last_scrub_age_ms),
            },
        );
        let cluster = ObservationScope::Cluster(self.cluster);
        self.push_sample(
            state,
            samples,
            cluster,
            metadata,
            period,
            TypedRequest(total_ops),
        );
        self.push_sample(
            state,
            samples,
            cluster,
            metadata,
            period,
            TypedMemory(MemorySignal {
                resident_bytes: total_resident,
                logical_bytes: total_logical,
            }),
        );
        self.push_sample(
            state,
            samples,
            cluster,
            metadata,
            period,
            TypedMemoryPressure(MemoryPressureSignal {
                pressure: UnitInterval::new(max_pressure).unwrap_or(UnitInterval::ZERO),
                reclaim_attempts: 0,
                reclaim_failures: 0,
            }),
        );
        self.push_sample(
            state,
            samples,
            cluster,
            metadata,
            period,
            TypedMaintenance(MaintenanceDebtSignal {
                checkpoint: CheckpointDebt {
                    bytes: inputs.checkpoint.wal_retained_bytes,
                    oldest_age: Duration::ZERO,
                },
                repair: repair_debt,
                scrub: scrub_debt,
            }),
        );
        if let Some(metrics) = redundancy_metrics {
            self.push_sample(
                state,
                samples,
                cluster,
                metadata,
                period,
                TypedRedundancy(RedundancySignal {
                    logical_bytes: metrics.logical_bytes,
                    physical_bytes: metrics.physical_bytes,
                    fragments: inputs
                        .redundancy
                        .as_ref()
                        .map_or(0, |snapshot| snapshot.census_fragments),
                }),
            );
            self.push_sample(
                state,
                samples,
                cluster,
                metadata,
                period,
                TypedDegraded(DegradedAssetsSignal {
                    assets: metrics.degraded,
                    oldest_age: Duration::from_millis(metrics.degraded_time_ms),
                }),
            );
        }
        self.push_sample(
            state,
            samples,
            cluster,
            metadata,
            period,
            TypedTopology(TopologyImbalanceSignal {
                busiest_worker_load: total_ops,
                mean_worker_load: total_ops / u64::try_from(workers.len().max(1)).unwrap_or(1),
                busiest_tablet_load: total_resident,
                mean_tablet_load: total_logical / u64::try_from(fabrics.len().max(1)).unwrap_or(1),
            }),
        );
        self.push_sample(
            state,
            samples,
            cluster,
            metadata,
            period,
            TypedConsensus(ConsensusLagSignal {
                commit_lag_ns: max_latency,
                replica_lag_ns: 0,
            }),
        );
        let _ = max_queue;
    }

    #[allow(clippy::unused_self)]
    fn push_sample<S>(
        &self,
        state: &mut RuntimeState,
        samples: &mut Vec<ObservationSample>,
        scope: ObservationScope,
        metadata: ObservationMetadata,
        period: ObservationPeriod,
        signal: S,
    ) where
        S: Into<RuntimeSignal>,
    {
        let signal = signal.into().into_observation();
        let stamp = ObservationStamp {
            window: state.window,
            sequence: state.sequence,
        };
        samples.push(ObservationSample::new(
            scope, stamp, period, metadata, signal,
        ));
        if let Ok(next) = state.sequence.next() {
            state.sequence = next;
        }
    }

    fn safety_context(
        &self,
        state: &RuntimeState,
        tablet: TabletId,
        snapshot: &kivi_observation::ScopeSnapshot,
    ) -> SafetyContext {
        let scope = ActionScope::new(self.cluster, self.namespace);
        let logical_bytes = snapshot.memory.map_or(0, |memory| memory.logical_bytes);
        let p99 = snapshot.latency.map_or(0, |latency| latency.p99_ns);
        let repair_debt = snapshot
            .maintenance_debt
            .map_or(0, |debt| debt.repair.assets);
        let repair_debt_age = snapshot
            .maintenance_debt
            .map_or(Duration::ZERO, |debt| debt.repair.oldest_age);
        SafetyContext {
            scope,
            primary_target: tablet,
            target_logical_bytes: logical_bytes,
            target_combined_bytes: logical_bytes,
            current_replicas: 1,
            current_failure_domains: 1,
            current_fence: kivi_control::MigrationFence::INITIAL,
            migration_fencing_valid: true,
            representation_reconstructible: false,
            critical_repair_reserve: RepairBudget::new(250_000)
                .unwrap_or_else(|| RepairBudget::new(0).expect("zero repair budget is valid")),
            repair_budget: RepairBudget::new(500_000)
                .unwrap_or_else(|| RepairBudget::new(0).expect("zero repair budget is valid")),
            scrub_budget: ScrubBudget::new(500_000)
                .unwrap_or_else(|| ScrubBudget::new(0).expect("zero scrub budget is valid")),
            memory_budget: MemoryBudget::new(state.current_memory_budget_ppm)
                .unwrap_or_else(|| MemoryBudget::new(0).expect("zero memory budget is valid")),
            materialization: kivi_control::MaterializationIntent::Materialize,
            repair_debt_age,
            critical_repair_debt: repair_debt > 0,
            active_topology_tablets: BTreeSet::new(),
            tablet_count: 1,
            migration_candidates: Vec::new(),
            merge_candidates: Vec::new(),
            actions_in_window: 0,
            resource_cost_in_window: ActionCost::zero(),
            foreground_latency_bad: p99 >= 1_000_000,
            confidence: snapshot.evidence.confidence,
            freshness: snapshot.evidence.freshness.status,
            freshness_age: snapshot.evidence.freshness.age,
        }
    }
}

struct MemoryActuator {
    engine: AdminHandle,
    policy: kivi_memory::MemoryControlPolicy,
    budget_ppm: u32,
}

impl Actuator for MemoryActuator {
    fn execute(
        &mut self,
        action_id: kivi_control::ActionId,
        action: &Action,
    ) -> Result<ActuatorReceipt, ActuationError> {
        let mut next_policy = self.policy;
        let mut next_budget = self.budget_ppm;
        let changed = match action {
            Action::AdjustMemoryBudget { budget, .. } => {
                let value = f64::from(budget.as_ppm().as_u32()) / 1_000_000.0;
                next_budget = budget.as_ppm().as_u32();
                next_policy.dram_residency_target =
                    UnitInterval::new(value).unwrap_or(next_policy.dram_residency_target);
                true
            }
            Action::ChangeMaterializationIntent { intent, .. } => {
                match intent {
                    kivi_control::MaterializationIntent::Materialize => {
                        next_policy.promotion_threshold = UnitInterval::ZERO;
                        next_policy.nvme_demotion_pressure =
                            UnitInterval::new(0.9).unwrap_or(UnitInterval::ONE);
                    }
                    kivi_control::MaterializationIntent::KeepWarm => {
                        next_policy.promotion_threshold =
                            UnitInterval::new(0.25).unwrap_or(UnitInterval::ZERO);
                    }
                    kivi_control::MaterializationIntent::Demote
                    | kivi_control::MaterializationIntent::Evict => {
                        next_policy.nvme_demotion_pressure =
                            UnitInterval::new(0.5).unwrap_or(UnitInterval::ZERO);
                        next_policy.promotion_threshold =
                            UnitInterval::new(0.75).unwrap_or(UnitInterval::ONE);
                    }
                }
                true
            }
            _ => false,
        };
        if !changed {
            return Err(ActuationError::Rejected);
        }
        if !self.engine.set_memory_control_policy(next_policy) {
            let _ = self.engine.set_memory_control_policy(self.policy);
            return Err(ActuationError::Rejected);
        }
        self.policy = next_policy;
        self.budget_ppm = next_budget;
        Ok(ActuatorReceipt::new(
            action_id,
            ActionOutcome::success(
                action.expected_benefit(),
                Duration::ZERO,
                ActionCost::zero(),
            ),
        ))
    }
}

enum RuntimeSignal {
    Request(u64),
    Queueing(QueueingSignal),
    Compute(ComputeSignal),
    Memory(MemorySignal),
    MemoryPressure(MemoryPressureSignal),
    Io(IoSignal),
    Criticality(ExecutionCriticalitySignal),
    HotKey(HotKeySignal),
    Maintenance(MaintenanceDebtSignal),
    Redundancy(RedundancySignal),
    Degraded(DegradedAssetsSignal),
    Topology(TopologyImbalanceSignal),
    Consensus(ConsensusLagSignal),
}

impl RuntimeSignal {
    fn into_observation(self) -> kivi_observation::TypedSignal {
        match self {
            Self::Request(reads) => {
                kivi_observation::TypedSignal::Request(RequestSignal { reads, writes: 0 })
            }
            Self::Queueing(signal) => kivi_observation::TypedSignal::Queueing(signal),
            Self::Compute(signal) => kivi_observation::TypedSignal::Compute(signal),
            Self::Memory(signal) => kivi_observation::TypedSignal::Memory(signal),
            Self::MemoryPressure(signal) => kivi_observation::TypedSignal::MemoryPressure(signal),
            Self::Io(signal) => kivi_observation::TypedSignal::Io(signal),
            Self::Criticality(signal) => {
                kivi_observation::TypedSignal::ExecutionCriticality(signal)
            }
            Self::HotKey(signal) => kivi_observation::TypedSignal::HotKey(signal),
            Self::Maintenance(signal) => kivi_observation::TypedSignal::MaintenanceDebt(signal),
            Self::Redundancy(signal) => kivi_observation::TypedSignal::Redundancy(signal),
            Self::Degraded(signal) => kivi_observation::TypedSignal::DegradedAssets(signal),
            Self::Topology(signal) => kivi_observation::TypedSignal::TopologyImbalance(signal),
            Self::Consensus(signal) => kivi_observation::TypedSignal::ConsensusLag(signal),
        }
    }
}

struct TypedRequest(u64);
struct TypedQueueing(QueueingSignal);
struct TypedCompute(ComputeSignal);
struct TypedMemory(MemorySignal);
struct TypedMemoryPressure(MemoryPressureSignal);
struct TypedIo(IoSignal);
struct TypedCriticality(ExecutionCriticalitySignal);
struct TypedHotKey(HotKeySignal);
struct TypedMaintenance(MaintenanceDebtSignal);
struct TypedRedundancy(RedundancySignal);
struct TypedDegraded(DegradedAssetsSignal);
struct TypedTopology(TopologyImbalanceSignal);
struct TypedConsensus(ConsensusLagSignal);

impl From<TypedRequest> for RuntimeSignal {
    fn from(value: TypedRequest) -> Self {
        Self::Request(value.0)
    }
}
impl From<TypedQueueing> for RuntimeSignal {
    fn from(value: TypedQueueing) -> Self {
        Self::Queueing(value.0)
    }
}
impl From<TypedCompute> for RuntimeSignal {
    fn from(value: TypedCompute) -> Self {
        Self::Compute(value.0)
    }
}
impl From<TypedMemory> for RuntimeSignal {
    fn from(value: TypedMemory) -> Self {
        Self::Memory(value.0)
    }
}
impl From<TypedMemoryPressure> for RuntimeSignal {
    fn from(value: TypedMemoryPressure) -> Self {
        Self::MemoryPressure(value.0)
    }
}
impl From<TypedIo> for RuntimeSignal {
    fn from(value: TypedIo) -> Self {
        Self::Io(value.0)
    }
}
impl From<TypedCriticality> for RuntimeSignal {
    fn from(value: TypedCriticality) -> Self {
        Self::Criticality(value.0)
    }
}
impl From<TypedHotKey> for RuntimeSignal {
    fn from(value: TypedHotKey) -> Self {
        Self::HotKey(value.0)
    }
}
impl From<TypedMaintenance> for RuntimeSignal {
    fn from(value: TypedMaintenance) -> Self {
        Self::Maintenance(value.0)
    }
}
impl From<TypedRedundancy> for RuntimeSignal {
    fn from(value: TypedRedundancy) -> Self {
        Self::Redundancy(value.0)
    }
}
impl From<TypedDegraded> for RuntimeSignal {
    fn from(value: TypedDegraded) -> Self {
        Self::Degraded(value.0)
    }
}
impl From<TypedTopology> for RuntimeSignal {
    fn from(value: TypedTopology) -> Self {
        Self::Topology(value.0)
    }
}
impl From<TypedConsensus> for RuntimeSignal {
    fn from(value: TypedConsensus) -> Self {
        Self::Consensus(value.0)
    }
}

fn fabric_stats_pressure(
    fabrics: &[(WorkerId, kivi_engine::FabricWorkerReport)],
    worker: WorkerId,
) -> f64 {
    fabrics
        .iter()
        .find(|(id, _)| *id == worker)
        .map_or(0.0, |(_, report)| report.arena_pressure)
}

fn fabric_resident_bytes(stats: kivi_engine::FabricStatsSnapshot) -> u64 {
    stats
        .inline_bytes
        .saturating_add(stats.dram_bytes)
        .saturating_add(stats.compressed_bytes)
        .saturating_add(stats.nvme_bytes)
}

fn duration_ns(duration: Duration) -> u64 {
    u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX)
}

fn aggregate_criticality(
    signals: &[(u64, kivi_memory::AccessSignals)],
) -> ExecutionCriticalitySignal {
    signals.iter().fold(
        ExecutionCriticalitySignal {
            accesses: 0,
            serial_accesses: 0,
            critical_path_accesses: 0,
            tail_accesses: 0,
            stall_time_ns: 0,
            access_latency_ns: 1,
            logical_bytes: 1,
        },
        |mut total, (_, signal)| {
            let accesses = signal.frequency();
            let latency = signal.cpu_ns.checked_div(accesses).unwrap_or(1);
            total.accesses = total.accesses.saturating_add(accesses);
            total.serial_accesses = total.serial_accesses.saturating_add(accesses);
            total.critical_path_accesses =
                total.critical_path_accesses.saturating_add(accesses / 2);
            total.stall_time_ns = total.stall_time_ns.saturating_add(signal.cpu_ns);
            total.access_latency_ns = total.access_latency_ns.max(latency);
            total.logical_bytes = total.logical_bytes.saturating_add(signal.logical_bytes);
            total
        },
    )
}

fn hot_key_signal(object: u64, access: &kivi_memory::AccessSignals) -> HotKeySignal {
    HotKeySignal {
        key: HotKeyId::from_u128(u128::from(object)),
        on_critical_path: access.cpu_ns > 0,
        affects_tail: access.cpu_ns > 0,
        stall_time_ns: access.cpu_ns,
    }
}
