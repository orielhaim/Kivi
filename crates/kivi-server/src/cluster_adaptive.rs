//! Cluster-side Phase 12 adaptive control runtime.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use kivi_consensus::{
    CATCH_UP_LAG_THRESHOLD, ConsensusGroupId, ConsensusNode, NodeStatus, ObservedMembership,
    ReplicaId, ReplicaRole,
};
use kivi_control::adaptive::ActionLedgerConfig;
use kivi_control::{
    Action, ActionCost, ActionOutcome, ActionRecord, ActionResult, ActionScope, ActionTarget,
    ActuatorReceipt, AdaptiveConfig, AdaptiveController, AdaptiveDecision, AdaptiveMode,
    BaselineConfig, BaselineController, ControlState, ControllerFrame, ControllerSource,
    LearnedController, MaterializationIntent, MemoryBudget, MigrationFence, MigrationIntent,
    MigrationRecommendation, NodeState, RepairBudget, SafetyContext, ScrubBudget, TabletSet,
};
use kivi_observation::{
    CheckpointDebt, Confidence, ConsensusLagSignal, FreshnessStatus, MaintenanceDebtSignal,
    MemorySignal, ObservationFabric, ObservationLimits, ObservationMetadata, ObservationPeriod,
    ObservationSample, ObservationScope, ObservationSequence, ObservationSnapshot,
    ObservationSource, ObservationStamp, ObservationWindowId, RepairDebt, ScrubDebt,
    TopologyImbalanceSignal, TypedSignal,
};
use kivi_state::StoreStats;
use kivi_tablet::{DirectorySnapshot, PartitionRange, TabletState};
use kivi_types::{ClusterId, NamespaceId, NodeId, TabletId, Ticks, WallTimestamp, WorkerId};
use serde::Serialize;

/// Configuration for the cluster adaptive runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ClusterAdaptiveConfig {
    /// Interval between observation and decision passes.
    pub interval: Duration,
    /// Whether baseline actions are shadow-only.
    pub shadow: bool,
    /// Maximum retained observation scopes.
    pub max_scopes: usize,
    /// Maximum retained samples per scope in one pass.
    pub max_samples_per_scope: usize,
    /// Maximum retained latency samples per scope.
    pub max_latency_samples: usize,
    /// Maximum retained hot keys per scope.
    pub hot_key_capacity: usize,
    /// Maximum retained decision traces.
    pub max_traces: usize,
    /// Maximum retained recommendation records.
    pub max_recommendations: usize,
    /// Maximum accepted observation age.
    pub freshness_budget: Duration,
}

impl Default for ClusterAdaptiveConfig {
    fn default() -> Self {
        Self {
            interval: Duration::from_millis(500),
            shadow: true,
            max_scopes: 128,
            max_samples_per_scope: 16,
            max_latency_samples: 8,
            hot_key_capacity: 32,
            max_traces: 64,
            max_recommendations: 64,
            freshness_budget: Duration::from_secs(5),
        }
    }
}

impl ClusterAdaptiveConfig {
    /// Reads optional cluster adaptive overrides from the environment.
    #[must_use]
    pub fn from_env() -> Self {
        let mut config = Self::default();
        let interval = std::env::var("KIVI_CLUSTER_ADAPTIVE_INTERVAL_MS")
            .ok()
            .or_else(|| std::env::var("KIVI_ADAPTIVE_INTERVAL_MS").ok())
            .and_then(|value| value.parse::<u64>().ok())
            .filter(|value| *value >= 50)
            .map(Duration::from_millis);
        if let Some(interval) = interval {
            config.interval = interval;
        }
        let shadow = std::env::var("KIVI_CLUSTER_ADAPTIVE_SHADOW")
            .ok()
            .or_else(|| std::env::var("KIVI_ADAPTIVE_SHADOW").ok());
        let active = std::env::var("KIVI_CLUSTER_ADAPTIVE_ACTIVE")
            .ok()
            .or_else(|| std::env::var("KIVI_ADAPTIVE_ACTIVE").ok());
        if shadow.is_some_and(|value| value == "1") {
            config.shadow = true;
        }
        if active.is_some_and(|value| value == "1") {
            config.shadow = false;
        }
        config
    }

    fn normalized(self) -> Self {
        let max_scopes = self.max_scopes.max(1);
        let max_samples_per_scope = self.max_samples_per_scope.max(1);
        Self {
            interval: if self.interval.is_zero() {
                Duration::from_millis(1)
            } else {
                self.interval
            },
            shadow: self.shadow,
            max_scopes,
            max_samples_per_scope,
            max_latency_samples: self.max_latency_samples.max(1).min(max_samples_per_scope),
            hot_key_capacity: self.hot_key_capacity.max(1),
            max_traces: self.max_traces.max(1),
            max_recommendations: self.max_recommendations.max(1),
            freshness_budget: if self.freshness_budget.is_zero() {
                Duration::from_secs(5)
            } else {
                self.freshness_budget
            },
        }
    }
}

/// One bounded adaptive decision trace.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ClusterAdaptiveTrace {
    /// Observation sequence used by the decision.
    pub sequence: u64,
    /// Observation window used by the decision.
    pub window: u64,
    /// Controller source.
    pub source: String,
    /// Trace status.
    pub status: String,
    /// Typed decision reason.
    pub reason: String,
    /// Action kind, when one was generated.
    pub action_kind: Option<String>,
    /// Safety rejection, when present.
    pub safety_violation: Option<String>,
    /// Ledger suppression, when present.
    pub ledger_rejection: Option<String>,
    /// Whether baseline fallback was used.
    pub fallback: bool,
}

/// One bounded controller recommendation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ClusterAdaptiveRecommendation {
    /// Ledger action identity, when the recommendation was admitted.
    pub action_id: Option<u64>,
    /// Stable action name.
    pub action: String,
    /// Canonical target tablet ids.
    pub targets: Vec<u64>,
    /// Migration source, when applicable.
    pub from: Option<u64>,
    /// Migration destination, when applicable.
    pub to: Option<u64>,
    /// Placement fence, when applicable.
    pub fence: Option<u64>,
    /// Maximum migration bytes, when applicable.
    pub max_bytes: Option<u64>,
    /// Policy summary, when applicable.
    pub policy: Option<String>,
    /// Controller source.
    pub source: String,
    /// Whether the recommendation was actuated.
    pub actuated: bool,
    /// Actuation or validation error, when applicable.
    pub error: Option<String>,
}

/// Public status and bounded history for the cluster adaptive runtime.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[allow(clippy::struct_excessive_bools)]
pub struct ClusterAdaptiveSnapshot {
    /// Whether the runtime is mounted.
    pub enabled: bool,
    /// Active operating mode.
    pub mode: String,
    /// Active controller source.
    pub active_controller: String,
    /// Whether learned decisions are shadow-only.
    pub learned_shadow: bool,
    /// Control interval in milliseconds.
    pub interval_ms: u64,
    /// Completed observation intervals.
    pub intervals: u64,
    /// Latest accepted observation sequence.
    pub observation_sequence: Option<u64>,
    /// Latest observation window.
    pub observation_window: u64,
    /// Number of retained observation scopes.
    pub observation_scopes: usize,
    /// Freshness classification.
    pub freshness: String,
    /// Freshness age in milliseconds.
    pub freshness_age_ms: u64,
    /// Confidence classification.
    pub confidence: String,
    /// Whether the local control replica is leader.
    pub control_leader: bool,
    /// Whether the control quorum gate passed.
    pub control_quorum: bool,
    /// Current placement generation observed by the runtime.
    pub placement_generation: Option<u64>,
    /// Number of controller decisions.
    pub decisions: u64,
    /// Baseline proposed actions.
    pub proposed: u64,
    /// Baseline admitted actions.
    pub accepted: u64,
    /// Baseline safety-rejected actions.
    pub rejected: u64,
    /// Baseline ledger-suppressed actions.
    pub suppressed: u64,
    /// Actions handed to the bounded actuator.
    pub executed: u64,
    /// Successfully completed actions.
    pub completed: u64,
    /// Failed actions.
    pub failed: u64,
    /// Learned or shadow actions emitted.
    pub shadow_actions: u64,
    /// Migration plans created through the control pathway.
    pub actuated: u64,
    /// Migration actuations rejected or failed.
    pub actuation_failures: u64,
    /// Active records in the controller ledger.
    pub active_actions: usize,
    /// Bounded decision traces.
    pub traces: Vec<ClusterAdaptiveTrace>,
    /// Bounded recommendation history.
    pub recommendations: Vec<ClusterAdaptiveRecommendation>,
    /// Last bounded collection, gate, or actuation error.
    pub last_error: Option<String>,
}

#[derive(Debug)]
struct RuntimeState {
    observations: ObservationFabric,
    controller: AdaptiveController,
    sequence: ObservationSequence,
    window: ObservationWindowId,
    ticks: Ticks,
    intervals: u64,
    decisions: u64,
    last_snapshot: Option<ObservationSnapshot>,
    freshness: FreshnessStatus,
    freshness_age: Duration,
    confidence: Confidence,
    control_leader: bool,
    control_quorum: bool,
    placement_generation: Option<u64>,
    traces: VecDeque<ClusterAdaptiveTrace>,
    recommendations: VecDeque<ClusterAdaptiveRecommendation>,
    shadow_actions: u64,
    actuated: u64,
    actuation_failures: u64,
    last_error: Option<String>,
}

#[derive(Debug)]
struct TickInput {
    statuses: Vec<NodeStatus>,
    control_status: NodeStatus,
    control_state: Option<ControlState>,
    control_membership: Option<ObservedMembership>,
    stats: BTreeMap<u64, StoreStats>,
    maintenance: Option<MaintenanceDebtSignal>,
    merge_pairs: Vec<(TabletId, TabletId)>,
}

#[derive(Debug, Clone, Copy)]
struct ControlGate {
    generation: u64,
}

#[derive(Debug, Clone, Copy)]
struct ExplicitSignals {
    topology: Option<TopologyImbalanceSignal>,
    consensus: Option<ConsensusLagSignal>,
    memory: Option<MemorySignal>,
    maintenance: Option<MaintenanceDebtSignal>,
}

struct SampleBuilder {
    samples: Vec<ObservationSample>,
    sequence: ObservationSequence,
    scopes: BTreeSet<ObservationScope>,
    per_scope: BTreeMap<ObservationScope, usize>,
    max_scopes: usize,
    max_samples_per_scope: usize,
    window: ObservationWindowId,
    observed_at: Ticks,
    period: ObservationPeriod,
}

impl SampleBuilder {
    fn new(
        sequence: ObservationSequence,
        window: ObservationWindowId,
        observed_at: Ticks,
        period: ObservationPeriod,
        max_scopes: usize,
        max_samples_per_scope: usize,
    ) -> Self {
        Self {
            samples: Vec::new(),
            sequence,
            scopes: BTreeSet::new(),
            per_scope: BTreeMap::new(),
            max_scopes,
            max_samples_per_scope,
            window,
            observed_at,
            period,
        }
    }

    fn add_scope(&mut self, scope: ObservationScope, signals: ExplicitSignals) {
        if !self.scopes.contains(&scope) {
            if self.scopes.len() >= self.max_scopes {
                return;
            }
            self.scopes.insert(scope);
            self.per_scope.insert(scope, 0);
        }
        if let Some(signal) = signals.topology {
            self.add(
                scope,
                TypedSignal::TopologyImbalance(signal),
                ObservationSource::ControlPlane,
                Confidence::High,
            );
        }
        if let Some(signal) = signals.consensus {
            self.add(
                scope,
                TypedSignal::ConsensusLag(signal),
                ObservationSource::RuntimeCounter,
                Confidence::Low,
            );
        }
        if let Some(signal) = signals.memory {
            self.add(
                scope,
                TypedSignal::Memory(signal),
                ObservationSource::RuntimeCounter,
                Confidence::Low,
            );
        }
        if let Some(signal) = signals.maintenance {
            self.add(
                scope,
                TypedSignal::MaintenanceDebt(signal),
                ObservationSource::ControlPlane,
                Confidence::Low,
            );
        }
    }

    fn add(
        &mut self,
        scope: ObservationScope,
        signal: TypedSignal,
        source: ObservationSource,
        confidence: Confidence,
    ) {
        let Some(count) = self.per_scope.get_mut(&scope) else {
            return;
        };
        if *count >= self.max_samples_per_scope {
            return;
        }
        *count += 1;
        let sample = ObservationSample::new(
            scope,
            ObservationStamp {
                window: self.window,
                sequence: self.sequence,
            },
            self.period,
            ObservationMetadata::new(self.observed_at, source, confidence),
            signal,
        );
        self.samples.push(sample);
        if let Ok(next) = self.sequence.next() {
            self.sequence = next;
        }
    }
}

/// Bounded cluster adaptive controller runtime.
#[derive(Debug)]
pub struct ClusterAdaptiveRuntime {
    node: Arc<ConsensusNode>,
    directory: Arc<arc_swap::ArcSwap<DirectorySnapshot>>,
    redundancy: Option<Arc<kivi_consensus::distcoord::RedundancyCoordinator>>,
    cluster: ClusterId,
    namespace: NamespaceId,
    config: ClusterAdaptiveConfig,
    state: Arc<Mutex<RuntimeState>>,
}

impl ClusterAdaptiveRuntime {
    /// Creates a cluster adaptive runtime over consensus and directory views.
    #[must_use]
    pub fn new(
        node: Arc<ConsensusNode>,
        directory: Arc<arc_swap::ArcSwap<DirectorySnapshot>>,
        redundancy: Option<Arc<kivi_consensus::distcoord::RedundancyCoordinator>>,
        cluster: ClusterId,
        namespace: NamespaceId,
        config: ClusterAdaptiveConfig,
    ) -> Self {
        let config = config.normalized();
        let limits = ObservationLimits::new(
            config.max_scopes,
            config.max_samples_per_scope,
            config.max_latency_samples,
            config.hot_key_capacity,
            config.freshness_budget,
        )
        .unwrap_or_else(|| {
            ObservationLimits::new(128, 16, 8, 32, Duration::from_secs(5))
                .expect("fallback observation limits are valid")
        });
        let mode = if config.shadow {
            AdaptiveMode::Shadow
        } else {
            AdaptiveMode::Active
        };
        let baseline = BaselineController::new(BaselineConfig {
            sustained_windows: 1,
            min_samples: 1,
            ..BaselineConfig::default()
        });
        let controller = AdaptiveController::new(
            AdaptiveConfig {
                mode,
                ledger: ActionLedgerConfig {
                    max_history: config.max_recommendations,
                    max_traces: config.max_traces,
                    ..ActionLedgerConfig::default()
                },
                max_traces: config.max_traces,
            },
            baseline,
            Some(LearnedController::default()),
        );
        Self {
            node,
            directory,
            redundancy,
            cluster,
            namespace,
            config,
            state: Arc::new(Mutex::new(RuntimeState {
                observations: ObservationFabric::new(limits),
                controller,
                sequence: ObservationSequence::FIRST,
                window: ObservationWindowId::FIRST,
                ticks: Ticks::from_micros(1),
                intervals: 0,
                decisions: 0,
                last_snapshot: None,
                freshness: FreshnessStatus::Future,
                freshness_age: Duration::ZERO,
                confidence: Confidence::Low,
                control_leader: false,
                control_quorum: false,
                placement_generation: None,
                traces: VecDeque::new(),
                recommendations: VecDeque::new(),
                shadow_actions: 0,
                actuated: 0,
                actuation_failures: 0,
                last_error: None,
            })),
        }
    }

    /// Returns the configured interval.
    #[must_use]
    pub const fn interval(&self) -> Duration {
        self.config.interval
    }

    /// Returns whether baseline actuation is disabled.
    #[must_use]
    pub const fn is_shadow(&self) -> bool {
        self.config.shadow
    }

    /// Returns the public runtime status.
    #[must_use]
    pub fn snapshot(&self) -> ClusterAdaptiveSnapshot {
        let state = self.state.lock().expect("cluster adaptive state mutex");
        let baseline = state
            .controller
            .stats()
            .get(&ControllerSource::Baseline)
            .copied()
            .unwrap_or_default();
        let target = state.last_snapshot.as_ref().and_then(|snapshot| {
            snapshot
                .scopes
                .iter()
                .find(|scope| matches!(scope.scope, ObservationScope::Tablet(_)))
                .or_else(|| snapshot.scopes.first())
        });
        let (freshness, freshness_age, confidence) = target.map_or_else(
            || ("unknown".to_owned(), Duration::ZERO, "unknown".to_owned()),
            |scope| {
                (
                    freshness_name(scope.evidence.freshness.status).to_owned(),
                    scope.evidence.freshness.age,
                    confidence_name(scope.evidence.confidence).to_owned(),
                )
            },
        );
        let observation_window =
            target.map_or(state.window.as_u64(), |scope| scope.latest_window.as_u64());
        ClusterAdaptiveSnapshot {
            enabled: true,
            mode: if is_shadow_mode(
                self.is_shadow(),
                state.control_leader && state.control_quorum,
            ) {
                "shadow".to_owned()
            } else {
                "active".to_owned()
            },
            active_controller: "baseline".to_owned(),
            learned_shadow: true,
            interval_ms: u64::try_from(self.interval().as_millis()).unwrap_or(u64::MAX),
            intervals: state.intervals,
            observation_sequence: state
                .last_snapshot
                .as_ref()
                .and_then(|snapshot| snapshot.last_sequence)
                .map(kivi_observation::ObservationSequence::as_u64),
            observation_window,
            observation_scopes: state
                .last_snapshot
                .as_ref()
                .map_or(0, |snapshot| snapshot.scopes.len()),
            freshness,
            freshness_age_ms: u64::try_from(freshness_age.as_millis()).unwrap_or(u64::MAX),
            confidence,
            control_leader: state.control_leader,
            control_quorum: state.control_quorum,
            placement_generation: state.placement_generation,
            decisions: state.decisions,
            proposed: baseline.proposed,
            accepted: baseline.accepted,
            rejected: baseline.rejected,
            suppressed: baseline.suppressed,
            executed: baseline.executed,
            completed: baseline.completed,
            failed: baseline.failed,
            shadow_actions: state.shadow_actions,
            actuated: state.actuated,
            actuation_failures: state.actuation_failures,
            active_actions: state.controller.ledger().active_records().count(),
            traces: state.traces.iter().cloned().collect(),
            recommendations: state.recommendations.iter().cloned().collect(),
            last_error: state.last_error.clone(),
        }
    }

    /// Returns the public runtime status.
    #[must_use]
    pub fn status(&self) -> ClusterAdaptiveSnapshot {
        self.snapshot()
    }

    /// Runs the bounded adaptive loop until shutdown.
    #[must_use]
    pub fn spawn(
        self: Arc<Self>,
        mut shutdown: tokio::sync::watch::Receiver<bool>,
    ) -> tokio::task::JoinHandle<()> {
        tokio::spawn(async move {
            let mut ticker = tokio::time::interval(self.config.interval);
            ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
            loop {
                tokio::select! {
                    _ = ticker.tick() => Arc::clone(&self).tick().await,
                    changed = shutdown.changed() => {
                        if changed.is_err() || *shutdown.borrow() {
                            break;
                        }
                    }
                }
            }
        })
    }

    /// Runs one observation, decision, and bounded actuation pass.
    #[allow(clippy::too_many_lines)]
    pub async fn tick(self: Arc<Self>) {
        let input = self.collect_input().await;
        let period = ObservationPeriod::new(self.config.interval)
            .unwrap_or_else(|| ObservationPeriod::new(Duration::from_millis(1)).expect("period"));
        let (snapshot, collect_error) = {
            let mut state = self.state.lock().expect("cluster adaptive state mutex");
            state.observations.begin_window();
            let mut builder = SampleBuilder::new(
                state.sequence,
                state.window,
                state.ticks,
                period,
                self.config.max_scopes,
                self.config.max_samples_per_scope,
            );
            self.build_samples(&input, &mut builder);
            let mut collect_error = None;
            for sample in &builder.samples {
                if let Err(error) = state.observations.collect(sample) {
                    collect_error = Some(error.to_string());
                    break;
                }
            }
            state.sequence = builder.sequence;
            let snapshot = state.observations.snapshot(state.ticks);
            let (freshness, freshness_age, confidence) = evidence_summary(&snapshot, None);
            state.freshness = freshness;
            state.freshness_age = freshness_age;
            state.confidence = confidence;
            state.last_snapshot = Some(snapshot.clone());
            state.intervals = state.intervals.saturating_add(1);
            state.ticks = state.ticks.advance_by(self.config.interval);
            if let Ok(next) = state.window.next() {
                state.window = next;
            }
            (snapshot, collect_error)
        };
        if let Some(error) = collect_error {
            self.set_error(format!("observation collection: {error}"));
            return;
        }
        let gate = control_gate(
            self.node.node(),
            &input.control_status,
            input.control_membership.as_ref(),
            input.control_state.as_ref(),
            self.namespace,
        );
        let actuation_allowed = gate.is_ok();
        let gate_error = gate.err();
        let decision_generation = input
            .control_state
            .as_ref()
            .map_or(0, |state| state.placement_version().as_u64());
        {
            let mut state = self.state.lock().expect("cluster adaptive state mutex");
            state.control_leader = input.control_status.role == ReplicaRole::Leader
                && input.control_status.leader == Some(ReplicaId::of_node(self.node.node()));
            state.control_quorum = actuation_allowed;
            state.placement_generation = input
                .control_state
                .as_ref()
                .map(|state| state.placement_version().as_u64());
        }
        let Some(primary) = primary_tablet(&input, &snapshot) else {
            self.set_error("adaptive paused: no observed local tablet");
            return;
        };
        let Some(target) = snapshot
            .scopes
            .iter()
            .find(|scope| scope.scope == ObservationScope::Tablet(primary))
        else {
            self.set_error("adaptive paused: primary tablet scope is missing");
            return;
        };
        {
            let mut state = self.state.lock().expect("cluster adaptive state mutex");
            state.freshness = target.evidence.freshness.status;
            state.freshness_age = target.evidence.freshness.age;
            state.confidence = target.evidence.confidence;
        }
        let context = self.safety_context(&input, &snapshot, primary, decision_generation);
        let sequence = snapshot.last_sequence.unwrap_or(ObservationSequence::FIRST);
        let target_window = target.latest_window;
        let target_at = target.evidence.latest_observed_at;
        let frame = ControllerFrame::new(
            snapshot,
            ActionScope::new(self.cluster, self.namespace),
            sequence,
            target_window,
            1,
            target_at,
        );
        let controller_mode = decision_mode(self.config.shadow, actuation_allowed);
        let decision = {
            let mut state = self.state.lock().expect("cluster adaptive state mutex");
            state.controller.set_mode(controller_mode);
            let decision = state.controller.decide(&frame, &context);
            state.controller.set_mode(if self.config.shadow {
                AdaptiveMode::Shadow
            } else {
                AdaptiveMode::Active
            });
            state.decisions = state.decisions.saturating_add(1);
            state.shadow_actions = state
                .shadow_actions
                .saturating_add(decision.shadow_actions.len() as u64);
            self.record_decision(&mut state, &decision);
            state.last_error = gate_error
                .as_ref()
                .map(|error| format!("adaptive shadow: {error}"));
            decision
        };
        let observation = frame.action_observation();
        for record in decision.active_actions {
            if !is_actuatable_action(&record.action) {
                let receipt = ActuatorReceipt::new(
                    record.id,
                    ActionOutcome::failed(
                        ActionResult::Rejected,
                        Duration::ZERO,
                        ActionCost::zero(),
                    ),
                );
                let mut state = self.state.lock().expect("cluster adaptive state mutex");
                if let Err(error) = state.controller.apply_receipt(receipt, observation) {
                    state.last_error = Some(error.to_string());
                }
                mark_recommendation(
                    &mut state.recommendations,
                    record.id,
                    false,
                    Some("action is recommendation-only".to_owned()),
                );
                continue;
            }
            if !actuation_allowed || self.config.shadow {
                let receipt = ActuatorReceipt::new(
                    record.id,
                    ActionOutcome::failed(
                        ActionResult::Rejected,
                        Duration::ZERO,
                        ActionCost::zero(),
                    ),
                );
                let mut state = self.state.lock().expect("cluster adaptive state mutex");
                if let Err(error) = state.controller.apply_receipt(receipt, observation) {
                    state.last_error = Some(error.to_string());
                }
                mark_recommendation(
                    &mut state.recommendations,
                    record.id,
                    false,
                    Some("adaptive actuation gate is closed".to_owned()),
                );
                continue;
            }
            let result = self.actuate_migration(&record).await;
            let mut state = self.state.lock().expect("cluster adaptive state mutex");
            match result {
                Ok(()) => {
                    let receipt = ActuatorReceipt::new(
                        record.id,
                        ActionOutcome::success(
                            record.action.expected_benefit(),
                            Duration::ZERO,
                            record.action.estimated_cost(),
                        ),
                    );
                    if let Err(error) = state.controller.apply_receipt(receipt, observation) {
                        state.last_error = Some(error.to_string());
                    }
                    state.actuated = state.actuated.saturating_add(1);
                    mark_recommendation(&mut state.recommendations, record.id, true, None);
                }
                Err(reason) => {
                    let receipt = ActuatorReceipt::new(
                        record.id,
                        ActionOutcome::failed(
                            ActionResult::Failed,
                            Duration::ZERO,
                            ActionCost::zero(),
                        ),
                    );
                    if let Err(error) = state.controller.apply_receipt(receipt, observation) {
                        state.last_error = Some(error.to_string());
                    }
                    state.actuation_failures = state.actuation_failures.saturating_add(1);
                    state.last_error = Some(bounded_error(&reason));
                    let error = state.last_error.clone();
                    mark_recommendation(&mut state.recommendations, record.id, false, error);
                }
            }
        }
    }

    async fn collect_input(&self) -> TickInput {
        let mut statuses = self.node.status_all().await;
        statuses.sort_by_key(|status| status.group.tablet().as_u64());
        let status_limit = self.config.max_scopes.saturating_mul(2).max(16);
        statuses.truncate(status_limit);
        let control_status = self.node.status_for(ConsensusGroupId::control()).await;
        let control_state = self.node.control_state().await;
        let control_membership = if self.node.hosts_control() {
            self.node
                .observe_membership(ConsensusGroupId::control().tablet())
                .await
                .ok()
        } else {
            None
        };
        let mut stats = BTreeMap::new();
        for status in &statuses {
            if let Some(value) = self
                .node
                .tablet_stats(status.group.tablet(), WallTimestamp::from_micros(0))
                .await
            {
                stats.insert(status.group.tablet().as_u64(), value);
            }
        }
        let maintenance = self
            .redundancy
            .as_ref()
            .and_then(|coordinator| maintenance_signal(&coordinator.maintenance_snapshot()));
        let merge_pairs = self.merge_pairs();
        TickInput {
            statuses,
            control_status,
            control_state,
            control_membership,
            stats,
            maintenance,
            merge_pairs,
        }
    }

    fn merge_pairs(&self) -> Vec<(TabletId, TabletId)> {
        let directory = self.directory.load();
        let active: Vec<(TabletId, PartitionRange)> = directory
            .tablets()
            .iter()
            .filter(|descriptor| descriptor.state() == TabletState::Active)
            .take(self.config.max_scopes.saturating_mul(2).max(16))
            .map(|descriptor| (descriptor.id(), descriptor.range().clone()))
            .collect();
        let mut pairs = Vec::new();
        for pair in active.windows(2) {
            let (Some((left_id, left)), Some((right_id, right))) = (pair.first(), pair.get(1))
            else {
                continue;
            };
            if let (PartitionRange::Ordered(left_range), PartitionRange::Ordered(right_range)) =
                (left, right)
                && left_range.is_adjacent_to(right_range)
            {
                pairs.push((*left_id, *right_id));
                if pairs.len() >= self.config.max_scopes {
                    break;
                }
            }
        }
        pairs
    }

    fn build_samples(&self, input: &TickInput, builder: &mut SampleBuilder) {
        let worker_limit = if self.config.max_scopes > 3 {
            self.node
                .worker_count()
                .min(self.config.max_scopes.saturating_sub(3))
        } else {
            0
        };
        let mut worker_loads = vec![0_u64; worker_limit];
        for status in &input.statuses {
            let worker =
                kivi_consensus::worker_for_tablet(status.group.tablet(), self.node.worker_count());
            if worker < worker_limit {
                worker_loads[worker] = worker_loads[worker].saturating_add(1);
            }
        }
        let (placement_worker_loads, placement_tablet_loads) =
            placement_loads(input.control_state.as_ref(), self.config.max_scopes);
        let (global_worker_loads, global_tablet_loads) = if placement_worker_loads.is_empty() {
            (worker_loads.clone(), Vec::new())
        } else {
            (placement_worker_loads, placement_tablet_loads)
        };
        let global_topology = topology_signal(&global_worker_loads, &global_tablet_loads);
        let logical_bytes = input.stats.values().fold(0_u64, |total, stats| {
            total.saturating_add(stats.logical_value_bytes)
        });
        let aggregate = ExplicitSignals {
            topology: global_topology,
            consensus: (!input.statuses.is_empty()).then_some(ConsensusLagSignal {
                commit_lag_ns: 0,
                replica_lag_ns: 0,
            }),
            memory: Some(MemorySignal {
                resident_bytes: 0,
                logical_bytes,
            }),
            maintenance: input.maintenance,
        };
        builder.add_scope(ObservationScope::Namespace(self.namespace), aggregate);
        builder.add_scope(ObservationScope::Cluster(self.cluster), aggregate);
        for worker in 0..worker_limit {
            let worker_id = WorkerId::from_u64(u64::try_from(worker).unwrap_or(u64::MAX));
            let has_status = input.statuses.iter().any(|status| {
                kivi_consensus::worker_for_tablet(status.group.tablet(), self.node.worker_count())
                    == worker
            });
            builder.add_scope(
                ObservationScope::Worker(worker_id),
                ExplicitSignals {
                    topology: global_topology,
                    consensus: has_status.then_some(ConsensusLagSignal {
                        commit_lag_ns: 0,
                        replica_lag_ns: 0,
                    }),
                    memory: None,
                    maintenance: None,
                },
            );
        }
        for status in &input.statuses {
            let tablet = status.group.tablet();
            let desired_replicas = input
                .control_state
                .as_ref()
                .and_then(|state| state.desired(tablet))
                .map(|desired| u64::try_from(desired.replicas.len()).unwrap_or(u64::MAX));
            let topology = match desired_replicas {
                Some(tablet_load) => topology_signal(&global_worker_loads, &[tablet_load]),
                None if !global_worker_loads.is_empty() => {
                    topology_signal(&global_worker_loads, &[])
                }
                None => None,
            };
            builder.add_scope(
                ObservationScope::Tablet(tablet),
                ExplicitSignals {
                    topology,
                    consensus: None,
                    memory: None,
                    maintenance: None,
                },
            );
        }
    }

    fn safety_context(
        &self,
        input: &TickInput,
        snapshot: &ObservationSnapshot,
        primary: TabletId,
        generation: u64,
    ) -> SafetyContext {
        let state = input.control_state.as_ref();
        let scope = ActionScope::new(self.cluster, self.namespace);
        let target = snapshot
            .scopes
            .iter()
            .find(|scope| scope.scope == ObservationScope::Tablet(primary));
        let desired = state.and_then(|state| state.desired(primary));
        let logical_bytes = input
            .stats
            .get(&primary.as_u64())
            .map_or(0, |stats| stats.logical_value_bytes);
        let merge_candidates = state
            .map(|state| self.merge_candidates(input, state))
            .unwrap_or_default();
        let target_combined_bytes = merge_candidates
            .iter()
            .find(|(tablets, _)| tablets.contains(primary))
            .map_or(logical_bytes, |(_, bytes)| *bytes);
        let mut active_topology_tablets = BTreeSet::new();
        if let Some(state) = state {
            for desired in state.placements().take(self.config.max_scopes) {
                if state.has_live_topology(desired.tablet) {
                    active_topology_tablets.insert(desired.tablet);
                }
            }
        }
        let current_replicas = desired.map_or(0, |desired| desired.replicas.len());
        let current_failure_domains = desired.map_or(0, |desired| {
            desired
                .replicas
                .iter()
                .filter_map(|node| state.and_then(|state| state.node(*node)))
                .filter(|record| !record.failure_domain.is_empty())
                .map(|record| record.failure_domain.as_str())
                .collect::<BTreeSet<_>>()
                .len()
        });
        let migration_candidates = if generation == 0 {
            Vec::new()
        } else {
            state
                .map(|state| Self::migration_candidates(state, primary, logical_bytes, generation))
                .unwrap_or_default()
        };
        let (confidence, freshness, freshness_age) = target.map_or_else(
            || (Confidence::Low, FreshnessStatus::Future, Duration::ZERO),
            |target| {
                (
                    target.evidence.confidence,
                    target.evidence.freshness.status,
                    target.evidence.freshness.age,
                )
            },
        );
        let foreground_latency_bad = target
            .and_then(|target| target.latency)
            .is_some_and(|latency| latency.p99_ns >= 1_000_000);
        SafetyContext {
            scope,
            primary_target: primary,
            target_logical_bytes: logical_bytes,
            target_combined_bytes,
            current_replicas: u8::try_from(current_replicas).unwrap_or(u8::MAX),
            current_failure_domains: u8::try_from(current_failure_domains).unwrap_or(u8::MAX),
            current_fence: MigrationFence::new(generation).unwrap_or(MigrationFence::INITIAL),
            migration_fencing_valid: generation != 0,
            representation_reconstructible: false,
            critical_repair_reserve: RepairBudget::new(250_000)
                .unwrap_or_else(|| RepairBudget::new(0).expect("zero repair budget")),
            repair_budget: RepairBudget::new(500_000)
                .unwrap_or_else(|| RepairBudget::new(0).expect("zero repair budget")),
            scrub_budget: ScrubBudget::new(500_000)
                .unwrap_or_else(|| ScrubBudget::new(0).expect("zero scrub budget")),
            memory_budget: MemoryBudget::new(500_000)
                .unwrap_or_else(|| MemoryBudget::new(0).expect("zero memory budget")),
            materialization: MaterializationIntent::Materialize,
            repair_debt_age: Duration::ZERO,
            critical_repair_debt: false,
            active_topology_tablets,
            tablet_count: state.map_or(0, |state| {
                u32::try_from(state.placements().count()).unwrap_or(u32::MAX)
            }),
            migration_candidates,
            merge_candidates,
            actions_in_window: 0,
            resource_cost_in_window: ActionCost::zero(),
            foreground_latency_bad,
            confidence,
            freshness,
            freshness_age,
        }
    }

    fn migration_candidates(
        state: &ControlState,
        tablet: TabletId,
        logical_bytes: u64,
        generation: u64,
    ) -> Vec<MigrationRecommendation> {
        if logical_bytes == 0 {
            return Vec::new();
        }
        let Some(desired) = state.desired(tablet) else {
            return Vec::new();
        };
        let Some(source) = desired.replicas.iter().copied().find(|node| {
            state
                .node(*node)
                .is_some_and(|record| record.state == NodeState::Active)
        }) else {
            return Vec::new();
        };
        let Some(destination) = state
            .active_nodes()
            .into_iter()
            .find(|node| !desired.contains(*node))
        else {
            return Vec::new();
        };
        let Some(fence) = MigrationFence::new(generation) else {
            return Vec::new();
        };
        MigrationRecommendation::new(source, destination, fence, logical_bytes)
            .into_iter()
            .collect()
    }

    fn merge_candidates(&self, input: &TickInput, state: &ControlState) -> Vec<(TabletSet, u64)> {
        input
            .merge_pairs
            .iter()
            .filter_map(|(left, right)| {
                if state.desired(*left).is_none()
                    || state.desired(*right).is_none()
                    || state.has_live_topology(*left)
                    || state.has_live_topology(*right)
                {
                    return None;
                }
                let bytes = input
                    .stats
                    .get(&left.as_u64())
                    .map_or(0, |stats| stats.logical_value_bytes)
                    .saturating_add(
                        input
                            .stats
                            .get(&right.as_u64())
                            .map_or(0, |stats| stats.logical_value_bytes),
                    );
                if bytes == 0 {
                    return None;
                }
                let tablets = TabletSet::new([*left, *right]).ok()?;
                Some((tablets, bytes))
            })
            .take(self.config.max_scopes)
            .collect()
    }

    fn observations_current(&self, target: TabletId) -> bool {
        let state = self.state.lock().expect("cluster adaptive state mutex");
        state.last_snapshot.as_ref().is_some_and(|snapshot| {
            snapshot
                .scopes
                .iter()
                .find(|scope| scope.scope == ObservationScope::Tablet(target))
                .is_some_and(|scope| {
                    scope.evidence.freshness.status == FreshnessStatus::Current
                        && scope.evidence.freshness.age <= self.config.freshness_budget
                        && scope.evidence.confidence >= Confidence::Medium
                })
        })
    }

    async fn actuate_migration(&self, record: &ActionRecord) -> Result<(), String> {
        if self.config.shadow {
            return Err("adaptive runtime is shadow-only".to_owned());
        }
        let Action::RecommendMigration {
            target,
            recommendation,
            ..
        } = record.action
        else {
            return Err("action is not a migration recommendation".to_owned());
        };
        if record.action.scope() != ActionScope::new(self.cluster, self.namespace) {
            return Err("migration recommendation scope changed".to_owned());
        }
        if !self.observations_current(target) {
            return Err("adaptive observations became stale before actuation".to_owned());
        }
        let target_status = self
            .node
            .status_for(ConsensusGroupId::of_tablet(target))
            .await;
        if !target_status.healthy {
            return Err("migration target replica is unhealthy".to_owned());
        }
        let control_status = self.node.status_for(ConsensusGroupId::control()).await;
        let control_membership = if self.node.hosts_control() {
            self.node
                .observe_membership(ConsensusGroupId::control().tablet())
                .await
                .ok()
        } else {
            None
        };
        let state = self
            .node
            .control_state()
            .await
            .ok_or_else(|| "control image disappeared before actuation".to_owned())?;
        let gate = control_gate(
            self.node.node(),
            &control_status,
            control_membership.as_ref(),
            Some(&state),
            self.namespace,
        )?;
        if recommendation.fence.as_u64() != gate.generation {
            return Err("migration recommendation carries a stale placement fence".to_owned());
        }
        let intent = migration_intent(&state, target, recommendation)?;
        let ids = crate::control::create_plans(&self.node, &state, &[intent]).await?;
        if ids.is_empty() {
            return Err("control pathway created no migration plan".to_owned());
        }
        Ok(())
    }

    fn record_decision(&self, state: &mut RuntimeState, decision: &AdaptiveDecision) {
        for trace in &decision.traces {
            push_bounded(
                &mut state.traces,
                trace_from_decision(trace),
                self.config.max_traces,
            );
        }
        for record in &decision.active_actions {
            push_bounded(
                &mut state.recommendations,
                recommendation_from_action(
                    Some(record.id.as_u64()),
                    &record.action,
                    ControllerSource::Baseline,
                ),
                self.config.max_recommendations,
            );
        }
        let learned_keys: BTreeSet<_> = decision
            .traces
            .iter()
            .filter(|trace| trace.source == ControllerSource::LearnedShadow)
            .filter_map(|trace| trace.action_key.clone())
            .collect();
        for action in &decision.shadow_actions {
            let source = if learned_keys.contains(&action.key()) {
                ControllerSource::LearnedShadow
            } else {
                ControllerSource::Baseline
            };
            push_bounded(
                &mut state.recommendations,
                recommendation_from_action(None, action, source),
                self.config.max_recommendations,
            );
        }
    }

    fn set_error(&self, error: impl AsRef<str>) {
        let mut state = self.state.lock().expect("cluster adaptive state mutex");
        state.last_error = Some(bounded_error(error.as_ref()));
    }
}

fn control_gate(
    local: NodeId,
    status: &NodeStatus,
    membership: Option<&ObservedMembership>,
    state: Option<&ControlState>,
    namespace: NamespaceId,
) -> Result<ControlGate, String> {
    if !status.group.is_control() {
        return Err("control status is not for the control group".to_owned());
    }
    if status.role != ReplicaRole::Leader || status.leader != Some(ReplicaId::of_node(local)) {
        return Err("local control replica is not leader".to_owned());
    }
    if !status.healthy {
        return Err("local control replica is unhealthy".to_owned());
    }
    let membership = membership.ok_or_else(|| "control membership is unknown".to_owned())?;
    if membership.is_joint {
        return Err("control membership is joint".to_owned());
    }
    if membership.leader != Some(ReplicaId::of_node(local)) || !membership.is_voter(local) {
        return Err("local control replica has no authoritative leadership view".to_owned());
    }
    let state = state.ok_or_else(|| "control image is unknown".to_owned())?;
    if !quorum_reachable(local, membership, Some(state)) {
        return Err("control quorum is not currently reachable".to_owned());
    }
    if state.namespace(namespace).is_none() {
        return Err("control namespace is unknown".to_owned());
    }
    if state
        .node(local)
        .is_none_or(|record| record.state != NodeState::Active)
    {
        return Err("local node is not active in control state".to_owned());
    }
    let generation = state.placement_version().as_u64();
    if generation == 0 {
        return Err("placement generation is unknown".to_owned());
    }
    Ok(ControlGate { generation })
}

fn quorum_reachable(
    local: NodeId,
    membership: &ObservedMembership,
    state: Option<&ControlState>,
) -> bool {
    let voters = membership.voters.len();
    let reachable = membership
        .voters
        .iter()
        .filter(|node| {
            if **node == local.as_u64() {
                return true;
            }
            let node = NodeId::from_u64(**node);
            let lifecycle_allows = state.is_none_or(|state| {
                state.node(node).is_some_and(|record| {
                    matches!(
                        record.state,
                        NodeState::Active | NodeState::Suspect | NodeState::Draining
                    )
                })
            });
            lifecycle_allows && membership.lag_of(node) <= CATCH_UP_LAG_THRESHOLD
        })
        .count();
    voters != 0 && reachable.saturating_mul(2) > voters
}

fn primary_tablet(input: &TickInput, snapshot: &ObservationSnapshot) -> Option<TabletId> {
    input
        .statuses
        .iter()
        .filter(|status| status.healthy)
        .find_map(|status| {
            let tablet = status.group.tablet();
            snapshot
                .scopes
                .iter()
                .any(|scope| scope.scope == ObservationScope::Tablet(tablet))
                .then_some(tablet)
        })
}

fn evidence_summary(
    snapshot: &ObservationSnapshot,
    primary: Option<TabletId>,
) -> (FreshnessStatus, Duration, Confidence) {
    let target = primary.and_then(|tablet| {
        snapshot
            .scopes
            .iter()
            .find(|scope| scope.scope == ObservationScope::Tablet(tablet))
    });
    let target = target.or_else(|| {
        snapshot
            .scopes
            .iter()
            .find(|scope| matches!(scope.scope, ObservationScope::Cluster(_)))
            .or_else(|| snapshot.scopes.first())
    });
    target.map_or(
        (FreshnessStatus::Future, Duration::ZERO, Confidence::Low),
        |scope| {
            (
                scope.evidence.freshness.status,
                scope.evidence.freshness.age,
                scope.evidence.confidence,
            )
        },
    )
}

fn maintenance_signal(
    snapshot: &kivi_redundancy::ControllerSnapshot,
) -> Option<MaintenanceDebtSignal> {
    if snapshot.debt.total_assets == 0 && snapshot.debt.total_bytes == 0 {
        return None;
    }
    Some(MaintenanceDebtSignal {
        checkpoint: CheckpointDebt {
            bytes: 0,
            oldest_age: Duration::ZERO,
        },
        repair: RepairDebt {
            assets: snapshot.debt.total_assets,
            bytes: snapshot.debt.total_bytes,
            oldest_age: Duration::ZERO,
        },
        scrub: ScrubDebt {
            bytes: 0,
            oldest_age: Duration::ZERO,
        },
    })
}

fn placement_loads(state: Option<&ControlState>, limit: usize) -> (Vec<u64>, Vec<u64>) {
    let Some(state) = state else {
        return (Vec::new(), Vec::new());
    };
    let nodes: Vec<NodeId> = state.active_nodes().into_iter().take(limit).collect();
    let indices: BTreeMap<NodeId, usize> = nodes
        .iter()
        .copied()
        .enumerate()
        .map(|(index, node)| (node, index))
        .collect();
    let mut worker_loads = vec![0_u64; nodes.len()];
    let mut tablet_loads = Vec::new();
    for desired in state.placements().take(limit) {
        tablet_loads.push(u64::try_from(desired.replicas.len()).unwrap_or(u64::MAX));
        for node in &desired.replicas {
            if let Some(index) = indices.get(node) {
                worker_loads[*index] = worker_loads[*index].saturating_add(1);
            }
        }
    }
    (worker_loads, tablet_loads)
}

fn topology_signal(worker_loads: &[u64], tablet_loads: &[u64]) -> Option<TopologyImbalanceSignal> {
    if worker_loads.is_empty() && tablet_loads.is_empty() {
        return None;
    }
    let busiest_worker_load = worker_loads.iter().copied().max().unwrap_or(0);
    let mean_worker_load = mean(worker_loads);
    let busiest_tablet_load = tablet_loads.iter().copied().max().unwrap_or(0);
    let mean_tablet_load = mean(tablet_loads);
    Some(TopologyImbalanceSignal {
        busiest_worker_load,
        mean_worker_load,
        busiest_tablet_load,
        mean_tablet_load,
    })
}

fn mean(values: &[u64]) -> u64 {
    if values.is_empty() {
        return 0;
    }
    let total = values
        .iter()
        .fold(0_u64, |sum, value| sum.saturating_add(*value));
    total / u64::try_from(values.len()).unwrap_or(u64::MAX)
}

fn migration_intent(
    state: &ControlState,
    target: TabletId,
    recommendation: MigrationRecommendation,
) -> Result<MigrationIntent, String> {
    if recommendation.max_bytes == 0 {
        return Err("migration recommendation has no byte bound".to_owned());
    }
    if recommendation.fence.as_u64() != state.placement_version().as_u64() {
        return Err("migration recommendation has a stale placement fence".to_owned());
    }
    let desired = state
        .desired(target)
        .ok_or_else(|| "migration target has no desired placement".to_owned())?;
    if state.has_live_topology(target) {
        return Err("migration target already has a live topology plan".to_owned());
    }
    if !desired.contains(recommendation.from) {
        return Err("migration source is not a desired voter".to_owned());
    }
    if state
        .node(recommendation.from)
        .is_none_or(|record| record.state != NodeState::Active)
    {
        return Err("migration source is not active".to_owned());
    }
    if desired.contains(recommendation.to) {
        return Err("migration destination already desires the tablet".to_owned());
    }
    if state
        .node(recommendation.to)
        .is_none_or(|record| record.state != NodeState::Active)
    {
        return Err("migration destination is not active".to_owned());
    }
    let mut desired_voters = desired.replicas.clone();
    desired_voters.retain(|node| *node != recommendation.from);
    desired_voters.push(recommendation.to);
    desired_voters.sort_by_key(|node| node.as_u64());
    desired_voters.dedup();
    Ok(MigrationIntent {
        tablet: target,
        from: recommendation.from,
        to: recommendation.to,
        desired_voters,
    })
}

fn recommendation_from_action(
    action_id: Option<u64>,
    action: &Action,
    source: ControllerSource,
) -> ClusterAdaptiveRecommendation {
    let mut recommendation = ClusterAdaptiveRecommendation {
        action_id,
        action: action_name(action).to_owned(),
        targets: action_targets(action),
        from: None,
        to: None,
        fence: None,
        max_bytes: None,
        policy: None,
        source: source_name(source).to_owned(),
        actuated: false,
        error: None,
    };
    match action {
        Action::RecommendMigration {
            recommendation: migration,
            ..
        } => {
            recommendation.from = Some(migration.from.as_u64());
            recommendation.to = Some(migration.to.as_u64());
            recommendation.fence = Some(migration.fence.as_u64());
            recommendation.max_bytes = Some(migration.max_bytes);
        }
        Action::RecommendTabletSplit { policy, .. } => {
            recommendation.policy = Some(format!(
                "min_child_bytes={},max_children={}",
                policy.min_child_bytes, policy.max_children
            ));
        }
        Action::RecommendTabletMerge { policy, .. } => {
            recommendation.policy = Some(format!(
                "min_combined_bytes={},max_sources={}",
                policy.min_combined_bytes, policy.max_sources
            ));
        }
        _ => {}
    }
    recommendation
}

fn is_shadow_mode(shadow: bool, actuation_allowed: bool) -> bool {
    shadow || !actuation_allowed
}

fn decision_mode(shadow: bool, actuation_allowed: bool) -> AdaptiveMode {
    if is_shadow_mode(shadow, actuation_allowed) {
        AdaptiveMode::Shadow
    } else {
        AdaptiveMode::Active
    }
}

fn is_actuatable_action(action: &Action) -> bool {
    matches!(action, Action::RecommendMigration { .. })
}

fn action_name(action: &Action) -> &'static str {
    match action {
        Action::AdjustMemoryBudget { .. } => "adjust_memory_budget",
        Action::ChangeMaterializationIntent { .. } => "change_materialization_intent",
        Action::ChangeRedundancyPreference { .. } => "change_redundancy_preference",
        Action::AdjustRepairBudget { .. } => "adjust_repair_budget",
        Action::AdjustScrubBudget { .. } => "adjust_scrub_budget",
        Action::RecommendTabletSplit { .. } => "recommend_tablet_split",
        Action::RecommendTabletMerge { .. } => "recommend_tablet_merge",
        Action::RecommendMigration { .. } => "recommend_migration",
    }
}

fn action_targets(action: &Action) -> Vec<u64> {
    match action.target() {
        ActionTarget::Tablets(tablets) => {
            tablets.iter().map(kivi_types::TabletId::as_u64).collect()
        }
        ActionTarget::Tablet(tablet) | ActionTarget::Migration { tablet, .. } => {
            vec![tablet.as_u64()]
        }
    }
}

fn source_name(source: ControllerSource) -> &'static str {
    match source {
        ControllerSource::Baseline => "baseline",
        ControllerSource::LearnedShadow => "learned_shadow",
        ControllerSource::Learned => "learned",
    }
}

fn trace_from_decision(trace: &kivi_control::DecisionTrace) -> ClusterAdaptiveTrace {
    ClusterAdaptiveTrace {
        sequence: trace.sequence.as_u64(),
        window: trace.window.as_u64(),
        source: source_name(trace.source).to_owned(),
        status: format!("{:?}", trace.status),
        reason: format!("{:?}", trace.reason),
        action_kind: trace
            .action_key
            .as_ref()
            .map(|key| format!("{:?}", key.kind)),
        safety_violation: trace
            .safety_violation
            .map(|violation| format!("{violation:?}")),
        ledger_rejection: trace
            .ledger_rejection
            .map(|rejection| format!("{rejection:?}")),
        fallback: trace.fallback,
    }
}

fn freshness_name(status: FreshnessStatus) -> &'static str {
    match status {
        FreshnessStatus::Current => "current",
        FreshnessStatus::Stale => "stale",
        FreshnessStatus::Future => "future",
    }
}

fn confidence_name(confidence: Confidence) -> &'static str {
    match confidence {
        Confidence::Low => "low",
        Confidence::Medium => "medium",
        Confidence::High => "high",
    }
}

fn push_bounded<T>(values: &mut VecDeque<T>, value: T, limit: usize) {
    values.push_back(value);
    while values.len() > limit.max(1) {
        values.pop_front();
    }
}

fn mark_recommendation(
    recommendations: &mut VecDeque<ClusterAdaptiveRecommendation>,
    action_id: kivi_control::ActionId,
    actuated: bool,
    error: Option<String>,
) {
    let action_id = action_id.as_u64();
    if let Some(recommendation) = recommendations
        .iter_mut()
        .find(|recommendation| recommendation.action_id == Some(action_id))
    {
        recommendation.actuated = actuated;
        recommendation.error = error;
    }
}

fn bounded_error(error: &str) -> String {
    error.chars().take(512).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use kivi_consensus::{ConsensusGroupId, ReplicaId};
    use kivi_control::{ActionScope, TabletSet};
    use kivi_observation::{
        Confidence, ObservationFabric, ObservationLimits, ObservationPeriod, ObservationScope,
        ObservationSequence, ObservationSource, ObservationWindowId, TopologyImbalanceSignal,
        TypedSignal,
    };
    use kivi_types::{ClusterId, NamespaceId, NodeId, TabletId, Ticks};

    #[test]
    fn sample_builder_bounds_scopes_and_samples() {
        let period = ObservationPeriod::new(Duration::from_millis(1)).expect("period");
        let mut builder = SampleBuilder::new(
            ObservationSequence::FIRST,
            ObservationWindowId::FIRST,
            Ticks::from_micros(1),
            period,
            2,
            1,
        );
        for id in 1..=3 {
            builder.add_scope(
                ObservationScope::Tablet(TabletId::from_u64(id)),
                ExplicitSignals {
                    topology: Some(TopologyImbalanceSignal {
                        busiest_worker_load: id,
                        mean_worker_load: 1,
                        busiest_tablet_load: id,
                        mean_tablet_load: 1,
                    }),
                    consensus: None,
                    memory: None,
                    maintenance: None,
                },
            );
        }
        assert_eq!(builder.scopes.len(), 2);
        assert_eq!(builder.samples.len(), 2);
        assert!(
            builder
                .samples
                .iter()
                .all(|sample| sample.metadata.confidence == Confidence::High)
        );
        assert!(
            builder
                .samples
                .iter()
                .all(|sample| sample.metadata.source == ObservationSource::ControlPlane)
        );
        assert!(
            builder
                .samples
                .iter()
                .all(|sample| matches!(sample.signal, TypedSignal::TopologyImbalance(_)))
        );
        let limits =
            ObservationLimits::new(2, 1, 1, 1, Duration::from_secs(5)).expect("observation limits");
        let mut fabric = ObservationFabric::new(limits);
        for sample in &builder.samples {
            fabric.collect(sample).expect("typed sample");
        }
        let snapshot = fabric.snapshot(Ticks::from_micros(1));
        assert_eq!(snapshot.scopes.len(), 2);
        assert!(snapshot.scopes.iter().all(|scope| scope.sample_count == 1));
    }

    #[test]
    fn recommendation_history_is_bounded() {
        let mut history = VecDeque::new();
        for value in 0..8 {
            push_bounded(&mut history, value, 3);
        }
        assert_eq!(history.len(), 3);
        assert_eq!(history.iter().copied().collect::<Vec<_>>(), vec![5, 6, 7]);
    }

    #[test]
    fn split_and_merge_are_recommendation_only() {
        let scope = ActionScope::new(ClusterId::from_u128(1), NamespaceId::from_u64(1));
        let split = Action::RecommendTabletSplit {
            scope,
            target: TabletId::from_u64(1),
            policy: kivi_control::TabletSplitPolicy::new(1, 2).expect("split policy"),
            expected_benefit: kivi_control::ExpectedBenefit::zero(),
        };
        let merge = Action::RecommendTabletMerge {
            scope,
            targets: TabletSet::new([TabletId::from_u64(1), TabletId::from_u64(2)])
                .expect("tablet set"),
            policy: kivi_control::TabletMergePolicy::new(1, 2).expect("merge policy"),
            expected_benefit: kivi_control::ExpectedBenefit::zero(),
        };
        assert!(!is_actuatable_action(&split));
        assert!(!is_actuatable_action(&merge));
        assert!(is_actuatable_action(&Action::RecommendMigration {
            scope,
            target: TabletId::from_u64(1),
            recommendation: MigrationRecommendation::new(
                NodeId::from_u64(1),
                NodeId::from_u64(2),
                MigrationFence::INITIAL,
                1,
            )
            .expect("migration recommendation"),
            expected_benefit: kivi_control::ExpectedBenefit::zero(),
        }));
    }

    #[test]
    fn non_leader_control_view_cannot_actuate() {
        let membership = ObservedMembership::new(
            ConsensusGroupId::control(),
            [1_u64, 2, 3].into_iter().collect(),
            BTreeSet::new(),
            Some(ReplicaId::of_node(NodeId::from_u64(2))),
            kivi_consensus::ConsensusTerm::new(1),
            false,
        );
        assert!(!quorum_reachable(
            NodeId::from_u64(1),
            &membership,
            Some(&ControlState::empty()),
        ));
        assert_ne!(
            membership.leader,
            Some(ReplicaId::of_node(NodeId::from_u64(1)))
        );
        assert_eq!(decision_mode(false, false), AdaptiveMode::Shadow);
        assert_eq!(decision_mode(false, true), AdaptiveMode::Active);
        assert_eq!(decision_mode(true, true), AdaptiveMode::Shadow);
    }
}
