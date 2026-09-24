//! Deterministic Phase 12 controller scenarios.
#![allow(clippy::field_reassign_with_default)]

use core::time::Duration;

use kivi_control::{
    Action, ActionCost, ActionOutcome, ActionScope, ActionState, ActuationError, Actuator,
    ActuatorReceipt, AdaptiveConfig, AdaptiveController, AdaptiveMode, BaselineConfig,
    BaselineController, ControllerFrame, ControllerSimulation, ExpectedBenefit, LearnedController,
    MigrationFence, MigrationRecommendation, SafetyContext, SafetyValidator, SimulationInput,
    SimulationWorkload, TabletSet,
};
use kivi_observation::{
    AggregateEvidence, CheckpointDebtSummary, Confidence, CriticalityScore, DegradedAssetsSummary,
    ExecutionClass, ExecutionCriticalitySummary, Freshness, HotKeyClassification,
    HotKeyConcentration, HotKeyEntry, HotKeyId, HotKeySummary, HotnessClass, LatencySummary,
    MaintenanceDebtSummary, MemoryPressureSummary, MemorySummary, ObservationScope,
    ObservationSequence, ObservationSnapshot, ObservationSource, ObservationSources,
    ObservationWindowId, RedundancySummary, RepairDebtSummary, ScopeSnapshot, ScrubDebtSummary,
    TopologySummary, UnitInterval,
};
use kivi_types::{ClusterId, NamespaceId, NodeId, TabletId, Ticks};

fn snapshot(window: u64, sequence: u64, pressure: f64, criticality: u64) -> ObservationSnapshot {
    let window_id = ObservationWindowId::from_u64(window);
    let sequence_id = ObservationSequence::from_u64(sequence);
    let observed = Ticks::from_micros(sequence);
    let evidence = AggregateEvidence {
        oldest_observed_at: observed,
        latest_observed_at: observed,
        sources: ObservationSources::from_source(ObservationSource::Simulator),
        confidence: Confidence::High,
        freshness: Freshness::classify(observed, observed, Duration::from_secs(5)),
    };
    let scope = ScopeSnapshot {
        scope: ObservationScope::Tablet(TabletId::from_u64(1)),
        first_window: window_id,
        latest_window: window_id,
        latest_sequence: sequence_id,
        sample_count: 8,
        evidence,
        request: None,
        latency: Some(LatencySummary {
            sample_count: 8,
            p50_ns: 1_000,
            p99_ns: 10_000,
            p999_ns: 20_000,
            max_ns: 20_000,
        }),
        queueing: None,
        compute: None,
        memory: None,
        memory_pressure: Some(MemoryPressureSummary {
            mean_pressure: UnitInterval::new(pressure).unwrap_or(UnitInterval::ZERO),
            reclaim_attempts: 0,
            reclaim_failures: 0,
        }),
        compression: None,
        offcore_latency: None,
        io: None,
        consensus_lag: None,
        maintenance_debt: None,
        redundancy: None,
        degraded_assets: None,
        migration_cost: None,
        topology: None,
        hot_keys: HotKeySummary {
            concentration: kivi_observation::HotKeyConcentration::empty(),
            entries: Vec::new(),
        },
        execution_criticality: Some(ExecutionCriticalitySummary {
            mean_score: CriticalityScore::from_u64(criticality),
            peak_score: CriticalityScore::from_u64(criticality),
            observation_count: 1,
            access_count: 1,
        }),
    };
    ObservationSnapshot {
        as_of: observed,
        last_sequence: Some(sequence_id),
        accepted_samples: sequence,
        scopes: vec![scope],
    }
}

fn frame(window: u64, sequence: u64, pressure: f64, criticality: u64) -> ControllerFrame {
    ControllerFrame::new(
        snapshot(window, sequence, pressure, criticality),
        ActionScope::new(ClusterId::from_u128(7), NamespaceId::from_u64(1)),
        ObservationSequence::from_u64(sequence),
        ObservationWindowId::from_u64(window),
        1,
        Ticks::from_micros(sequence),
    )
}

fn context() -> SafetyContext {
    SafetyContext::new(
        ActionScope::new(ClusterId::from_u128(7), NamespaceId::from_u64(1)),
        TabletId::from_u64(1),
    )
}

#[derive(Clone, Copy)]
enum HotPattern {
    None,
    Many,
    One,
}

#[derive(Clone, Copy)]
struct Scenario {
    pressure: f64,
    criticality: u64,
    logical_bytes: u64,
    resident_bytes: u64,
    repair_assets: u64,
    repair_bytes: u64,
    repair_age: Duration,
    latency_ns: u64,
    worker_imbalance: f64,
    tablet_imbalance: f64,
    hot_pattern: HotPattern,
    degraded_assets: u64,
    degraded_age: Duration,
    merge_candidate: bool,
}

impl Default for Scenario {
    fn default() -> Self {
        Self {
            pressure: 0.1,
            criticality: 0,
            logical_bytes: 500_000,
            resident_bytes: 500_000,
            repair_assets: 0,
            repair_bytes: 0,
            repair_age: Duration::ZERO,
            latency_ns: 10_000,
            worker_imbalance: 1.0,
            tablet_imbalance: 1.0,
            hot_pattern: HotPattern::None,
            degraded_assets: 0,
            degraded_age: Duration::ZERO,
            merge_candidate: false,
        }
    }
}

fn hot_entry(key: u128, share: f64) -> HotKeyEntry {
    HotKeyEntry {
        key: HotKeyId::from_u128(key),
        estimated_count: 5,
        error_bound: 0,
        share,
        critical_path_hits: 0,
        tail_hits: 0,
        stall_time_ns: 0,
        classification: HotKeyClassification {
            hotness: HotnessClass::Hot,
            execution: ExecutionClass::Ordinary,
        },
    }
}

#[allow(clippy::cast_precision_loss)]
fn scenario_snapshot(window: u64, sequence: u64, scenario: Scenario) -> ObservationSnapshot {
    let window_id = ObservationWindowId::from_u64(window);
    let sequence_id = ObservationSequence::from_u64(sequence);
    let observed = Ticks::from_micros(sequence);
    let (entries, top_one_share, top_ten_share) = match scenario.hot_pattern {
        HotPattern::None => (Vec::new(), 0.0, 0.0),
        HotPattern::Many => (vec![hot_entry(1, 0.4), hot_entry(2, 0.4)], 0.4, 0.8),
        HotPattern::One => (vec![hot_entry(1, 0.8)], 0.8, 0.8),
    };
    let scope = ScopeSnapshot {
        scope: ObservationScope::Tablet(TabletId::from_u64(1)),
        first_window: window_id,
        latest_window: window_id,
        latest_sequence: sequence_id,
        sample_count: 8,
        evidence: AggregateEvidence {
            oldest_observed_at: observed,
            latest_observed_at: observed,
            sources: ObservationSources::from_source(ObservationSource::Simulator),
            confidence: Confidence::High,
            freshness: Freshness::classify(observed, observed, Duration::from_secs(5)),
        },
        request: None,
        latency: Some(LatencySummary {
            sample_count: 8,
            p50_ns: scenario.latency_ns / 2,
            p99_ns: scenario.latency_ns,
            p999_ns: scenario.latency_ns,
            max_ns: scenario.latency_ns,
        }),
        queueing: None,
        compute: None,
        memory: Some(MemorySummary {
            resident_bytes: scenario.resident_bytes,
            logical_bytes: scenario.logical_bytes,
            residency_ratio: Some(
                scenario.resident_bytes as f64 / scenario.logical_bytes.max(1) as f64,
            ),
        }),
        memory_pressure: Some(MemoryPressureSummary {
            mean_pressure: UnitInterval::new(scenario.pressure).unwrap_or(UnitInterval::ZERO),
            reclaim_attempts: 0,
            reclaim_failures: 0,
        }),
        compression: None,
        offcore_latency: None,
        io: None,
        consensus_lag: None,
        maintenance_debt: Some(MaintenanceDebtSummary {
            checkpoint: CheckpointDebtSummary {
                bytes: 0,
                oldest_age: Duration::ZERO,
            },
            repair: RepairDebtSummary {
                assets: scenario.repair_assets,
                bytes: scenario.repair_bytes,
                oldest_age: scenario.repair_age,
            },
            scrub: ScrubDebtSummary {
                bytes: 0,
                oldest_age: Duration::ZERO,
            },
        }),
        redundancy: Some(RedundancySummary {
            logical_bytes: scenario.logical_bytes,
            physical_bytes: scenario.logical_bytes.saturating_mul(2),
            fragments: 3,
            amplification_ratio: Some(2.0),
        }),
        degraded_assets: Some(DegradedAssetsSummary {
            assets: scenario.degraded_assets,
            oldest_age: scenario.degraded_age,
        }),
        migration_cost: None,
        topology: Some(TopologySummary {
            worker_max_to_mean: Some(scenario.worker_imbalance),
            tablet_max_to_mean: Some(scenario.tablet_imbalance),
        }),
        hot_keys: HotKeySummary {
            concentration: HotKeyConcentration {
                top_one_share,
                top_ten_share,
                herfindahl: top_one_share * top_one_share,
            },
            entries,
        },
        execution_criticality: Some(ExecutionCriticalitySummary {
            mean_score: CriticalityScore::from_u64(scenario.criticality),
            peak_score: CriticalityScore::from_u64(scenario.criticality),
            observation_count: 1,
            access_count: 1,
        }),
    };
    ObservationSnapshot {
        as_of: observed,
        last_sequence: Some(sequence_id),
        accepted_samples: sequence,
        scopes: vec![scope],
    }
}

fn scenario_frame(window: u64, scenario: Scenario) -> ControllerFrame {
    ControllerFrame::new(
        scenario_snapshot(window, window, scenario),
        ActionScope::new(ClusterId::from_u128(7), NamespaceId::from_u64(1)),
        ObservationSequence::from_u64(window),
        ObservationWindowId::from_u64(window),
        1,
        Ticks::from_micros(window),
    )
}

fn scenario_context(scenario: Scenario) -> SafetyContext {
    let mut context = SafetyContext::new(
        ActionScope::new(ClusterId::from_u128(7), NamespaceId::from_u64(1)),
        TabletId::from_u64(1),
    );
    context.target_logical_bytes = scenario.logical_bytes;
    context.target_combined_bytes = scenario.logical_bytes.saturating_mul(2);
    context.current_replicas = 3;
    context.current_failure_domains = 2;
    context.current_fence = MigrationFence::INITIAL;
    if !scenario.merge_candidate {
        context.migration_candidates = vec![
            MigrationRecommendation::new(
                NodeId::from_u64(1),
                NodeId::from_u64(2),
                MigrationFence::INITIAL,
                scenario.logical_bytes.max(1),
            )
            .expect("migration recommendation"),
        ];
    }
    if scenario.merge_candidate {
        context.merge_candidates = vec![(
            TabletSet::new([TabletId::from_u64(1), TabletId::from_u64(2)])
                .expect("merge target set"),
            scenario.logical_bytes.saturating_mul(2),
        )];
    }
    context.tablet_count = 3;
    context.foreground_latency_bad = scenario.latency_ns >= 1_000_000;
    context.critical_repair_debt = scenario.repair_assets > 0;
    context.repair_debt_age = scenario.repair_age;
    context
}

fn run_scenario_actions(scenarios: &[Scenario], sustained_windows: u16) -> Vec<Vec<Action>> {
    let mut controller = BaselineController::new(BaselineConfig {
        sustained_windows,
        min_samples: 1,
        ..BaselineConfig::default()
    });
    let validator = SafetyValidator::default();
    scenarios
        .iter()
        .enumerate()
        .map(|(index, scenario)| {
            let proposal = controller.propose(
                &scenario_frame(index as u64 + 1, *scenario),
                &scenario_context(*scenario),
                &validator,
            );
            for action in &proposal.actions {
                controller.on_outcome(
                    action,
                    &ActionOutcome::success(
                        action.expected_benefit(),
                        Duration::ZERO,
                        ActionCost::zero(),
                    ),
                );
            }
            proposal.actions
        })
        .collect()
}

struct CountingActuator {
    calls: u64,
}

impl Actuator for CountingActuator {
    fn execute(
        &mut self,
        action_id: kivi_control::ActionId,
        action: &Action,
    ) -> Result<ActuatorReceipt, ActuationError> {
        self.calls += 1;
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

#[test]
fn sustained_pressure_executes_legal_bounded_actions() {
    let mut controller = AdaptiveController::new(
        AdaptiveConfig::default(),
        BaselineController::new(BaselineConfig {
            sustained_windows: 2,
            min_samples: 1,
            ..BaselineConfig::default()
        }),
        Some(LearnedController::default()),
    );
    let mut actuator = CountingActuator { calls: 0 };
    for window in 1..=8 {
        let current = frame(window, window, 0.95, 0);
        let decision = controller.decide(&current, &context());
        for record in decision.active_actions {
            let result = controller.execute(&mut actuator, record.id, current.action_observation());
            assert!(result.is_ok());
            let result = result.expect("action executes");
            assert!(matches!(
                result.state,
                ActionState::Completed | ActionState::Failed
            ));
        }
    }
    assert!(actuator.calls <= 8);
    assert!(controller.ledger().records().all(|record| matches!(
        record.action.target(),
        kivi_control::ActionTarget::Tablet(_)
    )));
}

#[test]
fn alternating_pressure_converges_without_flapping() {
    let config = BaselineConfig {
        sustained_windows: 2,
        min_samples: 1,
        action_cooldown_windows: 3,
        ..BaselineConfig::default()
    };
    let mut controller = BaselineController::new(config);
    let validator = kivi_control::SafetyValidator::default();
    let mut actions = 0usize;
    for window in 1..=40 {
        let pressure = if window % 2 == 0 { 0.95 } else { 0.05 };
        let proposal = controller.propose(
            &frame(
                window,
                window,
                pressure,
                if pressure < 0.1 { 500 } else { 0 },
            ),
            &context(),
            &validator,
        );
        actions += proposal.actions.len();
    }
    assert!(
        actions <= 8,
        "alternating workload produced {actions} actions"
    );
}

#[test]
fn shadow_mode_records_suggestions_without_actuation() {
    let config = AdaptiveConfig {
        mode: AdaptiveMode::Shadow,
        ..AdaptiveConfig::default()
    };
    let mut controller = AdaptiveController::new(
        config,
        BaselineController::new(BaselineConfig {
            sustained_windows: 1,
            min_samples: 1,
            ..BaselineConfig::default()
        }),
        Some(LearnedController::default()),
    );
    let decision = controller.decide(&frame(1, 1, 0.95, 0), &context());
    assert!(!decision.shadow_actions.is_empty());
    assert!(decision.active_actions.is_empty());
    let mut actuator = CountingActuator { calls: 0 };
    assert!(matches!(
        controller.execute(
            &mut actuator,
            kivi_control::ActionId::FIRST,
            frame(1, 1, 0.95, 0).action_observation(),
        ),
        Err(ActuationError::ShadowMode)
    ));
    assert_eq!(actuator.calls, 0);
}

#[test]
fn fixed_seed_replay_is_deterministic() {
    fn run() -> (Vec<Action>, Vec<kivi_control::DecisionTrace>) {
        let mut controller = AdaptiveController::new(
            AdaptiveConfig::default(),
            BaselineController::new(BaselineConfig {
                sustained_windows: 2,
                min_samples: 1,
                ..BaselineConfig::default()
            }),
            None,
        );
        let mut actions = Vec::new();
        let mut traces = Vec::new();
        for window in 1..=10 {
            let current = frame(window, window, if window % 3 == 0 { 0.9 } else { 0.2 }, 100);
            let decision = controller.decide(&current, &context());
            actions.extend(
                decision
                    .active_actions
                    .into_iter()
                    .map(|record| record.action),
            );
            traces.extend(decision.traces);
        }
        (actions, traces)
    }
    assert_eq!(run(), run());
}

#[test]
fn gradual_pressure_reaches_a_bounded_policy_without_overshooting() {
    let mut rising = Scenario::default();
    rising.pressure = 0.4;
    let mut warm = rising;
    warm.pressure = 0.6;
    let mut hot = rising;
    hot.pressure = 0.8;
    let actions = run_scenario_actions(&[rising, warm, hot, hot, hot], 3);
    assert!(actions[..4].iter().all(Vec::is_empty));
    assert!(
        actions[4]
            .iter()
            .any(|action| { matches!(action, Action::AdjustMemoryBudget { .. }) })
    );
    assert!(actions.iter().flatten().count() <= 8);
}

#[test]
fn burst_and_recovery_reverse_materialization_intent() {
    let mut burst = Scenario::default();
    burst.pressure = 0.95;
    let mut recovered = Scenario::default();
    recovered.pressure = 0.05;
    recovered.criticality = 500;
    let actions = run_scenario_actions(&[burst, burst, burst, recovered, recovered], 1);
    let all = actions.into_iter().flatten().collect::<Vec<_>>();
    let demotion = all
        .iter()
        .position(|action| {
            matches!(
                action,
                Action::ChangeMaterializationIntent {
                    intent: kivi_control::MaterializationIntent::Evict,
                    ..
                }
            )
        })
        .expect("burst evicts reconstructible data");
    let promotion = all
        .iter()
        .position(|action| {
            matches!(
                action,
                Action::ChangeMaterializationIntent {
                    intent: kivi_control::MaterializationIntent::Materialize,
                    ..
                }
            )
        })
        .expect("recovery promotes critical data");
    assert!(demotion < promotion);
}

#[test]
fn sudden_many_key_hotspot_recommends_split_not_single_key_migration() {
    let mut scenario = Scenario::default();
    scenario.logical_bytes = 4_000_000;
    scenario.resident_bytes = 2_000_000;
    scenario.hot_pattern = HotPattern::Many;
    let actions = run_scenario_actions(&[scenario], 1);
    assert!(
        actions[0]
            .iter()
            .any(|action| { matches!(action, Action::RecommendTabletSplit { .. }) })
    );
    assert!(
        !actions[0]
            .iter()
            .any(|action| { matches!(action, Action::RecommendMigration { .. }) })
    );
}

#[test]
fn moving_node_skew_recommends_fenced_migration_only_when_imbalanced() {
    let mut balanced = Scenario::default();
    balanced.logical_bytes = 500_000;
    let mut imbalanced = balanced;
    imbalanced.worker_imbalance = 3.0;
    let actions = run_scenario_actions(&[balanced, imbalanced, balanced, imbalanced], 1);
    assert!(actions[0].is_empty());
    assert!(
        actions[1]
            .iter()
            .any(|action| { matches!(action, Action::RecommendMigration { .. }) })
    );
    assert!(actions[2].is_empty());
    assert!(
        actions[3]
            .iter()
            .any(|action| { matches!(action, Action::RecommendMigration { .. }) })
    );
}

#[test]
fn foreground_spike_and_degraded_assets_throttle_scrub_before_repair() {
    let mut scenario = Scenario::default();
    scenario.latency_ns = 2_000_000;
    scenario.repair_assets = 4;
    scenario.repair_bytes = 10_000_000;
    scenario.repair_age = Duration::from_secs(60);
    scenario.degraded_assets = 5;
    scenario.degraded_age = Duration::from_secs(120);
    let actions = run_scenario_actions(&[scenario], 1);
    assert!(
        actions[0]
            .iter()
            .any(|action| { matches!(action, Action::AdjustScrubBudget { .. }) })
    );
    assert!(
        !actions[0]
            .iter()
            .any(|action| { matches!(action, Action::AdjustRepairBudget { .. }) })
    );
}

#[test]
fn cold_large_and_hot_small_assets_select_opposite_redundancy_families() {
    let mut cold = Scenario::default();
    cold.logical_bytes = 4_000_000;
    cold.resident_bytes = 2_000_000;
    let mut hot = Scenario::default();
    hot.logical_bytes = 500_000;
    hot.resident_bytes = 500_000;
    hot.hot_pattern = HotPattern::One;
    let cold_actions = run_scenario_actions(&[cold], 1);
    let hot_actions = run_scenario_actions(&[hot], 1);
    assert!(cold_actions[0].iter().any(|action| {
        matches!(
            action,
            Action::ChangeRedundancyPreference {
                preference: kivi_control::RedundancyPreference::ErasureCoding(_),
                ..
            }
        )
    }));
    assert!(hot_actions[0].iter().any(|action| {
        matches!(
            action,
            Action::ChangeRedundancyPreference {
                preference: kivi_control::RedundancyPreference::Replication(_),
                ..
            }
        )
    }));
}

#[test]
fn adjacent_merge_is_recommendation_only_and_respects_minimum_count() {
    let mut scenario = Scenario::default();
    scenario.logical_bytes = 600_000;
    scenario.resident_bytes = 600_000;
    scenario.worker_imbalance = 2.0;
    scenario.tablet_imbalance = 2.0;
    scenario.merge_candidate = true;
    let actions = run_scenario_actions(&[scenario], 1);
    assert!(
        actions[0]
            .iter()
            .any(|action| { matches!(action, Action::RecommendTabletMerge { .. }) })
    );
}

#[test]
fn stable_merge_candidates_do_not_create_topology_actions() {
    let mut scenario = Scenario::default();
    scenario.merge_candidate = true;
    let actions = run_scenario_actions(&[scenario], 1);
    assert!(
        !actions[0]
            .iter()
            .any(|action| matches!(action, Action::RecommendTabletMerge { .. }))
    );
}

#[test]
fn learned_alternative_uses_action_features_and_never_bypasses_safety() {
    let mut scenario = Scenario::default();
    scenario.pressure = 0.9;
    let frame = scenario_frame(1, scenario);
    let context = scenario_context(scenario);
    let action = Action::AdjustMemoryBudget {
        scope: context.scope,
        target: context.primary_target,
        budget: kivi_control::MemoryBudget::new(400_000).expect("memory budget"),
        expected_benefit: ExpectedBenefit::new(500_000, 0, 0, 500_000, 0, 0, 0, 0, 0, 0)
            .expect("benefit"),
    };
    let validator = SafetyValidator::default();
    let mut learned = LearnedController::default();
    let outcome = ActionOutcome::success(
        ExpectedBenefit::new(500_000, 0, 0, 500_000, 0, 0, 0, 0, 0, 0).expect("benefit"),
        Duration::ZERO,
        ActionCost::zero(),
    );
    for _ in 0..40 {
        learned
            .observe_outcome(&frame, &context, &action, &outcome, &validator)
            .expect("valid feedback");
    }
    let suggestion = learned.suggest(&frame, &context, Some(&action), &validator);
    assert!(!suggestion.fallback_to_baseline);
    assert!(suggestion.action.is_some());
    assert!(suggestion.action.as_ref().is_some_and(|candidate| {
        candidate.key() != action.key() && validator.validate(candidate, &context).is_ok()
    }));
}

fn simulation_workload() -> SimulationWorkload {
    let inputs = (1..=48)
        .map(|window| {
            let scenario = Scenario {
                pressure: 0.9,
                ..Scenario::default()
            };
            SimulationInput::new(
                snapshot(window, window, scenario.pressure, scenario.criticality),
                context(),
            )
        })
        .collect();
    SimulationWorkload::new(7, inputs)
}

#[test]
fn baseline_and_learned_shadow_reports_are_deterministic_and_bounded() {
    fn run(learned: bool) -> kivi_control::SimulationReport {
        let config = AdaptiveConfig {
            max_traces: 16,
            ..AdaptiveConfig::default()
        };
        let mut controller = AdaptiveController::new(
            config,
            BaselineController::new(BaselineConfig {
                sustained_windows: 2,
                min_samples: 1,
                ..BaselineConfig::default()
            }),
            learned.then(LearnedController::default),
        );
        let mut simulation = ControllerSimulation::new(
            7,
            ObservationSequence::FIRST,
            ObservationWindowId::FIRST,
            Ticks::from_micros(1),
            Duration::from_millis(100),
        );
        simulation.run(
            &mut controller,
            &simulation_workload(),
            &mut CountingActuator { calls: 0 },
        )
    }
    let baseline = run(false);
    let learned = run(true);
    assert_eq!(baseline, run(false));
    assert_eq!(learned, run(true));
    assert!(baseline.executed > 0);
    assert!(learned.executed > 0);
    assert!(learned.shadow > 0);
    assert!(learned.traces.len() <= 16);
    assert!(baseline.traces.len() <= 16);
}
