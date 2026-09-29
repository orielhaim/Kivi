//! Deterministic controller scenarios.
#![allow(clippy::field_reassign_with_default)]

use core::time::Duration;

use kivi_control::{
    Action, ActionCost, ActionOutcome, ActionScope, ActuationError, Actuator, ActuatorReceipt,
    AdaptiveConfig, AdaptiveController, AdaptiveMode, BaselineConfig, BaselineController,
    ControllerFrame, ControllerSimulation, ExpectedBenefit, LearnedController, MigrationFence,
    MigrationRecommendation, SafetyContext, SafetyValidator, SimulationInput, SimulationWorkload,
    TabletSet,
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

/// Everything a scenario can vary. One `Default`, so a scenario test states
/// only the dimensions it actually exercises.
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

impl Scenario {
    #[allow(clippy::cast_precision_loss)]
    fn hot_entries(self) -> (Vec<HotKeyEntry>, f64, f64) {
        match self.hot_pattern {
            HotPattern::None => (Vec::new(), 0.0, 0.0),
            HotPattern::Many => (vec![hot_entry(1, 0.4), hot_entry(2, 0.4)], 0.4, 0.8),
            HotPattern::One => (vec![hot_entry(1, 0.8)], 0.8, 0.8),
        }
    }

    #[allow(clippy::cast_precision_loss)]
    fn snapshot(self, window: u64) -> ObservationSnapshot {
        let window_id = ObservationWindowId::from_u64(window);
        let sequence_id = ObservationSequence::from_u64(window);
        let observed = Ticks::from_micros(window);
        let (entries, top_one_share, top_ten_share) = self.hot_entries();
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
                p50_ns: self.latency_ns / 2,
                p99_ns: self.latency_ns,
                p999_ns: self.latency_ns,
                max_ns: self.latency_ns,
            }),
            queueing: None,
            compute: None,
            memory: Some(MemorySummary {
                resident_bytes: self.resident_bytes,
                logical_bytes: self.logical_bytes,
                residency_ratio: Some(
                    self.resident_bytes as f64 / self.logical_bytes.max(1) as f64,
                ),
            }),
            memory_pressure: Some(MemoryPressureSummary {
                mean_pressure: UnitInterval::new(self.pressure).unwrap_or(UnitInterval::ZERO),
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
                    assets: self.repair_assets,
                    bytes: self.repair_bytes,
                    oldest_age: self.repair_age,
                },
                scrub: ScrubDebtSummary {
                    bytes: 0,
                    oldest_age: Duration::ZERO,
                },
            }),
            redundancy: Some(RedundancySummary {
                logical_bytes: self.logical_bytes,
                physical_bytes: self.logical_bytes.saturating_mul(2),
                fragments: 3,
                amplification_ratio: Some(2.0),
            }),
            degraded_assets: Some(DegradedAssetsSummary {
                assets: self.degraded_assets,
                oldest_age: self.degraded_age,
            }),
            migration_cost: None,
            topology: Some(TopologySummary {
                worker_max_to_mean: Some(self.worker_imbalance),
                tablet_max_to_mean: Some(self.tablet_imbalance),
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
                mean_score: CriticalityScore::from_u64(self.criticality),
                peak_score: CriticalityScore::from_u64(self.criticality),
                observation_count: 1,
                access_count: 1,
            }),
            hardware: kivi_observation::HardwareSummary::absent(),
        };
        ObservationSnapshot {
            as_of: observed,
            last_sequence: Some(sequence_id),
            accepted_samples: window,
            scopes: vec![scope],
        }
    }

    fn frame(self, window: u64) -> ControllerFrame {
        ControllerFrame::new(
            self.snapshot(window),
            ActionScope::new(ClusterId::from_u128(7), NamespaceId::from_u64(1)),
            ObservationSequence::from_u64(window),
            ObservationWindowId::from_u64(window),
            1,
            Ticks::from_micros(window),
        )
    }

    /// The safety context the frame is judged against, derived from the same
    /// scenario numbers so a case can never describe a cluster the snapshot
    /// does not report.
    fn context(self) -> SafetyContext {
        let mut context = context();
        context.target_logical_bytes = self.logical_bytes;
        context.target_combined_bytes = self.logical_bytes.saturating_mul(2);
        context.current_replicas = 3;
        context.current_failure_domains = 2;
        context.current_fence = MigrationFence::INITIAL;
        context.tablet_count = 3;
        context.foreground_latency_bad = self.latency_ns >= 1_000_000;
        context.critical_repair_debt = self.repair_assets > 0;
        context.repair_debt_age = self.repair_age;
        if self.merge_candidate {
            context.merge_candidates = vec![(
                TabletSet::new([TabletId::from_u64(1), TabletId::from_u64(2)])
                    .expect("merge target set"),
                self.logical_bytes.saturating_mul(2),
            )];
        } else {
            context.migration_candidates = vec![
                MigrationRecommendation::new(
                    NodeId::from_u64(1),
                    NodeId::from_u64(2),
                    MigrationFence::INITIAL,
                    self.logical_bytes.max(1),
                )
                .expect("migration recommendation"),
            ];
        }
        context
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
                &scenario.frame(index as u64 + 1),
                &scenario.context(),
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

/// Actions proposed at each window, flattened: the common shape of every
/// scenario assertion below.
fn flat_actions(scenarios: &[Scenario], sustained_windows: u16) -> Vec<Action> {
    run_scenario_actions(scenarios, sustained_windows)
        .into_iter()
        .flatten()
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
fn alternating_pressure_converges_without_flapping() {
    let mut controller = BaselineController::new(BaselineConfig {
        sustained_windows: 2,
        min_samples: 1,
        action_cooldown_windows: 3,
        ..BaselineConfig::default()
    });
    let validator = SafetyValidator::default();
    let mut actions = 0usize;
    for window in 1..=40u64 {
        let scenario = Scenario {
            pressure: if window % 2 == 0 { 0.95 } else { 0.05 },
            criticality: if window % 2 == 0 { 0 } else { 500 },
            ..Scenario::default()
        };
        actions += controller
            .propose(&scenario.frame(window), &scenario.context(), &validator)
            .actions
            .len();
    }
    assert!(
        actions <= 8,
        "alternating workload produced {actions} actions"
    );
}

#[test]
fn shadow_mode_records_suggestions_without_actuation() {
    let mut controller = AdaptiveController::new(
        AdaptiveConfig {
            mode: AdaptiveMode::Shadow,
            ..AdaptiveConfig::default()
        },
        BaselineController::new(BaselineConfig {
            sustained_windows: 1,
            min_samples: 1,
            ..BaselineConfig::default()
        }),
        Some(LearnedController::default()),
    );
    let hot = Scenario {
        pressure: 0.95,
        ..Scenario::default()
    };
    let decision = controller.decide(&hot.frame(1), &hot.context());
    assert!(!decision.shadow_actions.is_empty());
    assert!(decision.active_actions.is_empty());
    let mut actuator = CountingActuator { calls: 0 };
    assert!(matches!(
        controller.execute(
            &mut actuator,
            kivi_control::ActionId::FIRST,
            hot.frame(1).action_observation(),
        ),
        Err(ActuationError::ShadowMode)
    ));
    assert_eq!(actuator.calls, 0, "shadow mode reached the actuator");
}

#[test]
fn gradual_pressure_reaches_a_bounded_policy_without_overshooting() {
    let rising = Scenario {
        pressure: 0.4,
        ..Scenario::default()
    };
    let warm = Scenario {
        pressure: 0.6,
        ..Scenario::default()
    };
    let hot = Scenario {
        pressure: 0.8,
        ..Scenario::default()
    };
    // Sustained pressure crosses the window threshold only on the fourth
    // window, and then proposes rather than escalating further.
    let actions = run_scenario_actions(&[rising, warm, hot, hot, hot], 3);
    assert!(actions[..4].iter().all(Vec::is_empty));
    assert!(
        actions[4]
            .iter()
            .any(|action| matches!(action, Action::AdjustMemoryBudget { .. }))
    );
    assert!(actions.iter().flatten().count() <= 8);
}

#[test]
fn burst_and_recovery_reverse_materialization_intent() {
    let burst = Scenario {
        pressure: 0.95,
        ..Scenario::default()
    };
    let recovered = Scenario {
        pressure: 0.05,
        criticality: 500,
        ..Scenario::default()
    };
    let all = flat_actions(&[burst, burst, burst, recovered, recovered], 1);
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
    let spread = Scenario {
        logical_bytes: 4_000_000,
        resident_bytes: 2_000_000,
        hot_pattern: HotPattern::Many,
        ..Scenario::default()
    };
    let actions = flat_actions(&[spread], 1);
    assert!(
        actions
            .iter()
            .any(|action| matches!(action, Action::RecommendTabletSplit { .. }))
    );
    assert!(
        !actions
            .iter()
            .any(|action| matches!(action, Action::RecommendMigration { .. }))
    );
}

#[test]
fn moving_node_skew_recommends_fenced_migration_only_when_imbalanced() {
    let balanced = Scenario::default();
    let skewed = Scenario {
        worker_imbalance: 3.0,
        ..Scenario::default()
    };
    let actions = run_scenario_actions(&[balanced, skewed, balanced, skewed], 1);
    let migration = |actions: &[Action]| {
        actions
            .iter()
            .any(|action| matches!(action, Action::RecommendMigration { .. }))
    };
    assert!(!migration(&actions[0]));
    assert!(migration(&actions[1]));
    assert!(!migration(&actions[2]));
    assert!(migration(&actions[3]));
}

#[test]
fn foreground_spike_and_degraded_assets_throttle_scrub_before_repair() {
    let pressured = Scenario {
        latency_ns: 2_000_000,
        repair_assets: 4,
        repair_bytes: 10_000_000,
        repair_age: Duration::from_secs(60),
        degraded_assets: 5,
        degraded_age: Duration::from_secs(120),
        ..Scenario::default()
    };
    let actions = flat_actions(&[pressured], 1);
    assert!(
        actions
            .iter()
            .any(|action| matches!(action, Action::AdjustScrubBudget { .. }))
    );
    assert!(
        !actions
            .iter()
            .any(|action| matches!(action, Action::AdjustRepairBudget { .. }))
    );
}

#[test]
fn cold_large_and_hot_small_assets_select_opposite_redundancy_families() {
    let cold = Scenario {
        logical_bytes: 4_000_000,
        resident_bytes: 2_000_000,
        ..Scenario::default()
    };
    let hot = Scenario {
        hot_pattern: HotPattern::One,
        ..Scenario::default()
    };
    let family = |action: &Action| match action {
        Action::ChangeRedundancyPreference { preference, .. } => match preference {
            kivi_control::RedundancyPreference::ErasureCoding(_) => "erasure",
            kivi_control::RedundancyPreference::Replication(_) => "replication",
        },
        _ => "other",
    };
    let cold_family = flat_actions(&[cold], 1)
        .iter()
        .map(family)
        .find(|family| *family != "other");
    let hot_family = flat_actions(&[hot], 1)
        .iter()
        .map(family)
        .find(|family| *family != "other");
    assert_eq!(cold_family, Some("erasure"));
    assert_eq!(hot_family, Some("replication"));
}

#[test]
fn merge_needs_both_a_candidate_and_imbalance() {
    let mergeable = Scenario {
        logical_bytes: 600_000,
        resident_bytes: 600_000,
        worker_imbalance: 2.0,
        tablet_imbalance: 2.0,
        merge_candidate: true,
        ..Scenario::default()
    };
    let stable = Scenario {
        merge_candidate: true,
        ..Scenario::default()
    };
    let proposes_merge = |actions: &[Action]| {
        actions
            .iter()
            .any(|action| matches!(action, Action::RecommendTabletMerge { .. }))
    };
    assert!(proposes_merge(&flat_actions(&[mergeable], 1)));
    // A candidate with no skew must not churn the topology.
    assert!(!proposes_merge(&flat_actions(&[stable], 1)));
}

#[test]
fn learned_alternative_uses_action_features_and_never_bypasses_safety() {
    let scenario = Scenario {
        pressure: 0.9,
        ..Scenario::default()
    };
    let frame = scenario.frame(1);
    let context = scenario.context();
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
    assert!(suggestion.action.as_ref().is_some_and(|candidate| {
        candidate.key() != action.key() && validator.validate(candidate, &context).is_ok()
    }));
}

fn simulation_workload() -> SimulationWorkload {
    let hot = Scenario {
        pressure: 0.9,
        ..Scenario::default()
    };
    let inputs = (1..=48)
        .map(|window| SimulationInput::new(hot.snapshot(window), hot.context()))
        .collect();
    SimulationWorkload::new(7, inputs)
}

#[test]
fn shadow_reports_execute_shadow_and_stay_bounded() {
    fn run(learned: bool) -> kivi_control::SimulationReport {
        let mut controller = AdaptiveController::new(
            AdaptiveConfig {
                max_traces: 16,
                ..AdaptiveConfig::default()
            },
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
    assert!(baseline.executed > 0);
    assert!(learned.executed > 0);
    assert!(learned.shadow > 0, "the learned pass never shadowed");
    assert_eq!(learned.traces.len(), 16, "trace ring buffer is bounded");
    assert_eq!(baseline.traces.len(), 16);
}
