//! In-process baseline and learned-shadow adaptive controller benchmark.

use core::time::Duration;

use divan::{Bencher, black_box};
use kivi_control::{
    Action, ActionCost, ActionOutcome, ActionScope, ActuationError, Actuator, ActuatorReceipt,
    AdaptiveConfig, AdaptiveController, BaselineConfig, BaselineController, ControllerFrame,
    LearnedController, SafetyContext,
};
use kivi_observation::{
    AggregateEvidence, Confidence, CriticalityScore, ExecutionCriticalitySummary, Freshness,
    HotKeySummary, LatencySummary, MemoryPressureSummary, MemorySummary, ObservationScope,
    ObservationSequence, ObservationSnapshot, ObservationSource, ObservationSources,
    ObservationWindowId, ScopeSnapshot, UnitInterval,
};
use kivi_types::{ClusterId, NamespaceId, TabletId, Ticks};

const WINDOWS: u64 = 256;

fn snapshot(window: u64) -> ObservationSnapshot {
    let sequence = ObservationSequence::from_u64(window);
    let observed = Ticks::from_micros(window);
    let scope = ScopeSnapshot {
        scope: ObservationScope::Tablet(TabletId::from_u64(1)),
        first_window: ObservationWindowId::from_u64(window),
        latest_window: ObservationWindowId::from_u64(window),
        latest_sequence: sequence,
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
            p50_ns: 1_000,
            p99_ns: 10_000,
            p999_ns: 20_000,
            max_ns: 20_000,
        }),
        queueing: None,
        compute: None,
        memory: Some(MemorySummary {
            resident_bytes: 2_000_000,
            logical_bytes: 2_000_000,
            residency_ratio: Some(1.0),
        }),
        memory_pressure: Some(MemoryPressureSummary {
            mean_pressure: UnitInterval::new(0.95).expect("bounded pressure"),
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
            mean_score: CriticalityScore::from_u64(0),
            peak_score: CriticalityScore::from_u64(0),
            observation_count: 1,
            access_count: 1,
        }),
    };
    ObservationSnapshot {
        as_of: observed,
        last_sequence: Some(sequence),
        accepted_samples: window,
        scopes: vec![scope],
    }
}

fn frame(window: u64) -> ControllerFrame {
    ControllerFrame::new(
        snapshot(window),
        ActionScope::new(ClusterId::from_u128(1), NamespaceId::from_u64(1)),
        ObservationSequence::from_u64(window),
        ObservationWindowId::from_u64(window),
        1,
        Ticks::from_micros(window),
    )
}

fn context() -> SafetyContext {
    SafetyContext::new(
        ActionScope::new(ClusterId::from_u128(1), NamespaceId::from_u64(1)),
        TabletId::from_u64(1),
    )
}

struct NoopActuator;

impl Actuator for NoopActuator {
    fn execute(
        &mut self,
        action_id: kivi_control::ActionId,
        action: &Action,
    ) -> Result<ActuatorReceipt, ActuationError> {
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

fn run_controller(learned: bool) -> usize {
    let mut controller = AdaptiveController::new(
        AdaptiveConfig {
            max_traces: 32,
            ..AdaptiveConfig::default()
        },
        BaselineController::new(BaselineConfig {
            sustained_windows: 1,
            min_samples: 1,
            ..BaselineConfig::default()
        }),
        learned.then(LearnedController::default),
    );
    let mut actuator = NoopActuator;
    for window in 1..=WINDOWS {
        let current = frame(window);
        let decision = controller.decide(&current, &context());
        for record in decision.active_actions {
            let _ = controller.execute(&mut actuator, record.id, current.action_observation());
        }
    }
    controller.ledger().records().count()
}

#[divan::bench]
fn baseline_decisions(bencher: Bencher) {
    bencher.bench_local(|| black_box(run_controller(false)));
}

#[divan::bench]
fn learned_shadow_decisions(bencher: Bencher) {
    bencher.bench_local(|| black_box(run_controller(true)));
}

fn main() {
    divan::main();
}
