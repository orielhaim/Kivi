//! Prints what this machine actually offers, from the same code paths Kivi
//! runs. Its purpose is to be *executed* on a target platform, not to be a
//! documented example.
//!
//! Every line is produced by a real probe: a counter is opened or it is not, a
//! ring is created or it is not, a sysfs file is read or it is not.

use std::time::{Duration, Instant};

fn main() {
    let capabilities = kivi_hardware::snapshot();

    println!("== snapshot ==");
    println!("{}", capabilities.report());

    println!();
    println!("== topology ==");
    let topology = capabilities.topology_logical();
    println!("{}", topology.summary());
    for core in &capabilities.topology_logical().physical_cores {
        println!(
            "  core {:?} node={} cpus={:?} class={:?} llc_shared={}",
            core.key, core.numa_node.0, core.logical_cpus, core.class, core.llc_shared
        );
    }
    for device in &topology.devices {
        println!("  device {device:?}");
    }

    println!();
    println!("== live probes ==");
    println!("  pmu support           : {}", capabilities.pmu);
    println!("  pmu counters attached : {}", pmu_probe());
    println!("  pmu per-event probe:");
    for (event, outcome) in kivi_hardware::telemetry::pmu::diagnose() {
        println!("    {event:24} {outcome}");
    }
    println!(
        "  io_uring              : {}",
        kivi_hardware::telemetry::io_uring_support()
    );
    println!(
        "  damon                 : {}",
        kivi_hardware::telemetry::damon_support()
    );
    println!("  direct io             : {}", kivi_hardware::io::detect());
    println!(
        "  huge pages            : {}",
        kivi_hardware::pages::detect()
    );
    println!(
        "  current cpu           : {:?}",
        kivi_hardware::current_cpu()
    );
    println!(
        "  allowed cpus          : {}",
        kivi_hardware::CpuSet::current()
            .map_or_else(|| "unknown".to_string(), |set| set.to_string())
    );

    println!();
    println!("== numa ==");
    println!("{}", kivi_hardware::numa::report());

    println!();
    println!("== placement ==");
    let slots: Vec<kivi_hardware::WorkerSlot> = kivi_hardware::PlacementRole::ALL
        .iter()
        .flat_map(|role| {
            let count = match role {
                kivi_hardware::PlacementRole::DataWorker => 4,
                _ => 1,
            };
            (0..count).map(move |ordinal| kivi_hardware::WorkerSlot::new(*role, ordinal))
        })
        .collect();
    for strategy in [
        kivi_hardware::PlacementStrategy::Unbound,
        kivi_hardware::PlacementStrategy::LogicalIndex,
        kivi_hardware::PlacementStrategy::PhysicalCore,
        kivi_hardware::PlacementStrategy::NumaAware,
    ] {
        let plan = kivi_hardware::plan(capabilities.topology_logical(), strategy, &slots);
        println!("  {}", plan.diagnostics.summary());
        for placement in &plan.placements {
            println!("    {placement}");
        }
    }
}

/// Attaches counters, burns cycles, and prints what the machine reported.
///
/// The point is to prove the group was scheduled and produced non-zero counts:
/// a probe that prints "available" without printing a cycle count has not
/// demonstrated that anything was measured.
fn pmu_probe() -> String {
    let mut counters = kivi_hardware::telemetry::pmu::PmuCounters::attach();
    if !counters.is_attached() {
        return format!("unattached ({})", counters.support());
    }
    let first = counters.sample();
    let start = Instant::now();
    let mut acc = 0_u64;
    while start.elapsed() < Duration::from_millis(200) {
        acc = acc.wrapping_add((0..20_000).map(|i| i * i).sum::<u64>() & 0xFFFF);
    }
    std::hint::black_box(acc);
    let second = counters.sample();
    let delta = second.since(&first);
    format!(
        "counters=[{}] source={} comparable={} delta={delta:?}",
        counters.support(),
        counters.cycle_source(),
        second.comparable(),
    )
}
