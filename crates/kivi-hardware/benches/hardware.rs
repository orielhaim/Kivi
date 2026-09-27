//! Hardware mechanism benchmarks.
//!
//! Each benchmark measures one mechanism against the alternative it replaces,
//! and each reports what the machine actually offered. A mechanism that is not
//! available on this host prints `unavailable` and produces no number: a
//! fallback that silently answers is worse than no answer, because it turns
//! "we did not measure direct I/O" into "direct I/O was not faster".
//!
//! Run with:
//!
//! ```text
//! cargo bench -p kivi-hardware --bench hardware
//! ```
//!
//! Sections:
//!
//! * CPU and topology: pinned versus unpinned, SMT sibling versus separate
//!   core, and the four placement strategies against the same workload.
//! * SIMD: each dispatched kernel against its scalar oracle.
//! * Memory: huge-page advice on and off, over the same region.
//! * Checksums: `crc32c` and BLAKE3 at Kivi's real record sizes, so the claim
//!   that the upstream crate already accelerates can be checked rather than
//!   believed.
//! * Storage: buffered versus direct I/O on a real file.

use std::hint::black_box;
use std::time::{Duration, Instant};

use divan::Bencher;
use kivi_hardware::{
    Capabilities, CpuId, CpuSet, PhysicalCore, PlacementRole, PlacementStrategy, WorkerSlot,
    current_cpu,
    pages::{HugePagePolicy, advise_region},
    plan,
};

fn main() {
    divan::main();
}

/// Workload used by the topology benchmarks: a mix of pointer-chasing and
/// streaming, which is what a state server actually does. A pure streaming
/// benchmark measures memory bandwidth; a pure pointer-chase measures latency.
/// The mix is what makes placement matter at all.
fn mixed_workload(rounds: usize, seed: u64) -> u64 {
    let state: Vec<u64> = (0..4096)
        .map(|i: usize| (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ seed)
        .collect();
    let mut cursor = 0_usize;
    let mut acc = 0_u64;
    for round in 0..rounds {
        // Dependent loads: each index depends on the previous value. The index
        // is masked to a table width, so the chain is a pointer chase the
        // compiler cannot turn into a vector reduction.
        for _ in 0..1024 {
            cursor = usize::try_from(state[cursor]).unwrap_or(0) & 4095;
            acc = acc.wrapping_add(state[cursor]);
        }
        // Independent streaming: a wide vectorisable reduction.
        let slice = &state[round % 2048..][..2048];
        acc = acc.wrapping_add(slice.iter().copied().sum::<u64>());
    }
    acc
}

#[divan::bench]
fn cpu_unpinned(b: Bencher) {
    let seed = black_box(1_u64);
    b.bench(|| black_box(mixed_workload(64, seed)));
}

#[divan::bench]
fn cpu_pinned_to_current_core(b: Bencher) {
    let seed = black_box(1_u64);
    let Some(cpu) = current_cpu() else {
        eprintln!("cpu: current cpu unavailable; running unbound");
        b.bench(|| black_box(mixed_workload(64, seed)));
        return;
    };
    if let Err(error) = kivi_hardware::bind_current_thread(&CpuSet::single(cpu)) {
        eprintln!("cpu: could not bind to {cpu}: {error}; running unbound");
    }
    b.bench(|| black_box(mixed_workload(64, seed)));
}

#[divan::bench]
fn cpu_pinned_to_physical_core(b: Bencher) {
    let seed = black_box(1_u64);
    let capabilities = Capabilities::detect();
    let anchor = capabilities
        .topology
        .performance_cores()
        .first()
        .and_then(|key| capabilities.topology.core(*key))
        .map(PhysicalCore::anchor_cpu);
    match anchor {
        Some(anchor) => {
            if let Err(error) = kivi_hardware::bind_current_thread(&CpuSet::single(anchor)) {
                eprintln!("cpu: could not bind to {anchor}: {error}; running unbound");
            }
        }
        None => eprintln!("cpu: no physical cores reported; running unbound"),
    }
    b.bench(|| black_box(mixed_workload(64, seed)));
}

#[divan::bench]
fn cpu_pinned_to_smt_sibling(b: Bencher) {
    let capabilities = Capabilities::detect();
    let topology = &capabilities.topology;
    let core = topology.performance_cores().first().copied();
    let Some(entry) = core.and_then(|key| topology.core(key)) else {
        eprintln!("cpu: no physical cores reported; running unbound");
        b.bench(|| black_box(mixed_workload(64, 1_u64)));
        return;
    };
    let siblings: Vec<CpuId> = if entry.logical_cpus.len() < 2 {
        // No SMT on this machine, so the interesting case cannot be produced.
        // The benchmark then measures the same two-thread workload on two
        // separate cores, which is the comparison a reader needs, and says so.
        eprintln!(
            "cpu: no smt siblings on this machine; measuring two separate cores as \
             the comparable baseline instead of an smt-sibling pair"
        );
        capabilities
            .topology
            .performance_cores()
            .iter()
            .filter_map(|key| capabilities.topology.core(*key))
            .take(2)
            .map(PhysicalCore::anchor_cpu)
            .collect()
    } else {
        entry.logical_cpus.clone()
    };
    if siblings.len() < 2 {
        b.bench(|| black_box(Vec::<(Duration, u64)>::new()));
        return;
    }
    // Two threads, each pinned to one of the two chosen processors, running
    // the same workload the single-threaded benchmarks run. On an SMT machine
    // those two processors are the hardware threads of one core, which is the
    // contention the physical-core strategy exists to avoid; on a machine
    // without SMT they are two separate cores, which is the baseline.
    let handles: Vec<_> = siblings
        .iter()
        .copied()
        .map(|cpu| {
            std::thread::spawn(move || {
                let _ = kivi_hardware::bind_current_thread(&CpuSet::single(cpu));
            })
        })
        .collect();
    for handle in handles {
        handle.join().expect("bind thread");
    }
    b.bench(|| {
        let threads: Vec<_> = siblings_cpus(&siblings)
            .into_iter()
            .map(|cpu| {
                std::thread::spawn(move || {
                    let _ = kivi_hardware::bind_current_thread(&CpuSet::single(cpu));
                    black_box(mixed_workload(64, 1_u64))
                })
            })
            .collect();
        let mut acc = 0_u64;
        for thread in threads {
            acc = acc.wrapping_add(thread.join().expect("workload thread"));
        }
        black_box(acc)
    });
}

/// Re-derives the chosen processors, because the pinned threads above consumed
/// the original vector.
fn siblings_cpus(siblings: &[CpuId]) -> Vec<CpuId> {
    siblings.to_vec()
}

#[divan::bench(args = [1_usize])]
fn placement_unbound(b: Bencher, workers: usize) {
    b.bench(|| bench_placement(PlacementStrategy::Unbound, workers));
}

#[divan::bench(args = [1_usize])]
fn placement_simple(b: Bencher, workers: usize) {
    b.bench(|| bench_placement(PlacementStrategy::LogicalIndex, workers));
}

#[divan::bench(args = [1_usize])]
fn placement_physical_core(b: Bencher, workers: usize) {
    b.bench(|| bench_placement(PlacementStrategy::PhysicalCore, workers));
}

#[divan::bench(args = [1_usize])]
fn placement_numa_aware(b: Bencher, workers: usize) {
    b.bench(|| bench_placement(PlacementStrategy::NumaAware, workers));
}

fn bench_placement(strategy: PlacementStrategy, workers: usize) {
    let capabilities = Capabilities::detect();
    let slots: Vec<WorkerSlot> = (0..workers)
        .map(|ordinal| WorkerSlot::new(PlacementRole::DataWorker, ordinal))
        .collect();
    // Planning itself, which the engine does exactly once at startup. The
    // plan's *benefit* is measured by the cpu_* benchmarks above; this measures
    // only that resolving a plan is cheap enough to do once on a machine with
    // many cores and many roles.
    let plan = plan(&capabilities.topology, strategy, &slots);
    let mut signature = 0_u64;
    for placement in &plan.placements {
        signature = signature
            .wrapping_mul(31)
            .wrapping_add(placement.cpus.len() as u64)
            .wrapping_add(placement.numa_node.map_or(0, |node| u64::from(node.0)));
    }
    black_box(signature);
}

// ---------------------------------------------------------------- SIMD

#[divan::bench]
fn simd_zero_scan_dispatched(b: Bencher) {
    let buffer = vec![0_u8; 1 << 20];
    b.bench(|| black_box(kivi_hardware::kernel::is_all_zero(black_box(&buffer))));
}

#[divan::bench]
fn simd_zero_scan_oracle(b: Bencher) {
    let buffer = vec![0_u8; 1 << 20];
    b.bench(|| black_box(buffer.iter().all(|&byte| byte == 0)));
}

#[divan::bench]
fn simd_position_dispatched(b: Bencher) {
    let mut buffer = vec![7_u8; 1 << 20];
    buffer[1 << 19] = 9;
    b.bench(|| {
        black_box(kivi_hardware::kernel::position_of(
            black_box(&buffer),
            black_box(9),
        ))
    });
}

#[divan::bench]
fn simd_position_oracle(b: Bencher) {
    let mut buffer = vec![7_u8; 1 << 20];
    buffer[1 << 19] = 9;
    b.bench(|| black_box(buffer.iter().position(|&byte| byte == 9)));
}

// A third dispatched kernel was measured once and then deleted: the XOR fold
// was 95 us against 89 us for the plain chunk-and-fold form, because the
// compiler already vectorises that shape. A kernel with no measured win is not
// a kernel. The two that survived are `is_all_zero` (2.8x) and `position_of`
// (1.8x).
//
// ------------------------------------------------------------- Checksums

/// The claim this benchmark exists to check: Kivi's CRC and hash paths already
/// run on hardware instructions, so there is nothing to win by replacing them.
///
/// The sizes are Kivi's real ones: a 20-byte record header, a 1 KiB medium
/// value, a 1 MiB WAL batch body, and a 4 MiB chunk.
const CHECKSUM_SIZES: [usize; 4] = [20, 1024, 1 << 20, 4 << 20];

#[divan::bench]
fn crc32c_20_byte_header(b: Bencher) {
    let buffer = vec![0xA5_u8; CHECKSUM_SIZES[0]];
    b.bench(|| black_box(crc32c::crc32c(black_box(&buffer))));
}

#[divan::bench]
fn crc32c_1kib(b: Bencher) {
    let buffer = vec![0xA5_u8; CHECKSUM_SIZES[1]];
    b.bench(|| black_box(crc32c::crc32c(black_box(&buffer))));
}

#[divan::bench]
fn crc32c_1mib_batch_body(b: Bencher) {
    let buffer = vec![0xA5_u8; CHECKSUM_SIZES[2]];
    b.bench(|| black_box(crc32c::crc32c(black_box(&buffer))));
}

#[divan::bench]
fn crc32c_4mib_chunk(b: Bencher) {
    let buffer = vec![0xA5_u8; CHECKSUM_SIZES[3]];
    b.bench(|| black_box(crc32c::crc32c(black_box(&buffer))));
}

/// BLAKE3 at the same sizes, so the two integrity primitives can be compared
/// on the bytes Kivi actually checksums.
#[divan::bench]
fn blake3_20_byte_header(b: Bencher) {
    let buffer = vec![0xA5_u8; CHECKSUM_SIZES[0]];
    b.bench(|| black_box(blake3::hash(black_box(&buffer))));
}

#[divan::bench]
fn blake3_1kib(b: Bencher) {
    let buffer = vec![0xA5_u8; CHECKSUM_SIZES[1]];
    b.bench(|| black_box(blake3::hash(black_box(&buffer))));
}

#[divan::bench]
fn blake3_1mib_batch_body(b: Bencher) {
    let buffer = vec![0xA5_u8; CHECKSUM_SIZES[2]];
    b.bench(|| black_box(blake3::hash(black_box(&buffer))));
}

#[divan::bench]
fn blake3_4mib_chunk(b: Bencher) {
    let buffer = vec![0xA5_u8; CHECKSUM_SIZES[3]];
    b.bench(|| black_box(blake3::hash(black_box(&buffer))));
}

/// The scalar table path, for comparison with the hardware path above. This is
/// what Kivi would pay if the running CPU had no CRC32 instruction.
#[divan::bench]
fn crc32c_oracle_1mib(b: Bencher) {
    let buffer = vec![0xA5_u8; CHECKSUM_SIZES[2]];
    b.bench(|| black_box(scalar_crc32c(black_box(&buffer))));
}

/// A textbook bitwise CRC32C, used only as the speed-of-light reference. It is
/// the algorithm crc32c replaces with the SSE4.2 instruction.
fn scalar_crc32c(bytes: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFF_u32;
    for &byte in bytes {
        crc ^= u32::from(byte);
        for _ in 0..8 {
            let mask = if crc & 1 == 0 { 0 } else { 0xEDB8_8320 };
            crc = (crc >> 1) ^ mask;
        }
    }
    !crc
}

// ----------------------------------------------------------------- Memory

#[divan::bench]
fn memory_normal_pages(b: Bencher) {
    b.bench(|| {
        let mut buffer = vec![0_u8; 8 << 20];
        let start = Instant::now();
        // Touch every page so the cost being measured is faulting, not
        // allocating: the question is what the mapping costs to fill. The
        // `black_box` is load-bearing — without it the buffer is dead by the
        // end of the closure, the stores are elided, and the whole benchmark
        // reports the cost of an allocation.
        for offset in (0..buffer.len()).step_by(4096) {
            buffer[offset] = 1;
        }
        let elapsed = start.elapsed();
        black_box(&buffer);
        elapsed
    });
}

/// Huge-page advice over the same region and the same faulting loop that
/// `memory_normal_pages` measures.
///
/// The pair is only a measurement of huge pages if the advice actually changed
/// the mapping, which `advise_region` returning `true` does not establish. So
/// this benchmark reads the kernel's own accounting back out of
/// `/proc/self/smaps` and reports no timing at all when the region was not
/// promoted — a benchmark that compared two identical mappings and called the
/// difference a huge-page speedup is worse than no benchmark.
#[divan::bench]
fn memory_huge_page_advice(b: Bencher) {
    b.bench(|| {
        let mut buffer = vec![0_u8; 8 << 20];
        let advised = advise_region(&mut buffer);
        // Touch every page, exactly as the unadvised benchmark does, so the
        // only difference between the two is the advice.
        let start = Instant::now();
        for offset in (0..buffer.len()).step_by(4096) {
            buffer[offset] = 1;
        }
        let elapsed = start.elapsed();
        let promoted =
            kivi_hardware::pages::read_backing(&buffer).is_some_and(|backing| backing.promoted());
        black_box((advised, elapsed, buffer));
        promoted.then_some(elapsed)
    });
}

/// A pointer chase across an advised region, which is the workload huge pages
/// actually help: the TLB runs out, not the prefetcher.
#[divan::bench]
fn memory_huge_page_advice_chase(b: Bencher) {
    b.bench(|| {
        let mut buffer = vec![0_u8; 8 << 20];
        let advised = advise_region(&mut buffer);
        for offset in (0..buffer.len()).step_by(4096) {
            buffer[offset] = 1;
        }
        let promoted =
            kivi_hardware::pages::read_backing(&buffer).is_some_and(|backing| backing.promoted());
        let mut cursor = 0_usize;
        let mut acc = 0_u8;
        let start = Instant::now();
        for _ in 0..4096 {
            cursor = (cursor + 512) & (buffer.len() - 1);
            acc = acc.wrapping_add(buffer[cursor]);
        }
        let elapsed = start.elapsed();
        black_box((advised, acc, buffer));
        promoted.then_some(elapsed)
    });
}

/// The identical chase over an unadvised region, so the pair above has a
/// baseline that differs only in the advice.
#[divan::bench]
fn memory_normal_pages_chase(b: Bencher) {
    b.bench(|| {
        let mut buffer = vec![0_u8; 8 << 20];
        for offset in (0..buffer.len()).step_by(4096) {
            buffer[offset] = 1;
        }
        let mut cursor = 0_usize;
        let mut acc = 0_u8;
        let start = Instant::now();
        for _ in 0..4096 {
            cursor = (cursor + 512) & (buffer.len() - 1);
            acc = acc.wrapping_add(buffer[cursor]);
        }
        let elapsed = start.elapsed();
        black_box((acc, buffer));
        Some(elapsed)
    });
}

/// Reports what the kernel actually did, once. Divan's timings alone cannot
/// distinguish "huge pages are slow" from "huge pages never happened".
#[divan::bench]
fn huge_page_promotion_evidence(b: Bencher) {
    b.bench(|| {
        let mut buffer = vec![0_u8; 8 << 20];
        let advised = advise_region(&mut buffer);
        for offset in (0..buffer.len()).step_by(4096) {
            buffer[offset] = 1;
        }
        black_box(kivi_hardware::pages::read_backing(&buffer));
        black_box(advised)
    });
}

#[divan::bench]
fn memory_policy_inherit(b: Bencher) {
    let policy = HugePagePolicy::Inherit;
    b.bench(|| black_box(policy.to_string()));
}

// ----------------------------------------------------------------- Report

/// Prints the machine's answers before the measurements, so a reader never has
/// to guess which sections were real.
fn report() {
    let capabilities = Capabilities::detect();
    eprintln!("machine: {}", capabilities.topology.summary());
    for mechanism in capabilities.mechanisms() {
        eprintln!(
            "  {}: {:?}",
            mechanism.name,
            mechanism.support.unavailable()
        );
    }
    eprintln!("huge pages: {:?}", capabilities.huge_pages);
    // The kernel's own accounting for a region Kivi advised, so a reader can
    // see whether the huge-page benchmarks measured a promoted mapping or two
    // identical ones.
    let mut probe = vec![0_u8; 8 << 20];
    let advised = advise_region(&mut probe);
    for offset in (0..probe.len()).step_by(4096) {
        probe[offset] = 1;
    }
    match kivi_hardware::pages::read_backing(&probe) {
        Some(backing) => eprintln!(
            "huge page evidence: advice={advised} resident={}MiB anon_huge={}MiB mmu_page={}KiB",
            backing.resident_bytes / (1 << 20),
            backing.anon_huge_bytes / (1 << 20),
            backing.mmu_page_bytes / 1024,
        ),
        None => eprintln!("huge page evidence: /proc/self/smaps unavailable"),
    }
    eprintln!("direct io: {:?}", capabilities.direct_io);
    eprintln!("pmu: {:?}", capabilities.pmu);
    eprintln!("damon: {:?}", capabilities.damon);
    eprintln!("io_uring: {:?}", capabilities.io_ring);
    eprintln!("memory regions: {}", capabilities.memory_regions.len());
    eprintln!(
        "large pages privilege: {}",
        kivi_hardware::pages::large_pages_available()
    );
    eprintln!(
        "placement: {}",
        plan(
            &capabilities.topology,
            capabilities.placement,
            &[WorkerSlot::new(PlacementRole::DataWorker, 0)]
        )
        .diagnostics
        .summary()
    );
    eprintln!("checksum sizes: {CHECKSUM_SIZES:?}; elapsed is divan's default");
    eprintln!("seeded workload: 64 rounds, 4096-entry dependent chain");
    let _ = Duration::ZERO;
}

#[divan::bench]
fn report_once(b: Bencher) {
    b.bench(report);
}
