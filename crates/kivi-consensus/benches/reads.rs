//! Consistency-layer read-path benchmarks: planning cost decomposition
//! per contract and path (no I/O, no Raft — the hub plan is the
//! deterministic core; barrier/fence/coverage execution costs ride the
//! network and are measured by the cluster benches).
//!
//! Run in release mode for meaningful figures:
//! `cargo bench -p kivi-consensus --bench reads`.
//!
//! What each bench pins (planning only):
//!
//! ```text
//! conservative Latest ......... barrier decision, no evidence consulted
//! AtLeast covered ............. lineage check + cache probe, local serve
//! AtLeast behind .............. lineage check, wait-index derivation
//! BoundedStale proof .......... receipt usability + age check
//! BoundedStale escalation ..... failed proof lookup, fallback decision
//! Any ......................... cache probe, local serve
//! strong-cache hit ............ cache probe + outcome clone
//! ALR plan .................... provider dispatch to the batch path
//! roster-evidence hit ......... stable-roster + floor-coverage proof
//! lease drive ................. reconcile + tick + renew cadence work
//! evidence install ............ barrier receipt insert + cache fill
//! ```

use std::time::Duration;

use divan::{Bencher, black_box};

use kivi_consensus::ConsistencyHub;
use kivi_state::{Key, Operation, OperationResult};
use kivi_types::{
    CommitPosition, CommitToken, FreshnessReceipt, LeaseEligibility, LeaseParams, NodeId,
    NodeIncarnation, ReadAuthorityProvider, ReadContext, ReadContract, ReplicaFreshness,
    RosterTerm, TabletAuthority, TabletEpoch, TabletId, Ticks, WallTimestamp, WriteGuardGeneration,
};

const TABLET: TabletId = TabletId::from_u64(3);
const ELIGIBLE: LeaseEligibility = LeaseEligibility::Eligible;

fn authority() -> TabletAuthority {
    TabletAuthority::new(
        TABLET,
        TabletEpoch::from_u64(1),
        WriteGuardGeneration::from_u64(1),
    )
}

fn lease_params() -> LeaseParams {
    LeaseParams::default()
}

fn fresh() -> ReplicaFreshness {
    ReplicaFreshness::new(
        authority(),
        CommitPosition::from_u64(10),
        NodeIncarnation::from_u64(7),
    )
}

fn ctx() -> ReadContext {
    ReadContext::new(
        WallTimestamp::from_micros(1_000_000),
        Ticks::from_micros(50_000),
        Duration::from_secs(1),
    )
}

fn op() -> Operation {
    Operation::Get {
        key: Key::from(b"bench-key".to_vec()),
    }
}

fn token(position: u64) -> CommitToken {
    CommitToken::new(
        TABLET,
        TabletEpoch::from_u64(1),
        CommitPosition::from_u64(position),
    )
}

fn receipt() -> FreshnessReceipt {
    FreshnessReceipt::new(
        authority(),
        CommitPosition::from_u64(10),
        Ticks::from_micros(0),
        NodeIncarnation::from_u64(7),
    )
}

fn outcome() -> OperationResult {
    OperationResult::Exists(true)
}

fn conservative() -> (ConsistencyHub, Operation) {
    let hub = ConsistencyHub::new(
        NodeId::from_u64(1),
        NodeIncarnation::from_u64(7),
        ReadAuthorityProvider::ConservativeLeader,
        lease_params(),
    );
    (hub, op())
}

fn proven() -> (ConsistencyHub, Operation) {
    let (hub, op) = conservative();
    hub.note_barrier(TABLET, receipt());
    (hub, op)
}

fn cached() -> (ConsistencyHub, Operation) {
    let (hub, op) = conservative();
    hub.insert(
        TABLET,
        authority(),
        &op,
        &outcome(),
        CommitPosition::from_u64(10),
        None,
    );
    (hub, op)
}

fn almost_local() -> (ConsistencyHub, Operation) {
    let hub = ConsistencyHub::new(
        NodeId::from_u64(1),
        NodeIncarnation::from_u64(7),
        ReadAuthorityProvider::AlmostLocal,
        lease_params(),
    );
    hub.note_barrier(TABLET, receipt());
    (hub, op())
}

fn roster() -> (ConsistencyHub, Operation) {
    let hub = ConsistencyHub::new(
        NodeId::from_u64(1),
        NodeIncarnation::from_u64(7),
        ReadAuthorityProvider::RosterLease,
        lease_params(),
    );
    // Single-voter loopback stabilizes inside one drive past quarantine.
    let quarantine = lease_params().quarantine();
    let awakened = u64::try_from(quarantine.as_micros())
        .unwrap_or(u64::MAX)
        .saturating_add(1);
    let _ = hub.lease_drive(
        TABLET,
        authority(),
        RosterTerm::from_u64(5),
        vec![NodeId::from_u64(1)],
        true,
        CommitPosition::from_u64(10),
        CommitPosition::from_u64(10),
        Ticks::from_micros(0),
    );
    let _ = hub.lease_drive(
        TABLET,
        authority(),
        RosterTerm::from_u64(5),
        vec![NodeId::from_u64(1)],
        true,
        CommitPosition::from_u64(10),
        CommitPosition::from_u64(10),
        Ticks::from_micros(awakened),
    );
    (hub, op())
}

fn main() {
    divan::main();
}

#[divan::bench]
fn plan_latest_conservative(bencher: Bencher) {
    let (hub, op) = conservative();
    let fresh = fresh();
    let ctx = ctx();
    bencher.bench(|| {
        black_box(hub.plan(
            black_box(TABLET),
            black_box(&op),
            black_box(ReadContract::Latest),
            black_box(fresh),
            black_box(ctx),
            black_box(ELIGIBLE),
        ))
    });
}

#[divan::bench]
fn plan_at_least_covered(bencher: Bencher) {
    let (hub, op) = conservative();
    let fresh = fresh();
    let ctx = ctx();
    let contract = ReadContract::AtLeast(token(9));
    bencher.bench(|| {
        black_box(hub.plan(
            black_box(TABLET),
            black_box(&op),
            black_box(contract),
            black_box(fresh),
            black_box(ctx),
            black_box(ELIGIBLE),
        ))
    });
}

#[divan::bench]
fn plan_at_least_wait(bencher: Bencher) {
    let (hub, op) = conservative();
    let fresh = fresh();
    let ctx = ctx();
    let contract = ReadContract::AtLeast(token(99));
    bencher.bench(|| {
        black_box(hub.plan(
            black_box(TABLET),
            black_box(&op),
            black_box(contract),
            black_box(fresh),
            black_box(ctx),
            black_box(ELIGIBLE),
        ))
    });
}

#[divan::bench]
fn plan_bounded_stale_proof(bencher: Bencher) {
    let (hub, op) = proven();
    let fresh = fresh();
    let ctx = ctx();
    let contract = ReadContract::BoundedStale {
        max_staleness: Duration::from_millis(100),
    };
    bencher.bench(|| {
        black_box(hub.plan(
            black_box(TABLET),
            black_box(&op),
            black_box(contract),
            black_box(fresh),
            black_box(ctx),
            black_box(ELIGIBLE),
        ))
    });
}

#[divan::bench]
fn plan_bounded_stale_escalation(bencher: Bencher) {
    let (hub, op) = conservative();
    let fresh = fresh();
    let ctx = ctx();
    let contract = ReadContract::BoundedStale {
        max_staleness: Duration::from_millis(100),
    };
    bencher.bench(|| {
        black_box(hub.plan(
            black_box(TABLET),
            black_box(&op),
            black_box(contract),
            black_box(fresh),
            black_box(ctx),
            black_box(ELIGIBLE),
        ))
    });
}

#[divan::bench]
fn plan_any(bencher: Bencher) {
    let (hub, op) = conservative();
    let fresh = fresh();
    let ctx = ctx();
    bencher.bench(|| {
        black_box(hub.plan(
            black_box(TABLET),
            black_box(&op),
            black_box(ReadContract::Any),
            black_box(fresh),
            black_box(ctx),
            black_box(ELIGIBLE),
        ))
    });
}

#[divan::bench]
fn plan_cache_hit(bencher: Bencher) {
    let (hub, op) = cached();
    let fresh = fresh();
    let ctx = ctx();
    bencher.bench(|| {
        black_box(hub.plan(
            black_box(TABLET),
            black_box(&op),
            black_box(ReadContract::Any),
            black_box(fresh),
            black_box(ctx),
            black_box(ELIGIBLE),
        ))
    });
}

#[divan::bench]
fn plan_alr_batch(bencher: Bencher) {
    let (hub, op) = almost_local();
    let fresh = fresh();
    let ctx = ctx();
    bencher.bench(|| {
        black_box(hub.plan(
            black_box(TABLET),
            black_box(&op),
            black_box(ReadContract::Latest),
            black_box(fresh),
            black_box(ctx),
            black_box(ELIGIBLE),
        ))
    });
}

#[divan::bench]
fn plan_roster_evidence_hit(bencher: Bencher) {
    let (hub, op) = roster();
    let fresh = fresh();
    let ctx = ctx();
    bencher.bench(|| {
        black_box(hub.plan(
            black_box(TABLET),
            black_box(&op),
            black_box(ReadContract::Latest),
            black_box(fresh),
            black_box(ctx),
            black_box(ELIGIBLE),
        ))
    });
}

#[divan::bench]
fn lease_drive_steady_state(bencher: Bencher) {
    let (hub, _) = roster();
    bencher.bench(|| {
        black_box(hub.lease_drive(
            black_box(TABLET),
            black_box(authority()),
            black_box(RosterTerm::from_u64(5)),
            black_box(vec![NodeId::from_u64(1)]),
            black_box(true),
            black_box(CommitPosition::from_u64(10)),
            black_box(CommitPosition::from_u64(10)),
            black_box(Ticks::from_micros(9_000_000)),
        ))
    });
}

#[divan::bench]
fn evidence_install(bencher: Bencher) {
    let (hub, op) = conservative();
    let fresh = fresh();
    let receipt = receipt();
    let outcome = outcome();
    bencher.bench(|| {
        hub.note_barrier(black_box(TABLET), black_box(receipt));
        hub.fill(
            black_box(TABLET),
            black_box(authority()),
            black_box(&op),
            black_box(&outcome),
            black_box(fresh.applied),
        );
        black_box(());
    });
}
