//! Consistency-layer read-path benchmarks: planning cost decomposition
//! per contract and path (no I/O, no Raft — the hub plan is the
//! deterministic core; barrier/wait execution costs ride the network and
//! are measured by the cluster benches).
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
//! Almost-Local hit ............ receipt usability + age check
//! RosterLease hit ............. lease authorization + floor check
//! evidence install ............ barrier receipt insert + cache fill
//! ```

use std::time::Duration;

use divan::{Bencher, black_box};

use kivi_consensus::ConsistencyHub;
use kivi_state::{Key, Operation, OperationResult};
use kivi_types::{
    CommitPosition, CommitToken, FreshnessReceipt, NodeId, NodeIncarnation, ReadAuthorityProvider,
    ReadContext, ReadContract, ReplicaFreshness, RosterLease, TabletAuthority, TabletEpoch,
    TabletId, Ticks, UnixMicros, WriteGuardGeneration,
};

const TABLET: TabletId = TabletId::from_u64(3);

fn authority() -> TabletAuthority {
    TabletAuthority::new(
        TABLET,
        TabletEpoch::from_u64(1),
        WriteGuardGeneration::from_u64(1),
    )
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
        UnixMicros::from_micros(1_000_000),
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
        ReadAuthorityProvider::AlmostLocal {
            max_proof_age: Duration::from_millis(100),
        },
    );
    hub.note_barrier(TABLET, receipt());
    (hub, op())
}

fn roster() -> (ConsistencyHub, Operation) {
    let hub = ConsistencyHub::new(
        NodeId::from_u64(1),
        NodeIncarnation::from_u64(7),
        ReadAuthorityProvider::RosterLease,
    );
    hub.grant_lease(
        TABLET,
        RosterLease::new(
            authority(),
            1,
            vec![NodeId::from_u64(1)],
            CommitPosition::from_u64(10),
            Ticks::from_micros(1_000_000),
            NodeIncarnation::from_u64(7),
        ),
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
        ))
    });
}

#[divan::bench]
fn plan_almost_local_hit(bencher: Bencher) {
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
        ))
    });
}

#[divan::bench]
fn plan_roster_hit(bencher: Bencher) {
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
        black_box(hub.note_barrier(black_box(TABLET), black_box(receipt)));
        black_box(hub.fill(
            black_box(TABLET),
            black_box(authority()),
            black_box(&op),
            black_box(&outcome),
            black_box(fresh.applied),
        ));
    });
}
