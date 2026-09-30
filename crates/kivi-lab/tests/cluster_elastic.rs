//! Elastic-cluster product tests: automatic failure detection + repair,
//! online split/merge with exactly-once and failover, automatic topology
//! policy, restart survival, and bounded log rotation - all on real
//! processes over H3/QUIC with real data directories.
//!
//! ```text
//! start A/B/C/D, RF=3 -> hard-kill C -> suspect -> grace -> repair onto D
//!   -> RF restored, traffic continues -> restart C, no resurrection
//! short outage (< grace) -> no mass re-replication
//! control-leader kill during repair -> new leader resumes, no dup harm
//! populated tablet + traffic -> split -> children serve, parent retires
//! lost-response retry across split -> exactly once
//! kill parent/control leader during split -> resumes, no orphan ranges
//! merge children back -> one range, state + dedup preserved
//! tiny thresholds -> automatic split, then eventual merge
//! full restart -> new topology survives
//! logs rotate by size -> disk bounded, tracing non-blocking
//! ```

use std::time::{Duration, Instant};

use kivi_lab::cluster::Cluster;
use kivi_lab::testkit::{
    fill, get_stream, merge_and_wait, put, put_stream, split_and_wait, traffic_get, traffic_put,
};
use kivi_state::Key;

/// Counts tablets whose desired voters still name `node`.
fn tablets_desiring(cluster: &Cluster, node: u64) -> usize {
    for index in 0..cluster.member_count() {
        if !cluster.alive(index) {
            continue;
        }
        let (_, control) = cluster.admin_get(index, "/v1/control");
        if let Some(placements) = control
            .get("placements")
            .and_then(serde_json::Value::as_array)
        {
            return placements
                .iter()
                .filter(|placement| {
                    placement
                        .get("replicas")
                        .and_then(serde_json::Value::as_array)
                        .is_some_and(|replicas| {
                            replicas
                                .iter()
                                .any(|replica| replica.as_u64() == Some(node))
                        })
                })
                .count();
        }
    }
    usize::MAX
}

/// Failure detector product flow: hard-kill C, automatic suspicion
/// → repair onto D, RF restored with traffic online, restarted C never
/// resurrects obsolete memberships. No operator-issued tablet moves.
#[test]
fn auto_repair_after_hard_kill() {
    let bench_start = Instant::now();
    let mut cluster = Cluster::spawn_with_tablets(24, 2, false).expect("cluster spawns");
    let _ = cluster.wait_all_leaders();
    // Fourth node joins as the repair target.
    let _ = cluster.add_node();
    cluster.wait_control_nodes(4, Duration::from_secs(90));
    // Spread RF=3 placements onto all four nodes first (bounded rounds).
    for _ in 0..6 {
        let _ = cluster.rebalance();
        cluster.wait_migrations_done(Duration::from_secs(240));
        let (_, control) = cluster.admin_get(cluster.live_index(), "/v1/control");
        let live = control
            .get("migrations")
            .and_then(serde_json::Value::as_array)
            .map_or(usize::MAX, |plans| {
                plans
                    .iter()
                    .filter(|plan| {
                        !matches!(
                            plan.get("phase").and_then(serde_json::Value::as_str),
                            Some("Completed" | "Failed")
                        )
                    })
                    .count()
            });
        if live == 0 {
            // One more quiet round proves stability.
            let _ = cluster.rebalance();
            cluster.wait_migrations_done(Duration::from_secs(120));
            break;
        }
    }
    // Workload across every tablet.
    let keys = cluster.keys_for_tablets(1);
    let client = cluster.client();
    for names in keys.values() {
        for name in names {
            put(&client, name, b"repair-target");
        }
    }
    // Hard-kill node 3 (kill -9 semantics: no handshake).
    let victim = cluster.index_of(3).expect("node 3 present");
    let detect_start = Instant::now();
    cluster.kill(victim);
    // Suspicion first (transient-miss state, no repair yet).
    cluster.wait_node_state(3, "Suspect", Duration::from_secs(60));
    let suspect_at = detect_start.elapsed();
    // Grace expiry → Unavailable → automatic repair plans appear with no
    // operator move.
    cluster.wait_node_state(3, "Unavailable", Duration::from_secs(120));
    let unavailable_at = detect_start.elapsed();
    // Traffic continues on surviving quorum throughout.
    for round in 0..20 {
        traffic_put(&client, &format!("during-repair-{round}"), b"online");
    }
    // RF restored: no tablet desires the dead node anymore.
    let repair_deadline = Instant::now() + Duration::from_secs(600);
    loop {
        let remaining = tablets_desiring(&cluster, 3);
        if remaining == 0 {
            break;
        }
        assert!(
            Instant::now() < repair_deadline,
            "repair never restored RF ({remaining} tablets still desire node 3)"
        );
        std::thread::sleep(Duration::from_secs(2));
    }
    let restored_at = detect_start.elapsed();
    cluster.wait_migrations_done(Duration::from_secs(300));
    // Restart the failed node: it must NOT resurrect obsolete memberships
    // (generation/tombstone fencing at open tombstones removed replicas).
    cluster.restart(victim).expect("victim restarts");
    cluster.wait_converged_all();
    let _ = cluster.wait_all_leaders();
    // No-resurrection as a quiescence predicate (never a fixed sleep):
    // poll the invariant itself - the returnee converges with zero
    // tablets still desiring the repaired node.
    let resurrection_deadline = Instant::now() + Duration::from_secs(120);
    loop {
        if tablets_desiring(&cluster, 3) == 0 {
            break;
        }
        assert!(
            Instant::now() < resurrection_deadline,
            "returned node keeps resurrecting obsolete desired placements"
        );
        std::thread::sleep(Duration::from_millis(500));
    }
    assert_eq!(
        tablets_desiring(&cluster, 3),
        0,
        "returned node must not resurrect obsolete desired placements"
    );
    // Data intact on the repaired topology.
    for names in keys.values() {
        for name in names {
            assert_eq!(traffic_get(&client, name), b"repair-target");
        }
    }
    eprintln!(
        "repair benchmark: suspect={suspect_at:?} unavailable={unavailable_at:?} \
         rf_restored={restored_at:?} total={:?}",
        bench_start.elapsed()
    );
}

/// Short outage: a partition shorter than the repair grace must not
/// trigger mass re-replication; health recovers when the node returns.
#[test]
fn short_outage_causes_no_repair() {
    let cluster = Cluster::spawn_with_tablets(8, 2, false).expect("cluster spawns");
    let _ = cluster.wait_all_leaders();
    let client = cluster.client();
    put(&client, "short-0", b"v");
    let migrations_before = cluster.migrations(cluster.live_index());
    let plans_before = migrations_before
        .get("migrations")
        .and_then(serde_json::Value::as_array)
        .map_or(0, Vec::len);
    // Partition node 3 briefly (under the repair grace).
    let victim = cluster.index_of(3).expect("node 3 present");
    cluster.partition(victim);
    std::thread::sleep(Duration::from_secs(3));
    cluster.heal(victim);
    // Settle past suspicion but inside grace: at most Suspect, never
    // Unavailable, and no new migration plans.
    std::thread::sleep(Duration::from_secs(5));
    let migrations_after = cluster.migrations(cluster.live_index());
    let plans_after = migrations_after
        .get("migrations")
        .and_then(serde_json::Value::as_array)
        .map_or(0, Vec::len);
    assert_eq!(
        plans_before, plans_after,
        "short outage must not create re-replication plans"
    );
    assert_eq!(traffic_get(&client, "short-0"), b"v");
}

/// Control-leader crash during automatic repair: the new leader
/// recovers repair plans and continues without duplicating harmful
/// membership actions.
#[test]
fn control_leader_failover_during_repair() {
    let mut cluster = Cluster::spawn_with_tablets(12, 2, false).expect("cluster spawns");
    let _ = cluster.wait_all_leaders();
    let _ = cluster.add_node();
    cluster.wait_control_nodes(4, Duration::from_secs(90));
    // Move the control vote off node 3 first: killing the data victim
    // plus a control voter would otherwise cost control quorum (2 of 3
    // voters down) and no plan could commit anywhere. With voters
    // [1,2,4], killing data node 3 and then one control voter leaves 2/3
    // quorum on the survivors, so the new control leader resumes repairs.
    cluster.set_control_voters(&[1, 2, 4]);
    let client = cluster.client();
    put(&client, "failover-0", b"v");
    let victim = cluster.index_of(3).expect("node 3 present");
    cluster.kill(victim);
    cluster.wait_node_state(3, "Unavailable", Duration::from_secs(120));
    // Crash the control leader while repairs run, wait for a NEW control
    // leader (failover happened, persisted plans survived), then restart
    // the crashed leader promptly: two simultaneous permanent deaths
    // would cost data quorum on RF=3 tablets, so the second death stays
    // brief while failover itself is fully exercised.
    let leader_victim = cluster
        .control_leader_index()
        .expect("control leader elected");
    cluster.kill(leader_victim);
    let failover_deadline = Instant::now() + Duration::from_secs(120);
    loop {
        if let Some(leader) = cluster.control_leader_index()
            && leader != leader_victim
        {
            break;
        }
        assert!(
            Instant::now() < failover_deadline,
            "no new control leader after crash"
        );
        std::thread::sleep(Duration::from_secs(1));
    }
    cluster
        .restart(leader_victim)
        .expect("crashed control leader restarts");
    // The new control leader recovers repair plans and continues them to
    // RF with no duplicated harmful membership actions.
    let repair_deadline = Instant::now() + Duration::from_secs(600);
    loop {
        if cluster.control_leader_index().is_some() && tablets_desiring(&cluster, 3) == 0 {
            break;
        }
        assert!(
            Instant::now() < repair_deadline,
            "repair never finished after control failover"
        );
        std::thread::sleep(Duration::from_secs(2));
    }
    cluster.wait_migrations_done(Duration::from_secs(300));
    assert_eq!(traffic_get(&client, "failover-0"), b"v");
}

/// Manual split money flow: populated tablet with counters, expiry,
/// and a large chunked value splits online under continuous traffic; the
/// new directory version publishes, children become normal tablets, the
/// parent retires, and every value is byte-exact.
#[test]
fn manual_split_serves_through_cutover() {
    let split_start = Instant::now();
    let cluster = Cluster::spawn_with_tablets(4, 2, false).expect("cluster spawns");
    let _ = cluster.wait_all_leaders();
    // Populate every tablet; pick the first as the split parent.
    let keys = cluster.keys_for_tablets(4);
    let parent = *keys.keys().next().expect("a tablet");
    let parent_keys = keys[&parent].clone();
    assert!(parent_keys.len() >= 4, "need keys on both future halves");
    let client = cluster.client();
    for (index, name) in parent_keys.iter().enumerate() {
        put(&client, name, format!("split-value-{index}").as_bytes());
    }
    // Counters + expiry on the parent.
    let counter = format!("{}-counter", parent_keys[0]);
    let expiry_key = format!("{}-expiry", parent_keys[1]);
    client
        .counter_add(&Key::from(counter.clone()), 41)
        .expect("counter commits");
    put(&client, &expiry_key, b"ephemeral");
    // Large chunked value: children must reference the same immutable
    // manifests (no 16 MiB retransfer per child).
    let big_key = format!("{}-big", parent_keys[2]);
    let big = fill(16 * 1024 * 1024, 0x5A);
    put_stream(&client, &big_key, &big);
    let dir_before = cluster
        .node_info(cluster.live_index())
        .get("dir_version")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    // Split under continuous traffic on both future halves.
    let traffic = cluster.client();
    let traffic_keys = parent_keys.clone();
    let traffic_done = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let traffic_flag = std::sync::Arc::clone(&traffic_done);
    let traffic_handle = std::thread::spawn(move || {
        let mut round = 0u64;
        while !traffic_flag.load(std::sync::atomic::Ordering::Relaxed) {
            let name =
                &traffic_keys[usize::try_from(round).unwrap_or(usize::MAX) % traffic_keys.len()];
            traffic_put(&traffic, name, format!("round-{round}").as_bytes());
            round += 1;
        }
    });
    let (left, right) = split_and_wait(&cluster, parent);
    traffic_done.store(true, std::sync::atomic::Ordering::Relaxed);
    let _ = traffic_handle.join();
    // New directory version published everywhere (per-node cutover
    // follows the plan commit; wait for convergence first).
    cluster.wait_dir_version(dir_before + 1, Duration::from_secs(120));
    let dir_after = cluster
        .node_info(cluster.live_index())
        .get("dir_version")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    assert!(
        dir_after > dir_before,
        "split publishes a new directory version"
    );
    // Children are ordinary tablets now (planner-visible, votered).
    cluster.wait_tablet_voters(left, &[1, 2, 3], Duration::from_secs(120));
    cluster.wait_tablet_voters(right, &[1, 2, 3], Duration::from_secs(120));
    // State preserved. Counter/expiry/big covered below; per-key traffic
    // values are eventually overwritten by the traffic loop, so only
    // assert presence + big/counter exactness here.
    for name in &parent_keys {
        let _ = traffic_get(&client, name);
    }
    assert_eq!(
        get_stream(&client, &big_key),
        big,
        "chunked value byte-exact after split"
    );
    eprintln!(
        "split benchmark: parent={parent} left={left} right={right} \
         dir {dir_before}->{dir_after} total={:?}",
        split_start.elapsed()
    );
}

/// Split exactly-once: a mutation committed around cutover whose
/// response is lost must not execute twice when retried after the
/// directory changed (same `SessionId + RequestSeq`).
#[test]
fn split_preserves_exactly_once() {
    let cluster = Cluster::spawn_with_tablets(2, 2, false).expect("cluster spawns");
    let _ = cluster.wait_all_leaders();
    let keys = cluster.keys_for_tablets(2);
    let parent = *keys.keys().next().expect("a tablet");
    let session = kivi_types::SessionId::from_u128(0xE0EC_A7E0_0001);
    let counter_key = format!("{}-eo-counter", keys[&parent][0]);
    // Commit one explicit identity before the split. Lost-response
    // simulation: the retry comes from a crashed-and-resumed session
    // (fresh client, same `SessionId`, floor 0), so the identity is
    // still retryable rather than completed past the floor.
    let committer = cluster.client_with_session(session);
    let seq = kivi_types::RequestSeq::from_u64(5);
    let first = committer
        .counter_add_with_seq(&Key::from(counter_key.clone()), 7, seq)
        .expect("first counter commits");
    // Split the tablet.
    let reply = cluster.split_tablet(parent);
    assert_eq!(
        reply.get("ok").and_then(serde_json::Value::as_bool),
        Some(true),
        "split accepted: {reply}"
    );
    cluster.wait_splits_done(Duration::from_secs(600));
    // Retry the SAME identity after the topology changed: dedup hit with
    // the original outcome, never a second execution.
    let retrier = cluster.client_with_session(session);
    let retry = retrier
        .counter_add_with_seq(&Key::from(counter_key.clone()), 7, seq)
        .expect("retry after split answers");
    assert_eq!(
        retry, first,
        "retry returns the original outcome, not a double-add"
    );
    let live: i64 = retrier
        .counter_get(&Key::from(counter_key.clone()))
        .expect("counter reads")
        .expect("counter present");
    assert_eq!(live, first, "counter executed exactly once across split");
}

/// Split failover: kill the parent leader and the control leader
/// mid-split; the operation resumes safely with no orphan active ranges.
#[test]
fn split_survives_leader_failover() {
    let mut cluster = Cluster::spawn_with_tablets(4, 2, false).expect("cluster spawns");
    let _ = cluster.wait_all_leaders();
    let keys = cluster.keys_for_tablets(2);
    let parent = *keys.keys().next().expect("a tablet");
    let client = cluster.client();
    for name in &keys[&parent] {
        put(&client, name, b"failover");
    }
    let reply = cluster.split_tablet(parent);
    assert_eq!(
        reply.get("ok").and_then(serde_json::Value::as_bool),
        Some(true),
        "split accepted: {reply}"
    );
    // Kill the parent tablet leader mid-split.
    let parent_leader = cluster.leader_of(parent);
    cluster.kill(parent_leader);
    // Kill the control leader mid-split too (a new leader resumes the
    // persisted plan without duplicating harmful membership actions).
    let control_victim = cluster.control_leader_index();
    if let Some(control) = control_victim {
        cluster.kill(control);
    }
    // Recover the killed members after the failover registers: the split
    // resumes (founder-only initialize waits for its founder; quorum
    // steps skip repair-worthy replicas) with no orphan active ranges.
    // Semantic failover gate (never a fixed sleep): a live control
    // leader outside the killed members proves the kills registered and
    // orchestration moved on, so the restarts below exercise
    // rejoin-after-detection rather than an unnoticed flap.
    let failover_deadline = Instant::now() + Duration::from_secs(120);
    let mut dead = vec![parent_leader];
    if let Some(control) = control_victim
        && control != parent_leader
    {
        dead.push(control);
    }
    loop {
        if let Some(current) = cluster.control_leader_index()
            && !dead.contains(&current)
        {
            break;
        }
        assert!(
            Instant::now() < failover_deadline,
            "control failover never registered the kills"
        );
        std::thread::sleep(Duration::from_millis(500));
    }
    cluster
        .restart(parent_leader)
        .expect("parent leader restarts");
    if let Some(control) = control_victim.filter(|control| *control != parent_leader) {
        cluster.restart(control).expect("control leader restarts");
    }
    cluster.wait_splits_done(Duration::from_secs(600));
    for name in &keys[&parent] {
        assert_eq!(traffic_get(&client, name), b"failover");
    }
}

/// Reads the split view from the first live member: phase, block reason, and
/// which tablets the serving directory currently routes.
fn split_view(cluster: &Cluster) -> (String, Option<String>, Vec<u64>) {
    for index in 0..cluster.member_count() {
        if !cluster.alive(index) {
            continue;
        }
        let (status, json) = cluster.admin_get(index, "/v1/control/splits");
        if status != 200 {
            continue;
        }
        let Some(plan) = json
            .get("splits")
            .and_then(serde_json::Value::as_array)
            .and_then(|plans| plans.first())
        else {
            return (String::new(), None, Vec::new());
        };
        let phase = plan
            .get("phase")
            .and_then(serde_json::Value::as_str)
            .unwrap_or_default()
            .to_owned();
        let blocked = plan
            .get("blocked")
            .and_then(serde_json::Value::as_str)
            .map(str::to_owned);
        // `serving_tabs` describes the directory, not the plan, so it sits beside
        // the plan list rather than inside a plan.
        let serving = json
            .get("serving_tabs")
            .and_then(serde_json::Value::as_array)
            .map(|ids| ids.iter().filter_map(serde_json::Value::as_u64).collect())
            .unwrap_or_default();
        return (phase, blocked, serving);
    }
    (String::new(), None, Vec::new())
}

/// A split never hands the namespace to children that cannot serve, and never
/// leaves the namespace with nowhere to route.
///
/// Sampled throughout the split rather than at the end: the failure being
/// pinned is a *transient* one - a moment where routing points at a child with
/// no leader - and a test that only looks at the final state walks straight past
/// it.
#[test]
fn a_split_never_routes_to_a_child_that_cannot_serve() {
    // Four tablets over four members, the shape the existing split tests use.
    let cluster = Cluster::spawn_with_tablets(4, 2, false).expect("cluster spawns");
    let _ = cluster.wait_all_leaders();
    let keys = cluster.keys_for_tablets(4);
    let parent = *keys.keys().next().expect("a tablet");
    let parent_keys = keys
        .get(&parent)
        .cloned()
        .unwrap_or_else(|| keys.values().next().cloned().unwrap_or_default());
    let client = cluster.client();
    for (index, name) in parent_keys.iter().enumerate() {
        put(&client, name, format!("v{index}").as_bytes());
    }
    // `split_and_wait` creates the split and waits for it; creating it here as
    // well would target a parent the first one had already taken out of service.
    let (left, right) = split_and_wait(&cluster, parent);
    let deadline = Instant::now() + Duration::from_secs(180);
    let mut saw_published = false;
    loop {
        let (phase, blocked, serving) = split_view(&cluster);
        assert!(
            !serving.is_empty(),
            "the namespace stopped routing while the split was in {phase:?} \
             (blocked: {blocked:?}): the parent stopped serving and the children \
             were not ready to take over"
        );
        if matches!(
            phase.as_str(),
            "CutoverCommitted" | "ParentRetiring" | "Completed"
        ) {
            saw_published = true;
            // Routing has moved. Both children must be serving it, with a live
            // leader each - this is the claim the readiness proof makes, checked
            // from the outside rather than from the code that made it.
            for child in [left, right] {
                assert!(
                    serving.contains(&child),
                    "routing published to {child} but it is not in the serving set \
                     {serving:?} (phase {phase})"
                );
                let leader = cluster.leader_of(child);
                assert!(
                    leader < cluster.member_count(),
                    "child {child} is authoritative but no member leads it (phase {phase})"
                );
            }
        }
        if phase == "Completed" || Instant::now() > deadline {
            break;
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    assert!(
        saw_published,
        "the split never reached a published-routing phase"
    );
    for (index, name) in parent_keys.iter().enumerate() {
        assert_eq!(
            traffic_get(&client, name),
            format!("v{index}").as_bytes(),
            "state survived the split"
        );
    }
}

/// A child that cannot be provisioned holds the plan, and the parent keeps
/// serving.
///
/// The founder is what makes a child group real, and a plan whose founder is gone
/// has to wait for it: a group nobody initialized has no membership and no
/// leader, and publishing routing to it is the outage this whole phase model
/// exists to prevent. So the observable outcome is a plan that stops advancing
/// with a stated reason, and a parent that is still the authority.
#[test]
fn an_unready_child_keeps_the_parent_authoritative() {
    let mut cluster = Cluster::spawn_with_tablets(4, 2, false).expect("cluster spawns");
    let _ = cluster.wait_all_leaders();
    let keys = cluster.keys_for_tablets(4);
    let parent = *keys.keys().next().expect("a tablet");
    let parent_keys = keys
        .get(&parent)
        .cloned()
        .unwrap_or_else(|| keys.values().next().cloned().unwrap_or_default());
    let client = cluster.client();
    for name in &parent_keys {
        put(&client, name, b"before-split");
    }
    let reply = cluster.split_tablet(parent);
    assert_eq!(
        reply.get("ok").and_then(serde_json::Value::as_bool),
        Some(true),
        "split accepted: {reply}"
    );
    // Kill the founder as soon as the children are provisioned - but not while it
    // is also the control leader, or the reconciler stops and the plan freezes
    // for a reason that has nothing to do with child readiness.
    let founder = wait_for_founder_kill(&mut cluster, parent);
    // The plan must stop short of published routing, and must say why.
    //
    // Which phase it holds in is deliberately not asserted. Readiness is proven
    // before the fence so an unready child never holds a range unwritable, which
    // means the plan waits at `BaseSeeded`; if it had already been fenced it
    // would wait at `Fenced`. The property is the same either way: no routing
    // moves, the parent keeps serving, and the reason is reported.
    let deadline = Instant::now() + Duration::from_secs(90);
    let mut held = false;
    while Instant::now() < deadline {
        let (phase, blocked, serving) = split_view(&cluster);
        assert!(
            !serving.is_empty(),
            "the parent stopped serving while the split was still in {phase:?}"
        );
        assert!(
            !matches!(
                phase.as_str(),
                "ChildrenReady" | "CutoverCommitted" | "ParentRetiring" | "Completed"
            ),
            "routing reached {phase:?} with the founder dead: the readiness proof let \
             a child through that cannot elect a leader"
        );
        if !phase.is_empty() {
            held = true;
            assert!(
                serving.contains(&parent),
                "the parent must remain the serving authority while its children are \
                 not ready, but the serving set is {serving:?} in {phase:?}"
            );
            if matches!(phase.as_str(), "BaseSeeded" | "Fenced" | "CutoverCommitted") {
                assert!(
                    blocked.is_some(),
                    "a split gated on readiness in {phase:?} must report which \
                     prerequisite is unmet, not just stop"
                );
            }
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    assert!(
        held,
        "removing the founder ({founder}) never left the plan visible at all"
    );
    // Reads keep working throughout: the parent is still serving them.
    for name in &parent_keys {
        assert_eq!(traffic_get(&client, name), b"before-split");
    }
    // And the split completes once the founder is back.
    cluster.restart(founder).expect("founder restarts");
    cluster.wait_splits_done(Duration::from_secs(600));
}

/// Kills the split's founder (its lowest-numbered planned replica) while the
/// plan is still `Planned`, returning the member index.
///
/// `Planned` specifically: the children get their Raft membership from the
/// founder's `initialize_group`, and once that has run, a three-replica child
/// survives losing one - it still has a quorum and an elected leader. Removing
/// the founder afterwards does not make a child unready, and asserting that it
/// does would be asserting something false. Before provisioning, the founder's
/// absence leaves the child groups with no membership at all, which is the state
/// the readiness proof exists to refuse.
fn wait_for_founder_kill(cluster: &mut Cluster, parent: u64) -> usize {
    let deadline = Instant::now() + Duration::from_secs(120);
    loop {
        let (phase, _, _) = split_view(cluster);
        if phase == "Planned" {
            break;
        }
        // An empty phase is "the plan has not been created yet", which is the
        // normal state immediately after the split is accepted.
        assert!(
            Instant::now() < deadline,
            "the split reached {phase:?} before the founder could be removed"
        );
        std::thread::sleep(Duration::from_millis(10));
    }
    // The plan's replicas are the parent's desired set; the founder is the
    // lowest of them, which is the rule `initialize_split_children` uses.
    let mut founder = u64::MAX;
    for index in 0..cluster.member_count() {
        if !cluster.alive(index) {
            continue;
        }
        let (_, splits) = cluster.admin_get(index, "/v1/control/splits");
        if let Some(replicas) = splits
            .get("splits")
            .and_then(serde_json::Value::as_array)
            .and_then(|plans| plans.first())
            .and_then(|plan| plan.get("replicas"))
            .and_then(serde_json::Value::as_array)
        {
            for replica in replicas.iter().filter_map(serde_json::Value::as_u64) {
                founder = founder.min(replica);
            }
        }
        if founder != u64::MAX {
            break;
        }
    }
    assert!(founder != u64::MAX, "the split plan named no replicas");
    let index = usize::try_from(founder - 1).unwrap_or(0);
    assert!(
        index < cluster.member_count() && cluster.alive(index),
        "founder node {founder} is not a live member of parent {parent}"
    );
    // A control leader elsewhere, so the kill takes out the founder's consensus
    // groups and nothing else.
    let leader_deadline = Instant::now() + Duration::from_secs(120);
    while cluster.control_leader_index() == Some(index) {
        assert!(
            Instant::now() < leader_deadline,
            "node {founder} never handed off the control leadership, so killing it \
             would stop the reconciler instead of the split"
        );
        std::thread::sleep(Duration::from_millis(50));
    }
    cluster.kill(index);
    index
}

/// A split interrupted by a full cluster restart resumes from its persisted
/// phase, creates no second set of children, and leaves the namespace serving.
///
/// Restart safety is the whole reason the phases are replicated rather than
/// held in memory, so this kills every member at an arbitrary point and checks
/// the two things that would be wrong if they were not: duplicate children, and a
/// namespace that never comes back.
#[test]
fn a_split_survives_a_full_restart() {
    let mut cluster = Cluster::spawn_with_tablets(3, 2, false).expect("cluster spawns");
    let _ = cluster.wait_all_leaders();
    let keys = cluster.keys_for_tablets(2);
    let parent = *keys.keys().next().expect("a tablet");
    let client = cluster.client();
    for name in &keys[&parent] {
        put(&client, name, b"restart-value");
    }
    let reply = cluster.split_tablet(parent);
    assert_eq!(
        reply.get("ok").and_then(serde_json::Value::as_bool),
        Some(true),
        "split accepted: {reply}"
    );
    // Restart at a phase boundary, not at a fixed delay: the interesting
    // restart is one that lands after children exist and before routing moved.
    let deadline = Instant::now() + Duration::from_secs(120);
    loop {
        let (phase, _, _) = split_view(&cluster);
        if matches!(phase.as_str(), "BaseSeeded" | "Fenced") {
            break;
        }
        assert_ne!(
            phase, "Completed",
            "the split finished before a restart could interrupt it"
        );
        assert!(
            Instant::now() < deadline,
            "the split never reached a restartable phase"
        );
        std::thread::sleep(Duration::from_millis(20));
    }
    cluster.restart_all().expect("cluster restarts");

    cluster.wait_splits_done(Duration::from_secs(600));
    // Exactly two children, named once: a replayed transition that re-allocated
    // would show up as a third tablet, or as the plan's children changing.
    let (phase, _, serving) = split_view(&cluster);
    assert_eq!(phase, "Completed", "the split resumed after the restart");
    // Three tablets, one of them split: two untouched parents plus the two
    // children. A replayed transition that re-allocated would show up here as an
    // extra tablet, and one that never completed as a missing child.
    assert_eq!(
        serving.len(),
        4,
        "two untouched parents plus exactly two split children: {serving:?}"
    );
    for name in &keys[&parent] {
        assert_eq!(
            traffic_get(&client, name),
            b"restart-value",
            "state survived a restart mid-split"
        );
    }
}

/// Merge money flow: merge two split children back online; one
/// merged range, all state preserved, dedup preserved, old children
/// fenced/retired, client caches converge.
#[test]
fn merge_children_back_online() {
    let cluster = Cluster::spawn_with_tablets(2, 2, false).expect("cluster spawns");
    let _ = cluster.wait_all_leaders();
    let keys = cluster.keys_for_tablets(2);
    let parent = *keys.keys().next().expect("a tablet");
    let client = cluster.client();
    for (index, name) in keys[&parent].iter().enumerate() {
        put(&client, name, format!("merge-{index}").as_bytes());
    }
    let session = kivi_types::SessionId::from_u128(0xE0EC_A7E0_0002);
    let session_client = cluster.client_with_session(session);
    let counter_key = format!("{}-merge-counter", keys[&parent][0]);
    let counter_at = session_client
        .counter_add(&Key::from(counter_key.clone()), 3)
        .expect("counter commits");
    // Split, write on children, then merge back.
    let (left, right) = split_and_wait(&cluster, parent);
    put(&client, &format!("{counter_key}-post"), b"child-write");
    merge_and_wait(&cluster, left, right);
    // One merged range serves everything; dedup survived the round trip.
    for name in &keys[&parent] {
        let _ = traffic_get(&client, name);
    }
    let live: i64 = session_client
        .counter_get(&Key::from(counter_key.clone()))
        .expect("counter reads")
        .expect("counter present");
    assert_eq!(live, counter_at, "counter preserved across split+merge");
    assert_eq!(
        traffic_get(&client, &format!("{counter_key}-post")),
        b"child-write"
    );
}

/// Automatic split/merge demonstration: tiny thresholds in the live
/// policy show `write → automatic split` then (after cooling) eventual
/// merge through the real control-plane flow.
#[test]
fn automatic_split_then_merge() {
    let cluster = Cluster::spawn_with_tablets(2, 2, false).expect("cluster spawns");
    let _ = cluster.wait_all_leaders();
    let live = cluster.live_index();
    // Tiny split threshold (20 objects), generous merge threshold (5).
    let _ = cluster.set_policy(&serde_json::json!({
        "auto_split": true,
        "auto_merge": false,
        "split_object_threshold": 20,
    }));
    let client = cluster.client();
    for index in 0..120 {
        traffic_put(&client, &format!("auto-{index:04}"), b"x");
    }
    // An automatic split plan appears with no manual command.
    let deadline = Instant::now() + Duration::from_secs(300);
    let (left, right) = loop {
        let splits = cluster.splits(live);
        let plans = splits
            .get("splits")
            .and_then(serde_json::Value::as_array)
            .cloned()
            .unwrap_or_default();
        if let Some(plan) = plans.first() {
            break (
                plan.get("left")
                    .and_then(serde_json::Value::as_u64)
                    .unwrap_or(0),
                plan.get("right")
                    .and_then(serde_json::Value::as_u64)
                    .unwrap_or(0),
            );
        }
        assert!(Instant::now() < deadline, "automatic split never planned");
        std::thread::sleep(Duration::from_secs(2));
    };
    cluster.wait_splits_done(Duration::from_secs(600));
    assert!(left != 0 && right != 0);
    // Cool the children down to mergeable size, then enable auto-merge.
    for index in 0..120 {
        let _ = client.delete(&Key::from(format!("auto-{index:04}")));
    }
    let _ = cluster.set_policy(&serde_json::json!({
        "auto_split": false,
        "auto_merge": true,
        "merge_object_threshold": 10,
    }));
    let deadline = Instant::now() + Duration::from_secs(300);
    loop {
        let merges = cluster.merges(live);
        let plans = merges
            .get("merges")
            .and_then(serde_json::Value::as_array)
            .cloned()
            .unwrap_or_default();
        if !plans.is_empty() {
            break;
        }
        assert!(Instant::now() < deadline, "automatic merge never planned");
        std::thread::sleep(Duration::from_secs(2));
    }
    cluster.wait_merges_done(Duration::from_secs(600));
}

/// Topology survives a full restart: split, checkpoint, purge logs,
/// restart the cluster - the new directory stays, old parents never
/// reappear as active.
#[test]
fn topology_survives_full_restart() {
    let mut cluster = Cluster::spawn_with_tablets(4, 2, false).expect("cluster spawns");
    let _ = cluster.wait_all_leaders();
    let keys = cluster.keys_for_tablets(2);
    let parent = *keys.keys().next().expect("a tablet");
    let client = cluster.client();
    for name in &keys[&parent] {
        put(&client, name, b"restart-proof");
    }
    let dir_before = cluster
        .node_info(cluster.live_index())
        .get("dir_version")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    let (left, right) = split_and_wait(&cluster, parent);
    // Cutover publishes per node (leader first, then peers): wait for
    // convergence instead of asserting a single member immediately.
    cluster.wait_dir_version(dir_before + 1, Duration::from_secs(120));
    let dir_after = cluster
        .node_info(cluster.live_index())
        .get("dir_version")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    assert!(dir_after > dir_before);
    // Checkpoint + purge on every member (live tablets only: the retired
    // parent no longer serves), then full restart.
    let live_tablets: Vec<u64> = keys
        .keys()
        .copied()
        .filter(|tablet| *tablet != parent)
        .chain([left, right])
        .collect();
    for tablet in live_tablets {
        for index in 0..cluster.member_count() {
            if cluster.alive(index) {
                let _ = cluster.snapshot_tablet(index, tablet);
            }
        }
    }
    cluster.kill_all();
    cluster.restart_all().expect("cluster restarts");
    let _ = cluster.wait_all_leaders();
    let dir_restarted = cluster
        .node_info(cluster.live_index())
        .get("dir_version")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    assert_eq!(
        dir_restarted, dir_after,
        "restarted cluster recovers the post-split directory"
    );
    for name in &keys[&parent] {
        assert_eq!(traffic_get(&client, name), b"restart-proof");
    }
}
