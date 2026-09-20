//! Transaction-fabric properties: coordinator selection, write-set digest
//! binding, the decision state machine, and escrow planning. Each property
//! pins a Phase-8 invariant that hand-written examples can only sample:
//! order-independence (replicas, re-drives, and recovery must derive the
//! same coordinator and digest from the same set), digest sensitivity (the
//! confused deputy is bound, never obeyed), and conservation (planning can
//! neither mint nor destroy rights).

use std::collections::BTreeSet;

use bytes::Bytes;
use kivi_state::{
    Key, TxnCoordinator, TxnError, TxnExpect, TxnState, TxnWrite, TxnWriteKind, check_bounds,
    plan_escrow_transfer, reservation_cost, select_coordinator, verify_escrow_widths,
    write_set_digest,
};
use kivi_types::{NamespaceId, TabletId};
use proptest::prelude::*;

const NS: NamespaceId = NamespaceId::from_u64(7);

fn arb_key() -> impl Strategy<Value = Key> {
    prop::sample::select(vec!["a", "b", "c", "d"]).prop_map(Key::from)
}

fn arb_kind() -> impl Strategy<Value = TxnWriteKind> {
    prop_oneof![
        prop::collection::vec(any::<u8>(), 0..8).prop_map(|v| TxnWriteKind::Put(Bytes::from(v))),
        Just(TxnWriteKind::Delete),
        (-5i64..5).prop_map(TxnWriteKind::CounterAdd),
        (-5i64..5).prop_map(TxnWriteKind::CommutativeAdd),
        (-5i64..5).prop_map(TxnWriteKind::BoundedAdd),
        (0i64..4, 0i64..4).prop_map(|(min, span)| TxnWriteKind::EscrowSetShare {
            min,
            max: min + span,
        }),
    ]
}

fn arb_write() -> impl Strategy<Value = TxnWrite> {
    (arb_key(), arb_kind()).prop_map(|(key, kind)| TxnWrite {
        key,
        kind,
        expect: TxnExpect::Any,
    })
}

fn route(key: &[u8]) -> Option<TabletId> {
    key.first()
        .map(|byte| TabletId::from_u64(u64::from(*byte) % 3))
}

proptest! {
    /// The coordinator is the lowest participant tablet — never a service,
    /// never first-writer-wins.
    #[test]
    fn coordinator_select_is_min(tablets in prop::collection::vec(0u64..8, 0..6)) {
        let set: BTreeSet<TabletId> = tablets.into_iter().map(TabletId::from_u64).collect();
        prop_assert_eq!(
            TxnCoordinator::select(&set),
            set.iter().next().map(|tablet| TxnCoordinator { tablet: *tablet }),
        );
    }

    /// Coordinator derivation is order-independent: re-drives and recovery
    /// probes present the same set in any order and must agree.
    #[test]
    fn select_coordinator_order_free(writes in prop::collection::vec(arb_write(), 1..6)) {
        let mut reversed = writes.clone();
        reversed.reverse();
        let forward = select_coordinator(&writes, route);
        prop_assert_eq!(forward, select_coordinator(&reversed, route));
        let min = writes.iter().filter_map(|w| route(w.key.as_bytes())).min();
        prop_assert_eq!(forward, min);
    }

    /// The write-set digest binds the effective set: collapsing
    /// duplicates last-wins (by hand, in the test) changes nothing, and
    /// distinct-key sets hash identically in any order.
    #[test]
    fn write_set_digest_binds_effective_set(
        writes in prop::collection::vec(arb_write(), 0..8),
        distinct_names in prop::collection::hash_set(prop::sample::select(vec!["p", "q", "r", "s"]), 0..4),
    ) {
        let mut collapsed: std::collections::BTreeMap<Vec<u8>, TxnWrite> =
            std::collections::BTreeMap::new();
        for write in &writes {
            collapsed.insert(write.key.as_bytes().to_vec(), write.clone());
        }
        let canonical: Vec<TxnWrite> = collapsed.into_values().collect();
        prop_assert_eq!(
            write_set_digest(&writes, NS),
            write_set_digest(&canonical, NS),
        );
        let mut distinct: Vec<TxnWrite> = distinct_names
            .into_iter()
            .map(|name| TxnWrite {
                key: Key::from(name),
                kind: TxnWriteKind::CounterAdd(1),
                expect: TxnExpect::Any,
            })
            .collect();
        let mut reversed = distinct.clone();
        distinct.sort_by(|a, b| a.key.as_bytes().cmp(b.key.as_bytes()));
        reversed.sort_by(|a, b| b.key.as_bytes().cmp(a.key.as_bytes()));
        prop_assert_eq!(
            write_set_digest(&distinct, NS),
            write_set_digest(&reversed, NS),
        );
    }

    /// The digest is sensitive exactly where the transaction is: mutating
    /// the winning write for a key rebinds; mutating a shadowed duplicate
    /// is invisible (last-wins, as applied).
    #[test]
    fn write_set_digest_sensitive(writes in prop::collection::vec(arb_write(), 1..5)) {
        let before = write_set_digest(&writes, NS);
        let last_for_key = |key: &[u8]| {
            writes
                .iter()
                .rposition(|write| write.key.as_bytes() == key)
        };
        let target = writes.len() - 1;
        let target_key = writes[target].key.as_bytes().to_vec();
        prop_assert_eq!(Some(target), last_for_key(&target_key));
        let mut rebound = writes.clone();
        {
            let first = &mut rebound[target];
            match &mut first.kind {
                TxnWriteKind::Put(value) => {
                    let mut grown = value.to_vec();
                    grown.push(0xff);
                    *value = Bytes::from(grown);
                }
                TxnWriteKind::Delete => first.kind = TxnWriteKind::CounterAdd(1),
                TxnWriteKind::CounterAdd(delta)
                | TxnWriteKind::CommutativeAdd(delta)
                | TxnWriteKind::BoundedAdd(delta) => *delta += 1,
                TxnWriteKind::EscrowSetShare { max, .. } => *max += 1,
                TxnWriteKind::PutChunked { .. } => unreachable!("generator never emits chunked"),
                TxnWriteKind::PutFabric { .. } => unreachable!("generator never emits fabric"),
            }
        }
        prop_assert_ne!(before, write_set_digest(&rebound, NS));
        // A shadowed duplicate (not the last writer for its key) is dead:
        // changing it must not rebind.
        if let Some(shadowed) = (0..writes.len())
            .find(|index| *index != last_for_key(writes[*index].key.as_bytes()).expect("present"))
        {
            let mut invisible = writes.clone();
            {
                let victim = &mut invisible[shadowed];
                match &mut victim.kind {
                    TxnWriteKind::Put(value) => {
                        let mut grown = value.to_vec();
                        grown.push(0xfe);
                        *value = Bytes::from(grown);
                    }
                    TxnWriteKind::Delete => victim.kind = TxnWriteKind::CounterAdd(2),
                    TxnWriteKind::CounterAdd(delta)
                    | TxnWriteKind::CommutativeAdd(delta)
                    | TxnWriteKind::BoundedAdd(delta) => *delta += 2,
                    TxnWriteKind::EscrowSetShare { max, .. } => *max += 2,
                    TxnWriteKind::PutChunked { .. } => {
                        unreachable!("generator never emits chunked")
                    }
                    TxnWriteKind::PutFabric { .. } => {
                        unreachable!("generator never emits fabric")
                    }
                }
            }
            prop_assert_eq!(before, write_set_digest(&invisible, NS));
        }
        // Rekeying the winner always rebinds.
        let mut rekeyed = writes.clone();
        let mut raw = rekeyed[target].key.as_bytes().to_vec();
        raw.push(0xff);
        rekeyed[target].key = Key::from(raw);
        prop_assert_ne!(before, write_set_digest(&rekeyed, NS));
    }

    /// Every planned transfer verifies: planning conserves by construction,
    /// so a driver preparing both moves can never mint or destroy rights.
    #[test]
    fn planned_transfer_always_verifies(
        donor_min in 0i64..8,
        donor_span in 0i64..8,
        donor_offset in 0i64..8,
        recip_min in 0i64..8,
        recip_span in 0i64..8,
        extra_capacity in 0u64..8,
        amount in 1u64..8,
    ) {
        let donor_max = donor_min + donor_span;
        let donor_value = donor_min + donor_offset.min(donor_span);
        let recip_max = recip_min + recip_span;
        let capacity = recip_max.max(0).cast_unsigned() + extra_capacity;
        let planned = plan_escrow_transfer(
            (donor_min, donor_max),
            donor_value,
            (recip_min, recip_max),
            capacity,
            amount,
        );
        if let Ok((donor_after, recip_after)) = planned {
            prop_assert!(verify_escrow_widths(
                (donor_min, donor_max),
                donor_after,
                (recip_min, recip_max),
                recip_after,
            )
            .is_ok());
            let width = |(min, max): (i64, i64)| max - min;
            prop_assert_eq!(
                width(donor_after) + width(recip_after),
                width((donor_min, donor_max)) + width((recip_min, recip_max)),
            );
        }
    }

    /// Reservation cost counts one intent per key: admission reserves before
    /// acknowledging prepare, so prepared-then-unappliable is unreachable.
    #[test]
    fn reservation_counts_intents(writes in prop::collection::vec(arb_write(), 0..8)) {
        if check_bounds(2, &writes).is_ok() {
            let cost = reservation_cost(&writes).expect("bounded");
            prop_assert_eq!(cost.intents, writes.len());
        }
    }
}

/// The decision machine is exactly the documented table: decide once,
/// replay freely, never flip, never resurrect.
#[test]
fn decision_state_machine_matches_table() {
    use TxnState::{Aborted, Begun, Committed};
    let table = [
        (Begun, Begun, Ok(Begun)),
        (Begun, Committed, Ok(Committed)),
        (Begun, Aborted, Ok(Aborted)),
        (Committed, Committed, Ok(Committed)),
        (Aborted, Aborted, Ok(Aborted)),
        (Committed, Aborted, Err(TxnError::DecisionConflict)),
        (Aborted, Committed, Err(TxnError::DecisionConflict)),
        (Committed, Begun, Err(TxnError::DecisionConflict)),
        (Aborted, Begun, Err(TxnError::DecisionConflict)),
    ];
    for (from, next, want) in table {
        assert_eq!(from.transition_to(next), want, "{from:?} -> {next:?}");
    }
}
