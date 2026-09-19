---- MODULE escrow ----
(*
  Phase-8 formal model for Kivi's BoundedCounter escrow discipline:
  capacity rights split across tablets, local spend against a held
  share, paired narrow/credit transfer, crash/restart with durable
  survival, and idempotent transfer replay.

  Configuration: two tablets (1, 2), capacity 3, single in-flight
  transfer slot plus a message soup for transfer notices.

  Correspondence to crates/kivi-state/src/{txn,store}.rs (rule by rule):
    escrow width ................ Widths[p] (durable per-tablet share)
    local spend ................. Spend (guard share >= n; debit now —
                                  previewed by predict_cost, settled by
                                  apply BoundedAdd with EscrowShare)
    paired transfer ............. Narrow (debit source, escrow to the
                                  transit slot) + DeliverCredit (credit
                                  the destination, clear the slot) —
                                  plan_escrow_transfer / transfer_escrow
    idempotent replay ........... ResendNotice / DeliverNotice (credit
                                  guarded on the live slot; replays of a
                                  cleared slot are no-ops)
    crash/restart ............... Crash (Widths + transit + spent survive;
                                  notices wait) / Restart
    width verification .......... verify_escrow_widths: Narrow/Spend are
                                  disabled when a tablet holds no share
                                  for a transfer it must fund

  Deliberate abstractions (documented, not hidden):
    - One counter, one transfer at a time (the slot). Concurrent
      transfers compose by sequentialization; the conserved quantity
      is per-transfer, proved here.
    - Spent rights leave the system (capacity consumed by decrements);
      they are tracked explicitly so conservation is exact.
    - No wall clocks; transfer has no deadline in the model (the
      implementation re-drives; dropping a transfer is never modeled
      as minting).
    - AllowMint selects the BROKEN variant: Narrow credits the transit
      slot without debiting the source -> violates RightsConserved.

  Invariants:
    RightsConserved: widths + transit + spent always equal capacity.
    NoNegative: no width or the slot ever goes negative (guards hold).
*)

EXTENDS Integers, FiniteSets, TLC

CONSTANTS Tablets, Capacity, AllowMint
VARIABLES widths, transit, spent, notices, crashed

Vars == <<widths, transit, spent, notices, crashed>>

TypeOK ==
    /\ widths \in [Tablets -> 0..Capacity]
    /\ transit \in 0..Capacity
    /\ spent \in 0..Capacity
    /\ notices \in SUBSET [from : Tablets, to : Tablets, amount : 1..Capacity]
    /\ crashed \in [Tablets -> BOOLEAN]

Init ==
    /\ widths = [t \in Tablets |-> IF t = 1 THEN Capacity ELSE 0]
    /\ transit = 0
    /\ spent = 0
    /\ notices = {}
    /\ crashed = [t \in Tablets |-> FALSE]

Spend(p, n) ==
    /\ ~crashed[p]
    /\ n \in 1..Capacity
    /\ widths[p] >= n
    /\ widths' = [widths EXCEPT ![p] = @ - n]
    /\ spent' = spent + n
    /\ UNCHANGED <<transit, notices, crashed>>

Narrow(p, q, n) ==
    /\ ~crashed[p]
    /\ p /= q
    /\ n \in 1..Capacity
    /\ transit = 0
    /\ IF AllowMint
       THEN /\ widths' = widths
            /\ transit' = n
       ELSE /\ widths[p] >= n
            /\ widths' = [widths EXCEPT ![p] = @ - n]
            /\ transit' = n
    /\ notices' = notices \cup {[from |-> p, to |-> q, amount |-> n]}
    /\ UNCHANGED <<spent, crashed>>

DeliverNotice(p, q, n) ==
    LET m == [from |-> p, to |-> q, amount |-> n] IN
    /\ m \in notices
    /\ ~crashed[q]
    /\ notices' = notices \ {m}
    /\ IF transit = n
       THEN /\ widths' = [widths EXCEPT ![q] = @ + n]
            /\ transit' = 0
            /\ UNCHANGED <<spent, crashed>>
       ELSE UNCHANGED <<widths, transit, spent, crashed>>

ResendNotice(p, q, n) ==
    /\ transit = n
    /\ notices' = notices \cup {[from |-> p, to |-> q, amount |-> n]}
    /\ UNCHANGED <<widths, transit, spent, crashed>>

Crash(p) ==
    /\ ~crashed[p]
    /\ crashed' = [crashed EXCEPT ![p] = TRUE]
    /\ UNCHANGED <<widths, transit, spent, notices>>

Restart(p) ==
    /\ crashed[p]
    /\ crashed' = [crashed EXCEPT ![p] = FALSE]
    /\ UNCHANGED <<widths, transit, spent, notices>>

LoseNotice(m) ==
    /\ m \in notices
    /\ transit > 0
    /\ notices' = notices \ {m}
    /\ UNCHANGED <<widths, transit, spent, crashed>>

Next ==
    \/ \E p \in Tablets, n \in 1..Capacity : Spend(p, n)
    \/ \E p \in Tablets, q \in Tablets, n \in 1..Capacity : Narrow(p, q, n)
    \/ \E p \in Tablets, q \in Tablets, n \in 1..Capacity : DeliverNotice(p, q, n)
    \/ \E p \in Tablets, q \in Tablets, n \in 1..Capacity : ResendNotice(p, q, n)
    \/ \E p \in Tablets : Crash(p)
    \/ \E p \in Tablets : Restart(p)
    \/ \E m \in notices : LoseNotice(m)

Spec == Init /\ [][Next]_Vars

RightsConserved ==
    widths[1] + widths[2] + transit + spent = Capacity

NoNegative ==
    /\ \A p \in Tablets : widths[p] >= 0
    /\ transit >= 0

SoupBound == Cardinality(notices) <= 4

THEOREM Spec => []TypeOK

====
