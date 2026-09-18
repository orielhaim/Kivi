---- MODULE roster_lease ----
(*
  Phase-7 formal model for Kivi's read-authority layer: Bodega-style
  directional roster leases + Lazy-ALR + the write-coverage gate.

  Models one tablet's lease engine on three replicas, two terms, two
  roster generations per term, Guard -> Renew activation, revoke-first
  transitions, restart quarantine with incarnations, message loss /
  reordering / delay (stream transport: delivery consumes the message;
  no application-level duplication — the scalar-counting audit in the
  implementation covers that separately), partitions, crash + sat-out
  restart,
  authority transitions, Raft commit vs external completion, local reads,
  and ALR fence/subsumption reads.

  Correspondence to crates/kivi-types/src/roster.rs (exact, rule by rule):
    Guard send / open_guard ......... SendGuard inside Announce,
                                      DeliverGuard auto-grant, RepairGuard,
                                      GuardTimeout (fresh attempt per send;
                                      never overwrites a live exclusion)
    Guard receipt / on_guard ........ DeliverGuard (quarantine gate,
                                      authority/term checks, attempt
                                      freshness, auto-grant closure,
                                      supersede-revoke, generation prune)
    GuardReply receipt .............. DeliverGuardReply (exact attempt+seq
                                      match; first Renew carries committed)
    Renew receipt / on_renew ........ DeliverRenew (exact attempt, seq
                                      advance, in-window, floor folds
                                      max(guard-thresh, renew-committed))
    RenewReply receipt .............. DeliverRenewReply (exact match, unmute)
    Revoke / RevokeReply ............ DeliverRevoke / DeliverRevokeReply
    tick ............................ GrantorTick (renewal cadence by
                                      renew_due, mute past MUTE_AFTER,
                                      exclusion lapse) + HoldLapse
    reconcile ....................... folded into Elect (revoke stale
                                      outgoing; grantee state lapses,
                                      never dropped early)
    stable .......................... StableHere (majority incl leader leg,
                                      designated grantors, current term,
                                      known/pruned roster)
    evidence ........................ LocalReadFinish (stable + covers +
                                      applied >= floor)
    announce ........................ Announce (leader-only, next
                                      generation, revoke-superseded first)
    covered_grantees ................ CoveredSet (Renewing + Revoking)
    coverage_gate + takeover ........ CompleteWrite (cover all live
                                      exclusions + new-leader wait)
    propose_sync ..................... AlrBoundary (subsumed = committed,
                                      else fence entry with coverage)
    quarantine ...................... Crash (quarantine_until) + GrantGate
    grant_for (driver repair) ........ RepairGuard (re-open where no live
                                      pairing, idempotent retry)

  Deliberate abstractions (documented, not hidden):
    - One tablet only. Cross-tablet transactions run lease-ineligible
      (LeaseEligibility::ConservativeOnly) in the implementation, covered
      by Rust tests, not by this model.
    - One outstanding read/batch per node (reads and batches serialize
      per node here; concurrency across nodes is fully modeled).
    - Voter set fixed = Nodes. Voter-set change is covered by Rust tests.
    - Time is a global integer; the asymmetric grantor/grantee deadlines
      (exclusion L+A vs hold L-A) are structural constants. Clock-rate
      drift itself is proved arithmetically by the Rust containment
      tests (opposite_max_drift_*, exact to the microsecond), not here.
    - Attempt identities are (incarnation, guard-sequence) pairs
      encoded as a single nat, continuing per pairing like the
      implementation's open_guard and distinct across restarts;
      numbers/sequences range 0..MaxSeq with sends disabled at the top
      (no wrap, no overflow). The model checks identity matching, not
      64-bit exhaustion (Rust next() tests).
    - Designated responder sets are fixed per generation
      (gen 1 -> {1,2}, gen 2 -> {2,3}); a generation bump models a
      responder-set change.
    - Coverage polls are instantaneous predicates on applied[] (the Rust
      gate retries them over a bounded window; polling vs sampling does
      not change the verdict set, only latency).
    - Internal Raft commit and external completion are separate actions
      with the coverage gate between them (WriteInternal, CompleteWrite).
    - Revoke acceptance needs only the roster identity (no sequence
      gate): roster identities never repeat, so a stale revoke for an
      old roster cannot touch a new pairing.

  Weakening switch: CONSTANT Weaken selects one structural rule to drop.
  Only "floor" and "takeover" currently exhibit counterexamples (fast,
  targeted inits below); the other four weakenings are retained as
  documentation of the threat model but have no exhibiting config —
  each is safety-redundant under the fail-closed deadline gates, with
  the reason recorded in the ledger. Removing a rule must either
  exhibit a fresh counterexample or extend the ledger with its
  redundancy proof.
    "none" ....... base model (must satisfy all invariants)
    "floor" ...... local reads require stability only (no floor
                   coverage) — EXHIBITS (repair witness, WarmFloorInit)
    "takeover" ... writes complete without the new-leader takeover
                   wait — EXHIBITS (crash witness, WarmInit)
    "quarantine" . grant/announce ignore the restart fence —
                   REDUNDANT (ledger: the intact fence plus
                   current-commit promo floors close every window)
    "attempt" .... replies accepted for any attempt/sequence —
                   REDUNDANT (ledger: floor folds max, deadlines only
                   refresh live holds, phases gate promotion)
    "generation" . no pruning, no designated checks (old+new coexist) —
                   REDUNDANT (ledger: old holds need a live old
                   grantor, which covers; promo/coverage sandwich)
    "revocation" . new Guards overwrite live exclusions; term rise sends
                   no revokes — REDUNDANT (ledger: a departed leader's
                   old pairing can never refresh — repair is
                   same-term-only — so stale-term stability dies with
                   its holds)

  Scope flags: CONSTANT Flags selects which fault/auxiliary actions TLC
  may fire, so each negative config explores the smallest state space
  that still admits its intended witness — faults a witness never uses
  only multiply interleavings. CONSTANT SoupCap bounds in-flight
  messages per config. Only fast-verdict configs ship in spec/ (see
  the ledger); deeper explorations (full-composition base with all
  flags, larger bounds) are manual research: set all Flags,
  MaxTime=10, MaxSeq=4, MaxGen=2, SoupCap=20, INIT Init and run by
  hand — expect hundreds of millions of states, not CI minutes.
    "drop" ..... message loss (DropMsg; non-delivery is already explored
                 by simply never firing a Deliver)
    "part" ..... partitions and heals
    "crash" .... crash with restart quarantine (incarnation bump, fence)
    "down" ..... sat-out restarts (down, up)
    "elect" .... term rise to term 2
    "auth" ..... authority-lineage bump
    "alr" ...... Lazy-ALR batch reads (lease witnesses serve local reads)

  Fail-closed stall discipline (load-bearing, mirrors the
  implementation): every safety use of time-bound state revalidates the
  deadline against now and fails closed — stability counts only live
  holds (GrantsFor), completion covers only live exclusions
  (CoveredSet), renewal receipt requires a live held pairing
  (DeliverRenew), exactly like the implementation's tick-removal
  semantics but without assuming tick ever runs. A stalled node loses
  fast-path authority; it never retains stale authority. Bounded
  clock-rate drift is the only timing assumption (proved arithmetically
  by the Rust containment tests); no scheduler or tick fairness is
  assumed. The asymmetric deadlines (exclusion outlasts hold) keep the
  two   sides consistent: a grantor never drops coverage while the
  grantee can still serve.

  Fast suite (spec/run_tla.ps1 default, all green 2026-09-18):
    "canonical" .... EXHAUSTIVE clean from WarmFreshInit (670K states,
                   depth 31, 10s): fresh-regime interplay with serves
                   happening — stability, coverage chase, designation,
                   leader leg. The mandatory gate. Stall-fallback is
                   covered by Rust unit tests, depth by the nightly
                   base run.
    "takeover" ... EXHIBITS in seconds from WarmInit (crash empties
                   coverage, fence skipped, fresh-hold stale serve;
                   depth 6). Base twin: the restart fence (termStart
                   reset on crash) forces the 6-tick wait that lets
                   holds die.
    "floor" ...... EXHIBITS in ~1 minute from WarmFloorInit (lapsed
                   exclusions, intact-but-expired holds, fence already
                   satisfied): uncovered completion, post-commit driver
                   repair of two pairings, serve below the ignored
                   floor (depth 13). Base-side safety on the same shape
                   follows by trace analysis — the post-commit
                   promo-renew always raises the floor past applied,
                   and completing while covered chases applied past the
                   floor — plus the Rust floor/evidence tests.
  Removed from the suite (no exhibiting config exists; each rule is
  safety-redundant under the fail-closed deadline gates):
    "attempt" .... stale-accept is semantically void — floors fold max
                   (never lower), deadlines refresh only live holds
                   (receipt-continuity), promotion still consumes the
                   Guarding phase.
    "generation" . an old roster serves only on fresh old holds, which
                   need a live old grantor, which covers completions
                   (chase); promo-renews carry the current commit, so
                   floors track replacements.
    "quarantine" . the takeover fence (intact in this variant) forces
                   the 6-tick wait that closes every window early
                   activation would open; any activation's promo-renew
                   carries the current commit into the floor.
    "revocation" . structural: serving a stale term needs a fresh old-
                   term leader leg, but a departed leader's pairing
                   refreshes only same-term (RepairGuard keys on the
                   grantor's current term), so the leg dies with its
                   holds; live old grantors keep covering (chase).
*)

EXTENDS Naturals, FiniteSets, TLC

CONSTANTS Nodes, Terms, MaxTime, MaxLog, MaxSeq, MaxGen, Weaken, Flags, SoupCap

VARIABLES
    now,
    term,
    auth,
    inc,
    quar,
    down,
    part,
    accepted,
    applied,
    commitIdx,
    completeIdx,
    leaderOf,
    termStart,
    gen,
    known,
    gPhase,
    gRoster,
    gAttempt,
    gSeq,
    gDeadline,
    gDue,
    gUnacked,
    hPhase,
    hRoster,
    hAttempt,
    hSeq,
    hDeadline,
    hFloor,
    msgs,
    pending,
    lastServed,
    servedTarget,
    batchOn,
    formation,
    boundary

vars == <<now, term, auth, inc, quar, down, part,
          accepted, applied, commitIdx, completeIdx,
          leaderOf, termStart, gen, known,
          gPhase, gRoster, gAttempt, gSeq, gDeadline, gDue, gUnacked,
          hPhase, hRoster, hAttempt, hSeq, hDeadline, hFloor,
          msgs, pending, lastServed, servedTarget, batchOn, formation, boundary>>

(* ---- abstract protocol timing (units) ----
   Calibrated minimal: holds last Lease-Allow ticks, exclusions
   Lease+Allow, guard windows absorb one unit of skew each side, and
   the quarantine covers one full forgotten grant plus one guard
   attempt. Small enough to check, large enough that every timing
   relationship (hold < exclusion, renew < hold, quarantine > hold)
   is strict and load-bearing. *)
Lease == 2
Allow == 1
GuardWin == 2
RenewEvery == 1
Quarantine == (Lease + Allow) + (GuardWin + Allow)
TakeoverFence == Quarantine
MuteAfter == 2
Majority == 2
NONEREAD == MaxLog + 1
TimeDom == 0..(MaxTime + 20)

(* ---- roster helpers ---- *)
RosterOf(a, t, g) == [auth |-> a, term |-> t, gen |-> g]
Designated(g) == IF g = 1 THEN {1, 2} ELSE {2, 3}
AttemptId(i, k) == i * 10 + k
FlagNames == {"drop", "part", "crash", "down", "elect", "auth", "alr"}

TypeOK ==
    /\ now \in 0..MaxTime
    /\ term \in [Nodes -> Terms]
    /\ auth \in 0..1
    /\ inc \in [Nodes -> 1..2]
    /\ quar \in [Nodes -> TimeDom]
    /\ down \in [Nodes -> BOOLEAN]
    /\ part \in [Nodes -> [Nodes -> BOOLEAN]]
    /\ accepted \in [Nodes -> 0..MaxLog]
    /\ applied \in [Nodes -> 0..MaxLog]
    /\ commitIdx \in 0..MaxLog
    /\ completeIdx \in 0..MaxLog
    /\ completeIdx <= commitIdx
    /\ leaderOf \in [Terms -> Nodes]
    /\ termStart \in [Nodes -> 0..MaxTime]
    /\ gen \in [Terms -> 0..MaxGen]
    /\ known \in [Nodes -> [Terms -> SUBSET (1..MaxGen)]]
    /\ gPhase \in [Nodes -> [Nodes -> {"idle", "guarding", "renewing", "revoking"}]]
    /\ gAttempt \in [Nodes -> [Nodes -> 0..30]]
    /\ gSeq \in [Nodes -> [Nodes -> 0..MaxSeq]]
    /\ gDeadline \in [Nodes -> [Nodes -> TimeDom]]
    /\ gDue \in [Nodes -> [Nodes -> TimeDom]]
    /\ gUnacked \in [Nodes -> [Nodes -> 0..MuteAfter]]
    /\ hPhase \in [Nodes -> [Nodes -> {"none", "guarded", "renewed"}]]
    /\ hAttempt \in [Nodes -> [Nodes -> 0..30]]
    /\ hSeq \in [Nodes -> [Nodes -> 0..MaxSeq]]
    /\ hDeadline \in [Nodes -> [Nodes -> TimeDom]]
    /\ hFloor \in [Nodes -> [Nodes -> 0..MaxLog]]
    /\ pending \in [Nodes -> 0..NONEREAD]
    /\ lastServed \in [Nodes -> 0..MaxLog]
    /\ servedTarget \in [Nodes -> 0..MaxLog]
    /\ batchOn \in [Nodes -> BOOLEAN]
    /\ formation \in [Nodes -> 0..MaxLog]
    /\ boundary \in [Nodes -> 0..MaxLog]
    /\ Flags \in SUBSET FlagNames

Init ==
    /\ now = 0
    /\ term = [n \in Nodes |-> 1]
    /\ auth = 0
    /\ inc = [n \in Nodes |-> 1]
    /\ quar = [n \in Nodes |-> 0]
    /\ down = [n \in Nodes |-> FALSE]
    /\ part = [n \in Nodes |-> [m \in Nodes |-> FALSE]]
    /\ accepted = [n \in Nodes |-> 0]
    /\ applied = [n \in Nodes |-> 0]
    /\ commitIdx = 0
    /\ completeIdx = 0
    /\ leaderOf = [t \in Terms |-> 1]
    /\ termStart = [n \in Nodes |-> 0]
    /\ gen = [t \in Terms |-> 0]
    /\ known = [n \in Nodes |-> [t \in Terms |-> {}]]
    /\ gPhase = [n \in Nodes |-> [m \in Nodes |-> "idle"]]
    /\ gRoster = [n \in Nodes |-> [m \in Nodes |-> RosterOf(0, 1, 1)]]
    /\ gAttempt = [n \in Nodes |-> [m \in Nodes |-> 0]]
    /\ gSeq = [n \in Nodes |-> [m \in Nodes |-> 0]]
    /\ gDeadline = [n \in Nodes |-> [m \in Nodes |-> 0]]
    /\ gDue = [n \in Nodes |-> [m \in Nodes |-> 0]]
    /\ gUnacked = [n \in Nodes |-> [m \in Nodes |-> 0]]
    /\ hPhase = [n \in Nodes |-> [m \in Nodes |-> "none"]]
    /\ hRoster = [n \in Nodes |-> [m \in Nodes |-> RosterOf(0, 1, 1)]]
    /\ hAttempt = [n \in Nodes |-> [m \in Nodes |-> 0]]
    /\ hSeq = [n \in Nodes |-> [m \in Nodes |-> 0]]
    /\ hDeadline = [n \in Nodes |-> [m \in Nodes |-> 0]]
    /\ hFloor = [n \in Nodes |-> [m \in Nodes |-> 0]]
    /\ msgs = {}
    /\ pending = [n \in Nodes |-> NONEREAD]
    /\ lastServed = [n \in Nodes |-> 0]
    /\ servedTarget = [n \in Nodes |-> 0]
    /\ batchOn = [n \in Nodes |-> FALSE]
    /\ formation = [n \in Nodes |-> 0]
    /\ boundary = [n \in Nodes |-> 0]

(* ---- warm start: established gen-1 steady state ----
   Exactly the state the cold-start Announce(1) cascade reaches with
   every message consumed and no tick fired: the five pairings the
   auto-grant topology opens ({1}->{1,2,3}, {2}->{2}, {3}->{2}) all
   renewed at attempt 11 / seq 2, floors zero, soup empty. Reachable
   from Init by construction (announce; deliver every guard; answer
   every reply; promote; deliver every renew; consume every
   renew-reply — all at now=0). Variants start here so BFS spends its
   budget on the witness, not the handshake; the base config keeps the
   cold Init. *)
WarmInit ==
    /\ now = 0
    /\ term = [n \in Nodes |-> 1]
    /\ auth = 0
    /\ inc = [n \in Nodes |-> 1]
    /\ quar = [n \in Nodes |-> 0]
    /\ down = [n \in Nodes |-> FALSE]
    /\ part = [n \in Nodes |-> [m \in Nodes |-> FALSE]]
    /\ accepted = [n \in Nodes |-> 0]
    /\ applied = [n \in Nodes |-> 0]
    /\ commitIdx = 0
    /\ completeIdx = 0
    /\ leaderOf = [t \in Terms |-> 1]
    /\ termStart = [n \in Nodes |-> 0]
    /\ gen = [t \in Terms |-> IF t = 1 THEN 1 ELSE 0]
    /\ known = [n \in Nodes |-> [t \in Terms |-> IF t = 1 THEN {1} ELSE {}]]
    /\ gPhase = [a \in Nodes |-> [b \in Nodes |->
        IF \/ a = 1 \/ (a = 2 /\ b = 2) \/ (a = 3 /\ b = 2)
        THEN "renewing" ELSE "idle"]]
    /\ gRoster = [n \in Nodes |-> [m \in Nodes |-> RosterOf(0, 1, 1)]]
    /\ gAttempt = [a \in Nodes |-> [b \in Nodes |->
        IF \/ a = 1 \/ (a = 2 /\ b = 2) \/ (a = 3 /\ b = 2)
        THEN 11 ELSE 0]]
    /\ gSeq = [a \in Nodes |-> [b \in Nodes |->
        IF \/ a = 1 \/ (a = 2 /\ b = 2) \/ (a = 3 /\ b = 2)
        THEN 2 ELSE 0]]
    /\ gDeadline = [a \in Nodes |-> [b \in Nodes |->
        IF \/ a = 1 \/ (a = 2 /\ b = 2) \/ (a = 3 /\ b = 2)
        THEN 6 ELSE 0]]
    /\ gDue = [n \in Nodes |-> [m \in Nodes |-> 0]]
    /\ gUnacked = [n \in Nodes |-> [m \in Nodes |-> 0]]
    /\ hPhase = [a \in Nodes |-> [b \in Nodes |->
        IF \/ a = 1 \/ (a = 2 /\ b = 2) \/ (a = 3 /\ b = 2)
        THEN "renewed" ELSE "none"]]
    /\ hRoster = [n \in Nodes |-> [m \in Nodes |-> RosterOf(0, 1, 1)]]
    /\ hAttempt = [a \in Nodes |-> [b \in Nodes |->
        IF \/ a = 1 \/ (a = 2 /\ b = 2) \/ (a = 3 /\ b = 2)
        THEN 11 ELSE 0]]
    /\ hSeq = [a \in Nodes |-> [b \in Nodes |->
        IF \/ a = 1 \/ (a = 2 /\ b = 2) \/ (a = 3 /\ b = 2)
        THEN 2 ELSE 0]]
    /\ hDeadline = [a \in Nodes |-> [b \in Nodes |->
        IF \/ a = 1 \/ (a = 2 /\ b = 2) \/ (a = 3 /\ b = 2)
        THEN 1 ELSE 0]]
    /\ hFloor = [n \in Nodes |-> [m \in Nodes |-> 0]]
    /\ msgs = {}
    /\ pending = [n \in Nodes |-> NONEREAD]
    /\ lastServed = [n \in Nodes |-> 0]
    /\ servedTarget = [n \in Nodes |-> 0]
    /\ batchOn = [n \in Nodes |-> FALSE]
    /\ formation = [n \in Nodes |-> 0]
    /\ boundary = [n \in Nodes |-> 0]

(* ---- targeted warm start for the floor negative ----
   WarmInit, then six silent ticks (no lapse or delivery actions), then
   ExclLapse on all five grantor pairings at now=6: every exclusion is
   gone while every grantee hold is intact-but-expired. termStart stays
   0, so the takeover fence is already satisfied — the witness needs
   zero ticks: uncovered completion, driver repair of two pairings,
   stale serve below the (weakened-away) floor. Reachable by
   construction. *)
WarmFloorInit ==
    /\ now = 6
    /\ term = [n \in Nodes |-> 1]
    /\ auth = 0
    /\ inc = [n \in Nodes |-> 1]
    /\ quar = [n \in Nodes |-> 0]
    /\ down = [n \in Nodes |-> FALSE]
    /\ part = [n \in Nodes |-> [m \in Nodes |-> FALSE]]
    /\ accepted = [n \in Nodes |-> 0]
    /\ applied = [n \in Nodes |-> 0]
    /\ commitIdx = 0
    /\ completeIdx = 0
    /\ leaderOf = [t \in Terms |-> 1]
    /\ termStart = [n \in Nodes |-> 0]
    /\ gen = [t \in Terms |-> IF t = 1 THEN 1 ELSE 0]
    /\ known = [n \in Nodes |-> [t \in Terms |-> IF t = 1 THEN {1} ELSE {}]]
    /\ gPhase = [n \in Nodes |-> [m \in Nodes |-> "idle"]]
    /\ gRoster = [n \in Nodes |-> [m \in Nodes |-> RosterOf(0, 1, 1)]]
    /\ gAttempt = [a \in Nodes |-> [b \in Nodes |->
        IF \/ a = 1 \/ (a = 2 /\ b = 2) \/ (a = 3 /\ b = 2)
        THEN 11 ELSE 0]]
    /\ gSeq = [a \in Nodes |-> [b \in Nodes |->
        IF \/ a = 1 \/ (a = 2 /\ b = 2) \/ (a = 3 /\ b = 2)
        THEN 2 ELSE 0]]
    /\ gDeadline = [a \in Nodes |-> [b \in Nodes |->
        IF \/ a = 1 \/ (a = 2 /\ b = 2) \/ (a = 3 /\ b = 2)
        THEN 6 ELSE 0]]
    /\ gDue = [n \in Nodes |-> [m \in Nodes |-> 0]]
    /\ gUnacked = [n \in Nodes |-> [m \in Nodes |-> 0]]
    /\ hPhase = [a \in Nodes |-> [b \in Nodes |->
        IF \/ a = 1 \/ (a = 2 /\ b = 2) \/ (a = 3 /\ b = 2)
        THEN "renewed" ELSE "none"]]
    /\ hRoster = [n \in Nodes |-> [m \in Nodes |-> RosterOf(0, 1, 1)]]
    /\ hAttempt = [a \in Nodes |-> [b \in Nodes |->
        IF \/ a = 1 \/ (a = 2 /\ b = 2) \/ (a = 3 /\ b = 2)
        THEN 11 ELSE 0]]
    /\ hSeq = [a \in Nodes |-> [b \in Nodes |->
        IF \/ a = 1 \/ (a = 2 /\ b = 2) \/ (a = 3 /\ b = 2)
        THEN 2 ELSE 0]]
    /\ hDeadline = [a \in Nodes |-> [b \in Nodes |->
        IF \/ a = 1 \/ (a = 2 /\ b = 2) \/ (a = 3 /\ b = 2)
        THEN 1 ELSE 0]]
    /\ hFloor = [n \in Nodes |-> [m \in Nodes |-> 0]]
    /\ msgs = {}
    /\ pending = [n \in Nodes |-> NONEREAD]
    /\ lastServed = [n \in Nodes |-> 0]
    /\ servedTarget = [n \in Nodes |-> 0]
    /\ batchOn = [n \in Nodes |-> FALSE]
    /\ formation = [n \in Nodes |-> 0]
    /\ boundary = [n \in Nodes |-> 0]

(* ---- fresh targeted start for the canonical gate ----
   WarmInit, six silent ticks, lapse of all five grantor pairings, then
   driver repair of every repairable pairing ((1,2), (2,2), (1,1),
   (3,2); (1,3) can never repair — non-designated, non-leader target)
   plus their full handshakes, all pre-write at now=6: four pairings
   renewed-fresh with zero floors, one lapsed, everything else
   untouched. termStart stays 0 (fence satisfied), commitIdx 0.
   Repairs land in this order — (1,2), then (1,1), then (2,2), then
   (3,2) — so no auto-guard cascade fires and the soup stays empty.
   From here the gate checks the fresh-regime interplay — stability,
   coverage chase, floors, designation, leader leg — with serves
   actually happening, in a tiny space: no ticks (now=MaxTime), no
   faults, every sender at its sequence cap (repairs, renewals, and
   re-guards all need gSeq < MaxSeq). Reachable by construction. What
   this scope cannot see (expiry-driven fallback) is covered by the
   Rust stall tests plus the nightly base run.
   Coverage map for the fast suite: chase/fence/stability/designation
   by canonical-exhaustive; floor by the floor negative (fires iff the
   check is skipped); takeover fence by the takeover negative (fires
   iff the wait is skipped); stall-fallback by Rust unit tests;
   attempt/generation/quarantine/revocation rules documented redundant
   in the ledger above. *)
WarmFreshInit ==
    /\ now = 6
    /\ term = [n \in Nodes |-> 1]
    /\ auth = 0
    /\ inc = [n \in Nodes |-> 1]
    /\ quar = [n \in Nodes |-> 0]
    /\ down = [n \in Nodes |-> FALSE]
    /\ part = [n \in Nodes |-> [m \in Nodes |-> FALSE]]
    /\ accepted = [n \in Nodes |-> 0]
    /\ applied = [n \in Nodes |-> 0]
    /\ commitIdx = 0
    /\ completeIdx = 0
    /\ leaderOf = [t \in Terms |-> 1]
    /\ termStart = [n \in Nodes |-> 0]
    /\ gen = [t \in Terms |-> IF t = 1 THEN 1 ELSE 0]
    /\ known = [n \in Nodes |-> [t \in Terms |-> IF t = 1 THEN {1} ELSE {}]]
    /\ gPhase = [a \in Nodes |-> [b \in Nodes |->
        IF \/ (a = 1 /\ b = 2) \/ (a = 2 /\ b = 2) \/ (a = 1 /\ b = 1) \/ (a = 3 /\ b = 2)
        THEN "renewing" ELSE "idle"]]
    /\ gRoster = [n \in Nodes |-> [m \in Nodes |-> RosterOf(0, 1, 1)]]
    /\ gAttempt = [a \in Nodes |-> [b \in Nodes |->
        IF \/ (a = 1 /\ b = 2) \/ (a = 2 /\ b = 2) \/ (a = 1 /\ b = 1) \/ (a = 3 /\ b = 2)
        THEN 13
        ELSE IF /\ a = 1 /\ b = 3 THEN 11 ELSE 0]]
    /\ gSeq = [a \in Nodes |-> [b \in Nodes |->
        IF \/ (a = 1 /\ b = 2) \/ (a = 2 /\ b = 2) \/ (a = 1 /\ b = 1) \/ (a = 3 /\ b = 2)
        THEN 4
        ELSE IF /\ a = 1 /\ b = 3 THEN 2 ELSE 0]]
    /\ gDeadline = [a \in Nodes |-> [b \in Nodes |->
        IF \/ (a = 1 /\ b = 2) \/ (a = 2 /\ b = 2) \/ (a = 1 /\ b = 1) \/ (a = 3 /\ b = 2)
        THEN 12
        ELSE IF /\ a = 1 /\ b = 3 THEN 6 ELSE 0]]
    /\ gDue = [a \in Nodes |-> [b \in Nodes |->
        IF \/ (a = 1 /\ b = 2) \/ (a = 2 /\ b = 2) \/ (a = 1 /\ b = 1) \/ (a = 3 /\ b = 2)
        THEN 6 ELSE 0]]
    /\ gUnacked = [n \in Nodes |-> [m \in Nodes |-> 0]]
    /\ hPhase = [a \in Nodes |-> [b \in Nodes |->
        IF \/ (a = 1 /\ b = 2) \/ (a = 2 /\ b = 2) \/ (a = 1 /\ b = 1) \/ (a = 3 /\ b = 2)
            \/ (a = 1 /\ b = 3)
        THEN "renewed" ELSE "none"]]
    /\ hRoster = [n \in Nodes |-> [m \in Nodes |-> RosterOf(0, 1, 1)]]
    /\ hAttempt = [a \in Nodes |-> [b \in Nodes |->
        IF \/ (a = 1 /\ b = 2) \/ (a = 2 /\ b = 2) \/ (a = 1 /\ b = 1) \/ (a = 3 /\ b = 2)
        THEN 13
        ELSE IF /\ a = 1 /\ b = 3 THEN 11 ELSE 0]]
    /\ hSeq = [a \in Nodes |-> [b \in Nodes |->
        IF \/ (a = 1 /\ b = 2) \/ (a = 2 /\ b = 2) \/ (a = 1 /\ b = 1) \/ (a = 3 /\ b = 2)
        THEN 4
        ELSE IF /\ a = 1 /\ b = 3 THEN 2 ELSE 0]]
    /\ hDeadline = [a \in Nodes |-> [b \in Nodes |->
        IF \/ (a = 1 /\ b = 2) \/ (a = 2 /\ b = 2) \/ (a = 1 /\ b = 1) \/ (a = 3 /\ b = 2)
        THEN 7
        ELSE IF /\ a = 1 /\ b = 3 THEN 1 ELSE 0]]
    /\ hFloor = [n \in Nodes |-> [m \in Nodes |-> 0]]
    /\ msgs = {}
    /\ pending = [n \in Nodes |-> NONEREAD]
    /\ lastServed = [n \in Nodes |-> 0]
    /\ servedTarget = [n \in Nodes |-> 0]
    /\ batchOn = [n \in Nodes |-> FALSE]
    /\ formation = [n \in Nodes |-> 0]
    /\ boundary = [n \in Nodes |-> 0]

(* ---- stability: majority incl leader leg, designated, current term ---- *)
DesignatedGrantor(n, r, m) ==
    \/ Weaken = "generation"
    \/ m \in Designated(r.gen)
    \/ m = leaderOf[r.term]

GrantsFor(n, r) ==
    {m \in Nodes :
        /\ hPhase[m][n] = "renewed"
        /\ hRoster[m][n] = r
        \* Fail closed on stalls: a hold counts only while its own
        \* deadline still covers now. An unprocessed-expired hold (tick
        \* starved, long pause) authorizes nothing — the holder loses
        \* fast-path authority instead of retaining stale authority.
        \* Bounded clock-rate drift is the only timing assumption; no
        \* scheduler/tick fairness is assumed anywhere here.
        /\ now < hDeadline[m][n]
        /\ DesignatedGrantor(n, r, m)}

KnownHas(n, r) ==
    /\ r.auth = auth
    /\ r.term = term[n]
    /\ r.gen \in known[n][r.term]

StableHere(n, r) ==
    /\ KnownHas(n, r)
    /\ Cardinality(GrantsFor(n, r)) >= Majority
    /\ \E m \in GrantsFor(n, r) : m = leaderOf[r.term]

CoversReader(n, r) ==
    \/ Weaken = "generation"
    \/ n \in Designated(r.gen)
    \/ n = leaderOf[r.term]

\* Majority-th smallest floor contribution (Majority=2): the least value
\* with at least Majority contributors at or below it (multiset-exact:
\* duplicate contributions count separately through the domain set).
FloorOf(n, r) ==
    LET D == GrantsFor(n, r)
        F == [m \in D |-> hFloor[m][n]]
    IN CHOOSE v \in 0..MaxLog :
        /\ Cardinality({m \in D : F[m] <= v}) >= Majority
        /\ \A w \in 0..MaxLog :
            (Cardinality({m \in D : F[m] <= w}) >= Majority) => v <= w

CoveredSet(n) == {m \in Nodes :
    /\ gPhase[n][m] \in {"renewing", "revoking"}
    \* Fail closed on stalls, grantor side: an exclusion covers only
    \* while its own deadline still covers now. The asymmetric deadlines
    \* (exclusion outlasts hold) plus drift-bounded clocks keep this
    \* consistent with the grantee side: a grantor never drops coverage
    \* while the grantee can still serve.
    /\ now < gDeadline[n][m]}

Quarantined(n) ==
    /\ Weaken /= "quarantine"
    /\ now < quar[n]

GrantGate(n) == ~Quarantined(n) /\ ~down[n]

(* ---- time ---- *)
Tick ==
    /\ now < MaxTime
    /\ now' = now + 1
    /\ UNCHANGED <<term, auth, inc, quar, down, part,
                   accepted, applied, commitIdx, completeIdx,
                   leaderOf, termStart, gen, known,
                   gPhase, gRoster, gAttempt, gSeq, gDeadline, gDue, gUnacked,
                   hPhase, hRoster, hAttempt, hSeq, hDeadline, hFloor,
                   msgs, pending, lastServed, servedTarget,
                   batchOn, formation, boundary>>

Done == /\ now = MaxTime
        /\ UNCHANGED vars

(* ---- network faults ---- *)
DropMsg ==
    /\ msgs /= {}
    /\ \E m \in msgs :
        msgs' = msgs \ {m}
    /\ UNCHANGED <<now, term, auth, inc, quar, down, part,
                   accepted, applied, commitIdx, completeIdx,
                   leaderOf, termStart, gen, known,
                   gPhase, gRoster, gAttempt, gSeq, gDeadline, gDue, gUnacked,
                   hPhase, hRoster, hAttempt, hSeq, hDeadline, hFloor,
                   pending, lastServed, servedTarget,
                   batchOn, formation, boundary>>

Partition(n, m) ==
    /\ n /= m
    /\ part' = [a \in Nodes |-> [b \in Nodes |->
        IF {a, b} = {n, m} THEN TRUE ELSE part[a][b]]]
    /\ UNCHANGED <<now, term, auth, inc, quar, down,
                   accepted, applied, commitIdx, completeIdx,
                   leaderOf, termStart, gen, known,
                   gPhase, gRoster, gAttempt, gSeq, gDeadline, gDue, gUnacked,
                   hPhase, hRoster, hAttempt, hSeq, hDeadline, hFloor,
                   msgs, pending, lastServed, servedTarget,
                   batchOn, formation, boundary>>

Heal(n, m) ==
    /\ n /= m
    /\ part' = [a \in Nodes |-> [b \in Nodes |->
        IF {a, b} = {n, m} THEN FALSE ELSE part[a][b]]]
    /\ UNCHANGED <<now, term, auth, inc, quar, down,
                   accepted, applied, commitIdx, completeIdx,
                   leaderOf, termStart, gen, known,
                   gPhase, gRoster, gAttempt, gSeq, gDeadline, gDue, gUnacked,
                   hPhase, hRoster, hAttempt, hSeq, hDeadline, hFloor,
                   msgs, pending, lastServed, servedTarget,
                   batchOn, formation, boundary>>

(* ---- crash / sat-out ---- *)
(* ---- crash: volatile pairing state is lost, incarnation rises ----
   Matches the implementation: tick lapse removes records, and a
   restart recreates the engine empty (hence owes the takeover fence
   before completing — the hub sets takeover_until on a fresh engine).
   Durable Raft state (accepted/applied/commit/complete) survives;
   everything else this node held or granted is forgotten. *)
Crash(n) ==
    /\ ~down[n]
    /\ inc[n] < 2
    /\ inc' = [inc EXCEPT ![n] = @ + 1]
    /\ quar' = [quar EXCEPT ![n] = now + Quarantine]
    /\ termStart' = [termStart EXCEPT ![n] = now]
    /\ gPhase' = [a \in Nodes |-> [b \in Nodes |-> IF a = n THEN "idle" ELSE gPhase[a][b]]]
    /\ gAttempt' = [a \in Nodes |-> [b \in Nodes |-> IF a = n THEN 0 ELSE gAttempt[a][b]]]
    /\ gSeq' = [a \in Nodes |-> [b \in Nodes |-> IF a = n THEN 0 ELSE gSeq[a][b]]]
    /\ gDeadline' = [a \in Nodes |-> [b \in Nodes |-> IF a = n THEN 0 ELSE gDeadline[a][b]]]
    /\ gDue' = [a \in Nodes |-> [b \in Nodes |-> IF a = n THEN 0 ELSE gDue[a][b]]]
    /\ gUnacked' = [a \in Nodes |-> [b \in Nodes |-> IF a = n THEN 0 ELSE gUnacked[a][b]]]
    /\ hPhase' = [a \in Nodes |-> [b \in Nodes |-> IF b = n THEN "none" ELSE hPhase[a][b]]]
    /\ hAttempt' = [a \in Nodes |-> [b \in Nodes |-> IF b = n THEN 0 ELSE hAttempt[a][b]]]
    /\ hSeq' = [a \in Nodes |-> [b \in Nodes |-> IF b = n THEN 0 ELSE hSeq[a][b]]]
    /\ hDeadline' = [a \in Nodes |-> [b \in Nodes |-> IF b = n THEN 0 ELSE hDeadline[a][b]]]
    /\ hFloor' = [a \in Nodes |-> [b \in Nodes |-> IF b = n THEN 0 ELSE hFloor[a][b]]]
    /\ known' = [m \in Nodes |-> IF m = n THEN [t \in Terms |-> {}] ELSE known[m]]
    /\ UNCHANGED <<now, term, auth, down, part,
                   accepted, applied, commitIdx, completeIdx,
                   leaderOf, gen,
                   gRoster, hRoster,
                   msgs, pending, lastServed, servedTarget,
                   batchOn, formation, boundary>>

GoDown(n) ==
    /\ ~down[n]
    /\ down' = [down EXCEPT ![n] = TRUE]
    /\ UNCHANGED <<now, term, auth, inc, quar, part,
                   accepted, applied, commitIdx, completeIdx,
                   leaderOf, termStart, gen, known,
                   gPhase, gRoster, gAttempt, gSeq, gDeadline, gDue, gUnacked,
                   hPhase, hRoster, hAttempt, hSeq, hDeadline, hFloor,
                   msgs, pending, lastServed, servedTarget,
                   batchOn, formation, boundary>>

ComeUp(n) ==
    /\ down[n]
    /\ down' = [down EXCEPT ![n] = FALSE]
    /\ UNCHANGED <<now, term, auth, inc, quar, part,
                   accepted, applied, commitIdx, completeIdx,
                   leaderOf, termStart, gen, known,
                   gPhase, gRoster, gAttempt, gSeq, gDeadline, gDue, gUnacked,
                   hPhase, hRoster, hAttempt, hSeq, hDeadline, hFloor,
                   msgs, pending, lastServed, servedTarget,
                   batchOn, formation, boundary>>

(* ---- election: L rises to term 2 with its connected majority ---- *)
ConnectedSet(L) == {m \in Nodes : ~part[L][m] /\ ~down[m]} \cup {L}

Elect(L) ==
    /\ ~down[L]
    /\ Cardinality(ConnectedSet(L)) >= Majority
    /\ leaderOf' = [leaderOf EXCEPT ![2] = L]
    /\ term' = [m \in Nodes |-> IF m \in ConnectedSet(L) THEN 2 ELSE term[m]]
    /\ termStart' = [m \in Nodes |-> IF m \in ConnectedSet(L) THEN now ELSE termStart[m]]
    \* revoke stale outgoing on every observer (fast path)
    /\ gPhase' = [a \in Nodes |-> [b \in Nodes |->
        IF /\ a \in ConnectedSet(L) /\ Weaken /= "revocation"
           /\ gPhase[a][b] = "renewing" /\ gRoster[a][b].term = 1
        THEN "revoking" ELSE gPhase[a][b]]]
    /\ msgs' = (msgs \cup {[type |-> "revoke", from |-> p[1], to |-> p[2],
                             roster |-> gRoster[p[1]][p[2]], attempt |-> gAttempt[p[1]][p[2]],
                             seq |-> gSeq[p[1]][p[2]], thresh |-> 0, committed |-> 0] :
                            p \in {q \in ConnectedSet(L) \X Nodes :
                                /\ gPhase[q[1]][q[2]] = "renewing"
                                /\ gRoster[q[1]][q[2]].term = 1
                                /\ Weaken /= "revocation"}})
    /\ UNCHANGED <<now, auth, inc, quar, down, part,
                   accepted, applied, commitIdx, completeIdx,
                   gen, known,
                   gRoster, gAttempt, gSeq, gDeadline, gDue, gUnacked,
                   hPhase, hRoster, hAttempt, hSeq, hDeadline, hFloor,
                   pending, lastServed, servedTarget,
                   batchOn, formation, boundary>>

(* ---- roster announce (leader-only, next generation) ---- *)
Announce(n) ==
    /\ ~down[n]
    /\ GrantGate(n)
    /\ n = leaderOf[term[n]]
    /\ gen[term[n]] < MaxGen
    /\ LET g == gen[term[n]] + 1
            r == RosterOf(auth, term[n], g)
            \* A skipped Guard is deferred, never dropped (matches
            \* open_guard): pairings at the sequence cap wait for budget.
            targets == {m \in Nodes :
                         /\ gSeq[n][m] < MaxSeq
                         /\ \/ Weaken = "revocation"
                            \/ gPhase[n][m] \notin {"renewing", "revoking"}}
       IN /\ gen' = [gen EXCEPT ![term[n]] = g]
          /\ known' = [known EXCEPT ![n] = [@ EXCEPT ![term[n]] = {g}]]
          /\ gPhase' = [a \in Nodes |-> [b \in Nodes |->
                IF /\ a = n /\ Weaken /= "revocation"
                   /\ gPhase[a][b] = "renewing" /\ gRoster[a][b].term = term[n]
                THEN "revoking"
                ELSE IF /\ a = n /\ b \in targets
                     THEN "guarding"
                     ELSE gPhase[a][b]]]
          /\ gRoster' = [a \in Nodes |-> [b \in Nodes |->
                IF /\ a = n /\ b \in targets THEN r ELSE gRoster[a][b]]]
          \* Fresh attempt per Guard, continuing this pairing's sequence
          \* (matches open_guard's (incarnation, out_seq)): a re-guard of
          \* a new generation carries a higher sequence under the same
          \* incarnation, so grantees adopt it instead of rejecting it as
          \* a duplicate of the old generation's Guard.
          /\ gAttempt' = [a \in Nodes |-> [b \in Nodes |->
                IF /\ a = n /\ b \in targets THEN AttemptId(inc[n], gSeq[a][b] + 1)
                ELSE gAttempt[a][b]]]
          /\ gSeq' = [a \in Nodes |-> [b \in Nodes |->
                IF /\ a = n /\ b \in targets THEN gSeq[a][b] + 1 ELSE gSeq[a][b]]]
          /\ gDeadline' = [a \in Nodes |-> [b \in Nodes |->
                IF /\ a = n /\ b \in targets
                THEN now + GuardWin + Allow ELSE gDeadline[a][b]]]
          /\ gDue' = gDue
          /\ gUnacked' = gUnacked
          /\ msgs' = (msgs
                \cup {[type |-> "revoke", from |-> n, to |-> p,
                       roster |-> gRoster[n][p], attempt |-> gAttempt[n][p],
                       seq |-> gSeq[n][p], thresh |-> 0, committed |-> 0] :
                      p \in {x \in Nodes : gPhase[n][x] = "renewing"
                                          /\ gRoster[n][x].term = term[n]
                                          /\ Weaken /= "revocation"}}
                \cup {[type |-> "guard", from |-> n, to |-> m,
                       roster |-> r, attempt |-> AttemptId(inc[n], gSeq[n][m] + 1),
                       seq |-> gSeq[n][m] + 1, thresh |-> accepted[n], committed |-> 0] :
                      m \in targets})
    /\ UNCHANGED <<now, term, auth, inc, quar, down, part,
                   accepted, applied, commitIdx, completeIdx,
                   leaderOf, termStart,
                   hPhase, hRoster, hAttempt, hSeq, hDeadline, hFloor,
                   pending, lastServed, servedTarget,
                   batchOn, formation, boundary>>

(* ---- guard receipt with auto-grant closure ---- *)
DeliverGuard(m) ==
    /\ m \in msgs
    /\ m.type = "guard"
    /\ ~part[m.from][m.to]
    /\ ~down[m.to]
    /\ LET n == m.to
           g == m.from
           r == m.roster
       IN /\ GrantGate(n)
          /\ r.auth = auth
           /\ r.term >= term[n]
           \* Freshness is lexicographic (incarnation, sequence) — the
           \* single-nat attempt preserves that order numerically, so a
           \* re-guard under the same incarnation (new generation, same
           \* lineage) is accepted iff its attempt advances.
           /\ \/ hPhase[g][n] = "none"
              \/ m.attempt > hAttempt[g][n]
          /\ hPhase' = [hPhase EXCEPT ![g][n] = "guarded"]
          /\ hRoster' = [hRoster EXCEPT ![g][n] = r]
          /\ hAttempt' = [hAttempt EXCEPT ![g][n] = m.attempt]
          /\ hSeq' = [hSeq EXCEPT ![g][n] = m.seq]
          /\ hDeadline' = [hDeadline EXCEPT ![g][n] = now + GuardWin - Allow]
          /\ hFloor' = [hFloor EXCEPT ![g][n] = m.thresh]
          \* supersede-revoke own older same-lineage outgoing (fast path)
          /\ gPhase' = [a \in Nodes |-> [b \in Nodes |->
                IF /\ a = n /\ Weaken /= "revocation"
                   /\ gPhase[a][b] = "renewing"
                   /\ gRoster[a][b].auth = r.auth /\ gRoster[a][b].term = r.term
                   /\ gRoster[a][b].gen < r.gen
                THEN "revoking" ELSE gPhase[a][b]]]
          \* generation prune: only the newest learned generation counts
          /\ known' = IF Weaken = "generation" THEN known
                      ELSE [known EXCEPT ![n] = [@ EXCEPT ![r.term] =
                             {q \in @ \cup {r.gen} : q >= r.gen}]]
          /\ msgs' = ((msgs \ {m})
                \cup {[type |-> "guardreply", from |-> n, to |-> g,
                       roster |-> r, attempt |-> m.attempt,
                       seq |-> m.seq, thresh |-> 0, committed |-> 0]})
                      \cup {[type |-> "guard", from |-> n, to |-> x,
                              roster |-> r, attempt |-> AttemptId(inc[n], 1),
                              seq |-> 1, thresh |-> accepted[n], committed |-> 0] :
                             x \in ({y \in Nodes :
                                    /\ (y = leaderOf[term[n]] \/ y \in Designated(r.gen))
                                    /\ gPhase[n][y] \notin {"renewing", "revoking"}}
                                    \ {g})}
                      \cup {[type |-> "revoke", from |-> n, to |-> p,
                              roster |-> gRoster[n][p], attempt |-> gAttempt[n][p],
                              seq |-> gSeq[n][p], thresh |-> 0, committed |-> 0] :
                             p \in {z \in Nodes : gPhase[n][z] = "renewing"
                                    /\ Weaken /= "revocation"
                                    /\ gRoster[n][z].auth = r.auth
                                    /\ gRoster[n][z].term = r.term
                                    /\ gRoster[n][z].gen < r.gen}}
    /\ UNCHANGED <<now, term, auth, inc, quar, down, part,
                   accepted, applied, commitIdx, completeIdx,
                   leaderOf, termStart, gen,
                   gRoster, gAttempt, gSeq, gDeadline, gDue, gUnacked,
                   pending, lastServed, servedTarget,
                   batchOn, formation, boundary>>

(* ---- guard-reply receipt: exact attempt+seq promotes ----
   Promotion consumes the Guarding phase, so a stale duplicate reply
   can never promote twice; each Guard carries a fresh continuing
   attempt, so a late reply to a superseded Guard never matches. *)
DeliverGuardReply(m) ==
    /\ m \in msgs
    /\ m.type = "guardreply"
    /\ ~part[m.from][m.to]
    /\ ~down[m.to]
    /\ LET n == m.to
           g == m.from
       IN /\ gPhase[n][g] = "guarding"
          /\ gRoster[n][g] = m.roster
          /\ (Weaken = "attempt" \/ m.attempt = gAttempt[n][g])
          /\ (Weaken = "attempt" \/ m.seq = gSeq[n][g])
          /\ now < gDeadline[n][g]
          /\ gSeq[n][g] < MaxSeq
          /\ gPhase' = [gPhase EXCEPT ![n][g] = "renewing"]
          /\ gSeq' = [gSeq EXCEPT ![n][g] = @ + 1]
          /\ gDeadline' = [gDeadline EXCEPT ![n][g] = now + GuardWin + Allow + Lease + Allow]
          /\ gDue' = [gDue EXCEPT ![n][g] = now]
          /\ gUnacked' = [gUnacked EXCEPT ![n][g] = 0]
          /\ msgs' = (msgs \ {m}) \cup {[type |-> "renew", from |-> n, to |-> g,
                                   roster |-> gRoster[n][g], attempt |-> gAttempt[n][g],
                                   seq |-> gSeq[n][g] + 1,
                                   thresh |-> 0, committed |-> commitIdx]}
    /\ UNCHANGED <<now, term, auth, inc, quar, down, part,
                   accepted, applied, commitIdx, completeIdx,
                   leaderOf, termStart, gen, known,
                   gRoster, gAttempt,
                   hPhase, hRoster, hAttempt, hSeq, hDeadline, hFloor,
                   pending, lastServed, servedTarget,
                   batchOn, formation, boundary>>

(* ---- renew receipt: exact attempt, advancing seq, in-window ---- *)
DeliverRenew(m) ==
    /\ m \in msgs
    /\ m.type = "renew"
    /\ ~part[m.from][m.to]
    /\ ~down[m.to]
    /\ LET n == m.to
           g == m.from
       IN /\ hRoster[g][n] = m.roster
          /\ (Weaken = "attempt" \/ m.attempt = hAttempt[g][n])
          /\ (Weaken = "attempt" \/ m.seq > hSeq[g][n])
          /\ hPhase[g][n] \in {"guarded", "renewed"}
          /\ now < hDeadline[g][n]
          /\ hPhase' = [hPhase EXCEPT ![g][n] = "renewed"]
          /\ hSeq' = [hSeq EXCEPT ![g][n] = m.seq]
          /\ hFloor' = [hFloor EXCEPT ![g][n] =
                IF hFloor[g][n] > m.committed THEN hFloor[g][n] ELSE m.committed]
          /\ hDeadline' = [hDeadline EXCEPT ![g][n] = now + Lease - Allow]
          /\ msgs' = (msgs \ {m}) \cup {[type |-> "renewreply", from |-> n, to |-> g,
                                   roster |-> m.roster, attempt |-> m.attempt,
                                   seq |-> m.seq, thresh |-> 0, committed |-> 0]}
    /\ UNCHANGED <<now, term, auth, inc, quar, down, part,
                   accepted, applied, commitIdx, completeIdx,
                   leaderOf, termStart, gen, known,
                   gPhase, gRoster, gAttempt, gSeq, gDeadline, gDue, gUnacked,
                   hRoster, hAttempt,
                   pending, lastServed, servedTarget,
                   batchOn, formation, boundary>>

(* ---- renew-reply receipt: exact match unmutes ---- *)
DeliverRenewReply(m) ==
    /\ m \in msgs
    /\ m.type = "renewreply"
    /\ ~part[m.from][m.to]
    /\ ~down[m.to]
    /\ LET n == m.to
           g == m.from
       IN /\ gPhase[n][g] = "renewing"
          /\ gRoster[n][g] = m.roster
          /\ (Weaken = "attempt" \/ m.attempt = gAttempt[n][g])
          /\ (Weaken = "attempt" \/ m.seq = gSeq[n][g])
           /\ gUnacked' = [gUnacked EXCEPT ![n][g] = 0]
           /\ msgs' = msgs \ {m}
    /\ UNCHANGED <<now, term, auth, inc, quar, down, part,
                   accepted, applied, commitIdx, completeIdx,
                   leaderOf, termStart, gen, known,
                   gPhase, gRoster, gAttempt, gSeq, gDeadline, gDue,
                   hPhase, hRoster, hAttempt, hSeq, hDeadline, hFloor,
                   pending, lastServed, servedTarget,
                   batchOn, formation, boundary>>

(* ---- revoke receipt: fast drop + reply ---- *)
DeliverRevoke(m) ==
    /\ m \in msgs
    /\ m.type = "revoke"
    /\ ~part[m.from][m.to]
    /\ ~down[m.to]
    /\ LET n == m.to
           g == m.from
       IN /\ hPhase[g][n] \in {"guarded", "renewed"}
          /\ hRoster[g][n] = m.roster
           /\ hPhase' = [hPhase EXCEPT ![g][n] = "none"]
           /\ msgs' = (msgs \ {m}) \cup {[type |-> "revokereply", from |-> n, to |-> g,
                                   roster |-> m.roster, attempt |-> m.attempt,
                                   seq |-> m.seq, thresh |-> 0, committed |-> 0]}
    /\ UNCHANGED <<now, term, auth, inc, quar, down, part,
                   accepted, applied, commitIdx, completeIdx,
                   leaderOf, termStart, gen, known,
                   gPhase, gRoster, gAttempt, gSeq, gDeadline, gDue, gUnacked,
                   hRoster, hAttempt, hSeq, hDeadline, hFloor,
                   msgs, pending, lastServed, servedTarget,
                   batchOn, formation, boundary>>

(* ---- revoke-reply receipt: prompt clear ---- *)
DeliverRevokeReply(m) ==
    /\ m \in msgs
    /\ m.type = "revokereply"
    /\ ~part[m.from][m.to]
    /\ ~down[m.to]
    /\ LET n == m.to
           g == m.from
       IN /\ gPhase[n][g] = "revoking"
          /\ gRoster[n][g] = m.roster
          /\ (Weaken = "attempt" \/ m.attempt = gAttempt[n][g])
           /\ (Weaken = "attempt" \/ m.seq = gSeq[n][g])
           /\ gPhase' = [gPhase EXCEPT ![n][g] = "idle"]
           /\ msgs' = msgs \ {m}
    /\ UNCHANGED <<now, term, auth, inc, quar, down, part,
                   accepted, applied, commitIdx, completeIdx,
                   leaderOf, termStart, gen, known,
                   gRoster, gAttempt, gSeq, gDeadline, gDue, gUnacked,
                   hPhase, hRoster, hAttempt, hSeq, hDeadline, hFloor,
                   pending, lastServed, servedTarget,
                   batchOn, formation, boundary>>

(* ---- grantor timers ---- *)
GrantorTick(n, m) ==
    /\ ~down[n]
    /\ gPhase[n][m] = "renewing"
    /\ now >= gDue[n][m]
    /\ gDue' = [gDue EXCEPT ![n][m] = now + RenewEvery]
    /\ gSeq[n][m] < MaxSeq
    /\ IF gUnacked[n][m] < MuteAfter
       THEN /\ gSeq' = [gSeq EXCEPT ![n][m] = @ + 1]
            /\ gUnacked' = [gUnacked EXCEPT ![n][m] = @ + 1]
            /\ gDeadline' = [gDeadline EXCEPT ![n][m] = now + Lease + Allow]
            /\ msgs' = msgs \cup {[type |-> "renew", from |-> n, to |-> m,
                                    roster |-> gRoster[n][m], attempt |-> gAttempt[n][m],
                                    seq |-> gSeq[n][m] + 1,
                                    thresh |-> 0, committed |-> commitIdx]}
       ELSE UNCHANGED <<msgs, gSeq, gUnacked, gDeadline>>
    /\ UNCHANGED <<now, term, auth, inc, quar, down, part,
                   accepted, applied, commitIdx, completeIdx,
                   leaderOf, termStart, gen, known,
                   gPhase, gRoster, gAttempt,
                   hPhase, hRoster, hAttempt, hSeq, hDeadline, hFloor,
                   pending, lastServed, servedTarget,
                   batchOn, formation, boundary>>

GuardTimeout(n, m) ==
    /\ ~down[n]
    /\ gPhase[n][m] = "guarding"
    /\ now >= gDeadline[n][m]
    /\ gSeq[n][m] < MaxSeq
    /\ gAttempt' = [gAttempt EXCEPT ![n][m] = AttemptId(inc[n], gSeq[n][m] + 1)]
    /\ gSeq' = [gSeq EXCEPT ![n][m] = @ + 1]
    /\ gDeadline' = [gDeadline EXCEPT ![n][m] = now + GuardWin + Allow]
    /\ msgs' = (msgs \cup {[type |-> "guard", from |-> n, to |-> m,
                            roster |-> gRoster[n][m], attempt |-> AttemptId(inc[n], gSeq[n][m] + 1),
                            seq |-> gSeq[n][m] + 1,
                            thresh |-> accepted[n], committed |-> 0]})
    /\ UNCHANGED <<now, term, auth, inc, quar, down, part,
                   accepted, applied, commitIdx, completeIdx,
                   leaderOf, termStart, gen, known,
                   gPhase, gRoster, gDue, gUnacked,
                   hPhase, hRoster, hAttempt, hSeq, hDeadline, hFloor,
                   pending, lastServed, servedTarget,
                   batchOn, formation, boundary>>

\* Driver repair: re-open a Guard where no live pairing exists
\* (models grant_for idempotent retry after revocations lapse).
RepairGuard(n, m) ==
    /\ ~down[n]
    /\ GrantGate(n)
    /\ gPhase[n][m] \in {"idle"}
    /\ \E r \in {[auth |-> auth, term |-> term[n], gen |-> g] : g \in known[n][term[n]]} :
        /\ (m = leaderOf[term[n]] \/ m \in Designated(r.gen))
        /\ gSeq[n][m] < MaxSeq
        /\ gPhase' = [gPhase EXCEPT ![n][m] = "guarding"]
        /\ gRoster' = [gRoster EXCEPT ![n][m] = r]
        /\ gAttempt' = [gAttempt EXCEPT ![n][m] = AttemptId(inc[n], gSeq[n][m] + 1)]
        /\ gSeq' = [gSeq EXCEPT ![n][m] = @ + 1]
        /\ gDeadline' = [gDeadline EXCEPT ![n][m] = now + GuardWin + Allow]
        /\ gDue' = gDue
        /\ gUnacked' = [gUnacked EXCEPT ![n][m] = 0]
        /\ msgs' = msgs \cup {[type |-> "guard", from |-> n, to |-> m,
                                roster |-> r, attempt |-> AttemptId(inc[n], gSeq[n][m] + 1),
                                seq |-> gSeq[n][m] + 1,
                                thresh |-> accepted[n], committed |-> 0]}
    /\ UNCHANGED <<now, term, auth, inc, quar, down, part,
                   accepted, applied, commitIdx, completeIdx,
                   leaderOf, termStart, gen, known,
                   hPhase, hRoster, hAttempt, hSeq, hDeadline, hFloor,
                   pending, lastServed, servedTarget,
                   batchOn, formation, boundary>>

ExclLapse(n, m) ==
    /\ gPhase[n][m] \in {"renewing", "revoking"}
    /\ now >= gDeadline[n][m]
    /\ gPhase' = [gPhase EXCEPT ![n][m] = "idle"]
    /\ UNCHANGED <<now, term, auth, inc, quar, down, part,
                   accepted, applied, commitIdx, completeIdx,
                   leaderOf, termStart, gen, known,
                   gRoster, gAttempt, gSeq, gDeadline, gDue, gUnacked,
                   hPhase, hRoster, hAttempt, hSeq, hDeadline, hFloor,
                   msgs, pending, lastServed, servedTarget,
                   batchOn, formation, boundary>>

HoldLapse(g, n) ==
    /\ hPhase[g][n] \in {"guarded", "renewed"}
    /\ now >= hDeadline[g][n]
    /\ hPhase' = [hPhase EXCEPT ![g][n] = "none"]
    /\ UNCHANGED <<now, term, auth, inc, quar, down, part,
                   accepted, applied, commitIdx, completeIdx,
                   leaderOf, termStart, gen, known,
                   gPhase, gRoster, gAttempt, gSeq, gDeadline, gDue, gUnacked,
                   hRoster, hAttempt, hSeq, hDeadline, hFloor,
                   msgs, pending, lastServed, servedTarget,
                   batchOn, formation, boundary>>

(* ---- replication ---- *)
Replicate(n) ==
    /\ ~down[n]
    /\ accepted[n] < commitIdx
    /\ accepted' = [accepted EXCEPT ![n] = @ + 1]
    /\ UNCHANGED <<now, term, auth, inc, quar, down, part,
                   applied, commitIdx, completeIdx,
                   leaderOf, termStart, gen, known,
                   gPhase, gRoster, gAttempt, gSeq, gDeadline, gDue, gUnacked,
                   hPhase, hRoster, hAttempt, hSeq, hDeadline, hFloor,
                   msgs, pending, lastServed, servedTarget,
                   batchOn, formation, boundary>>

ApplyStep(n) ==
    /\ ~down[n]
    /\ applied[n] < accepted[n]
    /\ applied' = [applied EXCEPT ![n] = @ + 1]
    /\ UNCHANGED <<now, term, auth, inc, quar, down, part,
                   accepted, commitIdx, completeIdx,
                   leaderOf, termStart, gen, known,
                   gPhase, gRoster, gAttempt, gSeq, gDeadline, gDue, gUnacked,
                   hPhase, hRoster, hAttempt, hSeq, hDeadline, hFloor,
                   msgs, pending, lastServed, servedTarget,
                   batchOn, formation, boundary>>

WriteInternal(n) ==
    /\ ~down[n]
    /\ n = leaderOf[term[n]]
    /\ commitIdx < MaxLog
    /\ commitIdx' = commitIdx + 1
    /\ accepted' = [accepted EXCEPT ![n] = commitIdx + 1]
    /\ UNCHANGED <<now, term, auth, inc, quar, down, part,
                   applied, completeIdx,
                   leaderOf, termStart, gen, known,
                   gPhase, gRoster, gAttempt, gSeq, gDeadline, gDue, gUnacked,
                   hPhase, hRoster, hAttempt, hSeq, hDeadline, hFloor,
                   msgs, pending, lastServed, servedTarget,
                   batchOn, formation, boundary>>

CompleteWrite(n) ==
    /\ ~down[n]
    /\ n = leaderOf[term[n]]
    /\ completeIdx < commitIdx
    /\ \A m \in CoveredSet(n) : applied[m] >= commitIdx
    /\ Weaken = "takeover" \/ now - termStart[n] >= TakeoverFence
    /\ completeIdx' = commitIdx
    /\ UNCHANGED <<now, term, auth, inc, quar, down, part,
                   accepted, applied, commitIdx,
                   leaderOf, termStart, gen, known,
                   gPhase, gRoster, gAttempt, gSeq, gDeadline, gDue, gUnacked,
                   hPhase, hRoster, hAttempt, hSeq, hDeadline, hFloor,
                   msgs, pending, lastServed, servedTarget,
                   batchOn, formation, boundary>>

(* ---- authority transition: fresh lineage, everything voids ----
   In-flight reads and batches die with the lineage (the implementation
   retires the serving front, so no batch outlives its authority), and
   every in-flight message names the retired lineage: Guards check
   authority equality and all other handlers key on roster identities
   that no longer exist anywhere, so dropping the soup here changes no
   observable behavior and keeps the state space finite. *)
AuthBump ==
    /\ auth < 1
    /\ auth' = auth + 1
    /\ commitIdx' = 0
    /\ completeIdx' = 0
    /\ applied' = [n \in Nodes |-> 0]
    /\ accepted' = [n \in Nodes |-> 0]
    /\ pending' = [n \in Nodes |-> NONEREAD]
    /\ batchOn' = [n \in Nodes |-> FALSE]
    /\ formation' = [n \in Nodes |-> 0]
    /\ boundary' = [n \in Nodes |-> 0]
    /\ msgs' = {}
    /\ gPhase' = [n \in Nodes |-> [m \in Nodes |-> "idle"]]
    /\ hPhase' = [n \in Nodes |-> [m \in Nodes |-> "none"]]
    /\ known' = [n \in Nodes |-> [t \in Terms |-> {}]]
    /\ gen' = [t \in Terms |-> 0]
    /\ gAttempt' = [n \in Nodes |-> [m \in Nodes |-> 0]]
    /\ gSeq' = [n \in Nodes |-> [m \in Nodes |-> 0]]
    /\ gDeadline' = [n \in Nodes |-> [m \in Nodes |-> 0]]
    /\ gDue' = [n \in Nodes |-> [m \in Nodes |-> 0]]
    /\ gUnacked' = [n \in Nodes |-> [m \in Nodes |-> 0]]
    /\ hAttempt' = [n \in Nodes |-> [m \in Nodes |-> 0]]
    /\ hSeq' = [n \in Nodes |-> [m \in Nodes |-> 0]]
    /\ hDeadline' = [n \in Nodes |-> [m \in Nodes |-> 0]]
    /\ hFloor' = [n \in Nodes |-> [m \in Nodes |-> 0]]
    /\ UNCHANGED <<now, term, inc, quar, down, part,
                   leaderOf, termStart,
                   gRoster, hRoster,
                   lastServed, servedTarget>>

(* ---- local reads ---- *)
LocalReadStart(n) ==
    /\ ~down[n]
    /\ pending[n] = NONEREAD
    /\ ~batchOn[n]
    \* Majority hint: a local serve needs a majority of renewed,
    \* still-valid holds (plus leader leg, designation, and known
    \* roster at finish), so starting below that can never finish —
    \* it only churns pending. Pruning dead starts loses no
    \* violation: every finishable read passes this gate.
    /\ Cardinality({m \in Nodes :
        /\ hPhase[m][n] = "renewed"
        /\ now < hDeadline[m][n]}) >= Majority
    /\ pending' = [pending EXCEPT ![n] = completeIdx]
    /\ UNCHANGED <<now, term, auth, inc, quar, down, part,
                   accepted, applied, commitIdx, completeIdx,
                   leaderOf, termStart, gen, known,
                   gPhase, gRoster, gAttempt, gSeq, gDeadline, gDue, gUnacked,
                   hPhase, hRoster, hAttempt, hSeq, hDeadline, hFloor,
                   msgs, lastServed, servedTarget,
                   batchOn, formation, boundary>>

LocalReadFinish(n, r) ==
    /\ ~down[n]
    /\ pending[n] /= NONEREAD
    /\ StableHere(n, r)
    /\ CoversReader(n, r)
    /\ Weaken = "floor" \/ applied[n] >= FloorOf(n, r)
    /\ lastServed' = [lastServed EXCEPT ![n] = applied[n]]
    /\ servedTarget' = [servedTarget EXCEPT ![n] = pending[n]]
    /\ pending' = [pending EXCEPT ![n] = NONEREAD]
    /\ UNCHANGED <<now, term, auth, inc, quar, down, part,
                   accepted, applied, commitIdx, completeIdx,
                   leaderOf, termStart, gen, known,
                   gPhase, gRoster, gAttempt, gSeq, gDeadline, gDue, gUnacked,
                   hPhase, hRoster, hAttempt, hSeq, hDeadline, hFloor,
                   msgs, batchOn, formation, boundary>>

(* ---- Lazy-ALR reads ---- *)
AlrStart(n) ==
    /\ ~down[n]
    /\ pending[n] = NONEREAD
    /\ ~batchOn[n]
    /\ pending' = [pending EXCEPT ![n] = completeIdx]
    /\ formation' = [formation EXCEPT ![n] = applied[n]]
    /\ batchOn' = [batchOn EXCEPT ![n] = TRUE]
    \* Fresh boundary per batch (the coordinator always orders one;
    \* a stale boundary from an earlier batch never carries over).
    /\ boundary' = [boundary EXCEPT ![n] = 0]
    /\ UNCHANGED <<now, term, auth, inc, quar, down, part,
                   accepted, applied, commitIdx, completeIdx,
                   leaderOf, termStart, gen, known,
                   gPhase, gRoster, gAttempt, gSeq, gDeadline, gDue, gUnacked,
                   hPhase, hRoster, hAttempt, hSeq, hDeadline, hFloor,
                   msgs, lastServed, servedTarget, boundary>>

AlrBoundarySub(n) ==
    /\ batchOn[n]
    /\ ~down[n]
    /\ LET L == leaderOf[term[n]]
       IN /\ ~down[L] /\ ~part[n][L]
          /\ boundary' = [boundary EXCEPT ![n] = commitIdx]
    /\ UNCHANGED <<now, term, auth, inc, quar, down, part,
                   accepted, applied, commitIdx, completeIdx,
                   leaderOf, termStart, gen, known,
                   gPhase, gRoster, gAttempt, gSeq, gDeadline, gDue, gUnacked,
                   hPhase, hRoster, hAttempt, hSeq, hDeadline, hFloor,
                   msgs, pending, lastServed, servedTarget,
                   batchOn, formation>>

AlrBoundaryFence(n) ==
    /\ batchOn[n]
    /\ ~down[n]
    /\ LET L == leaderOf[term[n]]
       IN /\ ~down[L] /\ ~part[n][L] /\ L = n
          /\ commitIdx < MaxLog
          /\ \A m \in CoveredSet(n) : applied[m] >= commitIdx + 1
          /\ Weaken = "takeover" \/ now - termStart[n] >= TakeoverFence
          /\ commitIdx' = commitIdx + 1
          /\ accepted' = [accepted EXCEPT ![n] = commitIdx + 1]
          /\ boundary' = [boundary EXCEPT ![n] = commitIdx + 1]
    /\ UNCHANGED <<now, term, auth, inc, quar, down, part,
                   applied, completeIdx,
                   leaderOf, termStart, gen, known,
                   gPhase, gRoster, gAttempt, gSeq, gDeadline, gDue, gUnacked,
                   hPhase, hRoster, hAttempt, hSeq, hDeadline, hFloor,
                   msgs, pending, lastServed, servedTarget,
                   batchOn, formation>>

AlrFinish(n) ==
    /\ batchOn[n]
    /\ ~down[n]
    /\ pending[n] /= NONEREAD
    /\ applied[n] >= boundary[n]
    \* A boundary was ordered for this batch — except the vacuous case
    \* (nothing ever committed: pending is 0 and applied 0 covers it).
    /\ boundary[n] > 0 \/ commitIdx = 0
    /\ lastServed' = [lastServed EXCEPT ![n] = applied[n]]
    /\ servedTarget' = [servedTarget EXCEPT ![n] = pending[n]]
    /\ pending' = [pending EXCEPT ![n] = NONEREAD]
    /\ batchOn' = [batchOn EXCEPT ![n] = FALSE]
    /\ UNCHANGED <<now, term, auth, inc, quar, down, part,
                   accepted, applied, commitIdx, completeIdx,
                   leaderOf, termStart, gen, known,
                   gPhase, gRoster, gAttempt, gSeq, gDeadline, gDue, gUnacked,
                   hPhase, hRoster, hAttempt, hSeq, hDeadline, hFloor,
                   msgs, boundary, formation>>

(* ---- next-state relation ---- *)
Next ==
    \/ Tick
    \/ Done
    \/ ("drop" \in Flags /\ DropMsg)
    \/ \E n, m \in Nodes : ("part" \in Flags /\ (Partition(n, m) \/ Heal(n, m)))
    \/ \E n \in Nodes : (("crash" \in Flags /\ Crash(n))
        \/ ("down" \in Flags /\ (GoDown(n) \/ ComeUp(n))))
    \/ \E n \in Nodes : ("elect" \in Flags /\ Elect(n))
    \/ \E n \in Nodes : Announce(n)
    \/ \E m \in msgs : DeliverGuard(m) \/ DeliverGuardReply(m) \/ DeliverRenew(m)
    \/ \E m \in msgs : DeliverRenewReply(m) \/ DeliverRevoke(m) \/ DeliverRevokeReply(m)
    \/ \E n, m \in Nodes : GrantorTick(n, m) \/ GuardTimeout(n, m)
    \/ \E n, m \in Nodes : RepairGuard(n, m)
    \/ \E n, m \in Nodes : ExclLapse(n, m) \/ HoldLapse(n, m)
    \/ \E n \in Nodes : Replicate(n) \/ ApplyStep(n)
    \/ \E n \in Nodes : WriteInternal(n) \/ CompleteWrite(n)
    \/ AuthBump /\ "auth" \in Flags
    \/ \E n \in Nodes : LocalReadStart(n)
    \/ \E n \in Nodes : ("alr" \in Flags /\ AlrStart(n))
    \/ \E n \in Nodes : \E r \in {[auth |-> auth, term |-> term[n], gen |-> g] : g \in 1..MaxGen} :
        LocalReadFinish(n, r)
    \/ \E n \in Nodes : ("alr" \in Flags
        /\ (AlrBoundarySub(n) \/ AlrBoundaryFence(n) \/ AlrFinish(n)))

Spec == Init /\ [][Next]_vars

(* ---- invariants ---- *)
NoStaleRead == \A n \in Nodes : lastServed[n] >= servedTarget[n]

(* ---- state constraint: bounded channels ----
   Production transports bound in-flight messages (stream/buffer
   limits); overflow arrives as loss, which DropMsg already explores.
   The bound prunes only deep pileup, never protocol-distinguishable
   behavior: every minimal safety witness fits comfortably beneath it
   (activation bursts peak around a dozen; variants allow headroom for
   revoke/re-announce rounds), and every weakened variant below still
   exhibits its counterexample inside the bound — the canary proving
   the bound hides nothing load-bearing. *)
SoupBound == Cardinality(msgs) <= SoupCap

====
