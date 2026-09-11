---- MODULE directory ----
(*
  Phase-0 formal ownership model for Kivi's tablet directory.

  Models the essential authority lifecycle and its safety properties. Ranges
  are abstracted as sets of routing points: both hash prefixes (sets of
  partition hashes) and ordered intervals (sets of keys) are point sets for
  ownership purposes, so disjointness and single-authority arguments apply to
  both layouts. Layout-specific range arithmetic (prefix canonicity, interval
  overlap) is covered by Rust unit tests, not by this model.

  Correspondence to crates/kivi-tablet (exact, action by action):
    Bootstrap <-> DirectorySnapshot::bootstrap
    Allocate  <-> DirectorySnapshot::allocate
    Stage     <-> DirectorySnapshot::stage
    Activate  <-> DirectorySnapshot::activate
    Seal      <-> DirectorySnapshot::seal
    Retire    <-> DirectorySnapshot::retire
  Rust's validate() re-checks the state invariants below after every action;
  lookup determinism follows because routing reads only the snapshot.

  Deliberate abstractions (documented, not hidden):
    - Epochs are fixed to 1 (INITIAL) at allocation and immutable after,
      matching Rust, where generations are caller-supplied valid values and
      never mutated by directory transitions. Epoch *advance* exhaustion is
      specified separately by the CheckedNext operator below, mirroring
      TabletEpoch::next / WriteGuardGeneration::next / NodeIncarnation::next.
    - Write-guard generations behave identically to epochs here and share the
      same argument; only one generation counter is modeled.
    - Ranges must be nonempty, mirroring validated construction (every valid
      hash prefix and every valid ordered interval covers >= 1 point).

  TLC configuration sketch (no TLC run is wired into this stage):
    CONSTANTS TabletIds <- {t1, t2, t3}, Points <- {p1, p2, p3}, MaxGen <- 2
    INVARIANTS TypeOK NoOverlappingAuthority AtMostOneWritableAuthority
               TombstonesRedirectForward NoSilentSaturation
    PROPERTIES MonotonicEpoch FencedAuthorityCannotReturn VersionAdvancesByOne
*)

EXTENDS Naturals, FiniteSets, TLC

CONSTANTS TabletIds, Points, MaxGen

VARIABLES state, range, redirect, epoch, version

vars == <<state, range, redirect, epoch, version>>

States == {"absent", "allocated", "inactive", "active", "fenced", "tombstone"}

TypeOK ==
    /\ state \in [TabletIds -> States]
    /\ range \in [TabletIds -> SUBSET Points]
    /\ redirect \in [TabletIds -> SUBSET TabletIds]
    /\ epoch \in [TabletIds -> Nat]
    /\ version \in Nat

Init ==
    /\ state = [t \in TabletIds |-> "absent"]
    /\ range = [t \in TabletIds |-> {}]
    /\ redirect = [t \in TabletIds |-> {}]
    /\ epoch = [t \in TabletIds |-> 0]
    /\ version = 0

ActiveSet == {t \in TabletIds : state[t] = "active"}

Bootstrap(t, R) ==
    /\ version = 0
    /\ state[t] = "absent"
    /\ R /= {}
    /\ state' = [state EXCEPT ![t] = "allocated"]
    /\ range' = [range EXCEPT ![t] = R]
    /\ epoch' = [epoch EXCEPT ![t] = 1]
    /\ UNCHANGED redirect
    /\ version' = version + 1

Allocate(t, R) ==
    /\ version > 0
    /\ state[t] = "absent"
    /\ R /= {}
    /\ state' = [state EXCEPT ![t] = "allocated"]
    /\ range' = [range EXCEPT ![t] = R]
    /\ epoch' = [epoch EXCEPT ![t] = 1]
    /\ UNCHANGED redirect
    /\ version' = version + 1

Stage(t) ==
    /\ state[t] = "allocated"
    /\ state' = [state EXCEPT ![t] = "inactive"]
    /\ UNCHANGED <<range, redirect, epoch>>
    /\ version' = version + 1

Activate(t) ==
    /\ state[t] = "inactive"
    /\ \A u \in ActiveSet : range[u] \cap range[t] = {}
    /\ state' = [state EXCEPT ![t] = "active"]
    /\ UNCHANGED <<range, redirect, epoch>>
    /\ version' = version + 1

Seal(t) ==
    /\ state[t] = "active"
    /\ state' = [state EXCEPT ![t] = "fenced"]
    /\ UNCHANGED <<range, redirect, epoch>>
    /\ version' = version + 1

Retire(t, S) ==
    /\ state[t] \in {"fenced", "inactive"}
    /\ S /= {}
    /\ \A s \in S : state[s] = "active"
    /\ state' = [state EXCEPT ![t] = "tombstone"]
    /\ redirect' = [redirect EXCEPT ![t] = S]
    /\ UNCHANGED <<range, epoch>>
    /\ version' = version + 1

Next ==
    \/ \E t \in TabletIds, R \in SUBSET Points : Bootstrap(t, R) \/ Allocate(t, R)
    \/ \E t \in TabletIds : Stage(t) \/ Activate(t) \/ Seal(t)
    \/ \E t \in TabletIds, S \in SUBSET TabletIds : Retire(t, S)

Spec == Init /\ [][Next]_vars

(***************************************************************************)
(* Generation successor with explicit exhaustion (Rust: next() returning    *)
(* Result<Self, GenerationExhausted>). The top value has no successor;     *)
(* saturation and wrap-around are not representable.                       *)
(***************************************************************************)

CheckedNext(e) == IF e = MaxGen THEN "Exhausted" ELSE e + 1

(***************************************************************************)
(* State invariants (checkable by TLC as INVARIANTS).                       *)
(***************************************************************************)

NoOverlappingAuthority ==
    \A t, u \in ActiveSet : t /= u => range[t] \cap range[u] = {}

AtMostOneWritableAuthority ==
    \A p \in Points : Cardinality({t \in ActiveSet : p \in range[t]}) <= 1

TombstonesRedirectForward ==
    \A t \in TabletIds : state[t] = "tombstone" =>
        /\ redirect[t] /= {}
        /\ redirect[t] \subseteq {u \in TabletIds : state[u] /= "absent"}

NoSilentSaturation ==
    \A e \in 0..MaxGen : CheckedNext(e) = "Exhausted" \/ CheckedNext(e) = e + 1

(***************************************************************************)
(* Temporal safety properties (checkable by TLC as PROPERTIES).             *)
(***************************************************************************)

MonotonicEpochAction ==
    \A t \in TabletIds :
        \/ epoch'[t] = epoch[t]
        \/ (state[t] = "absent" /\ epoch[t] = 0 /\ epoch'[t] = 1)

MonotonicEpoch == [][MonotonicEpochAction]_vars

FencedAuthorityCannotReturn ==
    [][\A t \in TabletIds :
        (state[t] = "fenced" \/ state[t] = "tombstone") => state'[t] /= "active"]_vars

VersionAdvancesByOne ==
    [][version' = version \/ version' = version + 1]_vars

====
