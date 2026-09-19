---- MODULE txn_2pc ----
(*
  Phase-8 formal model for Kivi's cross-tablet OCC + 2PC discipline:
  durable prepare intents, a single durable coordinator decision,
  idempotent replays, crash/restart with durable survival, message
  loss/duplication, and recovery by record re-read (never guessing).

  Configuration: two participants (1, 2; coordinator = 1 = Min),
  two transactions sharing key 1 (txn 1 writes {1,2}, txn 2 writes {1}:
  they conflict on key 1, so at most one commits). Digests 1..2 name
  write sets; digest 3 models a foreign write set under a reused id
  (confused deputy).

  Correspondence to crates/kivi-state/src/{txn,store}.rs (rule by rule):
    OCC validate + reserve .... Prepare / DeliverPrepare (absence/version
                                expectation against applied[]; foreign or
                                digest-mismatched intent conflicts; the
                                identical reservation replays success)
    durable decision .......... DecideCommit (all prepared with the bound
                                digest, CAS on none) / DecideAbort (any
                                time pre-commit, CAS on none)
    finalize .................. Finalize / RecoverFinalize / DeliverFinalize
                                (commit applies iff the intent is present
                                with a matching digest AND the durable
                                decision is Commit for that txn; abort
                                discards; missing intent = idempotent
                                no-op; delivery consumes the message)
    idempotent replay ......... ResendPrepare / ResendFinalize /
                                RecoverFinalize (re-send after consumption;
                                every handler total and idempotent)
    crash/restart ............. Crash (durable intent[]/decision[] survive;
                                undelivered messages wait in the soup)
                                / Restart
    recovery .................. RecoverFinalize (re-read the durable
                                decision; re-drive the decided outcome;
                                undecided resolves nothing — never guesses)
    confused deputy ........... ForeignPrepare (same id, digest 3:
                                conflicts, never overwrites)

  Deliberate abstractions (documented, not hidden):
    - Soup bound 4: a full commit fan-out is 2 finalizes and a full
      prepare wave is 2..4 prepares, so every single-transaction shape
      fits; pruned are only states with two transactions' waves fully
      overlapped in flight. Cross-transaction interference (intent
      conflict, version rejection) is still explored sequentially as
      waves drain, which is equivalent for safety: interference acts
      through durable intent/applied, never through soup co-residence.
    - Message soup with explicit prepare-loss instead of a link matrix.
      Commit-finalizes are never lost, only delayed by recipient crash:
      this is the implementation's finalize re-drive discipline (retry
      until delivered), not an assumption of reliable links. Prepare
      loss is benign (resent) and IS modeled.
    - Delivery consumes the message (stream-transport semantics, as in
      roster_lease). Replays arrive via the explicit Resend*/Recover*
      actions, which is exactly the replay shape the implementation
      must survive.
    - The decide step atomically enqueues one finalize per key (a
      durable outbox write piggy-backed on the decision record); network
      delay is modeled by soup residence, not by staggered sends. The
      standalone Finalize/Resend actions model driver re-drives
      (duplicates), which delivery absorbs idempotently.
    - Coordinator co-located with participant 1 (lowest id). Identity is
      derivable and all coordinator state is the single durable
      decision; a separate coordinator node adds no behaviors.
    - One key per participant; values abstracted to applied[] versions.
    - No wall clocks: resolver leases/graces are Rust-tested timing
      policy; the model proves the safety core (waiting is always safe).
    - Flags select broken variants (canonical cfgs set them FALSE and
      must pass; broken cfgs set one TRUE and must violate its target):
        AllowGuessCommit: ForgeCommitFinalize lets a participant forge
          the commit outcome for its own live intent (applying with no
          durable Commit) -> violates CommitNeedsDecision.
        AllowGuessAbort: a participant discards a prepared intent without
          a decision; the coordinator still commits the other key ->
          violates NoPartialCommit.

  Invariants (NoPartialCommit is quiescence-conditioned: in-flight
  finalizes may transiently expose one key before the other lands;
  the guarantee is that no QUIESCENT state — nothing left for the
  transaction in the soup — is partial):
    NoPartialCommit: every quiescent transaction has all or none of its
                     writes applied; an Abort-decided one has none.
    CommitNeedsDecision: every applied write follows a durable Commit
                     decision for its transaction.
*)

EXTENDS Integers, FiniteSets, TLC

CONSTANTS Parts, Txns, AllowGuessCommit, AllowGuessAbort
VARIABLES intent, decision, applied, msgs, crashed

Vars == <<intent, decision, applied, msgs, crashed>>

Coordinator == CHOOSE p \in Parts : \A q \in Parts : p <= q

KeysOf(t) == IF t = 1 THEN {1, 2} ELSE {1}

NoneIntent == [txn |-> 0, digest |-> 0]

TypeOK ==
    /\ intent \in [Parts -> [txn : {0} \cup Txns, digest : 0..3]]
    /\ decision \in [Txns -> {"none", "commit", "abort"}]
    /\ applied \in [Parts -> [Txns -> BOOLEAN]]
    /\ msgs \in SUBSET [kind : {"prepare", "finalize"},
                        txn : Txns, key : Parts,
                        digest : 0..3, commit : BOOLEAN]
    /\ crashed \in [Parts -> BOOLEAN]

Init ==
    /\ intent = [p \in Parts |-> NoneIntent]
    /\ decision = [t \in Txns |-> "none"]
    /\ applied = [p \in Parts |-> [t \in Txns |-> FALSE]]
    /\ msgs = {}
    /\ crashed = [p \in Parts |-> FALSE]

Send(m) == msgs' = msgs \cup {m}

Prepare(t, p, d) ==
    (* Callable at any time: first send and every re-drive. Delivery is
       idempotent, so replays are safe. *)
    /\ ~crashed[Coordinator]
    /\ Send([kind |-> "prepare", txn |-> t, key |-> p, digest |-> d, commit |-> FALSE])
    /\ UNCHANGED <<intent, decision, applied, crashed>>

ForeignPrepare(t, p) ==
    (* Same id, foreign digest: the confused deputy. Must conflict,
       never overwrite; injectable at any time. *)
    /\ intent[p].txn = t
    /\ intent[p].digest /= 3
    /\ Send([kind |-> "prepare", txn |-> t, key |-> p, digest |-> 3, commit |-> FALSE])
    /\ UNCHANGED <<intent, decision, applied, crashed>>

DeliverPrepare(t, p, d) ==
    LET m == [kind |-> "prepare", txn |-> t, key |-> p, digest |-> d, commit |-> FALSE] IN
    /\ m \in msgs
    /\ ~crashed[p]
    /\ msgs' = msgs \ {m}
    /\ IF intent[p] = NoneIntent /\ \A t2 \in Txns : t2 = t \/ ~applied[p][t2]
       THEN intent' = [intent EXCEPT ![p] = [txn |-> t, digest |-> d]]
       ELSE intent' = intent
    /\ UNCHANGED <<decision, applied, crashed>>

DecideCommit(t) ==
    /\ decision[t] = "none"
    /\ ~crashed[Coordinator]
    /\ \A p \in KeysOf(t) : intent[p].txn = t /\ intent[p].digest = t
    /\ decision' = [decision EXCEPT ![t] = "commit"]
    /\ msgs' = msgs \cup { [kind |-> "finalize", txn |-> t, key |-> p,
                            digest |-> t, commit |-> TRUE] : p \in KeysOf(t) }
    /\ UNCHANGED <<intent, applied, crashed>>

DecideAbort(t) ==
    /\ decision[t] = "none"
    /\ ~crashed[Coordinator]
    /\ decision' = [decision EXCEPT ![t] = "abort"]
    /\ msgs' = msgs \cup { [kind |-> "finalize", txn |-> t, key |-> p,
                            digest |-> intent[p].digest, commit |-> FALSE]
                         : p \in KeysOf(t) }
    /\ UNCHANGED <<intent, applied, crashed>>

Finalize(t, p, c, d) ==
    (* Callable at any time subject to the guard: first send (post-
      decision the outbox already holds one; this covers driver
      re-drives) and every replay. Delivery is idempotent.
      Honest drivers only (re)send finalizes consistent with the durable
      decision (or none exist pre-decision: the decide step enqueues).
      A contradicting finalize cannot be produced under crash faults —
      the decision is single-assignment. *)
    /\ ~crashed[Coordinator]
    /\ \/ c /\ decision[t] = "commit"
       \/ ~c /\ decision[t] = "abort"
    /\ Send([kind |-> "finalize", txn |-> t, key |-> p, digest |-> d, commit |-> c])
    /\ UNCHANGED <<intent, decision, applied, crashed>>

DeliverFinalize(t, p, c, d) ==
    LET m == [kind |-> "finalize", txn |-> t, key |-> p, digest |-> d, commit |-> c] IN
    /\ m \in msgs
    /\ ~crashed[p]
    /\ msgs' = msgs \ {m}
    /\ IF intent[p].txn = t /\ intent[p].digest = d
       THEN IF c
            THEN IF decision[t] = "commit" \/ AllowGuessCommit
                 THEN /\ applied' = [applied EXCEPT ![p] = [@ EXCEPT ![t] = TRUE]]
                      /\ intent' = [intent EXCEPT ![p] = NoneIntent]
                      /\ UNCHANGED <<decision, crashed>>
                 ELSE UNCHANGED <<intent, decision, applied, crashed>>
            ELSE /\ intent' = [intent EXCEPT ![p] = NoneIntent]
                 /\ UNCHANGED <<decision, applied, crashed>>
       ELSE UNCHANGED <<intent, decision, applied, crashed>>

RecoverFinalize(t, p) ==
    (* Re-read the durable decision and re-drive the decided outcome.
       Undecided resolves nothing — never guesses. *)
    /\ ~crashed[Coordinator]
    /\ decision[t] = "commit"
    /\ intent[p].txn = t
    /\ Finalize(t, p, TRUE, intent[p].digest)
    /\ UNCHANGED <<intent, decision, applied, crashed>>

ForgeCommitFinalize(t, p) ==
    (* BROKEN VARIANT (AllowGuessCommit): a participant forges the commit
       outcome for its own live intent — applying without any durable
       decision. The coordinator may later abort -> applied without a
       Commit decision. *)
    /\ AllowGuessCommit
    /\ ~crashed[p]
    /\ intent[p].txn = t
    /\ Send([kind |-> "finalize", txn |-> t, key |-> p,
             digest |-> intent[p].digest, commit |-> TRUE])
    /\ UNCHANGED <<intent, decision, applied, crashed>>

GuessAbort(t, p) ==
    (* BROKEN VARIANT (AllowGuessAbort): discard a prepared intent with
       no durable decision. The coordinator may still commit the sibling
       key -> a quiescent partial state. *)
    /\ AllowGuessAbort
    /\ ~crashed[p]
    /\ intent[p].txn = t
    /\ intent' = [intent EXCEPT ![p] = NoneIntent]
    /\ UNCHANGED <<decision, applied, msgs, crashed>>

Crash(p) ==
    /\ ~crashed[p]
    /\ crashed' = [crashed EXCEPT ![p] = TRUE]
    /\ UNCHANGED <<intent, decision, applied, msgs>>

Restart(p) ==
    /\ crashed[p]
    /\ crashed' = [crashed EXCEPT ![p] = FALSE]
    /\ UNCHANGED <<intent, decision, applied, msgs>>

LosePrepare(m) ==
    (* Only prepares can be lost (benign: resent). Commit-finalizes are
       re-driven until delivered; their loss is modeled as recipient
       crash, with the message waiting in the soup. *)
    /\ m \in msgs
    /\ m.kind = "prepare"
    /\ msgs' = msgs \ {m}
    /\ UNCHANGED <<intent, decision, applied, crashed>>

Next ==
    \/ \E t \in Txns, p \in Parts, d \in 1..2 : Prepare(t, p, d)
    \/ \E t \in Txns, p \in Parts : ForeignPrepare(t, p)
    \/ \E t \in Txns, p \in Parts, d \in 0..3 : DeliverPrepare(t, p, d)
    \/ \E t \in Txns : DecideCommit(t)
    \/ \E t \in Txns : DecideAbort(t)
    \/ \E t \in Txns, p \in Parts, c \in BOOLEAN, d \in 1..2 : Finalize(t, p, c, d)
    \/ \E t \in Txns, p \in Parts, c \in BOOLEAN, d \in 0..3 : DeliverFinalize(t, p, c, d)
    \/ \E t \in Txns, p \in Parts : RecoverFinalize(t, p)
    \/ \E t \in Txns, p \in Parts : ForgeCommitFinalize(t, p)
    \/ \E t \in Txns, p \in Parts : GuessAbort(t, p)
    \/ \E p \in Parts : Crash(p)
    \/ \E p \in Parts : Restart(p)
    \/ \E m \in msgs : LosePrepare(m)

Spec == Init /\ [][Next]_Vars

AppliedKeys(t) == {p \in KeysOf(t) : applied[p][t]}

Quiescent(t) == ~\E m \in msgs : m.txn = t

NoPartialCommit ==
    \A t \in Txns : Quiescent(t) =>
        /\ \/ AppliedKeys(t) = {}
           \/ AppliedKeys(t) = KeysOf(t)
        /\ \/ decision[t] /= "abort"
           \/ AppliedKeys(t) = {}

CommitNeedsDecision ==
    \A t \in Txns : \A p \in KeysOf(t) :
        applied[p][t] => decision[t] = "commit"

SoupBound == Cardinality(msgs) <= 4

THEOREM Spec => []TypeOK

====
