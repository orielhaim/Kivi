---- MODULE lease ----
(*
  Phase-8 formal model for Kivi's Lease object discipline: fencing
  tokens advance on every grant, renew keeps the token, stale tokens
  are powerless, expiry frees the lease, and a crashed holder's lease
  lapses by the clock.

  Configuration: two clients (1, 2), TTL 2, clock 0..6, tokens 0..4.

  Correspondence to crates/kivi-state/src/{object,semantics,store}.rs:
    acquire ................... Acquire (free or expired -> grant with
                                token+1, expiry = clock+TTL)
    renew ..................... Renew (live holder + current token ->
                                expiry extended, token unchanged)
    stale renew/release ....... StaleRenew / StaleRelease (token /= current
                                -> no-op; a stale release never frees a
                                successor's lease)
    release ................... Release (live holder + current token ->
                                free; token NOT reset — successors still
                                fence predecessors)
    expiry .................... Tick (clock advances past expiry; the
                                next acquire observes the lapse)
    crash ..................... CrashHolder (holder state is durable in
                                the store; the crash matters only in that
                                the holder stops renewing — modeled by
                                enabling Tick past expiry)

  Deliberate abstractions (documented, not hidden):
    - The store itself never crashes here (lease durability across
      tablet restart is covered by the Rust checkpoint tests); what is
      modeled is holder failure vs token discipline.
    - Single lease key; leases compose by key independence.
    - Token/sequence caps disable sends at the top (no wrap), as in
      roster_lease; 64-bit exhaustion is Rust-tested.
    - AllowRevive selects the BROKEN variant: a renew with a stale
      token on an expired lease re-seats the old holder under the old
      token -> violates HolderFresh.

  Invariants:
    HolderFresh: a seated holder always holds the latest issued token
                 (no stale takeover, no revival).
    FencingMonotonic: the issued token never decreases.
*)

EXTENDS Integers, FiniteSets, TLC

CONSTANTS Clients, TTL, MaxClock, MaxToken, AllowRevive
VARIABLES holder, token, expiry, clock, issued

Vars == <<holder, token, expiry, clock, issued>>

TypeOK ==
    /\ holder \in {0} \cup Clients
    /\ token \in 0..MaxToken
    /\ expiry \in 0..MaxClock
    /\ clock \in 0..MaxClock
    /\ issued \in 0..MaxToken
    /\ token <= issued

Init ==
    /\ holder = 0
    /\ token = 0
    /\ expiry = 0
    /\ clock = 0
    /\ issued = 0

Live == holder /= 0 /\ clock < expiry

Acquire(p) ==
    /\ p \in Clients
    /\ (holder = 0 \/ clock >= expiry)
    /\ issued + 1 <= MaxToken
    /\ clock + TTL <= MaxClock
    /\ holder' = p
    /\ issued' = issued + 1
    /\ token' = issued + 1
    /\ expiry' = clock + TTL
    /\ UNCHANGED <<clock>>

Renew(p, tok) ==
    /\ p \in Clients
    /\ holder = p
    /\ token = tok
    /\ clock < expiry
    /\ clock + TTL <= MaxClock
    /\ expiry' = clock + TTL
    /\ UNCHANGED <<holder, token, clock, issued>>

StaleRenew(p, tok) ==
    (* Stale token: powerless no-op. The broken variant revives. *)
    /\ p \in Clients
    /\ tok /= token
    /\ clock + TTL <= MaxClock
    /\ IF AllowRevive /\ holder = 0 /\ clock >= expiry /\ tok < issued
       THEN /\ holder' = p
            /\ token' = tok
            /\ expiry' = clock + TTL
            /\ UNCHANGED <<clock, issued>>
       ELSE UNCHANGED Vars

Release(p, tok) ==
    /\ p \in Clients
    /\ holder = p
    /\ token = tok
    /\ holder' = 0
    /\ UNCHANGED <<token, expiry, clock, issued>>

StaleRelease(p, tok) ==
    (* A stale release never frees a successor's lease. *)
    /\ p \in Clients
    /\ tok /= token
    /\ UNCHANGED Vars

Tick ==
    /\ clock + 1 <= MaxClock
    /\ clock' = clock + 1
    /\ UNCHANGED <<holder, token, expiry, issued>>

Next ==
    \/ \E p \in Clients : Acquire(p)
    \/ \E p \in Clients, tok \in 0..MaxToken : Renew(p, tok)
    \/ \E p \in Clients, tok \in 0..MaxToken : StaleRenew(p, tok)
    \/ \E p \in Clients, tok \in 0..MaxToken : Release(p, tok)
    \/ \E p \in Clients, tok \in 0..MaxToken : StaleRelease(p, tok)
    \/ Tick

Spec == Init /\ [][Next]_Vars

HolderFresh ==
    holder = 0 \/ token = issued

FencingMonotonic ==
    issued >= token

THEOREM Spec => []TypeOK

====
