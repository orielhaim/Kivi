//! Deterministic 2PC transaction simulation (Phase 8).
//!
//! A message-level model of Kivi's cross-tablet OCC + 2PC discipline,
//! executed over [`SimCluster`](crate::SimCluster) with virtual time,
//! lossy/duplicating/partitioning links, and crash/restart lifecycle:
//!
//! ```text
//! driver --Prepare--> participants (OCC validate + durable intent)
//! driver <--Prepared/Conflict-- participants
//! driver --Decide--> coordinator (durable record, first-writer-wins)
//! driver --Finalize(commit/abort)--> participants (apply/discard)
//! driver --RecordRead--> coordinator (recovery; never guesses)
//! ```
//!
//! ## What this models — and what it does not
//!
//! Modeled faithfully: prepare validation (OCC versions incl. absence,
//! foreign-intent conflicts, digest binding), durable intents, the single
//! durable decision, idempotent replays, recovery by record re-read (never
//! guessing), coordinator/participant crash and full restart, message
//! loss/duplication/reorder/partitions, and resolver convergence.
//!
//! Abstracted: transaction ids are `u64` (production: BLAKE3-derived
//! 128-bit ids); durability is handler-retained maps standing in for the
//! checkpoint/WAL (a crash discards only routing of messages to the down
//! node — deliveries to non-running instances drop — while every durable
//! step survives, exactly the property under test); versions are plain
//! counters; ordering uses a logical sequence (virtual ticks order events
//! but handler logic must not read clocks). Disk-torn-write faults belong
//! to the durability-crate suites, not to this protocol model. Every
//! protocol rule the production code enforces appears here as an explicit
//! check, and the invariants below name exactly what must hold.
//!
//! ## Invariants (checked after every stepped event)
//!
//! ```text
//! NoPartialCommit:  a transaction with any applied write has ALL its
//!                   writes applied; aborted transactions have none.
//! NoSplitDecision:  no (coordinator, txn) holds both Commit and Abort.
//! NoGuess:          every applied write follows a durable Commit
//!                   decision for its digest (participants never finalize
//!                   an undecided transaction).
//! ```

use std::collections::{BTreeMap, BTreeSet};

use kivi_core::RandomSource;
use kivi_types::NodeId;

use crate::cluster::{AppHandler, ClusterAction};
use crate::net::Endpoint;

// ---------------------------------------------------------------------------
// Model types (abstractions of the production transaction model)
// ---------------------------------------------------------------------------

/// Model transaction id (production: content-derived 128-bit ids).
pub type TxnSimId = u64;
/// Model key (production: keys routed to tablets).
pub type TxnKey = u64;
/// Model write-set digest (production: BLAKE3 over the sorted write set).
pub type TxnDigest = u64;

/// One point write: blind put with an OCC expectation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TxnWrite {
    /// Target key.
    pub key: TxnKey,
    /// Value to store on commit.
    pub value: u64,
    /// Expected live version (`None` = key must be absent).
    pub expect: Option<u64>,
}

/// Durable coordinator decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum TxnDecision {
    /// Commit: every participant must apply.
    Commit,
    /// Abort: every participant must discard.
    Abort,
}

/// Durable participant intent.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TxnIntent {
    /// Transaction this reservation belongs to.
    pub txn: TxnSimId,
    /// Coordinator hosting the decision record.
    pub coordinator: NodeId,
    /// Prepared write.
    pub write: TxnWrite,
    /// Digest the reservation is bound to.
    pub digest: TxnDigest,
    /// Observed live version at prepare (`None` = absent).
    pub observed: Option<u64>,
}

/// Protocol messages (hand codec below; deterministic, no serde).
/// Every request carries a driver-minted `req` correlation id echoed in
/// its reply (production multiplexes responses onto requests the same
/// way); replies are attributed exactly, never heuristically.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TxnMsg {
    /// Reserve one key (OCC + intent).
    Prepare {
        /// Request correlation id.
        req: u64,
        /// Transaction.
        txn: TxnSimId,
        /// Coordinator for the decision record.
        coordinator: NodeId,
        /// Write to reserve.
        write: TxnWrite,
        /// Write-set digest binding.
        digest: TxnDigest,
    },
    /// Prepare outcome.
    Prepared {
        /// Echoed correlation id.
        req: u64,
        /// Transaction.
        txn: TxnSimId,
        /// Key reserved.
        key: TxnKey,
        /// Whether the intent was reserved.
        ok: bool,
    },
    /// Persist the coordinator decision (first-writer-wins, like the
    /// absence-guarded record prepare in production).
    Decide {
        /// Request correlation id.
        req: u64,
        /// Transaction.
        txn: TxnSimId,
        /// Decision to durably record.
        decision: TxnDecision,
        /// Digest the decision covers.
        digest: TxnDigest,
    },
    /// Decide outcome (whether this send persisted the decision).
    Decided {
        /// Echoed correlation id.
        req: u64,
        /// Transaction.
        txn: TxnSimId,
        /// Whether the decision is now durable (either by this send or a
        /// racing one — the driver re-reads to learn which).
        stored: bool,
    },
    /// Resolve one key's intent.
    Finalize {
        /// Request correlation id.
        req: u64,
        /// Transaction.
        txn: TxnSimId,
        /// Key to resolve.
        key: TxnKey,
        /// Whether to apply (`true`) or discard (`false`).
        commit: bool,
        /// Digest the intent must be bound to.
        digest: TxnDigest,
    },
    /// Finalize outcome.
    Finalized {
        /// Echoed correlation id.
        req: u64,
        /// Transaction.
        txn: TxnSimId,
        /// Key resolved.
        key: TxnKey,
        /// Whether a write was applied.
        applied: bool,
    },
    /// Recovery probe: what does the coordinator record say?
    RecordRead {
        /// Request correlation id.
        req: u64,
        /// Transaction.
        txn: TxnSimId,
    },
    /// Recovery reply: the durable decision, if any.
    RecordReply {
        /// Echoed correlation id.
        req: u64,
        /// Transaction.
        txn: TxnSimId,
        /// Durable decision (`None` = undecided).
        decision: Option<(TxnDecision, TxnDigest)>,
    },
}

/// Encodes one message (little-endian, versioned tags; unknown tags fail
/// decode loudly, mirroring the production wire discipline).
#[must_use]
pub fn encode_msg(message: &TxnMsg) -> Vec<u8> {
    let mut out = Vec::new();
    match message {
        TxnMsg::Prepare {
            req,
            txn,
            coordinator,
            write,
            digest,
        } => {
            out.push(1);
            encode_msg_prepare(&mut out, *req, *txn, *coordinator, write, *digest);
        }
        TxnMsg::Prepared { req, txn, key, ok } => {
            encode_msg_flag3(&mut out, 2, *req, *txn, *key, *ok);
        }
        TxnMsg::Decide {
            req,
            txn,
            decision,
            digest,
        } => {
            out.push(3);
            encode_msg_decide(&mut out, *req, *txn, *decision, *digest);
        }
        TxnMsg::Decided { req, txn, stored } => {
            encode_msg_flag2(&mut out, 8, *req, *txn, *stored);
        }
        TxnMsg::Finalize {
            req,
            txn,
            key,
            commit,
            digest,
        } => {
            out.push(4);
            encode_msg_finalize(&mut out, *req, *txn, *key, *commit, *digest);
        }
        TxnMsg::Finalized {
            req,
            txn,
            key,
            applied,
        } => {
            encode_msg_flag3(&mut out, 5, *req, *txn, *key, *applied);
        }
        TxnMsg::RecordRead { req, txn } => {
            encode_msg_pair(&mut out, 6, *req, *txn);
        }
        TxnMsg::RecordReply { req, txn, decision } => {
            out.push(7);
            encode_msg_record_reply(&mut out, *req, *txn, *decision);
        }
    }
    out
}

fn encode_msg_u64(out: &mut Vec<u8>, value: u64) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn encode_msg_opt_u64(out: &mut Vec<u8>, value: Option<u64>) {
    match value {
        None => out.push(0),
        Some(value) => {
            out.push(1);
            encode_msg_u64(out, value);
        }
    }
}

fn encode_msg_tag(decision: TxnDecision) -> u8 {
    match decision {
        TxnDecision::Commit => 1,
        TxnDecision::Abort => 2,
    }
}

/// Encodes `tag + three u64 + one flag` (prepare replies, finalize
/// receipts): one shape, three messages.
fn encode_msg_flag3(out: &mut Vec<u8>, tag: u8, first: u64, second: u64, third: u64, flag: bool) {
    out.push(tag);
    encode_msg_u64(out, first);
    encode_msg_u64(out, second);
    encode_msg_u64(out, third);
    out.push(u8::from(flag));
}

/// Encodes `tag + two u64 + one flag` (decision receipts).
fn encode_msg_flag2(out: &mut Vec<u8>, tag: u8, first: u64, second: u64, flag: bool) {
    out.push(tag);
    encode_msg_u64(out, first);
    encode_msg_u64(out, second);
    out.push(u8::from(flag));
}

/// Encodes `tag + two u64` (record reads).
fn encode_msg_pair(out: &mut Vec<u8>, tag: u8, first: u64, second: u64) {
    out.push(tag);
    encode_msg_u64(out, first);
    encode_msg_u64(out, second);
}

fn encode_msg_prepare(
    out: &mut Vec<u8>,
    req: u64,
    txn: TxnSimId,
    coordinator: NodeId,
    write: &TxnWrite,
    digest: TxnDigest,
) {
    encode_msg_u64(out, req);
    encode_msg_u64(out, txn);
    encode_msg_u64(out, coordinator.as_u64());
    encode_msg_u64(out, write.key);
    encode_msg_u64(out, write.value);
    encode_msg_opt_u64(out, write.expect);
    encode_msg_u64(out, digest);
}

fn encode_msg_decide(
    out: &mut Vec<u8>,
    req: u64,
    txn: TxnSimId,
    decision: TxnDecision,
    digest: TxnDigest,
) {
    encode_msg_u64(out, req);
    encode_msg_u64(out, txn);
    out.push(encode_msg_tag(decision));
    encode_msg_u64(out, digest);
}

fn encode_msg_finalize(
    out: &mut Vec<u8>,
    req: u64,
    txn: TxnSimId,
    key: TxnKey,
    commit: bool,
    digest: TxnDigest,
) {
    encode_msg_u64(out, req);
    encode_msg_u64(out, txn);
    encode_msg_u64(out, key);
    out.push(u8::from(commit));
    encode_msg_u64(out, digest);
}

fn encode_msg_record_reply(
    out: &mut Vec<u8>,
    req: u64,
    txn: TxnSimId,
    decision: Option<(TxnDecision, TxnDigest)>,
) {
    encode_msg_u64(out, req);
    encode_msg_u64(out, txn);
    match decision {
        None => out.push(0),
        Some((decision, digest)) => {
            out.push(1);
            out.push(encode_msg_tag(decision));
            encode_msg_u64(out, digest);
        }
    }
}

/// Decodes one message (`None` on truncation or unknown tags — never a
/// guess, mirroring production wire discipline).
#[must_use]
pub fn decode_msg(input: &[u8]) -> Option<TxnMsg> {
    let tag = *input.first()?;
    let mut at = 1;
    let message = match tag {
        1 => decode_msg_prepare(input, &mut at)?,
        2 => TxnMsg::Prepared {
            req: msg_take_u64(input, &mut at)?,
            txn: msg_take_u64(input, &mut at)?,
            key: msg_take_u64(input, &mut at)?,
            ok: msg_take_bool(input, &mut at)?,
        },
        3 => TxnMsg::Decide {
            req: msg_take_u64(input, &mut at)?,
            txn: msg_take_u64(input, &mut at)?,
            decision: msg_take_decision(input, &mut at)?,
            digest: msg_take_u64(input, &mut at)?,
        },
        8 => TxnMsg::Decided {
            req: msg_take_u64(input, &mut at)?,
            txn: msg_take_u64(input, &mut at)?,
            stored: msg_take_bool(input, &mut at)?,
        },
        4 => TxnMsg::Finalize {
            req: msg_take_u64(input, &mut at)?,
            txn: msg_take_u64(input, &mut at)?,
            key: msg_take_u64(input, &mut at)?,
            commit: msg_take_bool(input, &mut at)?,
            digest: msg_take_u64(input, &mut at)?,
        },
        5 => TxnMsg::Finalized {
            req: msg_take_u64(input, &mut at)?,
            txn: msg_take_u64(input, &mut at)?,
            key: msg_take_u64(input, &mut at)?,
            applied: msg_take_bool(input, &mut at)?,
        },
        6 => TxnMsg::RecordRead {
            req: msg_take_u64(input, &mut at)?,
            txn: msg_take_u64(input, &mut at)?,
        },
        7 => decode_msg_record_reply(input, &mut at)?,
        _ => return None,
    };
    if at != input.len() {
        return None;
    }
    Some(message)
}

fn msg_take_u64(input: &[u8], at: &mut usize) -> Option<u64> {
    let bytes: [u8; 8] = input.get(*at..*at + 8)?.try_into().ok()?;
    *at += 8;
    Some(u64::from_le_bytes(bytes))
}

fn msg_take_bool(input: &[u8], at: &mut usize) -> Option<bool> {
    match *input.get(*at)? {
        0 => {
            *at += 1;
            Some(false)
        }
        1 => {
            *at += 1;
            Some(true)
        }
        _ => None,
    }
}

fn msg_take_decision(input: &[u8], at: &mut usize) -> Option<TxnDecision> {
    let decision = match *input.get(*at)? {
        1 => TxnDecision::Commit,
        2 => TxnDecision::Abort,
        _ => return None,
    };
    *at += 1;
    Some(decision)
}

fn decode_msg_prepare(input: &[u8], at: &mut usize) -> Option<TxnMsg> {
    let req = msg_take_u64(input, at)?;
    let txn = msg_take_u64(input, at)?;
    let coordinator = NodeId::from_u64(msg_take_u64(input, at)?);
    let write = TxnWrite {
        key: msg_take_u64(input, at)?,
        value: msg_take_u64(input, at)?,
        expect: if msg_take_bool(input, at)? {
            Some(msg_take_u64(input, at)?)
        } else {
            None
        },
    };
    let digest = msg_take_u64(input, at)?;
    Some(TxnMsg::Prepare {
        req,
        txn,
        coordinator,
        write,
        digest,
    })
}

fn decode_msg_record_reply(input: &[u8], at: &mut usize) -> Option<TxnMsg> {
    let req = msg_take_u64(input, at)?;
    let txn = msg_take_u64(input, at)?;
    let decision = match *input.get(*at)? {
        0 => {
            *at += 1;
            None
        }
        1 => {
            *at += 1;
            let decision = msg_take_decision(input, at)?;
            Some((decision, msg_take_u64(input, at)?))
        }
        _ => return None,
    };
    Some(TxnMsg::RecordReply { req, txn, decision })
}

// ---------------------------------------------------------------------------
// Driver events (scenario control plane)
// ---------------------------------------------------------------------------

/// Scheduled driver commands (node 0 acts as the client driver).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TxnDriverEv {
    /// Begin a transaction: prepare every write.
    Begin {
        /// Transaction.
        txn: TxnSimId,
        /// Writes in request order.
        writes: Vec<TxnWrite>,
        /// Write-set digest.
        digest: TxnDigest,
    },
    /// Re-drive an ambiguous transaction: record first, never re-prepare
    /// blindly (re-prepares after a commit would spuriously conflict).
    Recover {
        /// Transaction.
        txn: TxnSimId,
        /// Writes (for finalizes after a Commit decision is found).
        writes: Vec<TxnWrite>,
        /// Write-set digest.
        digest: TxnDigest,
    },
}

// ---------------------------------------------------------------------------
// The simulated 2PC fabric
// ---------------------------------------------------------------------------

/// One applied write (for invariant checks).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AppliedWrite {
    /// Transaction that wrote it.
    pub txn: TxnSimId,
    /// Digest it committed under.
    pub digest: TxnDigest,
    /// Value applied.
    pub value: u64,
    /// Version assigned.
    pub version: u64,
    /// Logical sequence of the durable Commit decision this apply followed.
    pub decided_seq: u64,
    /// Logical sequence at which the write applied.
    pub applied_seq: u64,
}

/// Deterministic 2PC fabric: durable intents, durable decisions, and
/// committed state, plus the driver-side collection buffers that make
/// re-drives deterministic. All maps live in the handler (durable —
/// surviving the crash of any node); per-node volatile state is `()`.
/// Ordering uses a logical sequence bumped on every durable step (never a
/// clock): the `NoGuess` invariant compares sequences, not ticks.
#[derive(Debug, Default)]
pub struct TxnSim {
    /// Which node hosts which key (scenario routing; fixed within a run —
    /// Phase 8 never remaps prepared writes into another lineage).
    pub key_home: BTreeMap<TxnKey, NodeId>,
    /// Durable participant intents: `(host, key) -> intent`.
    pub intents: BTreeMap<(NodeId, TxnKey), TxnIntent>,
    /// Durable coordinator decisions: `(coordinator, txn) -> (decision,
    /// digest, decided_seq)`.
    pub decisions: BTreeMap<(NodeId, TxnSimId), (TxnDecision, TxnDigest, u64)>,
    /// Committed user state: `(host, key) -> applied write`.
    pub applied: BTreeMap<(NodeId, TxnKey), AppliedWrite>,
    /// Driver write sets (the client knows what it sent): `(txn, digest)
    /// -> writes`. Keyed by digest as well as id so concurrent
    /// conflicting attempts under one id stay isolated (production
    /// drivers serialize attempts; the sim models the conflict).
    driver_writes: BTreeMap<(TxnSimId, TxnDigest), Vec<TxnWrite>>,
    /// Driver prepare collections: `(txn, digest) -> (ok count, total,
    /// decided?)`. The first terminal signal (all-ok or first conflict)
    /// decides exactly once; late or duplicate replies are ignored, never
    /// a second decision.
    driver_prepares: BTreeMap<(TxnSimId, TxnDigest), (usize, usize, bool)>,
    /// Driver finalize collections: `(txn, digest) -> applied count`.
    driver_finalizes: BTreeMap<(TxnSimId, TxnDigest), usize>,
    /// Driver recovery replies: `(txn, digest) -> decision`.
    driver_records: BTreeMap<(TxnSimId, TxnDigest), Option<(TxnDecision, TxnDigest)>>,
    /// Outstanding driver requests: `req -> (txn, digest)` for exact
    /// reply attribution (production multiplexes the same way).
    outstanding: BTreeMap<u64, (TxnSimId, TxnDigest)>,
    /// Next request correlation id.
    req_seq: u64,
    /// Logical sequence bumped on every durable step.
    seq: u64,
}

/// Handler context for one simulated participant step: who sent the
/// message, who executes it, and where outbound actions go. Bundles the
/// three values every participant handler threads through.
struct StepCtx<'a> {
    reply_to: NodeId,
    host: NodeId,
    actions: &'a mut Vec<ClusterAction<TxnDriverEv>>,
}

impl TxnSim {
    /// Creates an empty fabric.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Bumps and returns the logical sequence for a durable step.
    fn next_seq(&mut self) -> u64 {
        self.seq += 1;
        self.seq
    }

    /// Deterministic coordinator: the lowest hosting node among the
    /// write set's keys (mirrors `lowest TabletId` selection).
    #[must_use]
    pub fn coordinator_for(&self, writes: &[TxnWrite]) -> Option<NodeId> {
        writes
            .iter()
            .filter_map(|write| self.key_home.get(&write.key).copied())
            .min_by_key(|node| node.as_u64())
    }

    /// Sends one message (handler-side helper; no sim state involved).
    fn send_to(to: NodeId, message: &TxnMsg, actions: &mut Vec<ClusterAction<TxnDriverEv>>) {
        actions.push(ClusterAction::NetSend {
            to,
            payload: encode_msg(message),
        });
    }

    /// The driver node (scenario convention: node 0).
    fn driver_node() -> NodeId {
        NodeId::from_u64(0)
    }

    /// Mints a request correlation id bound to one attempt, for exact
    /// reply attribution.
    fn mint_req(&mut self, txn: TxnSimId, digest: TxnDigest) -> u64 {
        self.req_seq += 1;
        let req = self.req_seq;
        self.outstanding.insert(req, (txn, digest));
        req
    }

    /// Handles one driver `Begin`: records the write set and prepares
    /// every write in request order.
    fn driver_begin(
        &mut self,
        txn: TxnSimId,
        writes: &[TxnWrite],
        digest: TxnDigest,
        actions: &mut Vec<ClusterAction<TxnDriverEv>>,
    ) {
        self.driver_writes.insert((txn, digest), writes.to_vec());
        self.driver_prepares
            .insert((txn, digest), (0, writes.len(), false));
        let coordinator = self.coordinator_for(writes).expect("keys route");
        for write in writes {
            if let Some(host) = self.key_home.get(&write.key).copied() {
                let req = self.mint_req(txn, digest);
                Self::send_to(
                    host,
                    &TxnMsg::Prepare {
                        req,
                        txn,
                        coordinator,
                        write: *write,
                        digest,
                    },
                    actions,
                );
            }
        }
    }

    /// Handles one driver recovery probe: records the write set (the
    /// client knows what it sent) and reads the coordinator record. The
    /// reply decides the next step: a Commit decision finalizes, anything
    /// else re-prepares (idempotent replays for already-reserved keys —
    /// never a blind fresh attempt, which would fork reservations).
    fn driver_recover(
        &mut self,
        txn: TxnSimId,
        writes: &[TxnWrite],
        digest: TxnDigest,
        actions: &mut Vec<ClusterAction<TxnDriverEv>>,
    ) {
        self.driver_writes.insert((txn, digest), writes.to_vec());
        if let Some(coordinator) = self.coordinator_for(writes) {
            let req = self.mint_req(txn, digest);
            Self::send_to(coordinator, &TxnMsg::RecordRead { req, txn }, actions);
        }
    }

    /// Handles one prepare on a participant: OCC validation plus digest
    /// binding, then a durable intent — or a deterministic conflict.
    fn on_prepare(
        &mut self,
        ctx: StepCtx<'_>,
        req: u64,
        txn: TxnSimId,
        coordinator: NodeId,
        write: &TxnWrite,
        digest: TxnDigest,
    ) {
        let StepCtx {
            reply_to,
            host,
            actions,
        } = ctx;
        let ok = self.validate_prepare(host, txn, write, digest);
        if ok {
            let observed = self
                .applied
                .get(&(host, write.key))
                .map(|applied| applied.version);
            // Re-preparing the identical reservation is idempotent
            // (transport retry): keep the original intent, same outcome.
            if self
                .intents
                .get(&(host, write.key))
                .is_none_or(|intent| intent.txn != txn || intent.digest != digest)
            {
                self.intents.insert(
                    (host, write.key),
                    TxnIntent {
                        txn,
                        coordinator,
                        write: *write,
                        digest,
                        observed,
                    },
                );
            }
        }
        Self::send_to(
            reply_to,
            &TxnMsg::Prepared {
                req,
                txn,
                key: write.key,
                ok,
            },
            actions,
        );
    }

    /// Pure OCC + binding validation for one prepare.
    fn validate_prepare(
        &self,
        host: NodeId,
        txn: TxnSimId,
        write: &TxnWrite,
        digest: TxnDigest,
    ) -> bool {
        // A foreign intent blocks the key; a same-`TxnId` intent with a
        // different digest is a conflicting driver (never an overwrite);
        // the identical reservation replays success (idempotent retry).
        if let Some(intent) = self.intents.get(&(host, write.key)) {
            return intent.txn == txn && intent.digest == digest && intent.write == *write;
        }
        // OCC expectation against live (committed) state, absence included:
        // a missing key created concurrently must fail validation.
        let live = self
            .applied
            .get(&(host, write.key))
            .map(|applied| applied.version);
        live == write.expect
    }

    /// Handles one finalize on a participant: applies or discards exactly
    /// the bound reservation, never an undecided or foreign one.
    fn on_finalize(
        &mut self,
        ctx: StepCtx<'_>,
        req: u64,
        txn: TxnSimId,
        key: TxnKey,
        commit: bool,
        digest: TxnDigest,
    ) {
        let StepCtx {
            reply_to,
            host,
            actions,
        } = ctx;
        let applied = match self.intents.get(&(host, key)) {
            // No intent (already resolved or never prepared): idempotent
            // no-op, never an error.
            None => false,
            // Foreign or digest-mismatched intent: resolve nothing.
            Some(intent) if intent.txn != txn || intent.digest != digest => false,
            Some(_) => {
                if commit {
                    // Commit requires a durable Commit decision for this
                    // digest: participants never guess. No decision (or a
                    // different one) leaves the intent in place and reports
                    // not-applied; the driver re-reads the record.
                    let intent = self.intents.get(&(host, key)).expect("checked");
                    let decided_seq = self
                        .decisions
                        .get(&(intent.coordinator, txn))
                        .filter(|(decision, record_digest, _)| {
                            *decision == TxnDecision::Commit && *record_digest == digest
                        })
                        .map(|(_, _, seq)| *seq);
                    match decided_seq {
                        None => false,
                        Some(decided_seq) => {
                            let intent = self.intents.remove(&(host, key)).expect("checked");
                            let version = self
                                .applied
                                .get(&(host, key))
                                .map_or(1, |applied| applied.version + 1);
                            let applied_seq = self.next_seq();
                            self.applied.insert(
                                (host, key),
                                AppliedWrite {
                                    txn,
                                    digest,
                                    value: intent.write.value,
                                    version,
                                    decided_seq,
                                    applied_seq,
                                },
                            );
                            true
                        }
                    }
                } else {
                    self.intents.remove(&(host, key));
                    false
                }
            }
        };
        Self::send_to(
            reply_to,
            &TxnMsg::Finalized {
                req,
                txn,
                key,
                applied,
            },
            actions,
        );
    }

    /// Sends a driver Decide and records its correlation.
    fn driver_send_decide(
        &mut self,
        txn: TxnSimId,
        digest: TxnDigest,
        decision: TxnDecision,
        actions: &mut Vec<ClusterAction<TxnDriverEv>>,
    ) {
        let writes = self
            .driver_writes
            .get(&(txn, digest))
            .cloned()
            .unwrap_or_default();
        let coordinator = self.coordinator_for(&writes).expect("keys route");
        let req = self.mint_req(txn, digest);
        Self::send_to(
            coordinator,
            &TxnMsg::Decide {
                req,
                txn,
                decision,
                digest,
            },
            actions,
        );
    }

    /// Sends driver Finalizes for one attempt's write set.
    fn driver_send_finalizes(
        &mut self,
        txn: TxnSimId,
        digest: TxnDigest,
        commit: bool,
        actions: &mut Vec<ClusterAction<TxnDriverEv>>,
    ) {
        let writes = self
            .driver_writes
            .get(&(txn, digest))
            .cloned()
            .unwrap_or_default();
        for write in &writes {
            if let Some(host) = self.key_home.get(&write.key).copied() {
                let req = self.mint_req(txn, digest);
                Self::send_to(
                    host,
                    &TxnMsg::Finalize {
                        req,
                        txn,
                        key: write.key,
                        commit,
                        digest,
                    },
                    actions,
                );
            }
        }
    }

    /// Sends driver Prepares for one attempt's write set (fresh attempt or
    /// recovery re-prepare after an undecided record read).
    fn driver_send_prepares(
        &mut self,
        txn: TxnSimId,
        digest: TxnDigest,
        actions: &mut Vec<ClusterAction<TxnDriverEv>>,
    ) {
        let writes = self
            .driver_writes
            .get(&(txn, digest))
            .cloned()
            .unwrap_or_default();
        self.driver_prepares
            .insert((txn, digest), (0, writes.len(), false));
        let coordinator = self.coordinator_for(&writes).expect("keys route");
        for write in &writes {
            if let Some(host) = self.key_home.get(&write.key).copied() {
                let req = self.mint_req(txn, digest);
                Self::send_to(
                    host,
                    &TxnMsg::Prepare {
                        req,
                        txn,
                        coordinator,
                        write: *write,
                        digest,
                    },
                    actions,
                );
            }
        }
    }

    /// Handles driver-side replies, attributed exactly by request
    /// correlation id (unknown ids — duplicates after collection close —
    /// drop silently; the decided flag already converged the attempt).
    fn on_driver_msg(
        &mut self,
        from: NodeId,
        message: &TxnMsg,
        actions: &mut Vec<ClusterAction<TxnDriverEv>>,
    ) {
        let _ = from;
        match message {
            TxnMsg::Prepared { req, ok, .. } => {
                let Some((txn, digest)) = self.outstanding.get(req).copied() else {
                    return;
                };
                let Some((ok_count, total, decided)) =
                    self.driver_prepares.get(&(txn, digest)).copied()
                else {
                    return;
                };
                if decided {
                    return;
                }
                if *ok {
                    let ok_count = ok_count + 1;
                    if ok_count == total {
                        self.driver_prepares
                            .insert((txn, digest), (ok_count, total, true));
                        self.driver_send_decide(txn, digest, TxnDecision::Commit, actions);
                    } else {
                        self.driver_prepares
                            .insert((txn, digest), (ok_count, total, false));
                    }
                } else {
                    self.driver_prepares
                        .insert((txn, digest), (ok_count, total, true));
                    self.driver_send_decide(txn, digest, TxnDecision::Abort, actions);
                }
            }
            TxnMsg::Decided { req, .. } => {
                // The decision is durable (by this send or a racing one):
                // re-read the record to learn which outcome governs, then
                // finalize accordingly. Never assume this send won.
                let Some((txn, digest)) = self.outstanding.get(req).copied() else {
                    return;
                };
                let writes = self
                    .driver_writes
                    .get(&(txn, digest))
                    .cloned()
                    .unwrap_or_default();
                if let Some(coordinator) = self.coordinator_for(&writes) {
                    let req = self.mint_req(txn, digest);
                    Self::send_to(coordinator, &TxnMsg::RecordRead { req, txn }, actions);
                }
            }
            TxnMsg::Finalized { req, .. } => {
                if let Some((txn, digest)) = self.outstanding.get(req).copied() {
                    *self.driver_finalizes.entry((txn, digest)).or_default() += 1;
                }
            }
            TxnMsg::RecordReply { req, decision, .. } => {
                let Some((txn, digest)) = self.outstanding.get(req).copied() else {
                    return;
                };
                self.driver_records.insert((txn, digest), *decision);
                self.driver_follow_record(txn, digest, *decision, actions);
            }
            _ => {}
        }
    }

    /// Follows a recovery reply for one attempt: a Commit decision for
    /// this attempt's digest finalizes its write set (apply); an Abort
    /// for it finalizes too (discard — missing intents are idempotent
    /// no-ops); anything else re-prepares the attempt (idempotent replays
    /// converge already-reserved keys). A decision for a different digest
    /// governs another attempt: this one cannot proceed and waits (its
    /// intents age out through the resolver path, which follows the
    /// winning record, never this one).
    fn driver_follow_record(
        &mut self,
        txn: TxnSimId,
        digest: TxnDigest,
        decision: Option<(TxnDecision, TxnDigest)>,
        actions: &mut Vec<ClusterAction<TxnDriverEv>>,
    ) {
        match decision {
            Some((TxnDecision::Commit, record_digest)) if record_digest == digest => {
                self.driver_send_finalizes(txn, digest, true, actions);
            }
            Some((TxnDecision::Abort, record_digest)) if record_digest == digest => {
                self.driver_send_finalizes(txn, digest, false, actions);
            }
            Some(_) => {}
            None => {
                self.driver_send_prepares(txn, digest, actions);
            }
        }
    }
}

impl AppHandler<TxnDriverEv, ()> for TxnSim {
    fn handle_app(
        &mut self,
        node: NodeId,
        event: TxnDriverEv,
        _state: &mut (),
        actions: &mut Vec<ClusterAction<TxnDriverEv>>,
        _rng: &mut dyn RandomSource,
    ) {
        match event {
            TxnDriverEv::Begin {
                txn,
                writes,
                digest,
            } => {
                debug_assert_eq!(node, Self::driver_node());
                self.driver_begin(txn, &writes, digest, actions);
            }
            TxnDriverEv::Recover {
                txn,
                writes,
                digest,
            } => {
                debug_assert_eq!(node, Self::driver_node());
                // Recovery-first: read the coordinator record before
                // touching any intent (re-prepares after a commit would
                // spuriously conflict with committed state). The
                // coordinator re-derives deterministically.
                self.driver_recover(txn, &writes, digest, actions);
            }
        }
    }

    fn handle_net(
        &mut self,
        from: Endpoint,
        to: NodeId,
        bytes: &[u8],
        _state: &mut (),
        actions: &mut Vec<ClusterAction<TxnDriverEv>>,
        _rng: &mut dyn RandomSource,
    ) {
        let Some(message) = decode_msg(bytes) else {
            return;
        };
        // The driver node only handles driver-side replies; participants
        // only handle participant-side messages (a real deployment
        // separates these by tablet role; the sim separates by node).
        if to == Self::driver_node() {
            self.on_driver_msg(from.node(), &message, actions);
            return;
        }
        match message {
            TxnMsg::Prepare {
                req,
                txn,
                coordinator,
                write,
                digest,
            } => {
                let ctx = StepCtx {
                    reply_to: from.node(),
                    host: to,
                    actions,
                };
                self.on_prepare(ctx, req, txn, coordinator, &write, digest);
            }
            TxnMsg::Finalize {
                req,
                txn,
                key,
                commit,
                digest,
            } => {
                let ctx = StepCtx {
                    reply_to: from.node(),
                    host: to,
                    actions,
                };
                self.on_finalize(ctx, req, txn, key, commit, digest);
            }
            TxnMsg::RecordRead { req, txn } => {
                let decision = self
                    .decisions
                    .get(&(to, txn))
                    .map(|(decision, digest, _)| (*decision, *digest));
                Self::send_to(
                    from.node(),
                    &TxnMsg::RecordReply { req, txn, decision },
                    actions,
                );
            }
            TxnMsg::Decide {
                req,
                txn,
                decision,
                digest,
            } => {
                // Coordinator record write: first-writer-wins (production:
                // absence-guarded CAS). A lost CAS replies not-stored; the
                // driver re-reads to learn which outcome governs. Cross-key
                // splits (same txn, different coordinators) are preserved
                // as-is for `NoSplitDecision` to condemn.
                let stored = if self.decisions.contains_key(&(to, txn)) {
                    false
                } else {
                    let seq = self.next_seq();
                    self.decisions.insert((to, txn), (decision, digest, seq));
                    true
                };
                Self::send_to(from.node(), &TxnMsg::Decided { req, txn, stored }, actions);
            }
            _ => {}
        }
    }
}

// ---------------------------------------------------------------------------
// Invariants
// ---------------------------------------------------------------------------

/// Asserts no partial commit while running: condemns applies that
/// contradict a durable Abort for the same digest, and applied writes
/// spanning digests within one transaction. Mid-flight partial
/// application (finalizes still traveling) is legal — terminal atomicity
/// is asserted at idle with [`assert_quiesced_atomic`].
///
/// # Errors
///
/// Returns [`Err`] naming the condemned transaction when an applied write
/// contradicts a durable Abort or spans digests.
pub fn invariant_no_partial_commit(sim: &TxnSim) -> Result<(), String> {
    for (txn, digest, key) in sim
        .applied
        .iter()
        .map(|((_, key), applied)| (applied.txn, applied.digest, *key))
    {
        let _ = key;
        let aborted = sim.decisions.values().any(|(decision, record_digest, _)| {
            *decision == TxnDecision::Abort && *record_digest == digest
        });
        if aborted {
            return Err(format!("txn {txn}: writes applied under a durable Abort"));
        }
    }
    // One transaction never applies under two digests.
    let mut digests: BTreeMap<TxnSimId, BTreeSet<TxnDigest>> = BTreeMap::new();
    for applied in sim.applied.values() {
        digests
            .entry(applied.txn)
            .or_default()
            .insert(applied.digest);
    }
    for (txn, set) in digests {
        if set.len() > 1 {
            return Err(format!("txn {txn}: applied writes span digests"));
        }
    }
    Ok(())
}

/// Terminal atomicity check at idle: every known write of `txn` is
/// applied exactly when the durable decision is Commit (and nothing is
/// applied otherwise), with no intents left behind.
///
/// # Errors
///
/// Returns [`Err`] naming the leftover intent or the key-set mismatch.
pub fn assert_quiesced_atomic(
    sim: &TxnSim,
    txn: TxnSimId,
    writes: &[TxnWrite],
    digest: TxnDigest,
    expect_commit: bool,
) -> Result<(), String> {
    for ((_, key), intent) in &sim.intents {
        if intent.txn == txn {
            return Err(format!("txn {txn}: intent left on key {key} at idle"));
        }
    }
    let mut applied_keys: BTreeSet<TxnKey> = BTreeSet::new();
    for ((_, key), applied) in &sim.applied {
        if applied.txn == txn && applied.digest == digest {
            applied_keys.insert(*key);
        }
    }
    let expected_keys: BTreeSet<TxnKey> = writes.iter().map(|write| write.key).collect();
    if expect_commit {
        if applied_keys != expected_keys {
            return Err(format!(
                "txn {txn}: committed keys {applied_keys:?} != writes {expected_keys:?}"
            ));
        }
    } else if !applied_keys.is_empty() {
        return Err(format!("txn {txn}: aborted but applied {applied_keys:?}"));
    }
    Ok(())
}

/// Asserts no split decision: all durable decisions for one transaction
/// agree on both outcome and digest — across coordinators too. A second
/// decision for one txn with a different digest (conflicting drivers
/// reusing an id, or a lineage split) is corruption, never a tie to
/// break. Honest flows decide once (first-writer-wins per record key, one
/// coordinator per write set), so this holds at every step, not just at
/// idle.
///
/// # Errors
///
/// Returns [`Err`] naming the transaction and its conflicting decisions.
pub fn invariant_no_split_decision(sim: &TxnSim) -> Result<(), String> {
    let mut by_txn: BTreeMap<TxnSimId, BTreeSet<(TxnDecision, TxnDigest, NodeId)>> =
        BTreeMap::new();
    for ((coordinator, txn), (decision, digest, _)) in &sim.decisions {
        by_txn
            .entry(*txn)
            .or_default()
            .insert((*decision, *digest, *coordinator));
    }
    // One coordinator holding one decision is the honest shape; anything
    // else (two coordinators, or two digests) condemns the run. The
    // coordinator rides in the set only to name the split.
    for (txn, set) in by_txn {
        let canonical: BTreeSet<(TxnDecision, TxnDigest)> = set
            .iter()
            .map(|(decision, digest, _)| (*decision, *digest))
            .collect();
        if canonical.len() > 1 {
            return Err(format!("txn {txn}: split decisions {set:?}"));
        }
    }
    Ok(())
}

/// Asserts no guessing: every applied write follows a durable Commit
/// decision for its digest, sequenced strictly before the apply.
///
/// # Errors
///
/// Returns [`Err`] naming the host, key, and transaction of the first
/// write applied without a preceding durable Commit.
pub fn invariant_no_guess(sim: &TxnSim) -> Result<(), String> {
    for ((host, key), applied) in &sim.applied {
        let mut found = false;
        for ((_, txn), (decision, digest, decided_seq)) in &sim.decisions {
            if *txn == applied.txn
                && *decision == TxnDecision::Commit
                && *digest == applied.digest
                && *decided_seq < applied.applied_seq
            {
                found = true;
                break;
            }
        }
        if !found {
            return Err(format!(
                "txn {} key {key} on node {} applied without a preceding durable Commit",
                applied.txn,
                host.as_u64()
            ));
        }
    }
    Ok(())
}
