//! Cluster compound requests: single-tablet scans and single-tablet atomic
//! batches over Raft-backed tablets.
//!
//! A `Scan` answers the slice of `[start, end)` owned by the routed tablet
//! through a linearizable (`Latest`) or weak (`Any`) group read; the client
//! fans out across tablets and resumes by logical key. An `AtomicBatch`
//! runs the uniform 2PC machinery (record + prepares + decide + finalizes)
//! with coordinator == sole participant through the normal propose path.
//! Multi-tablet batches are rejected with `TxnTooLarge`; the client driver
//! runs those over `TxnPrepare`/`TxnFinalize`.
//!
//! `TxnPrepare`/`TxnFinalize` need no special handling here: they translate
//! to single-key operations and flow through the standard propose path with
//! the shaping arms in `cluster.rs`.

use kivi_protocol::{Opcode, Request, Response, ResponseBody, Status};
use kivi_state::{
    Operation, OperationResult, ScanDirection, ScanProjection, ScanSpec, TxnDriverPlan, TxnError,
    TxnId, TxnState, plan_transaction, txn_record_key, txn_write_from_wire,
};
use kivi_tablet::{DirectorySnapshot, PartitionRange};
use kivi_types::{MutationIdentity, NamespaceId, ReadContract, RequestSeq, SessionId, TabletId};

use super::cluster::ClusterShared;

/// Server-side per-page caps (client budgets above these are clamped, never
/// an error: bounded pages are the contract, and callers page anyway).
const SERVER_SCAN_MAX_ITEMS: u32 = 10_000;
/// Server-side per-page inline-value budget cap.
const SERVER_SCAN_MAX_BYTES: u32 = 4 << 20;

/// Answers one `Scan` request: routes the page window's cursor through the
/// current directory, reads the serving tablet's slice, and reports that
/// tablet's range so the client routes the next page by key.
pub async fn handle_scan(shared: &ClusterShared, request: &Request) -> (Response, Opcode) {
    let directory = shared.directory_snapshot();
    if request.scan_max_items == 0 || request.scan_max_bytes == 0 {
        return (invalid("scan budgets must be nonzero"), Opcode::Scan);
    }
    if directory.namespace() != request.namespace {
        return (invalid("unknown namespace"), Opcode::Scan);
    }
    let direction = match request.scan_direction {
        kivi_protocol::SCAN_FORWARD => ScanDirection::Forward,
        kivi_protocol::SCAN_REVERSE => ScanDirection::Reverse,
        _ => return (invalid("unknown scan direction"), Opcode::Scan),
    };
    let projection = match request.scan_projection {
        kivi_protocol::SCAN_KEYS_ONLY => ScanProjection::KeysOnly,
        kivi_protocol::SCAN_KEYS_AND_VALUES => ScanProjection::KeysAndValues,
        _ => return (invalid("unknown scan projection"), Opcode::Scan),
    };
    if !matches!(
        request.scan_consistency,
        kivi_protocol::SCAN_LATEST_PER_TABLET | kivi_protocol::SCAN_ANY
    ) {
        return (invalid("unknown scan consistency"), Opcode::Scan);
    }
    if request
        .scan_start
        .as_ref()
        .is_some_and(|start| request.scan_end.as_ref().is_some_and(|end| start >= end))
    {
        return (invalid("scan end must exceed start"), Opcode::Scan);
    }
    let cursor: Vec<u8> = match direction {
        ScanDirection::Forward => request.scan_start.clone().unwrap_or_default(),
        ScanDirection::Reverse => match &request.scan_end {
            Some(end) => end.clone(),
            None => match last_tablet(&directory) {
                Some(tablet) => {
                    return scan_tablet(shared, &directory, tablet, request, direction, projection)
                        .await;
                }
                None => return (invalid("no ordered tablet covers the scan"), Opcode::Scan),
            },
        },
    };
    let Some(tablet) = route_scan_cursor(&directory, &cursor, direction) else {
        return (scan_gap(&directory, &cursor), Opcode::Scan);
    };
    scan_tablet(shared, &directory, tablet, request, direction, projection).await
}

/// Reads one tablet's clamped slice and shapes the page.
async fn scan_tablet(
    shared: &ClusterShared,
    directory: &DirectorySnapshot,
    tablet: TabletId,
    request: &Request,
    direction: ScanDirection,
    projection: ScanProjection,
) -> (Response, Opcode) {
    let Some(descriptor) = directory.get(tablet) else {
        return (invalid("unknown tablet"), Opcode::Scan);
    };
    let PartitionRange::Ordered(tablet_range) = descriptor.range().clone() else {
        return (
            Response {
                proof: None,
                status: Status::InvalidRequest,
                body: ResponseBody::Diagnostic("scan requires an ordered namespace".to_owned()),
            },
            Opcode::Scan,
        );
    };
    let lower = max_bound(request.scan_start.as_deref(), Some(tablet_range.start()));
    let upper = min_bound(request.scan_end.as_deref(), tablet_range.end());
    if matches!((&lower, &upper), (Some(lo), Some(hi)) if lo >= hi) {
        return (
            empty_page(tablet, &tablet_range, directory.version().as_u64(), true),
            Opcode::Scan,
        );
    }
    let spec = ScanSpec::new(
        lower,
        upper,
        direction,
        (request.scan_max_items.min(SERVER_SCAN_MAX_ITEMS)) as usize,
        (request.scan_max_bytes.min(SERVER_SCAN_MAX_BYTES)) as usize,
        projection,
    );
    let Ok(spec) = spec else {
        return (invalid("invalid scan window"), Opcode::Scan);
    };
    let contract = match request.scan_consistency {
        kivi_protocol::SCAN_ANY => ReadContract::Any,
        _ => ReadContract::Latest,
    };
    let ctx = super::cluster::read_ctx();
    match shared.node.scan(tablet, &spec, contract, ctx).await {
        Err(error) => (
            super::cluster::shape_read_error(shared, tablet, &error).await,
            Opcode::Scan,
        ),
        Ok(served) => (
            Response {
                proof: Some(served.receipt),
                status: Status::Ok,
                body: ResponseBody::ScanPage {
                    entries: served
                        .page
                        .entries
                        .iter()
                        .map(|entry| kivi_protocol::ScanEntryBody {
                            key: entry.key.as_bytes().to_vec(),
                            value: scan_value_body(entry.value.as_ref()),
                        })
                        .collect(),
                    exhausted: served.page.exhausted,
                    last_key: served.page.last_key.map(|key| key.as_bytes().to_vec()),
                    tablet: tablet.as_u64(),
                    range_start: tablet_range.start().to_vec(),
                    range_end: tablet_range.end().map(<[u8]>::to_vec),
                    dir_version: directory.version().as_u64(),
                },
            },
            Opcode::Scan,
        ),
    }
}

/// Projects one scanned value onto its wire body.
fn scan_value_body(value: Option<&kivi_state::ScannedValue>) -> kivi_protocol::ScanValueBody {
    match value {
        None => kivi_protocol::ScanValueBody::None,
        Some(kivi_state::ScannedValue::Inline(bytes)) => {
            kivi_protocol::ScanValueBody::Inline(bytes.to_vec())
        }
        Some(kivi_state::ScannedValue::Counter(counter)) => {
            kivi_protocol::ScanValueBody::Counter(*counter)
        }
        Some(kivi_state::ScannedValue::Chunked {
            manifest,
            logical_len,
        }) => kivi_protocol::ScanValueBody::Chunked {
            manifest: *manifest.as_bytes(),
            logical_len: *logical_len,
        },
        Some(kivi_state::ScannedValue::Oversize { logical_len }) => {
            kivi_protocol::ScanValueBody::Oversize {
                logical_len: *logical_len,
            }
        }
    }
}

/// Routes a scan cursor like the engine path: forward cursors are inclusive,
/// reverse cursors exclusive (step down on exact boundaries).
fn route_scan_cursor(
    directory: &DirectorySnapshot,
    cursor: &[u8],
    direction: ScanDirection,
) -> Option<TabletId> {
    match direction {
        ScanDirection::Forward => directory.lookup_by_key(cursor),
        ScanDirection::Reverse => {
            if let Some(tablet) = directory.lookup_by_key(cursor) {
                let descriptor = directory.get(tablet)?;
                let PartitionRange::Ordered(range) = descriptor.range() else {
                    return None;
                };
                if range.start() == cursor {
                    return previous_tablet(directory, cursor);
                }
                return Some(tablet);
            }
            previous_tablet(directory, cursor)
        }
    }
}

/// Greatest active ordered tablet starting strictly below `cursor`.
fn previous_tablet(directory: &DirectorySnapshot, cursor: &[u8]) -> Option<TabletId> {
    let mut best: Option<(&[u8], TabletId)> = None;
    for tablet in directory.tablets() {
        if !tablet.state().is_writable() {
            continue;
        }
        let PartitionRange::Ordered(range) = tablet.range() else {
            continue;
        };
        if range.start() < cursor && best.is_none_or(|(start, _)| range.start() > start) {
            best = Some((range.start(), tablet.id()));
        }
    }
    best.map(|(_, tablet)| tablet)
}

/// Last active ordered tablet (unbounded reverse tail).
fn last_tablet(directory: &DirectorySnapshot) -> Option<TabletId> {
    let mut best: Option<(&[u8], TabletId)> = None;
    for tablet in directory.tablets() {
        if !tablet.state().is_writable() {
            continue;
        }
        let PartitionRange::Ordered(range) = tablet.range() else {
            return None;
        };
        if best.is_none_or(|(start, _)| range.start() > start) {
            best = Some((range.start(), tablet.id()));
        }
    }
    best.map(|(_, tablet)| tablet)
}

/// Empty page over a known tablet range.
fn empty_page(
    tablet: TabletId,
    tablet_range: &kivi_tablet::OrderedRange,
    dir_version: u64,
    exhausted: bool,
) -> Response {
    Response {
        proof: None,
        status: Status::Ok,
        body: ResponseBody::ScanPage {
            entries: Vec::new(),
            exhausted,
            last_key: None,
            tablet: tablet.as_u64(),
            range_start: tablet_range.start().to_vec(),
            range_end: tablet_range.end().map(<[u8]>::to_vec),
            dir_version,
        },
    }
}

/// Gap page: no tablet covers the cursor (transient mid-split). Range
/// fields describe the gap; `tablet` is the zero sentinel.
fn scan_gap(directory: &DirectorySnapshot, cursor: &[u8]) -> Response {
    let mut prev_end: Option<&[u8]> = None;
    let mut next_start: Option<Vec<u8>> = None;
    for tablet in directory.tablets() {
        if !tablet.state().is_writable() {
            continue;
        }
        let PartitionRange::Ordered(range) = tablet.range() else {
            return invalid("scan requires an ordered namespace");
        };
        if range.start() > cursor
            && next_start
                .as_deref()
                .is_none_or(|next| range.start() < next)
        {
            next_start = Some(range.start().to_vec());
        }
        if let Some(end) = range.end()
            && end <= cursor
            && prev_end.is_none_or(|prev: &[u8]| end > prev)
        {
            prev_end = Some(end);
        }
    }
    Response {
        proof: None,
        status: Status::Ok,
        body: ResponseBody::ScanPage {
            entries: Vec::new(),
            exhausted: false,
            last_key: None,
            tablet: 0,
            range_start: prev_end.unwrap_or(&[]).to_vec(),
            range_end: next_start,
            dir_version: directory.version().as_u64(),
        },
    }
}

/// Inclusive-lower max of two optional bounds (`None` = unbounded below).
fn max_bound(first: Option<&[u8]>, second: Option<&[u8]>) -> Option<Vec<u8>> {
    match (first, second) {
        (None, None) => None,
        (Some(bound), None) | (None, Some(bound)) => Some(bound.to_vec()),
        (Some(left), Some(right)) => Some(left.max(right).to_vec()),
    }
}

/// Exclusive-upper min of two optional bounds (`None` = unbounded above).
fn min_bound(first: Option<&[u8]>, second: Option<&[u8]>) -> Option<Vec<u8>> {
    match (first, second) {
        (None, None) => None,
        (Some(bound), None) | (None, Some(bound)) => Some(bound.to_vec()),
        (Some(left), Some(right)) => Some(left.min(right).to_vec()),
    }
}

fn invalid(detail: &str) -> Response {
    Response {
        proof: None,
        status: Status::InvalidRequest,
        body: ResponseBody::Diagnostic(detail.to_owned()),
    }
}

// ---------------------------------------------------------------------------
// Atomic batch (single-tablet fast path over Raft)
// ---------------------------------------------------------------------------

/// Drives one single-tablet atomic batch through the normal propose path:
/// record(Begun), prepares, decide(Commit/Abort), finalizes. Multi-tablet
/// batches are rejected for the client 2PC driver; non-leader tablets
/// redirect so the leader drives.
pub async fn handle_batch(shared: &ClusterShared, request: &Request) -> (Response, Opcode) {
    let directory = shared.directory_snapshot();
    let txn = TxnId::from_bytes(request.batch_txn);
    let mut writes = Vec::with_capacity(request.batch_writes.len());
    for write in &request.batch_writes {
        match txn_write_from_wire(
            write.key.clone(),
            write.kind,
            write.value.clone(),
            write.delta,
            write.expect,
            write.expect_version,
        ) {
            Ok(write) => writes.push(write),
            Err(_) => return (invalid("invalid batch write"), Opcode::AtomicBatch),
        }
    }
    let namespace = request.namespace;
    let route = |key: &[u8]| route_batch_key(&directory, namespace, key);
    let plan = match plan_transaction(txn, writes, route, directory.version().as_u64(), namespace) {
        Ok(plan) => plan,
        Err(TxnError::TooLarge) => {
            return (
                Response {
                    proof: None,
                    status: Status::TxnTooLarge,
                    body: ResponseBody::Diagnostic(
                        "batch is empty or exceeds transaction bounds".to_owned(),
                    ),
                },
                Opcode::AtomicBatch,
            );
        }
        Err(TxnError::RoutingChanged) => {
            return (
                Response {
                    proof: None,
                    status: Status::TxnCoordinatorUnavailable,
                    body: ResponseBody::Diagnostic(
                        "batch key routes nowhere; retry after topology settles".to_owned(),
                    ),
                },
                Opcode::AtomicBatch,
            );
        }
        Err(_) => return (invalid("invalid batch"), Opcode::AtomicBatch),
    };
    if plan.groups.len() != 1 {
        return (
            Response {
                proof: None,
                status: Status::TxnTooLarge,
                body: ResponseBody::Diagnostic(format!(
                    "batch spans {} tablets; drive it over TxnPrepare/TxnFinalize",
                    plan.groups.len()
                )),
            },
            Opcode::AtomicBatch,
        );
    }
    if shared
        .node
        .tablet_intent_count(plan.coordinator)
        .await
        .is_some_and(|count| count > 0)
    {
        return (
            Response {
                proof: None,
                status: Status::TxnCoordinatorUnavailable,
                body: ResponseBody::Diagnostic(
                    "tablet holds unresolved intents; retry after recovery".to_owned(),
                ),
            },
            Opcode::AtomicBatch,
        );
    }
    run_single_tablet_plan(shared, &plan, request).await
}

/// Routes one batch key through the directory by the unified rule
/// (see `super::cluster::route_key`; record fiat, projection stripping,
/// then layout routing).
fn route_batch_key(
    directory: &DirectorySnapshot,
    namespace: NamespaceId,
    key: &[u8],
) -> Option<TabletId> {
    if let Some((coordinator, _)) = kivi_state::parse_txn_record_key(key)
        && directory
            .get(coordinator)
            .is_some_and(|descriptor| descriptor.state().is_writable())
    {
        return Some(coordinator);
    }
    if let Some(primary) = kivi_state::parse_projection_primary(key) {
        return route_normal_key(directory, namespace, primary);
    }
    route_normal_key(directory, namespace, key)
}

/// Layout routing for ordinary keys.
fn route_normal_key(
    directory: &DirectorySnapshot,
    namespace: NamespaceId,
    key: &[u8],
) -> Option<TabletId> {
    if let Some(tablet) = directory.lookup_by_key(key) {
        return Some(tablet);
    }
    let hash = kivi_state::PartitionHasher::V1.hash(namespace, key)?;
    directory.lookup_by_hash(hash)
}

/// Executes a validated single-participant plan through proposes:
/// prepares, guarded decide, finalizes. No `Begun` record exists in V1
/// (orphans without records abort by lease). Recovery-first: an existing
/// durable decision resolves without re-preparing.
async fn run_single_tablet_plan(
    shared: &ClusterShared,
    plan: &TxnDriverPlan,
    _request: &Request,
) -> (Response, Opcode) {
    let session = SessionId::from_u128(u128::from_le_bytes(plan.txn.as_bytes()));
    let tablet = plan.coordinator;
    let mut seq: u64 = 1;
    // Per-step identities derive from the TxnId session (stable across
    // transport retries → idempotent re-drive through dedup).
    let mut step_identity = || {
        let identity = MutationIdentity::new(
            kivi_types::RequestIdentity::new(session, RequestSeq::from_u64(seq)),
            RequestSeq::from_u64(0),
        );
        seq += 1;
        identity
    };
    let record_key = txn_record_key(plan.coordinator, plan.txn);
    if let Some(record) = read_record(shared, tablet, &record_key).await {
        match record.state {
            TxnState::Committed => {
                return finalize_wave(shared, tablet, plan, &mut step_identity).await;
            }
            TxnState::Aborted => {
                return (
                    Response {
                        proof: None,
                        status: Status::TxnAborted,
                        body: ResponseBody::Diagnostic("transaction aborted".to_owned()),
                    },
                    Opcode::AtomicBatch,
                );
            }
            TxnState::Begun => {}
        }
    }
    let prepared = match prepare_wave(shared, plan, &mut step_identity).await {
        Ok(prepared) => prepared,
        Err(abort) => {
            let (partial, abort) = *abort;
            return abort_then(shared, plan, &partial, session, &mut step_identity, abort).await;
        }
    };
    let mut decided = plan.record.clone();
    decided.state = TxnState::Committed;
    if !decide_commit(shared, tablet, plan, &decided).await {
        // Re-read: an ambiguous decide may actually have committed (never
        // blind-overwrite a decision with an abort).
        if let Some(record) = read_record(shared, tablet, &record_key).await
            && record.state == TxnState::Committed
        {
            return finalize_wave(shared, tablet, plan, &mut step_identity).await;
        }
        return abort_then(
            shared,
            plan,
            &prepared,
            session,
            &mut step_identity,
            Response {
                proof: None,
                status: Status::TxnAborted,
                body: ResponseBody::Diagnostic("commit undecided; aborted".to_owned()),
            },
        )
        .await;
    }
    finalize_wave(shared, tablet, plan, &mut step_identity).await
}

/// Prepare every key in request order (deterministic). Returns the
/// prepared indexes; on the first failure returns the partial prepared
/// set (the caller finalize-aborts those) plus the abort response. The
/// failure side is boxed: responses are large and failures are cold.
async fn prepare_wave(
    shared: &ClusterShared,
    plan: &TxnDriverPlan,
    step_identity: &mut impl FnMut() -> MutationIdentity,
) -> Result<Vec<usize>, Box<(Vec<usize>, Response)>> {
    let tablet = plan.coordinator;
    let mut prepared: Vec<usize> = Vec::with_capacity(plan.writes.len());
    for (index, write) in plan.writes.iter().enumerate() {
        let op = Operation::TxnPrepare {
            txn: plan.txn,
            coordinator: tablet,
            write: write.clone(),
        };
        match propose_step(shared, tablet, &op, step_identity()).await {
            Ok(OperationResult::TxnPrepared) => prepared.push(index),
            Ok(OperationResult::TxnConflict) => {
                return Err(Box::new((
                    prepared,
                    Response {
                        proof: None,
                        status: Status::TxnConflict,
                        body: ResponseBody::Diagnostic("transaction conflict".to_owned()),
                    },
                )));
            }
            Ok(_) => {
                return Err(Box::new((
                    prepared,
                    Response {
                        proof: None,
                        status: Status::Internal,
                        body: ResponseBody::Diagnostic("unexpected prepare outcome".to_owned()),
                    },
                )));
            }
            Err(response) => {
                return Err(Box::new((prepared, response)));
            }
        }
    }
    Ok(prepared)
}

/// Finalize-commit wave shared by fresh drives and recovery re-drives.
async fn finalize_wave(
    shared: &ClusterShared,
    tablet: TabletId,
    plan: &TxnDriverPlan,
    step_identity: &mut impl FnMut() -> MutationIdentity,
) -> (Response, Opcode) {
    let mut versions: Vec<Option<u64>> = vec![None; plan.writes.len()];
    for (order, write) in plan.writes.iter().enumerate() {
        let op = Operation::TxnFinalize {
            txn: plan.txn,
            key: write.key.clone(),
            commit: true,
        };
        match propose_step(shared, tablet, &op, step_identity()).await {
            // `applied:false` after a successful prepare + durable commit
            // means the resolver already finalized this intent
            // (idempotent convergence): success with no version.
            Ok(OperationResult::TxnFinalized {
                applied: true,
                version,
            }) => {
                versions[order] = version.map(kivi_state::ObjectVersion::as_u64);
            }
            Ok(OperationResult::TxnFinalized { applied: false, .. }) => {}
            Ok(_) => {
                return (
                    Response {
                        proof: None,
                        status: Status::Internal,
                        body: ResponseBody::Diagnostic("unexpected finalize outcome".to_owned()),
                    },
                    Opcode::AtomicBatch,
                );
            }
            Err(response) => return (response, Opcode::AtomicBatch),
        }
    }
    (
        Response {
            proof: None,
            status: Status::Ok,
            body: ResponseBody::AtomicCommitted { versions },
        },
        Opcode::AtomicBatch,
    )
}

/// Reads and decodes the coordinator record (`None` when absent or
/// undecodable). Transaction records are always lease-ineligible: the
/// 2PC outcome must integrate with the final committed participant
/// outcome, so coordinator reads run Lazy-ALR/conservative until
/// responder coverage integrates with transactions.
async fn read_record(
    shared: &ClusterShared,
    tablet: TabletId,
    record_key: &kivi_state::Key,
) -> Option<kivi_state::TxnRecord> {
    let ctx = super::cluster::read_ctx();
    match shared
        .node
        .read(
            tablet,
            &Operation::Get {
                key: record_key.clone(),
            },
            kivi_types::ReadContract::Latest,
            ctx,
            kivi_types::LeaseEligibility::ConservativeOnly,
        )
        .await
    {
        Ok(served) => match served.outcome {
            OperationResult::Value(Some(bytes)) => kivi_state::TxnRecord::decode(&bytes).ok(),
            _ => None,
        },
        _ => None,
    }
}

/// Persists the Commit decision guarded on absence (CAS): the record key
/// is unique per transaction, so a present record means a previous drive
/// already decided. The decide steps use dedicated sequence numbers
/// (8000/8001): the abort decide uses a different pair (9000/9001),
/// because dedup keys on (session, seq) alone and the two decisions must
/// never alias each other.
async fn decide_commit(
    shared: &ClusterShared,
    tablet: TabletId,
    plan: &TxnDriverPlan,
    decided: &kivi_state::TxnRecord,
) -> bool {
    decide_record(shared, tablet, plan, decided, 8000).await
}

/// Persists one terminal decision through an absence-guarded prepare +
/// commit-finalize on the record key (CAS — never a blind overwrite).
/// `seq_base`/`seq_base + 1` name the decide step identities.
async fn decide_record(
    shared: &ClusterShared,
    tablet: TabletId,
    plan: &TxnDriverPlan,
    decided: &kivi_state::TxnRecord,
    seq_base: u64,
) -> bool {
    let session = SessionId::from_u128(u128::from_le_bytes(plan.txn.as_bytes()));
    let prepare = Operation::TxnPrepare {
        txn: plan.txn,
        coordinator: tablet,
        write: kivi_state::TxnWrite {
            key: txn_record_key(plan.coordinator, plan.txn),
            kind: kivi_state::TxnWriteKind::Put(bytes::Bytes::from(decided.encode())),
            expect: kivi_state::TxnExpect::Absent,
        },
    };
    let identity = MutationIdentity::new(
        kivi_types::RequestIdentity::new(session, RequestSeq::from_u64(seq_base)),
        RequestSeq::from_u64(0),
    );
    if !matches!(
        propose_step(shared, tablet, &prepare, identity).await,
        Ok(OperationResult::TxnPrepared)
    ) {
        return false;
    }
    let finalize = Operation::TxnFinalize {
        txn: plan.txn,
        key: txn_record_key(plan.coordinator, plan.txn),
        commit: true,
    };
    let identity = MutationIdentity::new(
        kivi_types::RequestIdentity::new(session, RequestSeq::from_u64(seq_base + 1)),
        RequestSeq::from_u64(0),
    );
    matches!(
        propose_step(shared, tablet, &finalize, identity).await,
        Ok(OperationResult::TxnFinalized { .. })
    )
}

/// Which decision the coordinator record converged to after the guarded
/// abort path.
#[derive(Clone, Copy, PartialEq, Eq)]
enum AbortOutcome {
    Aborted,
    Committed,
}

/// Runs the guarded abort path and maps its outcome: `Aborted` yields the
/// caller's abort response; `Committed` (a racing drive committed this
/// transaction) converges through the commit wave instead of reporting a
/// false abort.
async fn abort_then(
    shared: &ClusterShared,
    plan: &TxnDriverPlan,
    prepared: &[usize],
    session: SessionId,
    step_identity: &mut impl FnMut() -> MutationIdentity,
    abort: Response,
) -> (Response, Opcode) {
    match abort_prepared(shared, plan, prepared, session).await {
        AbortOutcome::Aborted => (abort, Opcode::AtomicBatch),
        AbortOutcome::Committed => {
            finalize_wave(shared, plan.coordinator, plan, step_identity).await
        }
    }
}

/// Finalize-abort every prepared key plus a guarded Aborted decision
/// (CAS on absence — never a blind overwrite: a racing drive may have
/// committed this transaction, and an abort must not clobber that).
/// On a lost race the record is re-read: `Committed` means the caller
/// must converge via the commit wave instead.
async fn abort_prepared(
    shared: &ClusterShared,
    plan: &TxnDriverPlan,
    prepared: &[usize],
    session: SessionId,
) -> AbortOutcome {
    let tablet = plan.coordinator;
    let mut aborted = plan.record.clone();
    aborted.state = TxnState::Aborted;
    // Abort decide identities (9000/9001) never alias the commit decide
    // pair (8000/8001): dedup keys on (session, seq) alone.
    if decide_record(shared, tablet, plan, &aborted, 9000).await {
        for (seq, index) in (9100_u64..).zip(prepared.iter()) {
            let op = Operation::TxnFinalize {
                txn: plan.txn,
                key: plan.writes[*index].key.clone(),
                commit: false,
            };
            let identity = MutationIdentity::new(
                kivi_types::RequestIdentity::new(session, RequestSeq::from_u64(seq)),
                RequestSeq::from_u64(0),
            );
            let _ = propose_step(shared, tablet, &op, identity).await;
        }
        return AbortOutcome::Aborted;
    }
    // Lost the decide race: re-read (an ambiguous decide may actually
    // have committed — never finalize-abort a committed transaction).
    let record_key = txn_record_key(plan.coordinator, plan.txn);
    if read_record(shared, tablet, &record_key)
        .await
        .is_some_and(|record| record.state == TxnState::Committed)
    {
        return AbortOutcome::Committed;
    }
    AbortOutcome::Aborted
}

/// One validated propose: outcomes pass through, rejections and routing
/// failures shape onto the wire. Transport-level ambiguity (Overloaded,
/// leader-unknown) surfaces retryably; the driver aborts on anything that
/// is not a clean prepare. The `Response` error side is large by nature
/// (it is the wire type); boxing it would allocate on every conflict, so
/// the size is accepted here.
#[allow(clippy::result_large_err)]
async fn propose_step(
    shared: &ClusterShared,
    tablet: TabletId,
    op: &Operation,
    identity: MutationIdentity,
) -> Result<OperationResult, Response> {
    use kivi_consensus::ProposeOutcome;
    let now = super::cluster::wall_now();
    match shared
        .node
        .propose(tablet, op, Some(identity), None, now)
        .await
    {
        Ok(
            ProposeOutcome::Applied { outcome, .. }
            | ProposeOutcome::Duplicate { outcome }
            | ProposeOutcome::Read { outcome },
        ) => Ok(outcome),
        Ok(ProposeOutcome::Rejected { outcome }) => {
            Err(super::cluster::shape_durable(&outcome, Opcode::AtomicBatch))
        }
        Err(error) => Err(super::cluster::shape_propose_error(shared, tablet, &error).await),
    }
}

/// Guarded record abort for the transaction resolver (internal, never a
/// wire request): version-guarded prepare + commit-finalize of the Aborted
/// decision on the record tablet. Wins only when the record is untouched
/// (a concurrent commit conflicts instead, and the resolver follows that
/// decision next pass). Best-effort: any failure simply retries later.
pub async fn handle_guard_abort(
    node: &kivi_consensus::ConsensusNode,
    tablet: TabletId,
    record_key: kivi_state::Key,
    record_bytes: Vec<u8>,
    expected: kivi_state::ObjectVersion,
) {
    use kivi_state::{Operation, OperationResult, TxnExpect, TxnId, TxnWrite, TxnWriteKind};
    let Some((_, txn)) = kivi_state::parse_txn_record_key(record_key.as_bytes()) else {
        return;
    };
    // Deterministic abort-transaction id (re-drives idempotently).
    let abort_txn = TxnId::derive(u128::from_le_bytes(txn.as_bytes()), 0, 0);
    let session = SessionId::from_u128(u128::from_le_bytes(abort_txn.as_bytes()));
    let now = super::cluster::wall_now();
    let mut seq: u64 = 1;
    let mut identity = || {
        let id = MutationIdentity::new(
            kivi_types::RequestIdentity::new(session, RequestSeq::from_u64(seq)),
            RequestSeq::from_u64(0),
        );
        seq += 1;
        id
    };
    let prepare = Operation::TxnPrepare {
        txn: abort_txn,
        coordinator: tablet,
        write: TxnWrite {
            key: record_key.clone(),
            kind: TxnWriteKind::Put(bytes::Bytes::from(record_bytes.clone())),
            expect: TxnExpect::Version(expected),
        },
    };
    let prepared = matches!(
        node.propose(tablet, &prepare, Some(identity()), None, now)
            .await,
        Ok(kivi_consensus::ProposeOutcome::Applied {
            outcome: OperationResult::TxnPrepared,
            ..
        } | kivi_consensus::ProposeOutcome::Duplicate {
            outcome: OperationResult::TxnPrepared,
            ..
        },)
    );
    if !prepared {
        return;
    }
    let finalize = Operation::TxnFinalize {
        txn: abort_txn,
        key: record_key,
        commit: true,
    };
    let _ = node
        .propose(tablet, &finalize, Some(identity()), None, now)
        .await;
}
