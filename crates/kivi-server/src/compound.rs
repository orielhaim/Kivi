//! Cluster compound requests: single-tablet scans and single-tablet atomic
//! batches over Raft-backed tablets.
//!
//! A `Scan` answers the slice of `[start, end)` owned by the routed tablet
//! through a linearizable (`Latest`) or weak (`Any`) group read; the client
//! fans out across tablets and resumes by logical key. An `AtomicBatch`
//! commits as one ordered atomic record (`TxnCommitLocal`) through the
//! normal propose path: one validation, one consensus round, one ordered
//! apply — no coordinator record, no prepare/finalize waves. Multi-tablet
//! batches are rejected with `TxnTooLarge`; the client driver runs those
//! over `TxnPrepare`/`TxnFinalize`.
//!
//! `TxnPrepare`/`TxnFinalize` need no special handling here: they translate
//! to single-key operations and flow through the standard propose path with
//! the shaping arms in `cluster.rs`.

use kivi_protocol::{Opcode, Request, Response, ResponseBody, Status};
use kivi_state::{
    Operation, OperationResult, ScanDirection, ScanProjection, ScanSpec, TxnDriverPlan, TxnError,
    TxnId, plan_transaction, txn_write_from_wire,
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
        Some(kivi_state::ScannedValue::Fabric {
            fabric_id,
            logical_len,
            version,
        }) => kivi_protocol::ScanValueBody::Fabric {
            fabric_id: *fabric_id,
            logical_len: *logical_len,
            version: *version,
        },
        Some(kivi_state::ScannedValue::Oversize { logical_len }) => {
            kivi_protocol::ScanValueBody::Oversize {
                logical_len: *logical_len,
            }
        }
        Some(kivi_state::ScannedValue::Semantic { object, descriptor }) => {
            kivi_protocol::ScanValueBody::Semantic {
                object: object_to_tag(*object),
                descriptor: descriptor.to_vec(),
            }
        }
    }
}

/// Maps a semantic object type to its stable scan tag (mirrors the
/// embedded engine's mapping; tags mirror `ObjectType` wire tags).
fn object_to_tag(object: kivi_state::ObjectType) -> u8 {
    match object {
        kivi_state::ObjectType::Bytes => 1,
        kivi_state::ObjectType::StrictCounter => 2,
        kivi_state::ObjectType::CommutativeCounter => 3,
        kivi_state::ObjectType::BoundedCounter => 4,
        kivi_state::ObjectType::Semaphore => 5,
        kivi_state::ObjectType::Lease => 6,
        kivi_state::ObjectType::StreamShard => 7,
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
                status: Status::TxnCrossTablet,
                body: ResponseBody::Diagnostic(format!(
                    "batch spans {} tablets; drive it over TxnPrepare/TxnFinalize",
                    plan.groups.len()
                )),
            },
            Opcode::AtomicBatch,
        );
    }
    // No blanket intent fence: the atomic record validates per-key intents
    // precisely, so free keys stay servable while other transactions
    // resolve. Topology cutover still fences on unresolved intents.
    run_single_tablet_plan(shared, &plan).await
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

/// Executes a validated single-participant plan as one ordered atomic
/// record through the normal propose path: a single `TxnCommitLocal`
/// proposal carries the whole write set through one consensus round and
/// one ordered apply. No coordinator record, no prepare/finalize waves.
/// The step identity is stable per transaction (session = txn bytes,
/// seq = 1), so retries replay through dedup instead of re-executing.
async fn run_single_tablet_plan(
    shared: &ClusterShared,
    plan: &TxnDriverPlan,
) -> (Response, Opcode) {
    let session = SessionId::from_u128(u128::from_le_bytes(plan.txn.as_bytes()));
    let identity = MutationIdentity::new(
        kivi_types::RequestIdentity::new(session, RequestSeq::from_u64(1)),
        RequestSeq::from_u64(0),
    );
    let op = Operation::TxnCommitLocal {
        txn: plan.txn,
        writes: plan.writes.clone(),
    };
    match propose_step(shared, plan.coordinator, &op, identity).await {
        Ok(OperationResult::TxnLocalCommitted { versions }) => (
            Response {
                proof: None,
                status: Status::Ok,
                body: ResponseBody::AtomicCommitted {
                    versions: versions
                        .iter()
                        .map(|version| version.map(kivi_state::ObjectVersion::as_u64))
                        .collect(),
                },
            },
            Opcode::AtomicBatch,
        ),
        Ok(_) => (
            Response {
                proof: None,
                status: Status::Internal,
                body: ResponseBody::Diagnostic("unexpected local-commit outcome".to_owned()),
            },
            Opcode::AtomicBatch,
        ),
        // Terminal rejections and propose errors already shaped by
        // `propose_step`; ambiguous delivery is retryable under the same
        // transaction identity.
        Err(response) => (response, Opcode::AtomicBatch),
    }
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
        Ok(ProposeOutcome::Rejected { outcome }) => Err(super::cluster::shape_durable(
            &outcome,
            Opcode::AtomicBatch,
            tablet,
        )),
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
    // Salt 3 never aliases driver decide-transactions (commit 0 / abort
    // 1) or the absent-record CAS (salt 2): distinct intents, genuine OCC
    // on the key.
    let abort_txn = TxnId::derive(u128::from_le_bytes(txn.as_bytes()), 0, 3);
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
        // Record-key steps bind the zero digest (no user write set).
        digest: [0u8; 32],
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
        digest: [0u8; 32],
    };
    let _ = node
        .propose(tablet, &finalize, Some(identity()), None, now)
        .await;
}
