//! Compound request execution over local tablets: single-tablet scans and
//! single-tablet atomic batches.
//!
//! A `Scan` answers the slice of `[start, end)` owned by the routed tablet;
//! cross-tablet scans fan out from the client, which resumes by logical key
//! through the current directory (splits and merges need no cursor state).
//! An `AtomicBatch` commits as one ordered atomic record
//! (`TxnCommitLocal`): one validation, one WAL record, one durability
//! barrier, one ordered apply — no coordinator record, no prepare/finalize
//! waves, no 2PC recovery machinery. Multi-tablet batches are rejected
//! here with `TxnTooLarge`; the client driver runs those over
//! `TxnPrepare`/`TxnFinalize`.
//!
//! Both paths work in ephemeral mode (direct tablet execution) and durable
//! mode (each step admitted through the commit pipeline and awaited): the
//! driver never bypasses the WAL.

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use std::time::{Duration, Instant};

use kivi_protocol::{
    Opcode, RedirectInfo, Request, Response, ResponseBody, SCAN_ANY, SCAN_KEYS_AND_VALUES,
    SCAN_KEYS_ONLY, SCAN_LATEST_PER_TABLET, SCAN_REVERSE, Status,
};
use kivi_state::{
    DurableOutcome, OpError, Operation, OperationResult, ScanDirection, ScanProjection, ScanSpec,
    TxnDriverPlan, TxnError, TxnId, plan_transaction, txn_write_from_wire,
};
use kivi_tablet::{DirectorySnapshot, PartitionRange};
use kivi_types::{
    MutationIdentity, NamespaceId, RequestSeq, SessionId, TabletAuthority, TabletId, WorkerId,
};

use crate::clock::SystemClock;
use crate::commit::PendingEntry;
use crate::routing::RoutingSnapshot;
use crate::tablet::LiveTablet;
use crate::worker::WorkerDurability;

/// Server-side per-page caps (client budgets above these are clamped, never
/// an error: bounded pages are the contract, and callers page anyway).
const SERVER_SCAN_MAX_ITEMS: usize = 10_000;
/// Server-side per-page inline-value budget cap.
const SERVER_SCAN_MAX_BYTES: usize = 4 << 20;
/// Microsleep between durable-step polls (mirrors the outbox quantum).
const STEP_POLL: Duration = Duration::from_micros(250);
/// Bound on one durable step wait (then abort-safe failure).
const STEP_TIMEOUT: Duration = Duration::from_secs(30);

/// Tablet map shared by a worker's tasks (single-threaded `Rc`).
pub type TabletMap = Rc<RefCell<HashMap<TabletId, LiveTablet>>>;

/// Answers one `Scan` or `AtomicBatch` request against local tablets.
/// Returns the response plus its opcode for framing.
pub async fn handle_compound(
    tablets: &TabletMap,
    routing: &RoutingSnapshot,
    worker: WorkerId,
    endpoint_of: impl Fn(WorkerId) -> String,
    durability: Option<&Rc<RefCell<WorkerDurability>>>,
    fabric: &Rc<RefCell<crate::fabric::TabletFabric>>,
    request: &Request,
) -> (Response, Opcode) {
    match request.opcode {
        Opcode::Scan => (
            handle_scan(tablets, routing, worker, &endpoint_of, request),
            Opcode::Scan,
        ),
        Opcode::AtomicBatch => (
            drive_batch(
                tablets,
                routing,
                worker,
                &endpoint_of,
                durability,
                fabric,
                request,
            )
            .await,
            Opcode::AtomicBatch,
        ),
        _ => (
            Response {
                proof: None,
                status: Status::InvalidRequest,
                body: ResponseBody::Diagnostic("not a compound request".to_owned()),
            },
            request.opcode,
        ),
    }
}

// ---------------------------------------------------------------------------
// Scan
// ---------------------------------------------------------------------------

/// Answers one single-tablet scan page: routes the page window's cursor
/// through the current directory (never trusting a stale hint), clamps to
/// the serving tablet's range, and reports that range so the client routes
/// the next page by key.
#[allow(clippy::too_many_lines)]
fn handle_scan(
    tablets: &TabletMap,
    routing: &RoutingSnapshot,
    worker: WorkerId,
    endpoint_of: impl Fn(WorkerId) -> String,
    request: &Request,
) -> Response {
    use Status as S;
    let directory = routing.directory();
    if request.scan_max_items == 0 || request.scan_max_bytes == 0 {
        return invalid("scan budgets must be nonzero");
    }
    if directory.namespace() != request.namespace {
        return invalid("unknown namespace");
    }
    let direction = match request.scan_direction {
        kivi_protocol::SCAN_FORWARD => ScanDirection::Forward,
        SCAN_REVERSE => ScanDirection::Reverse,
        _ => return invalid("unknown scan direction"),
    };
    let projection = match request.scan_projection {
        SCAN_KEYS_ONLY => ScanProjection::KeysOnly,
        SCAN_KEYS_AND_VALUES => ScanProjection::KeysAndValues,
        _ => return invalid("unknown scan projection"),
    };
    if !matches!(request.scan_consistency, SCAN_LATEST_PER_TABLET | SCAN_ANY) {
        return invalid("unknown scan consistency");
    }
    if request
        .scan_start
        .as_ref()
        .is_some_and(|start| request.scan_end.as_ref().is_some_and(|end| start >= end))
    {
        return invalid("scan end must exceed start");
    }
    // Route the window cursor: forward pages start at `start` (or the first
    // key); reverse pages end at `end` (or the last key) and descend.
    let cursor: Vec<u8> = match direction {
        ScanDirection::Forward => request.scan_start.clone().unwrap_or_default(),
        ScanDirection::Reverse => match &request.scan_end {
            // `end` is exclusive: route by it and step down below.
            Some(end) => end.clone(),
            None => return scan_last_tablet(tablets, routing, worker, &endpoint_of, request),
        },
    };
    let Some(tablet) = route_cursor(directory, &cursor, direction) else {
        return scan_gap(directory, &cursor, request);
    };
    // Non-local tablets redirect exactly like point operations.
    match placement_redirect(routing, worker, &endpoint_of, tablets, tablet) {
        Placement::Local => {}
        Placement::Redirect(info) => {
            return Response {
                proof: None,
                status: if info.worker == worker {
                    S::NotLocal
                } else {
                    S::StaleRoute
                },
                body: ResponseBody::Redirect(info),
            };
        }
        Placement::Nowhere => {
            return Response {
                proof: None,
                status: S::NotLocal,
                body: ResponseBody::Diagnostic("tablet not live on owner".to_owned()),
            };
        }
    }
    let Some(descriptor) = directory.get(tablet) else {
        return invalid("unknown tablet");
    };
    let PartitionRange::Ordered(tablet_range) = descriptor.range().clone() else {
        return Response {
            proof: None,
            status: S::InvalidRequest,
            body: ResponseBody::Diagnostic("scan requires an ordered namespace".to_owned()),
        };
    };
    // Clamp the window to the serving tablet's range.
    let lower = max_bound(request.scan_start.as_deref(), Some(tablet_range.start()));
    let upper = min_bound(request.scan_end.as_deref(), tablet_range.end());
    if matches!((&lower, &upper), (Some(lo), Some(hi)) if lo >= hi) {
        // Window lies outside this tablet (routing edge): report the range
        // so the client advances by key, with no entries.
        return empty_page(tablet, &tablet_range, directory, true);
    }
    let spec = ScanSpec::new(
        lower,
        upper,
        direction,
        usize::try_from(request.scan_max_items)
            .unwrap_or(usize::MAX)
            .min(SERVER_SCAN_MAX_ITEMS),
        usize::try_from(request.scan_max_bytes)
            .unwrap_or(usize::MAX)
            .min(SERVER_SCAN_MAX_BYTES),
        projection,
    );
    let Ok(spec) = spec else {
        return invalid("invalid scan window");
    };
    let now = SystemClock::wall_now();
    let mut borrowed = tablets.borrow_mut();
    let Some(live) = borrowed.get_mut(&tablet) else {
        return Response {
            proof: None,
            status: S::NotLocal,
            body: ResponseBody::Diagnostic("tablet not live on owner".to_owned()),
        };
    };
    match live.scan_local(&spec, now) {
        Err(error) => Response {
            proof: None,
            status: S::InvalidRequest,
            body: ResponseBody::Diagnostic(error.to_string()),
        },
        Ok(page) => {
            let entries = page
                .entries
                .iter()
                .map(|entry| kivi_protocol::ScanEntryBody {
                    key: entry.key.as_bytes().to_vec(),
                    value: match &entry.value {
                        None => kivi_protocol::ScanValueBody::None,
                        Some(value) => scan_value_body(value),
                    },
                })
                .collect();
            Response {
                proof: None,
                status: S::Ok,
                body: ResponseBody::ScanPage {
                    entries,
                    exhausted: page.exhausted,
                    last_key: page.last_key.map(|key| key.as_bytes().to_vec()),
                    tablet: tablet.as_u64(),
                    range_start: tablet_range.start().to_vec(),
                    range_end: tablet_range.end().map(<[u8]>::to_vec),
                    dir_version: directory.version().as_u64(),
                },
            }
        }
    }
}

/// Routes a scan cursor to its tablet: forward cursors are inclusive lower
/// bounds; reverse cursors are exclusive upper bounds (step down into the
/// tablet strictly below when the cursor sits exactly on a boundary).
fn route_cursor(
    directory: &DirectorySnapshot,
    cursor: &[u8],
    direction: ScanDirection,
) -> Option<TabletId> {
    match direction {
        ScanDirection::Forward => directory.lookup_by_key(cursor),
        ScanDirection::Reverse => {
            if let Some(tablet) = directory.lookup_by_key(cursor) {
                // `cursor` is exclusive: when it opens a tablet, the page
                // belongs to the tablet strictly below.
                let descriptor = directory.get(tablet)?;
                let PartitionRange::Ordered(range) = descriptor.range() else {
                    return None;
                };
                if range.start() == cursor {
                    return previous_tablet(directory, cursor);
                }
                return Some(tablet);
            }
            // Cursor past the last tablet's end (or in a gap): the greatest
            // tablet starting strictly below the cursor owns the page.
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

/// Reverse scan with no upper bound: serve the last tablet's tail.
fn scan_last_tablet(
    tablets: &TabletMap,
    routing: &RoutingSnapshot,
    worker: WorkerId,
    endpoint_of: impl Fn(WorkerId) -> String,
    request: &Request,
) -> Response {
    let directory = routing.directory();
    let mut last: Option<TabletId> = None;
    let mut last_start: &[u8] = &[];
    for tablet in directory.tablets() {
        if !tablet.state().is_writable() {
            continue;
        }
        let PartitionRange::Ordered(range) = tablet.range() else {
            return invalid("scan requires an ordered namespace");
        };
        if last.is_none_or(|_| range.start() > last_start) {
            last_start = range.start();
            last = Some(tablet.id());
        }
    }
    let Some(tablet) = last else {
        return invalid("no ordered tablet covers the scan");
    };
    // The window clamps to the tail tablet inside the shared helper.
    handle_scan_window(tablets, routing, worker, &endpoint_of, request, tablet)
}

/// Shared tail of scan handling once the serving tablet is fixed.
fn handle_scan_window(
    tablets: &TabletMap,
    routing: &RoutingSnapshot,
    worker: WorkerId,
    endpoint_of: impl Fn(WorkerId) -> String,
    request: &Request,
    tablet: TabletId,
) -> Response {
    use Status as S;
    let directory = routing.directory();
    match placement_redirect(routing, worker, &endpoint_of, tablets, tablet) {
        Placement::Local => {}
        Placement::Redirect(info) => {
            return Response {
                proof: None,
                status: if info.worker == worker {
                    S::NotLocal
                } else {
                    S::StaleRoute
                },
                body: ResponseBody::Redirect(info),
            };
        }
        Placement::Nowhere => {
            return Response {
                proof: None,
                status: S::NotLocal,
                body: ResponseBody::Diagnostic("tablet not live on owner".to_owned()),
            };
        }
    }
    let Some(descriptor) = directory.get(tablet) else {
        return invalid("unknown tablet");
    };
    let PartitionRange::Ordered(tablet_range) = descriptor.range().clone() else {
        return invalid("scan requires an ordered namespace");
    };
    scan_tablet_slice(tablets, directory, tablet, &tablet_range, request)
}

/// Scans `[clamp(request, tablet_range)]` on a local tablet.
fn scan_tablet_slice(
    tablets: &TabletMap,
    directory: &DirectorySnapshot,
    tablet: TabletId,
    tablet_range: &kivi_tablet::OrderedRange,
    request: &Request,
) -> Response {
    use Status as S;
    let direction = match request.scan_direction {
        kivi_protocol::SCAN_FORWARD => ScanDirection::Forward,
        _ => ScanDirection::Reverse,
    };
    let projection = match request.scan_projection {
        SCAN_KEYS_AND_VALUES => ScanProjection::KeysAndValues,
        _ => ScanProjection::KeysOnly,
    };
    let lower = max_bound(request.scan_start.as_deref(), Some(tablet_range.start()));
    let upper = min_bound(request.scan_end.as_deref(), tablet_range.end());
    if matches!((&lower, &upper), (Some(lo), Some(hi)) if lo >= hi) {
        return empty_page(tablet, tablet_range, directory, true);
    }
    let Ok(spec) = ScanSpec::new(
        lower,
        upper,
        direction,
        usize::try_from(request.scan_max_items)
            .unwrap_or(usize::MAX)
            .min(SERVER_SCAN_MAX_ITEMS),
        usize::try_from(request.scan_max_bytes)
            .unwrap_or(usize::MAX)
            .min(SERVER_SCAN_MAX_BYTES),
        projection,
    ) else {
        return invalid("invalid scan window");
    };
    let now = SystemClock::wall_now();
    let mut borrowed = tablets.borrow_mut();
    let Some(live) = borrowed.get_mut(&tablet) else {
        return Response {
            proof: None,
            status: S::NotLocal,
            body: ResponseBody::Diagnostic("tablet not live on owner".to_owned()),
        };
    };
    match live.scan_local(&spec, now) {
        Err(error) => Response {
            proof: None,
            status: S::InvalidRequest,
            body: ResponseBody::Diagnostic(error.to_string()),
        },
        Ok(page) => Response {
            proof: None,
            status: S::Ok,
            body: ResponseBody::ScanPage {
                entries: page
                    .entries
                    .iter()
                    .map(|entry| kivi_protocol::ScanEntryBody {
                        key: entry.key.as_bytes().to_vec(),
                        value: match &entry.value {
                            None => kivi_protocol::ScanValueBody::None,
                            Some(value) => scan_value_body(value),
                        },
                    })
                    .collect(),
                exhausted: page.exhausted,
                last_key: page.last_key.map(|key| key.as_bytes().to_vec()),
                tablet: tablet.as_u64(),
                range_start: tablet_range.start().to_vec(),
                range_end: tablet_range.end().map(<[u8]>::to_vec),
                dir_version: directory.version().as_u64(),
            },
        },
    }
}

/// Empty page over a known tablet range (window/range miss): `exhausted`
/// tells the client whether to advance (`true`: move past this tablet).
fn empty_page(
    tablet: TabletId,
    tablet_range: &kivi_tablet::OrderedRange,
    directory: &DirectorySnapshot,
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
            dir_version: directory.version().as_u64(),
        },
    }
}

/// Gap page: no tablet covers the cursor (transient mid-split). The range
/// fields describe the gap so the client jumps it by key; `tablet` is the
/// zero sentinel (never a real tablet id).
fn scan_gap(directory: &DirectorySnapshot, cursor: &[u8], request: &Request) -> Response {
    let mut prev_end: Option<&[u8]> = None;
    let mut next_start: Option<Vec<u8>> = None;
    for tablet in directory.tablets() {
        if !tablet.state().is_writable() {
            continue;
        }
        let PartitionRange::Ordered(range) = tablet.range() else {
            return invalid("scan requires an ordered namespace");
        };
        if range.start() <= cursor && range.end().is_none_or(|end| cursor < end) {
            // Raced into coverage: shouldn't happen (caller found no
            // tablet), fail closed rather than misroute.
            return invalid("scan routing raced topology");
        }
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
    let _ = request;
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

/// Placement verdict for one tablet from this worker's view.
enum Placement {
    /// Owned here and live: execute.
    Local,
    /// Owned elsewhere (or hinted stale): redirect.
    Redirect(RedirectInfo),
    /// No placement or not live here.
    Nowhere,
}

/// Checks placement and liveness for one tablet.
fn placement_redirect(
    routing: &RoutingSnapshot,
    worker: WorkerId,
    endpoint_of: impl Fn(WorkerId) -> String,
    tablets: &TabletMap,
    tablet: TabletId,
) -> Placement {
    let directory = routing.directory();
    let Some(owner) = routing.placement().worker_of(tablet) else {
        return Placement::Nowhere;
    };
    if owner != worker {
        let descriptor = directory.get(tablet);
        let endpoint = endpoint_of(owner);
        let Some(descriptor) = descriptor else {
            return Placement::Nowhere;
        };
        let authority = tablets.borrow().get(&tablet).map_or_else(
            || {
                TabletAuthority::new(
                    tablet,
                    kivi_types::TabletEpoch::INITIAL,
                    kivi_types::WriteGuardGeneration::INITIAL,
                )
            },
            LiveTablet::authority,
        );
        let Ok(info) = RedirectInfo::new(
            routing.version(),
            tablet,
            authority.epoch(),
            owner,
            endpoint,
            descriptor.range().clone(),
        ) else {
            return Placement::Nowhere;
        };
        return Placement::Redirect(info);
    }
    if tablets.borrow().contains_key(&tablet) {
        Placement::Local
    } else {
        Placement::Nowhere
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

/// Projects one scanned value onto its wire body.
fn scan_value_body(value: &kivi_state::ScannedValue) -> kivi_protocol::ScanValueBody {
    match value {
        kivi_state::ScannedValue::Inline(bytes) => {
            kivi_protocol::ScanValueBody::Inline(bytes.to_vec())
        }
        kivi_state::ScannedValue::Counter(counter) => {
            kivi_protocol::ScanValueBody::Counter(*counter)
        }
        kivi_state::ScannedValue::Chunked {
            manifest,
            logical_len,
        } => kivi_protocol::ScanValueBody::Chunked {
            manifest: *manifest.as_bytes(),
            logical_len: *logical_len,
        },
        kivi_state::ScannedValue::Fabric {
            fabric_id,
            logical_len,
            version,
        } => kivi_protocol::ScanValueBody::Fabric {
            fabric_id: *fabric_id,
            logical_len: *logical_len,
            version: *version,
        },
        kivi_state::ScannedValue::Oversize { logical_len } => {
            kivi_protocol::ScanValueBody::Oversize {
                logical_len: *logical_len,
            }
        }
        kivi_state::ScannedValue::Semantic { object, descriptor } => {
            kivi_protocol::ScanValueBody::Semantic {
                object: object_to_tag(*object),
                descriptor: descriptor.to_vec(),
            }
        }
    }
}

/// Maps a semantic object type to its stable scan tag. The tags mirror
/// `kivi-state`'s `ObjectType` wire tags (1..7); unknown future types fail
/// closed at the call site's exhaustiveness check, never as a forged byte.
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

// ---------------------------------------------------------------------------
// Atomic batch (single-tablet fast path)
// ---------------------------------------------------------------------------

/// Stages medium batch puts into the fabric, rewriting them as
/// `PutFabric` and collecting the seal payloads the later `Commit` must
/// persist. Values past the chunk threshold stay caller's problem
/// (explicit streaming uploads); saturation fails the batch, never the
/// commit that would follow. The single caller returns the response
/// immediately, so the large error never sits in a `Result` chain.
#[allow(clippy::result_large_err)]
fn stage_batch_medium(
    plan: kivi_state::TxnDriverPlan,
    fabric: &Rc<RefCell<crate::fabric::TabletFabric>>,
) -> Result<(kivi_state::TxnDriverPlan, Vec<crate::fabric::StagedSeal>), Response> {
    let coordinator = plan.coordinator;
    let mut writes = Vec::with_capacity(plan.writes.len());
    let mut seals = Vec::new();
    for write in plan.writes {
        match write.kind {
            kivi_state::TxnWriteKind::Put(value)
                if value.len() > crate::fabric::FABRIC_INLINE_MAX =>
            {
                // Cap medium staging at the chunk threshold: larger
                // values ride the chunked path via explicit uploads.
                if (value.len() as u64) > kivi_chunk::policy::DEFAULT_INLINE_THRESHOLD {
                    return Err(Response {
                        proof: None,
                        status: Status::TxnTooLarge,
                        body: ResponseBody::Diagnostic(
                            "batch value exceeds the inline bound; stage through streaming upload"
                                .to_owned(),
                        ),
                    });
                }
                match fabric.borrow_mut().stage(coordinator, value.clone(), true) {
                    Ok((fabric_id, logical_len)) => {
                        seals.push(crate::fabric::StagedSeal {
                            fabric_id,
                            key: write.key.clone(),
                            bytes: value,
                        });
                        writes.push(kivi_state::TxnWrite {
                            key: write.key,
                            kind: kivi_state::TxnWriteKind::PutFabric {
                                fabric_id,
                                logical_len,
                                version: 0,
                            },
                            expect: write.expect,
                        });
                    }
                    Err(_) => {
                        return Err(Response {
                            proof: None,
                            status: Status::TxnCoordinatorUnavailable,
                            body: ResponseBody::Diagnostic(
                                "memory fabric saturated; retry".to_owned(),
                            ),
                        });
                    }
                }
            }
            _ => writes.push(write),
        }
    }
    Ok((kivi_state::TxnDriverPlan { writes, ..plan }, seals))
}

/// Drives one single-tablet atomic batch: plans (bounds, grouping,
/// coordinator, digest), rejects multi-tablet batches for the client 2PC
/// driver, then commits the whole set as one atomic record with a stable
/// per-transaction step identity (transport retries re-drive idempotently
/// through dedup).
async fn drive_batch(
    tablets: &TabletMap,
    routing: &RoutingSnapshot,
    worker: WorkerId,
    endpoint_of: impl Fn(WorkerId) -> String,
    durability: Option<&Rc<RefCell<WorkerDurability>>>,
    fabric: &Rc<RefCell<crate::fabric::TabletFabric>>,
    request: &Request,
) -> Response {
    use Status as S;
    let directory = routing.directory();
    if directory.namespace() != request.namespace {
        return invalid("unknown namespace");
    }
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
            Err(_) => return invalid("invalid batch write"),
        }
    }
    let route = |key: &[u8]| route_batch_key(directory, request.namespace, key);
    let plan = match plan_transaction(
        txn,
        writes,
        route,
        directory.version().as_u64(),
        request.namespace,
    ) {
        Ok(plan) => plan,
        Err(TxnError::TooLarge) => {
            return Response {
                proof: None,
                status: S::TxnTooLarge,
                body: ResponseBody::Diagnostic(
                    "batch is empty or exceeds transaction bounds".to_owned(),
                ),
            };
        }
        Err(TxnError::RoutingChanged) => {
            return Response {
                proof: None,
                status: S::TxnCoordinatorUnavailable,
                body: ResponseBody::Diagnostic(
                    "batch key routes nowhere; retry after topology settles".to_owned(),
                ),
            };
        }
        Err(_) => return invalid("invalid batch"),
    };
    if plan.groups.len() != 1 {
        return Response {
            proof: None,
            status: S::TxnCrossTablet,
            body: ResponseBody::Diagnostic(format!(
                "batch spans {} tablets; drive it over TxnPrepare/TxnFinalize",
                plan.groups.len()
            )),
        };
    }
    let tablet = plan.coordinator;
    match placement_redirect(routing, worker, &endpoint_of, tablets, tablet) {
        Placement::Local => {}
        Placement::Redirect(info) => {
            return Response {
                proof: None,
                status: if info.worker == worker {
                    S::NotLocal
                } else {
                    S::StaleRoute
                },
                body: ResponseBody::Redirect(info),
            };
        }
        Placement::Nowhere => {
            return Response {
                proof: None,
                status: S::NotLocal,
                body: ResponseBody::Diagnostic("tablet not live on owner".to_owned()),
            };
        }
    }
    // Stage medium transactional puts into the fabric before planning
    // (synchronous inserts): later `Commit` can never meet a durable
    // root with unavailable payload. Staging is tablet-agnostic here;
    // routing below may still reject cross-tablet sets.
    let (plan, seals) = match stage_batch_medium(plan, fabric) {
        Ok(staged) => staged,
        Err(response) => return response,
    };
    // No blanket intent fence: the atomic record validates per-key intents
    // precisely (a foreign reservation on key K conflicts only writes to
    // K), so free keys stay servable while other transactions resolve.
    // Topology cutover (split/merge/retire) still fences on unresolved
    // intents at the control layer.
    run_single_tablet_plan(tablets, durability, fabric, &plan, seals).await
}

/// Routes one batch key through the directory by namespace layout.
fn route_batch_key(
    directory: &DirectorySnapshot,
    namespace: NamespaceId,
    key: &[u8],
) -> Option<TabletId> {
    route_point_key(directory, namespace, key)
}

/// Routes one point key through a directory snapshot by the unified rule
/// (mirrored in `kivi-server/src/compound.rs` and
/// `kivi-server/src/cluster.rs`; keep the three in sync):
///
/// - Transaction record keys route to the embedded coordinator tablet
///   while it is active (fiat, not hash: fast paths stay local); after
///   it retires, normal routing finds the seeded record object.
/// - Projection keys route by their stripped primary (always co-located
///   with the primary).
/// - Non-unique index entry keys route by their embedded primary (always
///   co-located with the primary, so an `indexed_set` batch stays
///   single-tablet). Unique entries carry no primary in the key and route
///   by layout; unique indexed writes stay cross-tablet 2PC.
/// - Everything else routes by layout: ordered ranges by key bytes, hash
///   ranges by partition hash (snapshots are single-layout, so exactly one
///   lookup can hit).
#[must_use]
pub fn route_point_key(
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
    if let Some(primary) = kivi_state::parse_index_primary(key) {
        return route_normal_key(directory, namespace, &primary);
    }
    route_normal_key(directory, namespace, key)
}

/// Layout routing for ordinary keys (ordered by bytes, hash by hash).
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
/// record: a single [`TxnCommitLocal`](kivi_state::Operation::TxnCommitLocal)
/// step through the normal pipeline (one WAL record, one durability
/// barrier, one ordered apply), or one inline execution in ephemeral mode.
/// No coordinator record, no prepare/finalize waves, no 2PC recovery
/// machinery: same-tablet atomicity costs one mutation, not `2N+2`.
///
/// The step identity is stable per transaction (session = txn bytes,
/// seq = 1), so any transport retry or client re-drive replays through
/// dedup instead of re-executing. Ambiguous delivery answers
/// `TxnCoordinatorUnavailable`: re-driving the identical batch is safe.
async fn run_single_tablet_plan(
    tablets: &TabletMap,
    durability: Option<&Rc<RefCell<WorkerDurability>>>,
    fabric: &Rc<RefCell<crate::fabric::TabletFabric>>,
    plan: &TxnDriverPlan,
    seals: Vec<crate::fabric::StagedSeal>,
) -> Response {
    use Status as S;
    let session = SessionId::from_u128(u128::from_le_bytes(plan.txn.as_bytes()));
    let identity = MutationIdentity::new(
        kivi_types::RequestIdentity::new(session, RequestSeq::from_u64(1)),
        RequestSeq::from_u64(0),
    );
    let op = Operation::TxnCommitLocal {
        txn: plan.txn,
        writes: plan.writes.clone(),
    };
    match drive_one(
        tablets,
        durability,
        fabric,
        plan.coordinator,
        &op,
        identity,
        seals,
    )
    .await
    {
        Ok(OperationResult::TxnLocalCommitted { versions }) => Response {
            proof: None,
            status: S::Ok,
            body: ResponseBody::AtomicCommitted {
                versions: versions
                    .iter()
                    .map(|version| version.map(kivi_state::ObjectVersion::as_u64))
                    .collect(),
            },
        },
        Ok(_) => Response {
            proof: None,
            status: S::Internal,
            body: ResponseBody::Diagnostic("unexpected local-commit outcome".to_owned()),
        },
        Err(StepFault::Rejected(outcome)) => map_local_rejection(&outcome),
        Err(StepFault::Unavailable) => Response {
            proof: None,
            status: S::TxnCoordinatorUnavailable,
            body: ResponseBody::Diagnostic(
                "local commit ambiguous; re-drive the identical batch".to_owned(),
            ),
        },
    }
}

/// Maps a local-commit terminal rejection onto the wire (deterministic).
fn map_local_rejection(outcome: &DurableOutcome) -> Response {
    use Status as S;
    let (status, message) = match outcome {
        DurableOutcome::Rejected(OpError::TxnConflict) => (S::TxnConflict, "transaction conflict"),
        DurableOutcome::Rejected(OpError::WrongType { .. }) => (S::WrongType, "wrong type"),
        DurableOutcome::Rejected(OpError::CounterOverflow) => {
            (S::CounterOverflow, "counter overflow")
        }
        DurableOutcome::Rejected(OpError::BoundedExceeded) => (
            S::BoundedExceeded,
            "bounded counter would exceed owned rights",
        ),
        DurableOutcome::Rejected(OpError::SemaphoreExhausted) => {
            (S::SemaphoreExhausted, "semaphore capacity exhausted")
        }
        DurableOutcome::Rejected(OpError::LeaseConflict) => {
            (S::LeaseConflict, "lease held by another live owner")
        }
        DurableOutcome::Rejected(OpError::StaleFencing) => (S::StaleFencing, "stale fencing token"),
        DurableOutcome::Rejected(OpError::StreamFull) => (S::StreamFull, "stream shard full"),
        DurableOutcome::Rejected(OpError::NotFound) => (S::NotFound, "not found"),
        DurableOutcome::Rejected(OpError::StaleRangeBase) => {
            (S::InvalidRequest, "range base changed; retry")
        }
        DurableOutcome::Completed(_) => (S::Internal, "local commit completed without a record"),
        DurableOutcome::VersionExhausted => (S::VersionExhausted, "version exhausted"),
    };
    Response {
        proof: None,
        status,
        body: ResponseBody::Diagnostic(message.to_owned()),
    }
}

/// One validated step execution: ephemeral tablets execute inline;
/// durable tablets admit through the commit pipeline and await the proof.
async fn drive_one(
    tablets: &TabletMap,
    durability: Option<&Rc<RefCell<WorkerDurability>>>,
    fabric: &Rc<RefCell<crate::fabric::TabletFabric>>,
    tablet: TabletId,
    op: &Operation,
    identity: MutationIdentity,
    seals: Vec<crate::fabric::StagedSeal>,
) -> Result<OperationResult, StepFault> {
    let Some(durable) = durability else {
        let now = SystemClock::wall_now();
        let mut borrowed = tablets.borrow_mut();
        let live = borrowed.get_mut(&tablet).ok_or(StepFault::Unavailable)?;
        return live.execute(op, now).map_err(step_tablet_error);
    };
    // Durable: admit one pipeline entry and await its proof without ever
    // holding a borrow across the wait (sibling tasks on this thread keep
    // borrowing while we park).
    let (respond, receive) = crossbeam_channel::bounded::<crate::worker::WorkerResponse>(1);
    {
        let mut guard = durable.borrow_mut();
        let namespace = guard.namespace;
        guard.commit.admit(
            PendingEntry::new(
                tablet,
                op.clone(),
                SystemClock::wall_now(),
                Some(identity),
                respond,
            )
            .with_fabric_staged(seals),
        );
        guard.commit.poll(
            &mut tablets.borrow_mut(),
            namespace,
            &mut fabric.borrow_mut(),
        );
    }
    let start = Instant::now();
    loop {
        // Pump the pipeline each turn: our proof may already be sealed.
        {
            let mut guard = durable.borrow_mut();
            let namespace = guard.namespace;
            guard.commit.poll(
                &mut tablets.borrow_mut(),
                namespace,
                &mut fabric.borrow_mut(),
            );
        }
        match receive.try_recv() {
            Ok(Ok(result)) => return Ok(result),
            Ok(Err(error)) => return Err(step_worker_error(error)),
            Err(crossbeam_channel::TryRecvError::Empty) => {}
            Err(crossbeam_channel::TryRecvError::Disconnected) => {
                return Err(StepFault::Unavailable);
            }
        }
        if start.elapsed() >= STEP_TIMEOUT {
            return Err(StepFault::Unavailable);
        }
        compio::time::sleep(STEP_POLL).await;
    }
}

/// One driver step failure.
#[derive(Debug)]
enum StepFault {
    /// Deterministic terminal rejection (persisted, safe to surface).
    Rejected(DurableOutcome),
    /// Ambiguous (timeout, transport, overload): the step may or may not
    /// have executed; the driver aborts or re-drives, never assumes.
    Unavailable,
}

/// Maps tablet logic failures onto driver faults, mirroring durable
/// pipeline semantics: predictable rejections surface, everything else is
/// ambiguous (fail closed, never assume).
fn step_tablet_error(error: crate::tablet::TabletError) -> StepFault {
    use crate::tablet::TabletError as E;
    match error {
        E::Op(op) => StepFault::Rejected(DurableOutcome::Rejected(op)),
        E::Apply(kivi_state::ApplyError::TypeMismatch { expected, found }) => {
            StepFault::Rejected(DurableOutcome::Rejected(OpError::WrongType {
                expected,
                found,
            }))
        }
        E::Apply(kivi_state::ApplyError::CounterOverflow) => {
            StepFault::Rejected(DurableOutcome::Rejected(OpError::CounterOverflow))
        }
        E::Apply(kivi_state::ApplyError::BoundedExceeded) => {
            StepFault::Rejected(DurableOutcome::Rejected(OpError::BoundedExceeded))
        }
        E::Apply(kivi_state::ApplyError::SemaphoreExhausted) => {
            StepFault::Rejected(DurableOutcome::Rejected(OpError::SemaphoreExhausted))
        }
        E::Apply(kivi_state::ApplyError::LeaseConflict) => {
            StepFault::Rejected(DurableOutcome::Rejected(OpError::LeaseConflict))
        }
        E::Apply(kivi_state::ApplyError::StreamFull) => {
            StepFault::Rejected(DurableOutcome::Rejected(OpError::StreamFull))
        }
        E::Apply(kivi_state::ApplyError::VersionExhausted) => {
            StepFault::Rejected(DurableOutcome::VersionExhausted)
        }
        E::Apply(_) | E::AuthorityMismatch { .. } | E::CommitExhausted { .. } | E::OpScan(_) => {
            StepFault::Unavailable
        }
    }
}

/// Maps a worker-side step failure onto driver faults.
fn step_worker_error(error: crate::worker::WorkerRequestError) -> StepFault {
    use crate::worker::WorkerRequestError as E;
    match error {
        E::Tablet(error) => step_tablet_error(error),
        E::UnknownTablet { .. }
        | E::Storage(_)
        | E::ChunkStore(_)
        | E::Fabric(_)
        | E::DedupExpired
        | E::SessionOverloaded
        | E::InvalidRequest { .. } => StepFault::Unavailable,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kivi_state::Key;

    const NS: NamespaceId = NamespaceId::from_u64(1);

    fn tablet_map() -> (TabletMap, RoutingSnapshot) {
        let directory =
            DirectorySnapshot::static_ordered_tiles(NS, &[b"m".to_vec()]).expect("tiling builds");
        let placement = crate::routing::Placement::new([
            (TabletId::from_u64(1), WorkerId::from_u64(0)),
            (TabletId::from_u64(2), WorkerId::from_u64(0)),
        ]);
        let routing = RoutingSnapshot::build(NS, directory, placement, 1).expect("routing builds");
        let tablets: TabletMap = Rc::new(RefCell::new(HashMap::new()));
        for descriptor in routing.directory().tablets() {
            if !descriptor.state().is_writable() {
                continue;
            }
            let authority = TabletAuthority::new(
                descriptor.id(),
                kivi_types::TabletEpoch::INITIAL,
                kivi_types::WriteGuardGeneration::INITIAL,
            );
            let mut live =
                LiveTablet::from_descriptor(descriptor, authority).expect("tablet builds");
            live.set_ordered_indexing(true);
            tablets.borrow_mut().insert(descriptor.id(), live);
        }
        // Seed data on both tablets through the tablets directly.
        {
            let mut borrowed = tablets.borrow_mut();
            for (key, value) in [("a", "1"), ("b", "2"), ("z", "3")] {
                let tablet = routing
                    .directory()
                    .lookup_by_key(key.as_bytes())
                    .expect("covered");
                let live = borrowed.get_mut(&tablet).expect("live");
                live.execute(
                    &kivi_state::Operation::Set {
                        key: Key::from(key),
                        value: bytes::Bytes::from(value),
                    },
                    kivi_types::UnixMicros::from_micros(1_000_000),
                )
                .expect("seed works");
            }
        }
        (tablets, routing)
    }

    fn scan_request(start: Option<Vec<u8>>, end: Option<Vec<u8>>, max_items: u32) -> Request {
        Request {
            contract: kivi_types::ReadContract::Latest,
            namespace: NS,
            opcode: Opcode::Scan,
            hint: None,
            key: Vec::new(),
            value: None,
            delta: 0,
            expiry: 0,
            offset: 0,
            len: 0,
            condition: 0,
            expiry_policy: 0,
            identity: None,
            ack_floor: kivi_types::RequestSeq::from_u64(0),
            scan_start: start,
            scan_end: end,
            scan_direction: kivi_protocol::SCAN_FORWARD,
            scan_max_items: max_items,
            scan_max_bytes: 1 << 20,
            scan_projection: kivi_protocol::SCAN_KEYS_ONLY,
            scan_consistency: kivi_protocol::SCAN_LATEST_PER_TABLET,
            batch_txn: [0u8; 16],
            batch_writes: Vec::new(),
            txn_coordinator: 0,
            txn_commit: false,
            txn_digest: [0u8; 32],
            capacity: 0,
            holder: 0,
            permit: [0u8; 16],
            owner: 0,
            qty: 0,
            fencing: 0,
            ttl: 0,
            stream: [0u8; 16],
            shard: 0,
            partition: Vec::new(),
            share_min: 0,
            share_max: 0,
        }
    }

    #[test]
    fn compound_scan_serves_local_slice() {
        let (tablets, routing) = tablet_map();
        let request = scan_request(None, None, 100);
        let dir = tempfile::tempdir().expect("tempdir");
        let paths = crate::fabric::FabricPaths {
            root: dir.path().to_owned(),
        };
        let (fabric, guard) =
            crate::fabric::TabletFabric::open(WorkerId::from_u64(0), &paths, 1 << 20, 16 << 20)
                .expect("fabric opens");
        let fabric = std::rc::Rc::new(std::cell::RefCell::new(fabric));
        let (response, opcode) =
            compio::runtime::Runtime::new()
                .expect("runtime")
                .block_on(handle_compound(
                    &tablets,
                    &routing,
                    WorkerId::from_u64(0),
                    |_| String::new(),
                    None,
                    &fabric,
                    &request,
                ));
        guard.shutdown();
        assert_eq!(opcode, Opcode::Scan);
        match response.body {
            ResponseBody::ScanPage {
                entries, exhausted, ..
            } => {
                // First tablet holds a, b (page clamped to tablet 1).
                assert_eq!(entries.len(), 2);
                assert!(exhausted);
            }
            other => panic!("unexpected scan body: {other:?}"),
        }
    }
}
