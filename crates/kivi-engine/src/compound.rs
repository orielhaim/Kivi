//! Compound request execution over local tablets: single-tablet scans and
//! single-tablet atomic batches.
//!
//! A `Scan` answers the slice of `[start, end)` owned by the routed tablet;
//! cross-tablet scans fan out from the client, which resumes by logical key
//! through the current directory (splits and merges need no cursor state).
//! An `AtomicBatch` runs the uniform 2PC machinery (record + prepares +
//! decide + finalizes) with coordinator == sole participant: no cross-group
//! coordination, one client roundtrip. Multi-tablet batches are rejected
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
    DurableOutcome, Key, OpError, Operation, OperationResult, ScanDirection, ScanProjection,
    ScanSpec, TxnDriverPlan, TxnError, TxnId, TxnState, plan_transaction, txn_record_key,
    txn_write_from_wire,
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
    request: &Request,
) -> (Response, Opcode) {
    match request.opcode {
        Opcode::Scan => (
            handle_scan(tablets, routing, worker, &endpoint_of, request),
            Opcode::Scan,
        ),
        Opcode::AtomicBatch => (
            drive_batch(tablets, routing, worker, &endpoint_of, durability, request).await,
            Opcode::AtomicBatch,
        ),
        _ => (
            Response {
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
            status: S::NotLocal,
            body: ResponseBody::Diagnostic("tablet not live on owner".to_owned()),
        };
    };
    match live.scan_local(&spec, now) {
        Err(error) => Response {
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
            status: S::NotLocal,
            body: ResponseBody::Diagnostic("tablet not live on owner".to_owned()),
        };
    };
    match live.scan_local(&spec, now) {
        Err(error) => Response {
            status: S::InvalidRequest,
            body: ResponseBody::Diagnostic(error.to_string()),
        },
        Ok(page) => Response {
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
        kivi_state::ScannedValue::Oversize { logical_len } => {
            kivi_protocol::ScanValueBody::Oversize {
                logical_len: *logical_len,
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Atomic batch (single-tablet fast path)
// ---------------------------------------------------------------------------

/// Drives one single-tablet atomic batch: plans (bounds, grouping,
/// coordinator, record), rejects multi-tablet batches for the client 2PC
/// driver, then runs record + prepares + decide + finalizes against the
/// one participant group with per-step stable identities (transport
/// retries re-drive idempotently through dedup).
async fn drive_batch(
    tablets: &TabletMap,
    routing: &RoutingSnapshot,
    worker: WorkerId,
    endpoint_of: impl Fn(WorkerId) -> String,
    durability: Option<&Rc<RefCell<WorkerDurability>>>,
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
                status: S::TxnTooLarge,
                body: ResponseBody::Diagnostic(
                    "batch is empty or exceeds transaction bounds".to_owned(),
                ),
            };
        }
        Err(TxnError::RoutingChanged) => {
            return Response {
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
            status: S::TxnTooLarge,
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
                status: S::NotLocal,
                body: ResponseBody::Diagnostic("tablet not live on owner".to_owned()),
            };
        }
    }
    // Split/merge fence: tablets with unresolved intents do not take new
    // transactions (resolve first, then cut over).
    if tablets
        .borrow()
        .get(&tablet)
        .is_some_and(|live| live.pending_intent_count() > 0)
    {
        return Response {
            status: S::TxnCoordinatorUnavailable,
            body: ResponseBody::Diagnostic(
                "tablet holds unresolved intents; retry after recovery".to_owned(),
            ),
        };
    }
    run_single_tablet_plan(tablets, durability, &plan).await
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
/// (mirrored in `kivi-server/src/compound.rs`; keep the two in sync):
///
/// - Transaction record keys route to the embedded coordinator tablet
///   while it is active (fiat, not hash: fast paths stay local); after
///   it retires, normal routing finds the seeded record object.
/// - Projection keys route by their stripped primary (always co-located
///   with the primary).
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

/// Executes a validated single-participant plan: prepares, guarded
/// decide(Commit/Abort), finalizes. No `Begun` record exists in V1:
/// orphans without records abort by lease, decided records carry the
/// terminal state directly. Every step carries a stable identity derived
/// from the `TxnId` (session = txn bytes, seq = step index), so a
/// transport retry re-drives idempotently through dedup.
async fn run_single_tablet_plan(
    tablets: &TabletMap,
    durability: Option<&Rc<RefCell<WorkerDurability>>>,
    plan: &TxnDriverPlan,
) -> Response {
    use Status as S;
    let session = SessionId::from_u128(u128::from_le_bytes(plan.txn.as_bytes()));
    let tablet = plan.coordinator;
    let mut seq: u64 = 1;
    let mut step_identity = || {
        let identity = MutationIdentity::new(
            kivi_types::RequestIdentity::new(session, RequestSeq::from_u64(seq)),
            RequestSeq::from_u64(0),
        );
        seq += 1;
        identity
    };
    // Recovery-first: an existing durable decision resolves without
    // re-preparing (re-prepares after a commit would spuriously conflict
    // with the committed state).
    let record_key = txn_record_key(plan.coordinator, plan.txn);
    if let Some(record) = read_record(tablets, durability, tablet, &record_key).await {
        match record.state {
            TxnState::Committed => {
                return finalize_wave(tablets, durability, tablet, plan, &mut step_identity).await;
            }
            TxnState::Aborted => {
                return Response {
                    status: S::TxnAborted,
                    body: ResponseBody::Diagnostic("transaction aborted".to_owned()),
                };
            }
            TxnState::Begun => {}
        }
    }
    // 1. Prepare every key (request order for determinism).
    let prepared = match prepare_wave(tablets, durability, plan, &mut step_identity).await {
        Ok(prepared) => prepared,
        Err(abort) => {
            let (partial, abort) = *abort;
            return abort_then(
                tablets,
                durability,
                plan,
                &partial,
                session,
                &mut step_identity,
                abort,
            )
            .await;
        }
    };
    // 2. All prepared: persist the Commit decision, guarded on absence
    // (the record key is unique per transaction: present means a previous
    // drive already decided — recovery-first above would have caught a
    // commit, so re-read instead of overwriting).
    if !decide_commit(tablets, durability, tablet, plan).await {
        if let Some(record) = read_record(tablets, durability, tablet, &record_key).await
            && record.state == TxnState::Committed
        {
            return finalize_wave(tablets, durability, tablet, plan, &mut step_identity).await;
        }
        return abort_then(
            tablets,
            durability,
            plan,
            &prepared,
            session,
            &mut step_identity,
            Response {
                status: S::TxnAborted,
                body: ResponseBody::Diagnostic("commit undecided; aborted".to_owned()),
            },
        )
        .await;
    }
    // 3. Finalize-commit every key, collecting versions in request order.
    // `applied:false` after a successful prepare + durable commit means
    // the resolver already finalized this intent (idempotent convergence),
    // not a loss: the commit is durable, report success with no version.
    finalize_wave(tablets, durability, tablet, plan, &mut step_identity).await
}

/// Prepare every key in request order (deterministic). Returns the
/// prepared indexes; on the first failure returns the partial prepared
/// set (the caller finalize-aborts those) plus the abort response. The
/// failure side is boxed: responses are large and failures are cold.
async fn prepare_wave(
    tablets: &TabletMap,
    durability: Option<&Rc<RefCell<WorkerDurability>>>,
    plan: &TxnDriverPlan,
    step_identity: &mut impl FnMut() -> MutationIdentity,
) -> Result<Vec<usize>, Box<(Vec<usize>, Response)>> {
    use Status as S;
    let tablet = plan.coordinator;
    let mut prepared: Vec<usize> = Vec::with_capacity(plan.writes.len());
    for (index, write) in plan.writes.iter().enumerate() {
        let op = Operation::TxnPrepare {
            txn: plan.txn,
            coordinator: tablet,
            write: write.clone(),
        };
        match drive_one(tablets, durability, tablet, &op, step_identity()).await {
            Ok(OperationResult::TxnPrepared) => prepared.push(index),
            Ok(OperationResult::TxnConflict) => {
                return Err(Box::new((
                    prepared,
                    Response {
                        status: S::TxnConflict,
                        body: ResponseBody::Diagnostic("transaction conflict".to_owned()),
                    },
                )));
            }
            Ok(_) => {
                return Err(Box::new((
                    prepared,
                    Response {
                        status: S::Internal,
                        body: ResponseBody::Diagnostic("unexpected prepare outcome".to_owned()),
                    },
                )));
            }
            Err(StepFault::Rejected(error)) => {
                return Err(Box::new((prepared, map_prepare_rejection(&error))));
            }
            Err(StepFault::Unavailable) => {
                // Ambiguous prepare: the intent may or may not exist. No
                // decision ran, so the resolver aborts the orphan by
                // lease; report abort (never a partial commit).
                return Err(Box::new((
                    prepared,
                    Response {
                        status: S::TxnAborted,
                        body: ResponseBody::Diagnostic("prepare ambiguous; aborted".to_owned()),
                    },
                )));
            }
        }
    }
    Ok(prepared)
}

/// Finalize-commit wave shared by fresh drives and recovery re-drives.
async fn finalize_wave(
    tablets: &TabletMap,
    durability: Option<&Rc<RefCell<WorkerDurability>>>,
    tablet: TabletId,
    plan: &TxnDriverPlan,
    step_identity: &mut impl FnMut() -> MutationIdentity,
) -> Response {
    use Status as S;
    let mut versions: Vec<Option<u64>> = vec![None; plan.writes.len()];
    for (order, write) in plan.writes.iter().enumerate() {
        let op = Operation::TxnFinalize {
            txn: plan.txn,
            key: write.key.clone(),
            commit: true,
        };
        match drive_one(tablets, durability, tablet, &op, step_identity()).await {
            Ok(OperationResult::TxnFinalized {
                applied: true,
                version,
            }) => {
                versions[order] = version.map(kivi_state::ObjectVersion::as_u64);
            }
            Ok(OperationResult::TxnFinalized { applied: false, .. }) => {}
            Ok(_) => {
                return Response {
                    status: S::Internal,
                    body: ResponseBody::Diagnostic("unexpected finalize outcome".to_owned()),
                };
            }
            Err(_) => {
                // Commit is durable: the resolver replays missing
                // finalizes from the record. Report unavailable so the
                // client re-drives (idempotent) or waits out recovery.
                return Response {
                    status: S::TxnCoordinatorUnavailable,
                    body: ResponseBody::Diagnostic(
                        "finalize ambiguous; commit durable, retry".to_owned(),
                    ),
                };
            }
        }
    }
    Response {
        status: S::Ok,
        body: ResponseBody::AtomicCommitted { versions },
    }
}

/// Reads and decodes the coordinator record (`None` when absent or
/// undecodable).
async fn read_record(
    tablets: &TabletMap,
    durability: Option<&Rc<RefCell<WorkerDurability>>>,
    tablet: TabletId,
    record_key: &Key,
) -> Option<kivi_state::TxnRecord> {
    let op = Operation::Get {
        key: record_key.clone(),
    };
    let identity = MutationIdentity::new(
        kivi_types::RequestIdentity::new(
            SessionId::from_u128(0xC0DE_0000_0000_0002),
            RequestSeq::from_u64(1),
        ),
        RequestSeq::from_u64(0),
    );
    match drive_one(tablets, durability, tablet, &op, identity).await {
        Ok(OperationResult::Value(Some(bytes))) => kivi_state::TxnRecord::decode(&bytes).ok(),
        _ => None,
    }
}

/// Persists the Commit decision guarded on absence (CAS): the record key
/// is unique per transaction, so a present record means a previous drive
/// already decided. Returns whether the commit decision is durable.
///
/// The decide steps use dedicated sequence numbers (8000/8001) outside
/// the caller's step sequence: the abort decide uses a different pair
/// (9000/9001), because dedup is keyed on (session, seq) alone and the two
/// decisions must never alias each other.
async fn decide_commit(
    tablets: &TabletMap,
    durability: Option<&Rc<RefCell<WorkerDurability>>>,
    tablet: TabletId,
    plan: &TxnDriverPlan,
) -> bool {
    decide_record(tablets, durability, tablet, plan, TxnState::Committed, 8000).await
}

/// Persists a terminal decision (`Committed` or `Aborted`) guarded on
/// absence (CAS): the record key is unique per transaction, so a present
/// record means a previous drive already decided. Returns whether this
/// call persisted the decision. `seq_base`/`seq_base + 1` name the decide
/// step identities (commit and abort decides must use different pairs —
/// dedup keys on (session, seq) alone).
async fn decide_record(
    tablets: &TabletMap,
    durability: Option<&Rc<RefCell<WorkerDurability>>>,
    tablet: TabletId,
    plan: &TxnDriverPlan,
    state: TxnState,
    seq_base: u64,
) -> bool {
    let mut decided = plan.record.clone();
    decided.state = state;
    let prepare = Operation::TxnPrepare {
        txn: plan.txn,
        coordinator: tablet,
        write: kivi_state::TxnWrite {
            key: txn_record_key(plan.coordinator, plan.txn),
            kind: kivi_state::TxnWriteKind::Put(bytes::Bytes::from(decided.encode())),
            expect: kivi_state::TxnExpect::Absent,
        },
    };
    // The decide prepare runs under a fresh step identity (outside the
    // caller's step sequence; determinism comes from the absence guard,
    // not the identity).
    let identity = MutationIdentity::new(
        kivi_types::RequestIdentity::new(
            SessionId::from_u128(u128::from_le_bytes(plan.txn.as_bytes())),
            RequestSeq::from_u64(seq_base),
        ),
        RequestSeq::from_u64(0),
    );
    if !matches!(
        drive_one(tablets, durability, tablet, &prepare, identity).await,
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
        kivi_types::RequestIdentity::new(
            SessionId::from_u128(u128::from_le_bytes(plan.txn.as_bytes())),
            RequestSeq::from_u64(seq_base + 1),
        ),
        RequestSeq::from_u64(0),
    );
    matches!(
        drive_one(tablets, durability, tablet, &finalize, identity).await,
        Ok(OperationResult::TxnFinalized { .. })
    )
}

/// Outcome of the guarded abort path: which decision the coordinator
/// record converged to.
#[derive(Clone, Copy, PartialEq, Eq)]
enum AbortOutcome {
    Aborted,
    Committed,
}

/// Runs the guarded abort path and maps its outcome: `Aborted` yields the
/// caller's abort response; `Committed` (a racing drive committed this
/// transaction) converges through the commit wave instead of reporting a
/// false abort.
#[allow(clippy::too_many_arguments)]
async fn abort_then(
    tablets: &TabletMap,
    durability: Option<&Rc<RefCell<WorkerDurability>>>,
    plan: &TxnDriverPlan,
    prepared: &[usize],
    session: SessionId,
    step_identity: &mut impl FnMut() -> MutationIdentity,
    abort: Response,
) -> Response {
    match abort_prepared(tablets, durability, plan, prepared, session).await {
        AbortOutcome::Aborted => abort,
        AbortOutcome::Committed => {
            finalize_wave(tablets, durability, plan.coordinator, plan, step_identity).await
        }
    }
}

/// Finalize-abort every prepared key plus a guarded Aborted decision
/// (CAS on absence — never a blind overwrite: a racing drive may have
/// committed this transaction, and an abort must not clobber that).
/// On a lost race the record is re-read: `Committed` means the caller
/// must converge via the commit wave instead.
async fn abort_prepared(
    tablets: &TabletMap,
    durability: Option<&Rc<RefCell<WorkerDurability>>>,
    plan: &TxnDriverPlan,
    prepared: &[usize],
    session: SessionId,
) -> AbortOutcome {
    let tablet = plan.coordinator;
    // Decide Abort first so any racing resolver discards rather than waits.
    // Abort decide identities (9000/9001) never alias the commit decide
    // pair (8000/8001): dedup keys on (session, seq) alone.
    if decide_record(tablets, durability, tablet, plan, TxnState::Aborted, 9000).await {
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
            let _ = drive_one(tablets, durability, tablet, &op, identity).await;
        }
        return AbortOutcome::Aborted;
    }
    // Lost the decide race: re-read (an ambiguous decide may actually
    // have committed — never finalize-abort a committed transaction).
    let record_key = txn_record_key(plan.coordinator, plan.txn);
    if read_record(tablets, durability, tablet, &record_key)
        .await
        .is_some_and(|record| record.state == TxnState::Committed)
    {
        return AbortOutcome::Committed;
    }
    AbortOutcome::Aborted
}

/// Maps a prepare rejection onto the wire (terminal, deterministic).
fn map_prepare_rejection(error: &DurableOutcome) -> Response {
    match error {
        DurableOutcome::Rejected(OpError::TxnConflict) => Response {
            status: Status::TxnConflict,
            body: ResponseBody::Diagnostic("transaction conflict".to_owned()),
        },
        DurableOutcome::Rejected(OpError::WrongType { .. }) => Response {
            status: Status::WrongType,
            body: ResponseBody::Diagnostic("wrong type".to_owned()),
        },
        DurableOutcome::Rejected(OpError::CounterOverflow) => Response {
            status: Status::CounterOverflow,
            body: ResponseBody::Diagnostic("counter overflow".to_owned()),
        },
        _ => Response {
            status: Status::Internal,
            body: ResponseBody::Diagnostic("prepare rejected".to_owned()),
        },
    }
}

/// One validated step execution: ephemeral tablets execute inline;
/// durable tablets admit through the commit pipeline and await the proof.
async fn drive_one(
    tablets: &TabletMap,
    durability: Option<&Rc<RefCell<WorkerDurability>>>,
    tablet: TabletId,
    op: &Operation,
    identity: MutationIdentity,
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
        guard.commit.admit(PendingEntry::new(
            tablet,
            op.clone(),
            SystemClock::wall_now(),
            Some(identity),
            respond,
        ));
        guard.commit.poll(&mut tablets.borrow_mut(), namespace);
    }
    let start = Instant::now();
    loop {
        // Pump the pipeline each turn: our proof may already be sealed.
        {
            let mut guard = durable.borrow_mut();
            let namespace = guard.namespace;
            guard.commit.poll(&mut tablets.borrow_mut(), namespace);
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
        | E::DedupExpired
        | E::SessionOverloaded
        | E::InvalidRequest { .. } => StepFault::Unavailable,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        }
    }

    #[test]
    fn compound_scan_serves_local_slice() {
        let (tablets, routing) = tablet_map();
        let request = scan_request(None, None, 100);
        let (response, opcode) =
            compio::runtime::Runtime::new()
                .expect("runtime")
                .block_on(handle_compound(
                    &tablets,
                    &routing,
                    WorkerId::from_u64(0),
                    |_| String::new(),
                    None,
                    &request,
                ));
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
