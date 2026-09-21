//! Ordered data-plane client: streaming cross-tablet range scans,
//! atomic batches (single-tablet fast path plus client-driven OCC + 2PC),
//! and secondary index maintenance and queries.
//!
//! Scans walk the directory by logical key: each page resumes after the
//! last emitted key through the current server directory, so splits,
//! merges, and migrations cause at most one extra hop — never duplicates
//! or omissions. No central coordinator exists; the cursor (namespace,
//! direction, last key, bounds) is the only continuation truth.
//!
//! Transactions use one `TxnId` per attempt (stable across transport
//! retries, fresh per OCC retry — transport retry vs transaction retry
//! stay distinct). The single-tablet fast path runs server-side in one
//! roundtrip; multi-tablet batches run client-driven 2PC over
//! `TxnPrepare`/`TxnFinalize` with coordinator records for recovery.

use std::time::Duration;

use bytes::Bytes;
use kivi_state::{
    IndexId, IndexKind, Key, TxnId, TxnRecord, TxnState, decode_entry_value, decode_non_unique_key,
    encode_non_unique_key, encode_non_unique_value, encode_unique_key, encode_unique_value,
    parse_index_primary, parse_projection_primary, projection_key,
};
use kivi_types::{NamespaceId, TabletId};

use super::{ClientError, NativeClient, backoff, status_error};

// ---------------------------------------------------------------------------
// Scan
// ---------------------------------------------------------------------------

/// Scan direction over `[start, end)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ScanDirection {
    /// Ascending key order.
    Forward,
    /// Descending key order.
    Reverse,
}

/// Scan consistency contract (honest, no global snapshot claimed).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ScanConsistency {
    /// Strong `Latest` read per tablet. The full multi-tablet scan is NOT
    /// one global point-in-time snapshot: each tablet read is strong, but
    /// concurrent writes may appear on some tablets and not others.
    LatestPerTablet,
    /// Weakest local read per the tablet's existing contracts.
    Any,
}

/// What a scan returns per key.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ScanProjection {
    /// Keys only (no value bytes cross the wire).
    KeysOnly,
    /// Keys with bounded values.
    KeysAndValues,
}

/// Options for one scan.
#[derive(Debug, Clone)]
pub struct ScanOptions {
    /// Inclusive lower bound (`None` = first key).
    pub start: Option<Vec<u8>>,
    /// Exclusive upper bound (`None` = last key).
    pub end: Option<Vec<u8>>,
    /// Key order.
    pub direction: ScanDirection,
    /// Consistency per tablet.
    pub consistency: ScanConsistency,
    /// Per-key payload shape.
    pub projection: ScanProjection,
    /// Maximum entries per page (clamped server-side too).
    pub max_items_per_page: u32,
    /// Inline value budget per page (clamped server-side too).
    pub max_bytes_per_page: u32,
    /// Maximum total entries to collect (`None` = unbounded; callers
    /// streaming large keyspaces should page with [`ScanCursor`]).
    pub limit: Option<usize>,
}

impl Default for ScanOptions {
    fn default() -> Self {
        Self {
            start: None,
            end: None,
            direction: ScanDirection::Forward,
            consistency: ScanConsistency::LatestPerTablet,
            projection: ScanProjection::KeysAndValues,
            max_items_per_page: 1000,
            max_bytes_per_page: 1 << 20,
            limit: None,
        }
    }
}

/// One scanned entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClientScanEntry {
    /// Scanned key.
    pub key: Vec<u8>,
    /// Projected value.
    pub value: ClientScanValue,
}

/// Value payload of [`ClientScanEntry`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ClientScanValue {
    /// No value (`KeysOnly`).
    Absent,
    /// Resident bytes.
    Inline(Bytes),
    /// Counter value.
    Counter(i64),
    /// Chunked bytes by reference (resolve via `get_stream`).
    Chunked {
        /// Manifest addressing the immutable chunk sequence.
        manifest: [u8; 32],
        /// Total logical bytes.
        logical_len: u64,
    },
    /// Fabric bytes by reference (resolve via point `get`, which
    /// promotes transparently). Worker-local to the serving node;
    /// re-route by key after any topology change.
    Fabric {
        /// Fabric-scoped object id assigned at staging.
        fabric_id: u64,
        /// Total logical bytes.
        logical_len: u64,
        /// Logical version the bytes were published at.
        version: u64,
    },
    /// Resident bytes exceeding the page budget (length only).
    Oversize {
        /// Logical length.
        logical_len: u64,
    },
    /// Typed semantic value by small descriptor (object type tag plus
    /// canonical descriptor bytes; never inlined bulk).
    Semantic {
        /// Object type tag (mirrors `kivi-state`'s `ObjectType` tags).
        object: u8,
        /// Small canonical descriptor bytes for `object`.
        descriptor: Bytes,
    },
}

/// Maps one wire scan value onto its client shape (shared by page
/// decoding and per-tablet term fan-out, so both paths project
/// identically).
#[must_use]
fn scan_body_to_client(body: kivi_protocol::ScanValueBody) -> ClientScanValue {
    match body {
        kivi_protocol::ScanValueBody::Inline(bytes) => ClientScanValue::Inline(Bytes::from(bytes)),
        kivi_protocol::ScanValueBody::Counter(counter) => ClientScanValue::Counter(counter),
        kivi_protocol::ScanValueBody::Chunked {
            manifest,
            logical_len,
        } => ClientScanValue::Chunked {
            manifest,
            logical_len,
        },
        kivi_protocol::ScanValueBody::Fabric {
            fabric_id,
            logical_len,
            version,
        } => ClientScanValue::Fabric {
            fabric_id,
            logical_len,
            version,
        },
        kivi_protocol::ScanValueBody::Oversize { logical_len } => {
            ClientScanValue::Oversize { logical_len }
        }
        kivi_protocol::ScanValueBody::Semantic { object, descriptor } => {
            ClientScanValue::Semantic {
                object,
                descriptor: Bytes::from(descriptor),
            }
        }
        kivi_protocol::ScanValueBody::None => ClientScanValue::Absent,
    }
}

/// One scan page with its routing metadata.
#[derive(Debug, Clone)]
pub struct ClientScanPage {
    /// Entries in scan order.
    pub entries: Vec<ClientScanEntry>,
    /// Whether the serving tablet's slice is exhausted.
    pub exhausted: bool,
    /// Last emitted key, if any.
    pub last_key: Option<Vec<u8>>,
    /// Serving tablet (zero = gap page, no tablet).
    pub tablet: u64,
    /// Serving range start (gap bounds for gap pages).
    pub range_start: Vec<u8>,
    /// Serving range end (`None` = `+∞`).
    pub range_end: Option<Vec<u8>>,
    /// Serving directory version (staleness hint for transaction plans).
    pub dir_version: u64,
}

/// Resumable scan cursor: everything semantically required, nothing
/// server-local (no tablet ids that can go stale, no sockets, no iterator
/// handles). Survives leader changes, reconnects, splits, merges, and
/// migrations: resume re-routes by logical key.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScanCursor {
    /// Scanned namespace.
    pub namespace: NamespaceId,
    /// Key order.
    pub direction: ScanDirection,
    /// Original lower bound.
    pub start: Option<Vec<u8>>,
    /// Original upper bound.
    pub end: Option<Vec<u8>>,
    /// Resume point: next forward start (inclusive) or next reverse end
    /// (exclusive).
    pub resume: Option<Vec<u8>>,
    /// Whether the scan already completed.
    pub done: bool,
}

impl ScanCursor {
    /// Starts a cursor from scan bounds.
    #[must_use]
    pub fn new(
        namespace: NamespaceId,
        direction: ScanDirection,
        start: Option<Vec<u8>>,
        end: Option<Vec<u8>>,
    ) -> Self {
        Self {
            namespace,
            direction,
            start,
            end,
            resume: None,
            done: false,
        }
    }
}

/// Lexicographic successor: the smallest key strictly greater than `key`.
#[must_use]
pub fn key_successor(key: &[u8]) -> Vec<u8> {
    let mut next = key.to_vec();
    next.push(0x00);
    next
}

impl NativeClient {
    /// Fetches one scan page: routes the window cursor, follows redirects,
    /// and returns the serving tablet's slice with its range. Most callers
    /// want [`scan`](Self::scan) or cursor paging instead.
    ///
    /// # Errors
    ///
    /// Returns [`ClientError`] on transport/routing failure or server
    /// rejection.
    #[allow(clippy::too_many_arguments, clippy::too_many_lines)]
    pub fn scan_page(
        &self,
        namespace: NamespaceId,
        window_start: Option<Vec<u8>>,
        window_end: Option<Vec<u8>>,
        direction: ScanDirection,
        consistency: ScanConsistency,
        projection: ScanProjection,
        max_items: u32,
        max_bytes: u32,
    ) -> Result<ClientScanPage, ClientError> {
        use kivi_protocol::{Opcode, ResponseBody};
        let route_bytes: Vec<u8> = match direction {
            ScanDirection::Forward => window_start.clone().unwrap_or_default(),
            ScanDirection::Reverse => window_end
                .clone()
                .unwrap_or_else(|| window_start.clone().unwrap_or_default()),
        };
        let route_key = Key::from(route_bytes);
        let (wire_direction, wire_projection, wire_consistency) = (
            match direction {
                ScanDirection::Forward => kivi_protocol::SCAN_FORWARD,
                ScanDirection::Reverse => kivi_protocol::SCAN_REVERSE,
            },
            match projection {
                ScanProjection::KeysOnly => kivi_protocol::SCAN_KEYS_ONLY,
                ScanProjection::KeysAndValues => kivi_protocol::SCAN_KEYS_AND_VALUES,
            },
            match consistency {
                ScanConsistency::LatestPerTablet => kivi_protocol::SCAN_LATEST_PER_TABLET,
                ScanConsistency::Any => kivi_protocol::SCAN_ANY,
            },
        );
        // A scan request carries no single key; route by the window cursor
        // (the server re-routes every page by key regardless of the hint).
        let start = window_start;
        let end = window_end;
        let response = self.execute_raw(
            &route_key,
            Opcode::Scan,
            || kivi_protocol::Request {
                contract: kivi_types::ReadContract::Latest,
                namespace,
                opcode: Opcode::Scan,
                hint: None,
                key: Vec::new(),
                value: None,
                delta: 0,
                expiry: 0,
                offset: 0,
                len: 0,
                condition: kivi_protocol::COND_ALWAYS,
                expiry_policy: kivi_protocol::EXPIRY_CLEAR,
                identity: None,
                ack_floor: kivi_types::RequestSeq::from_u64(0),
                scan_start: start.clone(),
                scan_end: end.clone(),
                scan_direction: wire_direction,
                scan_max_items: max_items,
                scan_max_bytes: max_bytes,
                scan_projection: wire_projection,
                scan_consistency: wire_consistency,
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
            },
            None,
            kivi_types::ReadContract::Latest,
        )?;
        match (response.status, response.body) {
            (
                kivi_protocol::Status::Ok,
                ResponseBody::ScanPage {
                    entries,
                    exhausted,
                    last_key,
                    tablet,
                    range_start,
                    range_end,
                    dir_version,
                },
            ) => Ok(ClientScanPage {
                entries: entries
                    .into_iter()
                    .map(|entry| ClientScanEntry {
                        key: entry.key,
                        value: scan_body_to_client(entry.value),
                    })
                    .collect(),
                exhausted,
                last_key,
                tablet,
                range_start,
                range_end,
                dir_version,
            }),
            (status, body) => Err(status_error(status, &body)),
        }
    }

    /// Streams a whole range in order, paging by key across tablets.
    /// Results are strictly ordered within the requested range with no
    /// topology-induced duplicates or omissions; concurrent writes appear
    /// per the scan consistency (see [`ScanConsistency`]).
    ///
    /// # Errors
    ///
    /// Returns [`ClientError`] on transport/routing failure, server
    /// rejection, or an unresolvable gap.
    pub fn scan(
        &self,
        namespace: NamespaceId,
        options: &ScanOptions,
    ) -> Result<Vec<ClientScanEntry>, ClientError> {
        let mut cursor = ScanCursor::new(
            namespace,
            options.direction,
            options.start.clone(),
            options.end.clone(),
        );
        let mut out = Vec::new();
        let mut stale_restarts: u32 = 0;
        loop {
            if cursor.done {
                break;
            }
            if let Some(limit) = options.limit
                && out.len() >= limit
            {
                break;
            }
            let mut page = match self.scan_cursor_page(&cursor, options) {
                Err(ClientError::ScanCursorStale) if stale_restarts < 3 => {
                    // Cursor predates retention (namespace dropped or long
                    // topology churn): restart from the original bounds.
                    stale_restarts += 1;
                    cursor = ScanCursor::new(
                        namespace,
                        options.direction,
                        options.start.clone(),
                        options.end.clone(),
                    );
                    out.clear();
                    continue;
                }
                result => result?,
            };
            let empty = page.entries.is_empty();
            for entry in page.entries.drain(..) {
                if let Some(limit) = options.limit
                    && out.len() >= limit
                {
                    break;
                }
                out.push(entry);
            }
            Self::advance_cursor(&mut cursor, &page, empty)?;
        }
        Ok(out)
    }

    /// Fetches the next page for a cursor (resumable scan primitive).
    ///
    /// # Errors
    ///
    /// Returns [`ClientError`] on transport/routing failure or server
    /// rejection.
    pub fn scan_cursor_page(
        &self,
        cursor: &ScanCursor,
        options: &ScanOptions,
    ) -> Result<ClientScanPage, ClientError> {
        let (window_start, window_end) = match cursor.direction {
            ScanDirection::Forward => {
                let start = cursor.resume.clone().or(cursor.start.clone());
                (start, cursor.end.clone())
            }
            ScanDirection::Reverse => {
                let end = cursor.resume.clone().or(cursor.end.clone());
                (cursor.start.clone(), end)
            }
        };
        self.scan_page(
            cursor.namespace,
            window_start,
            window_end,
            cursor.direction,
            options.consistency,
            options.projection,
            options.max_items_per_page,
            options.max_bytes_per_page,
        )
    }

    /// Advances a cursor past one page. Gap pages (empty, unexhausted, no
    /// last key) jump by the reported gap bounds; a gap with no upper
    /// bound backs off and retries the same cursor (bounded).
    fn advance_cursor(
        cursor: &mut ScanCursor,
        page: &ClientScanPage,
        entries_empty: bool,
    ) -> Result<(), ClientError> {
        if entries_empty && !page.exhausted && page.last_key.is_none() {
            // Gap (transient mid-split): jump it by key, or wait it out.
            match cursor.direction {
                ScanDirection::Forward => {
                    if let Some(next) = &page.range_end {
                        cursor.resume = Some(next.clone());
                        return Ok(());
                    }
                    backoff(Duration::from_millis(20), 2);
                    return Ok(());
                }
                ScanDirection::Reverse => {
                    cursor.resume = Some(page.range_start.clone());
                    return Ok(());
                }
            }
        }
        if !page.exhausted {
            let Some(last) = page.last_key.clone() else {
                return Err(ClientError::Internal(
                    "page truncated its cursor".to_owned(),
                ));
            };
            match cursor.direction {
                ScanDirection::Forward => cursor.resume = Some(key_successor(&last)),
                ScanDirection::Reverse => cursor.resume = Some(last),
            }
            return Ok(());
        }
        // Tablet slice exhausted: continue on the next tablet or finish.
        match cursor.direction {
            ScanDirection::Forward => {
                let more = match (&page.range_end, &cursor.end) {
                    (None, _) => false,
                    (Some(_), None) => true,
                    (Some(tablet_end), Some(end)) => tablet_end.as_slice() < end.as_slice(),
                };
                if more {
                    cursor.resume.clone_from(&page.range_end);
                } else {
                    cursor.done = true;
                }
            }
            ScanDirection::Reverse => {
                let lower = cursor.start.as_deref().unwrap_or(&[]);
                if page.range_start.as_slice() > lower {
                    cursor.resume = Some(page.range_start.clone());
                } else {
                    cursor.done = true;
                }
            }
        }
        Ok(())
    }

    /// Reads one key's logical version (`None` when absent) without
    /// fetching its value. Works for chunked values without resolving any
    /// bytes.
    ///
    /// # Errors
    ///
    /// Returns [`ClientError`] on transport/routing failure.
    pub fn get_version(
        &self,
        namespace: NamespaceId,
        key: &[u8],
    ) -> Result<Option<u64>, ClientError> {
        use kivi_protocol::{Opcode, ResponseBody};
        let route_key = Key::from(key.to_vec());
        let response = self.execute_raw(
            &route_key,
            Opcode::GetVersion,
            || kivi_protocol::Request {
                contract: kivi_types::ReadContract::Latest,
                namespace,
                opcode: Opcode::GetVersion,
                hint: None,
                key: key.to_vec(),
                value: None,
                delta: 0,
                expiry: 0,
                offset: 0,
                len: 0,
                condition: kivi_protocol::COND_ALWAYS,
                expiry_policy: kivi_protocol::EXPIRY_CLEAR,
                identity: None,
                ack_floor: kivi_types::RequestSeq::from_u64(0),
                scan_start: None,
                scan_end: None,
                scan_direction: kivi_protocol::SCAN_FORWARD,
                scan_max_items: 0,
                scan_max_bytes: 0,
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
            },
            None,
            kivi_types::ReadContract::Latest,
        )?;
        match (response.status, response.body) {
            (kivi_protocol::Status::Ok, ResponseBody::Version(version)) => Ok(Some(version)),
            (kivi_protocol::Status::NotFound, _) => Ok(None),
            (status, body) => Err(status_error(status, &body)),
        }
    }
}

// ---------------------------------------------------------------------------
// Atomic batch + client-driven 2PC
// ---------------------------------------------------------------------------

/// One write inside an atomic batch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BatchWriteSpec {
    /// Target key.
    pub key: Vec<u8>,
    /// Write kind.
    pub kind: BatchWriteKind,
    /// OCC expectation.
    pub expect: BatchExpect,
}

/// Atomic batch write kinds (deterministic point ops only).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BatchWriteKind {
    /// Store bytes (SET semantics).
    Put(Vec<u8>),
    /// Store a staged large value by chunk reference (stage the bytes
    /// through a streaming upload first; the reference is proven at
    /// admission before any participant may prepare).
    PutChunked {
        /// Manifest addressing the immutable chunk sequence.
        manifest: [u8; 32],
        /// Total logical bytes across the manifest.
        logical_len: u64,
    },
    /// Remove the key.
    Delete,
    /// Add to a strict counter (ordinal result).
    CounterAdd(i64),
    /// Add to a commutative counter (order-free; result is `Applied`).
    CommutativeAdd(i64),
    /// Add to a bounded counter (escrow rights checked at prepare).
    BoundedAdd(i64),
    /// Move one bounded counter's escrow share to `[min, max]` (one side
    /// of a paired, conservation-checked transfer — see
    /// [`kivi_state::plan_escrow_transfer`]).
    EscrowSetShare {
        /// New share lower bound (inclusive).
        min: i64,
        /// New share upper bound (inclusive).
        max: i64,
    },
}

/// OCC expectation for one batch write.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BatchExpect {
    /// Blind write (still conflicts with prepared intents).
    Any,
    /// Key must be absent.
    Absent,
    /// Live version must equal this exactly.
    Version(u64),
}

/// Atomic batch outcome.
#[derive(Debug, Clone)]
pub struct AtomicBatchResult {
    /// Committing transaction id (stable across transport retries of the
    /// attempt, fresh per OCC attempt).
    pub txn: TxnId,
    /// Resulting versions in request order (`None` for deletes).
    /// Re-driven completions may carry `None` for keys finalized before
    /// the re-drive (commit confirmed, version unrecoverable).
    pub versions: Vec<Option<u64>>,
}

/// Maximum OCC attempts per batch before surfacing the conflict.
const MAX_BATCH_ATTEMPTS: u32 = 3;
/// Maximum same-transaction re-drives (ambiguous steps) per attempt.
const MAX_BATCH_REDRIVES: u32 = 3;
/// Maximum abort-finalize waves per prepared key: the Abort decision is
/// already durable, so each wave redrives the same idempotent discard
/// until the intent is observably resolved or the failure is terminal.
/// Eight waves back off ~2.5s total — past any election/redirection
/// window, far short of the 60s transaction lease the leak would
/// otherwise wait out.
const MAX_ABORT_WAVES: u32 = 8;

impl NativeClient {
    /// Executes an atomic batch: all writes commit or none does.
    /// Single-tablet batches run server-side in one roundtrip; wider
    /// batches run client-driven 2PC with OCC retries (fresh `TxnId` per
    /// attempt, stable across transport retries within an attempt).
    ///
    /// # Errors
    ///
    /// Returns [`ClientError`] on conflicts (after retries), oversize
    /// batches, uniqueness violations surfaced by index callers, or
    /// transport/routing failure.
    pub fn atomic_batch(
        &self,
        namespace: NamespaceId,
        writes: &[BatchWriteSpec],
    ) -> Result<AtomicBatchResult, ClientError> {
        if writes.is_empty() {
            return Err(ClientError::InvalidRequest);
        }
        if writes.len() > kivi_state::MAX_TXN_KEYS {
            return Err(ClientError::TxnTooLarge);
        }
        let bytes: usize = writes
            .iter()
            .map(|write| {
                write.key.len()
                    + match &write.kind {
                        BatchWriteKind::Put(value) => value.len(),
                        BatchWriteKind::PutChunked { .. } => 40,
                        BatchWriteKind::Delete
                        | BatchWriteKind::CounterAdd(_)
                        | BatchWriteKind::CommutativeAdd(_)
                        | BatchWriteKind::BoundedAdd(_)
                        | BatchWriteKind::EscrowSetShare { .. } => 8,
                    }
            })
            .sum();
        if bytes > kivi_state::MAX_TXN_BYTES {
            return Err(ClientError::TxnTooLarge);
        }
        self.note_txn_started();
        let mut attempt: u32 = 0;
        loop {
            let salt = self.next_id().as_u64();
            let txn = TxnId::derive(self.shared.session.as_u128(), salt, u64::from(attempt));
            match self.batch_attempt(namespace, writes, txn) {
                Ok(result) => return Ok(result),
                Err(BatchAttempt::Conflict) if attempt + 1 < MAX_BATCH_ATTEMPTS => {
                    attempt += 1;
                    backoff(Duration::from_millis(10), attempt.min(6));
                }
                Err(BatchAttempt::Conflict) => {
                    self.note_txn_aborted(true);
                    return Err(ClientError::TxnConflict);
                }
                Err(BatchAttempt::Redrive) => {
                    // Same-transaction re-drive: bounded internally.
                    return match self.redrive(namespace, writes, txn) {
                        Ok(result) => Ok(result),
                        Err(error) => {
                            self.note_txn_aborted(matches!(error, ClientError::TxnConflict));
                            Err(error)
                        }
                    };
                }
                Err(BatchAttempt::Error(error)) => {
                    self.note_txn_aborted(matches!(error, ClientError::TxnConflict));
                    return Err(error);
                }
            }
        }
    }

    /// Re-drives one transaction id to completion after ambiguity
    /// (bounded): reads the record first — a durable Commit skips straight
    /// to finalizes, so re-drives never duplicate prepared intent effects.
    fn redrive(
        &self,
        namespace: NamespaceId,
        writes: &[BatchWriteSpec],
        txn: TxnId,
    ) -> Result<AtomicBatchResult, ClientError> {
        let mut redrives: u32 = 0;
        loop {
            match self.batch_attempt(namespace, writes, txn) {
                Ok(result) => return Ok(result),
                Err(BatchAttempt::Conflict) => return Err(ClientError::TxnConflict),
                Err(BatchAttempt::Redrive) if redrives + 1 < MAX_BATCH_REDRIVES => {
                    redrives += 1;
                    backoff(Duration::from_millis(20), redrives.min(6));
                }
                Err(BatchAttempt::Redrive) => {
                    return Err(ClientError::TxnCoordinatorUnavailable);
                }
                Err(BatchAttempt::Error(error)) => return Err(error),
            }
        }
    }

    /// Encodes one batch write for the wire: kind, value, delta, and OCC
    /// expectation. Shared by the fast-path batch and the 2PC prepare
    /// step so both waves encode identically.
    fn encode_wire_write(write: &BatchWriteSpec) -> kivi_protocol::BatchWrite {
        let (kind, value, delta) = match &write.kind {
            BatchWriteKind::Put(value) => (kivi_protocol::BATCH_PUT, value.clone(), 0),
            BatchWriteKind::PutChunked {
                manifest,
                logical_len,
            } => {
                let mut blob = Vec::with_capacity(40);
                blob.extend_from_slice(manifest);
                blob.extend_from_slice(&logical_len.to_le_bytes());
                (kivi_protocol::BATCH_PUT_CHUNKED, blob, 0)
            }
            BatchWriteKind::Delete => (kivi_protocol::BATCH_DELETE, Vec::new(), 0),
            BatchWriteKind::CounterAdd(delta) => {
                (kivi_protocol::BATCH_COUNTER_ADD, Vec::new(), *delta)
            }
            BatchWriteKind::CommutativeAdd(delta) => {
                (kivi_protocol::BATCH_COMMUTATIVE_ADD, Vec::new(), *delta)
            }
            BatchWriteKind::BoundedAdd(delta) => {
                (kivi_protocol::BATCH_BOUNDED_ADD, Vec::new(), *delta)
            }
            BatchWriteKind::EscrowSetShare { min, max } => {
                let mut blob = Vec::with_capacity(16);
                blob.extend_from_slice(&min.to_le_bytes());
                blob.extend_from_slice(&max.to_le_bytes());
                (kivi_protocol::BATCH_ESCROW_SHARE, blob, 0)
            }
        };
        let (expect, expect_version) = match write.expect {
            BatchExpect::Any => (kivi_protocol::BATCH_EXPECT_ANY, 0),
            BatchExpect::Absent => (kivi_protocol::BATCH_EXPECT_ABSENT, 0),
            BatchExpect::Version(version) => (kivi_protocol::BATCH_EXPECT_VERSION, version),
        };
        kivi_protocol::BatchWrite {
            key: write.key.clone(),
            kind,
            value,
            delta,
            expect,
            expect_version,
        }
    }

    /// One batch attempt under a fixed `TxnId`: fast path first, then
    /// client-driven 2PC on multi-tablet rejection.
    fn batch_attempt(
        &self,
        namespace: NamespaceId,
        writes: &[BatchWriteSpec],
        txn: TxnId,
    ) -> Result<AtomicBatchResult, BatchAttempt> {
        use kivi_protocol::{Opcode, ResponseBody};
        let wire: Vec<kivi_protocol::BatchWrite> =
            writes.iter().map(Self::encode_wire_write).collect();
        let route_key = Key::from(writes[0].key.clone());
        let mut request = self.request_base(&route_key, Opcode::AtomicBatch);
        request.namespace = namespace;
        request.batch_txn = txn.as_bytes();
        request.batch_writes = wire;
        let response = self
            .execute_raw(
                &route_key,
                Opcode::AtomicBatch,
                || request.clone(),
                None,
                kivi_types::ReadContract::Latest,
            )
            .map_err(BatchAttempt::Error)?;
        match (response.status, response.body) {
            (kivi_protocol::Status::Ok, ResponseBody::AtomicCommitted { versions }) => {
                self.note_txn_committed(false);
                Ok(AtomicBatchResult { txn, versions })
            }
            // Typed cross-tablet signal (never string-matched):
            // single-tablet fast path cannot serve this set.
            (kivi_protocol::Status::TxnCrossTablet, _) => {
                match self.drive_2pc(namespace, writes, txn) {
                    Ok(result) => {
                        self.note_txn_committed(true);
                        Ok(result)
                    }
                    Err(attempt) => Err(attempt),
                }
            }
            (kivi_protocol::Status::TxnConflict, _) => Err(BatchAttempt::Conflict),
            (
                kivi_protocol::Status::TxnCoordinatorUnavailable
                | kivi_protocol::Status::Overloaded,
                _,
            ) => Err(BatchAttempt::Redrive),
            (status, body) => Err(BatchAttempt::Error(status_error(status, &body))),
        }
    }

    /// Client-driven OCC + 2PC across tablets: probe participants, prepare
    /// every key, persist the decision, finalize all. Recovery-first: an
    /// existing durable decision skips straight to finalizes.
    ///
    /// Coordinator selection is deterministic: the lowest probed
    /// participant tablet. Every drive of one write set under stable
    /// routing derives the same coordinator with no control-plane
    /// involvement, so recovery re-drives agree after failures.
    fn drive_2pc(
        &self,
        namespace: NamespaceId,
        writes: &[BatchWriteSpec],
        txn: TxnId,
    ) -> Result<AtomicBatchResult, BatchAttempt> {
        let typed = Self::typed_writes(writes)?;
        let digest = kivi_state::write_set_digest(&typed, namespace);
        let drive = self.probe_participants(namespace, writes)?;
        let attempt = TxnAttempt {
            txn,
            coordinator: drive.coordinator,
            digest,
        };
        // Recovery-first: a durable decision from an earlier drive of this
        // transaction resolves without re-preparing. Only a decision
        // carrying this exact digest governs (a foreign digest under one
        // `TxnId` is a conflicting driver — fail loudly, never follow).
        let record_key = kivi_state::txn_record_key(attempt.coordinator, txn);
        if let Some(record) = self.read_txn_record(namespace, &record_key)? {
            match record.state {
                TxnState::Committed if record.digest == digest => {
                    return self.finalize_all_recovery(namespace, writes, txn, true, digest);
                }
                TxnState::Aborted if record.digest == digest => {
                    return Err(BatchAttempt::Error(ClientError::TxnAborted));
                }
                TxnState::Begun => {}
                _ => return Err(BatchAttempt::Conflict),
            }
        }
        let ready = match self.prepare_wave(namespace, writes, attempt, &drive.probed) {
            Ok(ready) => ready,
            Err((aborted, prepared)) => {
                tracing::debug!(
                    probed = ?drive
                        .probed
                        .values()
                        .map(|(tablet, _)| tablet.as_u64())
                        .collect::<Vec<_>>(),
                    ?prepared,
                    "2pc prepare failed; aborting partial reservations",
                );
                return self.abort_or_commit(namespace, writes, attempt, &prepared, aborted);
            }
        };
        self.decide_and_finalize(namespace, writes, attempt, drive, ready)
    }

    /// Discovers one drive's participants (route-cache first, one-entry
    /// scan probes on misses) and derives the lowest-tablet coordinator
    /// plus the directory version the plan routes against.
    fn probe_participants(
        &self,
        namespace: NamespaceId,
        writes: &[BatchWriteSpec],
    ) -> Result<ProbedDrive, BatchAttempt> {
        // Probes warm the route cache for the waves below.
        let mut probed: std::collections::BTreeMap<Vec<u8>, (TabletId, u64)> =
            std::collections::BTreeMap::new();
        let mut dir_version: u64 = 0;
        for write in writes {
            let (tablet, version) = self.probe_tablet(namespace, &write.key)?;
            dir_version = dir_version.max(version);
            probed.insert(write.key.clone(), (tablet, version));
        }
        let coordinator = probed
            .values()
            .map(|(tablet, _)| *tablet)
            .min()
            .ok_or(BatchAttempt::Error(ClientError::InvalidRequest))?;
        Ok(ProbedDrive {
            probed,
            coordinator,
            dir_version,
        })
    }

    /// Prepares every key in request order (deterministic), verifying
    /// each prepare served on its probed tablet. Stops at the first
    /// failure with the partial reservations made: the driver aborts
    /// those explicitly (never a silent remap into another lineage, never
    /// an ambiguous half-drive).
    fn prepare_wave(
        &self,
        namespace: NamespaceId,
        writes: &[BatchWriteSpec],
        attempt: TxnAttempt,
        probed: &std::collections::BTreeMap<Vec<u8>, (TabletId, u64)>,
    ) -> Result<PreparedDrive, (BatchAttempt, Vec<usize>)> {
        let prepare_start = std::time::Instant::now();
        let mut prepared: Vec<usize> = Vec::with_capacity(writes.len());
        let mut observed: std::collections::BTreeSet<TabletId> = std::collections::BTreeSet::new();
        for (index, write) in writes.iter().enumerate() {
            let (expected_tablet, _) = probed[write.key.as_slice()];
            match self.txn_prepare(
                namespace,
                write,
                attempt.txn,
                attempt.coordinator,
                attempt.digest,
            ) {
                Ok(tablet) => {
                    if tablet != expected_tablet {
                        // The server APPLIED this prepare on `tablet`
                        // (intent created, `Ok` returned) but the probe
                        // named a retired tablet: a stale cached route.
                        // The intent MUST join the abort set (the abort
                        // wave finalizes it) and the proven-stale cache
                        // entries covering this key go, so the next
                        // attempt re-probes fresh. Dropping the index
                        // here orphans the intent with no record and no
                        // owner — a wedge until lease expiry.
                        prepared.push(index);
                        observed.insert(tablet);
                        self.evict_covering_for(namespace, &write.key);
                        tracing::debug!(
                            key = hex_key_trace(&write.key),
                            probed = expected_tablet.as_u64(),
                            served = tablet.as_u64(),
                            "2pc prepare served-elsewhere; intent kept for abort, route evicted",
                        );
                        return Err((BatchAttempt::Conflict, prepared));
                    }
                    observed.insert(tablet);
                    prepared.push(index);
                }
                Err(ClientError::TxnConflict) => {
                    return Err((BatchAttempt::Conflict, prepared));
                }
                Err(error) if is_ambiguous(&error) => {
                    return Err((BatchAttempt::Redrive, prepared));
                }
                // Deterministic semantic rejections (wrong type, overflow,
                // rights, fencing, bounds) abort the attempt and surface
                // intact — retrying the same writes would fail identically.
                Err(error) => {
                    return Err((BatchAttempt::Error(error), prepared));
                }
            }
        }
        self.note_prepare_latency(
            u64::try_from(prepare_start.elapsed().as_micros()).unwrap_or(u64::MAX),
        );
        Ok(PreparedDrive { prepared, observed })
    }

    /// Verifies the prepared set against the probe, persists the Commit
    /// decision (guarded CAS), re-verifies the durable record, and
    /// finalizes. A lost decide race re-reads before aborting (an
    /// ambiguous decide may actually have committed — never
    /// blind-overwrite a decision with an abort).
    fn decide_and_finalize(
        &self,
        namespace: NamespaceId,
        writes: &[BatchWriteSpec],
        attempt: TxnAttempt,
        drive: ProbedDrive,
        prepared: PreparedDrive,
    ) -> Result<AtomicBatchResult, BatchAttempt> {
        let TxnAttempt {
            txn,
            coordinator,
            digest,
        } = attempt;
        // The probed set must equal the observed set: any drift means
        // routing moved under the attempt. Evict the drifted keys'
        // stale routes so the fresh plan re-probes instead of
        // conflicting against retired tablets again.
        if prepared
            .observed
            .iter()
            .copied()
            .collect::<std::collections::BTreeSet<_>>()
            != drive.probed.values().map(|(tablet, _)| *tablet).collect()
        {
            tracing::debug!(
                probed = ?drive
                    .probed
                    .values()
                    .map(|(tablet, _)| tablet.as_u64())
                    .collect::<Vec<_>>(),
                observed = ?prepared
                    .observed
                    .iter()
                    .map(|tablet| tablet.as_u64())
                    .collect::<Vec<_>>(),
                "2pc probe drift; evicting plan routes",
            );
            for write in writes {
                self.evict_covering_for(namespace, &write.key);
            }
            return self.abort_or_commit(
                namespace,
                writes,
                attempt,
                &prepared.prepared,
                BatchAttempt::Conflict,
            );
        }
        // All prepared: persist the Commit decision with the observed
        // participants and the canonical digest — guarded (CAS): a
        // concurrent resolver abort conflicts instead of being silently
        // overwritten, and the driver follows the abort.
        let record = Self::build_record(
            namespace,
            writes,
            txn,
            coordinator,
            prepared.observed.into_iter().collect(),
            drive.dir_version,
            TxnState::Committed,
        );
        let record_key = kivi_state::txn_record_key(coordinator, txn);
        let decide_start = std::time::Instant::now();
        let decided = self.decide_record(namespace, coordinator, &record_key, &record, 0);
        self.note_decision_latency(
            u64::try_from(decide_start.elapsed().as_micros()).unwrap_or(u64::MAX),
        );
        if !decided {
            // Decide failed: re-read before aborting (an ambiguous decide
            // may actually have committed — never blind-overwrite a
            // decision with an abort).
            match self.read_txn_record(namespace, &record_key) {
                Ok(Some(record)) if record.state == TxnState::Committed => {
                    if record.digest != digest {
                        return Err(BatchAttempt::Conflict);
                    }
                    return self.finalize_all_recovery(namespace, writes, txn, true, digest);
                }
                Ok(_) => {}
                Err(BatchAttempt::Redrive) => return Err(BatchAttempt::Redrive),
                Err(BatchAttempt::Conflict) => return Err(BatchAttempt::Conflict),
                Err(BatchAttempt::Error(error)) => return Err(BatchAttempt::Error(error)),
            }
            return self.abort_or_commit(
                namespace,
                writes,
                attempt,
                &prepared.prepared,
                BatchAttempt::Redrive,
            );
        }
        // Verify the durable decision before finalizing: state, digest,
        // and lineage must be exactly what this attempt decided.
        match self.read_txn_record(namespace, &record_key) {
            Ok(Some(record)) if record.state == TxnState::Committed && record.digest == digest => {}
            Ok(_) => {
                return self.abort_or_commit(
                    namespace,
                    writes,
                    attempt,
                    &prepared.prepared,
                    BatchAttempt::Conflict,
                );
            }
            Err(attempt) => return Err(attempt),
        }
        let expected: std::collections::BTreeMap<Vec<u8>, TabletId> = drive
            .probed
            .into_iter()
            .map(|(key, (tablet, _))| (key, tablet))
            .collect();
        self.finalize_all(namespace, writes, txn, true, digest, &expected)
    }

    /// Persists one terminal decision through an absence-guarded
    /// single-key transaction on the record tablet (CAS — never a blind
    /// overwrite). `decide_salt` separates the commit decide-transaction
    /// from the abort one so the two never share intent identity.
    fn decide_record(
        &self,
        namespace: NamespaceId,
        coordinator: TabletId,
        record_key: &Key,
        record: &TxnRecord,
        decide_salt: u64,
    ) -> bool {
        let write = BatchWriteSpec {
            key: record_key.as_bytes().to_vec(),
            kind: BatchWriteKind::Put(record.encode()),
            expect: BatchExpect::Absent,
        };
        // The decide runs as its own single-key transaction on the record
        // tablet (any coordinator works for one key; use a derived id so
        // re-drives are idempotent). Record-key steps bind the zero digest.
        let decide_txn = TxnId::derive(u128::from_le_bytes(record.id.as_bytes()), 0, decide_salt);
        match self.txn_prepare(namespace, &write, decide_txn, coordinator, [0u8; 32]) {
            Ok(_) => {}
            Err(_) => return false,
        }
        // Only an applied finalize persists the decision: `Gone` means
        // the intent vanished without applying (never happens for a fresh
        // decide intent outside a racing abort — treat as failure and let
        // the caller re-read the record).
        matches!(
            self.txn_finalize(
                namespace,
                record_key.as_bytes(),
                decide_txn,
                true,
                [0u8; 32]
            ),
            Ok(FinalizeOutcome::Applied(_, _))
        )
    }

    /// Builds the canonical coordinator record for a write set.
    fn build_record(
        namespace: NamespaceId,
        writes: &[BatchWriteSpec],
        txn: TxnId,
        coordinator: TabletId,
        participants: Vec<TabletId>,
        dir_version: u64,
        state: TxnState,
    ) -> TxnRecord {
        let typed = Self::typed_writes(writes).unwrap_or_default();
        // Participants are observed prepare tablets (advisory for
        // operators); the record's decision + digest are authoritative.
        let digest = kivi_state::write_set_digest(&typed, namespace);
        TxnRecord {
            id: txn,
            coordinator,
            participants,
            state,
            dir_version,
            digest,
        }
    }

    /// Translates batch specs into typed writes (shared by digest
    /// computation and record building, so drivers and records agree).
    fn typed_writes(writes: &[BatchWriteSpec]) -> Result<Vec<kivi_state::TxnWrite>, BatchAttempt> {
        writes
            .iter()
            .map(|write| {
                let kind = match &write.kind {
                    BatchWriteKind::Put(value) => {
                        kivi_state::TxnWriteKind::Put(Bytes::from(value.clone()))
                    }
                    BatchWriteKind::PutChunked {
                        manifest,
                        logical_len,
                    } => kivi_state::TxnWriteKind::PutChunked {
                        manifest: kivi_types::ManifestId::from_bytes(*manifest),
                        logical_len: *logical_len,
                    },
                    BatchWriteKind::Delete => kivi_state::TxnWriteKind::Delete,
                    BatchWriteKind::CounterAdd(delta) => {
                        kivi_state::TxnWriteKind::CounterAdd(*delta)
                    }
                    BatchWriteKind::CommutativeAdd(delta) => {
                        kivi_state::TxnWriteKind::CommutativeAdd(*delta)
                    }
                    BatchWriteKind::BoundedAdd(delta) => {
                        kivi_state::TxnWriteKind::BoundedAdd(*delta)
                    }
                    BatchWriteKind::EscrowSetShare { min, max } => {
                        kivi_state::TxnWriteKind::EscrowSetShare {
                            min: *min,
                            max: *max,
                        }
                    }
                };
                let expect = match write.expect {
                    BatchExpect::Any => kivi_state::TxnExpect::Any,
                    BatchExpect::Absent => kivi_state::TxnExpect::Absent,
                    BatchExpect::Version(version) => {
                        kivi_state::TxnExpect::Version(kivi_state::ObjectVersion::from_u64(version))
                    }
                };
                Ok(kivi_state::TxnWrite {
                    key: Key::from(write.key.clone()),
                    kind,
                    expect,
                })
            })
            .collect()
    }

    /// Discovers one key's tablet: co-located system/index keys resolve
    /// through their primary first (route-cache first, one-entry scan
    /// probes on misses), else the key itself. Returns the tablet plus
    /// the freshest directory version observed.
    fn probe_tablet(
        &self,
        namespace: NamespaceId,
        key: &[u8],
    ) -> Result<(TabletId, u64), BatchAttempt> {
        // Projection keys and non-unique index entries live beside their
        // primary: probe the primary's tablet so prepares, probes, and
        // server routing agree on placement.
        if let Some(primary) = parse_projection_primary(key) {
            let owned = primary.to_vec();
            return self.probe_routable(namespace, &owned);
        }
        if let Some(primary) = parse_index_primary(key) {
            return self.probe_routable(namespace, &primary);
        }
        self.probe_routable(namespace, key)
    }

    /// Evicts cached routes covering `key`: called when a prepare or
    /// finalize proves the cache stale by serving on an unexpected
    /// tablet. The next probe re-scans fresh instead of conflicting
    /// against a retired tablet forever.
    fn evict_covering_for(&self, namespace: NamespaceId, key: &[u8]) {
        if let Some(hash) = kivi_state::PartitionHasher::V1.hash(namespace, key) {
            self.shared.routes.evict_covering(key, hash);
        }
    }

    /// Probes `route` for placement (route-cache first, one-entry scan
    /// probe on miss, which also warms the cache for the waves below).
    fn probe_routable(
        &self,
        namespace: NamespaceId,
        route: &[u8],
    ) -> Result<(TabletId, u64), BatchAttempt> {
        if let Some(hash) = kivi_state::PartitionHasher::V1.hash(namespace, route)
            && let Some(entry) = self.shared.routes.lookup(hash)
        {
            return Ok((entry.tablet, entry.dir_version));
        }
        if let Some(entry) = self.shared.routes.lookup_key(route) {
            return Ok((entry.tablet, entry.dir_version));
        }
        let mut tries: u32 = 0;
        loop {
            let page = self
                .scan_page(
                    namespace,
                    Some(route.to_vec()),
                    Some(key_successor(route)),
                    ScanDirection::Forward,
                    ScanConsistency::LatestPerTablet,
                    ScanProjection::KeysOnly,
                    1,
                    64,
                )
                .map_err(BatchAttempt::Error)?;
            if page.tablet != 0 {
                return Ok((TabletId::from_u64(page.tablet), page.dir_version));
            }
            tries += 1;
            if tries >= MAX_BATCH_REDRIVES {
                return Err(BatchAttempt::Redrive);
            }
            backoff(Duration::from_millis(20), tries.min(6));
        }
    }

    /// Reads and decodes a coordinator record (`None` when absent).
    fn read_txn_record(
        &self,
        namespace: NamespaceId,
        record_key: &Key,
    ) -> Result<Option<TxnRecord>, BatchAttempt> {
        match self.get_raw(namespace, record_key.as_bytes()) {
            Ok(Some(bytes)) => TxnRecord::decode(&bytes)
                .map(Some)
                .map_err(|_| BatchAttempt::Error(ClientError::TxnAborted)),
            Ok(None) => Ok(None),
            Err(error) if is_ambiguous(&error) => Err(BatchAttempt::Redrive),
            Err(error) => Err(BatchAttempt::Error(error)),
        }
    }

    /// Finalizes every key (commit or abort), collecting versions in
    /// request order. Missing intents (already resolved) report
    /// not-applied without failing: convergence, not error. Every
    /// finalize carries the plan digest and must be served by its expected
    /// participant: a foreign serving tablet means the tablet moved
    /// mid-transaction, which aborts the attempt for a fresh plan (never a
    /// silent remap into another lineage).
    fn finalize_all(
        &self,
        namespace: NamespaceId,
        writes: &[BatchWriteSpec],
        txn: TxnId,
        commit: bool,
        digest: [u8; 32],
        expected: &std::collections::BTreeMap<Vec<u8>, TabletId>,
    ) -> Result<AtomicBatchResult, BatchAttempt> {
        let mut versions: Vec<Option<u64>> = vec![None; writes.len()];
        for (order, write) in writes.iter().enumerate() {
            let participant = expected.get(&write.key).copied();
            match self.txn_finalize(namespace, &write.key, txn, commit, digest) {
                Ok(FinalizeOutcome::Applied(version, tablet)) => {
                    if participant.is_some_and(|expected| expected != tablet) {
                        // Same stale-probe class as a served-elsewhere
                        // prepare: evict so the redrive re-probes fresh
                        // instead of conflicting forever.
                        self.evict_covering_for(namespace, &write.key);
                        tracing::debug!(
                            key = hex_key_trace(&write.key),
                            expected = ?participant.map(kivi_types::TabletId::as_u64),
                            served = tablet.as_u64(),
                            commit,
                            "2pc finalize lineage drift; evicting route",
                        );
                        return Err(BatchAttempt::Conflict);
                    }
                    versions[order] = version;
                }
                Ok(FinalizeOutcome::Gone) => {}
                Ok(FinalizeOutcome::Conflict) => return Err(BatchAttempt::Conflict),
                Err(error) if is_ambiguous(&error) => return Err(BatchAttempt::Redrive),
                Err(error) => return Err(BatchAttempt::Error(error)),
            }
        }
        Ok(AtomicBatchResult { txn, versions })
    }

    /// Finalizes every key on the recovery path (no serving-tablet
    /// expectations available): the record decision plus the digest
    /// binding carry correctness here.
    fn finalize_all_recovery(
        &self,
        namespace: NamespaceId,
        writes: &[BatchWriteSpec],
        txn: TxnId,
        commit: bool,
        digest: [u8; 32],
    ) -> Result<AtomicBatchResult, BatchAttempt> {
        let mut versions: Vec<Option<u64>> = vec![None; writes.len()];
        for (order, write) in writes.iter().enumerate() {
            match self.txn_finalize(namespace, &write.key, txn, commit, digest) {
                Ok(FinalizeOutcome::Applied(version, _)) => versions[order] = version,
                Ok(FinalizeOutcome::Gone) => {}
                Ok(FinalizeOutcome::Conflict) => return Err(BatchAttempt::Conflict),
                Err(error) if is_ambiguous(&error) => return Err(BatchAttempt::Redrive),
                Err(error) => return Err(BatchAttempt::Error(error)),
            }
        }
        Ok(AtomicBatchResult { txn, versions })
    }

    /// Maps the guarded abort outcome onto the drive result: `Aborted`
    /// yields the caller's abort outcome; `Committed` (a racing drive of
    /// this transaction committed) converges through the commit wave
    /// instead of reporting a false abort.
    fn abort_or_commit(
        &self,
        namespace: NamespaceId,
        writes: &[BatchWriteSpec],
        attempt: TxnAttempt,
        prepared: &[usize],
        aborted: BatchAttempt,
    ) -> Result<AtomicBatchResult, BatchAttempt> {
        match self.abort_prepared(
            namespace,
            writes,
            attempt.txn,
            attempt.coordinator,
            attempt.digest,
            prepared,
        ) {
            Ok(AbortOutcome::Aborted) => Err(aborted),
            Ok(AbortOutcome::Committed) => {
                self.finalize_all_recovery(namespace, writes, attempt.txn, true, attempt.digest)
            }
            Err(attempt) => Err(attempt),
        }
    }

    /// Persists the Abort decision (guarded CAS, never a blind overwrite)
    /// and delivers an abort finalize per prepared key through the
    /// durable wave. Returns which decision the coordinator record
    /// converged to: a lost race means a concurrent drive of this
    /// transaction committed, and the caller must converge via commit.
    /// An ambiguous record read propagates as `Redrive` (the state is
    /// unknown — the same-transaction re-drive resolves it recovery-first).
    fn abort_prepared(
        &self,
        namespace: NamespaceId,
        writes: &[BatchWriteSpec],
        txn: TxnId,
        coordinator: TabletId,
        digest: [u8; 32],
        prepared: &[usize],
    ) -> Result<AbortOutcome, BatchAttempt> {
        let record_key = kivi_state::txn_record_key(coordinator, txn);
        let record = Self::build_record(
            namespace,
            writes,
            txn,
            coordinator,
            Vec::new(),
            0,
            TxnState::Aborted,
        );
        if !self.decide_record(namespace, coordinator, &record_key, &record, 1) {
            // Lost the decide race: re-read (an ambiguous decide may
            // actually have committed — never finalize-abort a committed
            // transaction). A foreign digest under one `TxnId` fails the
            // attempt instead of following another transaction.
            let reread = self.read_txn_record(namespace, &record_key);
            if let Ok(record) = &reread {
                tracing::debug!(
                    coordinator = coordinator.as_u64(),
                    state = ?record.as_ref().map(|record| record.state),
                    "2pc abort decide failed; re-read decides the outcome",
                );
            } else {
                tracing::debug!(
                    coordinator = coordinator.as_u64(),
                    "2pc abort decide failed and record unreadable; redriving",
                );
            }
            return match reread {
                Ok(Some(record)) if record.state == TxnState::Committed => {
                    if record.digest != digest {
                        return Err(BatchAttempt::Conflict);
                    }
                    Ok(AbortOutcome::Committed)
                }
                Ok(_) => Ok(AbortOutcome::Aborted),
                Err(attempt) => Err(attempt),
            };
        }
        for index in prepared {
            self.finalize_abort_durable(namespace, &writes[*index].key, txn, digest);
        }
        Ok(AbortOutcome::Aborted)
    }

    /// Delivers one abort finalize durably: the Abort decision is already
    /// persisted, so a lost finalize would wedge the key behind a foreign
    /// intent until transaction-lease expiry (the resolver only reaps
    /// expired or grace-aged intents, and only when its reader/proposer
    /// roles align). Retry transient failures on the SAME `TxnId`
    /// (idempotent discard of our own intent — never a fresh identity),
    /// bounded by [`MAX_ABORT_WAVES`]. Terminal outcomes end the wave:
    /// `Applied`/`Gone` mean nothing of ours remains (a `Conflict` likewise
    /// proves our intent is gone — prepares reserve the key, so no foreign
    /// intent can coexist with ours), and terminal errors leave the
    /// persisted record for the resolver/lease fallback.
    fn finalize_abort_durable(
        &self,
        namespace: NamespaceId,
        key: &[u8],
        txn: TxnId,
        digest: [u8; 32],
    ) {
        let mut waves = 0u32;
        loop {
            match self.txn_finalize(namespace, key, txn, false, digest) {
                Err(error) if Self::retry_abort_wave(&error) && waves + 1 < MAX_ABORT_WAVES => {
                    waves += 1;
                    tracing::debug!(
                        waves,
                        %error,
                        "2pc abort finalize transient; redriving same txn id",
                    );
                    backoff(Duration::from_millis(20), waves.min(6));
                }
                // Applied/Gone/Conflict (nothing of ours remains) and
                // terminal errors (the persisted record converges the
                // rest): the wave for this key is over either way.
                outcome => {
                    if waves > 0 || outcome.is_err() {
                        tracing::debug!(
                            waves,
                            resolved = outcome.is_ok(),
                            "2pc abort finalize wave done",
                        );
                    }
                    return;
                }
            }
        }
    }

    /// Whether an abort-finalize failure is worth redriving on the same
    /// `TxnId`: transport/routing failures (the finalize definitely did
    /// not execute, or may have — both safe under the same idempotent
    /// discard) plus redirect-budget exhaustion after convergence churn
    /// (a backed-off redrive lands on the converged directory).
    /// Semantic rejections are terminal for the wave.
    fn retry_abort_wave(error: &ClientError) -> bool {
        is_ambiguous(error) || matches!(error, ClientError::TooManyRedirects)
    }

    /// Sends one prepare (OCC + intent reservation, no visible mutation).
    /// Steps are idempotent by `TxnId` (same id + key replays the
    /// reservation rather than duplicating it), so transport retries carry
    /// no separate identity — request identity and transaction identity
    /// stay distinct, with exactly-once resting on the latter.
    fn txn_prepare(
        &self,
        namespace: NamespaceId,
        write: &BatchWriteSpec,
        txn: TxnId,
        coordinator: TabletId,
        digest: [u8; 32],
    ) -> Result<TabletId, ClientError> {
        use kivi_protocol::{Opcode, ResponseBody};
        let wire = Self::encode_wire_write(write);
        let route_key = Key::from(write.key.clone());
        let mut request = self.request_base(&route_key, Opcode::TxnPrepare);
        request.namespace = namespace;
        request.key.clone_from(&write.key);
        request.batch_txn = txn.as_bytes();
        request.batch_writes = vec![wire];
        request.txn_coordinator = coordinator.as_u64();
        request.txn_digest = digest;
        let response = self.execute_raw(
            &route_key,
            Opcode::TxnPrepare,
            || request.clone(),
            None,
            kivi_types::ReadContract::Latest,
        )?;
        match (response.status, response.body) {
            (kivi_protocol::Status::Ok, ResponseBody::TxnPrepared { tablet, .. }) => {
                Ok(TabletId::from_u64(tablet))
            }
            (status, body) => Err(status_error(status, &body)),
        }
    }

    /// Sends one finalize (commit applies, abort discards; idempotent).
    /// Reports the serving tablet with every application so the driver can
    /// verify lineage (a finalize served elsewhere means the tablet moved
    /// mid-transaction).
    fn txn_finalize(
        &self,
        namespace: NamespaceId,
        key: &[u8],
        txn: TxnId,
        commit: bool,
        digest: [u8; 32],
    ) -> Result<FinalizeOutcome, ClientError> {
        use kivi_protocol::{Opcode, ResponseBody};
        let route_key = Key::from(key.to_vec());
        let mut request = self.request_base(&route_key, Opcode::TxnFinalize);
        request.namespace = namespace;
        request.key = key.to_vec();
        request.batch_txn = txn.as_bytes();
        request.txn_commit = commit;
        request.txn_digest = digest;
        let response = self.execute_raw(
            &route_key,
            Opcode::TxnFinalize,
            || request.clone(),
            None,
            kivi_types::ReadContract::Latest,
        )?;
        match (response.status, response.body) {
            (
                kivi_protocol::Status::Ok,
                ResponseBody::TxnFinalized {
                    applied,
                    version,
                    tablet,
                },
            ) => {
                if applied {
                    Ok(FinalizeOutcome::Applied(
                        (version != 0).then_some(version),
                        TabletId::from_u64(tablet),
                    ))
                } else {
                    Ok(FinalizeOutcome::Gone)
                }
            }
            (kivi_protocol::Status::TxnConflict, _) => Ok(FinalizeOutcome::Conflict),
            (status, body) => Err(status_error(status, &body)),
        }
    }

    /// Raw point read (bytes, no version).
    fn get_raw(&self, namespace: NamespaceId, key: &[u8]) -> Result<Option<Vec<u8>>, ClientError> {
        use kivi_protocol::{Opcode, ResponseBody};
        let route_key = Key::from(key.to_vec());
        let response = self.execute_raw(
            &route_key,
            Opcode::Get,
            || kivi_protocol::Request {
                contract: kivi_types::ReadContract::Latest,
                namespace,
                opcode: Opcode::Get,
                hint: None,
                key: key.to_vec(),
                value: None,
                delta: 0,
                expiry: 0,
                offset: 0,
                len: 0,
                condition: kivi_protocol::COND_ALWAYS,
                expiry_policy: kivi_protocol::EXPIRY_CLEAR,
                identity: None,
                ack_floor: kivi_types::RequestSeq::from_u64(0),
                scan_start: None,
                scan_end: None,
                scan_direction: kivi_protocol::SCAN_FORWARD,
                scan_max_items: 0,
                scan_max_bytes: 0,
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
            },
            None,
            kivi_types::ReadContract::Latest,
        )?;
        match (response.status, response.body) {
            (kivi_protocol::Status::Ok, ResponseBody::Value(value)) => Ok(Some(value)),
            (kivi_protocol::Status::NotFound, _) => Ok(None),
            (status, body) => Err(status_error(status, &body)),
        }
    }
}

/// One finalize RPC outcome.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FinalizeOutcome {
    /// Intent applied (`Some` version, `None` for deletes / recovered),
    /// plus the serving tablet (lineage verification).
    Applied(Option<u64>, TabletId),
    /// No intent left (idempotent replay or abort).
    Gone,
    /// A foreign intent blocks the key.
    Conflict,
}

/// One batch-attempt outcome: success, OCC conflict (new attempt),
/// ambiguity (same-transaction re-drive), or terminal error.
#[derive(Debug)]
enum BatchAttempt {
    /// OCC conflict: retry with a fresh `TxnId`.
    Conflict,
    /// Ambiguous step: re-drive the same `TxnId` (bounded).
    Redrive,
    /// Terminal failure.
    Error(ClientError),
}

/// Which decision the coordinator record converged to after the guarded
/// abort path.
#[derive(Clone, Copy, PartialEq, Eq)]
enum AbortOutcome {
    Aborted,
    Committed,
}

/// Identity of one transaction attempt across every 2PC step: the id,
/// the derived coordinator, and the write-set digest binding the exact
/// transaction. Steps take this (never bare fields) so a step can
/// neither mix attempts nor forget the binding.
#[derive(Debug, Clone, Copy)]
struct TxnAttempt {
    txn: TxnId,
    coordinator: TabletId,
    digest: [u8; 32],
}

/// One 2PC drive's probed geography: the participant map, the derived
/// lowest-tablet coordinator, and the directory version the plan routes
/// against. Prepare/decide/finalize waves consume this (never
/// re-probe), so a routing move mid-drive aborts instead of silently
/// remapping.
struct ProbedDrive {
    probed: std::collections::BTreeMap<Vec<u8>, (TabletId, u64)>,
    coordinator: TabletId,
    dir_version: u64,
}

/// One 2PC drive's prepared state: which request indexes reserved, and
/// which tablets served them. The decide wave verifies this against
/// the probe before persisting anything.
struct PreparedDrive {
    prepared: Vec<usize>,
    observed: std::collections::BTreeSet<TabletId>,
}

impl From<BatchAttempt> for ClientError {
    fn from(attempt: BatchAttempt) -> Self {
        match attempt {
            BatchAttempt::Conflict => ClientError::TxnConflict,
            BatchAttempt::Redrive => ClientError::TxnCoordinatorUnavailable,
            BatchAttempt::Error(error) => error,
        }
    }
}

/// Short hex for 2PC debug lines (first key bytes only, no model data).
fn hex_key_trace(key: &[u8]) -> String {
    use std::fmt::Write as _;
    let mut out = String::new();
    for byte in key.iter().take(8) {
        let _ = write!(out, "{byte:02x}");
    }
    out
}

/// Whether an error is ambiguous (the step may or may not have executed):
/// re-driving the same `TxnId` is safe, assuming a new one is not.
fn is_ambiguous(error: &ClientError) -> bool {
    matches!(
        error,
        ClientError::AmbiguousOutcome
            | ClientError::TxnCoordinatorUnavailable
            | ClientError::Overloaded
            | ClientError::Io(_)
            | ClientError::Timeout
    )
}

// ---------------------------------------------------------------------------
// Escrow transfer + stream affinity
// ---------------------------------------------------------------------------

impl NativeClient {
    /// Maps a partition key to a shard index in `[0, shard_count)`:
    /// stable and deterministic (FNV-1a 64 over the partition bytes),
    /// so the same partition always routes to the same shard key while
    /// different partitions spread. A routing hint, not a correctness
    /// mechanism: per-shard order holds whatever the mapping, and changing
    /// `shard_count` remaps (streams scale by adding shards, then
    /// trimming the old ones once drained).
    #[must_use]
    pub fn stream_shard_for(partition: &[u8], shard_count: u32) -> u32 {
        if shard_count == 0 {
            return 0;
        }
        let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
        for byte in partition {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(0x0100_0000_01b3);
        }
        // The remainder under a `u32` count always fits `u32`.
        u32::try_from(hash % u64::from(shard_count)).unwrap_or(u32::MAX)
    }

    /// Atomically moves `amount` of escrow rights from the bounded counter
    /// at `donor_key` to the one at `recipient_key`: reads both shares and
    /// versions, plans the paired move
    /// ([`kivi_state::plan_escrow_transfer`], conservation by
    /// construction), and commits both share moves in one transaction with
    /// version expectations — a concurrent move fails the OCC check and
    /// surfaces [`ClientError::TxnConflict`] for a fresh retry instead of
    /// minting or destroying rights.
    ///
    /// Cross-tablet pairs commit via 2PC; colocated pairs via the
    /// single-tablet fast path. Either way the transfer is atomic.
    /// Operates in this client's namespace.
    ///
    /// # Errors
    ///
    /// Returns [`ClientError`] on missing/non-bounded keys
    /// ([`ClientError::BoundedExceeded`]), unplannable moves
    /// ([`ClientError::TxnConflict`] when the donor cannot cover `amount`,
    /// [`ClientError::InvalidRequest`] for malformed amounts or capacity),
    /// OCC conflicts (retry), or transport/routing failure.
    pub fn escrow_transfer(
        &self,
        donor_key: &[u8],
        recipient_key: &[u8],
        amount: u64,
    ) -> Result<(Option<u64>, Option<u64>), ClientError> {
        use super::ClientError as E;
        let namespace = self.shared.namespace;
        let donor_key = Key::from(donor_key.to_vec());
        let recipient_key = Key::from(recipient_key.to_vec());
        let donor = self.bounded_get(&donor_key)?.ok_or(E::BoundedExceeded)?;
        let recipient = self
            .bounded_get(&recipient_key)?
            .ok_or(E::BoundedExceeded)?;
        let donor_version = self.get_version(namespace, donor_key.as_bytes())?;
        let recipient_version = self.get_version(namespace, recipient_key.as_bytes())?;
        let ((donor_min, donor_new_max), (recipient_min, recipient_new_max)) =
            kivi_state::plan_escrow_transfer(
                (donor.share_min, donor.share_max),
                donor.value,
                (recipient.share_min, recipient.share_max),
                recipient.capacity,
                amount,
            )
            .map_err(|error| match error {
                kivi_state::TxnError::Conflict => E::TxnConflict,
                _ => E::InvalidRequest,
            })?;
        let writes = vec![
            BatchWriteSpec {
                key: donor_key.as_bytes().to_vec(),
                kind: BatchWriteKind::EscrowSetShare {
                    min: donor_min,
                    max: donor_new_max,
                },
                expect: donor_version.map_or(BatchExpect::Absent, BatchExpect::Version),
            },
            BatchWriteSpec {
                key: recipient_key.as_bytes().to_vec(),
                kind: BatchWriteKind::EscrowSetShare {
                    min: recipient_min,
                    max: recipient_new_max,
                },
                expect: recipient_version.map_or(BatchExpect::Absent, BatchExpect::Version),
            },
        ];
        let result = self.atomic_batch(namespace, &writes)?;
        let mut versions = result.versions.into_iter();
        Ok((versions.next().flatten(), versions.next().flatten()))
    }
}

// ---------------------------------------------------------------------------
// Secondary index
// ---------------------------------------------------------------------------

/// One index term set for a single index on one primary write.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndexTermSet {
    /// Index identity.
    pub index: IndexId,
    /// Index kind (encoding + uniqueness contract).
    pub kind: IndexKind,
    /// Current terms for the primary value.
    pub terms: Vec<Vec<u8>>,
}

/// One index query hit: a primary reference with its indexed version.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndexHit {
    /// Referenced primary key.
    pub primary_key: Vec<u8>,
    /// Primary version the entry was written against.
    pub version: u64,
}

/// Maximum index entries resolved per query page (bounds the N-way fan-out).
const INDEX_RESOLVE_LIMIT: usize = 1000;
/// Parallel primary-resolution width.
const INDEX_RESOLVE_WIDTH: usize = 8;

impl NativeClient {
    /// Writes a primary value with its index terms atomically: the primary
    /// Put, the projection record, stale index-entry deletes, and new index
    /// entries commit as one transaction across tablets. Terms for indexes
    /// not mentioned are left alone (pass empty terms to clear an index).
    ///
    /// Stale index entries from crashed or superseded writes are skipped
    /// by version validation at query time and removed here when observed.
    ///
    /// # Errors
    ///
    /// Returns [`ClientError`] on conflicts (retry), uniqueness violations
    /// (stable), or transport/routing failure.
    pub fn indexed_set(
        &self,
        namespace: NamespaceId,
        primary_key: &[u8],
        value: Vec<u8>,
        terms: &[IndexTermSet],
    ) -> Result<AtomicBatchResult, ClientError> {
        // Read current version + projection (TOCTOU is closed by OCC: the
        // primary write carries the observed version as its expectation).
        let observed_version = self.get_version(namespace, primary_key)?;
        let projection_bytes = self.get_raw(namespace, projection_key(primary_key).as_bytes())?;
        let old_projection = projection_bytes
            .map(|bytes| kivi_state::IndexProjections::decode(&bytes))
            .transpose()
            .map_err(|_| ClientError::TxnAborted)?;
        let mut old_terms: std::collections::BTreeMap<IndexId, Vec<Vec<u8>>> =
            std::collections::BTreeMap::new();
        if let Some(projection) = old_projection {
            old_terms = projection.terms;
        }
        let new_version = observed_version.map_or(1, |version| version + 1);
        let expect = match observed_version {
            None => BatchExpect::Absent,
            Some(version) => BatchExpect::Version(version),
        };
        // Diff old vs new terms per index: deletes for stale, puts for new.
        let mut writes: Vec<BatchWriteSpec> = Vec::new();
        let mut new_projection = kivi_state::IndexProjections::default();
        for term_set in terms {
            new_projection
                .terms
                .insert(term_set.index, term_set.terms.clone());
            let old = old_terms.remove(&term_set.index).unwrap_or_default();
            let (stale, fresh) = diff_terms(&old, &term_set.terms);
            for term in stale {
                let key = match term_set.kind {
                    IndexKind::NonUnique => {
                        encode_non_unique_key(term_set.index, &term, primary_key)
                    }
                    IndexKind::Unique => encode_unique_key(term_set.index, &term),
                };
                writes.push(BatchWriteSpec {
                    key,
                    kind: BatchWriteKind::Delete,
                    expect: BatchExpect::Any,
                });
            }
            for term in fresh {
                let (key, value) = match term_set.kind {
                    IndexKind::NonUnique => (
                        encode_non_unique_key(term_set.index, &term, primary_key),
                        encode_non_unique_value(kivi_state::ObjectVersion::from_u64(new_version)),
                    ),
                    IndexKind::Unique => (
                        encode_unique_key(term_set.index, &term),
                        encode_unique_value(
                            kivi_state::ObjectVersion::from_u64(new_version),
                            primary_key,
                        ),
                    ),
                };
                let expect = match term_set.kind {
                    // Unique claims must find no live owner; a live owner is
                    // a stable violation, not a blind overwrite.
                    IndexKind::Unique => BatchExpect::Absent,
                    IndexKind::NonUnique => BatchExpect::Any,
                };
                writes.push(BatchWriteSpec {
                    key,
                    kind: BatchWriteKind::Put(value),
                    expect,
                });
            }
        }
        // Dropped indexes (in old projection, absent now): remove all old.
        // (Kind unknown post-drop; try unique-key delete by term — the
        // non-unique form needs the primary suffix which we still have, so
        // delete both spellings to be thorough. Bounded: old term count.)
        for (index, old) in &old_terms {
            for term in old {
                writes.push(BatchWriteSpec {
                    key: encode_non_unique_key(*index, term, primary_key),
                    kind: BatchWriteKind::Delete,
                    expect: BatchExpect::Any,
                });
                writes.push(BatchWriteSpec {
                    key: encode_unique_key(*index, term),
                    kind: BatchWriteKind::Delete,
                    expect: BatchExpect::Any,
                });
            }
        }
        // Primary + projection last (request order is irrelevant to
        // atomicity, but stable for tests).
        writes.push(BatchWriteSpec {
            key: primary_key.to_vec(),
            kind: BatchWriteKind::Put(value),
            expect,
        });
        writes.push(BatchWriteSpec {
            key: projection_key(primary_key).as_bytes().to_vec(),
            kind: BatchWriteKind::Put(new_projection.encode()),
            expect: BatchExpect::Any,
        });
        match self.atomic_batch(namespace, &writes) {
            Ok(result) => Ok(result),
            Err(ClientError::TxnConflict)
                if terms.iter().any(|set| set.kind == IndexKind::Unique) =>
            {
                // A unique claim may have lost to a committed owner:
                // distinguish stable violation from plain contention.
                self.map_unique_conflict(namespace, primary_key, terms)
            }
            Err(error) => Err(error),
        }
    }

    /// Maps a conflict from an indexed write onto a stable uniqueness
    /// violation when another primary owns the claimed term.
    fn map_unique_conflict(
        &self,
        namespace: NamespaceId,
        primary_key: &[u8],
        terms: &[IndexTermSet],
    ) -> Result<AtomicBatchResult, ClientError> {
        for term_set in terms {
            if term_set.kind != IndexKind::Unique {
                continue;
            }
            for term in &term_set.terms {
                let key = encode_unique_key(term_set.index, term);
                if let Some(bytes) = self.get_raw(namespace, &key)?
                    && let Ok((_, owner)) = decode_entry_value(&bytes)
                    && owner != primary_key
                {
                    return Err(ClientError::UniqueViolation);
                }
            }
        }
        Err(ClientError::TxnConflict)
    }

    /// Deletes a primary with its index maintenance atomically: primary
    /// Delete, projection Delete, and stale index-entry Deletes commit as
    /// one transaction. Missing primaries delete nothing (still cleans any
    /// orphan index entries named by the projection).
    ///
    /// # Errors
    ///
    /// Returns [`ClientError`] on conflicts (retry) or transport failure.
    pub fn indexed_delete(
        &self,
        namespace: NamespaceId,
        primary_key: &[u8],
    ) -> Result<AtomicBatchResult, ClientError> {
        let observed_version = self.get_version(namespace, primary_key)?;
        let projection_bytes = self.get_raw(namespace, projection_key(primary_key).as_bytes())?;
        let mut writes: Vec<BatchWriteSpec> = Vec::new();
        if let Some(bytes) = projection_bytes
            && let Ok(projection) = kivi_state::IndexProjections::decode(&bytes)
        {
            for (index, old) in &projection.terms {
                for term in old {
                    // Both spellings (see `indexed_set` drop path).
                    writes.push(BatchWriteSpec {
                        key: encode_non_unique_key(*index, term, primary_key),
                        kind: BatchWriteKind::Delete,
                        expect: BatchExpect::Any,
                    });
                    writes.push(BatchWriteSpec {
                        key: encode_unique_key(*index, term),
                        kind: BatchWriteKind::Delete,
                        expect: BatchExpect::Any,
                    });
                }
            }
        }
        writes.push(BatchWriteSpec {
            key: primary_key.to_vec(),
            kind: BatchWriteKind::Delete,
            expect: match observed_version {
                None => BatchExpect::Any,
                Some(version) => BatchExpect::Version(version),
            },
        });
        writes.push(BatchWriteSpec {
            key: projection_key(primary_key).as_bytes().to_vec(),
            kind: BatchWriteKind::Delete,
            expect: BatchExpect::Any,
        });
        self.atomic_batch(namespace, &writes)
    }

    /// Equality lookup over a non-unique index: all primaries currently
    /// indexed under `term`, in primary-key order. Entries live beside
    /// their primaries, so the lookup fans out per tablet: every tablet
    /// reports the primaries it holds for the term (bounded page each),
    /// and version validation skips stale entries (primary moved or
    /// deleted since).
    ///
    /// # Errors
    ///
    /// Returns [`ClientError`] on transport/routing failure.
    pub fn index_equal(
        &self,
        namespace: NamespaceId,
        index: IndexId,
        term: &[u8],
        limit: usize,
    ) -> Result<Vec<Vec<u8>>, ClientError> {
        let mut hits: Vec<IndexHit> = Vec::new();
        for entry in self.index_term_entries(namespace, index, term, limit)? {
            let (found_index, found_term, primary) = decode_non_unique_key(&entry.key)
                .map_err(|_| ClientError::Internal("index entry failed to decode".to_owned()))?;
            if found_index != index || found_term != term {
                continue;
            }
            let bytes = match &entry.value {
                ClientScanValue::Inline(bytes) => bytes.to_vec(),
                _ => continue,
            };
            let (version, _) = decode_entry_value(&bytes)
                .map_err(|_| ClientError::Internal("index value failed to decode".to_owned()))?;
            hits.push(IndexHit {
                primary_key: primary,
                version: version.as_u64(),
            });
        }
        self.validate_hits(namespace, &hits)
    }

    /// Lists every tablet's entries for a non-unique term: fans the term
    /// out per tablet (each tablet answers from the entries beside its
    /// primaries) and merges the slices in primary-key order. The window
    /// is the term prefix itself (`[prefix, successor)`), so each
    /// tablet's `scan_index_term` path answers its own slice directly;
    /// the per-tablet walk below only advances the tiling cursor.
    #[allow(clippy::too_many_lines)]
    fn index_term_entries(
        &self,
        namespace: NamespaceId,
        index: IndexId,
        term: &[u8],
        limit: usize,
    ) -> Result<Vec<ClientScanEntry>, ClientError> {
        use kivi_protocol::{Opcode, ResponseBody};
        use kivi_state::{prefix_successor, term_prefix};
        let bound = limit.min(INDEX_RESOLVE_LIMIT);
        let prefix = term_prefix(index, term, false);
        let end = prefix_successor(&prefix);
        let mut out: Vec<ClientScanEntry> = Vec::new();
        // The server fans one term window out per tablet internally, so a
        // single windowed scan covers every tablet's co-located slice.
        let route_key = Key::from(prefix.clone());
        let mut request = self.request_base(&route_key, Opcode::Scan);
        request.namespace = namespace;
        request.scan_direction = kivi_protocol::SCAN_FORWARD;
        request.scan_projection = kivi_protocol::SCAN_KEYS_AND_VALUES;
        request.scan_consistency = kivi_protocol::SCAN_LATEST_PER_TABLET;
        request.scan_max_items = 1000;
        request.scan_max_bytes = 1 << 20;
        request.scan_start = Some(prefix);
        request.scan_end.clone_from(&end);
        let response = self.execute_raw(
            &route_key,
            Opcode::Scan,
            || request.clone(),
            None,
            kivi_types::ReadContract::Latest,
        )?;
        let entries = match (response.status, response.body) {
            (kivi_protocol::Status::Ok, ResponseBody::ScanPage { entries, .. }) => entries,
            (status, body) => return Err(status_error(status, &body)),
        };
        for entry in entries {
            let decoded = entry.key.clone();
            let value = scan_body_to_client(entry.value);
            if decode_non_unique_key(&decoded)
                .is_ok_and(|(found, found_term, _)| found == index && found_term == term)
            {
                out.push(ClientScanEntry {
                    key: decoded,
                    value,
                });
                if out.len() >= bound {
                    break;
                }
            }
        }
        out.sort_by(|left, right| left.key.cmp(&right.key));
        out.truncate(bound);
        Ok(out)
    }

    /// Unique-term lookup: the single primary owning `term`, if any and
    /// current. Returns `Ok(None)` for unclaimed or stale terms.
    ///
    /// # Errors
    ///
    /// Returns [`ClientError`] on transport/routing failure.
    pub fn index_equal_unique(
        &self,
        namespace: NamespaceId,
        index: IndexId,
        term: &[u8],
    ) -> Result<Option<Vec<u8>>, ClientError> {
        let key = encode_unique_key(index, term);
        let Some(bytes) = self.get_raw(namespace, &key)? else {
            return Ok(None);
        };
        let (version, primary) = decode_entry_value(&bytes)
            .map_err(|_| ClientError::Internal("index value failed to decode".to_owned()))?;
        match self.get_version(namespace, primary)? {
            Some(current) if current == version.as_u64() => Ok(Some(primary.to_vec())),
            _ => Ok(None),
        }
    }

    /// Range lookup over index terms `[start_term, end_term)`: primaries
    /// for every term in the window, in `(term, primary)` order. Entries
    /// live beside their primaries; the server fans the whole term range
    /// out per tablet, so one windowed scan covers every term at once.
    /// Stale entries are skipped by version validation.
    ///
    /// # Errors
    ///
    /// Returns [`ClientError`] on transport/routing failure.
    pub fn index_range(
        &self,
        namespace: NamespaceId,
        index: IndexId,
        start_term: &[u8],
        end_term: &[u8],
        limit: usize,
    ) -> Result<Vec<Vec<u8>>, ClientError> {
        use kivi_protocol::{Opcode, ResponseBody};
        use kivi_state::term_prefix;
        let bound = limit.min(INDEX_RESOLVE_LIMIT);
        // One windowed scan: `[prefix(start), prefix(end))` fans out per
        // tablet server-side, so every term in the range answers at once.
        let start = term_prefix(index, start_term, false);
        let end = term_prefix(index, end_term, false);
        let route_key = Key::from(start.clone());
        let mut request = self.request_base(&route_key, Opcode::Scan);
        request.namespace = namespace;
        request.scan_direction = kivi_protocol::SCAN_FORWARD;
        request.scan_projection = kivi_protocol::SCAN_KEYS_AND_VALUES;
        request.scan_consistency = kivi_protocol::SCAN_LATEST_PER_TABLET;
        request.scan_max_items = 1000;
        request.scan_max_bytes = 1 << 20;
        request.scan_start = Some(start);
        request.scan_end = Some(end);
        let response = self.execute_raw(
            &route_key,
            Opcode::Scan,
            || request.clone(),
            None,
            kivi_types::ReadContract::Latest,
        )?;
        let entries = match (response.status, response.body) {
            (kivi_protocol::Status::Ok, ResponseBody::ScanPage { entries, .. }) => entries,
            (status, body) => return Err(status_error(status, &body)),
        };
        let mut hits: Vec<IndexHit> = Vec::new();
        for entry in entries {
            if let Ok((found_index, found_term, primary)) = decode_non_unique_key(&entry.key) {
                if found_index != index {
                    continue;
                }
                if found_term.as_slice() < start_term || found_term.as_slice() >= end_term {
                    continue;
                }
                let bytes = match scan_body_to_client(entry.value) {
                    ClientScanValue::Inline(bytes) => bytes.to_vec(),
                    _ => continue,
                };
                let (version, _) = decode_entry_value(&bytes).map_err(|_| {
                    ClientError::Internal("index value failed to decode".to_owned())
                })?;
                hits.push(IndexHit {
                    primary_key: primary,
                    version: version.as_u64(),
                });
            }
            if hits.len() >= bound {
                break;
            }
        }
        hits.sort_by(|left, right| left.primary_key.cmp(&right.primary_key));
        hits.truncate(bound);
        self.validate_hits(namespace, &hits)
    }

    /// Validates hits against current primary versions (bounded parallel
    /// fan-out): entries whose primary moved or vanished are skipped.
    fn validate_hits(
        &self,
        namespace: NamespaceId,
        hits: &[IndexHit],
    ) -> Result<Vec<Vec<u8>>, ClientError> {
        if hits.is_empty() {
            return Ok(Vec::new());
        }
        let width = INDEX_RESOLVE_WIDTH;
        let mut valid: Vec<Vec<u8>> = Vec::new();
        for chunk in hits.chunks(width) {
            let chunk = chunk.to_vec();
            let (send, receive) = std::sync::mpsc::channel();
            std::thread::scope(|scope| {
                for hit in &chunk {
                    let send = send.clone();
                    let client = self.clone();
                    scope.spawn(move || {
                        let current = client.get_version(namespace, &hit.primary_key);
                        let _ = send.send((hit.primary_key.clone(), hit.version, current));
                    });
                }
            });
            drop(send);
            let mut ordered: std::collections::BTreeMap<Vec<u8>, (u64, Option<u64>)> =
                std::collections::BTreeMap::new();
            for (key, version, current) in receive {
                let current = current?;
                ordered.insert(key, (version, current));
            }
            for (key, (version, current)) in ordered {
                if current == Some(version) {
                    valid.push(key);
                }
            }
        }
        Ok(valid)
    }

    /// Fetches validated primary values for index hits (bounded parallel
    /// `get_version` + `get` pairs; pairs that moved between the two reads
    /// are skipped, never returned stale).
    ///
    /// # Errors
    ///
    /// Returns [`ClientError`] on transport/routing failure.
    pub fn index_resolve_values(
        &self,
        namespace: NamespaceId,
        keys: &[Vec<u8>],
    ) -> Result<Vec<(Vec<u8>, Bytes)>, ClientError> {
        let mut out = Vec::new();
        for chunk in keys.chunks(INDEX_RESOLVE_WIDTH) {
            let chunk = chunk.to_vec();
            let (send, receive) = std::sync::mpsc::channel();
            std::thread::scope(|scope| {
                for key in &chunk {
                    let send = send.clone();
                    let client = self.clone();
                    scope.spawn(move || {
                        let version = client.get_version(namespace, key);
                        let value = client.get_raw(namespace, key);
                        let _ = send.send((key.clone(), version, value));
                    });
                }
            });
            drop(send);
            for (key, version, value) in receive {
                if let (Some(_), Some(bytes)) = (version?, value?) {
                    out.push((key, Bytes::from(bytes)));
                }
            }
        }
        Ok(out)
    }

    /// Online application-driven backfill: installs index entries for one
    /// primary only if its version still matches, so concurrent indexed
    /// writes win deterministically and the backfill retries or skips.
    ///
    /// # Errors
    ///
    /// Returns [`ClientError::TxnConflict`] when the primary moved.
    pub fn index_backfill(
        &self,
        namespace: NamespaceId,
        primary_key: &[u8],
        expected_version: u64,
        terms: &[IndexTermSet],
    ) -> Result<AtomicBatchResult, ClientError> {
        let current = self.get_version(namespace, primary_key)?;
        if current != Some(expected_version) {
            return Err(ClientError::TxnConflict);
        }
        let projection_bytes = self.get_raw(namespace, projection_key(primary_key).as_bytes())?;
        let mut merged = projection_bytes
            .map(|bytes| kivi_state::IndexProjections::decode(&bytes))
            .transpose()
            .map_err(|_| ClientError::TxnAborted)?
            .unwrap_or_default();
        let mut writes: Vec<BatchWriteSpec> = Vec::new();
        for term_set in terms {
            let entry = merged.terms.entry(term_set.index).or_default();
            for term in &term_set.terms {
                if !entry.contains(term) {
                    entry.push(term.clone());
                }
                let (key, value) = match term_set.kind {
                    IndexKind::NonUnique => (
                        encode_non_unique_key(term_set.index, term, primary_key),
                        encode_non_unique_value(kivi_state::ObjectVersion::from_u64(
                            expected_version,
                        )),
                    ),
                    IndexKind::Unique => (
                        encode_unique_key(term_set.index, term),
                        encode_unique_value(
                            kivi_state::ObjectVersion::from_u64(expected_version),
                            primary_key,
                        ),
                    ),
                };
                writes.push(BatchWriteSpec {
                    key,
                    kind: BatchWriteKind::Put(value),
                    expect: match term_set.kind {
                        IndexKind::Unique => BatchExpect::Absent,
                        IndexKind::NonUnique => BatchExpect::Any,
                    },
                });
            }
        }
        writes.push(BatchWriteSpec {
            key: projection_key(primary_key).as_bytes().to_vec(),
            kind: BatchWriteKind::Put(merged.encode()),
            expect: BatchExpect::Any,
        });
        // The primary is untouched, but the version check above plus the
        // entries' embedded versions keep this safe: a concurrent primary
        // write supersedes these entries (they validate stale) or
        // conflicts here (unique claims race deterministically).
        self.atomic_batch(namespace, &writes)
    }
}

/// Splits old/new term lists into `(stale, fresh)` (order-free, deduped).
fn diff_terms(old: &[Vec<u8>], new: &[Vec<u8>]) -> (Vec<Vec<u8>>, Vec<Vec<u8>>) {
    use std::collections::BTreeSet;
    let old_set: BTreeSet<&[u8]> = old.iter().map(Vec::as_slice).collect();
    let new_set: BTreeSet<&[u8]> = new.iter().map(Vec::as_slice).collect();
    (
        old_set
            .difference(&new_set)
            .map(|term| term.to_vec())
            .collect(),
        new_set
            .difference(&old_set)
            .map(|term| term.to_vec())
            .collect(),
    )
}
