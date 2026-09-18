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
    decode_unique_key, encode_non_unique_key, encode_non_unique_value, encode_unique_key,
    encode_unique_value, prefix_successor, projection_key, term_prefix,
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
    /// Resident bytes exceeding the page budget (length only).
    Oversize {
        /// Logical length.
        logical_len: u64,
    },
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
                    dir_version: _,
                },
            ) => Ok(ClientScanPage {
                entries: entries
                    .into_iter()
                    .map(|entry| ClientScanEntry {
                        key: entry.key,
                        value: match entry.value {
                            kivi_protocol::ScanValueBody::None => ClientScanValue::Absent,
                            kivi_protocol::ScanValueBody::Inline(bytes) => {
                                ClientScanValue::Inline(Bytes::from(bytes))
                            }
                            kivi_protocol::ScanValueBody::Counter(counter) => {
                                ClientScanValue::Counter(counter)
                            }
                            kivi_protocol::ScanValueBody::Chunked {
                                manifest,
                                logical_len,
                            } => ClientScanValue::Chunked {
                                manifest,
                                logical_len,
                            },
                            kivi_protocol::ScanValueBody::Oversize { logical_len } => {
                                ClientScanValue::Oversize { logical_len }
                            }
                        },
                    })
                    .collect(),
                exhausted,
                last_key,
                tablet,
                range_start,
                range_end,
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
    /// Remove the key.
    Delete,
    /// Add to a counter.
    CounterAdd(i64),
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
                        BatchWriteKind::Delete | BatchWriteKind::CounterAdd(_) => 8,
                    }
            })
            .sum();
        if bytes > kivi_state::MAX_TXN_BYTES {
            return Err(ClientError::TxnTooLarge);
        }
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
                Err(BatchAttempt::Conflict) => return Err(ClientError::TxnConflict),
                Err(BatchAttempt::Redrive) => {
                    // Same-transaction re-drive: bounded internally.
                    return self.redrive(namespace, writes, txn);
                }
                Err(BatchAttempt::Error(error)) => return Err(error),
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

    /// One batch attempt under a fixed `TxnId`: fast path first, then
    /// client-driven 2PC on multi-tablet rejection.
    fn batch_attempt(
        &self,
        namespace: NamespaceId,
        writes: &[BatchWriteSpec],
        txn: TxnId,
    ) -> Result<AtomicBatchResult, BatchAttempt> {
        use kivi_protocol::{Opcode, ResponseBody};
        let mut wire = Vec::with_capacity(writes.len());
        for write in writes {
            let (kind, value, delta) = match &write.kind {
                BatchWriteKind::Put(value) => (kivi_protocol::BATCH_PUT, value.clone(), 0),
                BatchWriteKind::Delete => (kivi_protocol::BATCH_DELETE, Vec::new(), 0),
                BatchWriteKind::CounterAdd(delta) => {
                    (kivi_protocol::BATCH_COUNTER_ADD, Vec::new(), *delta)
                }
            };
            let (expect, expect_version) = match write.expect {
                BatchExpect::Any => (kivi_protocol::BATCH_EXPECT_ANY, 0),
                BatchExpect::Absent => (kivi_protocol::BATCH_EXPECT_ABSENT, 0),
                BatchExpect::Version(version) => (kivi_protocol::BATCH_EXPECT_VERSION, version),
            };
            wire.push(kivi_protocol::BatchWrite {
                key: write.key.clone(),
                kind,
                value,
                delta,
                expect,
                expect_version,
            });
        }
        let route_key = Key::from(writes[0].key.clone());
        let response = self
            .execute_raw(
                &route_key,
                Opcode::AtomicBatch,
                || kivi_protocol::Request {
                    contract: kivi_types::ReadContract::Latest,
                    namespace,
                    opcode: Opcode::AtomicBatch,
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
                    scan_start: None,
                    scan_end: None,
                    scan_direction: kivi_protocol::SCAN_FORWARD,
                    scan_max_items: 0,
                    scan_max_bytes: 0,
                    scan_projection: kivi_protocol::SCAN_KEYS_ONLY,
                    scan_consistency: kivi_protocol::SCAN_LATEST_PER_TABLET,
                    batch_txn: txn.as_bytes(),
                    batch_writes: wire.clone(),
                    txn_coordinator: 0,
                    txn_commit: false,
                },
                None,
                kivi_types::ReadContract::Latest,
            )
            .map_err(BatchAttempt::Error)?;
        match (response.status, response.body) {
            (kivi_protocol::Status::Ok, ResponseBody::AtomicCommitted { versions }) => {
                Ok(AtomicBatchResult { txn, versions })
            }
            (kivi_protocol::Status::TxnTooLarge, ResponseBody::Diagnostic(message))
                if message.contains("tablets") =>
            {
                self.drive_2pc(namespace, writes, txn)
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

    /// Client-driven OCC + 2PC across tablets: probe coordinator, prepare
    /// every key, persist the decision, finalize all. Recovery-first: an
    /// existing durable decision skips straight to finalizes.
    fn drive_2pc(
        &self,
        namespace: NamespaceId,
        writes: &[BatchWriteSpec],
        txn: TxnId,
    ) -> Result<AtomicBatchResult, BatchAttempt> {
        // Recovery-first: a durable decision from an earlier drive of this
        // transaction resolves without re-preparing (re-prepares after a
        // commit would spuriously conflict with the committed state).
        // The coordinator is unknown before the first prepare, so probe
        // the record optimistically is impossible: learn it below.
        let coordinator = self.probe_coordinator(namespace, &writes[0].key)?;
        let record_key = kivi_state::txn_record_key(coordinator, txn);
        if let Some(record) = self.read_txn_record(namespace, &record_key)? {
            match record.state {
                TxnState::Committed => {
                    return self.finalize_all(namespace, writes, txn, true);
                }
                TxnState::Aborted => return Err(BatchAttempt::Error(ClientError::TxnAborted)),
                TxnState::Begun => {}
            }
        }
        // Prepare every key in request order (deterministic), learning
        // participant tablets from the prepare responses.
        let mut prepared: Vec<usize> = Vec::with_capacity(writes.len());
        let mut observed: std::collections::BTreeSet<TabletId> = std::collections::BTreeSet::new();
        for (index, write) in writes.iter().enumerate() {
            match self.txn_prepare(namespace, write, txn, coordinator) {
                Ok(tablet) => {
                    observed.insert(tablet);
                    prepared.push(index);
                }
                Err(
                    ClientError::TxnConflict
                    | ClientError::WrongType
                    | ClientError::CounterOverflow,
                ) => {
                    return self.abort_or_commit(
                        namespace,
                        writes,
                        txn,
                        coordinator,
                        &prepared,
                        BatchAttempt::Conflict,
                    );
                }
                Err(error) if is_ambiguous(&error) => {
                    return self.abort_or_commit(
                        namespace,
                        writes,
                        txn,
                        coordinator,
                        &prepared,
                        BatchAttempt::Redrive,
                    );
                }
                Err(error) => {
                    return self.abort_or_commit(
                        namespace,
                        writes,
                        txn,
                        coordinator,
                        &prepared,
                        BatchAttempt::Error(error),
                    );
                }
            }
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
            observed.into_iter().collect(),
            TxnState::Committed,
        );
        if !self.decide_record(namespace, coordinator, &record_key, &record, 0) {
            // Decide failed: re-read before aborting (an ambiguous decide
            // may actually have committed — never blind-overwrite a
            // decision with an abort).
            match self.read_txn_record(namespace, &record_key) {
                Ok(Some(record)) if record.state == TxnState::Committed => {
                    return self.finalize_all(namespace, writes, txn, true);
                }
                Ok(_) => {}
                Err(BatchAttempt::Redrive) => return Err(BatchAttempt::Redrive),
                Err(BatchAttempt::Conflict) => return Err(BatchAttempt::Conflict),
                Err(BatchAttempt::Error(error)) => return Err(BatchAttempt::Error(error)),
            }
            return self.abort_or_commit(
                namespace,
                writes,
                txn,
                coordinator,
                &prepared,
                BatchAttempt::Redrive,
            );
        }
        self.finalize_all(namespace, writes, txn, true)
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
        // re-drives are idempotent).
        let decide_txn = TxnId::derive(u128::from_le_bytes(record.id.as_bytes()), 0, decide_salt);
        match self.txn_prepare(namespace, &write, decide_txn, coordinator) {
            Ok(_) => {}
            Err(_) => return false,
        }
        // Only an applied finalize persists the decision: `Gone` means
        // the intent vanished without applying (never happens for a fresh
        // decide intent outside a racing abort — treat as failure and let
        // the caller re-read the record).
        matches!(
            self.txn_finalize(namespace, record_key.as_bytes(), decide_txn, true),
            Ok(FinalizeOutcome::Applied(_))
        )
    }

    /// Builds the canonical coordinator record for a write set.
    fn build_record(
        namespace: NamespaceId,
        writes: &[BatchWriteSpec],
        txn: TxnId,
        coordinator: TabletId,
        participants: Vec<TabletId>,
        state: TxnState,
    ) -> TxnRecord {
        let typed: Vec<kivi_state::TxnWrite> = writes
            .iter()
            .map(|write| {
                let kind = match &write.kind {
                    BatchWriteKind::Put(value) => {
                        kivi_state::TxnWriteKind::Put(Bytes::from(value.clone()))
                    }
                    BatchWriteKind::Delete => kivi_state::TxnWriteKind::Delete,
                    BatchWriteKind::CounterAdd(delta) => {
                        kivi_state::TxnWriteKind::CounterAdd(*delta)
                    }
                };
                let expect = match write.expect {
                    BatchExpect::Any => kivi_state::TxnExpect::Any,
                    BatchExpect::Absent => kivi_state::TxnExpect::Absent,
                    BatchExpect::Version(version) => {
                        kivi_state::TxnExpect::Version(kivi_state::ObjectVersion::from_u64(version))
                    }
                };
                kivi_state::TxnWrite {
                    key: Key::from(write.key.clone()),
                    kind,
                    expect,
                }
            })
            .collect();
        // Participants are observed prepare tablets (advisory for
        // operators); the record's decision + digest are authoritative.
        let digest = kivi_state::write_set_digest(&typed, namespace);
        TxnRecord {
            id: txn,
            coordinator,
            participants,
            state,
            dir_version: 0,
            digest,
        }
    }

    /// Discovers the coordinator (first write's tablet) with a one-entry
    /// scan probe (also warms the route cache via the page range).
    fn probe_coordinator(
        &self,
        namespace: NamespaceId,
        key: &[u8],
    ) -> Result<TabletId, BatchAttempt> {
        let mut tries: u32 = 0;
        loop {
            let page = self
                .scan_page(
                    namespace,
                    Some(key.to_vec()),
                    Some(key_successor(key)),
                    ScanDirection::Forward,
                    ScanConsistency::LatestPerTablet,
                    ScanProjection::KeysOnly,
                    1,
                    64,
                )
                .map_err(BatchAttempt::Error)?;
            if page.tablet != 0 {
                return Ok(TabletId::from_u64(page.tablet));
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
    /// not-applied without failing: convergence, not error.
    fn finalize_all(
        &self,
        namespace: NamespaceId,
        writes: &[BatchWriteSpec],
        txn: TxnId,
        commit: bool,
    ) -> Result<AtomicBatchResult, BatchAttempt> {
        let mut versions: Vec<Option<u64>> = vec![None; writes.len()];
        for (order, write) in writes.iter().enumerate() {
            match self.txn_finalize(namespace, &write.key, txn, commit) {
                Ok(FinalizeOutcome::Applied(version)) => versions[order] = version,
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
        txn: TxnId,
        coordinator: TabletId,
        prepared: &[usize],
        aborted: BatchAttempt,
    ) -> Result<AtomicBatchResult, BatchAttempt> {
        match self.abort_prepared(namespace, writes, txn, coordinator, prepared) {
            Ok(AbortOutcome::Aborted) => Err(aborted),
            Ok(AbortOutcome::Committed) => self.finalize_all(namespace, writes, txn, true),
            Err(attempt) => Err(attempt),
        }
    }

    /// Persists the Abort decision (guarded CAS, never a blind overwrite)
    /// and finalize-aborts every prepared key (best effort; the record
    /// converges anything missed). Returns which decision the coordinator
    /// record converged to: a lost race means a concurrent drive of this
    /// transaction committed, and the caller must converge via commit.
    /// An ambiguous record read propagates as `Redrive` (the state is
    /// unknown — the same-transaction re-drive resolves it recovery-first).
    fn abort_prepared(
        &self,
        namespace: NamespaceId,
        writes: &[BatchWriteSpec],
        txn: TxnId,
        coordinator: TabletId,
        prepared: &[usize],
    ) -> Result<AbortOutcome, BatchAttempt> {
        let record_key = kivi_state::txn_record_key(coordinator, txn);
        let record = Self::build_record(
            namespace,
            writes,
            txn,
            coordinator,
            Vec::new(),
            TxnState::Aborted,
        );
        if !self.decide_record(namespace, coordinator, &record_key, &record, 1) {
            // Lost the decide race: re-read (an ambiguous decide may
            // actually have committed — never finalize-abort a committed
            // transaction).
            return match self.read_txn_record(namespace, &record_key) {
                Ok(Some(record)) if record.state == TxnState::Committed => {
                    Ok(AbortOutcome::Committed)
                }
                Ok(_) => Ok(AbortOutcome::Aborted),
                Err(attempt) => Err(attempt),
            };
        }
        for index in prepared {
            let _ = self.txn_finalize(namespace, &writes[*index].key, txn, false);
        }
        Ok(AbortOutcome::Aborted)
    }

    /// Sends one prepare (OCC + intent reservation, no visible mutation).
    fn txn_prepare(
        &self,
        namespace: NamespaceId,
        write: &BatchWriteSpec,
        txn: TxnId,
        coordinator: TabletId,
    ) -> Result<TabletId, ClientError> {
        use kivi_protocol::{Opcode, ResponseBody};
        let (kind, value, delta) = match &write.kind {
            BatchWriteKind::Put(value) => (kivi_protocol::BATCH_PUT, value.clone(), 0),
            BatchWriteKind::Delete => (kivi_protocol::BATCH_DELETE, Vec::new(), 0),
            BatchWriteKind::CounterAdd(delta) => {
                (kivi_protocol::BATCH_COUNTER_ADD, Vec::new(), *delta)
            }
        };
        let (expect, expect_version) = match write.expect {
            BatchExpect::Any => (kivi_protocol::BATCH_EXPECT_ANY, 0),
            BatchExpect::Absent => (kivi_protocol::BATCH_EXPECT_ABSENT, 0),
            BatchExpect::Version(version) => (kivi_protocol::BATCH_EXPECT_VERSION, version),
        };
        let route_key = Key::from(write.key.clone());
        let value_owned = value;
        let response = self.execute_raw(
            &route_key,
            Opcode::TxnPrepare,
            || kivi_protocol::Request {
                contract: kivi_types::ReadContract::Latest,
                namespace,
                opcode: Opcode::TxnPrepare,
                hint: None,
                key: write.key.clone(),
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
                batch_txn: txn.as_bytes(),
                batch_writes: vec![kivi_protocol::BatchWrite {
                    key: write.key.clone(),
                    kind,
                    value: value_owned.clone(),
                    delta,
                    expect,
                    expect_version,
                }],
                txn_coordinator: coordinator.as_u64(),
                txn_commit: false,
            },
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
    fn txn_finalize(
        &self,
        namespace: NamespaceId,
        key: &[u8],
        txn: TxnId,
        commit: bool,
    ) -> Result<FinalizeOutcome, ClientError> {
        use kivi_protocol::{Opcode, ResponseBody};
        let route_key = Key::from(key.to_vec());
        let response = self.execute_raw(
            &route_key,
            Opcode::TxnFinalize,
            || kivi_protocol::Request {
                contract: kivi_types::ReadContract::Latest,
                namespace,
                opcode: Opcode::TxnFinalize,
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
                batch_txn: txn.as_bytes(),
                batch_writes: Vec::new(),
                txn_coordinator: 0,
                txn_commit: commit,
            },
            None,
            kivi_types::ReadContract::Latest,
        )?;
        match (response.status, response.body) {
            (kivi_protocol::Status::Ok, ResponseBody::TxnFinalized { applied, version }) => {
                if applied {
                    Ok(FinalizeOutcome::Applied((version != 0).then_some(version)))
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
    /// Intent applied (`Some` version, `None` for deletes / recovered).
    Applied(Option<u64>),
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

impl From<BatchAttempt> for ClientError {
    fn from(attempt: BatchAttempt) -> Self {
        match attempt {
            BatchAttempt::Conflict => ClientError::TxnConflict,
            BatchAttempt::Redrive => ClientError::TxnCoordinatorUnavailable,
            BatchAttempt::Error(error) => error,
        }
    }
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
    /// indexed under `term`, in primary-key order. Stale entries (primary
    /// moved or deleted since) are skipped by version validation.
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
        let prefix = term_prefix(index, term, false);
        let end = prefix_successor(&prefix);
        let options = ScanOptions {
            start: Some(prefix),
            end,
            direction: ScanDirection::Forward,
            consistency: ScanConsistency::LatestPerTablet,
            projection: ScanProjection::KeysAndValues,
            max_items_per_page: 1000,
            max_bytes_per_page: 1 << 20,
            limit: Some(limit.min(INDEX_RESOLVE_LIMIT)),
        };
        let entries = self.scan(namespace, &options)?;
        let mut hits: Vec<IndexHit> = Vec::new();
        for entry in entries {
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
    /// for every term in the window, in `(term, primary)` order. Stale
    /// entries are skipped by version validation.
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
        let start = term_prefix(index, start_term, false);
        let end = term_prefix(index, end_term, false);
        let options = ScanOptions {
            start: Some(start),
            end: Some(end),
            direction: ScanDirection::Forward,
            consistency: ScanConsistency::LatestPerTablet,
            projection: ScanProjection::KeysAndValues,
            max_items_per_page: 1000,
            max_bytes_per_page: 1 << 20,
            limit: Some(limit.min(INDEX_RESOLVE_LIMIT)),
        };
        let entries = self.scan(namespace, &options)?;
        let mut hits: Vec<IndexHit> = Vec::new();
        for entry in entries {
            // A namespace may mix both spellings per index: try the
            // non-unique form (primary in key) then the unique form
            // (primary in value).
            if let Ok((found_index, _, primary)) = decode_non_unique_key(&entry.key) {
                if found_index != index {
                    continue;
                }
                let bytes = match &entry.value {
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
            } else if let Ok((found_index, _)) = decode_unique_key(&entry.key) {
                if found_index != index {
                    continue;
                }
                let bytes = match &entry.value {
                    ClientScanValue::Inline(bytes) => bytes.to_vec(),
                    _ => continue,
                };
                let (version, primary) = decode_entry_value(&bytes).map_err(|_| {
                    ClientError::Internal("index value failed to decode".to_owned())
                })?;
                hits.push(IndexHit {
                    primary_key: primary.to_vec(),
                    version: version.as_u64(),
                });
            }
        }
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
