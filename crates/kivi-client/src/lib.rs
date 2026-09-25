//! Native TCP client library for Kivi (blocking, no async runtime).
//!
//! [`NativeClient`] speaks the native protocol over persistent per-worker
//! TCP connections with a sparse route cache: the first request to a range
//! goes to the seed, learns the authority from the redirect, and routes
//! directly afterwards. One logical request identity is preserved across
//! route retries — redirects are transport retries, never logical replays.
//!
//! Safe retry: every mutating call allocates one [`RequestIdentity`] (plus
//! the acknowledgement floor) that is preserved across redirects, redials,
//! and delivery retries. Servers advertising `DURABLE_MUTATION_DEDUP`
//! persist each outcome under that identity, so a retried mutation returns
//! its original outcome instead of executing twice. One typed call is one
//! mutation identity; caller-level retries across calls are new mutations.
//!
//! Threading model: the client is cheaply clonable (`Arc` inside). Each
//! connection has one background reader thread dispatching responses by
//! request id; writers serialize per connection under a mutex. Blocking
//! calls wait on one-shot rendezvous channels with an explicit timeout.

pub mod ordered;
pub mod route;

use core::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use core::time::Duration;
use std::collections::{BTreeSet, HashMap};
use std::io::{Read, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

use bytes::Bytes;
use crossbeam_channel::{Sender, bounded, unbounded};
use kivi_protocol::{
    Capabilities, ClientHello, DEFAULT_MAX_FRAME, Frame, FrameKind, FrameReader,
    MAX_STREAM_UPLOAD_BYTES, ProtocolError, RequestId, Response, ResponseBody, RouteHint,
    ServerHello, Status, StreamAbort, StreamBegin, encode_frame,
};
use kivi_state::{Key, PartitionHasher};
use kivi_types::{
    CommitToken, NamespaceId, ReadContract, ReadReceipt, RequestIdentity, RequestSeq, SessionId,
    WallTimestamp, WorkerId,
};

pub use ordered::{
    AtomicBatchResult, BatchExpect, BatchWriteKind, BatchWriteSpec, ClientScanEntry,
    ClientScanPage, ClientScanValue, IndexHit, IndexTermSet, ScanConsistency, ScanCursor,
    ScanDirection, ScanOptions, ScanProjection, key_successor,
};
pub use route::{RouteCache, RouteEntry};

/// Connection establishment timeout default.
pub const DEFAULT_CONNECT_TIMEOUT: Duration = Duration::from_secs(5);
/// Per-request response wait default.
pub const DEFAULT_REQUEST_TIMEOUT: Duration = Duration::from_secs(30);
/// Write timeout default (no read timeout: idle pooled connections are healthy).
pub const DEFAULT_WRITE_TIMEOUT: Duration = Duration::from_secs(30);
/// Route retry budget default (redirects before surfacing).
pub const DEFAULT_MAX_REDIRECTS: usize = 8;
/// Zero-progress redirect streak before failing over: consecutive
/// `StaleRoute` hops carrying already-seen (tablet, endpoint, generation)
/// triples back off without spending redirect budget (they wait out an
/// election/observation window); past this many repeats the call evicts
/// the endpoint, spends one redirect, and fails over. Bounds every call
/// while riding out ordinary elections (which land in ~1-5 hops).
pub const MAX_STALE_REPEATS: u32 = 12;
/// In-flight requests per connection default (fast `Overloaded` past this).
pub const DEFAULT_MAX_PENDING: usize = 128;
/// Dial attempts per connection (handshake validation failures fail fast;
/// transport failures back off and retry).
pub const DEFAULT_DIAL_ATTEMPTS: u32 = 3;
/// Base dial backoff (doubled per attempt, capped at one second).
pub const DEFAULT_DIAL_BACKOFF: Duration = Duration::from_millis(20);
/// Ambiguous-delivery retries per mutating request (same identity, only
/// against `DURABLE_MUTATION_DEDUP` servers).
pub const DEFAULT_DELIVERY_RETRIES: u32 = 2;
/// Backoff cap for every retry schedule.
pub const MAX_RETRY_BACKOFF: Duration = Duration::from_secs(1);

/// Client configuration. All bounds explicit; everything has a default.
#[derive(Debug, Clone)]
pub struct ClientConfig {
    /// Seed addresses (`"ip:port"`); the first reachable seed bootstraps.
    pub seeds: Vec<String>,
    /// Namespace every request targets.
    pub namespace: NamespaceId,
    /// Largest frame accepted (negotiated down by servers).
    pub max_frame: usize,
    /// Redirect hops before failing a request.
    pub max_redirects: usize,
    /// In-flight requests per connection before fast rejection.
    pub max_pending: usize,
    /// TCP connect timeout.
    pub connect_timeout: Duration,
    /// Response wait timeout.
    pub request_timeout: Duration,
    /// Socket write timeout.
    pub write_timeout: Duration,
    /// Dial attempts per connection establishment.
    pub dial_attempts: u32,
    /// Base backoff between dial attempts (exponential, capped).
    pub dial_backoff: Duration,
    /// Ambiguous-delivery retries per mutating request.
    pub delivery_retries: u32,
    /// Adopt an existing mutation session instead of minting a fresh
    /// random one. `None` (the default) mints; `Some` resumes: the client
    /// speaks as that session, so retries of its uncompleted sequences
    /// dedup-hit server-side instead of executing twice. Resuming is the
    /// application-level exactly-once primitive — pair it with explicit
    /// sequences (`set_with_seq`, `counter_add_with_seq`) naming the
    /// sequences to resume.
    pub session: Option<SessionId>,
}

impl Default for ClientConfig {
    /// Loopback-friendly defaults for development and tests.
    fn default() -> Self {
        Self {
            seeds: Vec::new(),
            namespace: NamespaceId::from_u64(1),
            max_frame: DEFAULT_MAX_FRAME,
            max_redirects: DEFAULT_MAX_REDIRECTS,
            max_pending: DEFAULT_MAX_PENDING,
            connect_timeout: DEFAULT_CONNECT_TIMEOUT,
            request_timeout: DEFAULT_REQUEST_TIMEOUT,
            write_timeout: DEFAULT_WRITE_TIMEOUT,
            dial_attempts: DEFAULT_DIAL_ATTEMPTS,
            dial_backoff: DEFAULT_DIAL_BACKOFF,
            delivery_retries: DEFAULT_DELIVERY_RETRIES,
            session: None,
        }
    }
}

/// Native client failure modes. Wire-shape problems surface as `Protocol`;
/// semantic rejections surface typed (mirroring `LocalClient` outcomes).
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum ClientError {
    /// Transport I/O failure (connect, read, write, handshake I/O).
    #[error("transport I/O: {0}")]
    Io(String),
    /// Handshake rejected (version, capabilities, or malformed hello).
    #[error("handshake failed: {0}")]
    Handshake(String),
    /// Wire decode failure (strict frame/message validation).
    #[error("protocol: {0}")]
    Protocol(#[from] ProtocolError),
    /// No response within the request timeout.
    #[error("request timed out")]
    Timeout,
    /// Connection saturated past `max_pending` (fast rejection).
    #[error("connection saturated")]
    Overloaded,
    /// Redirect budget exhausted without reaching an authority.
    #[error("redirect budget exhausted")]
    TooManyRedirects,
    /// Key holds a different logical type.
    #[error("wrong type")]
    WrongType,
    /// Counter addition overflows `i64`.
    #[error("counter overflow")]
    CounterOverflow,
    /// An object version cannot advance.
    #[error("version exhausted")]
    VersionExhausted,
    /// Operation not implemented by the server.
    #[error("unsupported")]
    Unsupported,
    /// Structurally invalid request (should not happen from this client).
    #[error("invalid request")]
    InvalidRequest,
    /// A resource bound was exceeded server-side.
    #[error("resource exhausted")]
    ResourceExhausted,
    /// A key or value exceeds the per-value bound.
    #[error("value too large")]
    ValueTooLarge,
    /// Unexpected server-side failure, with its diagnostic if any.
    #[error("server internal error: {0}")]
    Internal(String),
    /// A mutating request may or may not have executed: delivery was
    /// ambiguous (timeout or transport failure after possible transmission)
    /// and the server does not advertise durable dedup, so retrying could
    /// execute it twice. Returned instead of risking duplicate execution.
    #[error("ambiguous outcome: request may or may not have executed")]
    AmbiguousOutcome,
    /// A retried identity fell below the session's durable floor: its
    /// outcome is gone and it will never execute again.
    #[error("mutation identity expired below the session floor")]
    DedupExpired,
    /// The session holds too many unacknowledged outcomes on the tablet.
    #[error("session outcome window exhausted")]
    SessionOverloaded,
    /// A transactional prepare failed OCC validation or met a reservation.
    #[error("transaction conflict")]
    TxnConflict,
    /// The transaction was aborted.
    #[error("transaction aborted")]
    TxnAborted,
    /// The transaction exceeds participant/key/byte bounds.
    #[error("transaction too large")]
    TxnTooLarge,
    /// A participant cannot reach the coordinator record yet.
    #[error("transaction coordinator unavailable")]
    TxnCoordinatorUnavailable,
    /// A unique index term is already owned by another primary.
    #[error("unique index violation")]
    UniqueViolation,
    /// The batch spans tablets: drive it over `TxnPrepare`/`TxnFinalize`.
    /// The batch driver intercepts this before it surfaces; reaching user
    /// code means a direct `AtomicBatch` crossed tablets unexpectedly.
    #[error("batch spans tablets; use a transaction driver")]
    TxnCrossTablet,
    /// A bounded-counter addition would leave locally owned escrow rights.
    #[error("bounded counter would exceed owned escrow rights")]
    BoundedExceeded,
    /// A semaphore acquisition would exceed capacity.
    #[error("semaphore capacity exhausted")]
    SemaphoreExhausted,
    /// A lease acquisition met a live holder owned by someone else.
    #[error("lease held by another live owner")]
    LeaseConflict,
    /// A lease renew/release named a stale fencing token.
    #[error("stale fencing token")]
    StaleFencing,
    /// A stream append would exceed the shard's bounds.
    #[error("stream shard full")]
    StreamFull,
    /// Conflicting final decisions for one transaction (durable corruption).
    #[error("transaction decision conflict")]
    TxnDecisionConflict,
    /// A scan cursor no longer resolves; re-issue the scan.
    #[error("scan cursor stale")]
    ScanCursorStale,
    /// An `AtLeast` token's lineage is foreign to the serving replica.
    /// Refresh the token from a read on its own lineage instead of
    /// retrying the same token.
    #[error("stale read token")]
    StaleToken,
}

/// Maps a response status onto a client error (Ok/NotFound handled by callers).
fn status_error(status: Status, body: &ResponseBody) -> ClientError {
    let diagnostic = match body {
        ResponseBody::Diagnostic(message) if !message.is_empty() => message.clone(),
        _ => status.to_string(),
    };
    match status {
        Status::Ok | Status::NotFound | Status::StaleRoute | Status::NotLocal => {
            ClientError::Internal(format!("unexpected {status} here: {diagnostic}"))
        }
        Status::InvalidRequest => ClientError::InvalidRequest,
        Status::Unsupported => ClientError::Unsupported,
        Status::WrongType => ClientError::WrongType,
        Status::CounterOverflow => ClientError::CounterOverflow,
        Status::VersionExhausted => ClientError::VersionExhausted,
        Status::Overloaded => ClientError::Overloaded,
        Status::ResourceExhausted => ClientError::ResourceExhausted,
        Status::ValueTooLarge => ClientError::ValueTooLarge,
        Status::Internal => ClientError::Internal(diagnostic),
        Status::DedupExpired => ClientError::DedupExpired,
        Status::SessionOverloaded => ClientError::SessionOverloaded,
        Status::TxnConflict => ClientError::TxnConflict,
        Status::TxnAborted => ClientError::TxnAborted,
        Status::TxnTooLarge => ClientError::TxnTooLarge,
        Status::TxnCoordinatorUnavailable => ClientError::TxnCoordinatorUnavailable,
        Status::UniqueViolation => ClientError::UniqueViolation,
        Status::TxnCrossTablet => ClientError::TxnCrossTablet,
        Status::BoundedExceeded => ClientError::BoundedExceeded,
        Status::SemaphoreExhausted => ClientError::SemaphoreExhausted,
        Status::LeaseConflict => ClientError::LeaseConflict,
        Status::StaleFencing => ClientError::StaleFencing,
        Status::StreamFull => ClientError::StreamFull,
        Status::TxnDecisionConflict => ClientError::TxnDecisionConflict,
        Status::ScanCursorStale => ClientError::ScanCursorStale,
        Status::StaleToken => ClientError::StaleToken,
        Status::CoverageUncertain => ClientError::AmbiguousOutcome,
    }
}

/// Encodes a conditional store's condition/policy onto wire bytes plus the
/// `expiry` stamp (nonzero only for `ExpireAt`).
fn encode_conditional(
    condition: kivi_state::SetCondition,
    expiry: kivi_state::ExpiryPolicy,
) -> (u8, u8, i64) {
    use kivi_protocol::{COND_ALWAYS, COND_IF_ABSENT, COND_IF_PRESENT};
    use kivi_protocol::{EXPIRY_AT, EXPIRY_CLEAR, EXPIRY_KEEP};
    let condition_wire = match condition {
        kivi_state::SetCondition::Always => COND_ALWAYS,
        kivi_state::SetCondition::IfAbsent => COND_IF_ABSENT,
        kivi_state::SetCondition::IfPresent => COND_IF_PRESENT,
    };
    let (policy_wire, stamp) = match expiry {
        kivi_state::ExpiryPolicy::Clear => (EXPIRY_CLEAR, 0),
        kivi_state::ExpiryPolicy::Keep => (EXPIRY_KEEP, 0),
        kivi_state::ExpiryPolicy::ExpireAt(stamp) => (EXPIRY_AT, stamp.as_micros()),
    };
    (condition_wire, policy_wire, stamp)
}

/// Sleeps the capped exponential backoff for retry `attempt` (from zero).
fn backoff(base: Duration, attempt: u32) {
    let shifted = base.checked_mul(1 << attempt.min(10)).unwrap_or(base);
    std::thread::sleep(shifted.min(MAX_RETRY_BACKOFF));
}

/// One stream event arriving on the background reader: upload control
/// (`Ready`, the final `Response`, `Abort`) or download frames
/// (`ValueBegin`, `ValueData`, `ValueEnd`).
#[derive(Debug)]
enum StreamEvent {
    /// Server accepted the upload; client must cap data frames here.
    Ready(u32),
    /// Upload committed: the ordinary stored response under the stream id.
    Committed(Response),
    /// Either side aborted: the stream is over, nothing was committed.
    Aborted(String),
    /// A streamed read starts; this many data bytes follow.
    ValueBegin(u64),
    /// Verbatim value bytes (already capped per frame by the server).
    ValueData(Vec<u8>),
    /// The streamed read completed exactly (byte count already checked).
    ValueEnd,
}

/// One pooled connection: a mutex-serialized writer plus a background reader
/// dispatching responses to per-request rendezvous channels by id, and
/// stream frames to per-stream rendezvous by stream id.
#[derive(Debug)]
struct Connection {
    worker: WorkerId,
    endpoint: String,
    server_caps: Capabilities,
    writer: Mutex<TcpStream>,
    pending: Arc<Mutex<HashMap<u64, Sender<Response>>>>,
    streams: Arc<Mutex<HashMap<u64, crossbeam_channel::Sender<StreamEvent>>>>,
    max_pending: usize,
    request_timeout: Duration,
    reader_alive: Arc<AtomicBool>,
    _reader: JoinHandle<()>,
}

impl Connection {
    /// Whether this connection's server persists mutation outcomes under
    /// client identity (safe to retry ambiguous mutating requests on).
    fn durable_dedup(&self) -> bool {
        self.server_caps
            .contains(Capabilities::DURABLE_MUTATION_DEDUP)
    }

    /// Whether this connection's server speaks chunk-fabric streaming
    /// (upload and streamed-read frames). Streaming calls check this
    /// before sending anything, so a legacy server fails fast with
    /// `Unsupported` instead of a violation.
    fn streaming(&self) -> bool {
        self.server_caps.contains(Capabilities::STREAMING)
    }

    /// Dials, handshakes, and spawns the reader thread. Validates the hello
    /// (worker identity when expected, base capability present). Transport
    /// (`Io`) failures retry with capped backoff; handshake validation
    /// failures are deterministic and fail fast.
    fn dial(
        endpoint: &str,
        expect_worker: Option<WorkerId>,
        config: &ClientConfig,
    ) -> Result<Self, ClientError> {
        let attempts = config.dial_attempts.max(1);
        let mut last = ClientError::Io(format!("unresolvable endpoint {endpoint}"));
        for attempt in 0..attempts {
            match Self::dial_once(endpoint, expect_worker, config) {
                Ok(conn) => return Ok(conn),
                Err(ClientError::Io(detail)) => {
                    last = ClientError::Io(detail);
                    if attempt + 1 < attempts {
                        backoff(config.dial_backoff, attempt);
                    }
                }
                Err(error) => return Err(error),
            }
        }
        Err(last)
    }

    /// One dial plus handshake attempt, no retry.
    fn dial_once(
        endpoint: &str,
        expect_worker: Option<WorkerId>,
        config: &ClientConfig,
    ) -> Result<Self, ClientError> {
        let addr = endpoint
            .to_socket_addrs()
            .map_err(|error| ClientError::Io(error.to_string()))?
            .next()
            .ok_or_else(|| ClientError::Io(format!("unresolvable endpoint {endpoint}")))?;
        let stream = TcpStream::connect_timeout(&addr, config.connect_timeout)
            .map_err(|error| ClientError::Io(error.to_string()))?;
        stream
            .set_nodelay(true)
            .map_err(|error| ClientError::Io(error.to_string()))?;
        stream
            .set_write_timeout(Some(config.write_timeout))
            .map_err(|error| ClientError::Io(error.to_string()))?;
        // No read timeout: pooled connections idle healthily for long stretches.
        let mut reader_stream = stream
            .try_clone()
            .map_err(|error| ClientError::Io(error.to_string()))?;
        let hello = ClientHello {
            max_frame: u32::try_from(config.max_frame).unwrap_or(u32::MAX),
            required_caps: Capabilities::BASE_V1,
            optional_caps: Capabilities::empty(),
            desired_worker: expect_worker,
        };
        let bytes = encode_frame(FrameKind::ClientHello, 0, &hello.encode());
        stream
            .try_clone()
            .map_err(|error| ClientError::Io(error.to_string()))?
            .write_all(&bytes)
            .map_err(|error| ClientError::Io(error.to_string()))?;
        let reply = read_one_frame(&mut reader_stream, config.max_frame)
            .map_err(|error| ClientError::Handshake(error.to_string()))?;
        if reply.kind != FrameKind::ServerHello {
            return Err(ClientError::Handshake(format!(
                "expected server-hello, got {}",
                reply.kind
            )));
        }
        let hello = ServerHello::decode(&reply.payload)
            .map_err(|error| ClientError::Handshake(error.to_string()))?;
        if !hello.caps.contains(Capabilities::BASE_V1) {
            return Err(ClientError::Handshake(
                "server lacks base capability".to_owned(),
            ));
        }
        if let Some(expect) = expect_worker
            && hello.worker != expect
        {
            return Err(ClientError::Handshake(format!(
                "dialed worker {} but reached {}",
                expect.as_u64(),
                hello.worker.as_u64()
            )));
        }
        let pending: Arc<Mutex<HashMap<u64, Sender<Response>>>> =
            Arc::new(Mutex::new(HashMap::new()));
        let dispatch = Arc::clone(&pending);
        // Stream ids live in the high-bit space (request ids ascend from
        // 1), so the two tables never share an id and the reader routes
        // response frames by table membership without ambiguity.
        let streams: Arc<Mutex<HashMap<u64, crossbeam_channel::Sender<StreamEvent>>>> =
            Arc::new(Mutex::new(HashMap::new()));
        let stream_dispatch = Arc::clone(&streams);
        let reader_alive = Arc::new(AtomicBool::new(true));
        let alive = Arc::clone(&reader_alive);
        let max_frame = config.max_frame.min(hello.max_frame as usize);
        let reader = std::thread::Builder::new()
            .name(format!("kivi-client-reader-{endpoint}"))
            .spawn(move || {
                reader_loop(reader_stream, max_frame, dispatch, stream_dispatch, alive);
            })
            .map_err(|error| ClientError::Io(error.to_string()))?;
        Ok(Self {
            worker: hello.worker,
            endpoint: endpoint.to_owned(),
            server_caps: hello.caps,
            writer: Mutex::new(stream),
            pending,
            streams,
            max_pending: config.max_pending,
            request_timeout: config.request_timeout,
            reader_alive,
            _reader: reader,
        })
    }

    /// Sends one request frame and waits for its response (matched by id). A
    /// wait that outlives its reader thread surfaces as `Io` promptly
    /// (connection lost, never a slow server): the reader drains the
    /// rendezvous table on exit, so the stranded wait wakes at once instead
    /// of burning the full request timeout. The reader only exits on EOF or
    /// fatal framing failure, after which no response can arrive.
    fn round_trip(&self, id: RequestId, payload: &[u8]) -> Result<Response, ClientError> {
        let (tx, rx) = bounded::<Response>(1);
        {
            let mut pending = self.pending.lock().map_err(|_| {
                ClientError::Io(format!("connection {} lock poisoned", self.endpoint))
            })?;
            if pending.len() >= self.max_pending {
                return Err(ClientError::Overloaded);
            }
            pending.insert(id.as_u64(), tx);
        }
        let bytes = encode_frame(FrameKind::Request, id.as_u64(), payload);
        let write_result = self
            .writer
            .lock()
            .map_err(|_| ClientError::Io(format!("connection {} lock poisoned", self.endpoint)))
            .and_then(|mut stream| {
                stream.write_all(&bytes).map_err(|error| {
                    self.drop_pending(id.as_u64());
                    ClientError::Io(error.to_string())
                })
            });
        write_result?;
        if let Ok(response) = rx.recv_timeout(self.request_timeout) {
            Ok(response)
        } else {
            self.drop_pending(id.as_u64());
            if self.reader_alive.load(Ordering::SeqCst) {
                Err(ClientError::Timeout)
            } else {
                Err(ClientError::Io(format!(
                    "connection {} reader ended mid-request",
                    self.endpoint
                )))
            }
        }
    }

    /// Forgets one rendezvous (after write/timeout failures).
    fn drop_pending(&self, id: u64) {
        if let Ok(mut pending) = self.pending.lock() {
            pending.remove(&id);
        }
    }

    /// Writes one frame on a connection. Each frame holds the writer
    /// mutex only for its own write: stream frames multiplex with
    /// ordinary requests by id, so uploads never pin the connection.
    fn send_frame(
        conn: &Arc<Connection>,
        kind: FrameKind,
        id: u64,
        payload: &[u8],
    ) -> Result<(), ClientError> {
        let bytes = encode_frame(kind, id, payload);
        conn.writer
            .lock()
            .map_err(|_| ClientError::Io(format!("connection {} lock poisoned", conn.endpoint)))
            .and_then(|mut stream| {
                use std::io::Write as _;
                stream
                    .write_all(&bytes)
                    .map_err(|error| ClientError::Io(error.to_string()))
            })
    }

    /// Registers one stream rendezvous, bounding open streams like
    /// in-flight requests (every registration pins server-side state).
    fn register_stream(
        conn: &Arc<Connection>,
        id: u64,
    ) -> Result<crossbeam_channel::Receiver<StreamEvent>, ClientError> {
        let (tx, rx) = unbounded();
        conn.streams
            .lock()
            .map_err(|_| ClientError::Io(format!("connection {} lock poisoned", conn.endpoint)))
            .and_then(|mut table| {
                if table.len() >= conn.max_pending {
                    return Err(ClientError::Overloaded);
                }
                table.insert(id, tx);
                Ok(rx)
            })
    }

    /// Forgets one stream rendezvous (after completion, timeout, or abort).
    fn unregister_stream(conn: &Arc<Connection>, id: u64) {
        if let Ok(mut table) = conn.streams.lock() {
            table.remove(&id);
        }
    }

    /// Best-effort stream abort (timeout hygiene: frees server-side
    /// assembler state). Delivery failure is ignored — the connection
    /// itself is already suspect when this runs.
    fn abort_stream(conn: &Arc<Connection>, id: u64, reason: &str) {
        let _ = Connection::send_frame(
            conn,
            FrameKind::StreamAbort,
            id,
            &StreamAbort {
                reason: reason.to_owned(),
            }
            .encode(),
        );
    }

    /// Waits for one stream event. Timeouts and dead readers surface like
    /// `round_trip` (`Timeout` vs `Io` via the reader liveness check), so
    /// stalled streams never hang past the request timeout.
    fn await_stream_event(
        conn: &Arc<Connection>,
        rx: &crossbeam_channel::Receiver<StreamEvent>,
    ) -> Result<StreamEvent, ClientError> {
        match rx.recv_timeout(conn.request_timeout) {
            Ok(event) => Ok(event),
            Err(crossbeam_channel::RecvTimeoutError::Timeout) => {
                if conn.reader_alive.load(Ordering::SeqCst) {
                    Err(ClientError::Timeout)
                } else {
                    Err(ClientError::Io(format!(
                        "connection {} reader ended mid-stream",
                        conn.endpoint
                    )))
                }
            }
            Err(crossbeam_channel::RecvTimeoutError::Disconnected) => {
                if conn.reader_alive.load(Ordering::SeqCst) {
                    Err(ClientError::Internal("stream rendezvous lost".to_owned()))
                } else {
                    Err(ClientError::Io(format!(
                        "connection {} reader ended mid-stream",
                        conn.endpoint
                    )))
                }
            }
        }
    }
}

/// Background reader: frames → responses → per-id rendezvous, plus stream
/// frames → per-stream rendezvous. Ends on EOF, timeout-free. On exit both
/// tables drain: dropping every waiter wakes blocked calls at once with a
/// channel disconnect (surfaced as `Io` by the `reader_alive` check),
/// instead of stranding each one until its full timeout. Malformed frames
/// end the connection the same way. Takes the tables by value: the reader
/// thread owns its dispatch handles. Clears `alive` on exit so waits can
/// distinguish a dead connection (redial) from a slow server (timeout).
#[allow(clippy::needless_pass_by_value)]
fn reader_loop(
    mut stream: TcpStream,
    max_frame: usize,
    pending: Arc<Mutex<HashMap<u64, Sender<Response>>>>,
    streams: Arc<Mutex<HashMap<u64, crossbeam_channel::Sender<StreamEvent>>>>,
    alive: Arc<AtomicBool>,
) {
    let mut reader = FrameReader::new(max_frame);
    let mut chunk = vec![0u8; 8192];
    loop {
        let count = match stream.read(&mut chunk) {
            Ok(0) | Err(_) => break,
            Ok(count) => count,
        };
        let Ok(frames) = reader.push(&chunk[..count]) else {
            break;
        };
        for frame in frames {
            dispatch_frame(&frame, &pending, &streams);
        }
    }
    alive.store(false, Ordering::SeqCst);
    // Fail-fast stranded waiters: dropping every rendezvous sender wakes
    // each blocked `round_trip` at once (its `recv` errors immediately and
    // the `reader_alive` check below reports `Io`, i.e. redial). Without
    // this, every in-flight request on a dead connection burns its full
    // request timeout before noticing — a ~30s stall per dead connection
    // under load, exactly the tail this fixes. Off the hot path: runs once
    // per connection death.
    if let Ok(mut table) = pending.lock() {
        let stranded = table.len();
        if stranded > 0 {
            tracing::warn!(
                stranded,
                "client connection reader exited with in-flight requests; failing them fast"
            );
        }
        table.clear();
    }
    // Stream waiters strand the same way: dropping their senders wakes
    // every in-flight upload and download at once.
    if let Ok(mut table) = streams.lock() {
        if !table.is_empty() {
            tracing::warn!(
                stranded = table.len(),
                "client connection reader exited with in-flight streams; failing them fast"
            );
        }
        table.clear();
    }
}

/// Routes one frame to its rendezvous. Undecodable frames are skipped
/// (the waiter stays and times out, exactly like before streaming):
/// framing violations already end the connection inside `reader.push`,
/// so nothing here re-litigates the connection over one bad payload.
/// Per-stream control decode failures degrade to that stream's abort
/// instead: one garbled control frame must not kill healthy multiplexed
/// streams.
fn dispatch_frame(
    frame: &Frame,
    pending: &Arc<Mutex<HashMap<u64, Sender<Response>>>>,
    streams: &Arc<Mutex<HashMap<u64, crossbeam_channel::Sender<StreamEvent>>>>,
) {
    use kivi_protocol::ValueStreamBegin;
    match frame.kind {
        FrameKind::Response => {
            // A committed upload's final response arrives under its
            // stream id — check the request table first (hot path),
            // then the stream table (id spaces never overlap).
            let Ok((response, _)) = Response::decode(&frame.payload) else {
                return;
            };
            if let Some(tx) = pending
                .lock()
                .ok()
                .and_then(|mut table| table.remove(&frame.request_id))
            {
                let _ = tx.try_send(response);
                return;
            }
            if let Some(tx) = streams
                .lock()
                .ok()
                .and_then(|mut table| table.remove(&frame.request_id))
            {
                let _ = tx.send(StreamEvent::Committed(response));
            }
        }
        FrameKind::StreamReady => {
            if let Some(tx) = streams
                .lock()
                .ok()
                .and_then(|table| table.get(&frame.request_id).cloned())
            {
                match kivi_protocol::StreamReady::decode(&frame.payload) {
                    Ok(ready) => {
                        let _ = tx.send(StreamEvent::Ready(ready.max_data));
                    }
                    Err(_) => {
                        let _ = tx.send(StreamEvent::Aborted("malformed stream-ready".to_owned()));
                    }
                }
            }
        }
        FrameKind::StreamAbort => {
            if let Some(tx) = streams
                .lock()
                .ok()
                .and_then(|mut table| table.remove(&frame.request_id))
            {
                let reason = StreamAbort::decode(&frame.payload)
                    .map_or_else(|_| "stream aborted".to_owned(), |abort| abort.reason);
                let _ = tx.send(StreamEvent::Aborted(reason));
            }
        }
        FrameKind::ValueStreamBegin => {
            if let Some(tx) = streams
                .lock()
                .ok()
                .and_then(|table| table.get(&frame.request_id).cloned())
            {
                match ValueStreamBegin::decode(&frame.payload) {
                    Ok(begin) => {
                        let _ = tx.send(StreamEvent::ValueBegin(begin.total_len));
                    }
                    Err(_) => {
                        let _ = tx.send(StreamEvent::Aborted(
                            "malformed value-stream-begin".to_owned(),
                        ));
                    }
                }
            }
        }
        FrameKind::ValueStreamData => {
            if let Some(tx) = streams
                .lock()
                .ok()
                .and_then(|table| table.get(&frame.request_id).cloned())
            {
                let _ = tx.send(StreamEvent::ValueData(frame.payload.to_vec()));
            }
        }
        FrameKind::ValueStreamEnd => {
            if let Some(tx) = streams
                .lock()
                .ok()
                .and_then(|mut table| table.remove(&frame.request_id))
            {
                let _ = tx.send(StreamEvent::ValueEnd);
            }
        }
        // The server never sends these on a client connection; anything
        // else arriving here is a peer bug. Ignoring (like before) keeps
        // one stray frame from killing healthy requests — framing
        // violations still end the connection in `reader.push`.
        _ => {}
    }
}

/// Reads exactly one frame (handshake path, no pipelining yet).
fn read_one_frame(stream: &mut TcpStream, max_frame: usize) -> Result<Frame, ProtocolError> {
    let mut reader = FrameReader::new(max_frame);
    let mut chunk = vec![0u8; 4096];
    loop {
        let count = stream
            .read(&mut chunk)
            .map_err(|_| ProtocolError::Truncated)?;
        if count == 0 {
            return Err(ProtocolError::Truncated);
        }
        let frames = reader.push(&chunk[..count])?;
        if let Some(frame) = frames.into_iter().next() {
            return Ok(frame);
        }
    }
}

/// One download attempt's outcome: finished, retry on a new route,
/// back off and retry, redial and retry, or surface.
enum GetOutcome {
    /// Terminal: the value (`None` when absent).
    Done(Option<Bytes>),
    /// The route moved: re-resolve and re-request (reads are side-effect free).
    /// Carries the redirect triple when the server named one, so the loop
    /// can tell fresh information from a stale repeat (same patience rules
    /// as `execute_with_seq`; `None` spends budget like before).
    Reroute(Option<RouteEntry>),
    /// The server is saturated: back off and re-request.
    Overloaded,
    /// The transport died: drop the connection, redial, re-request.
    Reconnect,
    /// Terminal failure.
    Fail(ClientError),
}

/// Mutation sequence state for one client session. Sequences start at 1
/// (0 is never issued); the acknowledgement floor only advances over
/// consecutively completed sequences, so an in-flight sequence is never
/// acknowledged beneath itself — even across tablets sharing the session.
#[derive(Debug, Default)]
struct SeqState {
    /// Next sequence to issue.
    next: u64,
    /// Highest consecutively completed sequence (0 = nothing acked).
    acked: u64,
    /// Issued but not yet completed sequences.
    in_flight: BTreeSet<u64>,
}

/// Shared client state behind every clone.
#[derive(Debug)]
struct Shared {
    namespace: NamespaceId,
    max_frame: u32,
    max_redirects: usize,
    max_pending: usize,
    connect_timeout: Duration,
    request_timeout: Duration,
    write_timeout: Duration,
    dial_attempts: u32,
    dial_backoff: Duration,
    delivery_retries: u32,
    seeds: Vec<String>,
    routes: route::RouteCache,
    pool: Mutex<HashMap<String, Arc<Connection>>>,
    id_counter: AtomicU64,
    stream_counter: AtomicU64,
    session: SessionId,
    seq: Mutex<SeqState>,
    requests: AtomicU64,
    redirects: AtomicU64,
    errors: AtomicU64,
    /// Transactions started (per `atomic_batch` call).
    txns_started: AtomicU64,
    /// Transactions committed (fast path and 2PC).
    txns_committed: AtomicU64,
    /// Transactions aborted on OCC conflict (retryable by the caller).
    txns_aborted_conflict: AtomicU64,
    /// Transactions aborted for any other reason.
    txns_aborted_other: AtomicU64,
    /// Commits via the single-tablet fast path.
    same_tablet_txns: AtomicU64,
    /// Commits via cross-tablet 2PC.
    cross_tablet_txns: AtomicU64,
    /// Prepare-wave latencies in micros (2PC only; lock held briefly on
    /// transaction completion, never on the point-op path).
    prepare_latency: Mutex<hdrhistogram::Histogram<u64>>,
    /// Decision-write latencies in micros (2PC only).
    decision_latency: Mutex<hdrhistogram::Histogram<u64>>,
}

/// Observable client counters (all cumulative across clones).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ClientStats {
    /// Logical requests issued (before routing retries multiply them).
    pub requests: u64,
    /// Route redirects followed (topology + overload retries).
    pub redirects: u64,
    /// Terminal failures surfaced to callers.
    pub errors: u64,
    /// Transactions started (per `atomic_batch` call).
    pub txns_started: u64,
    /// Transactions committed (fast path and 2PC).
    pub txns_committed: u64,
    /// Transactions aborted on OCC conflict.
    pub txns_aborted_conflict: u64,
    /// Transactions aborted for any other reason.
    pub txns_aborted_other: u64,
    /// Commits via the single-tablet fast path.
    pub same_tablet_txns: u64,
    /// Commits via cross-tablet 2PC.
    pub cross_tablet_txns: u64,
}

/// Transaction latency snapshot (2PC paths only; micros).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TxnLatencySnapshot {
    /// Prepare-wave p50 / p99 (`None` before the first sampled transaction).
    pub prepare_p50: Option<u64>,
    /// Prepare-wave p99.
    pub prepare_p99: Option<u64>,
    /// Decision-write p50 / p99.
    pub decision_p50: Option<u64>,
    /// Decision-write p99.
    pub decision_p99: Option<u64>,
    /// Sampled transactions behind this snapshot.
    pub samples: u64,
}

/// Native Kivi client: typed operations over sparse-routed TCP connections.
/// Cheaply clonable and safe to share across application threads.
#[derive(Debug, Clone)]
pub struct NativeClient {
    shared: Arc<Shared>,
}

impl NativeClient {
    /// Builds a client from configuration (no connections yet — dialing is
    /// lazy on first use, so construction never fails on network state).
    /// One client is one mutation session: every clone shares the session
    /// identity and its sequence space. Set `config.session` to resume a
    /// session this process (or a previous one) owned before.
    ///
    /// # Errors
    ///
    /// Returns [`ClientError::Io`] when no seed address is configured or the
    /// OS random source is unavailable for the session identity.
    ///
    /// # Panics
    ///
    /// Panics when the latency histograms fail to build (effectively
    /// unreachable: the significant-figures parameter is a constant).
    pub fn new(config: ClientConfig) -> Result<Self, ClientError> {
        if config.seeds.is_empty() {
            return Err(ClientError::Io("no seed addresses configured".to_owned()));
        }
        let session = if let Some(session) = config.session {
            session
        } else {
            let mut random = [0u8; 16];
            getrandom::fill(&mut random)
                .map_err(|error| ClientError::Io(format!("session identity: {error}")))?;
            SessionId::from_u128(u128::from_le_bytes(random))
        };
        Ok(Self {
            shared: Arc::new(Shared {
                namespace: config.namespace,
                max_frame: u32::try_from(config.max_frame).unwrap_or(u32::MAX),
                max_redirects: config.max_redirects,
                max_pending: config.max_pending,
                connect_timeout: config.connect_timeout,
                request_timeout: config.request_timeout,
                write_timeout: config.write_timeout,
                dial_attempts: config.dial_attempts,
                dial_backoff: config.dial_backoff,
                delivery_retries: config.delivery_retries,
                seeds: config.seeds,
                routes: route::RouteCache::new(),
                pool: Mutex::new(HashMap::new()),
                id_counter: AtomicU64::new(1),
                stream_counter: AtomicU64::new(1),
                session,
                seq: Mutex::new(SeqState {
                    next: 1,
                    acked: 0,
                    in_flight: BTreeSet::new(),
                }),
                requests: AtomicU64::new(0),
                redirects: AtomicU64::new(0),
                errors: AtomicU64::new(0),
                txns_started: AtomicU64::new(0),
                txns_committed: AtomicU64::new(0),
                txns_aborted_conflict: AtomicU64::new(0),
                txns_aborted_other: AtomicU64::new(0),
                same_tablet_txns: AtomicU64::new(0),
                cross_tablet_txns: AtomicU64::new(0),
                prepare_latency: Mutex::new(
                    hdrhistogram::Histogram::new(3).expect("latency histogram builds"),
                ),
                decision_latency: Mutex::new(
                    hdrhistogram::Histogram::new(3).expect("latency histogram builds"),
                ),
            }),
        })
    }

    /// This client's mutation session: the identity every mutation it
    /// sends (or resumes) speaks as. Record it to resume the session
    /// after a restart with `ClientConfig::session`.
    #[must_use]
    pub fn session(&self) -> SessionId {
        self.shared.session
    }

    /// Returns cumulative client counters.
    #[must_use]
    pub fn stats(&self) -> ClientStats {
        ClientStats {
            requests: self.shared.requests.load(Ordering::Relaxed),
            redirects: self.shared.redirects.load(Ordering::Relaxed),
            errors: self.shared.errors.load(Ordering::Relaxed),
            txns_started: self.shared.txns_started.load(Ordering::Relaxed),
            txns_committed: self.shared.txns_committed.load(Ordering::Relaxed),
            txns_aborted_conflict: self.shared.txns_aborted_conflict.load(Ordering::Relaxed),
            txns_aborted_other: self.shared.txns_aborted_other.load(Ordering::Relaxed),
            same_tablet_txns: self.shared.same_tablet_txns.load(Ordering::Relaxed),
            cross_tablet_txns: self.shared.cross_tablet_txns.load(Ordering::Relaxed),
        }
    }

    /// Snapshots 2PC transaction latencies (prepare wave, decision write).
    /// Point operations and fast-path commits never touch these
    /// histograms; samples count completed 2PC phases only.
    #[must_use]
    pub fn txn_latencies(&self) -> TxnLatencySnapshot {
        fn quantile(histogram: &hdrhistogram::Histogram<u64>, quantile: f64) -> Option<u64> {
            (!histogram.is_empty()).then(|| histogram.value_at_quantile(quantile))
        }
        let prepare = self.shared.prepare_latency.lock().ok();
        let decision = self.shared.decision_latency.lock().ok();
        TxnLatencySnapshot {
            prepare_p50: prepare
                .as_deref()
                .and_then(|histogram| quantile(histogram, 0.5)),
            prepare_p99: prepare
                .as_deref()
                .and_then(|histogram| quantile(histogram, 0.99)),
            decision_p50: decision
                .as_deref()
                .and_then(|histogram| quantile(histogram, 0.5)),
            decision_p99: decision
                .as_deref()
                .and_then(|histogram| quantile(histogram, 0.99)),
            samples: prepare.map_or(0, |histogram| histogram.len()),
        }
    }

    /// Records a started transaction attempt (2PC driver paths call this
    /// once per `atomic_batch` call).
    pub(crate) fn note_txn_started(&self) {
        self.shared.txns_started.fetch_add(1, Ordering::Relaxed);
    }

    /// Records a committed transaction and its path (fast or 2PC).
    pub(crate) fn note_txn_committed(&self, cross_tablet: bool) {
        self.shared.txns_committed.fetch_add(1, Ordering::Relaxed);
        if cross_tablet {
            self.shared
                .cross_tablet_txns
                .fetch_add(1, Ordering::Relaxed);
        } else {
            self.shared.same_tablet_txns.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Records an aborted transaction by cause family.
    pub(crate) fn note_txn_aborted(&self, conflict: bool) {
        if conflict {
            self.shared
                .txns_aborted_conflict
                .fetch_add(1, Ordering::Relaxed);
        } else {
            self.shared
                .txns_aborted_other
                .fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Records one 2PC prepare-wave latency sample (micros).
    pub(crate) fn note_prepare_latency(&self, micros: u64) {
        if let Ok(mut histogram) = self.shared.prepare_latency.lock() {
            let _ = histogram.record(micros);
        }
    }

    /// Records one 2PC decision-write latency sample (micros).
    pub(crate) fn note_decision_latency(&self, micros: u64) {
        if let Ok(mut histogram) = self.shared.decision_latency.lock() {
            let _ = histogram.record(micros);
        }
    }

    /// Returns the number of cached sparse routes (introspection for tests).
    #[must_use]
    pub fn cached_routes(&self) -> usize {
        self.shared.routes.len()
    }

    /// Allocates the next stream id: the high-bit space request ids never
    /// reach (they ascend from 1), so stream and request rendezvous never
    /// share an id and the reader routes by table membership.
    fn next_stream_id(&self) -> u64 {
        let low = self.shared.stream_counter.fetch_add(1, Ordering::SeqCst);
        (1u64 << 63) | low.max(1)
    }

    /// Allocates the next request identity (starts at 1, skips 0 on the
    /// practically unreachable wrap; outstanding sets stay far below 2^64).
    fn next_id(&self) -> RequestId {
        let id = self.shared.id_counter.fetch_add(1, Ordering::SeqCst);
        if id == 0 {
            RequestId::from_u64(self.shared.id_counter.fetch_add(1, Ordering::SeqCst).max(1))
        } else {
            RequestId::from_u64(id)
        }
    }

    /// Allocates one mutation identity plus the current acknowledgement
    /// floor. The identity names this logical mutation across every
    /// transport retry; the floor lets the server drop outcomes this
    /// session already acknowledged.
    ///
    /// # Errors
    ///
    /// Returns [`ClientError::Internal`] when the sequence space is
    /// exhausted (practically unreachable) or the state lock is poisoned.
    fn alloc_seq(&self) -> Result<(RequestIdentity, RequestSeq), ClientError> {
        let mut state = self
            .shared
            .seq
            .lock()
            .map_err(|_| ClientError::Internal("mutation sequence lock poisoned".to_owned()))?;
        if state.next == u64::MAX {
            return Err(ClientError::Internal(
                "request sequence exhausted".to_owned(),
            ));
        }
        let seq = state.next;
        state.next += 1;
        state.in_flight.insert(seq);
        Ok((
            RequestIdentity::new(self.shared.session, RequestSeq::from_u64(seq)),
            RequestSeq::from_u64(state.acked),
        ))
    }

    /// Marks one explicit sequence issued under this session: same
    /// bookkeeping as allocation, but the caller names the sequence (a
    /// resumed identity, never a fresh one). The sequence space advances
    /// past it so later automatic allocation cannot collide; skipped
    /// sequences complete past silently, so resuming out of order is safe
    /// but reusing a completed sequence fails closed server-side
    /// (`DedupExpired`) once the floor passes it.
    ///
    /// # Errors
    ///
    /// Returns [`ClientError::Internal`] when `seq` is 0 or exhausted, or
    /// the state lock is poisoned.
    fn adopt_seq(&self, seq: RequestSeq) -> Result<(RequestIdentity, RequestSeq), ClientError> {
        let mut state = self
            .shared
            .seq
            .lock()
            .map_err(|_| ClientError::Internal("mutation sequence lock poisoned".to_owned()))?;
        let raw = seq.as_u64();
        if raw == 0 || raw == u64::MAX {
            return Err(ClientError::Internal(
                "explicit request sequence out of range".to_owned(),
            ));
        }
        state.in_flight.insert(raw);
        if raw >= state.next {
            state.next = raw + 1;
        }
        Ok((
            RequestIdentity::new(self.shared.session, seq),
            RequestSeq::from_u64(state.acked),
        ))
    }

    /// Marks one mutation complete (success, terminal failure, or
    /// abandonment), advancing the acknowledgement floor over consecutively
    /// completed sequences. Abandoned sequences complete too: the floor may
    /// pass them because their identity will never be retried under this
    /// call again, so no live retry can fall beneath it.
    fn complete_seq(&self, seq: u64) {
        if let Ok(mut state) = self.shared.seq.lock() {
            state.in_flight.remove(&seq);
            while let Some(candidate) = state.acked.checked_add(1) {
                if candidate >= state.next || state.in_flight.contains(&candidate) {
                    break;
                }
                state.acked = candidate;
            }
        }
    }

    /// Returns the pooled connection for `endpoint` (dialing + handshake on
    /// miss), validating the expected worker when known.
    fn connection(
        &self,
        endpoint: &str,
        expect_worker: Option<WorkerId>,
    ) -> Result<Arc<Connection>, ClientError> {
        if let Some(conn) = self
            .shared
            .pool
            .lock()
            .map_err(|_| ClientError::Io("connection pool lock poisoned".to_owned()))?
            .get(endpoint)
            .cloned()
            && expect_worker.is_none_or(|want| conn.worker == want)
            && conn.reader_alive.load(Ordering::SeqCst)
        {
            return Ok(conn);
        }
        self.drop_connection(endpoint);
        let config = ClientConfig {
            seeds: Vec::new(),
            namespace: self.shared.namespace,
            max_frame: self.shared.max_frame as usize,
            max_redirects: self.shared.max_redirects,
            max_pending: self.shared.max_pending,
            connect_timeout: self.shared.connect_timeout,
            request_timeout: self.shared.request_timeout,
            write_timeout: self.shared.write_timeout,
            dial_attempts: self.shared.dial_attempts,
            dial_backoff: self.shared.dial_backoff,
            delivery_retries: self.shared.delivery_retries,
            // Transport-only redial config: the session lives in `Shared`,
            // not the connection, so there is nothing to adopt here.
            session: None,
        };
        let conn = Arc::new(Connection::dial(endpoint, expect_worker, &config)?);
        self.shared
            .pool
            .lock()
            .map_err(|_| ClientError::Io("connection pool lock poisoned".to_owned()))?
            .insert(endpoint.to_owned(), Arc::clone(&conn));
        Ok(conn)
    }

    /// Issues one call identity: allocates fresh, or adopts the named
    /// sequence for resumption. Returns the identity, the floor snapshot,
    /// and the raw sequence for completion bookkeeping (`None` for reads).
    ///
    /// # Errors
    ///
    /// Returns [`ClientError::Internal`] when the sequence space is
    /// exhausted or an explicit sequence is out of range.
    fn issue_identity(
        &self,
        mutating: bool,
        seq: Option<RequestSeq>,
    ) -> Result<Option<(RequestIdentity, RequestSeq, u64)>, ClientError> {
        if !mutating {
            return Ok(None);
        }
        let (identity, floor) = match seq {
            Some(explicit) => self.adopt_seq(explicit)?,
            None => self.alloc_seq()?,
        };
        Ok(Some((identity, floor, identity.seq().as_u64())))
    }

    /// Executes one request payload against the authority for `key`,
    /// following redirects with the same request identity throughout.
    /// Mutating opcodes carry one mutation identity (allocated here, never
    /// reallocated across retries); reads carry none.
    fn execute(
        &self,
        key: &Key,
        opcode: kivi_protocol::Opcode,
        build: impl Fn() -> kivi_protocol::Request,
    ) -> Result<Response, ClientError> {
        self.execute_with_seq(key, opcode, build, None)
    }

    /// Executes one request, naming the mutation sequence explicitly when
    /// `seq` is `Some` (session resumption: the identity must have been
    /// issued under this session before — e.g., sent but unacknowledged
    /// across a crash — so the server dedups it instead of executing
    /// twice). `None` allocates a fresh sequence. Reads ignore `seq`.
    /// Only `Ok`/`NotFound` responses return `Ok`; every other application
    /// status becomes its [`ClientError`] (transaction/scan drivers use
    /// [`execute_raw`](Self::execute_raw) to match statuses themselves).
    fn execute_with_seq(
        &self,
        key: &Key,
        opcode: kivi_protocol::Opcode,
        build: impl Fn() -> kivi_protocol::Request,
        seq: Option<RequestSeq>,
    ) -> Result<Response, ClientError> {
        use kivi_protocol::Status;
        use kivi_types::ReadContract;
        let response = self.execute_raw(key, opcode, build, seq, ReadContract::Latest)?;
        match response.status {
            Status::Ok | Status::NotFound => Ok(response),
            other => Err(status_error(other, &response.body)),
        }
    }

    /// Executes one read request under an explicit freshness contract.
    /// The contract is stamped onto every transport retry, so client
    /// retries preserve it; only `Ok`/`NotFound` return `Ok`.
    fn execute_with_contract(
        &self,
        key: &Key,
        opcode: kivi_protocol::Opcode,
        build: impl Fn() -> kivi_protocol::Request,
        contract: kivi_types::ReadContract,
    ) -> Result<Response, ClientError> {
        use kivi_protocol::Status;
        let response = self.execute_raw(key, opcode, build, None, contract)?;
        match response.status {
            Status::Ok | Status::NotFound => Ok(response),
            other => Err(status_error(other, &response.body)),
        }
    }

    /// Executes one request payload and returns the terminal response
    /// verbatim (any application status, including error statuses):
    /// redirects and overload are still followed/retried inside; transport
    /// failures still error. Scan, batch, and transaction drivers match on
    /// `(status, body)` themselves.
    #[allow(clippy::too_many_lines)]
    fn execute_raw(
        &self,
        key: &Key,
        opcode: kivi_protocol::Opcode,
        build: impl Fn() -> kivi_protocol::Request,
        seq: Option<RequestSeq>,
        contract: kivi_types::ReadContract,
    ) -> Result<Response, ClientError> {
        use kivi_protocol::Status;
        let hash = PartitionHasher::V1
            .hash(self.shared.namespace, key.as_bytes())
            .ok_or(ClientError::Internal(
                "partition hash unsupported".to_owned(),
            ))?;
        let id = self.next_id();
        let mutating = opcode.is_mutating();
        // One identity per typed call. The floor is snapshotted once;
        // retries reuse it (a stale floor only evicts less, never wrongly).
        // Explicit sequences resume a previously issued identity; fresh
        // sequences allocate.
        let identity = self.issue_identity(mutating, seq)?;
        let mut redirects = 0usize;
        let mut reconnects = 0usize;
        let mut deliveries = 0u32;
        // Consecutive `Overloaded` answers from one endpoint (a deaf or
        // partitioned member): bounded retries honor genuine elections,
        // then the route evicts and fails over instead of pinning on a
        // black hole.
        let mut overloaded_streak = 0u32;
        // Dead-member failover: endpoints that fail this call are
        // remembered (`tried`) and never redialed within the call, so a
        // killed or retired member fails over across untried seeds
        // instead of stranding the request on a grave (§34). Rotation
        // terminates: every seed is tried at most once per call, and
        // the redirect budget backstops everything else.
        // Redirect convergence guard: (tablet, endpoint, dir_version)
        // triples already followed this call, plus per-tablet version
        // knowledge. A redirect is progress only if routing knowledge
        // actually improves: a newer directory version, or a never-seen
        // tablet at the current version (a sibling at the same generation).
        // Anything else — an exact repeat, a revisited tablet at the same
        // or older version, or an older version for a new tablet
        // (`A@v10 → B@v9`) — is non-progress: back off and retry WITHOUT
        // spending redirect budget (the retried hop re-reads fresh state
        // and converges when the election/topology lands). Only a long
        // streak of zero-progress hops fails over to an untried seed by
        // refreshing routing from the seeds (the trusted source) instead
        // of burning the normal budget on a cycle like `A → B → A`.
        // Retries stay bounded: the streak cap times the redirect budget
        // bounds every call.
        let mut tried: Vec<String> = Vec::new();
        let mut visited_redirects: Vec<(u64, String, u64)> = Vec::new();
        let mut visited_tablets: std::collections::HashSet<u64> = std::collections::HashSet::new();
        let mut max_dir_version: u64 = 0;
        let mut stale_repeats: u32 = 0;
        let mut seed_cursor = 0usize;
        // Fresh-redirect follow: a redirect names the authority directly,
        // so the next hop goes there instead of re-resolving (re-resolution
        // can miss at exclusive range boundaries — e.g. reverse-scan
        // cursors — and ping-pong on the seed forever). Cleared once used;
        // never overrides dead-endpoint failover.
        let mut follow: Option<(String, kivi_protocol::RouteHint)> = None;
        self.shared.requests.fetch_add(1, Ordering::Relaxed);
        let outcome: Result<Response, ClientError> = loop {
            if redirects > self.shared.max_redirects {
                self.shared.errors.fetch_add(1, Ordering::Relaxed);
                break Err(ClientError::TooManyRedirects);
            }
            // Route: a fresh redirect target first (unless tried dead),
            // then cached range authority (ordered ranges by key, hash
            // ranges by partition hash — snapshots are single-layout, so
            // exactly one lookup can hit), unless already tried dead this
            // call, else the next untried seed with no hint. The mutation
            // identity above is preserved across all hops.
            let (endpoint, hint) = match follow.take() {
                Some((endpoint, hint)) if !tried.contains(&endpoint) => (endpoint, Some(hint)),
                _ => match self
                    .shared
                    .routes
                    .lookup_key(key.as_bytes())
                    .or_else(|| self.shared.routes.lookup(hash))
                {
                    Some(route) if !tried.contains(&route.endpoint) => {
                        (route.endpoint.clone(), Some(route.hint()))
                    }
                    _ => {
                        let mut pick = None;
                        for _ in 0..self.shared.seeds.len() {
                            let candidate =
                                self.shared.seeds[seed_cursor % self.shared.seeds.len()].clone();
                            seed_cursor += 1;
                            if !tried.contains(&candidate) {
                                pick = Some(candidate);
                                break;
                            }
                        }
                        let Some(endpoint) = pick else {
                            self.shared.errors.fetch_add(1, Ordering::Relaxed);
                            break Err(ClientError::Io("all known endpoints failed".to_owned()));
                        };
                        (endpoint, None)
                    }
                },
            };
            let expect = hint.map(|hint| hint.worker);
            let Ok(conn) = self.connection(&endpoint, expect) else {
                // A dead endpoint (killed member, retired replica)
                // evicts its cached routes and is never retried this
                // call; the next iteration picks an untried seed or
                // exhausts. Rotation does not consume redirect
                // budget (the tried-set already bounds it); the
                // budget counts routing loops only. No sleep: a
                // refused dial is definitive (backoff lives on
                // congestion paths, not graves).
                self.drop_connection(&endpoint);
                self.shared.routes.evict_endpoint(&endpoint);
                tried.push(endpoint);
                continue;
            };
            let mut request = build();
            request.hint = hint;
            // The freshness contract is stamped onto every transport
            // retry: redirects and redials preserve it exactly, so a
            // retried read never silently weakens (or strengthens) its
            // semantics mid-call.
            request.contract = contract;
            if let Some((identity, floor, _)) = &identity {
                request.identity = Some(*identity);
                request.ack_floor = *floor;
            }
            let response = match conn.round_trip(id, &request.encode()) {
                Ok(response) => response,
                Err(ClientError::Io(_)) if reconnects < 2 => {
                    // Delivery may or may not have happened. Reads have no
                    // side effects; mutating requests are safe to redial
                    // only against durable-dedup servers under the same
                    // identity. Otherwise the outcome is ambiguous: surface
                    // it instead of risking a duplicate execution.
                    let safe = !mutating || conn.durable_dedup();
                    reconnects += 1;
                    self.drop_connection(&endpoint);
                    if safe {
                        backoff(
                            self.shared.dial_backoff,
                            u32::try_from(reconnects).unwrap_or(u32::MAX),
                        );
                        continue;
                    }
                    self.shared.errors.fetch_add(1, Ordering::Relaxed);
                    break Err(ClientError::AmbiguousOutcome);
                }
                Err(ClientError::Timeout)
                    if mutating
                        && conn.durable_dedup()
                        && deliveries < self.shared.delivery_retries =>
                {
                    // Possibly transmitted, possibly not — but the identity
                    // makes redelivery safe: a completed first attempt is a
                    // dedup hit, otherwise it executes exactly once. Drop
                    // the connection (its waiter is gone) and redial.
                    deliveries += 1;
                    self.drop_connection(&endpoint);
                    backoff(self.shared.dial_backoff, deliveries.min(10));
                    continue;
                }
                Err(ClientError::Timeout) if mutating => {
                    self.shared.errors.fetch_add(1, Ordering::Relaxed);
                    break Err(ClientError::AmbiguousOutcome);
                }
                Err(ClientError::Io(_)) => {
                    // The endpoint died mid-request (kill, restart,
                    // retire): same failover as a refused dial when the
                    // retry is provably safe (reads, or mutating
                    // requests under durable dedup with a preserved
                    // identity). Otherwise the outcome stays ambiguous
                    // rather than risking a duplicate execution. The
                    // tried-set bounds rotation; the redirect budget
                    // backstops it.
                    let safe = !mutating || conn.durable_dedup();
                    if !safe {
                        self.shared.errors.fetch_add(1, Ordering::Relaxed);
                        break Err(ClientError::AmbiguousOutcome);
                    }
                    self.drop_connection(&endpoint);
                    self.shared.routes.evict_endpoint(&endpoint);
                    tried.push(endpoint);
                    reconnects = 0;
                    backoff(
                        self.shared.dial_backoff,
                        u32::try_from(redirects).unwrap_or(u32::MAX).min(6),
                    );
                    continue;
                }
                Err(error) => {
                    self.shared.errors.fetch_add(1, Ordering::Relaxed);
                    break Err(error);
                }
            };
            match response.status {
                Status::Ok | Status::NotFound => break Ok(response),
                Status::StaleRoute | Status::NotLocal => {
                    if let ResponseBody::Redirect(info) = &response.body {
                        let triple = (
                            info.tablet.as_u64(),
                            info.endpoint.clone(),
                            info.dir_version.as_u64(),
                        );
                        // Progress iff knowledge improves: exact repeats
                        // never do; newer versions always do; same-version
                        // siblings (new tablet, version == max) do; older
                        // versions and same-version revisits do not.
                        let tablet = info.tablet.as_u64();
                        let version = info.dir_version.as_u64();
                        let exact_repeat = visited_redirects.contains(&triple);
                        let is_progress = !exact_repeat
                            && (version > max_dir_version
                                || (version == max_dir_version
                                    && !visited_tablets.contains(&tablet)));
                        // The very first redirect establishes the baseline:
                        // `max_dir_version == 0` means nothing seen yet, so
                        // any non-repeat is progress (covers version 0/1).
                        let is_progress = is_progress || (!exact_repeat && max_dir_version == 0);
                        if is_progress {
                            visited_redirects.push(triple);
                            visited_tablets.insert(tablet);
                            max_dir_version = max_dir_version.max(version);
                            stale_repeats = 0;
                            // Update only the affected range. The next hop
                            // follows the redirect target directly:
                            // re-resolution can miss at exclusive range
                            // boundaries (reverse-scan cursors) and loop
                            // on the seed forever.
                            self.shared.routes.insert(&RouteEntry::from_redirect(info));
                            follow = Some((
                                info.endpoint.clone(),
                                kivi_protocol::RouteHint {
                                    tablet: info.tablet,
                                    epoch: info.epoch,
                                    worker: info.worker,
                                },
                            ));
                            overloaded_streak = 0;
                            redirects += 1;
                            self.shared.redirects.fetch_add(1, Ordering::Relaxed);
                            // Bounded backoff on redirects: stale hints during
                            // election churn retry safely without spinning.
                            backoff(
                                self.shared.dial_backoff,
                                u32::try_from(redirects).unwrap_or(u32::MAX).min(6),
                            );
                        } else {
                            // No forward progress: a cycle (`A → B → A`),
                            // a stale generation (`A@v10 → B@v9`), or the
                            // same authority answering the same generation
                            // again (election/observation window). Back off
                            // and retry without spending redirect budget;
                            // only a long zero-progress streak fails over
                            // by refreshing routing from the seeds (evict,
                            // mark tried, spend one redirect, continue
                            // elsewhere). Retries stay bounded: the streak
                            // cap times the redirect budget bounds calls.
                            stale_repeats += 1;
                            overloaded_streak = 0;
                            if stale_repeats > MAX_STALE_REPEATS {
                                stale_repeats = 0;
                                // Refresh from the trusted source: drop the
                                // suspect connection, evict its routes, and
                                // re-resolve through untried seeds instead
                                // of following another stale hint.
                                self.drop_connection(&endpoint);
                                self.shared.routes.evict_endpoint(&endpoint);
                                if !tried.contains(&endpoint) {
                                    tried.push(endpoint);
                                }
                                // Clear the direct-follow so the next hop
                                // re-resolves (the stale hint is not
                                // followed further).
                                follow = None;
                                redirects += 1;
                                self.shared.redirects.fetch_add(1, Ordering::Relaxed);
                                backoff(
                                    self.shared.dial_backoff,
                                    u32::try_from(redirects).unwrap_or(u32::MAX).min(6),
                                );
                            } else {
                                backoff(self.shared.dial_backoff, stale_repeats.min(6));
                            }
                        }
                    } else {
                        overloaded_streak = 0;
                        stale_repeats = 0;
                        redirects += 1;
                        self.shared.redirects.fetch_add(1, Ordering::Relaxed);
                        backoff(
                            self.shared.dial_backoff,
                            u32::try_from(redirects).unwrap_or(u32::MAX).min(6),
                        );
                    }
                }
                Status::Overloaded => {
                    // Election or migration churn: back off (like
                    // redirects) instead of spinning the budget dry in
                    // milliseconds while a leader emerges. Past a short
                    // streak, the endpoint is deaf rather than busy:
                    // evict and fail over instead of pinning on it.
                    redirects += 1;
                    overloaded_streak += 1;
                    self.shared.redirects.fetch_add(1, Ordering::Relaxed);
                    if overloaded_streak >= 3 {
                        overloaded_streak = 0;
                        self.drop_connection(&endpoint);
                        self.shared.routes.evict_endpoint(&endpoint);
                        tried.push(endpoint);
                    } else {
                        backoff(
                            self.shared.dial_backoff,
                            u32::try_from(redirects).unwrap_or(u32::MAX).min(6),
                        );
                    }
                }
                _ => {
                    // Raw terminal response: the typed wrappers map
                    // `Ok`/`NotFound`, drivers match the rest themselves.
                    self.shared.errors.fetch_add(1, Ordering::Relaxed);
                    break Ok(response);
                }
            }
        };
        // Every terminal path completes the identity — including abandonment
        // — so the acknowledgement floor can advance past it and the outcome
        // (if any) becomes eligible for eviction. An identity is never
        // retried after its call ends, so completing it cannot strand a
        // live retry beneath the floor.
        if let Some((_, _, raw)) = identity {
            self.complete_seq(raw);
        }
        outcome
    }

    /// Forgets one pooled connection so the next use redials. Used after
    /// transport failures; the replacement handshake revalidates everything.
    fn drop_connection(&self, endpoint: &str) {
        if let Ok(mut pool) = self.shared.pool.lock() {
            pool.remove(endpoint);
        }
    }

    /// Fetches bytes (`None` when absent).
    ///
    /// # Errors
    ///
    /// Returns [`ClientError`] on transport/routing failure or wrong-type access.
    pub fn get(&self, key: &Key) -> Result<Option<Bytes>, ClientError> {
        use kivi_protocol::{Opcode, ResponseBody};
        let response = self.execute(key, Opcode::Get, || kivi_protocol::Request {
            contract: kivi_types::ReadContract::Latest,
            namespace: self.shared.namespace,
            opcode: Opcode::Get,
            hint: None,
            key: key.as_bytes().to_vec(),
            value: None,
            delta: 0,
            expiry: 0,
            offset: 0,
            len: 0,
            condition: kivi_protocol::COND_ALWAYS,
            expiry_policy: kivi_protocol::EXPIRY_CLEAR,
            identity: None,
            ack_floor: RequestSeq::from_u64(0),
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
        })?;
        match response.body {
            ResponseBody::Value(value) => Ok(Some(Bytes::from(value))),
            ResponseBody::Diagnostic(_) => Ok(None),
            _ => Err(ClientError::Internal("unexpected get body".to_owned())),
        }
    }

    /// Fetches bytes under an explicit freshness contract, returning the
    /// value with the server's read proof (`None` when the server predates
    /// proofs: treat as no evidence, never as freshness).
    ///
    /// The contract rides every transport retry of this call, so redirects
    /// and redials preserve it. Feed `proof.map(|proof| proof.token())`
    /// back into [`get_at_least`](Self::get_at_least) to chain reads.
    ///
    /// # Errors
    ///
    /// Returns [`ClientError`] on transport/routing failure, wrong-type
    /// access, or (for `AtLeast`) [`ClientError::StaleToken`] when the
    /// token's lineage is foreign to the serving replica.
    pub fn get_with_contract(
        &self,
        key: &Key,
        contract: ReadContract,
    ) -> Result<(Option<Bytes>, Option<ReadReceipt>), ClientError> {
        use kivi_protocol::{Opcode, ResponseBody};
        let response = self.execute_with_contract(
            key,
            Opcode::Get,
            || kivi_protocol::Request {
                contract: kivi_types::ReadContract::Latest,
                namespace: self.shared.namespace,
                opcode: Opcode::Get,
                hint: None,
                key: key.as_bytes().to_vec(),
                value: None,
                delta: 0,
                expiry: 0,
                offset: 0,
                len: 0,
                condition: kivi_protocol::COND_ALWAYS,
                expiry_policy: kivi_protocol::EXPIRY_CLEAR,
                identity: None,
                ack_floor: RequestSeq::from_u64(0),
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
            contract,
        )?;
        let value = match response.body {
            ResponseBody::Value(value) => Ok(Some(Bytes::from(value))),
            ResponseBody::Diagnostic(_) => Ok(None),
            _ => Err(ClientError::Internal("unexpected get body".to_owned())),
        }?;
        Ok((value, response.proof))
    }

    /// Fetches bytes covered by `token` from any replica that has applied
    /// it (served without contacting the leader when already covered;
    /// bounded catch-up wait when behind; [`ClientError::StaleToken`]
    /// when the token names a foreign lineage).
    ///
    /// # Errors
    ///
    /// Returns [`ClientError`] on transport/routing failure, wrong-type
    /// access, or stale-token lineage mismatch.
    pub fn get_at_least(
        &self,
        key: &Key,
        token: CommitToken,
    ) -> Result<(Option<Bytes>, Option<ReadReceipt>), ClientError> {
        self.get_with_contract(key, ReadContract::AtLeast(token))
    }

    /// Fetches bytes acceptable up to `max_staleness` old. The server
    /// serves from proven-fresh state or escalates to the authority path;
    /// it never returns data outside the bound as success.
    ///
    /// # Errors
    ///
    /// Returns [`ClientError`] on transport/routing failure or wrong-type
    /// access.
    pub fn get_bounded_stale(
        &self,
        key: &Key,
        max_staleness: core::time::Duration,
    ) -> Result<(Option<Bytes>, Option<ReadReceipt>), ClientError> {
        self.get_with_contract(key, ReadContract::BoundedStale { max_staleness })
    }

    /// Fetches whatever applied state the contacted replica holds, with no
    /// freshness claim. May be arbitrarily stale; never blocks on
    /// coverage.
    ///
    /// # Errors
    ///
    /// Returns [`ClientError`] on transport/routing failure or wrong-type
    /// access.
    pub fn get_any(&self, key: &Key) -> Result<(Option<Bytes>, Option<ReadReceipt>), ClientError> {
        self.get_with_contract(key, ReadContract::Any)
    }

    /// Stores bytes, overwriting any type and clearing expiry. Takes the value
    /// by value: `Bytes` is a cheap shared handle and callers already own it.
    ///
    /// # Errors
    ///
    /// Returns [`ClientError`] on transport/routing failure.
    #[allow(clippy::needless_pass_by_value)]
    pub fn set(&self, key: &Key, value: Bytes) -> Result<(), ClientError> {
        use kivi_protocol::{Opcode, ResponseBody};
        let response = self.execute(key, Opcode::Set, || kivi_protocol::Request {
            contract: kivi_types::ReadContract::Latest,
            namespace: self.shared.namespace,
            opcode: Opcode::Set,
            hint: None,
            key: key.as_bytes().to_vec(),
            value: Some(value.to_vec()),
            delta: 0,
            expiry: 0,
            offset: 0,
            len: 0,
            condition: kivi_protocol::COND_ALWAYS,
            expiry_policy: kivi_protocol::EXPIRY_CLEAR,
            identity: None,
            ack_floor: RequestSeq::from_u64(0),
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
        })?;
        match response.body {
            ResponseBody::Stored { .. } => Ok(()),
            _ => Err(ClientError::Internal("unexpected set body".to_owned())),
        }
    }

    /// Patches a byte range of a value (partial update): `patch` overwrites
    /// the value starting at `offset`, zero-padding past-the-end gaps. The
    /// live expiry survives (unlike [`set`](Self::set)): partial writes
    /// touch bytes, never the TTL. Counters fail with wrong-type; absurd
    /// ranges fail inadmittable. The patch travels in one request frame —
    /// bulk rewrites belong on the streaming upload path instead.
    ///
    /// # Errors
    ///
    /// Returns [`ClientError`] on transport/routing failure, wrong-type
    /// access, or an inadmittable range.
    #[allow(clippy::needless_pass_by_value)]
    pub fn set_range(&self, key: &Key, offset: u64, patch: Bytes) -> Result<(), ClientError> {
        use kivi_protocol::{Opcode, ResponseBody};
        let response = self.execute(key, Opcode::SetRange, || kivi_protocol::Request {
            contract: kivi_types::ReadContract::Latest,
            namespace: self.shared.namespace,
            opcode: Opcode::SetRange,
            hint: None,
            key: key.as_bytes().to_vec(),
            value: Some(patch.to_vec()),
            delta: 0,
            expiry: 0,
            offset,
            len: 0,
            condition: kivi_protocol::COND_ALWAYS,
            expiry_policy: kivi_protocol::EXPIRY_CLEAR,
            identity: None,
            ack_floor: RequestSeq::from_u64(0),
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
        })?;
        match response.body {
            ResponseBody::Stored { .. } => Ok(()),
            // Restaged range writes answer `Stored` (the server shapes the
            // internal conditional outcome by request opcode); anything
            // else here is a contract break, never a runtime condition.
            _ => Err(ClientError::Internal(
                "unexpected set-range body".to_owned(),
            )),
        }
    }

    /// Streams a value upload of any size up to the 1 GiB stream bound:
    /// frames carry slices while the server stages chunk by chunk, and one
    /// commit barrier proves the manifest — neither side materializes the
    /// value outside chunking. Values over the inline threshold land as
    /// chunked roots and read back identically through [`get`](Self::get).
    /// Pass `total_len` when known (the server aborts on mismatch instead
    /// of committing short); `None` streams until `source` ends.
    ///
    /// Routing warms from the cache (probing when cold); a mid-upload
    /// range move surfaces as an error and the caller retries with a
    /// fresh source — pass an explicit sequence (see
    /// [`put_stream_with_seq`](Self::put_stream_with_seq)) when the retry
    /// must dedup against a possibly-committed first attempt.
    ///
    /// # Errors
    ///
    /// Returns [`ClientError`] on transport/routing failure, a legacy
    /// server (`Unsupported`, checked before sending anything), an
    /// over-bound or inconsistent upload, or a lost commit response
    /// (`AmbiguousOutcome`: the commit may have applied).
    pub fn put_stream(
        &self,
        key: &Key,
        total_len: Option<u64>,
        source: &mut impl Read,
    ) -> Result<(), ClientError> {
        self.put_stream_with_seq(key, total_len, source, None)
    }

    /// Streams an upload under an explicitly named sequence of this
    /// session (resumption: same contract as `set_with_seq` — a retried
    /// upload with the same identity commits at most once, dedup-hitting
    /// when the first attempt already applied). Pair with
    /// `ClientConfig::session` to resume across restarts.
    ///
    /// # Errors
    ///
    /// Returns [`ClientError`] on transport/routing failure, an
    /// out-of-range sequence, over-bound uploads, or a lost commit
    /// response (`AmbiguousOutcome`).
    pub fn put_stream_with_seq(
        &self,
        key: &Key,
        total_len: Option<u64>,
        source: &mut impl Read,
        seq: Option<RequestSeq>,
    ) -> Result<(), ClientError> {
        // One identity per typed call, mirroring `execute_with_seq`: fresh
        // sequences allocate, explicit ones resume. Completed here on
        // every terminal path so the floor can advance past it.
        let identity = self.issue_identity(true, seq)?;
        let outcome = self.put_stream_inner(key, total_len, source, identity.as_ref());
        if let Some((_, _, raw)) = identity {
            self.complete_seq(raw);
        }
        outcome
    }

    /// Streams a value download of any size: the server fans the value
    /// into `ValueStream*` frames (entry by entry for chunked values, so
    /// server memory stays flat) and the client reassembles. Returns
    /// `None` when absent. Downloads materialize client-side, bounded by
    /// the 1 GiB stream bound — bulk processing without materializing is
    /// a future reader API, not this call.
    ///
    /// # Errors
    ///
    /// Returns [`ClientError`] on transport/routing failure, a legacy
    /// server (`Unsupported`, checked before sending anything),
    /// wrong-type access, or a truncated stream (`Protocol`: the byte
    /// count, not just the end marker, must agree).
    pub fn get_stream(&self, key: &Key) -> Result<Option<Bytes>, ClientError> {
        let hash = PartitionHasher::V1
            .hash(self.shared.namespace, key.as_bytes())
            .ok_or(ClientError::Internal(
                "partition hash unsupported".to_owned(),
            ))?;
        let mut redirects = 0usize;
        let mut reconnects = 0usize;
        // Same dead-member failover as `execute_with_seq` (reads
        // re-run safely, so rotation never risks duplication): failed
        // endpoints are tried at most once per call. Same stale-repeat
        // patience: zero-progress redirect triples back off without
        // spending budget (election windows), failing over only after a
        // long streak.
        let mut tried: Vec<String> = Vec::new();
        let mut visited_redirects: Vec<(u64, String, u64)> = Vec::new();
        let mut visited_tablets: std::collections::HashSet<u64> = std::collections::HashSet::new();
        let mut max_dir_version: u64 = 0;
        let mut stale_repeats: u32 = 0;
        let mut seed_cursor = 0usize;
        self.shared.requests.fetch_add(1, Ordering::Relaxed);
        loop {
            if redirects > self.shared.max_redirects {
                self.shared.errors.fetch_add(1, Ordering::Relaxed);
                return Err(ClientError::TooManyRedirects);
            }
            let Some((endpoint, hint)) =
                self.next_stream_endpoint(key, hash, &tried, &mut seed_cursor)
            else {
                self.shared.errors.fetch_add(1, Ordering::Relaxed);
                return Err(ClientError::Io("all known endpoints failed".to_owned()));
            };
            let expect = hint.map(|hint| hint.worker);
            let Ok(conn) = self.connection(&endpoint, expect) else {
                // Refused dials fail over immediately (no sleep: the
                // refusal is definitive; the tried-set terminates
                // without consuming redirect budget).
                self.drop_connection(&endpoint);
                self.shared.routes.evict_endpoint(&endpoint);
                tried.push(endpoint);
                continue;
            };
            if !conn.streaming() {
                self.shared.errors.fetch_add(1, Ordering::Relaxed);
                return Err(ClientError::Unsupported);
            }
            match self.get_stream_attempt(&conn, key, hint) {
                GetOutcome::Done(value) => return Ok(value),
                GetOutcome::Reroute(route) => {
                    let progress = Self::note_stream_reroute_versioned(
                        self,
                        route.as_ref(),
                        &endpoint,
                        &mut visited_redirects,
                        &mut visited_tablets,
                        &mut max_dir_version,
                        &mut stale_repeats,
                        &mut tried,
                        &mut redirects,
                    );
                    if progress && let Some(route) = route {
                        self.shared.routes.insert(&route);
                    }
                }
                GetOutcome::Overloaded => {
                    redirects += 1;
                    self.shared.redirects.fetch_add(1, Ordering::Relaxed);
                    backoff(
                        self.shared.dial_backoff,
                        u32::try_from(redirects).unwrap_or(u32::MAX).min(10),
                    );
                }
                GetOutcome::Reconnect => {
                    // Reads have no side effects: redial and retry, like
                    // `execute_with_seq`, failing over across untried
                    // seeds when one member stays down.
                    reconnects += 1;
                    self.drop_connection(&endpoint);
                    if reconnects > 2 {
                        self.shared.routes.evict_endpoint(&endpoint);
                        tried.push(endpoint);
                        reconnects = 0;
                    }
                    backoff(
                        self.shared.dial_backoff,
                        u32::try_from(redirects).unwrap_or(u32::MAX).min(6),
                    );
                }
                GetOutcome::Fail(error) => {
                    self.shared.errors.fetch_add(1, Ordering::Relaxed);
                    return Err(error);
                }
            }
        }
    }

    fn next_stream_endpoint(
        &self,
        key: &Key,
        hash: kivi_types::PartitionHash,
        tried: &[String],
        seed_cursor: &mut usize,
    ) -> Option<(String, Option<RouteHint>)> {
        if let Some(route) = self
            .shared
            .routes
            .lookup_key(key.as_bytes())
            .or_else(|| self.shared.routes.lookup(hash))
            && !tried.contains(&route.endpoint)
        {
            return Some((route.endpoint.clone(), Some(route.hint())));
        }
        let seed_count = self.shared.seeds.len();
        (0..seed_count).find_map(|_| {
            let endpoint = self.shared.seeds[*seed_cursor % seed_count].clone();
            *seed_cursor += 1;
            (!tried.contains(&endpoint)).then_some((endpoint, None))
        })
    }

    /// Records one stream reroute: progress (newer version, or a new
    /// tablet at the current version) spends redirect budget;
    /// non-progress (exact repeats, revisited tablets, older versions —
    /// `A → B → A`, `A@v10 → B@v9`) backs off without spending it
    /// (election/topology windows), failing over only after
    /// [`MAX_STALE_REPEATS`]. Mirrors the point-path convergence guard.
    #[allow(clippy::too_many_arguments)]
    fn note_stream_reroute_versioned(
        &self,
        route: Option<&RouteEntry>,
        endpoint: &str,
        visited_redirects: &mut Vec<(u64, String, u64)>,
        visited_tablets: &mut std::collections::HashSet<u64>,
        max_dir_version: &mut u64,
        stale_repeats: &mut u32,
        tried: &mut Vec<String>,
        redirects: &mut usize,
    ) -> bool {
        let Some(route) = route else {
            *stale_repeats = 0;
            *redirects += 1;
            self.shared.redirects.fetch_add(1, Ordering::Relaxed);
            return false;
        };
        let tablet = route.tablet.as_u64();
        let triple = (tablet, route.endpoint.clone(), route.dir_version);
        let exact_repeat = visited_redirects.contains(&triple);
        let is_progress = !exact_repeat
            && (route.dir_version > *max_dir_version
                || (route.dir_version == *max_dir_version && !visited_tablets.contains(&tablet))
                || *max_dir_version == 0);
        if is_progress {
            visited_redirects.push(triple);
            visited_tablets.insert(tablet);
            *max_dir_version = (*max_dir_version).max(route.dir_version);
            *stale_repeats = 0;
        } else {
            *stale_repeats += 1;
        }
        let repeat = !is_progress;
        if repeat && *stale_repeats <= MAX_STALE_REPEATS {
            backoff(self.shared.dial_backoff, (*stale_repeats).min(10));
        } else {
            if repeat {
                *stale_repeats = 0;
                self.drop_connection(endpoint);
                self.shared.routes.evict_endpoint(endpoint);
                tried.push(endpoint.to_owned());
            }
            *redirects += 1;
            self.shared.redirects.fetch_add(1, Ordering::Relaxed);
        }
        is_progress
    }

    /// One upload attempt against the routed endpoint: begin, data, commit.
    /// Pre-commit failures are definitely uncommitted (the commit is the
    /// single atomic point, and this call never retries past it with a
    /// consumed source); only a lost commit response is ambiguous.
    #[allow(clippy::too_many_arguments)]
    fn put_stream_inner(
        &self,
        key: &Key,
        total_len: Option<u64>,
        source: &mut impl Read,
        identity: Option<&(RequestIdentity, RequestSeq, u64)>,
    ) -> Result<(), ClientError> {
        let hash = PartitionHasher::V1
            .hash(self.shared.namespace, key.as_bytes())
            .ok_or(ClientError::Internal(
                "partition hash unsupported".to_owned(),
            ))?;
        // Cold routes probe first: one cheap `Exists` warms the cache
        // through the ordinary redirect machinery, so the begin below
        // lands on the owner the first time (a move between probe and
        // begin still aborts cleanly and surfaces — never miscommits).
        if self
            .shared
            .routes
            .lookup_key(key.as_bytes())
            .or_else(|| self.shared.routes.lookup(hash))
            .is_none()
        {
            let _ = self.exists(key);
        }
        self.shared.requests.fetch_add(1, Ordering::Relaxed);
        let (endpoint, hint) = match self
            .shared
            .routes
            .lookup_key(key.as_bytes())
            .or_else(|| self.shared.routes.lookup(hash))
        {
            Some(route) => (route.endpoint.clone(), Some(route.hint())),
            None => (self.shared.seeds[0].clone(), None),
        };
        let expect = hint.map(|hint| hint.worker);
        let conn = match self.connection(&endpoint, expect) {
            Ok(conn) => conn,
            Err(error) => {
                self.shared.errors.fetch_add(1, Ordering::Relaxed);
                return Err(error);
            }
        };
        if !conn.streaming() {
            self.shared.errors.fetch_add(1, Ordering::Relaxed);
            return Err(ClientError::Unsupported);
        }
        let outcome = self.upload_attempt(&conn, &endpoint, key, total_len, source, identity);
        if outcome.is_err() {
            self.shared.errors.fetch_add(1, Ordering::Relaxed);
        }
        outcome
    }

    /// Runs begin → data → commit on one connection. Returns the terminal
    /// error with server state already accounted for (aborts sent where
    /// the server may hold an assembler).
    fn upload_attempt(
        &self,
        conn: &Arc<Connection>,
        endpoint: &str,
        key: &Key,
        total_len: Option<u64>,
        source: &mut impl Read,
        identity: Option<&(RequestIdentity, RequestSeq, u64)>,
    ) -> Result<(), ClientError> {
        if total_len.is_some_and(|total| total > MAX_STREAM_UPLOAD_BYTES) {
            return Err(ClientError::ValueTooLarge);
        }
        let stream = self.next_stream_id();
        let rx = Connection::register_stream(conn, stream)?;
        let (id, floor) = match identity {
            Some((identity, floor, _)) => (Some(*identity), *floor),
            None => (None, RequestSeq::from_u64(0)),
        };
        let begin = StreamBegin {
            namespace: self.shared.namespace,
            key: key.as_bytes().to_vec(),
            total_len,
            identity: id,
            ack_floor: floor,
        };
        if let Err(error) =
            Connection::send_frame(conn, FrameKind::StreamBegin, stream, &begin.encode())
        {
            Connection::unregister_stream(conn, stream);
            return Err(match error {
                ClientError::Io(_) => {
                    self.drop_connection(endpoint);
                    ClientError::AmbiguousOutcome
                }
                other => other,
            });
        }
        // Ready (or an immediate abort: unknown namespace, over bound,
        // misrouted key — all pre-commit, so all definitely uncommitted).
        let max_data = match Connection::await_stream_event(conn, &rx) {
            Ok(StreamEvent::Ready(max_data)) => max_data.max(1) as usize,
            Ok(StreamEvent::Aborted(reason)) => {
                Connection::unregister_stream(conn, stream);
                return Err(ClientError::Internal(reason));
            }
            Ok(_) => {
                Connection::unregister_stream(conn, stream);
                Connection::abort_stream(conn, stream, "client closing malformed stream");
                return Err(ClientError::Protocol(ProtocolError::Malformed {
                    context: "stream-ready",
                }));
            }
            Err(error) => {
                Connection::unregister_stream(conn, stream);
                return Err(match error {
                    ClientError::Timeout => ClientError::Timeout,
                    ClientError::Io(_) => {
                        self.drop_connection(endpoint);
                        ClientError::AmbiguousOutcome
                    }
                    other => other,
                });
            }
        };
        self.upload_data(conn, endpoint, stream, source, total_len, max_data)?;
        // Commit: the atomic point. A lost response from here on is
        // ambiguous (the root may have committed) — same contract as
        // `execute_with_seq` past delivery.
        if let Err(error) = Connection::send_frame(conn, FrameKind::StreamCommit, stream, &[]) {
            Connection::unregister_stream(conn, stream);
            self.drop_connection(endpoint);
            return Err(match error {
                ClientError::Io(_) => ClientError::AmbiguousOutcome,
                other => other,
            });
        }
        match Connection::await_stream_event(conn, &rx) {
            // The commit response arrives under the stream id and
            // unregisters the rendezvous on arrival.
            Ok(StreamEvent::Committed(response)) => match response.status {
                Status::Ok => Ok(()),
                other => Err(status_error(other, &response.body)),
            },
            Ok(StreamEvent::Aborted(reason)) => Err(ClientError::Internal(reason)),
            Ok(_) => {
                Connection::abort_stream(conn, stream, "client closing malformed stream");
                Err(ClientError::Protocol(ProtocolError::Malformed {
                    context: "stream-commit",
                }))
            }
            Err(ClientError::Timeout | ClientError::Io(_)) => {
                Connection::unregister_stream(conn, stream);
                self.drop_connection(endpoint);
                Err(ClientError::AmbiguousOutcome)
            }
            Err(other) => {
                Connection::unregister_stream(conn, stream);
                Err(other)
            }
        }
    }

    /// Streams source bytes as data frames capped at the negotiated size.
    /// Anything failing here is pre-commit, hence definitely uncommitted —
    /// plain errors, never ambiguous. Short writer holds per frame, reads
    /// chunked to the cap.
    fn upload_data(
        &self,
        conn: &Arc<Connection>,
        endpoint: &str,
        stream: u64,
        source: &mut impl Read,
        total_len: Option<u64>,
        max_data: usize,
    ) -> Result<(), ClientError> {
        let mut delivered = 0u64;
        let mut buf = vec![0u8; max_data];
        loop {
            let count = match source.read(&mut buf) {
                Ok(0) => break,
                Ok(count) => count,
                Err(error) => {
                    Connection::unregister_stream(conn, stream);
                    Connection::abort_stream(conn, stream, "client source failed");
                    return Err(ClientError::Io(format!("stream source: {error}")));
                }
            };
            delivered += count as u64;
            if delivered > MAX_STREAM_UPLOAD_BYTES {
                Connection::unregister_stream(conn, stream);
                Connection::abort_stream(conn, stream, "upload exceeds the stream bound");
                return Err(ClientError::ValueTooLarge);
            }
            if total_len.is_some_and(|total| delivered > total) {
                Connection::unregister_stream(conn, stream);
                Connection::abort_stream(conn, stream, "upload exceeds its declared total");
                return Err(ClientError::InvalidRequest);
            }
            if let Err(error) =
                Connection::send_frame(conn, FrameKind::StreamData, stream, &buf[..count])
            {
                Connection::unregister_stream(conn, stream);
                self.drop_connection(endpoint);
                return Err(match error {
                    ClientError::Io(detail) => ClientError::Io(detail),
                    other => other,
                });
            }
        }
        if total_len.is_some_and(|total| delivered != total) {
            Connection::unregister_stream(conn, stream);
            Connection::abort_stream(conn, stream, "upload length mismatches its declared total");
            return Err(ClientError::InvalidRequest);
        }
        Ok(())
    }

    /// One download attempt: request, then collect frames to the declared
    /// total. Reads never mutate, so every failure mode either retries
    /// (redirects, redials) or surfaces plainly — nothing is ambiguous.
    fn get_stream_attempt(
        &self,
        conn: &Arc<Connection>,
        key: &Key,
        hint: Option<kivi_protocol::RouteHint>,
    ) -> GetOutcome {
        use kivi_protocol::{Opcode, ResponseBody};
        let stream = self.next_stream_id();
        let rx = match Connection::register_stream(conn, stream) {
            Ok(rx) => rx,
            Err(ClientError::Overloaded) => return GetOutcome::Overloaded,
            Err(error) => return GetOutcome::Fail(error),
        };
        let mut request = self.request_base(key, Opcode::GetStream);
        request.hint = hint;
        if let Err(error) =
            Connection::send_frame(conn, FrameKind::Request, stream, &request.encode())
        {
            Connection::unregister_stream(conn, stream);
            return match error {
                ClientError::Io(_) => GetOutcome::Reconnect,
                other => GetOutcome::Fail(other),
            };
        }
        // Head: begin (stream the body), a terminal response (absent,
        // redirect, error), or an abort.
        let total = match Connection::await_stream_event(conn, &rx) {
            Ok(StreamEvent::ValueBegin(total)) => total,
            Ok(StreamEvent::Committed(response)) => {
                return match response.status {
                    Status::NotFound => GetOutcome::Done(None),
                    Status::StaleRoute | Status::NotLocal => {
                        if let ResponseBody::Redirect(info) = &response.body {
                            GetOutcome::Reroute(Some(RouteEntry::from_redirect(info)))
                        } else {
                            GetOutcome::Reroute(None)
                        }
                    }
                    Status::Overloaded => GetOutcome::Overloaded,
                    other => GetOutcome::Fail(status_error(other, &response.body)),
                };
            }
            Ok(StreamEvent::Aborted(reason)) => {
                return GetOutcome::Fail(ClientError::Internal(reason));
            }
            Ok(_) => {
                Connection::unregister_stream(conn, stream);
                Connection::abort_stream(conn, stream, "client closing malformed stream");
                return GetOutcome::Fail(ClientError::Protocol(ProtocolError::Malformed {
                    context: "value-stream-begin",
                }));
            }
            Err(ClientError::Timeout) => {
                Connection::unregister_stream(conn, stream);
                return GetOutcome::Fail(ClientError::Timeout);
            }
            Err(ClientError::Io(_)) => {
                Connection::unregister_stream(conn, stream);
                return GetOutcome::Reconnect;
            }
            Err(other) => {
                Connection::unregister_stream(conn, stream);
                return GetOutcome::Fail(other);
            }
        };
        if total > MAX_STREAM_UPLOAD_BYTES {
            Connection::unregister_stream(conn, stream);
            Connection::abort_stream(conn, stream, "download exceeds the stream bound");
            return GetOutcome::Fail(ClientError::ValueTooLarge);
        }
        Self::collect_stream_body(conn, stream, &rx, total)
    }

    /// Collects download frames to the declared total, then the end
    /// marker. The count — not just the marker — must agree, or the
    /// download is truncated and fails instead of returning short bytes.
    fn collect_stream_body(
        conn: &Arc<Connection>,
        stream: u64,
        rx: &crossbeam_channel::Receiver<StreamEvent>,
        total: u64,
    ) -> GetOutcome {
        let mut out = Vec::new();
        if total > 0
            && let Ok(capacity) = usize::try_from(total)
            && out.try_reserve_exact(capacity).is_err()
        {
            Connection::unregister_stream(conn, stream);
            Connection::abort_stream(conn, stream, "download allocation exceeds client budget");
            return GetOutcome::Fail(ClientError::ResourceExhausted);
        }
        loop {
            match Connection::await_stream_event(conn, rx) {
                Ok(StreamEvent::ValueData(bytes)) => {
                    if out.len() as u64 + bytes.len() as u64 > total {
                        Connection::unregister_stream(conn, stream);
                        Connection::abort_stream(
                            conn,
                            stream,
                            "download overran its declared total",
                        );
                        return GetOutcome::Fail(ClientError::Protocol(ProtocolError::Malformed {
                            context: "value-stream-data",
                        }));
                    }
                    out.extend_from_slice(&bytes);
                }
                Ok(StreamEvent::ValueEnd) => {
                    Connection::unregister_stream(conn, stream);
                    if out.len() as u64 != total {
                        return GetOutcome::Fail(ClientError::Protocol(ProtocolError::Malformed {
                            context: "value-stream-end",
                        }));
                    }
                    return GetOutcome::Done(Some(Bytes::from(out)));
                }
                Ok(StreamEvent::Aborted(reason)) => {
                    return GetOutcome::Fail(ClientError::Internal(reason));
                }
                Ok(_) => {
                    Connection::unregister_stream(conn, stream);
                    Connection::abort_stream(conn, stream, "client closing malformed stream");
                    return GetOutcome::Fail(ClientError::Protocol(ProtocolError::Malformed {
                        context: "value-stream",
                    }));
                }
                Err(ClientError::Timeout) => {
                    Connection::unregister_stream(conn, stream);
                    return GetOutcome::Fail(ClientError::Timeout);
                }
                Err(ClientError::Io(_)) => {
                    Connection::unregister_stream(conn, stream);
                    return GetOutcome::Reconnect;
                }
                Err(other) => {
                    Connection::unregister_stream(conn, stream);
                    return GetOutcome::Fail(other);
                }
            }
        }
    }

    /// Stores bytes under an explicitly named sequence of this session
    /// (resumption: the identity was issued before — sent but never
    /// acknowledged — so this either executes it for the first time or
    /// dedup-hits the stored outcome; it never executes twice). Pair with
    /// `ClientConfig::session` to resume across restarts.
    ///
    /// # Errors
    ///
    /// Returns [`ClientError`] on transport/routing failure, an out-of-range
    /// sequence, or a server-side rejection (including `DedupExpired` when
    /// the session floor already passed the sequence).
    #[allow(clippy::needless_pass_by_value)]
    pub fn set_with_seq(
        &self,
        key: &Key,
        value: Bytes,
        seq: RequestSeq,
    ) -> Result<(), ClientError> {
        use kivi_protocol::{Opcode, ResponseBody};
        let response = self.execute_with_seq(
            key,
            Opcode::Set,
            || kivi_protocol::Request {
                contract: kivi_types::ReadContract::Latest,
                namespace: self.shared.namespace,
                opcode: Opcode::Set,
                hint: None,
                key: key.as_bytes().to_vec(),
                value: Some(value.to_vec()),
                delta: 0,
                expiry: 0,
                offset: 0,
                len: 0,
                condition: kivi_protocol::COND_ALWAYS,
                expiry_policy: kivi_protocol::EXPIRY_CLEAR,
                identity: None,
                ack_floor: RequestSeq::from_u64(0),
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
            Some(seq),
        )?;
        match response.body {
            ResponseBody::Stored { .. } => Ok(()),
            _ => Err(ClientError::Internal("unexpected set body".to_owned())),
        }
    }
    /// Removes a key, reporting whether a live object existed.
    ///
    /// # Errors
    ///
    /// Returns [`ClientError`] on transport/routing failure.
    pub fn delete(&self, key: &Key) -> Result<bool, ClientError> {
        use kivi_protocol::{Opcode, ResponseBody};
        let response = self.execute(key, Opcode::Delete, || kivi_protocol::Request {
            contract: kivi_types::ReadContract::Latest,
            namespace: self.shared.namespace,
            opcode: Opcode::Delete,
            hint: None,
            key: key.as_bytes().to_vec(),
            value: None,
            delta: 0,
            expiry: 0,
            offset: 0,
            len: 0,
            condition: kivi_protocol::COND_ALWAYS,
            expiry_policy: kivi_protocol::EXPIRY_CLEAR,
            identity: None,
            ack_floor: RequestSeq::from_u64(0),
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
        })?;
        match response.body {
            ResponseBody::Deleted { existed } => Ok(existed),
            _ => Err(ClientError::Internal("unexpected delete body".to_owned())),
        }
    }

    /// Reports whether a live value of any type exists.
    ///
    /// # Errors
    ///
    /// Returns [`ClientError`] on transport/routing failure.
    pub fn exists(&self, key: &Key) -> Result<bool, ClientError> {
        use kivi_protocol::{Opcode, ResponseBody};
        let response = self.execute(key, Opcode::Exists, || kivi_protocol::Request {
            contract: kivi_types::ReadContract::Latest,
            namespace: self.shared.namespace,
            opcode: Opcode::Exists,
            hint: None,
            key: key.as_bytes().to_vec(),
            value: None,
            delta: 0,
            expiry: 0,
            offset: 0,
            len: 0,
            condition: kivi_protocol::COND_ALWAYS,
            expiry_policy: kivi_protocol::EXPIRY_CLEAR,
            identity: None,
            ack_floor: RequestSeq::from_u64(0),
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
        })?;
        match response.body {
            ResponseBody::Exists(present) => Ok(present),
            _ => Err(ClientError::Internal("unexpected exists body".to_owned())),
        }
    }

    /// Reads a counter (`None` when absent).
    ///
    /// # Errors
    ///
    /// Returns [`ClientError`] on transport/routing failure or wrong-type access.
    pub fn counter_get(&self, key: &Key) -> Result<Option<i64>, ClientError> {
        use kivi_protocol::{Opcode, ResponseBody};
        let response = self.execute(key, Opcode::CounterGet, || kivi_protocol::Request {
            contract: kivi_types::ReadContract::Latest,
            namespace: self.shared.namespace,
            opcode: Opcode::CounterGet,
            hint: None,
            key: key.as_bytes().to_vec(),
            value: None,
            delta: 0,
            expiry: 0,
            offset: 0,
            len: 0,
            condition: kivi_protocol::COND_ALWAYS,
            expiry_policy: kivi_protocol::EXPIRY_CLEAR,
            identity: None,
            ack_floor: RequestSeq::from_u64(0),
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
        })?;
        match response.body {
            ResponseBody::Counter(value) => Ok(Some(value)),
            ResponseBody::Diagnostic(_) => Ok(None),
            _ => Err(ClientError::Internal(
                "unexpected counter-get body".to_owned(),
            )),
        }
    }

    /// Adds `delta` to a counter and returns the new value.
    ///
    /// # Errors
    ///
    /// Returns [`ClientError`] on transport/routing failure, wrong-type
    /// access, or counter overflow.
    pub fn counter_add(&self, key: &Key, delta: i64) -> Result<i64, ClientError> {
        use kivi_protocol::{Opcode, ResponseBody};
        let response = self.execute(key, Opcode::CounterAdd, || kivi_protocol::Request {
            contract: kivi_types::ReadContract::Latest,
            namespace: self.shared.namespace,
            opcode: Opcode::CounterAdd,
            hint: None,
            key: key.as_bytes().to_vec(),
            value: None,
            delta,
            expiry: 0,
            offset: 0,
            len: 0,
            condition: kivi_protocol::COND_ALWAYS,
            expiry_policy: kivi_protocol::EXPIRY_CLEAR,
            identity: None,
            ack_floor: RequestSeq::from_u64(0),
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
        })?;
        match response.body {
            ResponseBody::CounterUpdated { value, .. } => Ok(value),
            _ => Err(ClientError::Internal(
                "unexpected counter-add body".to_owned(),
            )),
        }
    }

    /// Adds `delta` under an explicitly named sequence of this session
    /// (resumption: same contract as `set_with_seq` — executes at most
    /// once per identity, dedup-hitting when the first attempt already
    /// applied). A dedup hit returns the originally recorded post-add
    /// value, not a re-execution.
    ///
    /// # Errors
    ///
    /// Returns [`ClientError`] on transport/routing failure, wrong-type
    /// access, counter overflow, an out-of-range sequence, or a
    /// server-side rejection (including `DedupExpired`).
    pub fn counter_add_with_seq(
        &self,
        key: &Key,
        delta: i64,
        seq: RequestSeq,
    ) -> Result<i64, ClientError> {
        use kivi_protocol::{Opcode, ResponseBody};
        let response = self.execute_with_seq(
            key,
            Opcode::CounterAdd,
            || kivi_protocol::Request {
                contract: kivi_types::ReadContract::Latest,
                namespace: self.shared.namespace,
                opcode: Opcode::CounterAdd,
                hint: None,
                key: key.as_bytes().to_vec(),
                value: None,
                delta,
                expiry: 0,
                offset: 0,
                len: 0,
                condition: kivi_protocol::COND_ALWAYS,
                expiry_policy: kivi_protocol::EXPIRY_CLEAR,
                identity: None,
                ack_floor: RequestSeq::from_u64(0),
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
            Some(seq),
        )?;
        match response.body {
            ResponseBody::CounterUpdated { value, .. } => Ok(value),
            _ => Err(ClientError::Internal(
                "unexpected counter-add body".to_owned(),
            )),
        }
    }

    /// Attaches an absolute expiry; `false` when the key is absent.
    ///
    /// # Errors
    ///
    /// Returns [`ClientError`] on transport/routing failure.
    pub fn expire_at(&self, key: &Key, expires_at: WallTimestamp) -> Result<bool, ClientError> {
        use kivi_protocol::{Opcode, ResponseBody};
        let response = self.execute(key, Opcode::ExpireAt, || kivi_protocol::Request {
            contract: kivi_types::ReadContract::Latest,
            namespace: self.shared.namespace,
            opcode: Opcode::ExpireAt,
            hint: None,
            key: key.as_bytes().to_vec(),
            value: None,
            delta: 0,
            expiry: expires_at.as_micros(),
            offset: 0,
            len: 0,
            condition: kivi_protocol::COND_ALWAYS,
            expiry_policy: kivi_protocol::EXPIRY_CLEAR,
            identity: None,
            ack_floor: RequestSeq::from_u64(0),
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
        })?;
        match response.body {
            ResponseBody::ExpirySet { applied } => Ok(applied),
            _ => Err(ClientError::Internal(
                "unexpected expire-at body".to_owned(),
            )),
        }
    }

    /// Removes any expiry; `false` when the key has none or is absent.
    ///
    /// # Errors
    ///
    /// Returns [`ClientError`] on transport/routing failure.
    pub fn persist_expiry(&self, key: &Key) -> Result<bool, ClientError> {
        use kivi_protocol::{Opcode, ResponseBody};
        let response = self.execute(key, Opcode::PersistExpiry, || kivi_protocol::Request {
            contract: kivi_types::ReadContract::Latest,
            namespace: self.shared.namespace,
            opcode: Opcode::PersistExpiry,
            hint: None,
            key: key.as_bytes().to_vec(),
            value: None,
            delta: 0,
            expiry: 0,
            offset: 0,
            len: 0,
            condition: kivi_protocol::COND_ALWAYS,
            expiry_policy: kivi_protocol::EXPIRY_CLEAR,
            identity: None,
            ack_floor: RequestSeq::from_u64(0),
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
        })?;
        match response.body {
            ResponseBody::ExpiryPersisted { removed } => Ok(removed),
            _ => Err(ClientError::Internal("unexpected persist body".to_owned())),
        }
    }

    /// Reads a key's expiry (`None` when absent).
    ///
    /// # Errors
    ///
    /// Returns [`ClientError`] on transport/routing failure.
    pub fn get_expiry(&self, key: &Key) -> Result<Option<kivi_types::Expiry>, ClientError> {
        use kivi_protocol::{Opcode, ResponseBody};
        let response = self.execute(key, Opcode::GetExpiry, || kivi_protocol::Request {
            contract: kivi_types::ReadContract::Latest,
            namespace: self.shared.namespace,
            opcode: Opcode::GetExpiry,
            hint: None,
            key: key.as_bytes().to_vec(),
            value: None,
            delta: 0,
            expiry: 0,
            offset: 0,
            len: 0,
            condition: kivi_protocol::COND_ALWAYS,
            expiry_policy: kivi_protocol::EXPIRY_CLEAR,
            identity: None,
            ack_floor: RequestSeq::from_u64(0),
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
        })?;
        match response.body {
            ResponseBody::ExpiryAt(stamp) => Ok(Some(kivi_types::Expiry::at(
                kivi_types::WallTimestamp::from_micros(stamp),
            ))),
            ResponseBody::ExpiryNever => Ok(Some(kivi_types::Expiry::NEVER)),
            ResponseBody::Diagnostic(_) => Ok(None),
            _ => Err(ClientError::Internal(
                "unexpected get-expiry body".to_owned(),
            )),
        }
    }

    /// Reads a byte slice `[offset, offset + len)` clamped to the logical
    /// length (`None` when absent; possibly empty when past the end).
    ///
    /// # Errors
    ///
    /// Returns [`ClientError`] on transport/routing failure or wrong-type access.
    pub fn get_range(
        &self,
        key: &Key,
        offset: u64,
        len: u64,
    ) -> Result<Option<Bytes>, ClientError> {
        use kivi_protocol::{Opcode, ResponseBody};
        let response = self.execute(key, Opcode::GetRange, || kivi_protocol::Request {
            contract: kivi_types::ReadContract::Latest,
            namespace: self.shared.namespace,
            opcode: Opcode::GetRange,
            hint: None,
            key: key.as_bytes().to_vec(),
            value: None,
            delta: 0,
            expiry: 0,
            offset,
            len,
            condition: kivi_protocol::COND_ALWAYS,
            expiry_policy: kivi_protocol::EXPIRY_CLEAR,
            identity: None,
            ack_floor: RequestSeq::from_u64(0),
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
        })?;
        match response.body {
            ResponseBody::Value(value) => Ok(Some(Bytes::from(value))),
            ResponseBody::Diagnostic(_) => Ok(None),
            _ => Err(ClientError::Internal(
                "unexpected get-range body".to_owned(),
            )),
        }
    }

    /// Reports the logical byte length (`None` when absent). Chunked roots
    /// answer from metadata without reading chunk payloads.
    ///
    /// # Errors
    ///
    /// Returns [`ClientError`] on transport/routing failure or wrong-type access.
    pub fn bytes_length(&self, key: &Key) -> Result<Option<u64>, ClientError> {
        use kivi_protocol::{Opcode, ResponseBody};
        let response = self.execute(key, Opcode::BytesLength, || kivi_protocol::Request {
            contract: kivi_types::ReadContract::Latest,
            namespace: self.shared.namespace,
            opcode: Opcode::BytesLength,
            hint: None,
            key: key.as_bytes().to_vec(),
            value: None,
            delta: 0,
            expiry: 0,
            offset: 0,
            len: 0,
            condition: kivi_protocol::COND_ALWAYS,
            expiry_policy: kivi_protocol::EXPIRY_CLEAR,
            identity: None,
            ack_floor: RequestSeq::from_u64(0),
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
        })?;
        match response.body {
            ResponseBody::Length(len) => Ok(Some(len)),
            ResponseBody::Diagnostic(_) => Ok(None),
            _ => Err(ClientError::Internal(
                "unexpected bytes-length body".to_owned(),
            )),
        }
    }

    /// Conditionally stores bytes, atomically at the owning tablet.
    /// Returns `(applied, version)`: `applied` names whether the condition
    /// held, `version` the new version when applied.
    ///
    /// # Errors
    ///
    /// Returns [`ClientError`] on transport/routing failure.
    #[allow(clippy::needless_pass_by_value)]
    pub fn set_conditional(
        &self,
        key: &Key,
        value: Bytes,
        condition: kivi_state::SetCondition,
        expiry: kivi_state::ExpiryPolicy,
    ) -> Result<(bool, Option<u64>), ClientError> {
        use kivi_protocol::{Opcode, ResponseBody};
        let (condition_wire, policy_wire, stamp) = encode_conditional(condition, expiry);
        let response = self.execute(key, Opcode::SetConditional, || kivi_protocol::Request {
            contract: kivi_types::ReadContract::Latest,
            namespace: self.shared.namespace,
            opcode: Opcode::SetConditional,
            hint: None,
            key: key.as_bytes().to_vec(),
            value: Some(value.to_vec()),
            delta: 0,
            expiry: stamp,
            offset: 0,
            len: 0,
            condition: condition_wire,
            expiry_policy: policy_wire,
            identity: None,
            ack_floor: RequestSeq::from_u64(0),
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
        })?;
        match response.body {
            ResponseBody::ConditionalSet { applied, version } => {
                Ok((applied, applied.then_some(version)))
            }
            _ => Err(ClientError::Internal(
                "unexpected set-conditional body".to_owned(),
            )),
        }
    }

    /// Conditionally stores bytes under an explicitly named sequence of
    /// this session (resumption: same exactly-once contract as
    /// [`set_with_seq`](Self::set_with_seq)).
    ///
    /// # Errors
    ///
    /// Returns [`ClientError`] on transport/routing failure, an out-of-range
    /// sequence, or a server-side rejection.
    #[allow(clippy::needless_pass_by_value)]
    pub fn set_conditional_with_seq(
        &self,
        key: &Key,
        value: Bytes,
        condition: kivi_state::SetCondition,
        expiry: kivi_state::ExpiryPolicy,
        seq: RequestSeq,
    ) -> Result<(bool, Option<u64>), ClientError> {
        use kivi_protocol::{Opcode, ResponseBody};
        let (condition_wire, policy_wire, stamp) = encode_conditional(condition, expiry);
        let response = self.execute_with_seq(
            key,
            Opcode::SetConditional,
            || kivi_protocol::Request {
                contract: kivi_types::ReadContract::Latest,
                namespace: self.shared.namespace,
                opcode: Opcode::SetConditional,
                hint: None,
                key: key.as_bytes().to_vec(),
                value: Some(value.to_vec()),
                delta: 0,
                expiry: stamp,
                offset: 0,
                len: 0,
                condition: condition_wire,
                expiry_policy: policy_wire,
                identity: None,
                ack_floor: RequestSeq::from_u64(0),
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
            Some(seq),
        )?;
        match response.body {
            ResponseBody::ConditionalSet { applied, version } => {
                Ok((applied, applied.then_some(version)))
            }
            _ => Err(ClientError::Internal(
                "unexpected set-conditional body".to_owned(),
            )),
        }
    }

    /// Sends a keepalive ping, expecting a pong (connection health probe).
    ///
    /// # Errors
    ///
    /// Returns [`ClientError`] on transport failure.
    pub fn ping(&self, endpoint: &str) -> Result<(), ClientError> {
        use kivi_protocol::FrameKind;
        let conn = self.connection(endpoint, None)?;
        let id = self.next_id();
        let bytes = encode_frame(FrameKind::Ping, id.as_u64(), &[]);
        conn.writer
            .lock()
            .map_err(|_| ClientError::Io("connection lock poisoned".to_owned()))?
            .write_all(&bytes)
            .map_err(|error| ClientError::Io(error.to_string()))?;
        Ok(())
    }
}

/// A lease grant: the fencing token that authorizes the external resource
/// plus the grant's logical expiry. Present the token on every guarded
/// action; the resource MUST reject tokens older than the one it last saw
/// (time expiry alone never retires an old holder's authority).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LeaseGrant {
    /// Fencing token issued with this grant (strictly increasing).
    pub fencing: u64,
    /// Logical expiry (`WallTimestamp::MAX` = immortal until released).
    pub expires_at: WallTimestamp,
}

/// Lease inspection answer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LeaseInfo {
    /// Live holder's owner (`None` when free or expired).
    pub owner: Option<u64>,
    /// Live holder's fencing token (zero when no live holder).
    pub fencing: u64,
    /// Live holder's logical expiry (epoch when no live holder).
    pub expires_at: WallTimestamp,
    /// Next fencing token to be issued.
    pub next_fencing: u64,
}

/// Semaphore load answer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SemaphoreLoad {
    /// Total outstanding quantity.
    pub outstanding: u64,
    /// Total capacity.
    pub capacity: u64,
}

/// Bounded-counter read answer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BoundedValue {
    /// Current value.
    pub value: i64,
    /// Global capacity.
    pub capacity: u64,
    /// Locally owned escrow share lower bound.
    pub share_min: i64,
    /// Locally owned escrow share upper bound.
    pub share_max: i64,
}

/// One stream entry read from a shard.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StreamEntry {
    /// Shard-local offset.
    pub offset: u64,
    /// Partition key that routed this entry here.
    pub partition: Vec<u8>,
    /// Entry payload bytes.
    pub payload: Vec<u8>,
}

/// One stream-shard read page.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StreamPage {
    /// Entries in shard order.
    pub entries: Vec<StreamEntry>,
    /// Shard's next offset (exclusive upper bound; resume cursor).
    pub next_offset: u64,
}

impl NativeClient {
    /// Builds a zeroed native request for `key`/`opcode`: per-operation
    /// helpers set only their meaningful fields. Keeps call sites small
    /// and future-proof as the protocol grows new opcodes.
    pub(crate) fn request_base(
        &self,
        key: &Key,
        opcode: kivi_protocol::Opcode,
    ) -> kivi_protocol::Request {
        kivi_protocol::Request {
            contract: kivi_types::ReadContract::Latest,
            namespace: self.shared.namespace,
            opcode,
            hint: None,
            key: key.as_bytes().to_vec(),
            value: None,
            delta: 0,
            expiry: 0,
            offset: 0,
            len: 0,
            condition: kivi_protocol::COND_ALWAYS,
            expiry_policy: kivi_protocol::EXPIRY_CLEAR,
            identity: None,
            ack_floor: RequestSeq::from_u64(0),
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
        }
    }

    /// Adds `delta` to a commutative counter. The result is `Applied`
    /// (no ordinal): permutations of additions produce identical observable
    /// histories, which is what will make a future unordered fast path
    /// safe. For exact ordinals use [`counter_add`](Self::counter_add).
    ///
    /// # Errors
    ///
    /// Returns [`ClientError`] on transport/routing failure, wrong-type
    /// access, or counter overflow.
    pub fn commutative_add(&self, key: &Key, delta: i64) -> Result<(), ClientError> {
        use kivi_protocol::{Opcode, ResponseBody};
        let response = self.execute(key, Opcode::CommutativeAdd, || {
            let mut request = self.request_base(key, Opcode::CommutativeAdd);
            request.delta = delta;
            request
        })?;
        match response.body {
            ResponseBody::CommutativeApplied => Ok(()),
            _ => Err(ClientError::Internal(
                "unexpected commutative-add body".to_owned(),
            )),
        }
    }

    /// Reads a commutative counter's current sum (`None` when absent; a
    /// state query, not an ordinal).
    ///
    /// # Errors
    ///
    /// Returns [`ClientError`] on transport/routing failure or wrong-type access.
    pub fn commutative_get(&self, key: &Key) -> Result<Option<i64>, ClientError> {
        use kivi_protocol::{Opcode, ResponseBody};
        let response = self.execute(key, Opcode::CommutativeGet, || {
            self.request_base(key, Opcode::CommutativeGet)
        })?;
        match response.body {
            ResponseBody::CommutativeValue(value) => Ok(Some(value)),
            ResponseBody::Diagnostic(_) => Ok(None),
            _ => Err(ClientError::Internal(
                "unexpected commutative-get body".to_owned(),
            )),
        }
    }

    /// Creates a bounded counter with global `capacity` and a full local
    /// escrow share `[0, capacity]` on the serving tablet.
    ///
    /// # Errors
    ///
    /// Returns [`ClientError`] on transport/routing failure, when the key
    /// already holds live state, or when the capacity is unrepresentable.
    pub fn bounded_create(&self, key: &Key, capacity: u64) -> Result<u64, ClientError> {
        use kivi_protocol::{Opcode, ResponseBody};
        let response = self.execute(key, Opcode::BoundedCounterCreate, || {
            let mut request = self.request_base(key, Opcode::BoundedCounterCreate);
            request.capacity = capacity;
            request
        })?;
        match response.body {
            ResponseBody::Stored { version } => Ok(version),
            _ => Err(ClientError::Internal(
                "unexpected bounded-create body".to_owned(),
            )),
        }
    }

    /// Adds `delta` to a bounded counter within locally owned escrow
    /// rights, returning the new value and version. Fails without
    /// mutating when the result would leave owned rights (move rights
    /// first) or the global bound.
    ///
    /// # Errors
    ///
    /// Returns [`ClientError`] on transport/routing failure, wrong-type
    /// access, overflow, or [`ClientError::BoundedExceeded`].
    pub fn bounded_add(&self, key: &Key, delta: i64) -> Result<(i64, u64), ClientError> {
        use kivi_protocol::{Opcode, ResponseBody};
        let response = self.execute(key, Opcode::BoundedCounterAdd, || {
            let mut request = self.request_base(key, Opcode::BoundedCounterAdd);
            request.delta = delta;
            request
        })?;
        match response.body {
            ResponseBody::BoundedUpdated { value, version } => Ok((value, version)),
            _ => Err(ClientError::Internal(
                "unexpected bounded-add body".to_owned(),
            )),
        }
    }

    /// Reads a bounded counter's value, capacity, and owned share
    /// (`None` when absent).
    ///
    /// # Errors
    ///
    /// Returns [`ClientError`] on transport/routing failure or wrong-type access.
    pub fn bounded_get(&self, key: &Key) -> Result<Option<BoundedValue>, ClientError> {
        use kivi_protocol::{Opcode, ResponseBody};
        let response = self.execute(key, Opcode::BoundedCounterGet, || {
            self.request_base(key, Opcode::BoundedCounterGet)
        })?;
        match response.body {
            ResponseBody::BoundedValue {
                value,
                capacity,
                share_min,
                share_max,
            } => Ok(Some(BoundedValue {
                value,
                capacity,
                share_min,
                share_max,
            })),
            ResponseBody::Diagnostic(_) => Ok(None),
            _ => Err(ClientError::Internal(
                "unexpected bounded-get body".to_owned(),
            )),
        }
    }

    /// Narrows one bounded counter's escrow share to `[new_min, new_max]`
    /// (lone operations narrow only — widening pairs inside atomic
    /// transfers, see `OrderedClient` batch composition with
    /// [`kivi_state::plan_escrow_transfer`]).
    ///
    /// # Errors
    ///
    /// Returns [`ClientError`] on transport/routing failure, wrong-type
    /// access, or [`ClientError::BoundedExceeded`] (not a narrowing, the
    /// value uncovered, or outside capacity).
    pub fn escrow_narrow(&self, key: &Key, new_min: i64, new_max: i64) -> Result<u64, ClientError> {
        use kivi_protocol::{Opcode, ResponseBody};
        let response = self.execute(key, Opcode::EscrowTransfer, || {
            let mut request = self.request_base(key, Opcode::EscrowTransfer);
            request.share_min = new_min;
            request.share_max = new_max;
            request
        })?;
        match response.body {
            ResponseBody::Stored { version } => Ok(version),
            _ => Err(ClientError::Internal(
                "unexpected escrow-transfer body".to_owned(),
            )),
        }
    }

    /// Creates a semaphore with `capacity` total permits.
    ///
    /// # Errors
    ///
    /// Returns [`ClientError`] on transport/routing failure or when the key
    /// already holds live state.
    pub fn semaphore_create(&self, key: &Key, capacity: u64) -> Result<u64, ClientError> {
        use kivi_protocol::{Opcode, ResponseBody};
        let response = self.execute(key, Opcode::SemaphoreCreate, || {
            let mut request = self.request_base(key, Opcode::SemaphoreCreate);
            request.capacity = capacity;
            request
        })?;
        match response.body {
            ResponseBody::Stored { version } => Ok(version),
            _ => Err(ClientError::Internal(
                "unexpected semaphore-create body".to_owned(),
            )),
        }
    }

    /// Acquires `qty` permits under `permit`: idempotent by permit id (a
    /// retry with the same id returns success without acquiring twice —
    /// mint stable ids per acquisition attempt, e.g. via
    /// [`kivi_state::PermitId::derive`]).
    ///
    /// # Errors
    ///
    /// Returns [`ClientError`] on transport/routing failure, wrong-type
    /// access, unknown semaphore, or [`ClientError::SemaphoreExhausted`].
    pub fn semaphore_acquire(
        &self,
        key: &Key,
        permit: [u8; 16],
        owner: u64,
        qty: u64,
    ) -> Result<(), ClientError> {
        use kivi_protocol::{Opcode, ResponseBody};
        let response = self.execute(key, Opcode::SemaphoreAcquire, || {
            let mut request = self.request_base(key, Opcode::SemaphoreAcquire);
            request.permit = permit;
            request.owner = owner;
            request.qty = qty;
            request
        })?;
        match response.body {
            ResponseBody::SemaphoreAcquired => Ok(()),
            _ => Err(ClientError::Internal(
                "unexpected semaphore-acquire body".to_owned(),
            )),
        }
    }

    /// Releases the permits held under `permit`: idempotent (unknown or
    /// already-released ids report `false` and mint nothing).
    ///
    /// # Errors
    ///
    /// Returns [`ClientError`] on transport/routing failure or wrong-type access.
    pub fn semaphore_release(&self, key: &Key, permit: [u8; 16]) -> Result<bool, ClientError> {
        use kivi_protocol::{Opcode, ResponseBody};
        let response = self.execute(key, Opcode::SemaphoreRelease, || {
            let mut request = self.request_base(key, Opcode::SemaphoreRelease);
            request.permit = permit;
            request
        })?;
        match response.body {
            ResponseBody::SemaphoreReleased { released } => Ok(released),
            _ => Err(ClientError::Internal(
                "unexpected semaphore-release body".to_owned(),
            )),
        }
    }

    /// Inspects a semaphore's outstanding load and capacity.
    ///
    /// # Errors
    ///
    /// Returns [`ClientError`] on transport/routing failure, wrong-type
    /// access, or unknown semaphore.
    pub fn semaphore_inspect(&self, key: &Key) -> Result<SemaphoreLoad, ClientError> {
        use kivi_protocol::{Opcode, ResponseBody};
        let response = self.execute(key, Opcode::SemaphoreInspect, || {
            self.request_base(key, Opcode::SemaphoreInspect)
        })?;
        match response.body {
            ResponseBody::SemaphoreLoad {
                outstanding,
                capacity,
            } => Ok(SemaphoreLoad {
                outstanding,
                capacity,
            }),
            _ => Err(ClientError::Internal(
                "unexpected semaphore-inspect body".to_owned(),
            )),
        }
    }

    /// Acquires (or re-acquires after expiry) the lease for `owner` with a
    /// logical TTL in micros (`0` = immortal until released). Success
    /// issues a fresh fencing token: present it on every guarded external
    /// action, and require the resource to reject older tokens — that
    /// rejection (not the expiry) is what retires old holders.
    ///
    /// # Errors
    ///
    /// Returns [`ClientError`] on transport/routing failure, wrong-type
    /// access, or [`ClientError::LeaseConflict`] (live holder — renew
    /// instead of re-granting).
    pub fn lease_acquire(
        &self,
        key: &Key,
        owner: u64,
        ttl_micros: u64,
    ) -> Result<LeaseGrant, ClientError> {
        use kivi_protocol::{Opcode, ResponseBody};
        let response = self.execute(key, Opcode::LeaseAcquire, || {
            let mut request = self.request_base(key, Opcode::LeaseAcquire);
            request.owner = owner;
            request.ttl = ttl_micros;
            request
        })?;
        match response.body {
            ResponseBody::LeaseAcquired {
                fencing,
                expires_at,
            } => Ok(LeaseGrant {
                fencing,
                expires_at: WallTimestamp::from_micros(expires_at),
            }),
            _ => Err(ClientError::Internal(
                "unexpected lease-acquire body".to_owned(),
            )),
        }
    }

    /// Extends a live grant: requires the exact current fencing token. A
    /// stale token fails without mutating and can never revive a
    /// replaced or expired lease.
    ///
    /// # Errors
    ///
    /// Returns [`ClientError`] on transport/routing failure, wrong-type
    /// access, or [`ClientError::StaleFencing`].
    pub fn lease_renew(
        &self,
        key: &Key,
        owner: u64,
        fencing: u64,
        ttl_micros: u64,
    ) -> Result<LeaseGrant, ClientError> {
        use kivi_protocol::{Opcode, ResponseBody};
        let response = self.execute(key, Opcode::LeaseRenew, || {
            let mut request = self.request_base(key, Opcode::LeaseRenew);
            request.owner = owner;
            request.fencing = fencing;
            request.ttl = ttl_micros;
            request
        })?;
        match response.body {
            ResponseBody::LeaseRenewed {
                fencing,
                expires_at,
            } => Ok(LeaseGrant {
                fencing,
                expires_at: WallTimestamp::from_micros(expires_at),
            }),
            _ => Err(ClientError::Internal(
                "unexpected lease-renew body".to_owned(),
            )),
        }
    }

    /// Releases a live grant: requires the exact current fencing token. A
    /// stale token fails without mutating and can never free a newer
    /// holder's lease. Releasing a free lease reports `false`.
    ///
    /// # Errors
    ///
    /// Returns [`ClientError`] on transport/routing failure, wrong-type
    /// access, or [`ClientError::StaleFencing`].
    pub fn lease_release(&self, key: &Key, owner: u64, fencing: u64) -> Result<bool, ClientError> {
        use kivi_protocol::{Opcode, ResponseBody};
        let response = self.execute(key, Opcode::LeaseRelease, || {
            let mut request = self.request_base(key, Opcode::LeaseRelease);
            request.owner = owner;
            request.fencing = fencing;
            request
        })?;
        match response.body {
            ResponseBody::LeaseReleased { released } => Ok(released),
            _ => Err(ClientError::Internal(
                "unexpected lease-release body".to_owned(),
            )),
        }
    }

    /// Inspects a lease: the live holder at now (`None` when free or
    /// expired) plus the next fencing token to be issued. Never mutates.
    ///
    /// # Errors
    ///
    /// Returns [`ClientError`] on transport/routing failure or wrong-type access.
    pub fn lease_inspect(&self, key: &Key) -> Result<LeaseInfo, ClientError> {
        use kivi_protocol::{Opcode, ResponseBody};
        let response = self.execute(key, Opcode::LeaseInspect, || {
            self.request_base(key, Opcode::LeaseInspect)
        })?;
        match response.body {
            ResponseBody::LeaseInfo {
                owner,
                fencing,
                expires_at,
                next_fencing,
            } => Ok(LeaseInfo {
                owner,
                fencing,
                expires_at: WallTimestamp::from_micros(expires_at),
                next_fencing,
            }),
            _ => Err(ClientError::Internal(
                "unexpected lease-inspect body".to_owned(),
            )),
        }
    }

    /// Creates one shard of a sharded stream. Shards are independent
    /// ordered logs: total order within the shard, none across shards.
    /// Callers own shard keys (any bytes; distinct keys hash to distinct
    /// tablets) and map partitions to shards with
    /// [`OrderedClient::stream_shard_for`] (stable, deterministic).
    ///
    /// # Errors
    ///
    /// Returns [`ClientError`] on transport/routing failure or when the key
    /// already holds live state.
    pub fn stream_create(
        &self,
        key: &Key,
        stream: [u8; 16],
        shard: u32,
    ) -> Result<(), ClientError> {
        use kivi_protocol::{Opcode, ResponseBody};
        let response = self.execute(key, Opcode::StreamCreate, || {
            let mut request = self.request_base(key, Opcode::StreamCreate);
            request.stream = stream;
            request.shard = shard;
            request
        })?;
        match response.body {
            ResponseBody::StreamCreated => Ok(()),
            _ => Err(ClientError::Internal(
                "unexpected stream-create body".to_owned(),
            )),
        }
    }

    /// Appends one entry to a stream shard, assigning the shard's next
    /// offset. The `partition` key is retained for audit (it is what
    /// routed the entry here); the offset is shard-local and monotonic,
    /// never global.
    ///
    /// # Errors
    ///
    /// Returns [`ClientError`] on transport/routing failure, wrong-type
    /// access, unknown shard, or [`ClientError::StreamFull`].
    pub fn stream_append(
        &self,
        key: &Key,
        partition: &[u8],
        payload: &[u8],
    ) -> Result<u64, ClientError> {
        use kivi_protocol::{Opcode, ResponseBody};
        let response = self.execute(key, Opcode::StreamAppend, || {
            let mut request = self.request_base(key, Opcode::StreamAppend);
            request.partition = partition.to_vec();
            request.value = Some(payload.to_vec());
            request
        })?;
        match response.body {
            ResponseBody::StreamAppended { offset } => Ok(offset),
            _ => Err(ClientError::Internal(
                "unexpected stream-append body".to_owned(),
            )),
        }
    }

    /// Reads one shard's entries from `from_offset` (inclusive), up to
    /// `max_entries`. Implies nothing about other shards: merged
    /// cross-shard reads are the caller's explicit composition, never a
    /// fabricated global order.
    ///
    /// # Errors
    ///
    /// Returns [`ClientError`] on transport/routing failure or wrong-type access.
    pub fn stream_read(
        &self,
        key: &Key,
        from_offset: u64,
        max_entries: u32,
    ) -> Result<StreamPage, ClientError> {
        use kivi_protocol::{Opcode, ResponseBody};
        let response = self.execute(key, Opcode::StreamRead, || {
            let mut request = self.request_base(key, Opcode::StreamRead);
            request.offset = from_offset;
            request.len = u64::from(max_entries);
            request
        })?;
        match response.body {
            ResponseBody::StreamEntries {
                entries,
                next_offset,
            } => Ok(StreamPage {
                entries: entries
                    .into_iter()
                    .map(|entry| StreamEntry {
                        offset: entry.offset,
                        partition: entry.partition,
                        payload: entry.payload,
                    })
                    .collect(),
                next_offset,
            }),
            _ => Err(ClientError::Internal(
                "unexpected stream-read body".to_owned(),
            )),
        }
    }

    /// Trims a shard's retained prefix through `through_offset`
    /// (inclusive). The cursor never rewinds: trimmed offsets are gone,
    /// not reusable.
    ///
    /// # Errors
    ///
    /// Returns [`ClientError`] on transport/routing failure or wrong-type access.
    pub fn stream_trim(&self, key: &Key, through_offset: u64) -> Result<u64, ClientError> {
        use kivi_protocol::{Opcode, ResponseBody};
        let response = self.execute(key, Opcode::StreamTrim, || {
            let mut request = self.request_base(key, Opcode::StreamTrim);
            request.offset = through_offset;
            request
        })?;
        match response.body {
            ResponseBody::StreamTrimmed { removed } => Ok(removed),
            _ => Err(ClientError::Internal(
                "unexpected stream-trim body".to_owned(),
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_client() -> NativeClient {
        NativeClient::new(ClientConfig {
            seeds: vec!["127.0.0.1:1".to_owned()],
            ..ClientConfig::default()
        })
        .expect("client builds without dialing")
    }

    #[test]
    fn sessions_carry_distinct_csprng_identities() {
        let first = test_client();
        let second = test_client();
        assert_ne!(
            first.shared.session, second.shared.session,
            "sessions must differ (CSPRNG collision would break dedup isolation)"
        );
    }

    #[test]
    fn floor_advances_only_over_consecutive_completions() {
        let client = test_client();
        // Allocate 1, 2, 3; complete out of order: the floor must wait.
        let (_, floor) = client.alloc_seq().expect("seq 1");
        assert_eq!(floor.as_u64(), 0);
        let (_, _) = client.alloc_seq().expect("seq 2");
        let (_, _) = client.alloc_seq().expect("seq 3");
        client.complete_seq(3);
        client.complete_seq(1);
        let (_, floor) = client.alloc_seq().expect("seq 4");
        // 2 still in flight: floor stays at 1 even though 3 completed.
        assert_eq!(floor.as_u64(), 1);
        client.complete_seq(2);
        let (_, floor) = client.alloc_seq().expect("seq 5");
        // 2 completed, but 4 is still in flight: floor stops at 3.
        assert_eq!(floor.as_u64(), 3);
    }

    #[test]
    fn abandonment_completes_so_floors_never_stall() {
        let client = test_client();
        let _ = client.alloc_seq().expect("seq 1");
        // Abandoned (timeout give-up): completing it lets the floor pass.
        client.complete_seq(1);
        let (_, floor) = client.alloc_seq().expect("seq 2");
        assert_eq!(floor.as_u64(), 1);
    }

    #[test]
    fn adopted_sequences_reserve_space_without_colliding() {
        let client = test_client();
        let (identity, _) = client
            .adopt_seq(RequestSeq::from_u64(41))
            .expect("adopt 41");
        assert_eq!(identity.seq().as_u64(), 41);
        // Automatic allocation continues past the adopted sequence.
        let (next, _) = client.alloc_seq().expect("next");
        assert_eq!(next.seq().as_u64(), 42);
    }

    #[test]
    fn adopted_sequences_reject_the_reserved_endpoints() {
        let client = test_client();
        assert!(client.adopt_seq(RequestSeq::from_u64(0)).is_err());
        assert!(client.adopt_seq(RequestSeq::from_u64(u64::MAX)).is_err());
    }

    #[test]
    fn resumed_sessions_keep_their_identity() {
        let first = test_client();
        let resumed = NativeClient::new(ClientConfig {
            seeds: vec!["127.0.0.1:1".to_owned()],
            session: Some(first.session()),
            ..ClientConfig::default()
        })
        .expect("resume builds without dialing");
        assert_eq!(resumed.session(), first.session());
    }

    /// Regression for the ~30s durable tail: when the server side of a
    /// connection dies mid-request (close, violation kill, crash), the
    /// in-flight waiter must fail at once and redial — never burn the full
    /// request timeout first. Deterministic stub: completes the handshake,
    /// reads exactly one request frame, then closes silently with no
    /// response. With `request_timeout` at 10s, stranded code surfaces
    /// only after ~10s; fixed code surfaces `Io` (reader ended) in
    /// milliseconds and the redial then fails fast on the closed listener.
    #[test]
    fn dead_connection_fails_inflight_fast() {
        use std::io::{Read, Write};
        use std::net::TcpListener;
        use std::time::{Duration, Instant};

        let listener = TcpListener::bind("127.0.0.1:0").expect("stub binds");
        let addr = listener.local_addr().expect("stub addr");
        let stub = std::thread::spawn(move || {
            let (mut socket, _) = listener.accept().expect("accept");
            // No further accepts: the client's redial must fail fast with
            // connection-refused instead of hanging in a second handshake.
            drop(listener);
            socket.set_read_timeout(Some(Duration::from_secs(10))).ok();
            let mut reader = FrameReader::new(DEFAULT_MAX_FRAME);
            let mut chunk = vec![0u8; 4096];
            let hello = loop {
                let count = socket.read(&mut chunk).expect("hello bytes");
                assert!(count > 0, "client must send a hello");
                let frames = reader.push(&chunk[..count]).expect("hello parses");
                if let Some(frame) = frames.into_iter().next() {
                    break frame;
                }
            };
            assert_eq!(hello.kind, FrameKind::ClientHello);
            let reply = ServerHello {
                major: kivi_protocol::PROTOCOL_MAJOR,
                minor: kivi_protocol::PROTOCOL_MINOR,
                caps: Capabilities::BASE_V1 | Capabilities::DURABLE_MUTATION_DEDUP,
                cluster: kivi_types::ClusterId::from_u128(1),
                node: kivi_types::NodeId::from_u64(1),
                incarnation: kivi_types::NodeIncarnation::from_u64(1),
                worker: WorkerId::from_u64(0),
                dir_version: kivi_tablet::DirectoryVersion::from_u64(1),
                max_frame: u32::try_from(DEFAULT_MAX_FRAME).expect("frame bound fits u32"),
                endpoints: Vec::new(),
            };
            socket
                .write_all(&encode_frame(FrameKind::ServerHello, 0, &reply.encode()))
                .expect("hello reply");
            loop {
                let count = socket.read(&mut chunk).expect("request bytes");
                assert!(count > 0, "client must send its request");
                let frames = reader.push(&chunk[..count]).expect("request parses");
                if frames
                    .into_iter()
                    .any(|frame| frame.kind == FrameKind::Request)
                {
                    break;
                }
            }
            // Die silently: close with the request unanswered.
        });
        let client = NativeClient::new(ClientConfig {
            seeds: vec![addr.to_string()],
            request_timeout: Duration::from_secs(10),
            connect_timeout: Duration::from_secs(2),
            dial_attempts: 1,
            ..ClientConfig::default()
        })
        .expect("client builds");
        let started = Instant::now();
        let outcome = client.set(&Key::from("k"), bytes::Bytes::from_static(b"v"));
        let elapsed = started.elapsed();
        stub.join().expect("stub exits");
        // The waiter must notice the dead connection promptly: well under
        // the 10s timeout it would otherwise burn. The redial then fails
        // fast (refused), so the surfaced error is transport `Io` — never
        // a `Timeout` that pretends the server is merely slow.
        assert!(
            elapsed < Duration::from_secs(5),
            "in-flight waiter stranded {elapsed:?} on a dead connection"
        );
        assert!(
            matches!(outcome, Err(ClientError::Io(_))),
            "expected fast transport Io, got {outcome:?}"
        );
    }

    /// Reads frames until the handshake completes, then until one
    /// request arrives. Returns the request id, the decoded request,
    /// and the now-established socket for the reply.
    fn stub_request(socket: &mut std::net::TcpStream, node: u64) -> (u64, kivi_protocol::Request) {
        use std::io::{Read, Write};

        let mut reader = FrameReader::new(DEFAULT_MAX_FRAME);
        let mut chunk = vec![0u8; 4096];
        let hello = loop {
            let count = socket.read(&mut chunk).expect("hello bytes");
            assert!(count > 0, "client must send a hello");
            let frames = reader.push(&chunk[..count]).expect("hello parses");
            if let Some(frame) = frames.into_iter().next() {
                break frame;
            }
        };
        assert_eq!(hello.kind, FrameKind::ClientHello);
        let reply = ServerHello {
            major: kivi_protocol::PROTOCOL_MAJOR,
            minor: kivi_protocol::PROTOCOL_MINOR,
            caps: Capabilities::BASE_V1 | Capabilities::DURABLE_MUTATION_DEDUP,
            cluster: kivi_types::ClusterId::from_u128(1),
            node: kivi_types::NodeId::from_u64(node),
            incarnation: kivi_types::NodeIncarnation::from_u64(1),
            worker: WorkerId::from_u64(0),
            dir_version: kivi_tablet::DirectoryVersion::from_u64(1),
            max_frame: u32::try_from(DEFAULT_MAX_FRAME).expect("frame bound fits u32"),
            endpoints: Vec::new(),
        };
        socket
            .write_all(&encode_frame(FrameKind::ServerHello, 0, &reply.encode()))
            .expect("hello reply");
        loop {
            let count = socket.read(&mut chunk).expect("request bytes");
            assert!(count > 0, "client must send its request");
            let frames = reader.push(&chunk[..count]).expect("request parses");
            for frame in frames {
                if frame.kind == FrameKind::Request {
                    let request =
                        kivi_protocol::Request::decode(&frame.payload).expect("request decodes");
                    return (frame.request_id, request);
                }
            }
        }
    }

    /// Leader redirect preserves the mutation identity (task O): a
    /// follower answers `StaleRoute` with the leader's endpoint, and the
    /// retry at the leader carries the identical session, sequence, and
    /// ack floor — bounded by the redirect budget, never an infinite
    /// loop, never a reallocated identity.
    #[test]
    fn redirect_retry_preserves_mutation_identity() {
        use std::io::Write;
        use std::net::TcpListener;
        use std::sync::mpsc;

        use kivi_protocol::RedirectInfo;

        // Completer stub (the "leader"): answers `Stored` and reports the
        // identity it served.
        let leader_listener = TcpListener::bind("127.0.0.1:0").expect("binds");
        let leader_addr = leader_listener.local_addr().expect("addr");
        let (leader_seen_tx, leader_seen_rx) = mpsc::channel();
        let leader = std::thread::spawn(move || {
            let (mut socket, _) = leader_listener.accept().expect("accept");
            let (id, request) = stub_request(&mut socket, 1);
            let identity = request.identity.expect("mutations carry identity");
            leader_seen_tx
                .send((identity, request.ack_floor))
                .expect("report");
            let response = Response {
                proof: None,
                status: Status::Ok,
                body: ResponseBody::Stored { version: 7 },
            };
            socket
                .write_all(&encode_frame(
                    FrameKind::Response,
                    id,
                    &response.encode(request.opcode),
                ))
                .expect("reply");
        });
        // Redirector stub (the "follower"): answers `StaleRoute` at the
        // leader and reports the identity it saw.
        let follower_listener = TcpListener::bind("127.0.0.1:0").expect("binds");
        let follower_addr = follower_listener.local_addr().expect("addr");
        let (follower_seen_tx, follower_seen_rx) = mpsc::channel();
        let follower = std::thread::spawn(move || {
            let (mut socket, _) = follower_listener.accept().expect("accept");
            let (id, request) = stub_request(&mut socket, 2);
            let identity = request.identity.expect("mutations carry identity");
            follower_seen_tx
                .send((identity, request.ack_floor))
                .expect("report");
            let info = RedirectInfo::new(
                kivi_tablet::DirectoryVersion::from_u64(1),
                kivi_types::TabletId::from_u64(9),
                kivi_types::TabletEpoch::INITIAL,
                WorkerId::from_u64(0),
                leader_addr.to_string(),
                kivi_tablet::PartitionRange::Hash(
                    kivi_tablet::HashPrefix::new(0, 0).expect("root range valid"),
                ),
            )
            .expect("redirect builds");
            let response = Response {
                proof: None,
                status: Status::StaleRoute,
                body: ResponseBody::Redirect(info),
            };
            socket
                .write_all(&encode_frame(
                    FrameKind::Response,
                    id,
                    &response.encode(request.opcode),
                ))
                .expect("reply");
        });
        // Seeded at the follower: the write transparently follows the
        // redirect with the same identity, inside the redirect budget.
        let client = NativeClient::new(ClientConfig {
            seeds: vec![follower_addr.to_string()],
            ..ClientConfig::default()
        })
        .expect("client builds");
        client
            .set(&Key::from("k"), bytes::Bytes::from_static(b"v"))
            .expect("redirected write succeeds");
        let (first, first_floor) = follower_seen_rx
            .recv_timeout(std::time::Duration::from_secs(10))
            .expect("follower saw the attempt");
        let (second, second_floor) = leader_seen_rx
            .recv_timeout(std::time::Duration::from_secs(10))
            .expect("leader saw the retry");
        assert_eq!(
            (first, first_floor),
            (second, second_floor),
            "redirect retries must preserve session, sequence, and ack floor"
        );
        follower.join().expect("follower exits");
        leader.join().expect("leader exits");
    }
}
