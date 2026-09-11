//! Native TCP client library for Kivi (blocking, no async runtime).
//!
//! [`NativeClient`] speaks the native protocol over persistent per-worker
//! TCP connections with a sparse route cache: the first request to a range
//! goes to the seed, learns the authority from the redirect, and routes
//! directly afterwards. One logical request identity is preserved across
//! route retries — redirects are transport retries, never logical replays
//! (the design already anticipates future `SessionId + RequestSeq`
//! server-side dedup: identities stay stable across retries today).
//!
//! Threading model: the client is cheaply clonable (`Arc` inside). Each
//! connection has one background reader thread dispatching responses by
//! request id; writers serialize per connection under a mutex. Blocking
//! calls wait on one-shot rendezvous channels with an explicit timeout.

pub mod route;

use core::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use core::time::Duration;
use std::collections::HashMap;
use std::io::{Read, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

use bytes::Bytes;
use crossbeam_channel::{Sender, bounded};
use kivi_protocol::{
    Capabilities, ClientHello, DEFAULT_MAX_FRAME, Frame, FrameKind, FrameReader, ProtocolError,
    RequestId, Response, ResponseBody, ServerHello, Status, encode_frame,
};
use kivi_state::{Key, PartitionHasher};
use kivi_types::{NamespaceId, UnixMicros, WorkerId};

pub use route::{RouteCache, RouteEntry};

/// Connection establishment timeout default.
pub const DEFAULT_CONNECT_TIMEOUT: Duration = Duration::from_secs(5);
/// Per-request response wait default.
pub const DEFAULT_REQUEST_TIMEOUT: Duration = Duration::from_secs(30);
/// Write timeout default (no read timeout: idle pooled connections are healthy).
pub const DEFAULT_WRITE_TIMEOUT: Duration = Duration::from_secs(30);
/// Route retry budget default (redirects before surfacing).
pub const DEFAULT_MAX_REDIRECTS: usize = 8;
/// In-flight requests per connection default (fast `Overloaded` past this).
pub const DEFAULT_MAX_PENDING: usize = 128;

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
    }
}

/// One pooled connection: a mutex-serialized writer plus a background reader
/// dispatching responses to per-request rendezvous channels by id.
#[derive(Debug)]
struct Connection {
    worker: WorkerId,
    endpoint: String,
    writer: Mutex<TcpStream>,
    pending: Arc<Mutex<HashMap<u64, Sender<Response>>>>,
    max_pending: usize,
    request_timeout: Duration,
    reader_alive: Arc<AtomicBool>,
    _reader: JoinHandle<()>,
}

impl Connection {
    /// Dials, handshakes, and spawns the reader thread. Validates the hello
    /// (worker identity when expected, base capability present).
    fn dial(
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
        let reader_alive = Arc::new(AtomicBool::new(true));
        let alive = Arc::clone(&reader_alive);
        let max_frame = config.max_frame.min(hello.max_frame as usize);
        let reader = std::thread::Builder::new()
            .name(format!("kivi-client-reader-{endpoint}"))
            .spawn(move || {
                reader_loop(reader_stream, max_frame, dispatch, alive);
            })
            .map_err(|error| ClientError::Io(error.to_string()))?;
        Ok(Self {
            worker: hello.worker,
            endpoint: endpoint.to_owned(),
            writer: Mutex::new(stream),
            pending,
            max_pending: config.max_pending,
            request_timeout: config.request_timeout,
            reader_alive,
            _reader: reader,
        })
    }

    /// Sends one request frame and waits for its response (matched by id). A
    /// wait that outlives its reader thread surfaces as `Io` (connection
    /// lost, never a slow server): the reader only exits on EOF or fatal
    /// framing failure, after which no response can arrive.
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
}

/// Background reader: frames → responses → per-id rendezvous. Ends on EOF,
/// timeout-free; pending waiters fail via their own timeouts or the next
/// connection teardown. Malformed frames end the connection (pending
/// responses fail when their waiters time out or the pool drops the conn).
/// Takes the table by value: the reader thread owns its dispatch handle.
/// Clears `alive` on exit so waits can distinguish a dead connection
/// (redial) from a slow server (timeout).
#[allow(clippy::needless_pass_by_value)]
fn reader_loop(
    mut stream: TcpStream,
    max_frame: usize,
    pending: Arc<Mutex<HashMap<u64, Sender<Response>>>>,
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
            if frame.kind != FrameKind::Response {
                continue;
            }
            let Ok((response, _)) = Response::decode(&frame.payload) else {
                continue;
            };
            let waiter = pending
                .lock()
                .ok()
                .and_then(|mut table| table.remove(&frame.request_id));
            if let Some(tx) = waiter {
                let _ = tx.try_send(response);
            }
        }
    }
    alive.store(false, Ordering::SeqCst);
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
    seeds: Vec<String>,
    routes: route::RouteCache,
    pool: Mutex<HashMap<String, Arc<Connection>>>,
    id_counter: AtomicU64,
    requests: AtomicU64,
    redirects: AtomicU64,
    errors: AtomicU64,
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
    ///
    /// # Errors
    ///
    /// Returns [`ClientError::Io`] when no seed address is configured.
    pub fn new(config: ClientConfig) -> Result<Self, ClientError> {
        if config.seeds.is_empty() {
            return Err(ClientError::Io("no seed addresses configured".to_owned()));
        }
        Ok(Self {
            shared: Arc::new(Shared {
                namespace: config.namespace,
                max_frame: u32::try_from(config.max_frame).unwrap_or(u32::MAX),
                max_redirects: config.max_redirects,
                max_pending: config.max_pending,
                connect_timeout: config.connect_timeout,
                request_timeout: config.request_timeout,
                write_timeout: config.write_timeout,
                seeds: config.seeds,
                routes: route::RouteCache::new(),
                pool: Mutex::new(HashMap::new()),
                id_counter: AtomicU64::new(1),
                requests: AtomicU64::new(0),
                redirects: AtomicU64::new(0),
                errors: AtomicU64::new(0),
            }),
        })
    }

    /// Returns cumulative client counters.
    #[must_use]
    pub fn stats(&self) -> ClientStats {
        ClientStats {
            requests: self.shared.requests.load(Ordering::Relaxed),
            redirects: self.shared.redirects.load(Ordering::Relaxed),
            errors: self.shared.errors.load(Ordering::Relaxed),
        }
    }

    /// Returns the number of cached sparse routes (introspection for tests).
    #[must_use]
    pub fn cached_routes(&self) -> usize {
        self.shared.routes.len()
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
        };
        let conn = Arc::new(Connection::dial(endpoint, expect_worker, &config)?);
        self.shared
            .pool
            .lock()
            .map_err(|_| ClientError::Io("connection pool lock poisoned".to_owned()))?
            .insert(endpoint.to_owned(), Arc::clone(&conn));
        Ok(conn)
    }

    /// Executes one request payload against the authority for `key`,
    /// following redirects with the same request identity throughout.
    fn execute(
        &self,
        key: &Key,
        build: impl Fn() -> kivi_protocol::Request,
    ) -> Result<Response, ClientError> {
        use kivi_protocol::Status;
        let hash = PartitionHasher::V1
            .hash(self.shared.namespace, key.as_bytes())
            .ok_or(ClientError::Internal(
                "partition hash unsupported".to_owned(),
            ))?;
        let id = self.next_id();
        let mut redirects = 0usize;
        let mut reconnects = 0usize;
        self.shared.requests.fetch_add(1, Ordering::Relaxed);
        loop {
            if redirects > self.shared.max_redirects {
                self.shared.errors.fetch_add(1, Ordering::Relaxed);
                return Err(ClientError::TooManyRedirects);
            }
            // Route: cached authority, else the seed with no hint.
            let (endpoint, hint) = match self.shared.routes.lookup(hash) {
                Some(route) => (route.endpoint.clone(), Some(route.hint())),
                None => (self.shared.seeds[0].clone(), None),
            };
            let expect = hint.map(|hint| hint.worker);
            let conn = self.connection(&endpoint, expect)?;
            let mut request = build();
            request.hint = hint;
            let response = match conn.round_trip(id, &request.encode()) {
                Ok(response) => response,
                // Transport failure (including a wait that outlived its
                // reader): drop the pooled connection and redial. The request
                // identity is preserved; servers track no per-connection
                // state that a redial could confuse. Retried writes may
                // re-execute a request the dead server already ran — safe for
                // idempotent ops; non-idempotent ops need the future
                // `SessionId + RequestSeq` dedup before relying on this.
                Err(ClientError::Io(_)) if reconnects < 2 => {
                    reconnects += 1;
                    self.drop_connection(&endpoint);
                    continue;
                }
                Err(error) => {
                    self.shared.errors.fetch_add(1, Ordering::Relaxed);
                    return Err(error);
                }
            };
            match response.status {
                Status::Ok | Status::NotFound => return Ok(response),
                Status::StaleRoute | Status::NotLocal => {
                    if let ResponseBody::Redirect(info) = &response.body {
                        self.shared.routes.insert(RouteEntry::from_redirect(info));
                    }
                    redirects += 1;
                    self.shared.redirects.fetch_add(1, Ordering::Relaxed);
                }
                Status::Overloaded => {
                    redirects += 1;
                    self.shared.redirects.fetch_add(1, Ordering::Relaxed);
                }
                other => {
                    self.shared.errors.fetch_add(1, Ordering::Relaxed);
                    return Err(status_error(other, &response.body));
                }
            }
        }
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
        let response = self.execute(key, || kivi_protocol::Request {
            namespace: self.shared.namespace,
            opcode: Opcode::Get,
            hint: None,
            key: key.as_bytes().to_vec(),
            value: None,
            delta: 0,
            expiry: 0,
        })?;
        match response.body {
            ResponseBody::Value(value) => Ok(Some(Bytes::from(value))),
            ResponseBody::Diagnostic(_) => Ok(None),
            _ => Err(ClientError::Internal("unexpected get body".to_owned())),
        }
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
        let response = self.execute(key, || kivi_protocol::Request {
            namespace: self.shared.namespace,
            opcode: Opcode::Set,
            hint: None,
            key: key.as_bytes().to_vec(),
            value: Some(value.to_vec()),
            delta: 0,
            expiry: 0,
        })?;
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
        let response = self.execute(key, || kivi_protocol::Request {
            namespace: self.shared.namespace,
            opcode: Opcode::Delete,
            hint: None,
            key: key.as_bytes().to_vec(),
            value: None,
            delta: 0,
            expiry: 0,
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
        let response = self.execute(key, || kivi_protocol::Request {
            namespace: self.shared.namespace,
            opcode: Opcode::Exists,
            hint: None,
            key: key.as_bytes().to_vec(),
            value: None,
            delta: 0,
            expiry: 0,
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
        let response = self.execute(key, || kivi_protocol::Request {
            namespace: self.shared.namespace,
            opcode: Opcode::CounterGet,
            hint: None,
            key: key.as_bytes().to_vec(),
            value: None,
            delta: 0,
            expiry: 0,
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
        let response = self.execute(key, || kivi_protocol::Request {
            namespace: self.shared.namespace,
            opcode: Opcode::CounterAdd,
            hint: None,
            key: key.as_bytes().to_vec(),
            value: None,
            delta,
            expiry: 0,
        })?;
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
    pub fn expire_at(&self, key: &Key, expires_at: UnixMicros) -> Result<bool, ClientError> {
        use kivi_protocol::{Opcode, ResponseBody};
        let response = self.execute(key, || kivi_protocol::Request {
            namespace: self.shared.namespace,
            opcode: Opcode::ExpireAt,
            hint: None,
            key: key.as_bytes().to_vec(),
            value: None,
            delta: 0,
            expiry: expires_at.as_micros(),
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
        let response = self.execute(key, || kivi_protocol::Request {
            namespace: self.shared.namespace,
            opcode: Opcode::PersistExpiry,
            hint: None,
            key: key.as_bytes().to_vec(),
            value: None,
            delta: 0,
            expiry: 0,
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
        let response = self.execute(key, || kivi_protocol::Request {
            namespace: self.shared.namespace,
            opcode: Opcode::GetExpiry,
            hint: None,
            key: key.as_bytes().to_vec(),
            value: None,
            delta: 0,
            expiry: 0,
        })?;
        match response.body {
            ResponseBody::ExpiryAt(stamp) => Ok(Some(kivi_types::Expiry::at(
                kivi_types::UnixMicros::from_micros(stamp),
            ))),
            ResponseBody::ExpiryNever => Ok(Some(kivi_types::Expiry::NEVER)),
            ResponseBody::Diagnostic(_) => Ok(None),
            _ => Err(ClientError::Internal(
                "unexpected get-expiry body".to_owned(),
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
