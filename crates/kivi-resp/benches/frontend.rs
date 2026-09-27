//! RESP frontend cost, isolated from the engine and from the socket.
//!
//! The end-to-end number mixes four things: the frontend, the socket, the
//! engine rendezvous, and the tablet store. This bench answers the question
//! an end-to-end number cannot: how much of a request is the frontend, and
//! how much of that is allocation.
//!
//! The executor here is a hash map and nothing else — no thread, no channel,
//! no rendezvous — so every nanosecond below is parse, dispatch, key
//! handling, reply construction, and the write that would carry them. The gap
//! between this and the server's end-to-end number is the part that is *not*
//! the frontend.
//!
//! Allocations are measured with `dhat` rather than a hand-written counting
//! allocator, because this workspace denies hand-written `unsafe` and a
//! `GlobalAlloc` implementation cannot be written without it. Run with
//! `--features alloc-profile` to emit the allocation report.

use std::collections::HashMap;
use std::sync::Mutex;

use bytes::Bytes;
use divan::Bencher;
use kivi_resp::{ConnConfig, ExecuteError, Executor, RespConnection};
use kivi_state::{Operation, OperationResult};

// ------------------------------------------------------------------ executor

/// A map and nothing else: the frontend's entire dependency.
///
/// `Mutex` rather than `RefCell` because [`Executor`] requires `Send + Sync`;
/// uncontended, it is two atomic operations the map lookup already dominates.
struct MapExec {
    store: Mutex<HashMap<Vec<u8>, Bytes>>,
    now: kivi_types::WallTimestamp,
}

impl MapExec {
    fn new() -> Self {
        Self {
            store: Mutex::new(HashMap::new()),
            now: kivi_types::WallTimestamp::from_micros(1_700_000_000_000_000),
        }
    }

    /// Every arm is infallible by construction: the map is the whole backend,
    /// and a key it does not hold is a miss, not a failure. Returning a
    /// `Result` the bench never exercises would only be noise.
    #[allow(clippy::unnecessary_wraps)]
    fn one(&self, op: &Operation) -> Result<OperationResult, ExecuteError> {
        let mut store = self.store.lock().expect("bench map");
        match op {
            Operation::Get { key } => {
                Ok(OperationResult::Value(store.get(key.as_bytes()).cloned()))
            }
            Operation::SetConditional { key, value, .. } => {
                store.insert(key.as_bytes().to_vec(), value.clone());
                Ok(OperationResult::ConditionalSet {
                    applied: true,
                    version: None,
                })
            }
            Operation::Exists { key } => {
                Ok(OperationResult::Exists(store.contains_key(key.as_bytes())))
            }
            Operation::Delete { key } => Ok(OperationResult::Deleted {
                existed: store.remove(key.as_bytes()).is_some(),
            }),
            Operation::BytesLength { key } => Ok(OperationResult::Length(
                store
                    .get(key.as_bytes())
                    .map(|value| u64::try_from(value.len()).unwrap_or(0)),
            )),
            _ => Ok(OperationResult::Exists(false)),
        }
    }
}

impl Executor for MapExec {
    fn execute(&self, op: &Operation) -> Result<OperationResult, ExecuteError> {
        self.one(op)
    }

    fn now(&self) -> kivi_types::WallTimestamp {
        self.now
    }
}

/// The same map, with the batched path the server actually uses. Separate type
/// so a default-method call can never silently stand in for the real one.
struct BatchedExec(MapExec);

impl BatchedExec {
    fn new() -> Self {
        Self(MapExec::new())
    }
}

impl Executor for BatchedExec {
    fn execute(&self, op: &Operation) -> Result<OperationResult, ExecuteError> {
        self.0.one(op)
    }

    fn execute_batch(&self, ops: &[Operation]) -> Vec<Result<OperationResult, ExecuteError>> {
        let mut out = Vec::with_capacity(ops.len());
        for op in ops {
            out.push(self.0.one(op));
        }
        out
    }

    fn now(&self) -> kivi_types::WallTimestamp {
        self.0.now
    }
}

// ---------------------------------------------------------------------- wire

fn cmd(parts: &[&[u8]]) -> Vec<u8> {
    let mut out = format!("*{}\r\n", parts.len()).into_bytes();
    for part in parts {
        out.extend_from_slice(format!("${}\r\n", part.len()).as_bytes());
        out.extend_from_slice(part);
        out.extend_from_slice(b"\r\n");
    }
    out
}

fn get_batch(depth: usize) -> Vec<u8> {
    (0..depth)
        .flat_map(|_| cmd(&[b"GET", b"k0000000000000000"]))
        .collect()
}

/// One prepared request, replayed through a live connection.
struct Harness<E: Executor> {
    conn: RespConnection<E>,
    out: Vec<u8>,
    /// The request bytes, and how many requests they carry.
    batch: Vec<u8>,
    per_batch: usize,
}

impl<E: Executor> Harness<E> {
    fn new(exec: E, batch: Vec<u8>, per_batch: usize) -> Self {
        Self {
            conn: RespConnection::new(exec, ConnConfig::default(), 1),
            out: Vec::with_capacity(64 * 1024),
            batch,
            per_batch,
        }
    }

    /// One turn: push the prepared request, drain, return bytes appended.
    fn turn(&mut self) -> usize {
        self.out.clear();
        let _ = self.conn.push(&self.batch);
        let before = self.out.len();
        self.conn.drain(&mut self.out);
        divan::black_box(self.per_batch);
        self.out.len() - before
    }
}

/// divan drives a bench with a `Send + Sync` `Fn` closure and a turn mutates
/// the connection, so the connection is held behind a `Mutex` and each
/// iteration takes it. It is uncontended, so the lock costs two atomics
/// against a parse that touches every byte of the request.
fn harness(batch: Vec<u8>, per_batch: usize) -> Mutex<Harness<MapExec>> {
    Mutex::new(Harness::new(MapExec::new(), batch, per_batch))
}

fn batched(batch: Vec<u8>, per_batch: usize) -> Mutex<Harness<BatchedExec>> {
    Mutex::new(Harness::new(BatchedExec::new(), batch, per_batch))
}

const KEY: &[u8] = b"k0000000000000000";

#[divan::bench]
fn ping(bencher: Bencher) {
    let h = harness(cmd(&[b"PING"]), 1);
    bencher.bench(|| h.lock().expect("harness").turn());
}

#[divan::bench]
fn get_16b(bencher: Bencher) {
    let h = harness(cmd(&[b"GET", KEY]), 1);
    h.lock().expect("harness").turn();
    bencher.bench(|| h.lock().expect("harness").turn());
}

#[divan::bench]
fn get_1kib(bencher: Bencher) {
    let value = vec![b'x'; 1024];
    let h = harness(cmd(&[b"SET", KEY, &value]), 1);
    h.lock().expect("harness").turn();
    bencher.bench(|| h.lock().expect("harness").turn());
}

#[divan::bench]
fn set_16b(bencher: Bencher) {
    let h = harness(cmd(&[b"SET", KEY, b"v0123456789abcdef"]), 1);
    bencher.bench(|| h.lock().expect("harness").turn());
}

#[divan::bench]
fn set_1kib(bencher: Bencher) {
    let value = vec![b'x'; 1024];
    let h = harness(cmd(&[b"SET", KEY, &value]), 1);
    bencher.bench(|| h.lock().expect("harness").turn());
}

#[divan::bench]
fn exists(bencher: Bencher) {
    let h = harness(cmd(&[b"EXISTS", KEY]), 1);
    h.lock().expect("harness").turn();
    bencher.bench(|| h.lock().expect("harness").turn());
}

#[divan::bench]
fn ttl(bencher: Bencher) {
    let h = harness(cmd(&[b"TTL", KEY]), 1);
    h.lock().expect("harness").turn();
    bencher.bench(|| h.lock().expect("harness").turn());
}

#[divan::bench]
fn getrange_64b(bencher: Bencher) {
    let value = vec![b'x'; 1024];
    let seed = harness(cmd(&[b"SET", KEY, &value]), 1);
    seed.lock().expect("harness").turn();
    let h = harness(cmd(&[b"GETRANGE", KEY, b"0", b"63"]), 1);
    h.lock().expect("harness").turn();
    bencher.bench(|| h.lock().expect("harness").turn());
}

#[divan::bench]
fn strlen_1kib(bencher: Bencher) {
    let value = vec![b'x'; 1024];
    let h = harness(cmd(&[b"SET", KEY, &value]), 1);
    h.lock().expect("harness").turn();
    let h = harness(cmd(&[b"STRLEN", KEY]), 1);
    h.lock().expect("harness").turn();
    bencher.bench(|| h.lock().expect("harness").turn());
}

#[divan::bench]
fn pipeline1_get(bencher: Bencher) {
    let h = harness(get_batch(1), 1);
    h.lock().expect("harness").turn();
    bencher.bench(|| h.lock().expect("harness").turn());
}

#[divan::bench]
fn pipeline4_get(bencher: Bencher) {
    let h = harness(get_batch(4), 4);
    h.lock().expect("harness").turn();
    bencher.bench(|| h.lock().expect("harness").turn());
}

#[divan::bench]
fn pipeline16_get(bencher: Bencher) {
    let h = harness(get_batch(16), 16);
    h.lock().expect("harness").turn();
    bencher.bench(|| h.lock().expect("harness").turn());
}

#[divan::bench]
fn pipeline64_get(bencher: Bencher) {
    let h = harness(get_batch(64), 64);
    h.lock().expect("harness").turn();
    bencher.bench(|| h.lock().expect("harness").turn());
}

#[divan::bench]
fn pipeline16_get_batched_exec(bencher: Bencher) {
    let h = batched(get_batch(16), 16);
    h.lock().expect("harness").turn();
    bencher.bench(|| h.lock().expect("harness").turn());
}

#[divan::bench]
fn pipeline64_get_batched_exec(bencher: Bencher) {
    let h = batched(get_batch(64), 64);
    h.lock().expect("harness").turn();
    bencher.bench(|| h.lock().expect("harness").turn());
}

// ------------------------------------------------- allocation report (dhat)

/// Emits a dhat report for one request shape, so allocations per request are a
/// measurement rather than an argument. One shape per process, chosen by
/// argument, because dhat profiles the whole process.
#[cfg(feature = "alloc-profile")]
fn allocation_report() {
    const REQUESTS: u64 = 100_000;
    let shape = std::env::args().nth(1).unwrap_or_else(|| "get".to_owned());
    let (batch, per_batch, seed) = match shape.as_str() {
        "ping" => (cmd(&[b"PING"]), 1, false),
        "get16" => (cmd(&[b"GET", KEY]), 1, true),
        "get1kib" => {
            let value = vec![b'x'; 1024];
            (cmd(&[b"SET", KEY, &value]), 1, true)
        }
        "set16" => (cmd(&[b"SET", KEY, b"v0123456789abcdef"]), 1, false),
        "set1kib" => {
            let value = vec![b'x'; 1024];
            (cmd(&[b"SET", KEY, &value]), 1, false)
        }
        "pipe16" => (get_batch(16), 16, true),
        other => panic!("unknown shape {other:?}"),
    };
    let profiler = dhat::Profiler::new_heap();
    let h = harness(batch.clone(), per_batch);
    if seed {
        h.lock().expect("harness").turn();
    }
    for _ in 0..REQUESTS {
        h.lock().expect("harness").turn();
    }
    let stats = dhat::HeapStats::get();
    // Per-request figures are scaled by 1000 and printed as integers: the
    // workspace forbids lossy float casts, and a scaled integer says the same
    // thing to three decimal places.
    let requests = REQUESTS + u64::from(seed);
    let blocks = stats.total_blocks;
    let bytes = stats.total_bytes;
    println!(
        "shape={shape:<8} requests={requests} total_blocks={blocks} \
         milli_blocks_per_request={} total_bytes={bytes} milli_bytes_per_request={}",
        blocks.saturating_mul(1000) / requests.max(1),
        bytes.saturating_mul(1000) / requests.max(1),
    );
    drop(profiler);
}

#[cfg(feature = "alloc-profile")]
fn main() {
    allocation_report();
}

#[cfg(not(feature = "alloc-profile"))]
fn main() {
    divan::main();
}
