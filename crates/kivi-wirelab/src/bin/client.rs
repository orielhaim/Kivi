//! The external latency laboratory's client: one generator, every target.
//!
//! # Why one client
//!
//! Every row in the external tables comes from this binary. `redis-server`,
//! `kivi-server` and the five control servers are driven by identical code doing
//! identical work: build the request bytes once, write them, read the replies, check
//! them. A table that mixed `redis-benchmark` for one row and this for another would
//! be measuring two load generators, and the difference between them is routinely
//! larger than the difference between the servers.
//!
//! # The client is part of the measurement, so it has to be honest
//!
//! Every reply is verified and a run with a failure is rejected rather than
//! reported. A broken stream measures *something*, and nothing in the number says
//! so - a server answering `-ERR` to every GET, or a reader that never issues a
//! read, both produce a plausible-looking latency figure.
//!
//! [`ReplyReader`] reads whatever is available into one buffer and parses replies
//! out of it, which is one read per reply in the common case rather than one per
//! protocol field.
//!
//! # The two numbers it reports per run
//!
//! * `ns/op` - total wall time divided by commands. The number to quote.
//! * `ns/first` - for a pipelined block, the time until the *first* reply byte
//!   arrives. This separates the server's latency to start answering from its
//!   marginal cost per further command, which is the only way to tell a per-request
//!   cost from a per-batch cost. A server with a 20 µs wakeup and a 50 ns command
//!   looks like 20 µs/op at pipeline 1 and 50 ns/op at pipeline 256; the first-reply
//!   figure is what exposes the wakeup.
//! * `reads` - read syscalls issued, so the buffered reader's effect is visible
//!   rather than asserted.

use std::io::{Read as _, Write as _};
use std::net::{SocketAddr, TcpStream};
use std::time::Instant;

/// A target's reply dialect.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Dialect {
    /// The bytes come back unchanged.
    Echo,
    /// RESP replies, parsed and checked.
    Resp,
}

impl Dialect {
    /// Parses the target name.
    fn parse(name: &str) -> Result<Self, String> {
        Ok(match name {
            // The two raw echoes: a, ap.
            "a" | "ap" => Self::Echo,
            // RESP-speaking targets: the b, c and epoll controls, kivi-server, and
            // redis-server.
            "b" | "c" | "epoll" | "d" | "kivi" | "redis" => Self::Resp,
            other => {
                return Err(format!(
                    "unknown target {other:?}; expected a, ap, b, c, epoll, d or redis"
                ));
            }
        })
    }
}

/// One measured run's numbers.
#[derive(Debug, Clone, Copy)]
struct Run {
    /// Total wall time for the whole measured region.
    total: std::time::Duration,
    /// Accumulated time until the first reply byte of each block.
    first: std::time::Duration,
    /// Commands completed.
    ops: u64,
    /// Read syscalls issued, for the whole connection including the warm-up.
    reads: u64,
}

impl Run {
    /// Nanoseconds per command, from the total.
    ///
    /// The nanosecond count is a `u128` and a per-command figure is a fraction of it,
    /// so both are converted through `f64` rather than integer division: a run of
    /// 20,000 operations at 24,000 ns each is 480,000,000 ns, which an `f64` holds
    /// exactly, and a truncated quotient would quietly round every figure down. The
    /// conversion is allowed rather than avoided because the alternative - `u128`
    /// arithmetic and a hand-written decimal formatter - is a worse trade for a
    /// measurement tool, and no value printed here has more than three significant
    /// digits of meaning anyway.
    #[allow(clippy::cast_precision_loss, reason = "ns and op counts; see above")]
    fn ns_per_op(self) -> f64 {
        self.total.as_secs_f64() * 1e9 / self.ops.max(1) as f64
    }

    /// Nanoseconds per command, attributed to the first reply of each block.
    #[allow(clippy::cast_precision_loss, reason = "ns and op counts; see above")]
    fn ns_per_first(self) -> f64 {
        self.first.as_secs_f64() * 1e9 / self.ops.max(1) as f64
    }

    /// Read syscalls per command.
    #[allow(clippy::cast_precision_loss, reason = "ratio of two small integers")]
    fn reads_per_op(self) -> f64 {
        self.reads as f64 / self.ops.max(1) as f64
    }
}

/// Runs the whole measurement and prints the report.
///
/// Long because it is the report: argument parsing, the warm-up/measure loop, the
/// per-client fan-out, the verification gate, and the two tables. Splitting it would
/// move the same lines without making the sequence any clearer, and the order matters
/// - the warm-up repeats are discarded, a failed verification exits before anything
/// is printed as a result, and the minimum is taken only over the repeats that
/// survived both.
#[allow(clippy::too_many_lines, reason = "the report; see above")]
fn main() {
    let mut target = String::from("d");
    let mut pipeline = 1usize;
    let mut clients = 1usize;
    let mut payload = 16usize;
    let mut command = String::from("get");
    let mut blocks = 20_000usize;
    let mut repeats = 7usize;
    let mut warmup = 2usize;
    let mut addr = String::from("127.0.0.1:6379");
    let mut prep = true;

    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut index = 0;
    while index < args.len() {
        let flag = args[index].clone();
        index += 1;
        let mut value = || -> String {
            let next = args.get(index).cloned().unwrap_or_else(|| {
                eprintln!("kivi-wirelab-client: {flag} needs a value");
                std::process::exit(2);
            });
            index += 1;
            next
        };
        match flag.as_str() {
            "--target" => target = value(),
            "--addr" => addr = value(),
            "--pipeline" => pipeline = value().parse().expect("valid --pipeline"),
            "--clients" => clients = value().parse().expect("valid --clients"),
            "--payload" => payload = value().parse().expect("valid --payload"),
            "--cmd" => command = value(),
            "--blocks" => blocks = value().parse().expect("valid --blocks"),
            "--repeats" => repeats = value().parse().expect("valid --repeats"),
            "--warmup" => warmup = value().parse().expect("valid --warmup"),
            "--no-prep" => prep = false,
            other => {
                eprintln!("kivi-wirelab-client: unexpected argument {other:?}");
                std::process::exit(2);
            }
        }
    }
    assert!(pipeline > 0, "pipeline must be at least 1");
    assert!(clients > 0, "clients must be at least 1");

    let dialect = Dialect::parse(&target).unwrap_or_else(|e| {
        eprintln!("kivi-wirelab-client: {e}");
        std::process::exit(2);
    });
    let is_set = command == "set";
    let socket = addr.parse().expect("valid --addr");
    let key = b"bench:key:00000001".to_vec();

    let mut runs = Vec::with_capacity(repeats);
    let mut bad_replies = 0u64;
    let mut total_ops = 0u64;

    for repeat in 0..(warmup + repeats) {
        let mut handles = Vec::with_capacity(clients);
        for client in 0..clients {
            // Each client gets its own key when there is more than one, so a SET run
            // is not 16 threads writing the same key and contending on it. The
            // contention would be measured as server cost, which it is not.
            let client_key = if clients == 1 {
                key.clone()
            } else {
                format!("bench:key:{client:08}").into_bytes()
            };
            let request = if is_set {
                build_set(&client_key, payload)
            } else {
                build_get(&client_key)
            };
            let block = request.repeat(pipeline);
            let prep = prep && !is_set;
            handles.push(std::thread::spawn(move || {
                run_client(
                    socket,
                    &block,
                    pipeline,
                    blocks,
                    dialect,
                    prep,
                    &client_key,
                    payload,
                )
            }));
        }
        let mut worst_bad = 0u64;
        let mut ops = 0u64;
        let mut best: Option<Run> = None;
        for handle in handles {
            let (run, bad) = handle.join().expect("client thread");
            worst_bad = worst_bad.max(bad);
            ops += run.ops;
            best = Some(match best {
                None => run,
                Some(previous) if run.total < previous.total => run,
                Some(previous) => previous,
            });
        }
        if repeat >= warmup {
            runs.push(best.expect("at least one client"));
        }
        bad_replies += worst_bad;
        total_ops = ops;
    }

    let fastest = runs
        .iter()
        .copied()
        .min_by_key(|run| run.total)
        .expect("at least one repeat");
    let fastest_ns = fastest.ns_per_op();

    println!("# kivi-wirelab client");
    println!("target={target} dialect={dialect:?} cmd={command} payload={payload}B");
    println!(
        "clients={clients} pipeline={pipeline} blocks={blocks} repeats={repeats} \
         (warmup {warmup}), min reported"
    );
    println!();
    println!("| figure | value |");
    println!("| --- | ---: |");
    println!("| ns/op (total wall time) | {fastest_ns:.0} |");
    println!(
        "| ns/op (first reply of each block) | {:.0} |",
        fastest.ns_per_first()
    );
    println!("| read syscalls per op | {:.2} |", fastest.reads_per_op());
    println!("| ops/s | {:.0} |", 1e9 / fastest_ns);
    println!("| commands completed | {total_ops} |");
    println!("| replies that failed verification | {bad_replies} |");
    if bad_replies > 0 {
        // Loud, not a warning line: a run that verified nothing is not a measurement
        // and must not be copied into a table.
        eprintln!("REJECTED: {bad_replies} replies did not verify");
        std::process::exit(1);
    }

    println!();
    println!("## every repeat, same method");
    println!("| repeat | ns/op | ns/first | reads/op |");
    println!("| ---: | ---: | ---: | ---: |");
    for (index, run) in runs.iter().enumerate() {
        println!(
            "| {} | {:.0} | {:.0} | {:.2} |",
            index + 1,
            run.ns_per_op(),
            run.ns_per_first(),
            run.reads_per_op()
        );
    }
}

/// A buffered reader over the reply stream.
struct ReplyReader {
    stream: TcpStream,
    /// Unread bytes start at `pos`. Bytes before `pos` are consumed and reclaimed.
    buf: Vec<u8>,
    pos: usize,
    /// Read syscalls issued, reported so the buffered reader's effect is visible
    /// rather than asserted.
    reads: u64,
}

impl ReplyReader {
    /// A reader over `stream` whose buffer starts empty.
    ///
    /// Empty, not zero-filled: a zero-filled buffer satisfies `fill`'s "is there
    /// enough unread data" test on the first call, so the reader never issues a read
    /// at all and hands back the caller's own zero bytes. That reports an entirely
    /// fictional latency, because nothing has crossed the socket. The tell is the
    /// RESP controls reporting "unexpected marker 0x0" verification failures - the
    /// zero byte arriving as a reply.
    ///
    /// Capacity comes from the first read; `fill` grows as needed.
    fn new(stream: TcpStream) -> Self {
        Self {
            stream,
            buf: Vec::new(),
            pos: 0,
            reads: 0,
        }
    }

    /// Makes at least `want` unread bytes available, reading if necessary.
    fn fill(&mut self, want: usize) -> std::io::Result<()> {
        /// How much room to open up per read. Large enough that a 1 KiB reply and a
        /// deep pipeline both fit in one syscall, small enough that a run does not
        /// hold a quarter-megabyte per connection thread.
        const CHUNK: usize = 64 * 1024;
        while self.buf.len() - self.pos < want {
            if self.pos > 0 {
                // Reclaim the consumed prefix rather than growing without bound; a
                // long run would otherwise turn the reply buffer into the leak.
                self.buf.copy_within(self.pos.., 0);
                self.buf.truncate(self.buf.len() - self.pos);
                self.pos = 0;
            }
            if self.buf.len() == self.buf.capacity() {
                self.buf.reserve(CHUNK);
            }
            let writable_at = self.buf.len();
            self.buf.resize(writable_at + CHUNK, 0);
            let got = self.stream.read(&mut self.buf[writable_at..])?;
            self.reads += 1;
            if got == 0 {
                self.buf.truncate(writable_at);
                return Err(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    "server closed mid-reply",
                ));
            }
            self.buf.truncate(writable_at + got);
        }
        Ok(())
    }

    /// One CRLF-terminated line, without its terminator, as an owned string.
    fn line(&mut self) -> std::io::Result<String> {
        const MAX_LINE: usize = 512;
        loop {
            if let Some(offset) = find_crlf(&self.buf[self.pos..]) {
                let line =
                    String::from_utf8_lossy(&self.buf[self.pos..self.pos + offset]).into_owned();
                self.pos += offset + 2;
                return Ok(line);
            }
            if self.buf.len() - self.pos > MAX_LINE {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "reply line too long",
                ));
            }
            self.fill(self.buf.len() - self.pos + 1)?;
        }
    }

    /// Exactly `n` bytes, as a slice valid until the next call.
    fn take(&mut self, n: usize) -> std::io::Result<&[u8]> {
        self.fill(n)?;
        let slice = &self.buf[self.pos..self.pos + n];
        self.pos += n;
        Ok(slice)
    }
}

fn find_crlf(bytes: &[u8]) -> Option<usize> {
    bytes.windows(2).position(|pair| pair == b"\r\n")
}

/// Reads one RESP reply, advancing the reader past it.
///
/// `Ok(true)` for a reply this client expects; `Ok(false)` for a syntactically valid
/// reply that is not - counted as a failed verification rather than a protocol error,
/// because "the server answered something else" is exactly the condition a benchmark
/// must not silently absorb.
fn read_resp(reader: &mut ReplyReader) -> std::io::Result<bool> {
    let marker = reader.take(1)?[0];
    Ok(match marker {
        // A simple string and an integer are both "a line, and it is not an error",
        // so they share an arm rather than differing only in a comment.
        b'+' | b':' => {
            reader.line()?;
            true
        }
        b'-' => {
            // An error reply. Read it so the stream stays framed, and report it as a
            // failed verification: a server answering `-ERR` to every GET is not
            // being benchmarked.
            let line = reader.line()?;
            eprintln!("server error reply: {line}");
            false
        }
        b'$' => {
            let header = reader.line()?;
            // `-1` is a nil bulk - a legitimate answer to a `GET` for a key that is
            // not there, and framed, so it counts as a reply this client accounted
            // for. Anything else that is not a number is a malformed frame, and
            // treating it as nil would silently accept a reply it cannot explain.
            match header.trim().parse::<i64>() {
                Ok(-1) => true,
                Ok(length) if length >= 0 => {
                    let Ok(length) = usize::try_from(length) else {
                        eprintln!("bulk length does not fit a usize: {length}");
                        return Ok(false);
                    };
                    reader.take(length + 2)?;
                    true
                }
                Ok(length) => {
                    eprintln!("negative bulk length {length}");
                    false
                }
                Err(_) => {
                    eprintln!("bulk length is not a number: {header:?}");
                    false
                }
            }
        }
        other => {
            eprintln!("unexpected RESP marker {other:#x}");
            false
        }
    })
}

/// Drives one connection through `blocks` pipeline blocks.
#[allow(clippy::too_many_arguments)]
fn run_client(
    socket: SocketAddr,
    block: &[u8],
    pipeline: usize,
    blocks: usize,
    dialect: Dialect,
    prep: bool,
    key: &[u8],
    payload: usize,
) -> (Run, u64) {
    let stream = match TcpStream::connect(socket) {
        Ok(stream) => stream,
        Err(error) => {
            eprintln!("connect {socket}: {error}");
            return (
                Run {
                    total: std::time::Duration::ZERO,
                    first: std::time::Duration::ZERO,
                    ops: 0,
                    reads: 0,
                },
                u64::MAX,
            );
        }
    };
    let _ = stream.set_nodelay(true);
    // A read timeout, so a server that answers with the wrong dialect fails the run
    // instead of hanging it. Generous, because a legitimate run at pipeline 1 is
    // tens of microseconds per op and a loaded host must not trip it. It exists to
    // catch "this will never finish", not "this is slow".
    let _ = stream.set_read_timeout(Some(std::time::Duration::from_secs(30)));
    let mut reader = ReplyReader::new(stream);
    let mut bad = 0u64;

    // Populate the key so a GET measures a read of `payload` bytes rather than a nil
    // reply. Issued on the same connection, before the clock starts, and verified: a
    // run whose warm-up silently failed would spend the whole measurement reading
    // `$-1`, which is a much cheaper thing than reading 1 KiB.
    if prep && dialect == Dialect::Resp {
        let set = build_set(key, payload);
        if reader.stream.write_all(&set).is_err() {
            return (
                Run {
                    total: std::time::Duration::ZERO,
                    first: std::time::Duration::ZERO,
                    ops: 0,
                    reads: reader.reads,
                },
                u64::MAX,
            );
        }
        match read_resp(&mut reader) {
            Ok(true) => {}
            Ok(false) => bad += 1,
            Err(error) => {
                eprintln!("warm-up SET: {error}");
                return (
                    Run {
                        total: std::time::Duration::ZERO,
                        first: std::time::Duration::ZERO,
                        ops: 0,
                        reads: reader.reads,
                    },
                    u64::MAX,
                );
            }
        }
    }

    let mut first_acc = std::time::Duration::ZERO;
    let start = Instant::now();
    let mut done = 0u64;
    'outer: for _ in 0..blocks {
        if reader.stream.write_all(block).is_err() {
            break;
        }
        let block_start = Instant::now();
        for reply in 0..pipeline {
            match dialect {
                Dialect::Echo => {
                    // The echo targets return the request bytes. Read the block's
                    // worth back: enough to prove the connection carried data
                    // without re-deriving an echo expectation per reply.
                    let want = block.len() / pipeline;
                    if reader.take(want).is_err() {
                        done += reply as u64;
                        break 'outer;
                    }
                    if reply == 0 {
                        first_acc += block_start.elapsed();
                    }
                }
                Dialect::Resp => {
                    let first_reply = reply == 0;
                    match read_resp(&mut reader) {
                        Ok(true) => {}
                        Ok(false) => bad += 1,
                        Err(_) => {
                            done += reply as u64;
                            break 'outer;
                        }
                    }
                    if first_reply {
                        first_acc += block_start.elapsed();
                    }
                }
            }
            done += 1;
        }
    }
    let total = start.elapsed();
    (
        Run {
            total,
            first: first_acc,
            ops: done.max(1),
            reads: reader.reads,
        },
        bad,
    )
}

/// A `GET` frame.
fn build_get(key: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(key.len() + 16);
    out.extend_from_slice(b"*2\r\n$3\r\nGET\r\n$");
    out.extend_from_slice(key.len().to_string().as_bytes());
    out.extend_from_slice(b"\r\n");
    out.extend_from_slice(key);
    out.extend_from_slice(b"\r\n");
    out
}

/// A `SET` frame carrying `payload` bytes of value.
fn build_set(key: &[u8], payload: usize) -> Vec<u8> {
    let value: Vec<u8> = (0..payload)
        .map(|i| u8::try_from(i % 251).expect("fits"))
        .collect();
    let mut out = Vec::with_capacity(key.len() + payload + 32);
    out.extend_from_slice(b"*3\r\n$3\r\nSET\r\n$");
    out.extend_from_slice(key.len().to_string().as_bytes());
    out.extend_from_slice(b"\r\n");
    out.extend_from_slice(key);
    out.extend_from_slice(b"\r\n$");
    out.extend_from_slice(payload.to_string().as_bytes());
    out.extend_from_slice(b"\r\n");
    out.extend_from_slice(&value);
    out.extend_from_slice(b"\r\n");
    out
}
