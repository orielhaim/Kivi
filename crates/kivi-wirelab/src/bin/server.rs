//! The external latency laboratory's server: the same code path with one layer
//! removed at a time.
//!
//! # Why subtraction, and why these particular layers
//!
//! `read_path.rs` measures a 46 ns core operation. The external question is what
//! happens to the other ~5,800 ns. A single instrument cannot separate "the kernel
//! did a syscall", "the runtime woke a task", and "the parser did work", because all
//! three are inside one `read().await` on the page. This binary makes them separate
//! by *removing* them one at a time from an otherwise identical server:
//!
//! | mode | what it does | what its answer means |
//! | --- | --- | --- |
//! | `a` | compio, read, write the same bytes back | the floor: what any compio TCP server costs on this host |
//! | `ap` | `a`, plus Kivi's per-read `Vec` + copy | what Kivi's buffer handling costs, with everything else held fixed |
//! | `b` | `ap`, plus RESP parse, fixed bulk reply | what RESP costs over a raw echo |
//! | `c` | `b`, plus a real `ObjectStore` lookup and the stored value | what Kivi's state layer costs over a fixed reply |
//! | `epoll` | `b` on blocking threads, no async runtime at all | whether compio is the problem or the kernel is |
//!
//! The interesting comparisons are `a` against `b` (how much RESP really costs),
//! `b` against `c` (the state layer, which `read_path.rs` already measures at 6.7 ns),
//! and `b` against `epoll` (the runtime question).
//!
//! # What this binary deliberately does not do
//!
//! It does not use `kivi-engine`. Every mode here is a few dozen lines, and if any
//! of them stopped being a few dozen lines the subtraction would stop meaning what
//! the table says it means. Control D - the real server - is `kivi-server`, measured
//! with the same client from the same harness.
//!
//! # Modes `a` and `ap` speak no RESP
//!
//! They echo. The client is told which it is talking to and verifies the bytes came
//! back rather than parsing a reply. This is what makes them a floor rather than a
//! weaker `b`: they have no protocol to get wrong.

use std::io::{Read as _, Write as _};
use std::net::SocketAddr;

use compio::buf::BufResult;
use compio::io::{AsyncRead, AsyncWriteExt};
use kivi_resp::{Limits, Parsed};
use kivi_state::{Key, ObjectStore, Operation, Prepared};
use kivi_types::WallTimestamp;

/// Receive buffer size. Matches `kivi-engine`'s `READ_CHUNK` so the per-read
/// allocation in `ap` is the same allocation the real server makes, not a smaller
/// stand-in that the allocator would treat differently.
const READ_CHUNK: usize = 64 * 1024;

/// The value every key holds in mode `c`, and the value mode `b` replies with.
///
/// 16 bytes, because that is the size the internal decomposition was measured at.
/// A control that replied with a different length would not be comparable.
const VALUE: &[u8] = b"0123456789abcdef";

/// The fixed instant every control uses. A wall-clock read inside the hot path would
/// add a syscall-visible dependency and measure `clock_gettime` as much as the store.
const NOW: WallTimestamp = WallTimestamp::from_micros(1_700_000_000_000_000);

/// The key every request addresses, fixed so the table lookup is a hit.
const KEY: &str = "bench:key:00000001";

/// Which layers this server runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mode {
    /// compio, raw echo, one persistent buffer.
    Echo,
    /// `Echo` plus Kivi's per-read allocation and copy.
    EchoPooled,
    /// `EchoPooled` plus a RESP parse and a fixed bulk reply.
    RespNoop,
    /// `RespNoop` plus a real `ObjectStore` lookup.
    Table,
    /// `RespNoop` on blocking threads. No async runtime.
    Blocking,
}

impl Mode {
    /// Parses the mode name.
    fn parse(name: &str) -> Result<Self, String> {
        Ok(match name {
            "a" => Self::Echo,
            "ap" => Self::EchoPooled,
            "b" => Self::RespNoop,
            "c" => Self::Table,
            "epoll" => Self::Blocking,
            other => {
                return Err(format!(
                    "unknown mode {other:?}; expected a, ap, b, c or epoll"
                ));
            }
        })
    }

    /// One line describing what this mode is, printed at startup and copied into
    /// every report row so a table cannot be read without its definitions.
    const fn describe(self) -> &'static str {
        match self {
            Self::Echo => "compio, raw echo, persistent buffer (runtime floor)",
            Self::EchoPooled => "compio, raw echo, per-read Vec + copy (Kivi's buffer pattern)",
            Self::RespNoop => "compio, RESP parse, fixed bulk reply",
            Self::Table => "compio, RESP parse, ObjectStore lookup, stored reply",
            Self::Blocking => "blocking threads, RESP parse, fixed bulk reply (no async runtime)",
        }
    }
}

fn main() {
    let mut mode = None;
    let mut event_interval: Option<usize> = None;
    let mut listen: SocketAddr = "127.0.0.1:0".parse().expect("valid default");
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "--mode" => {
                index += 1;
                mode = Some(Mode::parse(&args[index]).unwrap_or_else(|e| {
                    eprintln!("kivi-wirelab-server: {e}");
                    std::process::exit(2);
                }));
            }
            "--event-interval" => {
                index += 1;
                event_interval = Some(args[index].parse().unwrap_or_else(|_| {
                    eprintln!("kivi-wirelab-server: --event-interval needs a number");
                    std::process::exit(2);
                }));
            }
            "--listen" => {
                index += 1;
                listen = args[index].parse().expect("valid --listen");
            }
            other => {
                eprintln!("kivi-wirelab-server: unexpected argument {other:?}");
                std::process::exit(2);
            }
        }
        index += 1;
    }
    let mode = mode.unwrap_or_else(|| {
        eprintln!("kivi-wirelab-server: --mode is required (a, ap, b, c or epoll)");
        std::process::exit(2);
    });

    if mode == Mode::Blocking {
        if let Err(error) = run_blocking(listen, mode) {
            eprintln!("kivi-wirelab-server: {error}");
            std::process::exit(1);
        }
        // `incoming()` never ends, so this is unreachable. Parked rather than exited
        // so a killed listener looks like a kill and not like a crash.
        loop {
            std::thread::park();
        }
    }

    // `event_interval` is how many scheduler ticks pass between polls of the io_uring
    // ring for external events. compio's default is 61, which is a throughput
    // choice: a busy runtime amortises a ring poll over many completions. A server
    // that handles one request per connection turn is not that runtime, and a request
    // that arrives while the runtime is mid-interval waits for the interval to expire.
    // Exposed as a flag because it is the cheapest available test of "is the runtime
    // sleeping when it should be working". Measured: it makes no difference, which is
    // itself the result - see `docs/perf-program.md` Part V §3.
    let mut builder = compio::runtime::Runtime::builder();
    if let Some(interval) = event_interval {
        builder.event_interval(interval);
    }
    let runtime = match builder.build() {
        Ok(runtime) => runtime,
        Err(error) => {
            eprintln!("kivi-wirelab-server: compio runtime: {error}");
            std::process::exit(1);
        }
    };
    // Announces its own bound address before the accept loop, which never returns.
    if let Err(error) = runtime.block_on(run_compio(listen, mode)) {
        eprintln!("kivi-wirelab-server: {error}");
        std::process::exit(1);
    }
}

/// Prints the bound address on a line of its own, so a harness can read the port
/// when the server was asked for port 0.
fn announce(mode: Mode, bound: SocketAddr) {
    println!(
        "mode={}",
        match mode {
            Mode::Echo => "a",
            Mode::EchoPooled => "ap",
            Mode::RespNoop => "b",
            Mode::Table => "c",
            Mode::Blocking => "epoll",
        }
    );
    println!("listen={bound}");
    println!("desc={}", mode.describe());
    let _ = std::io::stdout().flush();
}

// ---------------------------------------------------------------- compio modes

/// Binds, reports, then serves forever.
///
/// The accept shape is the real server's: one `accept` per connection and a detached
/// task per connection, with no timeout on the accept. Matching that is the point - a
/// control that used a cheaper accept shape would be measuring a different
/// architecture rather than a smaller one.
async fn run_compio(listen: SocketAddr, mode: Mode) -> std::io::Result<()> {
    let listener = compio::net::TcpListener::bind(listen).await?;
    announce(mode, listener.local_addr()?);
    loop {
        let (stream, _) = listener.accept().await?;
        // Nagle would make a reply wait on a delayed ACK. Every mode turns it off,
        // including the raw echoes, so no row carries a latency the others do not.
        let _ = stream.set_nodelay(true);
        compio::runtime::spawn(async move {
            serve_compio_conn(stream, mode).await;
        })
        .detach();
    }
}

/// Serves one connection in the given mode.
async fn serve_compio_conn(stream: compio::net::TcpStream, mode: Mode) {
    let mut stream = stream;
    match mode {
        Mode::Echo => {
            // The floor, with the best buffer handling compio's API allows.
            //
            // `read` takes a buffer and hands it back filled; `write_all` takes a
            // buffer and hands it back. So one `Vec` can ping-pong between them:
            // truncate it to what arrived, write exactly that, and receive it back
            // as the next read's buffer. No allocation per read, no copy between a
            // receive buffer and a write buffer, and no second buffer to keep warm.
            //
            // A `Vec` whose length is `count` and whose capacity is `READ_CHUNK`
            // still works as a read target: compio's `IoBufMut` for `Vec` exposes
            // the spare capacity, so the next read writes past the current length
            // and returns a buffer with the new length. The client verifies the
            // echoed bytes, so a mistake here cannot pass unnoticed.
            let mut buf = vec![0u8; READ_CHUNK];
            loop {
                let BufResult(result, filled) = stream.read(buf).await;
                match result {
                    Ok(0) | Err(_) => return,
                    Ok(count) => {
                        let mut filled = filled;
                        filled.truncate(count);
                        let BufResult(written, returned) = stream.write_all(filled).await;
                        if written.is_err() {
                            return;
                        }
                        buf = returned;
                    }
                }
            }
        }
        Mode::EchoPooled => {
            // Kivi's buffer pattern, with no protocol: a fresh `Vec` is built per
            // read, filled by the kernel, copied into a persistent buffer, and
            // dropped, and the persistent buffer is what gets echoed.
            //
            // This is the mode that prices the buffer handling on its own. The
            // client verifies echoed bytes and enforces a read timeout, so a mode
            // that accidentally answered from the RESP path fails loudly rather
            // than reporting a number for the wrong work.
            let mut read_buf = vec![0u8; READ_CHUNK];
            loop {
                let chunk = Vec::with_capacity(READ_CHUNK);
                let BufResult(result, filled) = stream.read(chunk).await;
                let count = match result {
                    Ok(0) | Err(_) => return,
                    Ok(count) => count,
                };
                if read_buf.len() < count {
                    read_buf.resize(count, 0);
                }
                read_buf[..count].copy_from_slice(&filled[..count]);
                let out = read_buf[..count].to_vec();
                let BufResult(written, returned) = stream.write_all(out).await;
                if written.is_err() {
                    return;
                }
                // Keep the written buffer's capacity for the next turn, so this mode
                // differs from `a` by the per-read allocation and the copy and by
                // nothing else.
                read_buf.resize(READ_CHUNK.max(returned.len()), 0);
            }
        }
        _ => serve_buffered(stream, mode).await,
    }
}

/// The modes that parse: Kivi's own read shape, one read then everything it
/// decoded then one write.
async fn serve_buffered(mut stream: compio::net::TcpStream, mode: Mode) {
    let mut read_buf = vec![0u8; READ_CHUNK];
    let mut out: Vec<u8> = Vec::with_capacity(READ_CHUNK);
    let limits = Limits::default();
    let mut store = (mode == Mode::Table).then(ObjectStore::new);
    if let Some(store) = store.as_mut()
        && let Prepared::Write(mutation) = store
            .prepare(
                &Operation::Set {
                    key: Key::new(KEY.as_bytes().to_vec()),
                    value: bytes::Bytes::from_static(VALUE),
                },
                NOW,
            )
            .expect("set prepares")
    {
        store.apply(&mutation, NOW).expect("set applies");
    }

    loop {
        // The per-read allocation, exactly as `kivi-engine`'s `serve_resp_conn` does
        // it: compio's `read` is buffer-passing and hands back a fresh allocation
        // each call, so a `Vec` is built, filled, copied, and dropped per read.
        // Mode `ap` exists to price this.
        let chunk = Vec::with_capacity(READ_CHUNK);
        let BufResult(result, filled) = stream.read(chunk).await;
        let count = match result {
            Ok(0) | Err(_) => return,
            Ok(count) => count,
        };
        if read_buf.len() < count {
            read_buf.resize(count, 0);
        }
        read_buf[..count].copy_from_slice(&filled[..count]);

        out.clear();
        let mut cursor = 0;
        while cursor < count {
            let (parsed, used) = kivi_resp::parse_command(&read_buf[cursor..count], &limits);
            cursor += used;
            let Parsed::Command(command) = parsed else {
                // Every control answers an unparseable frame the same way: flush
                // whatever was produced, then close. A control that looped or
                // panicked would report a number for work the real server does not
                // do. `write_all` takes its buffer by value, so `out` moves here and
                // the connection is finished with it.
                let _ = stream.write_all(out).await;
                return;
            };
            match store.as_ref() {
                Some(store) => {
                    let key = command.args().first().copied().unwrap_or_default();
                    match store.get(&Key::new(key.to_vec()), NOW) {
                        Some(stored) => write_stored(&mut out, stored),
                        None => out.extend_from_slice(b"$-1\r\n"),
                    }
                }
                None => write_fixed_bulk(&mut out),
            }
        }

        if !out.is_empty() {
            let BufResult(written, returned) = stream.write_all(out).await;
            if written.is_err() {
                return;
            }
            out = returned;
        }
    }
}

/// Appends the fixed 16-byte bulk reply mode `b` sends for every command.
fn write_fixed_bulk(out: &mut Vec<u8>) {
    write_bulk(out, VALUE);
}

/// Appends a stored object's bytes as a bulk reply.
///
/// Only the inline `Bytes` representation is a real value here. A chunked or
/// fabric-backed object is answered empty rather than resolved, because resolving
/// it would pull the chunk layer into a control whose job is to measure a local
/// table probe.
fn write_stored(out: &mut Vec<u8>, stored: &kivi_state::StoredObject) {
    match stored.value() {
        kivi_state::LogicalValue::Bytes(bytes) => write_bulk(out, bytes),
        _ => write_bulk(out, b""),
    }
}

/// Appends a RESP bulk reply carrying `value`.
///
/// Written by hand rather than through a codec so the control's reply cost is a
/// `to_string` and two `extend_from_slice` calls that are visible here. A control
/// that borrowed its reply path from the thing it is measuring could not be
/// subtracted from it.
fn write_bulk(out: &mut Vec<u8>, value: &[u8]) {
    out.push(b'$');
    out.extend_from_slice(value.len().to_string().as_bytes());
    out.extend_from_slice(b"\r\n");
    out.extend_from_slice(value);
    out.extend_from_slice(b"\r\n");
}

// ------------------------------------------------------------- blocking control

/// The non-Compio floor: one thread per connection, blocking sockets, no async
/// runtime, no executor, no wakers, no ring.
///
/// This is the control that decides whether compio is the problem. It is *not* a
/// production candidate - a thread per connection does not scale to the
/// connection counts a cluster sees, and it is written for clarity rather than
/// throughput. It is here to answer one question with a number: what does a TCP
/// request/response cost on this host when nothing at all is between the syscall and
/// the bytes?
fn run_blocking(listen: SocketAddr, mode: Mode) -> std::io::Result<()> {
    let listener = std::net::TcpListener::bind(listen)?;
    announce(mode, listener.local_addr()?);
    for incoming in listener.incoming() {
        let Ok(stream) = incoming else { continue };
        let _ = stream.set_nodelay(true);
        std::thread::spawn(move || serve_blocking_conn(stream, mode));
    }
    Ok(())
}

/// Serves one connection with blocking reads and writes.
fn serve_blocking_conn(mut stream: std::net::TcpStream, mode: Mode) {
    let limits = Limits::default();
    let mut read_buf = vec![0u8; READ_CHUNK];
    let mut out: Vec<u8> = Vec::with_capacity(READ_CHUNK);
    let mut store = (mode == Mode::Table).then(ObjectStore::new);
    if let Some(store) = store.as_mut()
        && let Prepared::Write(mutation) = store
            .prepare(
                &Operation::Set {
                    key: Key::new(KEY.as_bytes().to_vec()),
                    value: bytes::Bytes::from_static(VALUE),
                },
                NOW,
            )
            .expect("set prepares")
    {
        store.apply(&mutation, NOW).expect("set applies");
    }

    loop {
        let Ok(count) = stream.read(&mut read_buf) else {
            return;
        };
        if count == 0 {
            return;
        }
        out.clear();
        let mut cursor = 0;
        while cursor < count {
            let (parsed, used) = kivi_resp::parse_command(&read_buf[cursor..count], &limits);
            cursor += used;
            let Parsed::Command(command) = parsed else {
                let _ = stream.write_all(&out);
                return;
            };
            match store.as_ref() {
                Some(store) => {
                    let key = command.args().first().copied().unwrap_or_default();
                    match store.get(&Key::new(key.to_vec()), NOW) {
                        Some(stored) => write_stored(&mut out, stored),
                        None => out.extend_from_slice(b"$-1\r\n"),
                    }
                }
                None => write_fixed_bulk(&mut out),
            }
        }
        if stream.write_all(&out).is_err() {
            return;
        }
    }
}
