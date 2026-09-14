//! Minimal raw RESP client over blocking TCP.
//!
//! Deliberately small and dependency-free: it speaks just enough RESP2 to
//! drive the profiled commands (arrays of bulk strings out; full frame
//! parsing back), with per-connection byte accounting for benchmark I/O
//! metrics. Benchmarks and conformance share this transport so Kivi-vs-Redis
//! comparisons isolate *server* differences, never client libraries.
//! Ecosystem/client-compatibility tests use `redis-rs` instead (see
//! `tests/resp.rs`).

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::time::Duration;

/// Default I/O timeout for lab connections.
pub const IO_TIMEOUT: Duration = Duration::from_secs(10);

/// One parsed RESP reply. Bulk payloads stay binary-safe (`Vec<u8>`);
/// command names and integer parsing validate ASCII upstream in the caller.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Reply {
    /// `+...`.
    Simple(Vec<u8>),
    /// `-...` (prefix + message, without the leading `-`).
    Error(Vec<u8>),
    /// `:...`.
    Integer(i64),
    /// `$len` (`None` is the null bulk string).
    Bulk(Option<Vec<u8>>),
    /// `*n` (`None` is the null array).
    Array(Option<Vec<Reply>>),
}

impl Reply {
    /// Whether this reply is an error frame.
    #[must_use]
    pub const fn is_error(&self) -> bool {
        matches!(self, Self::Error(_))
    }
}

/// Errors from the lab client itself (transport/parse, never server semantics).
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum ClientError {
    /// TCP or timeout failure.
    #[error("transport: {0}")]
    Transport(String),
    /// Reply bytes do not parse as RESP.
    #[error("malformed reply: {0}")]
    Malformed(String),
}

impl From<std::io::Error> for ClientError {
    fn from(error: std::io::Error) -> Self {
        Self::Transport(error.to_string())
    }
}

/// One owned connection. Not `Sync` by design: benchmark threads each own
/// one, so connections never serialize across threads.
#[derive(Debug)]
pub struct RespClient {
    reader: BufReader<TcpStream>,
    /// Bytes written so far (benchmark I/O accounting).
    pub bytes_out: u64,
    /// Bytes read so far (benchmark I/O accounting).
    pub bytes_in: u64,
}

impl RespClient {
    /// Connects to `addr` (`host:port`, with an optional `redis://` scheme
    /// prefix tolerated) with [`IO_TIMEOUT`] timeouts.
    ///
    /// # Errors
    ///
    /// Returns [`ClientError::Transport`] when the endpoint is unreachable.
    pub fn connect(addr: &str) -> Result<Self, ClientError> {
        let host = addr
            .strip_prefix("redis://")
            .unwrap_or(addr)
            .split('/')
            .next()
            .unwrap_or(addr);
        let endpoint: Vec<std::net::SocketAddr> = host
            .to_socket_addrs()
            .map_err(|error| ClientError::Transport(error.to_string()))?
            .collect();
        let endpoint = endpoint
            .into_iter()
            .next()
            .ok_or_else(|| ClientError::Transport(format!("no address for {addr:?}")))?;
        let stream = TcpStream::connect_timeout(&endpoint, IO_TIMEOUT)
            .map_err(|error| ClientError::Transport(error.to_string()))?;
        stream
            .set_read_timeout(Some(IO_TIMEOUT))
            .map_err(ClientError::from)?;
        stream
            .set_write_timeout(Some(IO_TIMEOUT))
            .map_err(ClientError::from)?;
        // Small request/response frames both ways: never wait for delayed ACKs.
        let _ = stream.set_nodelay(true);
        Ok(Self {
            reader: BufReader::new(stream),
            bytes_out: 0,
            bytes_in: 0,
        })
    }

    /// Sends one command (array of bulk strings) without reading the reply.
    ///
    /// # Errors
    ///
    /// Returns [`ClientError::Transport`] on write failure.
    pub fn send(&mut self, args: &[&[u8]]) -> Result<(), ClientError> {
        let mut out = format!("*{}\r\n", args.len()).into_bytes();
        for arg in args {
            out.extend_from_slice(format!("${}\r\n", arg.len()).as_bytes());
            out.extend_from_slice(arg);
            out.extend_from_slice(b"\r\n");
        }
        self.reader.get_mut().write_all(&out)?;
        self.reader.get_mut().flush()?;
        self.bytes_out += out.len() as u64;
        Ok(())
    }

    /// Reads one reply frame (`bytes_in` grows by the wire length of the
    /// frame headers and bulk payloads as they are consumed).
    ///
    /// # Errors
    ///
    /// Returns [`ClientError`] on transport failure or malformed bytes.
    pub fn read(&mut self) -> Result<Reply, ClientError> {
        self.read_frame()
    }

    /// Sends one command and reads its reply.
    ///
    /// # Errors
    ///
    /// Returns [`ClientError`] on transport failure or malformed bytes.
    pub fn round_trip(&mut self, args: &[&[u8]]) -> Result<Reply, ClientError> {
        self.send(args)?;
        self.read()
    }

    /// Sends `cmds` back-to-back, then reads that many replies in order
    /// (pipelining; responses match requests positionally).
    ///
    /// # Errors
    ///
    /// Returns [`ClientError`] on transport failure or malformed bytes.
    pub fn pipeline(&mut self, cmds: &[Vec<Vec<u8>>]) -> Result<Vec<Reply>, ClientError> {
        let mut out = Vec::new();
        for cmd in cmds {
            out.extend_from_slice(format!("*{}\r\n", cmd.len()).as_bytes());
            for arg in cmd {
                out.extend_from_slice(format!("${}\r\n", arg.len()).as_bytes());
                out.extend_from_slice(arg);
                out.extend_from_slice(b"\r\n");
            }
        }
        self.reader.get_mut().write_all(&out)?;
        self.reader.get_mut().flush()?;
        self.bytes_out += out.len() as u64;
        let mut replies = Vec::with_capacity(cmds.len());
        for _ in cmds {
            replies.push(self.read()?);
        }
        Ok(replies)
    }

    /// Drains and returns byte counters since the last call.
    pub fn take_bytes(&mut self) -> (u64, u64) {
        let counters = (self.bytes_out, self.bytes_in);
        self.bytes_out = 0;
        self.bytes_in = 0;
        counters
    }

    fn read_frame(&mut self) -> Result<Reply, ClientError> {
        let mut line = Vec::new();
        self.reader
            .read_until(b'\n', &mut line)
            .map_err(ClientError::from)?;
        self.bytes_in += line.len() as u64;
        let (kind, rest) = line
            .split_first()
            .ok_or_else(|| ClientError::Malformed("empty".to_owned()))?;
        let text = rest
            .strip_suffix(b"\r\n")
            .or_else(|| rest.strip_suffix(b"\n"))
            .unwrap_or(rest);
        match kind {
            b'-' => Ok(Reply::Error(text.to_vec())),
            b':' => {
                let text = core::str::from_utf8(text)
                    .map_err(|_| ClientError::Malformed("bad integer".to_owned()))?;
                text.trim()
                    .parse::<i64>()
                    .map(Reply::Integer)
                    .map_err(|_| ClientError::Malformed("bad integer".to_owned()))
            }
            b'$' => {
                let text = core::str::from_utf8(text)
                    .map_err(|_| ClientError::Malformed("bad bulk length".to_owned()))?;
                let len: i64 = text
                    .trim()
                    .parse()
                    .map_err(|_| ClientError::Malformed("bad bulk length".to_owned()))?;
                if len < 0 {
                    return Ok(Reply::Bulk(None));
                }
                let len = usize::try_from(len)
                    .map_err(|_| ClientError::Malformed("bulk length overflow".to_owned()))?;
                let mut data = vec![0u8; len + 2];
                self.reader
                    .read_exact(&mut data)
                    .map_err(ClientError::from)?;
                self.bytes_in += data.len() as u64;
                if !data.ends_with(b"\r\n") {
                    return Err(ClientError::Malformed("bulk missing CRLF".to_owned()));
                }
                data.truncate(len);
                Ok(Reply::Bulk(Some(data)))
            }
            b'*' => {
                let text = core::str::from_utf8(text)
                    .map_err(|_| ClientError::Malformed("bad array length".to_owned()))?;
                let len: i64 = text
                    .trim()
                    .parse()
                    .map_err(|_| ClientError::Malformed("bad array length".to_owned()))?;
                if len < 0 {
                    return Ok(Reply::Array(None));
                }
                let len = usize::try_from(len)
                    .map_err(|_| ClientError::Malformed("array length overflow".to_owned()))?;
                if len > 1024 * 1024 {
                    return Err(ClientError::Malformed("array too large".to_owned()));
                }
                let mut items = Vec::with_capacity(len);
                for _ in 0..len {
                    items.push(self.read_frame()?);
                }
                Ok(Reply::Array(Some(items)))
            }
            // RESP3 frames the profiled commands may return: null, maps,
            // blobs, doubles, booleans. Parse enough to stay in sync.
            b'_' => Ok(Reply::Bulk(None)),
            b'%' | b'~' | b'>' => {
                let text = core::str::from_utf8(text)
                    .map_err(|_| ClientError::Malformed("bad aggregate length".to_owned()))?;
                let len: i64 = text
                    .trim()
                    .parse()
                    .map_err(|_| ClientError::Malformed("bad aggregate length".to_owned()))?;
                if len < 0 {
                    return Ok(Reply::Array(None));
                }
                let pairs = usize::try_from(len)
                    .map_err(|_| ClientError::Malformed("aggregate length overflow".to_owned()))?;
                let count = if *kind == b'%' {
                    pairs.saturating_mul(2)
                } else {
                    pairs
                };
                if count > 2 * 1024 * 1024 {
                    return Err(ClientError::Malformed("aggregate too large".to_owned()));
                }
                let mut items = Vec::with_capacity(count);
                for _ in 0..count {
                    items.push(self.read_frame()?);
                }
                Ok(Reply::Array(Some(items)))
            }
            // Plain-text RESP3 scalars (doubles, booleans, big numbers,
            // verbatim strings, bulk errors, attributes): content-preserving
            // passthrough; callers that need them match on bytes.
            b'+' | b',' | b'#' | b'(' | b'=' | b'!' | b'|' => Ok(Reply::Simple(text.to_vec())),
            _ => Err(ClientError::Malformed(format!("unknown frame type {kind}"))),
        }
    }
}
