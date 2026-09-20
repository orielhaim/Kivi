//! Per-connection RESP state: version, bounds, pipeline, and encoding.
//!
//! One connection speaks one version at a time (RESP2 initially, `HELLO 3`
//! upgrades to RESP3). Requests decode in order and replies encode in the
//! same order; pipelined commands never reorder. Decoding uses
//! `redis-protocol`'s direct slice interface with Kivi bounds applied
//! before any attacker-controlled length can allocate.

use crate::command::{REGISTRY, arity_ok, lookup};
use crate::error::RespError;
use crate::translate::{
    Action, Executor, Immediate, RedisOp, Reply, map_execute_error, map_parse_error, map_result,
    parse_unsigned, split_command, translate, uppercase_token,
};

/// RESP version spoken on one connection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum RespVersion {
    /// Initial mode, matching normal Redis behavior.
    #[default]
    V2,
    /// Upgraded via `HELLO 3`.
    V3,
}

/// Resource bounds for one RESP connection. Rejects before allocating
/// attacker-controlled claimed lengths where possible.
#[derive(Debug, Clone, Copy)]
pub struct ConnConfig {
    /// Maximum buffered input bytes (all pipelined requests combined).
    pub max_input_bytes: usize,
    /// Maximum single bulk-string bytes (keys and values).
    pub max_bulk_bytes: usize,
    /// Maximum array elements in one command frame.
    pub max_array_elements: usize,
    /// Maximum command argument count (including the name).
    pub max_args: usize,
    /// Maximum queued pipelined requests answered per connection.
    pub max_pipelined: usize,
    /// Maximum commands executed per drain turn (fairness).
    pub max_commands_per_turn: usize,
    /// Maximum single value bytes accepted from clients (`SET`).
    pub max_value_bytes: usize,
}

impl Default for ConnConfig {
    fn default() -> Self {
        Self {
            // 64 MiB input (matches the legacy value ceiling; requests are
            // small, pipelines share this, never per-command).
            max_input_bytes: 64 * 1024 * 1024,
            // Bulk strings cap at the legacy value ceiling: larger payloads
            // belong on the native streaming interface.
            max_bulk_bytes: 64 * 1024 * 1024,
            // Commands are small; 256 elements is already generous.
            max_array_elements: 256,
            // No supported command takes more than a handful of args.
            max_args: 32,
            // Pipelines deeper than this are rejected, never queued
            // unboundedly.
            max_pipelined: 128,
            // One client sending a huge pipeline cannot monopolize the
            // frontend: this many commands per turn, then yield.
            max_commands_per_turn: 32,
            // RESP values cap at the legacy ceiling with the same message
            // as the native bound.
            max_value_bytes: 64 * 1024 * 1024,
        }
    }
}

/// Mutable per-connection state (version, client name, id).
#[derive(Debug)]
pub struct ConnState {
    /// Version spoken (starts V2).
    pub version: RespVersion,
    /// `CLIENT SETNAME` value (`None` unset).
    pub name: Option<String>,
    /// Connection id (reported in `HELLO`).
    pub id: u64,
    /// Selected database (always 0; other values are rejected, never
    /// stored).
    pub db: u64,
}

impl ConnState {
    /// Creates initial state for one connection.
    #[must_use]
    pub const fn new(id: u64) -> Self {
        Self {
            version: RespVersion::V2,
            name: None,
            id,
            db: 0,
        }
    }
}

/// Per-connection metrics (frontend-local, no shared atomics here; the
/// server aggregates into its own counters without a per-request lock by
/// moving these out when the connection closes).
#[derive(Debug, Clone, Copy, Default)]
pub struct ConnMetrics {
    /// Requests executed (including immediate replies).
    pub requests: u64,
    /// Incoming bytes consumed.
    pub bytes_in: u64,
    /// Outgoing bytes produced.
    pub bytes_out: u64,
    /// Protocol/parse errors answered.
    pub protocol_errors: u64,
    /// Unsupported commands answered.
    pub unsupported: u64,
}

/// One decoded request: raw args plus its connection-order index.
#[derive(Debug)]
struct Pending {
    /// Command name (uppercased ASCII).
    name: Vec<u8>,
    /// Raw byte args (binary-safe, verbatim).
    args: Vec<Vec<u8>>,
}

/// A RESP connection: buffer, version state, metrics, and bounded
/// decode/dispatch. I/O lives in the server; this type never touches a
/// socket, so unit tests drive it with byte strings.
pub struct RespConnection<E> {
    /// Engine access (shared handle, cheap to clone if needed).
    executor: E,
    /// Resource bounds.
    config: ConnConfig,
    /// Mutable version/name state.
    pub state: ConnState,
    /// Undecoded input tail.
    buffer: Vec<u8>,
    /// Per-connection metrics.
    pub metrics: ConnMetrics,
}

impl<E> core::fmt::Debug for RespConnection<E> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("RespConnection")
            .field("config", &self.config)
            .field("state", &self.state)
            .field("buffered", &self.buffer.len())
            .field("metrics", &self.metrics)
            .finish_non_exhaustive()
    }
}

impl<E: Executor> RespConnection<E> {
    /// Creates one connection over `executor` with `config` and `id`.
    pub fn new(executor: E, config: ConnConfig, id: u64) -> Self {
        Self {
            executor,
            config,
            state: ConnState::new(id),
            buffer: Vec::new(),
            metrics: ConnMetrics::default(),
        }
    }

    /// Appends newly read bytes, enforcing the input bound.
    ///
    /// # Errors
    ///
    /// Returns [`RespError::TooLarge`] when the buffered tail would exceed
    /// the configured input ceiling.
    pub fn push(&mut self, bytes: &[u8]) -> Result<(), RespError> {
        if self.buffer.len().saturating_add(bytes.len()) > self.config.max_input_bytes {
            return Err(RespError::TooLarge);
        }
        self.buffer.extend_from_slice(bytes);
        self.metrics.bytes_in += bytes.len() as u64;
        Ok(())
    }

    /// Drains up to the per-turn command budget, executing each request in
    /// order and returning their encoded replies in the same order. `quit`
    /// is included as a `bool` (the server closes after flushing when set).
    /// Incomplete trailing bytes stay buffered for the next read.
    pub fn drain(&mut self) -> (Vec<Vec<u8>>, bool) {
        let mut replies = Vec::new();
        let mut quit = false;
        for _ in 0..self.config.max_commands_per_turn {
            let Some((pending, consumed)) = self.decode_one() else {
                break;
            };
            self.buffer.drain(..consumed);
            if replies.len() >= self.config.max_pipelined {
                replies.push(encode_error(
                    self.state.version,
                    "BUSY pipeline depth exceeded",
                ));
                self.metrics.protocol_errors += 1;
                continue;
            }
            let (bytes, close) = self.dispatch(pending);
            self.metrics.requests += 1;
            self.metrics.bytes_out += bytes.len() as u64;
            replies.push(bytes);
            if close {
                quit = true;
                break;
            }
        }
        (replies, quit)
    }

    /// Whether buffered bytes hold no complete frame (idle, not an error).
    #[must_use]
    pub fn needs_more(&self) -> bool {
        self.buffer.is_empty()
    }

    /// Decodes one command frame from the buffered tail (bounds-checked).
    /// Malformed input fails the connection closed (never resynced by
    /// guessing): the sentinel below carries the whole tail length so the
    /// dispatcher can answer once and signal close.
    fn decode_one(&mut self) -> Option<(Pending, usize)> {
        let (frame, consumed) = match redis_protocol::resp2::decode::decode(self.buffer.as_slice())
        {
            Ok(Some(pair)) => pair,
            Ok(None) => return None,
            Err(_) => {
                let consumed = self.buffer.len();
                return Some((
                    Pending {
                        name: b"__MALFORMED__".to_vec(),
                        args: Vec::new(),
                    },
                    consumed,
                ));
            }
        };
        // Inline commands (`PING` over telnet) are outside this
        // profile: answer a clear error rather than guessing.
        let redis_protocol::resp2::types::OwnedFrame::Array(elements) = frame else {
            return Some((
                Pending {
                    name: Vec::new(),
                    args: Vec::new(),
                },
                consumed,
            ));
        };
        if elements.len() > self.config.max_array_elements || elements.len() > self.config.max_args
        {
            return Some((
                Pending {
                    name: b"__TOO_MANY__".to_vec(),
                    args: Vec::new(),
                },
                consumed,
            ));
        }
        let mut raw: Vec<Vec<u8>> = Vec::with_capacity(elements.len());
        for element in elements {
            match element {
                redis_protocol::resp2::types::OwnedFrame::BulkString(bytes)
                | redis_protocol::resp2::types::OwnedFrame::SimpleString(bytes) => {
                    if bytes.len() > self.config.max_bulk_bytes {
                        return Some((
                            Pending {
                                name: b"__TOO_LARGE__".to_vec(),
                                args: Vec::new(),
                            },
                            consumed,
                        ));
                    }
                    raw.push(bytes);
                }
                // Integers/errors/nils are not valid command arguments.
                _ => {
                    return Some((
                        Pending {
                            name: Vec::new(),
                            args: Vec::new(),
                        },
                        consumed,
                    ));
                }
            }
        }
        let Ok((name, args)) = split_command(raw.as_slice()) else {
            return Some((
                Pending {
                    name: Vec::new(),
                    args: Vec::new(),
                },
                consumed,
            ));
        };
        Some((Pending { name, args }, consumed))
    }

    /// Dispatches one decoded request to an encoded reply (plus close flag).
    #[allow(clippy::too_many_lines, clippy::needless_pass_by_value)]
    fn dispatch(&mut self, pending: Pending) -> (Vec<u8>, bool) {
        let version = self.state.version;
        // Decode sentinels from `decode_one`.
        if pending.name == b"__MALFORMED__" {
            self.metrics.protocol_errors += 1;
            return (
                encode_error(
                    version,
                    "ERR malformed request: expected an array of bulk strings",
                ),
                true,
            );
        }
        if pending.name == b"__TOO_MANY__" {
            self.metrics.protocol_errors += 1;
            return (encode_error(version, "ERR too many arguments"), false);
        }
        if pending.name == b"__TOO_LARGE__" {
            self.metrics.protocol_errors += 1;
            return (
                encode_error(version, "ERR bulk string exceeds bound"),
                false,
            );
        }
        if pending.name.is_empty() {
            self.metrics.protocol_errors += 1;
            return (
                encode_error(
                    version,
                    "ERR malformed request: expected an array of bulk strings",
                ),
                false,
            );
        }
        // Connection/bootstrap commands bypass the engine.
        match pending.name.as_slice() {
            b"HELLO" => return self.dispatch_hello(&pending.args),
            b"CLIENT" => return self.dispatch_client(&pending.args),
            b"SELECT" => return self.dispatch_select(&pending.args),
            b"COMMAND" => return self.dispatch_command(&pending.args),
            b"AUTH" => {
                self.metrics.unsupported += 1;
                return (
                    encode_error(
                        version,
                        "ERR AUTH not implemented in Kivi RESP profile v1; no authentication is enforced",
                    ),
                    false,
                );
            }
            _ => {}
        }
        let spec = lookup(core::str::from_utf8(&pending.name).unwrap_or(""));
        let Some(spec) = spec else {
            self.metrics.protocol_errors += 1;
            let name = String::from_utf8_lossy(&pending.name).into_owned();
            return (
                encode_error(version, &format!("ERR unknown command '{name}'")),
                false,
            );
        };
        if !arity_ok(spec, pending.args.len() + 1) {
            self.metrics.protocol_errors += 1;
            let want = format!(
                "ERR wrong number of arguments for '{}' command",
                spec.name.to_lowercase()
            );
            return (encode_error(version, &want), false);
        }
        if spec.compat == crate::command::CompatClass::Unsupported {
            self.metrics.unsupported += 1;
            let reply = map_parse_error(&RespError::Unsupported {
                command: spec.name.to_owned(),
            });
            return (encode_reply(version, &reply), false);
        }
        // Multi-key forms are rejected, never faked with a loop.
        if matches!(spec.name, "DEL" | "EXISTS") && pending.args.len() != 1 {
            self.metrics.protocol_errors += 1;
            return (
                encode_error(
                    version,
                    "ERR multi-key form unsupported in Kivi RESP profile v1; send single-key requests",
                ),
                false,
            );
        }
        let action = match translate(&pending.name, &pending.args, self.executor.now_micros()) {
            Ok(action) => action,
            Err(error) => {
                match &error {
                    RespError::Unsupported { .. } => self.metrics.unsupported += 1,
                    _ => self.metrics.protocol_errors += 1,
                }
                let reply = map_parse_error(&error);
                return (encode_reply(version, &reply), false);
            }
        };
        // Value-size bound: RESP bulk strings larger than the legacy
        // ceiling belong on native streaming, never buffered here.
        if let Action::Execute { op, .. } = &action {
            let size = match op {
                kivi_state::Operation::SetConditional { value, .. }
                | kivi_state::Operation::Set { value, .. } => Some(value.len()),
                kivi_state::Operation::SetRange { patch, .. } => Some(patch.len()),
                _ => None,
            };
            if size.is_some_and(|len| len > self.config.max_value_bytes) {
                self.metrics.protocol_errors += 1;
                return (
                    encode_error(
                        version,
                        "ERR value exceeds RESP bound; use native streaming",
                    ),
                    false,
                );
            }
        }
        match action {
            Action::Reply(immediate) => {
                let (bytes, close) = self.dispatch_immediate(immediate);
                (bytes, close)
            }
            Action::Execute { op, redis } => self.dispatch_execute(op, redis),
        }
    }

    /// Dispatches an immediate (engine-free) reply.
    fn dispatch_immediate(&mut self, immediate: Immediate) -> (Vec<u8>, bool) {
        let version = self.state.version;
        match immediate {
            Immediate::Simple(text) => (encode_simple(version, text), false),
            Immediate::Echo(bytes) => (encode_bulk(version, &bytes), false),
            Immediate::Nil => (encode_nil(version), false),
            Immediate::Integer(value) => (encode_integer(version, value), false),
            Immediate::Quit => (encode_simple(version, "OK"), true),
        }
    }

    /// Executes one typed operation and maps its outcome (including the
    /// `SETRANGE` read-after-write length).
    #[allow(clippy::needless_pass_by_value)]
    fn dispatch_execute(&mut self, op: kivi_state::Operation, redis: RedisOp) -> (Vec<u8>, bool) {
        let version = self.state.version;
        // `SETRANGE` answers its post-write length: write, then measure.
        // Single-client observable behavior is exact; under concurrent
        // writers the length may reflect a newer write (documented).
        //
        // An empty patch is a pure length read, never a write: Redis treats
        // it as a complete no-op (missing keys answer 0 and are NOT
        // created, TTLs are untouched). Skipping the mutation is
        // atomicity-neutral — there is nothing to race — and it keeps the
        // native splice semantics (explicit offsets are real operations
        // there) out of the compatibility profile.
        if redis == RedisOp::SetRange {
            let key = op.key().clone();
            let empty = matches!(
                &op,
                kivi_state::Operation::SetRange { patch, .. } if patch.is_empty()
            );
            if !empty {
                match self.executor.execute(&op) {
                    Ok(_) => {}
                    Err(error) => return (encode_reply(version, &map_execute_error(error)), false),
                }
            }
            match self
                .executor
                .execute(&kivi_state::Operation::BytesLength { key })
            {
                Ok(kivi_state::OperationResult::Length(value)) => (
                    encode_integer(
                        version,
                        value.map_or(0, |len| i64::try_from(len).unwrap_or(i64::MAX)),
                    ),
                    false,
                ),
                Ok(_) => (encode_error(version, "ERR internal error"), false),
                Err(error) => (encode_reply(version, &map_execute_error(error)), false),
            }
        } else {
            match self.executor.execute(&op) {
                Ok(result) => {
                    let reply = map_result(&result, redis, self.executor.now_micros());
                    (encode_reply(version, &reply), false)
                }
                Err(error) => (encode_reply(version, &map_execute_error(error)), false),
            }
        }
    }

    /// Dispatches `HELLO [protover [AUTH user pass] [SETNAME name]]`.
    fn dispatch_hello(&mut self, args: &[Vec<u8>]) -> (Vec<u8>, bool) {
        let version = self.state.version;
        // No args: report the current context without switching.
        if args.is_empty() {
            return (self.encode_hello(), false);
        }
        let proto = uppercase_token(&args[0]).unwrap_or_default();
        let target = match proto.as_slice() {
            b"2" => RespVersion::V2,
            b"3" => RespVersion::V3,
            _ => {
                self.metrics.protocol_errors += 1;
                return (
                    encode_error(
                        version,
                        "NOPROTO sorry, this protocol version is not supported",
                    ),
                    false,
                );
            }
        };
        // Optional AUTH / SETNAME tail.
        let mut index = 1;
        while index < args.len() {
            let token = uppercase_token(&args[index]).unwrap_or_default();
            match token.as_slice() {
                b"AUTH" => {
                    // Security is a separate feature: never succeed
                    // silently, never switch versions on the way out.
                    self.metrics.unsupported += 1;
                    return (
                        encode_error(
                            version,
                            "ERR AUTH not implemented in Kivi RESP profile v1; no authentication is enforced",
                        ),
                        false,
                    );
                }
                b"SETNAME" => {
                    index += 1;
                    let name = args.get(index).map_or(String::new(), |bytes| {
                        String::from_utf8_lossy(bytes).into_owned()
                    });
                    self.state.name = Some(name);
                    index += 1;
                }
                _ => {
                    self.metrics.protocol_errors += 1;
                    return (
                        encode_error(version, "ERR syntax error in HELLO options"),
                        false,
                    );
                }
            }
        }
        self.state.version = target;
        (self.encode_hello(), false)
    }

    /// Encodes the `HELLO` reply in the (possibly just switched) version:
    /// a flat array in RESP2, a map in RESP3.
    fn encode_hello(&self) -> Vec<u8> {
        const VERSION: &str = env!("CARGO_PKG_VERSION");
        let id = self.state.id;
        match self.state.version {
            RespVersion::V2 => {
                use redis_protocol::resp2::types::OwnedFrame as F;
                let frame = F::Array(vec![
                    F::BulkString(b"server".to_vec()),
                    F::BulkString(b"kivi".to_vec()),
                    F::BulkString(b"version".to_vec()),
                    F::BulkString(VERSION.as_bytes().to_vec()),
                    F::BulkString(b"proto".to_vec()),
                    F::Integer(match self.state.version {
                        RespVersion::V2 => 2,
                        RespVersion::V3 => 3,
                    }),
                    F::BulkString(b"id".to_vec()),
                    F::Integer(i64::try_from(id).unwrap_or(i64::MAX)),
                    F::BulkString(b"mode".to_vec()),
                    F::BulkString(b"standalone".to_vec()),
                    F::BulkString(b"role".to_vec()),
                    F::BulkString(b"master".to_vec()),
                    F::BulkString(b"modules".to_vec()),
                    F::Array(Vec::new()),
                ]);
                encode_frame_v2(&frame)
            }
            RespVersion::V3 => {
                use redis_protocol::resp3::types::OwnedFrame as F;
                use std::collections::HashMap;
                let mut map: HashMap<F, F> = HashMap::new();
                let blob = |bytes: &[u8]| F::BlobString {
                    data: bytes.to_vec(),
                    attributes: None,
                };
                map.insert(blob(b"server"), blob(b"kivi"));
                map.insert(blob(b"version"), blob(VERSION.as_bytes()));
                map.insert(
                    blob(b"proto"),
                    F::Number {
                        data: match self.state.version {
                            RespVersion::V2 => 2,
                            RespVersion::V3 => 3,
                        },
                        attributes: None,
                    },
                );
                map.insert(
                    blob(b"id"),
                    F::Number {
                        data: i64::try_from(id).unwrap_or(i64::MAX),
                        attributes: None,
                    },
                );
                map.insert(blob(b"mode"), blob(b"standalone"));
                map.insert(blob(b"role"), blob(b"master"));
                map.insert(
                    blob(b"modules"),
                    F::Array {
                        data: Vec::new(),
                        attributes: None,
                    },
                );
                encode_frame_v3(&F::Map {
                    data: map,
                    attributes: None,
                })
            }
        }
    }

    /// Dispatches `CLIENT ...` bootstrap subcommands.
    fn dispatch_client(&mut self, args: &[Vec<u8>]) -> (Vec<u8>, bool) {
        let version = self.state.version;
        let Some((first, rest)) = args.split_first() else {
            self.metrics.protocol_errors += 1;
            return (
                encode_error(
                    version,
                    "ERR wrong number of arguments for 'client' command",
                ),
                false,
            );
        };
        let sub = first.to_ascii_uppercase();
        match (sub.as_slice(), rest) {
            (b"SETINFO", [_, _]) => (encode_simple(version, "OK"), false),
            (b"SETNAME", [name]) => {
                if name.len() > 1024 {
                    self.metrics.protocol_errors += 1;
                    return (encode_error(version, "ERR client name too long"), false);
                }
                self.state.name = Some(String::from_utf8_lossy(name).into_owned());
                (encode_simple(version, "OK"), false)
            }
            (b"GETNAME", []) => match &self.state.name {
                Some(name) => (encode_bulk(version, name.as_bytes()), false),
                None => (encode_nil(version), false),
            },
            _ => {
                self.metrics.protocol_errors += 1;
                (
                    encode_error(
                        version,
                        "ERR unsupported CLIENT subcommand in Kivi RESP profile v1",
                    ),
                    false,
                )
            }
        }
    }

    /// Dispatches `SELECT index`: `0` maps to the configured namespace,
    /// anything else is rejected (never faked as extra databases).
    fn dispatch_select(&mut self, args: &[Vec<u8>]) -> (Vec<u8>, bool) {
        let version = self.state.version;
        let [index] = args else {
            self.metrics.protocol_errors += 1;
            return (
                encode_error(
                    version,
                    "ERR wrong number of arguments for 'select' command",
                ),
                false,
            );
        };
        match parse_unsigned(index) {
            Ok(0) => (encode_simple(version, "OK"), false),
            Ok(_) => {
                self.metrics.protocol_errors += 1;
                (
                    encode_error(
                        version,
                        "ERR database numbers other than 0 are unsupported in Kivi RESP profile v1",
                    ),
                    false,
                )
            }
            Err(_) => {
                self.metrics.protocol_errors += 1;
                (
                    encode_error(version, "ERR value is not an integer or out of range"),
                    false,
                )
            }
        }
    }

    /// Dispatches `COMMAND [COUNT|INFO|DOCS|LIST]` from the registry.
    fn dispatch_command(&mut self, args: &[Vec<u8>]) -> (Vec<u8>, bool) {
        let version = self.state.version;
        if args.is_empty() {
            return (self.encode_command_list(), false);
        }
        let sub = args[0].to_ascii_uppercase();
        match (sub.as_slice(), &args[1..]) {
            (b"COUNT", []) => (
                encode_integer(version, i64::try_from(REGISTRY.len()).unwrap_or(i64::MAX)),
                false,
            ),
            (b"LIST", _) => (self.encode_command_names(), false),
            (b"INFO", names) => (self.encode_command_info(names), false),
            (b"DOCS", _) => (self.encode_command_docs(), false),
            _ => {
                self.metrics.protocol_errors += 1;
                (
                    encode_error(version, "ERR unsupported COMMAND subcommand"),
                    false,
                )
            }
        }
    }

    /// Encodes the full `COMMAND` reply (one entry per registry command).
    fn encode_command_list(&self) -> Vec<u8> {
        match self.state.version {
            RespVersion::V2 => {
                use redis_protocol::resp2::types::OwnedFrame as F;
                let entries = REGISTRY
                    .iter()
                    .map(|spec| {
                        F::Array(vec![
                            F::BulkString(spec.name.as_bytes().to_vec()),
                            F::Integer(command_arity(spec)),
                            F::Array(
                                command_flags(spec)
                                    .iter()
                                    .map(|flag| F::BulkString(flag.as_bytes().to_vec()))
                                    .collect(),
                            ),
                            F::Integer(0),
                            F::Integer(0),
                            F::Integer(0),
                        ])
                    })
                    .collect();
                encode_frame_v2(&F::Array(entries))
            }
            RespVersion::V3 => {
                use redis_protocol::resp3::types::OwnedFrame as F;
                let entries = REGISTRY
                    .iter()
                    .map(|spec| F::Array {
                        data: vec![
                            blob3(spec.name.as_bytes()),
                            F::Number {
                                data: command_arity(spec),
                                attributes: None,
                            },
                            F::Array {
                                data: command_flags(spec)
                                    .iter()
                                    .map(|flag| blob3(flag.as_bytes()))
                                    .collect(),
                                attributes: None,
                            },
                            F::Number {
                                data: 0,
                                attributes: None,
                            },
                            F::Number {
                                data: 0,
                                attributes: None,
                            },
                            F::Number {
                                data: 0,
                                attributes: None,
                            },
                        ],
                        attributes: None,
                    })
                    .collect();
                encode_frame_v3(&F::Array {
                    data: entries,
                    attributes: None,
                })
            }
        }
    }

    /// Encodes `COMMAND INFO [names...]` (all commands when empty).
    fn encode_command_info(&self, names: &[Vec<u8>]) -> Vec<u8> {
        let wanted: Vec<&[u8]> = if names.is_empty() {
            REGISTRY.iter().map(|spec| spec.name.as_bytes()).collect()
        } else {
            names.iter().map(Vec::as_slice).collect()
        };
        match self.state.version {
            RespVersion::V2 => {
                use redis_protocol::resp2::types::OwnedFrame as F;
                let entries = wanted
                    .iter()
                    .map(|name| {
                        let upper = name.to_ascii_uppercase();
                        match lookup(core::str::from_utf8(&upper).unwrap_or("")) {
                            None => F::Null,
                            Some(spec) => F::Array(vec![
                                F::BulkString(spec.name.as_bytes().to_vec()),
                                F::Integer(command_arity(spec)),
                                F::Array(
                                    command_flags(spec)
                                        .iter()
                                        .map(|flag| F::BulkString(flag.as_bytes().to_vec()))
                                        .collect(),
                                ),
                                F::Integer(0),
                                F::Integer(0),
                                F::Integer(0),
                            ]),
                        }
                    })
                    .collect();
                encode_frame_v2(&F::Array(entries))
            }
            RespVersion::V3 => {
                use redis_protocol::resp3::types::OwnedFrame as F;
                let entries = wanted
                    .iter()
                    .map(|name| {
                        let upper = name.to_ascii_uppercase();
                        match lookup(core::str::from_utf8(&upper).unwrap_or("")) {
                            None => F::Null,
                            Some(spec) => F::Array {
                                data: vec![
                                    blob3(spec.name.as_bytes()),
                                    F::Number {
                                        data: command_arity(spec),
                                        attributes: None,
                                    },
                                    F::Array {
                                        data: command_flags(spec)
                                            .iter()
                                            .map(|flag| blob3(flag.as_bytes()))
                                            .collect(),
                                        attributes: None,
                                    },
                                    F::Number {
                                        data: 0,
                                        attributes: None,
                                    },
                                    F::Number {
                                        data: 0,
                                        attributes: None,
                                    },
                                    F::Number {
                                        data: 0,
                                        attributes: None,
                                    },
                                ],
                                attributes: None,
                            },
                        }
                    })
                    .collect();
                encode_frame_v3(&F::Array {
                    data: entries,
                    attributes: None,
                })
            }
        }
    }

    /// Encodes `COMMAND NAMES`: the registry's command names.
    fn encode_command_names(&self) -> Vec<u8> {
        match self.state.version {
            RespVersion::V2 => {
                use redis_protocol::resp2::types::OwnedFrame as F;
                encode_frame_v2(&F::Array(
                    REGISTRY
                        .iter()
                        .map(|spec| F::BulkString(spec.name.as_bytes().to_vec()))
                        .collect(),
                ))
            }
            RespVersion::V3 => {
                use redis_protocol::resp3::types::OwnedFrame as F;
                encode_frame_v3(&F::Array {
                    data: REGISTRY
                        .iter()
                        .map(|spec| blob3(spec.name.as_bytes()))
                        .collect(),
                    attributes: None,
                })
            }
        }
    }

    /// Encodes `COMMAND DOCS` as an empty map (no Redis docs universe here;
    /// the registry notes are the docs and ship in crate documentation).
    fn encode_command_docs(&self) -> Vec<u8> {
        match self.state.version {
            RespVersion::V2 => {
                use redis_protocol::resp2::types::OwnedFrame as F;
                encode_frame_v2(&F::Array(Vec::new()))
            }
            RespVersion::V3 => {
                use redis_protocol::resp3::types::OwnedFrame as F;
                encode_frame_v3(&F::Map {
                    data: std::collections::HashMap::default(),
                    attributes: None,
                })
            }
        }
    }
}

/// Redis arity: positive when fixed, negative minimum when variable.
fn command_arity(spec: &crate::command::CommandSpec) -> i64 {
    match spec.max_arity {
        Some(max) if max == spec.min_arity => i64::try_from(spec.min_arity).unwrap_or(i64::MAX),
        _ => -(i64::try_from(spec.min_arity).unwrap_or(i64::MAX)),
    }
}

/// Redis command flags derived from the command kind.
fn command_flags(spec: &crate::command::CommandSpec) -> &'static [&'static str] {
    use crate::command::CommandKind as K;
    match spec.kind {
        K::Write => &["write", "denyoom"],
        K::Connection => &["fast", "connection"],
        K::Read | K::Introspection => &["readonly", "fast"],
    }
}

/// Builds a RESP3 blob string without attributes.
fn blob3(bytes: &[u8]) -> redis_protocol::resp3::types::OwnedFrame {
    redis_protocol::resp3::types::OwnedFrame::BlobString {
        data: bytes.to_vec(),
        attributes: None,
    }
}

/// Encodes a version-agnostic reply.
fn encode_reply(version: RespVersion, reply: &Reply) -> Vec<u8> {
    match reply {
        Reply::Simple(text) => encode_simple(version, text),
        Reply::Bulk(bytes) => encode_bulk(version, bytes),
        Reply::Nil => encode_nil(version),
        Reply::Int(value) => encode_integer(version, *value),
        Reply::Error(message) => encode_error(version, message),
    }
}

/// Encodes a simple string.
fn encode_simple(version: RespVersion, text: &str) -> Vec<u8> {
    match version {
        RespVersion::V2 => encode_frame_v2(
            &redis_protocol::resp2::types::OwnedFrame::SimpleString(text.as_bytes().to_vec()),
        ),
        RespVersion::V3 => {
            encode_frame_v3(&redis_protocol::resp3::types::OwnedFrame::SimpleString {
                data: text.as_bytes().to_vec(),
                attributes: None,
            })
        }
    }
}

/// Encodes a bulk string (binary-safe).
fn encode_bulk(version: RespVersion, bytes: &[u8]) -> Vec<u8> {
    match version {
        RespVersion::V2 => encode_frame_v2(&redis_protocol::resp2::types::OwnedFrame::BulkString(
            bytes.to_vec(),
        )),
        RespVersion::V3 => encode_frame_v3(&redis_protocol::resp3::types::OwnedFrame::BlobString {
            data: bytes.to_vec(),
            attributes: None,
        }),
    }
}

/// Encodes nil (null bulk string in RESP2, null in RESP3).
fn encode_nil(version: RespVersion) -> Vec<u8> {
    match version {
        RespVersion::V2 => encode_frame_v2(&redis_protocol::resp2::types::OwnedFrame::Null),
        RespVersion::V3 => encode_frame_v3(&redis_protocol::resp3::types::OwnedFrame::Null),
    }
}

/// Encodes an integer.
fn encode_integer(version: RespVersion, value: i64) -> Vec<u8> {
    match version {
        RespVersion::V2 => {
            encode_frame_v2(&redis_protocol::resp2::types::OwnedFrame::Integer(value))
        }
        RespVersion::V3 => encode_frame_v3(&redis_protocol::resp3::types::OwnedFrame::Number {
            data: value,
            attributes: None,
        }),
    }
}

/// Encodes an error (without the leading `-`; added by framing).
fn encode_error(version: RespVersion, message: &str) -> Vec<u8> {
    match version {
        RespVersion::V2 => encode_frame_v2(&redis_protocol::resp2::types::OwnedFrame::Error(
            message.to_owned(),
        )),
        RespVersion::V3 => {
            encode_frame_v3(&redis_protocol::resp3::types::OwnedFrame::SimpleError {
                data: message.to_owned(),
                attributes: None,
            })
        }
    }
}

/// Encodes one RESP2 frame via the direct slice interface (no `codec`
/// feature, so no `tokio-util` enters this path).
fn encode_frame_v2(frame: &redis_protocol::resp2::types::OwnedFrame) -> Vec<u8> {
    use redis_protocol::resp2::types::Resp2Frame as _;
    let mut buf = vec![0u8; frame.encode_len(false)];
    redis_protocol::resp2::encode::encode(&mut buf, frame, false)
        .expect("pre-sized RESP2 encoding cannot fail");
    buf
}

/// Encodes one RESP3 frame via the direct slice interface (no `codec`
/// feature, so no `tokio-util` enters this path).
fn encode_frame_v3(frame: &redis_protocol::resp3::types::OwnedFrame) -> Vec<u8> {
    use redis_protocol::resp3::types::Resp3Frame as _;
    let mut buf = vec![0u8; frame.encode_len(false)];
    redis_protocol::resp3::encode::complete::encode(&mut buf, frame, false)
        .expect("pre-sized RESP3 encoding cannot fail");
    buf
}

#[cfg(test)]
mod tests {
    use super::*;
    use kivi_state::{ObjectStore, Operation, OperationResult};
    use std::sync::Mutex;

    #[derive(Debug, Default)]
    struct FakeExec {
        store: Mutex<ObjectStore>,
        now: u64,
    }

    impl FakeExec {
        fn new(now: u64) -> Self {
            Self {
                store: Mutex::new(ObjectStore::new()),
                now,
            }
        }
    }

    impl Executor for FakeExec {
        fn execute(
            &self,
            op: &Operation,
        ) -> Result<OperationResult, crate::translate::ExecuteError> {
            use kivi_state::Prepared;
            let now = kivi_types::UnixMicros::from_micros(self.now);
            let mut store = self.store.lock().expect("fake lock");
            // Mirror the engine: ephemeral prepare/apply, chunked never appears here.
            match store.prepare(op, now) {
                Ok(Prepared::Read(result)) => Ok(result),
                Ok(Prepared::Write(mutation)) => {
                    let outcome = store
                        .apply(&mutation, now)
                        .map_err(|_| crate::translate::ExecuteError::Rejected)?;
                    let is_persist = matches!(op, Operation::PersistExpiry { .. });
                    Ok(kivi_state::outcome_for(&mutation, &outcome, is_persist))
                }
                Err(kivi_state::OpError::WrongType { .. }) => {
                    Err(crate::translate::ExecuteError::WrongType)
                }
                Err(_) => Err(crate::translate::ExecuteError::Rejected),
            }
        }

        fn now_micros(&self) -> u64 {
            self.now
        }
    }

    /// Encodes one command as a RESP2 array of bulk strings.
    fn cmd(parts: &[&[u8]]) -> Vec<u8> {
        use redis_protocol::resp2::types::OwnedFrame as F;
        let frame = F::Array(
            parts
                .iter()
                .map(|part| F::BulkString(part.to_vec()))
                .collect(),
        );
        super::encode_frame_v2(&frame)
    }

    fn conn() -> RespConnection<FakeExec> {
        RespConnection::new(FakeExec::new(1_000_000_000), ConnConfig::default(), 7)
    }

    fn round_trip(connection: &mut RespConnection<FakeExec>, bytes: &[u8]) -> Vec<Vec<u8>> {
        connection.push(bytes).expect("push fits");
        let (replies, quit) = connection.drain();
        assert!(!quit, "no quit in these flows");
        replies
    }

    #[test]
    fn ping_echo_and_bootstrap_commands() {
        let mut connection = conn();
        let replies = round_trip(&mut connection, &cmd(&[b"PING"]));
        assert_eq!(replies.len(), 1);
        assert_eq!(replies[0], b"+PONG\r\n");
        let replies = round_trip(&mut connection, &cmd(&[b"PING", b"hi"]));
        assert_eq!(replies[0], b"$2\r\nhi\r\n");
        let binary = [0x00, 0xFF, 0x10];
        let replies = round_trip(&mut connection, &cmd(&[b"ECHO", &binary]));
        assert_eq!(replies[0], b"$3\r\n\x00\xFF\x10\r\n");
        // CLIENT bootstrap.
        let replies = round_trip(
            &mut connection,
            &cmd(&[b"CLIENT", b"SETINFO", b"LIB-NAME", b"redis-rs"]),
        );
        assert_eq!(replies[0], b"+OK\r\n");
        let replies = round_trip(&mut connection, &cmd(&[b"CLIENT", b"SETNAME", b"web"]));
        assert_eq!(replies[0], b"+OK\r\n");
        let replies = round_trip(&mut connection, &cmd(&[b"CLIENT", b"GETNAME"]));
        assert_eq!(replies[0], b"$3\r\nweb\r\n");
        // SELECT 0 maps; others are rejected, never faked.
        let replies = round_trip(&mut connection, &cmd(&[b"SELECT", b"0"]));
        assert_eq!(replies[0], b"+OK\r\n");
        let replies = round_trip(&mut connection, &cmd(&[b"SELECT", b"3"]));
        assert!(replies[0].starts_with(b"-ERR"));
        // AUTH never succeeds silently.
        let replies = round_trip(&mut connection, &cmd(&[b"AUTH", b"secret"]));
        assert!(replies[0].starts_with(b"-ERR"));
    }

    #[test]
    fn hello_switches_encoding_and_reports() {
        let mut connection = conn();
        // Bare HELLO reports RESP2 with an array.
        let replies = round_trip(&mut connection, &cmd(&[b"HELLO"]));
        assert!(replies[0].starts_with(b"*"));
        assert_eq!(connection.state.version, RespVersion::V2);
        // HELLO 3 upgrades; the reply itself is already a map.
        let replies = round_trip(&mut connection, &cmd(&[b"HELLO", b"3"]));
        assert!(replies[0].starts_with(b"%"));
        assert_eq!(connection.state.version, RespVersion::V3);
        // Nil now encodes as RESP3 null, not `$-1`.
        let replies = round_trip(&mut connection, &cmd(&[b"GET", b"missing"]));
        assert_eq!(replies[0], b"_\r\n");
        // Unknown versions fail with NOPROTO without switching.
        let replies = round_trip(&mut connection, &cmd(&[b"HELLO", b"9"]));
        assert!(replies[0].starts_with(b"-NOPROTO"));
        assert_eq!(connection.state.version, RespVersion::V3);
        // AUTH inside HELLO errors without switching back.
        let replies = round_trip(
            &mut connection,
            &cmd(&[b"HELLO", b"2", b"AUTH", b"default", b"x"]),
        );
        assert!(replies[0].starts_with(b"-ERR"));
        assert_eq!(connection.state.version, RespVersion::V3);
    }

    #[test]
    #[allow(clippy::too_many_lines)]
    fn data_commands_cover_the_exact_profile() {
        let mut connection = conn();
        // SET then GET then DEL then EXISTS.
        assert_eq!(
            round_trip(&mut connection, &cmd(&[b"SET", b"k", b"v"]))[0],
            b"+OK\r\n"
        );
        assert_eq!(
            round_trip(&mut connection, &cmd(&[b"GET", b"k"]))[0],
            b"$1\r\nv\r\n"
        );
        assert_eq!(
            round_trip(&mut connection, &cmd(&[b"EXISTS", b"k"]))[0],
            b":1\r\n"
        );
        assert_eq!(
            round_trip(&mut connection, &cmd(&[b"STRLEN", b"k"]))[0],
            b":1\r\n"
        );
        // Missing: GET nil, STRLEN 0, DEL 0, TTL -2.
        assert_eq!(
            round_trip(&mut connection, &cmd(&[b"GET", b"nope"]))[0],
            b"$-1\r\n"
        );
        assert_eq!(
            round_trip(&mut connection, &cmd(&[b"STRLEN", b"nope"]))[0],
            b":0\r\n"
        );
        assert_eq!(
            round_trip(&mut connection, &cmd(&[b"DEL", b"nope"]))[0],
            b":0\r\n"
        );
        assert_eq!(
            round_trip(&mut connection, &cmd(&[b"TTL", b"nope"]))[0],
            b":-2\r\n"
        );
        assert_eq!(
            round_trip(&mut connection, &cmd(&[b"DEL", b"k"]))[0],
            b":1\r\n"
        );
        // SET NX/XX.
        assert_eq!(
            round_trip(&mut connection, &cmd(&[b"SET", b"n", b"1", b"NX"]))[0],
            b"+OK\r\n"
        );
        assert_eq!(
            round_trip(&mut connection, &cmd(&[b"SET", b"n", b"2", b"NX"]))[0],
            b"$-1\r\n"
        );
        assert_eq!(
            round_trip(&mut connection, &cmd(&[b"SET", b"n", b"2", b"XX"]))[0],
            b"+OK\r\n"
        );
        // SETRANGE answers the new length; GETRANGE slices inclusively.
        assert_eq!(
            round_trip(&mut connection, &cmd(&[b"SETRANGE", b"s", b"5", b"hi"]))[0],
            b":7\r\n"
        );
        assert_eq!(
            round_trip(&mut connection, &cmd(&[b"GETRANGE", b"s", b"0", b"1"]))[0],
            b"$2\r\n\x00\x00\r\n"
        );
        assert_eq!(
            round_trip(&mut connection, &cmd(&[b"GETRANGE", b"s", b"-2", b"-1"]))[0],
            b"$2\r\nhi\r\n"
        );
        assert_eq!(
            round_trip(
                &mut connection,
                &cmd(&[b"GETRANGE", b"missing", b"0", b"3"])
            )[0],
            b"$0\r\n\r\n"
        );
        // Empty patches are pure length reads: missing keys answer 0 and
        // are NOT created, existing values and TTLs are untouched.
        assert_eq!(
            round_trip(
                &mut connection,
                &cmd(&[b"SETRANGE", b"e-missing", b"11", b""])
            )[0],
            b":0\r\n"
        );
        assert_eq!(
            round_trip(&mut connection, &cmd(&[b"EXISTS", b"e-missing"]))[0],
            b":0\r\n"
        );
        assert_eq!(
            round_trip(&mut connection, &cmd(&[b"SETRANGE", b"s", b"99", b""]))[0],
            b":7\r\n"
        );
        assert_eq!(
            round_trip(&mut connection, &cmd(&[b"GETRANGE", b"s", b"5", b"6"]))[0],
            b"$2\r\nhi\r\n"
        );
        assert_eq!(
            round_trip(&mut connection, &cmd(&[b"EXPIRE", b"s", b"100"]))[0],
            b":1\r\n"
        );
        assert_eq!(
            round_trip(&mut connection, &cmd(&[b"SETRANGE", b"s", b"99", b""]))[0],
            b":7\r\n"
        );
        let ttl = round_trip(&mut connection, &cmd(&[b"TTL", b"s"]))[0].clone();
        assert!(
            ttl.starts_with(b":") && !ttl.starts_with(b":-"),
            "empty patch preserves TTL, got {ttl:?}"
        );
        // Expiry round-trip.
        assert_eq!(
            round_trip(&mut connection, &cmd(&[b"EXPIRE", b"n", b"100"]))[0],
            b":1\r\n"
        );
        let ttl = round_trip(&mut connection, &cmd(&[b"TTL", b"n"]))[0].clone();
        assert!(ttl.starts_with(b":"));
        assert_eq!(
            round_trip(&mut connection, &cmd(&[b"PERSIST", b"n"]))[0],
            b":1\r\n"
        );
        assert_eq!(
            round_trip(&mut connection, &cmd(&[b"PERSIST", b"n"]))[0],
            b":0\r\n"
        );
        // Unsupported stays explicit, never faked.
        for name in [b"INCR".as_slice(), b"MGET".as_slice(), b"MSET".as_slice()] {
            let replies = round_trip(&mut connection, &cmd(&[name, b"k"]));
            assert!(
                replies[0].starts_with(b"-ERR"),
                "{name:?} must error, got {:?}",
                replies[0]
            );
        }
        // SET ... GET stays unsupported rather than materializing.
        let replies = round_trip(&mut connection, &cmd(&[b"SET", b"k", b"v", b"GET"]));
        assert!(replies[0].starts_with(b"-ERR"));
    }

    #[test]
    fn pipelining_preserves_order_and_partial_frames_wait() {
        let mut connection = conn();
        let mut batch = cmd(&[b"SET", b"a", b"1"]);
        batch.extend_from_slice(&cmd(&[b"SET", b"b", b"2"]));
        batch.extend_from_slice(&cmd(&[b"GET", b"a"]));
        batch.extend_from_slice(&cmd(&[b"GET", b"b"]));
        connection.push(&batch).expect("batch fits");
        let (replies, _) = connection.drain();
        assert_eq!(replies.len(), 4);
        assert_eq!(replies[0], b"+OK\r\n");
        assert_eq!(replies[1], b"+OK\r\n");
        assert_eq!(replies[2], b"$1\r\n1\r\n");
        assert_eq!(replies[3], b"$1\r\n2\r\n");
        // Partial frames wait for the rest.
        let mut connection = conn();
        let full = cmd(&[b"PING"]);
        connection
            .push(&full[..full.len() - 2])
            .expect("partial fits");
        let (replies, _) = connection.drain();
        assert!(replies.is_empty());
        connection.push(&full[full.len() - 2..]).expect("rest fits");
        let (replies, _) = connection.drain();
        assert_eq!(replies, vec![b"+PONG\r\n".to_vec()]);
    }

    #[test]
    fn malformed_input_never_panics_and_bounds_hold() {
        let mut connection = conn();
        // Not an array: clear error, connection stays usable afterwards.
        connection.push(b"+PING\r\n").expect("fits");
        let (replies, quit) = connection.drain();
        assert_eq!(replies.len(), 1);
        assert!(replies[0].starts_with(b"-ERR"));
        assert!(!quit);
        // Oversized arrays and bulks are rejected before allocation games.
        let mut small = RespConnection::new(
            FakeExec::new(0),
            ConnConfig {
                max_array_elements: 2,
                max_bulk_bytes: 4,
                ..ConnConfig::default()
            },
            1,
        );
        small.push(&cmd(&[b"GET", b"a", b"b"])).expect("fits");
        let (replies, _) = small.drain();
        assert!(replies[0].starts_with(b"-ERR"));
        small.push(&cmd(&[b"GET", b"12345"])).expect("fits");
        let (replies, _) = small.drain();
        assert!(replies[0].starts_with(b"-ERR"));
        // Random bytes never panic (close is acceptable, panic is not).
        let mut fuzz = conn();
        for chunk in [
            b"\xff\x00*".as_slice(),
            b"*-1\r\n".as_slice(),
            b"$100\r\nhi".as_slice(),
            b"*3\r\n$1\r\na".as_slice(),
        ] {
            fuzz.push(chunk).expect("fits");
            let _ = fuzz.drain();
        }
    }

    #[test]
    fn wrongtype_maps_to_its_prefix() {
        let mut connection = conn();
        connection
            .executor
            .execute(&Operation::CounterAdd {
                key: kivi_state::Key::from("c"),
                delta: 1,
            })
            .expect("counter seeds");
        let replies = round_trip(&mut connection, &cmd(&[b"GET", b"c"]));
        assert!(replies[0].starts_with(b"-WRONGTYPE"));
        let replies = round_trip(&mut connection, &cmd(&[b"STRLEN", b"c"]));
        assert!(replies[0].starts_with(b"-WRONGTYPE"));
    }
}

#[cfg(test)]
mod fuzz {
    use super::*;
    use proptest::prelude::*;

    /// Byte alphabet biased toward RESP structure (star, dollar, digits,
    /// CRLF) mixed with arbitrary binary: finds panics; hangs are impossible
    /// here because `drain` always consumes progress or stops.
    fn respish() -> impl Strategy<Value = Vec<u8>> {
        prop::collection::vec(
            prop_oneof![
                Just(42u8),
                Just(36u8),
                Just(13u8),
                Just(10u8),
                Just(58u8),
                Just(43u8),
                Just(45u8),
                any::<u8>(),
            ],
            0..256,
        )
    }

    #[derive(Debug)]
    struct VoidExec;

    impl Executor for VoidExec {
        fn execute(
            &self,
            op: &kivi_state::Operation,
        ) -> Result<kivi_state::OperationResult, crate::translate::ExecuteError> {
            // Total over operations: every op maps to some reply without I/O.
            if let Some(result) = Self::execute_semantic(op) {
                return Ok(result);
            }
            Ok(match op {
                kivi_state::Operation::Get { .. } => kivi_state::OperationResult::Value(None),
                kivi_state::Operation::GetRange { .. } => {
                    kivi_state::OperationResult::Value(Some(bytes::Bytes::new()))
                }
                kivi_state::Operation::BytesLength { .. } => {
                    kivi_state::OperationResult::Length(None)
                }
                kivi_state::Operation::SetConditional { .. }
                | kivi_state::Operation::SetConditionalChunked { .. }
                | kivi_state::Operation::SetConditionalFabric { .. } => {
                    kivi_state::OperationResult::ConditionalSet {
                        applied: false,
                        version: None,
                    }
                }
                kivi_state::Operation::Set { .. }
                | kivi_state::Operation::SetChunked { .. }
                | kivi_state::Operation::SetFabric { .. }
                | kivi_state::Operation::SetRange { .. }
                | kivi_state::Operation::BoundedCounterCreate { .. }
                | kivi_state::Operation::EscrowTransfer { .. }
                | kivi_state::Operation::SemaphoreCreate { .. } => {
                    // Creates answer stored (new version, never the rights).
                    kivi_state::OperationResult::Stored {
                        version: kivi_state::ObjectVersion::FIRST,
                    }
                }
                kivi_state::Operation::Delete { .. } => {
                    kivi_state::OperationResult::Deleted { existed: false }
                }
                kivi_state::Operation::Exists { .. } => kivi_state::OperationResult::Exists(false),
                kivi_state::Operation::CounterGet { .. } => {
                    kivi_state::OperationResult::Counter(None)
                }
                kivi_state::Operation::CounterAdd { .. } => {
                    kivi_state::OperationResult::CounterUpdated {
                        value: 0,
                        version: kivi_state::ObjectVersion::FIRST,
                    }
                }
                kivi_state::Operation::ExpireAt { .. } => {
                    kivi_state::OperationResult::ExpirySet { applied: false }
                }
                kivi_state::Operation::PersistExpiry { .. } => {
                    kivi_state::OperationResult::ExpiryPersisted { removed: false }
                }
                kivi_state::Operation::GetExpiry { .. } => {
                    kivi_state::OperationResult::Expiry(None)
                }
                // The RESP edge never issues transaction steps (no SQL, no
                // multi-key verbs): the executor names the conflict so the
                // totality of this match is explicit, never silent.
                kivi_state::Operation::TxnPrepare { .. }
                | kivi_state::Operation::TxnFinalize { .. }
                | kivi_state::Operation::TxnCommitLocal { .. } => {
                    kivi_state::OperationResult::TxnConflict
                }
                // No Redis verb reads bare versions; version checks arrive
                // through conditional verbs, never this reader.
                kivi_state::Operation::GetVersion { .. } => {
                    kivi_state::OperationResult::Version(None)
                }
                kivi_state::Operation::CommutativeGet { .. }
                | kivi_state::Operation::CommutativeAdd { .. }
                | kivi_state::Operation::BoundedCounterAdd { .. }
                | kivi_state::Operation::BoundedCounterGet { .. }
                | kivi_state::Operation::SemaphoreAcquire { .. }
                | kivi_state::Operation::SemaphoreRelease { .. }
                | kivi_state::Operation::SemaphoreInspect { .. }
                | kivi_state::Operation::LeaseAcquire { .. }
                | kivi_state::Operation::LeaseRenew { .. }
                | kivi_state::Operation::LeaseRelease { .. }
                | kivi_state::Operation::LeaseInspect { .. }
                | kivi_state::Operation::StreamCreate { .. }
                | kivi_state::Operation::StreamAppend { .. }
                | kivi_state::Operation::StreamRead { .. }
                | kivi_state::Operation::StreamTrim { .. } => {
                    unreachable!("semantic ops dispatch above")
                }
            })
        }

        fn now_micros(&self) -> u64 {
            0
        }
    }

    impl VoidExec {
        /// Answers the semantic verbs with inert dummies (`Some`) or
        /// declines (`None`): the RESP edge exposes no semantic verbs
        /// (native API only), so the void executor answers their shapes
        /// with dummies to keep the match total and explicit.
        fn execute_semantic(op: &kivi_state::Operation) -> Option<kivi_state::OperationResult> {
            Some(match op {
                kivi_state::Operation::CommutativeGet { .. } => {
                    kivi_state::OperationResult::CommutativeValue(None)
                }
                kivi_state::Operation::CommutativeAdd { .. } => {
                    kivi_state::OperationResult::CommutativeApplied
                }
                kivi_state::Operation::BoundedCounterAdd { .. } => {
                    kivi_state::OperationResult::BoundedUpdated {
                        value: 0,
                        version: kivi_state::ObjectVersion::FIRST,
                    }
                }
                kivi_state::Operation::BoundedCounterGet { .. } => {
                    kivi_state::OperationResult::BoundedValue {
                        value: None,
                        capacity: None,
                        share: None,
                    }
                }
                kivi_state::Operation::SemaphoreAcquire { .. } => {
                    kivi_state::OperationResult::SemaphoreAcquired
                }
                kivi_state::Operation::SemaphoreRelease { .. } => {
                    kivi_state::OperationResult::SemaphoreReleased { released: false }
                }
                kivi_state::Operation::SemaphoreInspect { .. } => {
                    kivi_state::OperationResult::SemaphoreLoad {
                        outstanding: 0,
                        capacity: 0,
                    }
                }
                kivi_state::Operation::LeaseAcquire { .. } => {
                    kivi_state::OperationResult::LeaseAcquired {
                        fencing: kivi_state::FencingToken::from_u64(1),
                        expires_at: kivi_types::UnixMicros::from_micros(0),
                    }
                }
                kivi_state::Operation::LeaseRenew { .. } => {
                    kivi_state::OperationResult::LeaseRenewed {
                        fencing: kivi_state::FencingToken::from_u64(1),
                        expires_at: kivi_types::UnixMicros::from_micros(0),
                    }
                }
                kivi_state::Operation::LeaseRelease { .. } => {
                    kivi_state::OperationResult::LeaseReleased { released: false }
                }
                kivi_state::Operation::LeaseInspect { .. } => {
                    kivi_state::OperationResult::LeaseInfo {
                        holder: None,
                        next_fencing: kivi_state::FencingToken::from_u64(1),
                    }
                }
                kivi_state::Operation::StreamCreate { .. } => {
                    kivi_state::OperationResult::StreamCreated
                }
                kivi_state::Operation::StreamAppend { .. } => {
                    kivi_state::OperationResult::StreamAppended { offset: 0 }
                }
                kivi_state::Operation::StreamRead { .. } => {
                    kivi_state::OperationResult::StreamEntries {
                        entries: Vec::new(),
                        next_offset: 0,
                    }
                }
                kivi_state::Operation::StreamTrim { .. } => {
                    kivi_state::OperationResult::StreamTrimmed { removed: 0 }
                }
                _ => return None,
            })
        }
    }

    proptest! {
        /// Arbitrary bytes through one connection never panic and always
        /// make progress (buffer drains or waits for more input).
        #[test]
        fn arbitrary_input_never_panics(bytes in respish()) {
            let mut connection = RespConnection::new(VoidExec, ConnConfig::default(), 1);
            // Tiny chunks force every partial-frame path.
            for chunk in bytes.chunks(7) {
                let _ = connection.push(chunk);
                let (replies, quit) = connection.drain();
                // Replies are always well-formed RESP in the connection version.
                for reply in &replies {
                    prop_assert!(!reply.is_empty());
                }
                if quit {
                    break;
                }
            }
        }
    }
}
