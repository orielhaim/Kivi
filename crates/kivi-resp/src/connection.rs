//! Per-connection RESP state: version, bounds, pipeline, and encoding.
//!
//! One connection speaks one version at a time (RESP2 initially, `HELLO 3`
//! upgrades to RESP3). Requests decode in order and replies encode in the
//! same order; pipelined commands never reorder. Decoding uses
//! `redis-protocol`'s direct slice interface with Kivi bounds applied
//! before any attacker-controlled length can allocate.

use crate::command::{REGISTRY, arity_ok, lookup};
use crate::error::RespError;
use crate::frame::{Command, Limits, Parsed, fold_command_name, parse_command};
use crate::translate::{
    Action, ExecuteError, Executor, Immediate, RedisOp, Reply, map_execute_error, map_parse_error,
    map_result, parse_unsigned, token_is, translate,
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
            // Argument count is not configurable: the parser's inline
            // capacity (`frame::MAX_ARGS`) is the bound, because that array
            // is what keeps a parse allocation-free.
            //
            // There is deliberately no separate pipeline-depth cap. The
            // previous `max_pipelined` answered `BUSY pipeline depth
            // exceeded` past 128 requests and *dropped* them, which Redis
            // never does — Redis processes whatever depth a client sends.
            // Fairness comes from the per-turn budget, not from refusing
            // work a client is entitled to.
            max_commands_per_turn: 64,
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

/// What one drain turn produced.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct DrainOutcome {
    /// Replies appended to the caller's output buffer.
    pub replies: u64,
    /// The connection must close once the output is flushed.
    pub close: bool,
}

/// One turn's decoded commands, carried from decoding to encoding.
///
/// The engine operations are held separately from the reply slots because they
/// are the only part that can suspend: encoding has to wait for every answer
/// before it writes the first reply, or replies would reach the client out of
/// order.
struct DecodedTurn {
    slots: Vec<Slot>,
    engine_ops: Vec<kivi_state::Operation>,
    outcome: DrainOutcome,
}

/// A RESP connection: buffer, version state, metrics, and bounded
/// decode/dispatch. I/O lives in the server; this type never touches a
/// socket, so unit tests drive it with byte strings.
///
/// Input is consumed with a cursor rather than by draining from the front,
/// and replies are appended into a caller-owned buffer rather than returned
/// as a `Vec` of `Vec`s. Both exist because the previous shape memmoved the
/// whole tail per command (quadratic in pipeline depth) and allocated and
/// copied every reply twice.
pub struct RespConnection<E> {
    /// Engine access (shared handle, cheap to clone if needed).
    executor: E,
    /// Resource bounds.
    config: ConnConfig,
    /// Mutable version/name state.
    pub state: ConnState,
    /// Undecoded input tail; `cursor` is how much of it has been consumed.
    buffer: Vec<u8>,
    /// Parse offset into `buffer`.
    cursor: usize,
    /// Set by `QUIT` so the turn loop can stop without threading a flag
    /// through every dispatch arm.
    quit_requested: bool,
    /// Per-connection metrics.
    pub metrics: ConnMetrics,
}

/// One command's answer, in request order.
///
/// An engine-bound command records *where* in the turn's result list it
/// belongs rather than waiting: the whole turn is issued together, so no
/// reply can be known until every command has been decoded.
#[derive(Debug)]
enum Slot {
    /// Bytes already encoded (bootstrap and introspection replies).
    Raw(Vec<u8>),
    /// A reply the turn can encode without the engine.
    Ready(Reply),
    /// One engine operation, answered as `redis`.
    Execute {
        /// How to shape the engine's answer.
        redis: RedisOp,
    },
    /// `SETRANGE`: write, then measure. Two operations, one reply.
    SetRange {
        /// Whether the write itself is issued (an empty patch is a pure read).
        writes: bool,
    },
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
            // Sized for a typical pipelined batch so the steady state does
            // not reallocate on the first turn.
            buffer: Vec::with_capacity(16 * 1024),
            cursor: 0,
            quit_requested: false,
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

    /// Executes every complete command currently buffered, up to the
    /// per-turn budget, appending encoded replies to `out` in request order.
    ///
    /// `out` is caller-owned so the server keeps one reusable buffer per
    /// connection and writes it once per turn. The previous shape returned a
    /// `Vec<Vec<u8>>` the caller then copied into a second buffer: an
    /// allocation and a full copy of every reply, for nothing.
    ///
    /// The input buffer is moved out for the duration of the turn because the
    /// parser hands out slices into it, and a slice cannot outlive a borrow
    /// of `self` that dispatch also needs mutably. Moving a `Vec` is three
    /// words, so this costs nothing.
    pub fn drain(&mut self, out: &mut Vec<u8>) -> DrainOutcome {
        let turn = self.decode_turn(out);
        let replies_before = out.len();
        let results = if turn.engine_ops.is_empty() {
            Vec::new()
        } else {
            self.executor.execute_batch(&turn.engine_ops)
        };
        self.encode_turn(out, &turn, &results, replies_before)
    }

    /// One turn that may wait on something other than this thread.
    ///
    /// Identical to [`Self::drain`] except that the executor is given the
    /// chance to suspend, which it needs for exactly one reason: a request
    /// whose answer depends on another thread — a durability proof, or a
    /// payload in the chunk lane — cannot be collected by a blocking call from
    /// a reactor, because that call would park the very thread that has to
    /// produce the answer. An executor with nothing to wait for takes the
    /// default and pays nothing.
    pub async fn drain_async(&mut self, out: &mut Vec<u8>) -> DrainOutcome {
        let turn = self.decode_turn(out);
        let replies_before = out.len();
        let results = if turn.engine_ops.is_empty() {
            Vec::new()
        } else {
            self.executor.execute_batch_async(&turn.engine_ops).await
        };
        self.encode_turn(out, &turn, &results, replies_before)
    }

    /// Decodes one turn: every command the per-turn budget allows, classified
    /// into the slots that encode it and the operations the engine must answer.
    ///
    /// Shared by both turn shapes, because the order of replies is settled here
    /// and a second copy of this loop is a second chance to read the same bytes
    /// differently.
    fn decode_turn(&mut self, out: &mut Vec<u8>) -> DecodedTurn {
        let mut buffer = core::mem::take(&mut self.buffer);
        let mut outcome = DrainOutcome::default();
        let limits = Limits {
            max_bulk_bytes: self.config.max_bulk_bytes,
            max_array_elements: self.config.max_array_elements,
        };
        let budget = self.config.max_commands_per_turn;

        // One turn collects every command, then executes the engine-bound ones
        // together, then encodes every reply in request order.
        //
        // The alternative — execute as you decode — makes pipelining actively
        // harmful: a pipeline of N commands became N sequential engine
        // round-trips, so a deeper pipeline cost proportionally more latency
        // for the same work. Measured against Redis, which answers a whole
        // pipeline in one event-loop pass, that was the single largest gap on
        // this edge.
        let mut slots: Vec<Slot> = Vec::with_capacity(budget);
        let mut engine_ops: Vec<kivi_state::Operation> = Vec::new();
        while outcome.replies < budget as u64 {
            let Some(tail) = buffer.get(self.cursor..) else {
                break;
            };
            if tail.is_empty() {
                break;
            }
            let (parsed, used) = parse_command(tail, &limits);
            match parsed {
                Parsed::Incomplete => break,
                Parsed::Command(command) => {
                    self.cursor += used;
                    self.classify(&command, &mut slots, &mut engine_ops);
                }
                refused => {
                    // A refused frame must still leave the buffer or the
                    // connection spins on it forever. A parser that consumed
                    // nothing cannot be trusted to advance, so drop everything
                    // it inspected.
                    self.cursor += used.max(tail.len());
                    self.write_refusal(&refused, out);
                    self.metrics.requests += 1;
                    outcome.replies += 1;
                    self.note_close(&mut outcome);
                    if outcome.close {
                        break;
                    }
                    continue;
                }
            }
            self.metrics.requests += 1;
            outcome.replies += 1;
            self.note_close(&mut outcome);
            if outcome.close {
                break;
            }
        }

        // One compaction per turn rather than one per command: the previous
        // code memoved the entire remaining tail for every command it decoded,
        // which is quadratic in pipeline depth.
        if self.cursor > 0 {
            let keep = buffer.len() - self.cursor;
            buffer.copy_within(self.cursor.., 0);
            buffer.truncate(keep);
            self.cursor = 0;
        }
        self.buffer = buffer;
        DecodedTurn {
            slots,
            engine_ops,
            outcome,
        }
    }

    /// Encodes a decoded turn's replies in request order.
    ///
    /// `results` lines up with the turn's engine operations positionally, which
    /// is why nothing here may reorder: a RESP client has no other way to tell
    /// which answer belongs to which command.
    fn encode_turn(
        &mut self,
        out: &mut Vec<u8>,
        turn: &DecodedTurn,
        results: &[Result<kivi_state::OperationResult, ExecuteError>],
        replies_before: usize,
    ) -> DrainOutcome {
        let mut next = 0usize;
        for slot in &turn.slots {
            match slot {
                Slot::Raw(bytes) => out.extend_from_slice(bytes),
                Slot::Ready(reply) => write_reply(out, self.state.version, reply),
                Slot::Execute { redis } => {
                    let Some(result) = results.get(next) else {
                        write_error(out, self.state.version, "ERR internal error");
                        continue;
                    };
                    next += 1;
                    write_reply(
                        out,
                        self.state.version,
                        &shape(result, *redis, &self.executor),
                    );
                }
                Slot::SetRange { writes } => {
                    let consumed = 1 + usize::from(*writes);
                    let Some(results) = results.get(next..next + consumed) else {
                        write_error(out, self.state.version, "ERR internal error");
                        continue;
                    };
                    next += consumed;
                    if let Some(Err(error)) = results.first() {
                        write_reply(out, self.state.version, &map_execute_error(*error));
                        continue;
                    }
                    match results.last() {
                        Some(Ok(kivi_state::OperationResult::Length(value))) => write_integer(
                            out,
                            self.state.version,
                            value.map_or(0, |len| i64::try_from(len).unwrap_or(i64::MAX)),
                        ),
                        _ => write_error(out, self.state.version, "ERR internal error"),
                    }
                }
            }
        }
        self.metrics.bytes_out += (out.len() - replies_before) as u64;
        turn.outcome
    }

    /// Records a close request the turn loop observed.
    fn note_close(&mut self, outcome: &mut DrainOutcome) {
        if core::mem::take(&mut self.quit_requested) {
            outcome.close = true;
        }
    }

    /// Answers a frame the parser refused, keeping or closing the
    /// connection exactly as the refusal class says.
    fn write_refusal(&mut self, refused: &Parsed<'_>, out: &mut Vec<u8>) {
        let version = self.state.version;
        self.metrics.protocol_errors += 1;
        let message: &str = match refused {
            Parsed::TooManyArgs => "ERR too many arguments",
            Parsed::TooLarge => "ERR bulk string exceeds bound",
            // An empty array, a non-string element, and a well-formed
            // non-command frame are all "this is not a command" to a Redis
            // client, and all leave the connection usable.
            Parsed::Malformed | Parsed::Empty | Parsed::BadArgument | Parsed::NotACommand => {
                "ERR malformed request: expected an array of bulk strings"
            }
            Parsed::Incomplete | Parsed::Command(_) => return,
        };
        write_error(out, version, message);
    }

    /// Whether any bytes remain that no command has been decoded from yet.
    ///
    /// Deliberately distinct from "the buffer is empty": a frame split across
    /// two reads leaves bytes behind that no command can come from yet, and
    /// the server must not read that as "nothing to do".
    #[must_use]
    pub fn has_unconsumed(&self) -> bool {
        self.cursor < self.buffer.len()
    }

    /// Dispatches one decoded request to an encoded reply (plus close flag).
    #[allow(clippy::too_many_lines, clippy::needless_pass_by_value)]
    fn classify(
        &mut self,
        command: &Command<'_>,
        slots: &mut Vec<Slot>,
        engine_ops: &mut Vec<kivi_state::Operation>,
    ) {
        let name = command.name();
        let args = command.args();

        // One case-insensitive fold, then one match. The previous path
        // compared five bootstrap literals, then scanned a 30-entry registry
        // for a `&str` (which meant UTF-8-validating the name), then matched
        // the name a third time inside `translate`.
        let Some(folded) = fold_command_name(name) else {
            self.metrics.protocol_errors += 1;
            slots.push(Slot::Ready(Reply::Error(malformed_request())));
            return;
        };
        // The registry is keyed by upper-case names, so the *folded* name is
        // what has to be looked up. Passing the raw name here made every
        // lower-case command "unknown" — which is what `redis-cli` sends.
        let folded_name = &folded[..name.len()];
        match folded_name {
            b"HELLO" => {
                let mut out = Vec::new();
                self.dispatch_hello(args, &mut out);
                slots.push(Slot::Raw(out));
                return;
            }
            b"CLIENT" => {
                let mut out = Vec::new();
                self.dispatch_client(args, &mut out);
                slots.push(Slot::Raw(out));
                return;
            }
            b"SELECT" => {
                let mut out = Vec::new();
                self.dispatch_select(args, &mut out);
                slots.push(Slot::Raw(out));
                return;
            }
            b"COMMAND" => {
                let mut out = Vec::new();
                self.dispatch_command(args, &mut out);
                slots.push(Slot::Raw(out));
                return;
            }
            b"AUTH" => {
                self.metrics.unsupported += 1;
                slots.push(Slot::Ready(Reply::Error(
                    "ERR AUTH not implemented in Kivi RESP profile v1; no authentication is enforced"
                        .to_owned(),
                )));
                return;
            }
            _ => {}
        }

        let Some(spec) = lookup(std::str::from_utf8(folded_name).unwrap_or("")) else {
            self.metrics.protocol_errors += 1;
            let printable = String::from_utf8_lossy(name);
            slots.push(Slot::Ready(Reply::Error(format!(
                "ERR unknown command '{printable}'"
            ))));
            return;
        };
        if !arity_ok(spec, args.len() + 1) {
            self.metrics.protocol_errors += 1;
            slots.push(Slot::Ready(Reply::Error(format!(
                "ERR wrong number of arguments for '{}' command",
                spec.name.to_lowercase()
            ))));
            return;
        }
        if spec.compat == crate::command::CompatClass::Unsupported {
            self.metrics.unsupported += 1;
            slots.push(Slot::Ready(map_parse_error(&RespError::Unsupported {
                command: spec.name.to_owned(),
            })));
            return;
        }
        // Multi-key forms are rejected, never faked with a loop.
        if matches!(spec.name, "DEL" | "EXISTS") && args.len() != 1 {
            self.metrics.protocol_errors += 1;
            slots.push(Slot::Ready(Reply::Error(
                "ERR multi-key form unsupported in Kivi RESP profile v1; send single-key requests"
                    .to_owned(),
            )));
            return;
        }
        let action = match translate(name, args, self.executor.now()) {
            Ok(action) => action,
            Err(error) => {
                match &error {
                    RespError::Unsupported { .. } => self.metrics.unsupported += 1,
                    _ => self.metrics.protocol_errors += 1,
                }
                slots.push(Slot::Ready(map_parse_error(&error)));
                return;
            }
        };
        // Value-size bound: RESP bulk strings larger than the configured
        // ceiling are refused here rather than after the engine has copied
        // them.
        if let Action::Execute { op, .. } = &action {
            let size = match op {
                kivi_state::Operation::SetConditional { value, .. }
                | kivi_state::Operation::Set { value, .. } => Some(value.len()),
                kivi_state::Operation::SetRange { patch, .. } => Some(patch.len()),
                _ => None,
            };
            if size.is_some_and(|len| len > self.config.max_value_bytes) {
                self.metrics.protocol_errors += 1;
                slots.push(Slot::Ready(Reply::Error(
                    "ERR value exceeds RESP bound; use native streaming".to_owned(),
                )));
                return;
            }
        }
        match action {
            Action::Reply(immediate) => slots.push(Slot::Ready(immediate_reply(immediate))),
            Action::Execute { op, redis } => {
                if redis == RedisOp::SetRange {
                    // An empty patch is a pure length read: Redis treats it as
                    // a complete no-op (missing keys answer 0 and are NOT
                    // created, TTLs are untouched), so no mutation is issued
                    // and there is nothing to race.
                    let writes = !matches!(
                        &op,
                        kivi_state::Operation::SetRange { patch, .. } if patch.is_empty()
                    );
                    if writes {
                        engine_ops.push(op.clone());
                    }
                    engine_ops.push(kivi_state::Operation::BytesLength {
                        key: op.key().clone(),
                    });
                    slots.push(Slot::SetRange { writes });
                } else {
                    engine_ops.push(op);
                    slots.push(Slot::Execute { redis });
                }
            }
        }
    }

    /// Dispatches `HELLO [protover [AUTH user pass] [SETNAME name]]`.
    fn dispatch_hello(&mut self, args: &[&[u8]], out: &mut Vec<u8>) {
        let version = self.state.version;
        // No args: report the current context without switching.
        if args.is_empty() {
            self.encode_hello(out);
            return;
        }
        let target = match args[0] {
            b"2" => RespVersion::V2,
            b"3" => RespVersion::V3,
            _ => {
                self.metrics.protocol_errors += 1;
                write_error(
                    out,
                    version,
                    "NOPROTO sorry, this protocol version is not supported",
                );
                return;
            }
        };
        // Optional AUTH / SETNAME tail.
        let mut index = 1;
        while index < args.len() {
            let token = args[index];
            if token_is(token, b"AUTH") {
                // Security is a separate feature: never succeed silently,
                // never switch versions on the way out.
                self.metrics.unsupported += 1;
                write_error(
                    out,
                    version,
                    "ERR AUTH not implemented in Kivi RESP profile v1; no authentication is enforced",
                );
                return;
            }
            if token_is(token, b"SETNAME") {
                index += 1;
                let name = args.get(index).map_or_else(String::new, |bytes| {
                    String::from_utf8_lossy(bytes).into_owned()
                });
                self.state.name = Some(name);
                index += 1;
                continue;
            }
            self.metrics.protocol_errors += 1;
            write_error(out, version, "ERR syntax error in HELLO options");
            return;
        }
        self.state.version = target;
        self.encode_hello(out);
    }

    /// Encodes the `HELLO` reply in the (possibly just switched) version:
    /// a flat array in RESP2, a map in RESP3.
    fn encode_hello(&self, out: &mut Vec<u8>) {
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
                out.extend_from_slice(&encode_frame_v2(&frame));
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
                out.extend_from_slice(&encode_frame_v3(&F::Map {
                    data: map,
                    attributes: None,
                }));
            }
        }
    }

    /// Dispatches `CLIENT ...` bootstrap subcommands.
    fn dispatch_client(&mut self, args: &[&[u8]], out: &mut Vec<u8>) {
        let version = self.state.version;
        let Some((first, rest)) = args.split_first() else {
            self.metrics.protocol_errors += 1;
            write_error(
                out,
                version,
                "ERR wrong number of arguments for 'client' command",
            );
            return;
        };
        if token_is(first, b"SETINFO") && rest.len() == 2 {
            write_simple(out, version, "OK");
        } else if token_is(first, b"SETNAME") && rest.len() == 1 {
            let name = rest[0];
            if name.len() > 1024 {
                self.metrics.protocol_errors += 1;
                write_error(out, version, "ERR client name too long");
                return;
            }
            self.state.name = Some(String::from_utf8_lossy(name).into_owned());
            write_simple(out, version, "OK");
        } else if token_is(first, b"GETNAME") && rest.is_empty() {
            match &self.state.name {
                Some(name) => write_bulk(out, version, name.as_bytes()),
                None => write_nil(out, version),
            }
        } else {
            self.metrics.protocol_errors += 1;
            write_error(
                out,
                version,
                "ERR unsupported CLIENT subcommand in Kivi RESP profile v1",
            );
        }
    }

    /// Dispatches `SELECT index`: `0` maps to the configured namespace,
    /// anything else is rejected (never faked as extra databases).
    fn dispatch_select(&mut self, args: &[&[u8]], out: &mut Vec<u8>) {
        let version = self.state.version;
        let [index] = args else {
            self.metrics.protocol_errors += 1;
            write_error(
                out,
                version,
                "ERR wrong number of arguments for 'select' command",
            );
            return;
        };
        match parse_unsigned(index) {
            Ok(0) => write_simple(out, version, "OK"),
            Ok(_) => {
                self.metrics.protocol_errors += 1;
                write_error(
                    out,
                    version,
                    "ERR database numbers other than 0 are unsupported in Kivi RESP profile v1",
                );
            }
            Err(_) => {
                self.metrics.protocol_errors += 1;
                write_error(out, version, "ERR value is not an integer or out of range");
            }
        }
    }

    /// Dispatches `COMMAND [COUNT|INFO|DOCS|LIST]` from the registry.
    fn dispatch_command(&mut self, args: &[&[u8]], out: &mut Vec<u8>) {
        let version = self.state.version;
        if args.is_empty() {
            out.extend_from_slice(&self.encode_command_list());
            return;
        }
        let sub = args[0];
        let rest = &args[1..];
        if token_is(sub, b"COUNT") && rest.is_empty() {
            write_integer(
                out,
                version,
                i64::try_from(REGISTRY.len()).unwrap_or(i64::MAX),
            );
        } else if token_is(sub, b"LIST") {
            out.extend_from_slice(&self.encode_command_names());
        } else if token_is(sub, b"INFO") {
            out.extend_from_slice(&self.encode_command_info(rest));
        } else if token_is(sub, b"DOCS") {
            out.extend_from_slice(&self.encode_command_docs());
        } else {
            self.metrics.protocol_errors += 1;
            write_error(out, version, "ERR unsupported COMMAND subcommand");
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
    fn encode_command_info(&self, names: &[&[u8]]) -> Vec<u8> {
        let wanted: Vec<&[u8]> = if names.is_empty() {
            REGISTRY.iter().map(|spec| spec.name.as_bytes()).collect()
        } else {
            names.to_vec()
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

/// The one error string every "this is not a command" refusal uses.
fn malformed_request() -> String {
    "ERR malformed request: expected an array of bulk strings".to_owned()
}

/// The version-agnostic reply an immediate action produces.
fn immediate_reply(immediate: Immediate) -> Reply {
    match immediate {
        Immediate::Simple(text) => Reply::Simple(text),
        Immediate::Echo(bytes) => Reply::Bulk(bytes),
        Immediate::Nil => Reply::Nil,
        Immediate::Integer(value) => Reply::Int(value),
        Immediate::Quit => Reply::Simple("OK"),
    }
}

/// Maps one engine answer onto its Redis reply.
fn shape<E: Executor>(
    result: &Result<kivi_state::OperationResult, ExecuteError>,
    redis: RedisOp,
    executor: &E,
) -> Reply {
    match result {
        Ok(value) => map_result(value, redis, executor.now()),
        Err(error) => map_execute_error(*error),
    }
}

/// Appends a version-agnostic reply to `out`.
///
/// Every writer here appends into a buffer the connection reuses, so a reply
/// costs no allocation of its own and a pipelined batch costs one write. The
/// previous encoders built an owned frame — copying the payload — then
/// allocated an exactly-sized buffer and encoded into it, and the server
/// copied the whole batch into a third.
pub fn write_reply(out: &mut Vec<u8>, version: RespVersion, reply: &Reply) {
    match reply {
        Reply::Simple(text) => write_simple(out, version, text),
        Reply::Bulk(bytes) => write_bulk(out, version, bytes),
        Reply::Nil => write_nil(out, version),
        Reply::Int(value) => write_integer(out, version, *value),
        Reply::Error(message) => write_error(out, version, message),
    }
}

/// Appends a simple string.
pub fn write_simple(out: &mut Vec<u8>, _version: RespVersion, text: &str) {
    // RESP2 and RESP3 spell a simple string the same way.
    out.push(b'+');
    out.extend_from_slice(text.as_bytes());
    out.extend_from_slice(b"\r\n");
}

/// Appends a bulk string (binary-safe).
pub fn write_bulk(out: &mut Vec<u8>, _version: RespVersion, bytes: &[u8]) {
    // RESP2's bulk string and RESP3's blob string have the same wire form;
    // only RESP2's *null* is spelled differently (see `write_nil`).
    out.push(b'$');
    write_usize(out, bytes.len());
    out.extend_from_slice(b"\r\n");
    out.extend_from_slice(bytes);
    out.extend_from_slice(b"\r\n");
}

/// Appends nil (null bulk string in RESP2, `_` in RESP3).
pub fn write_nil(out: &mut Vec<u8>, version: RespVersion) {
    match version {
        RespVersion::V2 => out.extend_from_slice(b"$-1\r\n"),
        RespVersion::V3 => out.extend_from_slice(b"_\r\n"),
    }
}

/// Appends an integer.
pub fn write_integer(out: &mut Vec<u8>, _version: RespVersion, value: i64) {
    // RESP2's integer and RESP3's number have the same wire form.
    out.push(b':');
    write_i64(out, value);
    out.extend_from_slice(b"\r\n");
}

/// Appends an error (without the leading `-`; framing adds it).
pub fn write_error(out: &mut Vec<u8>, _version: RespVersion, message: &str) {
    // RESP2's error and RESP3's simple error have the same wire form.
    out.push(b'-');
    out.extend_from_slice(message.as_bytes());
    out.extend_from_slice(b"\r\n");
}

/// Appends a decimal length. Hand-rolled because `format!` allocates and
/// every bulk reply needs one.
fn write_usize(out: &mut Vec<u8>, value: usize) {
    let mut digits = [0u8; 20];
    let mut index = digits.len();
    let mut remaining = value;
    loop {
        index -= 1;
        digits[index] = b'0' + u8::try_from(remaining % 10).unwrap_or(0);
        remaining /= 10;
        if remaining == 0 {
            break;
        }
    }
    out.extend_from_slice(&digits[index..]);
}

/// Appends a signed decimal, including `i64::MIN` (whose magnitude does not
/// fit in an `i64`).
fn write_i64(out: &mut Vec<u8>, value: i64) {
    if value < 0 {
        out.push(b'-');
        let magnitude = value.unsigned_abs();
        let mut digits = [0u8; 20];
        let mut index = digits.len();
        let mut remaining = magnitude;
        loop {
            index -= 1;
            digits[index] = b'0' + u8::try_from(remaining % 10).unwrap_or(0);
            remaining /= 10;
            if remaining == 0 {
                break;
            }
        }
        out.extend_from_slice(&digits[index..]);
    } else {
        write_usize(out, usize::try_from(value).unwrap_or(0));
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

    #[derive(Debug)]
    struct FakeExec {
        store: Mutex<ObjectStore>,
        now: kivi_types::WallTimestamp,
    }

    /// Records the batches it is asked to execute, and answers with a
    /// deterministic per-key string so reply ordering is visible on the wire.
    #[derive(Debug, Default)]
    struct BatchSpy {
        batches: Mutex<Vec<Vec<Operation>>>,
        now: kivi_types::WallTimestamp,
    }

    impl BatchSpy {
        fn new() -> Self {
            Self {
                batches: Mutex::new(Vec::new()),
                now: kivi_types::WallTimestamp::from_micros(1_000_000_000),
            }
        }

        fn batches(&self) -> Vec<Vec<Operation>> {
            self.batches.lock().expect("spy lock").clone()
        }

        fn answer(op: &Operation) -> OperationResult {
            let key = String::from_utf8_lossy(op.key().as_bytes()).into_owned();
            match op {
                // `SET` compiles to `SetConditional`, and the engine answers
                // a conditional write with `ConditionalSet`.
                Operation::SetConditional { .. } => OperationResult::ConditionalSet {
                    applied: true,
                    version: Some(kivi_state::ObjectVersion::from_u64(1)),
                },
                Operation::Get { .. } | Operation::GetRange { .. } => {
                    OperationResult::Value(Some(bytes::Bytes::from(format!("{key}:1"))))
                }
                Operation::Exists { .. } => OperationResult::Exists(true),
                Operation::Delete { .. } => OperationResult::Deleted { existed: true },
                Operation::BytesLength { .. } => OperationResult::Length(Some(1)),
                _ => OperationResult::Exists(false),
            }
        }
    }

    impl Executor for BatchSpy {
        fn execute(
            &self,
            op: &Operation,
        ) -> Result<OperationResult, crate::translate::ExecuteError> {
            Ok(Self::answer(op))
        }

        fn execute_batch(
            &self,
            ops: &[Operation],
        ) -> Vec<Result<OperationResult, crate::translate::ExecuteError>> {
            self.batches.lock().expect("spy lock").push(ops.to_vec());
            ops.iter().map(|op| Ok(Self::answer(op))).collect()
        }

        fn now(&self) -> kivi_types::WallTimestamp {
            self.now
        }
    }

    impl FakeExec {
        fn new(now: kivi_types::WallTimestamp) -> Self {
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
            let now = self.now;
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

        fn now(&self) -> kivi_types::WallTimestamp {
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
        RespConnection::new(
            FakeExec::new(kivi_types::WallTimestamp::from_micros(1_000_000_000)),
            ConnConfig::default(),
            7,
        )
    }

    /// Splits a RESP2 reply stream into per-reply frames.
    ///
    /// Tests assert on framing rather than on a container shape, so this
    /// walks the bytes the way a client would.
    pub(crate) fn split_frames(bytes: &[u8]) -> Vec<Vec<u8>> {
        let mut frames = Vec::new();
        let mut at = 0usize;
        while at < bytes.len() {
            let len = split_one(&bytes[at..]).expect("well-formed frame");
            assert!(len > 0, "frame made no progress in {bytes:?}");
            assert!(at + len <= bytes.len(), "truncated frame in {bytes:?}");
            frames.push(bytes[at..at + len].to_vec());
            at += len;
        }
        frames
    }

    /// Length of the single frame at the front of `bytes`.
    pub(crate) fn split_one(bytes: &[u8]) -> Option<usize> {
        let kind = *bytes.first()?;
        let line_end = bytes.windows(2).position(|pair| pair == b"\r\n")? + 2;
        match kind {
            b'$' | b'=' | b'(' => {
                let count: i64 = std::str::from_utf8(&bytes[1..line_end - 2])
                    .ok()?
                    .parse()
                    .ok()?;
                Some(if count < 0 {
                    line_end
                } else {
                    line_end + usize::try_from(count).ok()? + 2
                })
            }
            b'*' | b'%' | b'~' | b'>' => {
                let count: i64 = std::str::from_utf8(&bytes[1..line_end - 2])
                    .ok()?
                    .parse()
                    .ok()?;
                if count < 0 {
                    return Some(line_end);
                }
                let mut at = line_end;
                for _ in 0..count {
                    at += split_one(&bytes[at..])?;
                }
                Some(at)
            }
            _ => Some(line_end),
        }
    }

    /// Drains one push and returns the replies as separate frames.
    fn drain_frames(connection: &mut RespConnection<FakeExec>) -> (Vec<Vec<u8>>, bool) {
        let mut out = Vec::new();
        let outcome = connection.drain(&mut out);
        (split_frames(&out), outcome.close)
    }

    fn round_trip(connection: &mut RespConnection<FakeExec>, bytes: &[u8]) -> Vec<Vec<u8>> {
        connection.push(bytes).expect("push fits");
        let (replies, quit) = drain_frames(connection);
        assert!(!quit, "no quit in these flows");
        replies
    }

    /// Every command name is case-insensitive, as in Redis.
    ///
    /// This is not a detail: `redis-cli` sends `set`, `get` and `ttl` in lower
    /// case, while most libraries send upper case. A dispatcher that folds
    /// the name for one lookup and not another answers real clients
    /// `ERR unknown command` while passing every upper-case test.
    #[test]
    fn command_names_are_case_insensitive() {
        for (lower, upper) in [
            (&b"get"[..], &b"GET"[..]),
            (&b"set"[..], &b"SET"[..]),
            (&b"del"[..], &b"DEL"[..]),
            (&b"exists"[..], &b"EXISTS"[..]),
            (&b"strlen"[..], &b"STRLEN"[..]),
            (&b"ttl"[..], &b"TTL"[..]),
            (&b"expire"[..], &b"EXPIRE"[..]),
            (&b"getrange"[..], &b"GETRANGE"[..]),
            (&b"setrange"[..], &b"SETRANGE"[..]),
            (&b"persist"[..], &b"PERSIST"[..]),
            (&b"ping"[..], &b"PING"[..]),
            (&b"hello"[..], &b"HELLO"[..]),
            (&b"client"[..], &b"CLIENT"[..]),
            (&b"select"[..], &b"SELECT"[..]),
            (&b"quit"[..], &b"QUIT"[..]),
        ] {
            let mut lower_conn = conn();
            let mut upper_conn = conn();
            let lower_replies = round_trip(&mut lower_conn, &cmd(&[lower, b"k"]));
            let upper_replies = round_trip(&mut upper_conn, &cmd(&[upper, b"k"]));
            assert_eq!(
                lower_replies,
                upper_replies,
                "{} and {} must dispatch identically",
                String::from_utf8_lossy(lower),
                String::from_utf8_lossy(upper)
            );
            assert!(
                !lower_replies.is_empty() && !lower_replies[0].starts_with(b"-ERR unknown"),
                "{} was reported unknown: {lower_replies:?}",
                String::from_utf8_lossy(lower)
            );
        }
    }

    /// The registry is keyed by upper case, so the fold has to reach the
    /// lookup, not only the bootstrap match.
    #[test]
    fn a_known_command_in_lower_case_is_never_reported_unknown() {
        let mut connection = conn();
        // `TTL` on a missing key is `-2`, never an unknown-command error.
        let replies = round_trip(&mut connection, &cmd(&[b"ttl", b"nope"]));
        assert_eq!(replies[0], b":-2\r\n");
        let replies = round_trip(&mut connection, &cmd(&[b"pErSiSt", b"nope"]));
        assert_eq!(replies[0], b":0\r\n");
        let replies = round_trip(&mut connection, &cmd(&[b"sTrLeN", b"nope"]));
        assert_eq!(replies[0], b":0\r\n");
        // An actually unknown command is still unknown, and names it.
        let replies = round_trip(&mut connection, &cmd(&[b"frobnicate"]));
        assert!(
            replies[0].starts_with(b"-ERR unknown command"),
            "{replies:?}"
        );
    }

    /// A pipeline is issued as one batch, not as N round-trips.
    ///
    /// This is the property that makes pipelining worth anything. Executing
    /// as it decodes turns a pipeline of N commands into N sequential engine
    /// round-trips, so a deeper pipeline costs proportionally more latency for
    /// the same work — the opposite of the intent, and measured against
    /// Redis, the largest single gap on this edge.
    #[test]
    fn a_pipeline_is_issued_as_one_batch_in_request_order() {
        let executor = BatchSpy::new();
        let mut connection = RespConnection::new(executor, ConnConfig::default(), 1);
        let mut batch = cmd(&[b"SET", b"a", b"1"]);
        batch.extend_from_slice(&cmd(&[b"GET", b"a"]));
        batch.extend_from_slice(&cmd(&[b"SET", b"b", b"22"]));
        batch.extend_from_slice(&cmd(&[b"GET", b"b"]));
        batch.extend_from_slice(&cmd(&[b"DEL", b"a"]));
        connection.push(&batch).expect("batch fits");
        let mut out = Vec::new();
        let outcome = connection.drain(&mut out);
        assert_eq!(outcome.replies, 5);

        let executor = connection.executor;
        assert_eq!(
            executor.batches(),
            vec![vec![
                // `SET` compiles to `SetConditional` so the write stays atomic
                // at the owning tablet; that is the operation, not a wrapper.
                kivi_state::Operation::SetConditional {
                    key: kivi_state::Key::from("a"),
                    value: bytes::Bytes::from_static(b"1"),
                    condition: kivi_state::SetCondition::Always,
                    expiry: kivi_state::ExpiryPolicy::Clear,
                },
                kivi_state::Operation::Get {
                    key: kivi_state::Key::from("a"),
                },
                kivi_state::Operation::SetConditional {
                    key: kivi_state::Key::from("b"),
                    value: bytes::Bytes::from_static(b"22"),
                    condition: kivi_state::SetCondition::Always,
                    expiry: kivi_state::ExpiryPolicy::Clear,
                },
                kivi_state::Operation::Get {
                    key: kivi_state::Key::from("b"),
                },
                kivi_state::Operation::Delete {
                    key: kivi_state::Key::from("a"),
                },
            ]],
            "the whole pipeline must arrive in one batch, in order"
        );
        // The spy answers `key:1` for reads, so the wire proves the answers
        // came back positionally rather than in completion order.
        let frames = split_frames(&out);
        assert_eq!(frames[0], b"+OK\r\n");
        assert_eq!(frames[1], b"$3\r\na:1\r\n");
        assert_eq!(frames[2], b"+OK\r\n");
        assert_eq!(frames[3], b"$3\r\nb:1\r\n");
        assert_eq!(frames[4], b":1\r\n");
    }

    /// A pipelined refusal does not shift anybody else's answer.
    #[test]
    fn a_refusal_inside_a_pipeline_keeps_every_answer_in_place() {
        let executor = BatchSpy::new();
        let mut connection = RespConnection::new(executor, ConnConfig::default(), 1);
        let mut batch = cmd(&[b"FROBNICATE"]);
        batch.extend_from_slice(&cmd(&[b"GET", b"a"]));
        batch.extend_from_slice(&cmd(&[b"GET"]));
        batch.extend_from_slice(&cmd(&[b"GET", b"b"]));
        connection.push(&batch).expect("batch fits");
        let mut out = Vec::new();
        let outcome = connection.drain(&mut out);
        assert_eq!(outcome.replies, 4);
        let frames = split_frames(&out);
        assert!(frames[0].starts_with(b"-ERR unknown command"));
        assert_eq!(frames[1], b"$3\r\na:1\r\n");
        assert!(frames[2].starts_with(b"-ERR wrong number of arguments"));
        assert_eq!(frames[3], b"$3\r\nb:1\r\n");
        // Only the two well-formed GETs reached the engine.
        assert_eq!(connection.executor.batches().len(), 1);
        assert_eq!(connection.executor.batches()[0].len(), 2);
    }

    /// A pipeline deeper than one turn's budget still answers once, in order.
    ///
    /// The turn budget is what stops one client monopolising the frontend, so
    /// a deep pipeline necessarily spans several turns. Each turn allocates
    /// its own slots and issues its own engine batch: a slot or an operation
    /// left over from the previous turn would show up here as a shifted or
    /// duplicated reply.
    #[test]
    fn a_pipeline_deeper_than_one_turn_answers_exactly_once_in_order() {
        let executor = BatchSpy::new();
        let mut connection = RespConnection::new(executor, ConnConfig::default(), 1);
        let depth = ConnConfig::default().max_commands_per_turn * 3;
        let mut batch = Vec::new();
        for index in 0..depth {
            batch.extend_from_slice(&cmd(&[b"GET", format!("k{index}").as_bytes()]));
        }
        connection.push(&batch).expect("batch fits");
        let mut out = Vec::new();
        let mut total = 0u64;
        while connection.has_unconsumed() {
            let outcome = connection.drain(&mut out);
            total += outcome.replies;
            assert!(outcome.replies > 0, "a turn must make progress");
        }
        assert_eq!(total, depth as u64, "exactly one reply per command");
        // One engine batch per turn, and every command answered exactly once.
        let batches = connection.executor.batches();
        assert_eq!(batches.len(), 3, "one batch per turn, not per command");
        assert_eq!(
            batches.iter().map(Vec::len).sum::<usize>(),
            depth,
            "every command must be issued exactly once"
        );
        assert!(!connection.has_unconsumed());
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
        let (replies, _) = drain_frames(&mut connection);
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
        let (replies, _) = drain_frames(&mut connection);
        assert!(replies.is_empty());
        connection.push(&full[full.len() - 2..]).expect("rest fits");
        let (replies, _) = drain_frames(&mut connection);
        assert_eq!(replies, vec![b"+PONG\r\n".to_vec()]);
    }

    #[test]
    fn malformed_input_never_panics_and_bounds_hold() {
        let mut connection = conn();
        // Not an array: clear error, connection stays usable afterwards.
        connection.push(b"+PING\r\n").expect("fits");
        let (replies, quit) = drain_frames(&mut connection);
        assert_eq!(replies.len(), 1);
        assert!(replies[0].starts_with(b"-ERR"));
        assert!(!quit);
        // Oversized arrays and bulks are rejected before allocation games.
        let mut small = RespConnection::new(
            FakeExec::new(kivi_types::WallTimestamp::EPOCH),
            ConnConfig {
                max_array_elements: 2,
                max_bulk_bytes: 4,
                ..ConnConfig::default()
            },
            1,
        );
        small.push(&cmd(&[b"GET", b"a", b"b"])).expect("fits");
        let (replies, _) = drain_frames(&mut small);
        assert!(replies[0].starts_with(b"-ERR"));
        small.push(&cmd(&[b"GET", b"12345"])).expect("fits");
        let (replies, _) = drain_frames(&mut small);
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
            let mut out = Vec::new();
            let _ = fuzz.drain(&mut out);
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
    use super::tests::split_frames;
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

        fn now(&self) -> kivi_types::WallTimestamp {
            kivi_types::WallTimestamp::EPOCH
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
                        expires_at: kivi_types::WallTimestamp::from_micros(0),
                    }
                }
                kivi_state::Operation::LeaseRenew { .. } => {
                    kivi_state::OperationResult::LeaseRenewed {
                        fencing: kivi_state::FencingToken::from_u64(1),
                        expires_at: kivi_types::WallTimestamp::from_micros(0),
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
                let mut out = Vec::new();
                let outcome = connection.drain(&mut out);
                // Every reply is well-formed RESP: splitting must not panic
                // and must consume the output exactly.
                let frames = split_frames(&out);
                prop_assert_eq!(frames.len() as u64, outcome.replies);
                if outcome.close {
                    break;
                }
            }
            // Whatever the input was, the connection either consumed it or
            // refused and closed. It must never be left claiming progress it
            // did not make.
            let mut out = Vec::new();
            let outcome = connection.drain(&mut out);
            prop_assert!(out.is_empty() || outcome.close || connection.has_unconsumed());
        }
    }
}
