//! Redis-to-Kivi and Kivi-to-Redis translation.
//!
//! Every supported command maps onto the same typed [`Operation`] the
//! native frontend executes.
//! Conditional writes stay atomic at the owning tablet via
//! [`SetConditional`](kivi_state::Operation::SetConditional): the adapter
//! never performs `GET`-then-`SET`, and expiry compares (`GT`/`LT` family)
//! are left unsupported rather than faked with a read-then-write loop.

use bytes::Bytes;
use jiff::SignedDuration;
use kivi_state::{ExpiryPolicy, Operation, OperationResult, SetCondition};
use kivi_types::{TabletRoute, WallTimestamp};

use crate::error::{KiviErrorKind, RespError};

/// Which Redis command produced an engine operation: drives exact reply
/// mapping (nil vs empty vs zero, `+OK` vs nil, seconds vs milliseconds).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RedisOp {
    /// `GET`: missing answers nil.
    Get,
    /// `SET`: applied answers `+OK`, refused answers nil.
    Set,
    /// `DEL`: answers `1`/`0`.
    Del,
    /// `EXISTS`: answers `1`/`0`.
    Exists,
    /// `EXPIRE` family: answers `1`/`0`.
    Expire,
    /// `TTL`: seconds remaining, `-1` immortal, `-2` missing.
    Ttl,
    /// `PTTL`: milliseconds remaining, `-1` immortal, `-2` missing.
    PTtl,
    /// `EXPIRETIME`: absolute seconds, `-1` immortal, `-2` missing.
    ExpireTime,
    /// `PEXPIRETIME`: absolute milliseconds, `-1` immortal, `-2` missing.
    PExpireTime,
    /// `PERSIST`: answers `1`/`0`.
    Persist,
    /// `GETRANGE` with a non-negative window: missing answers empty.
    GetRange,
    /// `GETRANGE` with a negative bound: full-value read sliced locally.
    GetRangeSlice {
        /// Inclusive start (negative counts from the end).
        start: i64,
        /// Inclusive end (negative counts from the end).
        end: i64,
    },
    /// `SETRANGE`: answers the post-write length (read-after-write).
    SetRange,
    /// `STRLEN`: missing answers `0`.
    StrLen,
}

/// How one request executes: against the engine or immediately.
#[derive(Debug)]
pub enum Action {
    /// Execute this typed operation against the engine, then map the
    /// result as `redis` dictates.
    Execute {
        /// Typed Kivi operation (shared with the native frontend).
        op: Operation,
        /// Originating Redis command (reply mapping only).
        redis: RedisOp,
    },
    /// Reply immediately without touching the engine.
    Reply(Immediate),
}

/// Replies that need no engine access.
#[derive(Debug)]
pub enum Immediate {
    /// Simple string (`+OK`, `+PONG`).
    Simple(&'static str),
    /// Echo bytes as a bulk string.
    Echo(Bytes),
    /// Nil (null bulk string / null).
    Nil,
    /// Integer reply.
    Integer(i64),
    /// Signal the server to answer `+OK` and close the connection.
    Quit,
}

/// Engine execution failures, as seen by the adapter (mapped by the server
/// from its own error type; this crate never depends on the engine).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecuteError {
    /// Key holds a different logical type; nothing mutated.
    WrongType,
    /// Counter overflow or version exhaustion; nothing mutated.
    Rejected,
    /// Bounded queues full or lane saturated; safe to retry with backoff.
    Overloaded,
    /// Unexpected failure; safe to retry, likely pointless.
    Internal,
}

impl ExecuteError {
    /// Maps onto the Redis error surface.
    #[must_use]
    pub const fn kind(self) -> KiviErrorKind {
        match self {
            Self::WrongType => KiviErrorKind::WrongType,
            Self::Rejected => KiviErrorKind::Rejected,
            Self::Overloaded => KiviErrorKind::Overloaded,
            Self::Internal => KiviErrorKind::Internal,
        }
    }
}

/// Engine access for one RESP connection.
///
/// Deliberately **not** `Send + Sync`. A connection may be served on the same
/// thread that owns the data it addresses, and then the cheapest correct
/// executor is a handle to that owner's state - no lock, no channel, no second
/// wakeup. Demanding `Send + Sync` here would forbid exactly the executor
/// worth having and force a rendezvous per request to satisfy a bound no
/// caller needs.
pub trait Executor {
    /// Executes one typed operation (reads resolve chunked values to bytes
    /// before returning, so `Value` is always inline here).
    ///
    /// # Errors
    ///
    /// Returns [`ExecuteError`] when the key holds the wrong type, a
    /// counter operation rejects, the engine is overloaded, or an
    /// unexpected failure occurs.
    fn execute(&self, op: &Operation) -> Result<OperationResult, ExecuteError>;

    /// Executes a batch of operations, issuing them **together** rather than
    /// one at a time.
    ///
    /// The default loops, which is correct but serial: a pipelined client
    /// would still pay one engine rendezvous per command, so deepening a
    /// pipeline would cost more rather than less. An executor that can hold
    /// several operations in flight should override this, and its answers
    /// must line up with `ops` positionally - RESP replies are positional.
    fn execute_batch(&self, ops: &[Operation]) -> Vec<Result<OperationResult, ExecuteError>> {
        ops.iter().map(|op| self.execute(op)).collect()
    }

    /// Current wall time (relative expiry computation only; never
    /// persisted as a decision - the materialized stamp is).
    fn now(&self) -> WallTimestamp;

    /// Resolves the route for a drain, once, or `None` if this turn has no direct
    /// route.
    ///
    /// # Why this is a separate call
    ///
    /// The route is the same for every command in a pipeline, so asking for it per
    /// command asks the same question two hundred and fifty-six times. On the
    /// worker that question is two `arc-swap` loads and a version compare, and the
    /// profile charged it 95 instructions per command - enough that removing 142
    /// instructions of `TabletId` hashing only bought 47. Precomputation is only
    /// worth anything if it is hoisted; leaving it in the loop is the same work with
    /// a comment.
    ///
    /// The value is [`Copy`] and sixteen bytes, so passing it per command costs two
    /// moves and no allocation, no lifetime, and no borrow of the publication.
    ///
    /// Defaulted to `None` so every executor - including `Box<dyn Executor>` and
    /// the trait's own test doubles - keeps working unchanged. `None` is always
    /// safe: it means "use the general path", and the general path is the authority.
    #[must_use]
    fn direct_route(&self) -> Option<TabletRoute> {
        None
    }

    /// Answers a [`FastKind`](crate::fast::FastKind) for a borrowed key without
    /// suspending, or `None` to use the general path.
    ///
    /// `route` is the [`Self::direct_route`] resolved for this drain. It is passed
    /// in rather than looked up so that the per-command path is an index and
    /// nothing else; an executor that ignored it and re-derived the tablet from
    /// `key` would be doing the work this type exists to have already done.
    ///
    /// Implementing it is worth roughly 2,500 instructions per command. See
    /// [`crate::fast`] for the cost map that says what it removes, and for why a
    /// `GET` needs none of the machinery the general path builds.
    fn fast_read(
        &self,
        route: TabletRoute,
        kind: crate::fast::FastKind,
        key: &[u8],
        now: WallTimestamp,
    ) -> Option<Reply> {
        let _ = (route, kind, key, now);
        None
    }

    /// Executes a batch, allowed to suspend.
    ///
    /// Only an executor whose answer depends on another thread needs this: a
    /// durability proof, or a payload that lives in a chunk lane. Such an
    /// answer cannot be collected by a blocking call from a reactor, because
    /// the call would park the thread that has to produce it - so it is awaited
    /// instead.
    ///
    /// A boxed future rather than `async fn` so this trait stays object-safe
    /// and so this crate stays free of any async runtime: a caller with no
    /// runtime simply never polls it.
    fn execute_batch_async<'a>(&'a self, ops: &'a [Operation]) -> BatchFuture<'a> {
        Box::pin(core::future::ready(self.execute_batch(ops)))
    }
}

/// One turn's answers, in request order.
pub type BatchResults = Vec<Result<OperationResult, ExecuteError>>;

/// A turn's answers, produced later.
///
/// Boxed rather than an `async fn` so [`Executor`] stays object-safe and this
/// crate stays free of any async runtime: a caller with no runtime simply never
/// polls one.
pub type BatchFuture<'a> =
    core::pin::Pin<Box<dyn core::future::Future<Output = BatchResults> + 'a>>;

impl<T: Executor + ?Sized> Executor for Box<T> {
    fn execute(&self, op: &Operation) -> Result<OperationResult, ExecuteError> {
        (**self).execute(op)
    }

    fn execute_batch(&self, ops: &[Operation]) -> Vec<Result<OperationResult, ExecuteError>> {
        (**self).execute_batch(ops)
    }

    fn now(&self) -> WallTimestamp {
        (**self).now()
    }

    fn execute_batch_async<'a>(&'a self, ops: &'a [Operation]) -> BatchFuture<'a> {
        (**self).execute_batch_async(ops)
    }
}

/// Parses an ASCII decimal integer argument (optional leading `+`/`-`,
/// digits only, no whitespace). Rejects floats, empty input, and overflow.
///
/// # Errors
///
/// Returns [`RespError::InvalidArguments`] when the bytes are not a valid
/// decimal integer.
pub fn parse_integer(arg: &[u8]) -> Result<i64, RespError> {
    if arg.is_empty() || arg.len() > 20 {
        return Err(RespError::InvalidArguments {
            reason: "not an integer".to_owned(),
        });
    }
    let text = core::str::from_utf8(arg).map_err(|_| RespError::InvalidArguments {
        reason: "not an integer".to_owned(),
    })?;
    text.parse::<i64>()
        .map_err(|_| RespError::InvalidArguments {
            reason: "not an integer".to_owned(),
        })
}

/// Parses an ASCII decimal unsigned argument (digits only).
///
/// # Errors
///
/// Returns [`RespError::InvalidArguments`] when the bytes are not a valid
/// decimal unsigned integer.
pub fn parse_unsigned(arg: &[u8]) -> Result<u64, RespError> {
    if arg.is_empty() || arg.len() > 20 {
        return Err(RespError::InvalidArguments {
            reason: "not an integer".to_owned(),
        });
    }
    let text = core::str::from_utf8(arg).map_err(|_| RespError::InvalidArguments {
        reason: "not an integer".to_owned(),
    })?;
    if text.starts_with('+') || text.starts_with('-') {
        return Err(RespError::InvalidArguments {
            reason: "not an integer".to_owned(),
        });
    }
    text.parse::<u64>()
        .map_err(|_| RespError::InvalidArguments {
            reason: "not an integer".to_owned(),
        })
}

/// Case-insensitive equality against an upper-case literal, for the option
/// tokens a `SET` or `HELLO` carries. Allocation-free, so the option loops do
/// not build a `Vec` per token.
#[must_use]
pub fn token_is(token: &[u8], upper: &[u8]) -> bool {
    token.len() == upper.len()
        && token
            .iter()
            .zip(upper)
            .all(|(actual, expected)| actual.to_ascii_uppercase() == *expected)
}

/// Translates one command (name plus borrowed byte args) into an [`Action`].
/// `now` anchors relative expiries (`EX`/`PX`) to absolute stamps.
///
/// # Errors
///
/// Returns [`RespError::InvalidArguments`] on bad arity or values,
/// [`RespError::Unsupported`] for commands or options outside this profile,
/// and [`RespError::Malformed`] for a name that is not a short ASCII token.
#[allow(clippy::too_many_lines)]
pub fn translate(name: &[u8], args: &[&[u8]], now: WallTimestamp) -> Result<Action, RespError> {
    let folded = crate::frame::fold_command_name(name).ok_or(RespError::Malformed)?;
    match &folded[..name.len()] {
        b"PING" => match args {
            [] => Ok(Action::Reply(Immediate::Simple("PONG"))),
            [message] => Ok(Action::Reply(Immediate::Echo(Bytes::copy_from_slice(
                message,
            )))),
            _ => Err(RespError::InvalidArguments {
                reason: "wrong number of arguments for 'ping'".to_owned(),
            }),
        },
        b"ECHO" => match args {
            [message] => Ok(Action::Reply(Immediate::Echo(Bytes::copy_from_slice(
                message,
            )))),
            _ => Err(RespError::InvalidArguments {
                reason: "wrong number of arguments for 'echo'".to_owned(),
            }),
        },
        b"QUIT" => {
            if !args.is_empty() {
                return Err(RespError::InvalidArguments {
                    reason: "wrong number of arguments for 'quit'".to_owned(),
                });
            }
            Ok(Action::Reply(Immediate::Quit))
        }
        b"GET" => match args {
            [key] => Ok(Action::Execute {
                op: Operation::Get {
                    key: kivi_state::Key::from(*key),
                },
                redis: RedisOp::Get,
            }),
            _ => Err(RespError::InvalidArguments {
                reason: "wrong number of arguments for 'get'".to_owned(),
            }),
        },
        b"SET" => translate_set(args, now),
        b"DEL" => match args {
            [key] => Ok(Action::Execute {
                op: Operation::Delete {
                    key: kivi_state::Key::from(*key),
                },
                redis: RedisOp::Del,
            }),
            _ => Err(RespError::InvalidArguments {
                reason: "wrong number of arguments for 'del'".to_owned(),
            }),
        },
        b"EXISTS" => match args {
            [key] => Ok(Action::Execute {
                op: Operation::Exists {
                    key: kivi_state::Key::from(*key),
                },
                redis: RedisOp::Exists,
            }),
            _ => Err(RespError::InvalidArguments {
                reason: "wrong number of arguments for 'exists'".to_owned(),
            }),
        },
        b"EXPIRE" => translate_expire(args, now, 1_000_000, "expire", RedisOp::Expire),
        b"PEXPIRE" => translate_expire(args, now, 1_000, "pexpire", RedisOp::Expire),
        b"EXPIREAT" => translate_expire_at(args, 1_000_000, "expireat"),
        b"PEXPIREAT" => translate_expire_at(args, 1_000, "pexpireat"),
        b"TTL" => match args {
            [key] => Ok(Action::Execute {
                op: Operation::GetExpiry {
                    key: kivi_state::Key::from(*key),
                },
                redis: RedisOp::Ttl,
            }),
            _ => Err(RespError::InvalidArguments {
                reason: "wrong number of arguments for 'ttl'".to_owned(),
            }),
        },
        b"PTTL" => match args {
            [key] => Ok(Action::Execute {
                op: Operation::GetExpiry {
                    key: kivi_state::Key::from(*key),
                },
                redis: RedisOp::PTtl,
            }),
            _ => Err(RespError::InvalidArguments {
                reason: "wrong number of arguments for 'pttl'".to_owned(),
            }),
        },
        b"EXPIRETIME" => match args {
            [key] => Ok(Action::Execute {
                op: Operation::GetExpiry {
                    key: kivi_state::Key::from(*key),
                },
                redis: RedisOp::ExpireTime,
            }),
            _ => Err(RespError::InvalidArguments {
                reason: "wrong number of arguments for 'expiretime'".to_owned(),
            }),
        },
        b"PEXPIRETIME" => match args {
            [key] => Ok(Action::Execute {
                op: Operation::GetExpiry {
                    key: kivi_state::Key::from(*key),
                },
                redis: RedisOp::PExpireTime,
            }),
            _ => Err(RespError::InvalidArguments {
                reason: "wrong number of arguments for 'pexpiretime'".to_owned(),
            }),
        },
        b"PERSIST" => match args {
            [key] => Ok(Action::Execute {
                op: Operation::PersistExpiry {
                    key: kivi_state::Key::from(*key),
                },
                redis: RedisOp::Persist,
            }),
            _ => Err(RespError::InvalidArguments {
                reason: "wrong number of arguments for 'persist'".to_owned(),
            }),
        },
        b"GETRANGE" => translate_getrange(args),
        b"SETRANGE" => match args {
            [key, offset, value] => {
                let offset = parse_unsigned(offset).map_err(|_| RespError::InvalidArguments {
                    reason: "offset is not an integer".to_owned(),
                })?;
                Ok(Action::Execute {
                    op: Operation::SetRange {
                        key: kivi_state::Key::from(*key),
                        offset,
                        patch: Bytes::copy_from_slice(value),
                    },
                    redis: RedisOp::SetRange,
                })
            }
            _ => Err(RespError::InvalidArguments {
                reason: "wrong number of arguments for 'setrange'".to_owned(),
            }),
        },
        b"STRLEN" => match args {
            [key] => Ok(Action::Execute {
                op: Operation::BytesLength {
                    key: kivi_state::Key::from(*key),
                },
                redis: RedisOp::StrLen,
            }),
            _ => Err(RespError::InvalidArguments {
                reason: "wrong number of arguments for 'strlen'".to_owned(),
            }),
        },
        _ => Err(RespError::Unsupported {
            command: String::from_utf8_lossy(&folded[..name.len()]).into_owned(),
        }),
    }
}

/// Translates `SET key value [options]` onto [`Operation::SetConditional`].
///
/// Relative expiries materialize `now + delta` with Jiff arithmetic; an
/// unrepresentable sum is an invalid expiry (loud error), never a silent
/// clamp to maximum time.
fn translate_set(args: &[&[u8]], now: WallTimestamp) -> Result<Action, RespError> {
    let [key, value, options @ ..] = args else {
        return Err(RespError::InvalidArguments {
            reason: "wrong number of arguments for 'set'".to_owned(),
        });
    };
    let mut condition = SetCondition::Always;
    let mut condition_set = false;
    let mut expiry = ExpiryPolicy::Clear;
    let mut expiry_set = false;
    let mut index = 0;
    while index < options.len() {
        let token = options[index];
        if token_is(token, b"NX") {
            if condition_set {
                return err_set_syntax();
            }
            condition = SetCondition::IfAbsent;
            condition_set = true;
            index += 1;
        } else if token_is(token, b"XX") {
            if condition_set {
                return err_set_syntax();
            }
            condition = SetCondition::IfPresent;
            condition_set = true;
            index += 1;
        } else if token_is(token, b"GET")
            || token_is(token, b"IFEQ")
            || token_is(token, b"IFNE")
            || token_is(token, b"IFDEQ")
            || token_is(token, b"IFDNE")
        {
            return Err(RespError::Unsupported {
                command: "SET".to_owned(),
            });
        } else if token_is(token, b"KEEPTTL") {
            if expiry_set {
                return err_set_syntax();
            }
            expiry = ExpiryPolicy::Keep;
            expiry_set = true;
            index += 1;
        } else if token_is(token, b"EX")
            || token_is(token, b"PX")
            || token_is(token, b"EXAT")
            || token_is(token, b"PXAT")
        {
            if expiry_set {
                return err_set_syntax();
            }
            let arg = options.get(index + 1).ok_or(RespError::InvalidArguments {
                reason: "wrong number of arguments for 'set'".to_owned(),
            })?;
            expiry = ExpiryPolicy::ExpireAt(set_expiry_stamp(token, arg, now)?);
            expiry_set = true;
            index += 2;
        } else {
            return err_set_syntax();
        }
    }
    Ok(Action::Execute {
        op: Operation::SetConditional {
            key: kivi_state::Key::from(*key),
            value: Bytes::copy_from_slice(value),
            condition,
            expiry,
        },
        redis: RedisOp::Set,
    })
}

/// Materializes a relative expiry (`now + delta micros`) with Jiff signed
/// arithmetic. Overflow is an invalid expiry, never a silent clamp.
fn relative_stamp(now: WallTimestamp, delta_micros: i64) -> Result<WallTimestamp, RespError> {
    now.checked_add(SignedDuration::from_micros(delta_micros))
        .map_err(|_| invalid_expire())
}

/// Computes one `SET` expiry stamp: `EX`/`PX` anchor `now + delta` with
/// Jiff arithmetic, `EXAT`/`PXAT` are absolute. Zero, overflow, and
/// unrepresentable sums are all invalid expiries, never silent clamps.
fn set_expiry_stamp(
    token: &[u8],
    arg: &[u8],
    now: WallTimestamp,
) -> Result<WallTimestamp, RespError> {
    let amount = parse_unsigned(arg).map_err(|_| invalid_expire())?;
    if amount == 0 {
        return Err(invalid_expire());
    }
    let unit_micros: i64 = match token {
        b"EX" | b"EXAT" => 1_000_000,
        _ => 1_000,
    };
    let micros = i64::try_from(amount)
        .ok()
        .and_then(|amount| amount.checked_mul(unit_micros))
        .ok_or_else(invalid_expire)?;
    match token {
        b"EX" | b"PX" => relative_stamp(now, micros),
        _ => Ok(WallTimestamp::from_micros(micros)),
    }
}

fn err_set_syntax() -> Result<Action, RespError> {
    Err(RespError::InvalidArguments {
        reason: "syntax error in SET options".to_owned(),
    })
}

fn invalid_expire() -> RespError {
    RespError::InvalidArguments {
        reason: "invalid expire time".to_owned(),
    }
}

/// Translates relative expiries (`EXPIRE`/`PEXPIRE`). Only the bare
/// three-argument form is exact in this profile; the `NX`/`XX`/`GT`/`LT`
/// family would need an atomic compare-and-set-expiry the engine does not
/// yet own, so those forms error clearly instead of racing a
/// read-then-write loop.
fn translate_expire(
    args: &[&[u8]],
    now: WallTimestamp,
    unit_micros: i64,
    name: &'static str,
    redis: RedisOp,
) -> Result<Action, RespError> {
    match args {
        [key, amount] => {
            let units = parse_unsigned(amount).map_err(|_| RespError::InvalidArguments {
                reason: "value is not an integer or out of range".to_owned(),
            })?;
            let delta = i64::try_from(units)
                .ok()
                .and_then(|units| units.checked_mul(unit_micros))
                .ok_or_else(|| RespError::InvalidArguments {
                    reason: "value is not an integer or out of range".to_owned(),
                })?;
            let stamp = relative_stamp(now, delta).map_err(|_| RespError::InvalidArguments {
                reason: "value is not an integer or out of range".to_owned(),
            })?;
            Ok(Action::Execute {
                op: Operation::ExpireAt {
                    key: kivi_state::Key::from(*key),
                    expires_at: stamp,
                },
                redis,
            })
        }
        [_, _, _, ..] => Err(RespError::Unsupported {
            command: name.to_uppercase(),
        }),
        _ => Err(RespError::InvalidArguments {
            reason: format!("wrong number of arguments for '{name}'"),
        }),
    }
}

/// Translates absolute expiries (`EXPIREAT`/`PEXPIREAT`); same option policy
/// as [`translate_expire`].
fn translate_expire_at(
    args: &[&[u8]],
    unit_micros: i64,
    name: &'static str,
) -> Result<Action, RespError> {
    match args {
        [key, amount] => {
            let units = parse_unsigned(amount).map_err(|_| RespError::InvalidArguments {
                reason: "value is not an integer or out of range".to_owned(),
            })?;
            let micros = i64::try_from(units)
                .ok()
                .and_then(|units| units.checked_mul(unit_micros))
                .ok_or_else(|| RespError::InvalidArguments {
                    reason: "value is not an integer or out of range".to_owned(),
                })?;
            Ok(Action::Execute {
                op: Operation::ExpireAt {
                    key: kivi_state::Key::from(*key),
                    expires_at: WallTimestamp::from_micros(micros),
                },
                redis: RedisOp::Expire,
            })
        }
        [_, _, _, ..] => Err(RespError::Unsupported {
            command: name.to_uppercase(),
        }),
        _ => Err(RespError::InvalidArguments {
            reason: format!("wrong number of arguments for '{name}'"),
        }),
    }
}

/// Translates `GETRANGE key start end` (inclusive, negatives from the end).
/// Non-negative windows compile to a single native `GetRange` (atomic and
/// lane-friendly); windows with a negative bound compile to a single `Get`
/// whose snapshot the connection slices locally (still one atomic read,
/// never a length-then-slice race).
fn translate_getrange(args: &[&[u8]]) -> Result<Action, RespError> {
    match args {
        [key, start, end] => {
            let start = parse_integer(start).map_err(|_| RespError::InvalidArguments {
                reason: "value is not an integer or out of range".to_owned(),
            })?;
            let end = parse_integer(end).map_err(|_| RespError::InvalidArguments {
                reason: "value is not an integer or out of range".to_owned(),
            })?;
            let key = kivi_state::Key::from(*key);
            if start >= 0 && end >= 0 {
                if end < start {
                    // Empty window: still a read (for WRONGTYPE parity),
                    // answered as an empty string below.
                    return Ok(Action::Execute {
                        op: Operation::GetRange {
                            key,
                            offset: 0,
                            len: 0,
                        },
                        redis: RedisOp::GetRange,
                    });
                }
                let offset = u64::try_from(start).unwrap_or(0);
                let len = u64::try_from(end)
                    .unwrap_or(0)
                    .saturating_sub(offset)
                    .saturating_add(1);
                Ok(Action::Execute {
                    op: Operation::GetRange { key, offset, len },
                    redis: RedisOp::GetRange,
                })
            } else {
                Ok(Action::Execute {
                    op: Operation::Get { key },
                    redis: RedisOp::GetRangeSlice { start, end },
                })
            }
        }
        _ => Err(RespError::InvalidArguments {
            reason: "wrong number of arguments for 'getrange'".to_owned(),
        }),
    }
}

/// Version-agnostic reply: the connection encodes it as RESP2 or RESP3.
///
/// `Bulk` carries the engine's own handle rather than a copy of it. A `GET`
/// result is already a `Bytes` the tablet store owns, so copying it here
/// would duplicate every value the database returns.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Reply {
    /// Simple string (`+OK`, `+PONG`).
    Simple(&'static str),
    /// Bulk string (binary-safe).
    Bulk(Bytes),
    /// Nil (null bulk string in RESP2, null in RESP3).
    Nil,
    /// Integer reply.
    Int(i64),
    /// Error reply (without the leading `-`; framing adds it).
    Error(String),
}

/// Maps an engine result onto its exact Redis reply for `redis`.
#[must_use]
#[allow(clippy::too_many_lines)]
pub fn map_result(result: &OperationResult, redis: RedisOp, now: WallTimestamp) -> Reply {
    match (result, redis) {
        (OperationResult::Value(Some(value)), RedisOp::Get | RedisOp::GetRange) => {
            Reply::Bulk(value.clone())
        }
        (OperationResult::Value(None), RedisOp::Get) => Reply::Nil,
        (OperationResult::Value(None), RedisOp::GetRange | RedisOp::GetRangeSlice { .. }) => {
            Reply::Bulk(Bytes::new())
        }
        (OperationResult::Value(Some(value)), RedisOp::GetRangeSlice { start, end }) => {
            Reply::Bulk(slice_getrange(value, start, end))
        }
        // Chunked reads resolve before mapping, and `SETRANGE` never
        // surfaces `Stored` here (the connection measures after writing):
        // reaching either is a missed resolution path, never wrong bytes.
        (OperationResult::ChunkedValue { .. }, _)
        | (OperationResult::Stored { .. }, RedisOp::SetRange) => {
            Reply::Error(KiviErrorKind::Internal.message().to_owned())
        }
        (OperationResult::ConditionalSet { applied, .. }, RedisOp::Set) => {
            if *applied {
                Reply::Simple("OK")
            } else {
                Reply::Nil
            }
        }
        (OperationResult::Deleted { existed }, RedisOp::Del) => Reply::Int(i64::from(*existed)),
        (OperationResult::Exists(present), RedisOp::Exists) => Reply::Int(i64::from(*present)),
        (OperationResult::ExpirySet { applied }, RedisOp::Expire) => {
            Reply::Int(i64::from(*applied))
        }
        (OperationResult::ExpiryPersisted { removed }, RedisOp::Persist) => {
            Reply::Int(i64::from(*removed))
        }
        (OperationResult::Expiry(expiry), RedisOp::Ttl) => Reply::Int(ttl_seconds(*expiry, now)),
        (OperationResult::Expiry(expiry), RedisOp::PTtl) => Reply::Int(ttl_millis(*expiry, now)),
        (OperationResult::Expiry(expiry), RedisOp::ExpireTime) => {
            Reply::Int(expire_seconds(*expiry))
        }
        (OperationResult::Expiry(expiry), RedisOp::PExpireTime) => {
            Reply::Int(expire_millis(*expiry))
        }
        (OperationResult::Length(value), RedisOp::StrLen) => {
            Reply::Int(value.map_or(0, |len| i64::try_from(len).unwrap_or(i64::MAX)))
        }
        // Unreachable pairings: loud errors, never wrong bytes.
        _ => Reply::Error(KiviErrorKind::Internal.message().to_owned()),
    }
}

/// Maps an engine failure onto its Redis error.
#[must_use]
pub fn map_execute_error(error: ExecuteError) -> Reply {
    Reply::Error(error.kind().message().to_owned())
}

/// Maps a parse/validation failure onto its Redis error.
#[must_use]
pub fn map_parse_error(error: &RespError) -> Reply {
    match error {
        RespError::Incomplete => Reply::Error("ERR incomplete".to_owned()),
        RespError::Malformed => Reply::Error("ERR malformed request".to_owned()),
        RespError::TooLarge => Reply::Error("ERR request exceeds bound".to_owned()),
        RespError::PipelineFull => Reply::Error(KiviErrorKind::Overloaded.message().to_owned()),
        RespError::Unsupported { command } => {
            if command.eq_ignore_ascii_case("SET") {
                Reply::Error("ERR unsupported SET option in Kivi RESP profile v1".to_owned())
            } else if crate::command::lookup(command).is_some() {
                Reply::Error(format!(
                    "ERR unsupported command '{command}' in Kivi RESP profile v1"
                ))
            } else {
                Reply::Error(format!("ERR unknown command '{command}'"))
            }
        }
        RespError::InvalidArguments { reason } => Reply::Error(format!("ERR {reason}")),
    }
}

/// Remaining TTL in seconds: `-2` missing, `-1` immortal, else the
/// ceiling of remaining milliseconds divided by 1000 (Redis rounds up
/// partial seconds; a 1 ms remainder reports `1`, never `0`).
/// Whole seconds of TTL remaining: `-2` missing, `-1` immortal.
///
/// Public because the direct read path in [`crate::fast`] must produce the same
/// number, and the alternative - a second implementation of Redis's TTL rounding -
/// is a second thing to get wrong. One authority, two callers.
#[must_use]
pub fn ttl_seconds(expiry: Option<kivi_types::Expiry>, now: WallTimestamp) -> i64 {
    match expiry {
        None => -2,
        Some(value) => match value.as_stamp() {
            None => -1,
            Some(stamp) => {
                let remaining = stamp.as_micros().saturating_sub(now.as_micros());
                if remaining <= 0 {
                    // Expired but unreclaimed races as missing downstream;
                    // reaching here means the read saw it live, so one
                    // second of grace keeps the contract total.
                    1
                } else {
                    remaining.saturating_add(999_999) / 1_000_000
                }
            }
        },
    }
}

/// Remaining TTL in milliseconds: `-2` missing, `-1` immortal.
/// Milliseconds of TTL remaining: `-2` missing, `-1` immortal.
///
/// Public for the same reason as [`ttl_seconds`].
#[must_use]
pub fn ttl_millis(expiry: Option<kivi_types::Expiry>, now: WallTimestamp) -> i64 {
    match expiry {
        None => -2,
        Some(value) => match value.as_stamp() {
            None => -1,
            Some(stamp) => (stamp.as_micros().saturating_sub(now.as_micros()) / 1_000).max(0),
        },
    }
}

/// Absolute expiry in seconds: `-2` missing, `-1` immortal.
fn expire_seconds(expiry: Option<kivi_types::Expiry>) -> i64 {
    match expiry {
        None => -2,
        Some(value) => match value.as_stamp() {
            None => -1,
            Some(stamp) => stamp.as_micros().div_euclid(1_000_000),
        },
    }
}

/// Absolute expiry in milliseconds: `-2` missing, `-1` immortal.
fn expire_millis(expiry: Option<kivi_types::Expiry>) -> i64 {
    match expiry {
        None => -2,
        Some(value) => match value.as_stamp() {
            None => -1,
            Some(stamp) => stamp.as_micros().div_euclid(1_000),
        },
    }
}

/// Slices a full value snapshot to a Redis `GETRANGE` window (inclusive,
/// negatives from the end, clamped; empty when the window misses).
#[must_use]
pub fn slice_getrange(value: &[u8], start: i64, end: i64) -> Bytes {
    let total = i64::try_from(value.len()).unwrap_or(i64::MAX);
    let norm = |index: i64| {
        if index < 0 {
            total.saturating_add(index).max(0)
        } else {
            index.min(total)
        }
    };
    let from = norm(start);
    let to = norm(end);
    if from > to || from >= total {
        return Bytes::new();
    }
    let to = to.min(total - 1);
    let from_usize = usize::try_from(from).unwrap_or(0);
    let to_usize = usize::try_from(to).unwrap_or(0);
    if from_usize > to_usize || to_usize >= value.len() {
        return Bytes::new();
    }
    Bytes::copy_from_slice(&value[from_usize..=to_usize])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn integers_reject_floats_empty_and_overflow() {
        assert_eq!(parse_integer(b"41"), Ok(41));
        assert_eq!(parse_integer(b"-41"), Ok(-41));
        assert_eq!(parse_integer(b"+7"), Ok(7));
        assert!(parse_integer(b"").is_err());
        assert!(parse_integer(b"4.5").is_err());
        assert!(parse_integer(b" 4").is_err());
        assert!(parse_integer(b"99999999999999999999").is_err());
        assert!(parse_integer(b"\xff").is_err());
        assert_eq!(parse_unsigned(b"41"), Ok(41));
        assert!(parse_unsigned(b"-1").is_err());
        assert!(parse_unsigned(b"+1").is_err());
        assert!(parse_unsigned(b"").is_err());
    }

    #[test]
    fn set_options_map_onto_kivi_conditions_atomically() {
        let now = WallTimestamp::from_micros(1_000_000_000);
        // Bare SET clears expiry.
        let Action::Execute { op, redis } =
            translate(b"SET", &[b"k", b"v"], now).expect("bare set")
        else {
            panic!("bare SET must execute");
        };
        assert_eq!(redis, RedisOp::Set);
        let Operation::SetConditional {
            condition, expiry, ..
        } = op
        else {
            panic!("SET must be conditional");
        };
        assert_eq!(condition, SetCondition::Always);
        assert_eq!(expiry, ExpiryPolicy::Clear);
        // NX / XX.
        let Action::Execute { op, .. } = translate(b"SET", &[b"k", b"v", b"NX"], now).expect("nx")
        else {
            panic!("NX must execute");
        };
        assert!(matches!(
            op,
            Operation::SetConditional {
                condition: SetCondition::IfAbsent,
                ..
            }
        ));
        let Action::Execute { op, .. } =
            translate(b"SET", &[b"k", b"v", b"xx"], now).expect("xx lowercase")
        else {
            panic!("XX must execute");
        };
        assert!(matches!(
            op,
            Operation::SetConditional {
                condition: SetCondition::IfPresent,
                ..
            }
        ));
        // Conflicts and bad values error, never approximate.
        assert!(translate(b"SET", &[b"k", b"v", b"NX", b"XX"], now).is_err());
        assert!(translate(b"SET", &[b"k", b"v", b"EX", b"1", b"PX", b"1"], now).is_err());
        assert!(translate(b"SET", &[b"k", b"v", b"EX", b"0"], now).is_err());
        assert!(translate(b"SET", &[b"k", b"v", b"EX", b"nope"], now).is_err());
        // GET and the value-compare family stay unsupported, never faked.
        assert!(matches!(
            translate(b"SET", &[b"k", b"v", b"GET"], now),
            Err(RespError::Unsupported { .. })
        ));
        assert!(matches!(
            translate(b"SET", &[b"k", b"v", b"IFEQ", b"x"], now),
            Err(RespError::Unsupported { .. })
        ));
        // EX anchors relative to now; EXAT is absolute; KEEPTTL keeps.
        let Action::Execute { op, .. } =
            translate(b"SET", &[b"k", b"v", b"EX", b"10"], now).expect("ex")
        else {
            panic!("EX must execute");
        };
        let Operation::SetConditional { expiry, .. } = op else {
            panic!("EX must be conditional");
        };
        assert_eq!(
            expiry,
            ExpiryPolicy::ExpireAt(WallTimestamp::from_micros(now.as_micros() + 10_000_000))
        );
        let Action::Execute { op, .. } =
            translate(b"SET", &[b"k", b"v", b"PXAT", b"5000"], now).expect("pxat")
        else {
            panic!("PXAT must execute");
        };
        let Operation::SetConditional { expiry, .. } = op else {
            panic!("PXAT must be conditional");
        };
        // PXAT is unix *milliseconds*: 5000 ms anchors 5_000_000 µs, never
        // the raw millisecond count (which would land in 1970 and expire
        // immediately - caught by differential testing against Redis).
        assert_eq!(
            expiry,
            ExpiryPolicy::ExpireAt(WallTimestamp::from_micros(5_000_000))
        );
        let Action::Execute { op, .. } =
            translate(b"SET", &[b"k", b"v", b"KEEPTTL"], now).expect("keepttl")
        else {
            panic!("KEEPTTL must execute");
        };
        assert!(matches!(
            op,
            Operation::SetConditional {
                expiry: ExpiryPolicy::Keep,
                ..
            }
        ));
    }

    #[test]
    fn getrange_windows_compile_to_one_atomic_read() {
        // Non-negative windows use the native range op directly.
        let Action::Execute { op, redis } =
            translate(b"GETRANGE", &[b"k", b"0", b"3"], WallTimestamp::EPOCH).expect("getrange")
        else {
            panic!("GETRANGE must execute");
        };
        assert_eq!(redis, RedisOp::GetRange);
        assert!(matches!(
            op,
            Operation::GetRange {
                offset: 0,
                len: 4,
                ..
            }
        ));
        // Negative bounds read the full snapshot once and slice locally.
        let Action::Execute { op, redis } =
            translate(b"GETRANGE", &[b"k", b"-3", b"-1"], WallTimestamp::EPOCH)
                .expect("negative getrange")
        else {
            panic!("negative GETRANGE must execute");
        };
        assert!(matches!(
            redis,
            RedisOp::GetRangeSlice { start: -3, end: -1 }
        ));
        assert!(matches!(op, Operation::Get { .. }));
        // Docs examples slice exactly.
        assert_eq!(slice_getrange(b"This is a string", 0, 3).as_ref(), b"This");
        assert_eq!(slice_getrange(b"This is a string", -3, -1).as_ref(), b"ing");
        assert_eq!(
            slice_getrange(b"This is a string", 0, -1).as_ref(),
            b"This is a string"
        );
        assert_eq!(
            slice_getrange(b"This is a string", 10, 100).as_ref(),
            b"string"
        );
        assert!(slice_getrange(b"abc", 5, 9).is_empty());
        assert!(slice_getrange(b"abc", 2, 1).is_empty());
    }

    #[test]
    fn result_mapping_keeps_nil_empty_and_zero_distinct() {
        use kivi_state::{ObjectVersion, OperationResult};
        // GET missing is nil; GETRANGE missing is empty; STRLEN missing is 0.
        assert_eq!(
            map_result(
                &OperationResult::Value(None),
                RedisOp::Get,
                WallTimestamp::EPOCH
            ),
            Reply::Nil
        );
        assert_eq!(
            map_result(
                &OperationResult::Value(None),
                RedisOp::GetRange,
                WallTimestamp::EPOCH
            ),
            Reply::Bulk(Bytes::new())
        );
        assert_eq!(
            map_result(
                &OperationResult::Length(None),
                RedisOp::StrLen,
                WallTimestamp::EPOCH
            ),
            Reply::Int(0)
        );
        // SET applied is +OK; refused is nil.
        assert_eq!(
            map_result(
                &OperationResult::ConditionalSet {
                    applied: true,
                    version: Some(ObjectVersion::FIRST),
                },
                RedisOp::Set,
                WallTimestamp::EPOCH
            ),
            Reply::Simple("OK")
        );
        assert_eq!(
            map_result(
                &OperationResult::ConditionalSet {
                    applied: false,
                    version: None,
                },
                RedisOp::Set,
                WallTimestamp::EPOCH
            ),
            Reply::Nil
        );
        // TTL family: -2 missing, -1 immortal, remaining otherwise.
        let at_sec = WallTimestamp::from_micros(1_000_000);
        assert_eq!(
            map_result(&OperationResult::Expiry(None), RedisOp::Ttl, at_sec),
            Reply::Int(-2)
        );
        assert_eq!(
            map_result(
                &OperationResult::Expiry(Some(kivi_types::Expiry::NEVER)),
                RedisOp::Ttl,
                at_sec
            ),
            Reply::Int(-1)
        );
        assert_eq!(
            map_result(
                &OperationResult::Expiry(Some(kivi_types::Expiry::at(WallTimestamp::from_micros(
                    3_500_000
                )))),
                RedisOp::Ttl,
                at_sec
            ),
            Reply::Int(3)
        );
        assert_eq!(
            map_result(
                &OperationResult::Expiry(Some(kivi_types::Expiry::at(WallTimestamp::from_micros(
                    3_500_000
                )))),
                RedisOp::PTtl,
                at_sec
            ),
            Reply::Int(2_500)
        );
        assert_eq!(
            map_result(
                &OperationResult::Expiry(Some(kivi_types::Expiry::at(WallTimestamp::from_micros(
                    3_500_000
                )))),
                RedisOp::ExpireTime,
                WallTimestamp::EPOCH
            ),
            Reply::Int(3)
        );
        assert_eq!(
            map_result(
                &OperationResult::Expiry(Some(kivi_types::Expiry::at(WallTimestamp::from_micros(
                    3_500_000
                )))),
                RedisOp::PExpireTime,
                WallTimestamp::EPOCH
            ),
            Reply::Int(3_500)
        );
    }

    #[test]
    fn relative_expiry_materializes_once_and_rejects_overflow() {
        let now = WallTimestamp::from_micros(1_000_000_000);
        // EXPIRE anchors `now + delta` exactly once at the edge.
        let Action::Execute { op, .. } = translate(b"EXPIRE", &[b"k", b"10"], now).expect("expire")
        else {
            panic!("EXPIRE must execute");
        };
        assert!(matches!(
            op,
            Operation::ExpireAt {
                expires_at,
                ..
            } if expires_at == WallTimestamp::from_micros(1_010_000_000)
        ));
        // Overflow is an error, never a silent clamp to maximum time.
        let huge = u64::MAX.to_string().into_bytes();
        assert!(
            translate(
                b"SET",
                &[
                    b"k".as_slice(),
                    b"v".as_slice(),
                    b"EX".as_slice(),
                    huge.as_slice()
                ],
                now
            )
            .is_err()
        );
        assert!(translate(b"EXPIRE", &[b"k".as_slice(), huge.as_slice()], now).is_err());
        // Absolute expiries accept the full signed range.
        let Action::Execute { op, .. } =
            translate(b"PEXPIREAT", &[b"k", b"5000"], WallTimestamp::EPOCH).expect("pexpireat")
        else {
            panic!("PEXPIREAT must execute");
        };
        assert!(matches!(
            op,
            Operation::ExpireAt { expires_at, .. }
            if expires_at == WallTimestamp::from_micros(5_000_000)
        ));
    }

    #[test]
    fn ttl_at_exact_boundary_reports_grace() {
        // Exactly at the deadline the read saw the key live: one second of
        // grace keeps the contract total; PTTL floors at zero.
        let at = WallTimestamp::from_micros(2_000_000);
        let expiry = Some(kivi_types::Expiry::at(at));
        assert_eq!(
            map_result(&OperationResult::Expiry(expiry), RedisOp::Ttl, at),
            Reply::Int(1)
        );
        assert_eq!(
            map_result(&OperationResult::Expiry(expiry), RedisOp::PTtl, at),
            Reply::Int(0)
        );
    }

    #[test]
    fn binary_keys_and_values_survive_verbatim() {
        let key = vec![0x00, 0xFF, 0x10, 0x00];
        let value = vec![0x00, 0x00, 0xFF, 0xFE, 0x61];
        let Action::Execute { op, .. } = translate(
            b"SET",
            &[key.as_slice(), value.as_slice()],
            WallTimestamp::EPOCH,
        )
        .expect("binary set") else {
            panic!("binary SET must execute");
        };
        let Operation::SetConditional {
            key: got,
            value: got_value,
            ..
        } = op
        else {
            panic!("binary SET must be conditional");
        };
        assert_eq!(got.as_bytes(), key.as_slice());
        assert_eq!(&got_value[..], value.as_slice());
    }
}
