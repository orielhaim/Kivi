//! Zero-copy RESP request parsing.
//!
//! A command frame is a slice of the connection's input buffer, not a
//! freshly built tree. Arguments are borrowed in place, so a `GET` costs one
//! allocation in total — the `Key` the engine has to own — instead of one per
//! argument, a copy of every argument while splitting the command name, and
//! another copy while building the operation.
//!
//! Every claimed length is validated against the connection's bounds before
//! it is used to slice, so a client cannot make the parser act on a length it
//! never sent. The parser allocates nothing for itself: [`MAX_ARGS`]
//! arguments fit inline, and a frame carrying more is refused rather than
//! truncated.
//!
//! # Incomplete is not malformed
//!
//! TCP splits frames anywhere. A scan that runs out of bytes mid-frame is
//! [`Parsed::Incomplete`] and the caller waits; only a frame that is
//! syntactically impossible ([`Parsed::Malformed`]) closes the connection.
//! Conflating the two would reject every request larger than one segment.

/// Maximum arguments in one command frame, name included.
///
/// The longest command this profile supports is `SET key value [EX seconds]`
/// at six elements, so ten is generous. The number also bounds the parser's
/// inline argument array, and therefore the size of the value it returns: a
/// larger array would make the result big enough to force a heap box, which
/// is the one allocation this parser exists to avoid. A frame carrying more
/// is refused, never truncated.
pub const MAX_ARGS: usize = 10;

/// Longest legal length line (`$<20 digits>\r\n`). A length header that has
/// not terminated within this many bytes cannot become legal, so refusing it
/// bounds the parser's per-byte work on hostile input instead of scanning a
/// line an attacker is still writing.
const MAX_LEN_LINE: usize = 24;

/// Longest command name the dispatcher recognises. The registry's longest
/// entry is `PEXPIRETIME` (11 bytes); 24 leaves room for the introspection
/// commands without letting a client make the fold buffer grow.
pub const MAX_COMMAND_NAME: usize = 24;

/// Folds an ASCII command name to upper case in a fixed stack buffer.
///
/// Redis accepts any case, so the name has to be folded somewhere. Doing it
/// here — once per request, into an array on the stack — is what lets the
/// dispatcher be a single `match` on the result. The previous version
/// allocated a `Vec` per request purely to make that `match` work, and then
/// matched the same name a second and third time in two other places.
#[must_use]
pub fn fold_command_name(name: &[u8]) -> Option<[u8; MAX_COMMAND_NAME]> {
    if name.is_empty() || name.len() > MAX_COMMAND_NAME || !name.is_ascii() {
        return None;
    }
    let mut folded = [0u8; MAX_COMMAND_NAME];
    for (slot, byte) in folded.iter_mut().zip(name) {
        *slot = byte.to_ascii_uppercase();
    }
    Some(folded)
}

/// Bounds the parser enforces while scanning.
#[derive(Debug, Clone, Copy)]
pub struct Limits {
    /// Maximum bytes in one bulk string (keys and values).
    pub max_bulk_bytes: usize,
    /// Maximum elements in one command frame.
    pub max_array_elements: usize,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            max_bulk_bytes: 64 * 1024 * 1024,
            max_array_elements: 256,
        }
    }
}

/// The outcome of scanning one command frame.
#[derive(Debug)]
pub enum Parsed<'a> {
    /// The buffer does not hold a complete frame yet. Consume nothing and
    /// wait for more bytes.
    Incomplete,
    /// Syntactically impossible. The connection is answered once and closed:
    /// the parser never resynchronises by guessing where a frame ends.
    Malformed,
    /// An empty command array.
    Empty,
    /// More elements or arguments than the connection allows. The frame is
    /// skipped whole and refused, so pipelined commands behind it still run
    /// and the connection stays usable.
    TooManyArgs,
    /// A non-string element (integer, error, nil, nested array, push).
    BadArgument,
    /// A bulk string past the connection's bound. Fatal, like Redis's
    /// `proto-max-bulk-len` refusal: the frame cannot be buffered, so its
    /// end is unknowable and the connection cannot be resynchronised.
    TooLarge,
    /// A syntactically valid RESP frame that is not a command — an inline
    /// string, a bare integer, a lone bulk. Answered with an error and the
    /// connection stays open, which is what Kivi has always done here.
    NotACommand,
    /// A complete command, borrowed from the input buffer.
    Command(Command<'a>),
}

impl Parsed<'_> {
    /// Whether the connection must close after answering.
    #[must_use]
    pub const fn is_fatal(&self) -> bool {
        matches!(self, Self::Malformed | Self::TooLarge)
    }
}

/// A complete command frame, borrowed from the connection's input buffer.
///
/// `args[0]` is the command name, so the name and the argument list are one
/// contiguous inline array and neither costs an allocation.
#[derive(Debug, Clone)]
pub struct Command<'a> {
    args: [&'a [u8]; MAX_ARGS],
    argc: usize,
}

impl<'a> Command<'a> {
    /// Command name exactly as sent (case preserved: the dispatcher matches
    /// case-insensitively, Redis does the same).
    #[must_use]
    pub fn name(&self) -> &'a [u8] {
        self.args[0]
    }

    /// Arguments after the name.
    #[must_use]
    pub fn args(&self) -> &[&'a [u8]] {
        &self.args[1..self.argc]
    }

    /// Name plus arguments, in wire order.
    #[must_use]
    pub fn all(&self) -> &[&'a [u8]] {
        &self.args[..self.argc]
    }
}

/// Outcome of skipping one well-formed non-command frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ScalarScan {
    /// The frame occupies this many bytes.
    Length(usize),
    /// Not all of it has arrived.
    Incomplete,
    /// It can never be legal.
    Malformed,
}

/// Measures one well-formed RESP frame that is not a command array, so the
/// connection can answer with an error and keep its place in the stream.
fn scan_scalar(buf: &[u8], limits: &Limits) -> ScalarScan {
    match buf.first() {
        Some(b'+' | b'-' | b':') => match find_line_end(buf.get(1..).unwrap_or(&[])) {
            LineEnd::At { used, .. } => ScalarScan::Length(1 + used),
            LineEnd::Incomplete => ScalarScan::Incomplete,
            LineEnd::TooLong => ScalarScan::Malformed,
        },
        Some(b'$') => {
            let (len, used) = match read_length(buf.get(1..).unwrap_or(&[])) {
                Ok(Some(parsed)) => parsed,
                Ok(None) => return ScalarScan::Incomplete,
                Err(()) => return ScalarScan::Malformed,
            };
            if len > limits.max_bulk_bytes {
                return ScalarScan::Malformed;
            }
            let start = 1 + used;
            let Some(end) = start.checked_add(len) else {
                return ScalarScan::Malformed;
            };
            if buf.get(start..end).is_none() {
                return ScalarScan::Incomplete;
            }
            match buf.get(end..end + 2) {
                None => ScalarScan::Incomplete,
                Some(b"\r\n") => ScalarScan::Length(end + 2),
                Some(_) => ScalarScan::Malformed,
            }
        }
        _ => ScalarScan::Malformed,
    }
}

/// Where a length line ends inside a body, and why.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LineEnd {
    /// The line's content is `body[..content]` and it consumed `used` bytes
    /// including the terminator.
    At {
        /// Bytes before the terminator.
        content: usize,
        /// Bytes including the terminator.
        used: usize,
    },
    /// No terminator yet, and the line can still become legal.
    Incomplete,
    /// No terminator, and the line is already longer than any legal one.
    TooLong,
}

/// Scans one command frame from the front of `buf`.
///
/// Returns the outcome and the frame's byte length. A consumed count of zero
/// accompanies only [`Parsed::Incomplete`]: wait for more bytes, buffer
/// intact.
#[must_use]
pub fn parse_command<'a>(buf: &'a [u8], limits: &Limits) -> (Parsed<'a>, usize) {
    // Inline commands (a bare `PING\r\n` typed at a socket) are outside this
    // profile. A frame that is nonetheless well-formed RESP is answered with
    // an error and the connection stays open; bytes that are not RESP at all
    // cannot be resynchronised, so the connection closes.
    match buf.first() {
        Some(b'*') => {}
        Some(b'+' | b'-' | b':' | b'$') => {
            return match scan_scalar(buf, limits) {
                ScalarScan::Length(len) => (Parsed::NotACommand, len),
                ScalarScan::Incomplete => (Parsed::Incomplete, 0),
                ScalarScan::Malformed => (Parsed::Malformed, buf.len()),
            };
        }
        _ => return (Parsed::Malformed, buf.len()),
    }

    let elements = match read_length(buf.get(1..).unwrap_or(&[])) {
        Ok(Some((value, used))) => (value, 1 + used),
        Ok(None) => return (Parsed::Incomplete, 0),
        Err(()) => return (Parsed::Malformed, buf.len()),
    };
    let (elements, mut pos) = elements;
    if elements == 0 {
        return (Parsed::Empty, 0);
    }
    if elements > limits.max_array_elements || elements > MAX_ARGS {
        // The element count is in hand, so the frame's end is computable
        // without materialising arguments. Skipping it keeps the connection
        // in sync, which matters: a pipelined client may have queued real
        // work behind the refused frame.
        return match frame_length(buf, elements, limits) {
            Some(len) => (Parsed::TooManyArgs, len),
            None => (Parsed::Incomplete, 0),
        };
    }

    let mut args = [&[] as &[u8]; MAX_ARGS];
    for slot in args.iter_mut().take(elements) {
        let Some(byte) = buf.get(pos).copied() else {
            return (Parsed::Incomplete, 0);
        };
        let body = pos + 1;
        match byte {
            b'$' => {
                let (len, used) = match read_length(buf.get(body..).unwrap_or(&[])) {
                    Ok(Some(parsed)) => parsed,
                    Ok(None) => return (Parsed::Incomplete, 0),
                    Err(()) => return (Parsed::Malformed, buf.len()),
                };
                if len > limits.max_bulk_bytes {
                    return (Parsed::TooLarge, buf.len());
                }
                let start = body + used;
                let Some(end) = start.checked_add(len) else {
                    return (Parsed::Malformed, buf.len());
                };
                let Some(payload) = buf.get(start..end) else {
                    return (Parsed::Incomplete, 0);
                };
                // A missing terminator is "not all here yet", not garbage: a
                // frame split across two reads must not be refused.
                match buf.get(end..end + 2) {
                    None => return (Parsed::Incomplete, 0),
                    Some(b"\r\n") => {}
                    Some(_) => return (Parsed::Malformed, buf.len()),
                }
                *slot = payload;
                pos = end + 2;
            }
            b'+' => match find_line_end(buf.get(body..).unwrap_or(&[])) {
                LineEnd::At { content, used } => {
                    *slot = &buf[body..body + content];
                    pos = body + used;
                }
                LineEnd::Incomplete => return (Parsed::Incomplete, 0),
                LineEnd::TooLong => return (Parsed::Malformed, buf.len()),
            },
            _ => return (Parsed::BadArgument, buf.len()),
        }
    }

    (
        Parsed::Command(Command {
            args,
            argc: elements,
        }),
        pos,
    )
}

/// Byte length of a frame carrying `elements` elements, or `None` while the
/// buffer does not hold it all. Only used to skip a refused frame.
fn frame_length(buf: &[u8], elements: usize, limits: &Limits) -> Option<usize> {
    let mut pos = 1 + read_length(buf.get(1..)?).ok()?.or(None)?.1;
    for _ in 0..elements {
        let byte = *buf.get(pos)?;
        let body = pos + 1;
        match byte {
            b'$' => {
                let (len, used) = read_length(buf.get(body..)?).ok()?.or(None)?;
                if len > limits.max_bulk_bytes {
                    return None;
                }
                let end = body.checked_add(used)?.checked_add(len)?;
                buf.get(body + used..end)?;
                buf.get(end..end + 2)?;
                pos = end + 2;
            }
            b'+' => match find_line_end(buf.get(body..)?) {
                LineEnd::At { used, .. } => pos = body + used,
                LineEnd::Incomplete | LineEnd::TooLong => return None,
            },
            _ => return None,
        }
    }
    buf.get(..pos)?;
    Some(pos)
}

/// Reads a RESP length line body (the digits after a type byte).
///
/// `Ok(Some((value, used)))` is a parsed length and `used` counts the digits
/// plus the terminator. `Ok(None)` means the line has not fully arrived;
/// `Err(())` means it can never be legal.
fn read_length(body: &[u8]) -> Result<Option<(usize, usize)>, ()> {
    let (content, used) = match find_line_end(body) {
        LineEnd::At { content, used } => (content, used),
        LineEnd::Incomplete => return Ok(None),
        LineEnd::TooLong => return Err(()),
    };
    let digits = &body[..content];
    if digits.is_empty() {
        return Err(());
    }
    let mut value: usize = 0;
    for byte in digits {
        if !byte.is_ascii_digit() {
            return Err(());
        }
        value = value
            .checked_mul(10)
            .and_then(|shifted| shifted.checked_add(usize::from(*byte - b'0')))
            .ok_or(())?;
    }
    Ok(Some((value, used)))
}

/// Locates a line terminator, distinguishing "not yet" from "never legal".
fn find_line_end(body: &[u8]) -> LineEnd {
    let limit = body.len().min(MAX_LEN_LINE);
    for index in 0..limit {
        if body[index] == b'\n' {
            // A bare `\n` terminates too: `content` is the same either way,
            // but `used` has to count what was actually consumed.
            let content = if index > 0 && body[index - 1] == b'\r' {
                index - 1
            } else {
                index
            };
            return LineEnd::At {
                content,
                used: index + 1,
            };
        }
    }
    if body.len() > MAX_LEN_LINE {
        LineEnd::TooLong
    } else {
        LineEnd::Incomplete
    }
}
