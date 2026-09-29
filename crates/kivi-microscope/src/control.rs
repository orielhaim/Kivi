//! The control servers: isolating infrastructure cost from database cost.
//!
//! A performance comparison needs a third point. "Kivi is 0.5× Redis" says
//! nothing about which of the two is wrong, because the difference could be in
//! either server, in the client, or in the machine. A control server answers
//! that by being a *deliberately incomplete* server whose only job is to perform
//! one stage of the path and nothing else, so the cost of each stage is
//! measurable on its own.
//!
//! # Why this is the most useful thing in the measurement program
//!
//! The previous campaign reached a wrong diagnosis from a plausible number. It
//! attributed a whole per-request cost to "the engine rendezvous" on the
//! strength of an in-process measurement, and the real split turned out to be
//! roughly half frontend, half engine - neither of which could have been found
//! from the end-to-end ratio alone. One extra server, ~80 lines, would have
//! separated them immediately. That is the entire justification for these.
//!
//! # The ladder
//!
//! Each server adds exactly one stage to the one above it, so the difference
//! between adjacent rows is the marginal cost of the added stage:
//!
//! | server | socket | RESP parse | classify | hash | table | semantics |
//! | --- | :-: | :-: | :-: | :-: | :-: | :-: |
//! | [`NoOp`] | ✓ | | | | | |
//! | [`ParseOnly`] | ✓ | ✓ | | | | |
//! | [`HashOnly`] | ✓ | ✓ | ✓ | ✓ | | |
//! | [`TableOnly`] | ✓ | ✓ | ✓ | ✓ | ✓ | |
//! | Kivi | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
//!
//! The final row is the real server, so the difference between `TableOnly` and
//! Kivi is the semantic layer: typed values, versions, expiry, transactions,
//! durability selection. That is a number nothing else in the program can
//! produce, and it is often the largest one.
//!
//! # The threading model is part of the control
//!
//! The previous campaign also measured that thread-per-connection costs 1.38×
//! against a single event loop at the same work. That is only visible if the
//! controls share the *real* server's threading, and change only what they
//! answer. A control with a different threading model measures the wrong
//! difference, so every server here runs one thread per connection on a blocking
//! socket, which is what Kivi's RESP edge does.
//!
//! # What these are not
//!
//! Not competitors. They do not implement Redis semantics, do not validate
//! replies, and are not proposed for any purpose beyond subtraction. A control
//! that returns a wrong answer quickly is still a valid control, because the
//! question is what the stage costs, not what the server means.

use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};

/// How many staged servers exist, and the order they add up in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(usize)]
pub enum Control {
    /// Reads a request and answers `+PONG`. No parsing.
    NoOp,
    /// Parses RESP frames and answers a fixed reply per frame. No hashing.
    ParseOnly,
    /// Parses, classifies, and hashes the key, then answers. No table.
    HashOnly,
    /// Parses, classifies, hashes, and looks the key up in a table, then
    /// answers. No semantic layer: no versions, expiry, or types.
    TableOnly,
}

impl Control {
    /// Every control, in the order they add up.
    pub const LADDER: [Control; 4] = [
        Control::NoOp,
        Control::ParseOnly,
        Control::HashOnly,
        Control::TableOnly,
    ];

    /// Name for logs and command lines.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::NoOp => "noop",
            Self::ParseOnly => "parse-only",
            Self::HashOnly => "hash-only",
            Self::TableOnly => "table-only",
        }
    }

    /// Parses a name.
    #[must_use]
    pub fn from_name(name: &str) -> Option<Self> {
        Control::LADDER
            .into_iter()
            .find(|control| control.name() == name)
    }

    /// The row above this one in the ladder, whose cost this one adds to.
    #[must_use]
    pub const fn builds_on(self) -> Option<Control> {
        match self {
            Self::NoOp => None,
            Self::ParseOnly => Some(Self::NoOp),
            Self::HashOnly => Some(Self::ParseOnly),
            Self::TableOnly => Some(Self::HashOnly),
        }
    }
}

/// Empty-bucket marker. `u32` indices keep a bucket two words wide with its
/// fingerprint, which is what makes a control table's access pattern resemble a
/// real one's.
const EMPTY: u32 = u32::MAX;

/// A counting hash table standing in for the state store.
///
/// Owns the shape a real table has - hash, probe, compare - without the
/// semantics, so the difference between [`Control::HashOnly`] and
/// [`Control::TableOnly`] is the table's cost and nothing else.
///
/// Keys are generated from their index, so a miss is a genuine absent key rather
/// than a special case the probe can shortcut.
///
/// Explicitly **not** a `KiviTable` prototype. A real one is a separate program
/// with its own exit gate, and a control that grew into a competing design would
/// make the subtraction meaningless: the rung would measure the control's table,
/// not the store's.
#[derive(Debug)]
pub struct Table {
    /// Fingerprint by bucket, so a miss usually costs one cache line.
    fingerprints: Vec<u8>,
    /// Index into `entries` by bucket, [`EMPTY`] for a free bucket.
    slots: Vec<u32>,
    /// The keys and values this table holds, in insertion order.
    entries: Vec<(Vec<u8>, Vec<u8>)>,
    /// Bucket count minus one. Always a power of two minus one, so the bucket
    /// index is a mask rather than a division.
    mask: usize,
    /// Hits since the last [`Table::take_stats`].
    hits: u64,
    /// Misses since the last [`Table::take_stats`].
    misses: u64,
}

impl Table {
    /// A table of `capacity` rounded up to a power of two and half-filled, so it
    /// sits at a realistic load factor and probe chains are non-empty.
    #[must_use]
    pub fn with_capacity(capacity: usize, value_bytes: usize) -> Self {
        let buckets = capacity.next_power_of_two().max(8);
        let entries: Vec<(Vec<u8>, Vec<u8>)> = (0..buckets / 2)
            .map(|index| {
                let key = format!("bench:key:{index:016x}").into_bytes();
                let value = vec![b'x'; value_bytes];
                (key, value)
            })
            .collect();
        let mut table = Self {
            fingerprints: vec![0u8; buckets],
            slots: vec![EMPTY; buckets],
            entries,
            mask: buckets - 1,
            hits: 0,
            misses: 0,
        };
        for index in 0..table.entries.len() {
            let (key, _) = &table.entries[index];
            let hash = hash_key(key);
            let slot = u32::try_from(index).unwrap_or(EMPTY);
            let mut placed = table.bucket_for(hash);
            if table.slots[placed] != EMPTY {
                placed = (placed + 1) & table.mask;
                while table.slots[placed] != EMPTY {
                    placed = (placed + 1) & table.mask;
                }
            }
            table.slots[placed] = slot;
            // The fingerprint belongs to the slot the entry *landed in*, not to
            // its home bucket. Recording it at the home bucket instead makes
            // every probed entry invisible to its own lookup - the key is in the
            // table and the table cannot find it, which is precisely the class of
            // bug a control server must not have.
            table.fingerprints[placed] = fingerprint(hash);
        }
        table
    }

    /// The bucket a hash lands in.
    #[inline]
    fn bucket_for(&self, hash: u64) -> usize {
        // Mixing the high bits into the index is what keeps the FNV prime's weak
        // low bits from clustering the control's own dataset. A control whose
        // dataset clustered would measure a degenerate probe and understate a
        // real table's cost.
        #[allow(
            clippy::cast_possible_truncation,
            reason = "the mask bounds the index to the bucket count, which is a usize"
        )]
        let mixed = (hash ^ (hash >> 32)) as usize;
        mixed & self.mask
    }

    /// Looks `key` up, returning its value.
    #[inline]
    #[must_use]
    pub fn get(&mut self, key: &[u8]) -> Option<&[u8]> {
        let hash = hash_key(key);
        let wanted = fingerprint(hash);
        let mut bucket = self.bucket_for(hash);
        loop {
            let index = self.slots[bucket];
            if index == EMPTY {
                self.misses += 1;
                return None;
            }
            if self.fingerprints[bucket] == wanted {
                let (stored_key, value) = &self.entries[index as usize];
                if stored_key == key {
                    self.hits += 1;
                    return Some(value);
                }
            }
            bucket = (bucket + 1) & self.mask;
        }
    }

    /// Hits and misses since the last call, clearing them.
    #[must_use]
    pub fn take_stats(&mut self) -> (u64, u64) {
        let stats = (self.hits, self.misses);
        self.hits = 0;
        self.misses = 0;
        stats
    }
}

/// The FNV-1a offset basis, as a `u64`.
///
/// A control server's hash exists to be *a* hash of the right cost, not to be a
/// good one. Using anything cleverer would import a variable this stage is meant
/// to hold constant, and the program compares hashing functions in a different
/// place entirely.
const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;

/// The FNV-1a prime.
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// Hashes a key with FNV-1a, 64-bit.
///
/// Byte-at-a-time, so it is a realistic short-key hash and not a single
/// instruction. The FNV prime's low bits are poor, which is fine: the table
/// masks to the low bits and a control only needs the same access pattern a real
/// table produces.
#[inline]
fn hash_key(key: &[u8]) -> u64 {
    let mut hash = FNV_OFFSET;
    for byte in key {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

/// Seven bits of the hash, stored per bucket, with the top bit set.
///
/// The top bit is set so a fingerprint is never zero, which lets a table use a
/// zeroed control array as an "empty" signal. A control does not need that, but
/// the constraint is what makes its access pattern resemble a real table's.
#[inline]
fn fingerprint(hash: u64) -> u8 {
    #[allow(
        clippy::cast_possible_truncation,
        reason = "the mask keeps the value in 0..=0x7f, so the cast is a truncation by design"
    )]
    let low = (hash >> 32) as u8 & 0x7f;
    low | 0x80
}

/// The reply a control server gives, chosen so the bytes are plausible and the
/// encoding work is the same on both sides.
const REPLY: &[u8] = b"+PONG\r\n";

/// Serves `control` on `addr` until the process is killed.
///
/// One thread per connection, blocking sockets. This is deliberately Kivi's
/// threading model: a control with a different model would measure the threading
/// difference instead of the stage difference, which is the mistake that made the
/// previous campaign's no-op floor incomparable to Redis.
///
/// # Errors
///
/// Returns the bind or accept error. A control that cannot bind has nothing to
/// measure, and reporting it is better than listening on nothing.
pub fn serve(control: Control, addr: SocketAddr) -> std::io::Result<()> {
    let listener = TcpListener::bind(addr)?;
    let bound = listener.local_addr()?;
    println!(
        "control {} listening on {bound} (one thread per connection)",
        control.name()
    );
    // A pre-sized table shared read-only would be faster, but a control has to
    // own its table to count its probes, and probe counting is the point of the
    // top rung.
    let table = Table::with_capacity(1024 * 1024, 128);
    for stream in listener.incoming() {
        let stream = stream?;
        let _ = stream.set_nodelay(true);
        let table = clone_table(&table);
        std::thread::spawn(move || {
            let _ = serve_connection(control, stream, table);
        });
    }
    Ok(())
}

/// A per-connection copy of the table's contents.
///
/// Cloning per connection is a cost the real server does not pay, and it is paid
/// once per connection rather than per request, so it does not enter a
/// throughput measurement. The alternative - a shared table behind a lock - would
/// add a contention point that has no counterpart in Kivi and would corrupt the
/// very comparison it exists to support.
fn clone_table(template: &Table) -> Table {
    Table {
        fingerprints: template.fingerprints.clone(),
        slots: template.slots.clone(),
        entries: template.entries.clone(),
        mask: template.mask,
        hits: 0,
        misses: 0,
    }
}

/// Serves one connection until it closes.
fn serve_connection(
    control: Control,
    mut stream: TcpStream,
    mut table: Table,
) -> std::io::Result<()> {
    let mut read_buffer = vec![0u8; 64 * 1024];
    let mut write_buffer = Vec::with_capacity(64 * 1024);
    let mut parser = FrameParser::default();
    loop {
        let count = match stream.read(&mut read_buffer) {
            Ok(0) | Err(_) => return Ok(()),
            Ok(count) => count,
        };
        parser.push(&read_buffer[..count]);
        while let Some(frame) = parser.next_frame() {
            write_buffer.clear();
            match control {
                // The two rungs that only read bytes answer the same thing, and
                // the shared arm is written out here rather than as a single
                // merged pattern so the ladder stays readable top to bottom next
                // to the table in this module's documentation.
                Control::NoOp | Control::ParseOnly => write_buffer.extend_from_slice(REPLY),
                Control::HashOnly => {
                    if frame.has_key() {
                        // The two rungs differ here and nowhere else: the parse is
                        // already done, so what is added is exactly one hash.
                        let key = frame
                            .argument(parser.buffer(), KEY_ARGUMENT)
                            .unwrap_or_default();
                        std::hint::black_box(hash_key(key));
                    }
                    write_buffer.extend_from_slice(REPLY);
                }
                Control::TableOnly => {
                    // `has_key` is checked before the lookup because an empty key
                    // is a legal key to a real table and a meaningless one here:
                    // without the check a `PING` would probe for `""` and the
                    // miss count would attribute it to the table.
                    match frame
                        .has_key()
                        .then(|| {
                            frame
                                .argument(parser.buffer(), KEY_ARGUMENT)
                                .unwrap_or_default()
                        })
                        .and_then(|key| table.get(key))
                    {
                        Some(value) => encode_bulk(&mut write_buffer, value),
                        None => write_buffer.extend_from_slice(b"$-1\r\n"),
                    }
                }
            }
        }
        if write_buffer.is_empty() {
            continue;
        }
        if stream.write_all(&write_buffer).is_err() {
            return Ok(());
        }
    }
}

/// Writes a RESP bulk string: `$<len>\r\n<bytes>\r\n`.
fn encode_bulk(out: &mut Vec<u8>, value: &[u8]) {
    out.push(b'$');
    out.extend_from_slice(value.len().to_string().as_bytes());
    out.extend_from_slice(b"\r\n");
    out.extend_from_slice(value);
    out.extend_from_slice(b"\r\n");
}

/// A minimal RESP frame scanner.
///
/// Finds frame boundaries and element spans, and nothing else: no validation
/// beyond the marker and the digits, no arity, no semantics. It exists so a
/// control server pays a *parse* cost that resembles Kivi's without becoming a
/// second parser to keep correct.
#[derive(Debug, Default)]
struct FrameParser {
    /// Unconsumed input, and how much of it has been parsed.
    buffer: Vec<u8>,
    cursor: usize,
}

/// Element count a control parser tracks.
///
/// Enough for every command a control reads a key from. A control is driven by a
/// benchmark client, so an oversized frame is a client bug, and ignoring the
/// surplus elements is the correct response to a benchmark that sent nonsense.
const MAX_ARGUMENTS: usize = 8;

/// Index of the key within a command frame.
///
/// Element 1, because element 0 is the command name. A control that hashed
/// `"GET"` instead of the key would produce a plausible number for entirely the
/// wrong work: a table with one hot key and a near-perfect hit rate, reported as
/// a hashing cost.
const KEY_ARGUMENT: usize = 1;

impl FrameParser {
    /// Appends newly read bytes.
    ///
    /// Reclaims the consumed prefix here rather than in
    /// [`FrameParser::next_frame`], because a frame handed out from the previous
    /// call still points into this buffer. Clearing on the way out would
    /// invalidate a frame the caller is still reading - the buffer empties and
    /// every element of the last frame comes back `None`.
    fn push(&mut self, bytes: &[u8]) {
        if self.cursor > 0 {
            self.buffer.drain(..self.cursor);
            self.cursor = 0;
        }
        self.buffer.extend_from_slice(bytes);
    }

    /// The parser's buffer, for reading a [`Frame`]'s elements.
    ///
    /// Exposed because a frame carries offsets rather than a borrow - see
    /// [`FrameParser::next_frame`] for why - so a caller reads its elements
    /// through the buffer it came from.
    fn buffer(&self) -> &[u8] {
        &self.buffer
    }

    /// Pops the next complete frame, or `None` while one is still arriving.
    ///
    /// A real parser distinguishes "incomplete" from "malformed" and refuses the
    /// second; a control does not, because it is only ever driven by a benchmark
    /// client that sends well-formed frames. A malformed frame stalls the
    /// connection, which is acceptable for a subtraction-only server and is
    /// recorded here so nobody mistakes it for a parser.
    ///
    /// Returns *offsets*, not a borrow. A frame carrying a borrow of the parser
    /// would pin the parser mutably for as long as the frame lives, so two
    /// frames from one buffer could not coexist - which is the borrow conflict
    /// Kivi's own RESP connection resolves by moving the buffer aside, and a
    /// control should not have to think about. Offsets cost two words and let the
    /// caller read bytes and continue parsing in any order.
    fn next_frame(&mut self) -> Option<Frame> {
        let scanned = self.scan()?;
        let start = self.cursor;
        self.cursor += scanned.length;
        Some(Frame {
            arguments: scanned.arguments,
            argument_count: scanned.argument_count,
            start,
        })
    }

    /// Scans the frame at the cursor.
    ///
    /// `None` means the frame has not fully arrived. The same `None` is returned
    /// for a frame that can never be legal, which is the documented
    /// simplification above.
    fn scan(&self) -> Option<Scanned> {
        let tail = self.buffer.get(self.cursor..)?;
        let mut position = 0usize;
        let mut arguments = [(0usize, 0usize); MAX_ARGUMENTS];
        let mut count = 0usize;

        // `read_length` validates the marker itself, so it is given the slice
        // *starting at* the marker. Handing it the slice after the marker would
        // make the marker check compare a digit against `*` and refuse every
        // frame.
        let (elements, used) = read_length(tail.get(position..)?, b'*')?;
        position += used;
        for _ in 0..elements {
            let (length, used) = read_length(tail.get(position..)?, b'$')?;
            position += used;
            // The payload and its CRLF must both be present before the span is
            // recorded, so a truncated frame never yields a short key that a
            // later stage would hash as if it were whole.
            tail.get(position..position + length)?;
            tail.get(position + length..position + length + 2)?;
            if count < MAX_ARGUMENTS {
                arguments[count] = (position, position + length);
                count += 1;
            }
            position += length + 2;
        }
        Some(Scanned {
            length: position,
            arguments,
            argument_count: count,
        })
    }
}

/// What one scan of a frame found.
#[derive(Debug, Clone, Copy)]
struct Scanned {
    /// Frame length in bytes.
    length: usize,
    /// Span of each argument, relative to the frame's start.
    arguments: [(usize, usize); MAX_ARGUMENTS],
    /// How many of `arguments` are populated.
    argument_count: usize,
}

/// One frame's location in the parser's buffer.
///
/// Holds offsets rather than borrows, so several frames from one buffer can be
/// held at once while the parser keeps consuming.
#[derive(Debug, Clone, Copy)]
struct Frame {
    /// Byte offset the frame starts at.
    start: usize,
    /// Span of each element, relative to `start`.
    arguments: [(usize, usize); MAX_ARGUMENTS],
    /// How many of `arguments` are populated.
    argument_count: usize,
}

impl Frame {
    /// The `index`th element of the frame, read out of `buffer`.
    ///
    /// Bounded by `argument_count`, not by the array's length. The unused tail
    /// of `arguments` holds a zero span, and indexing it directly would return
    /// `Some(&[])` for an element the frame does not have - an empty key that
    /// hashes successfully and misses the table, blaming the table for a frame
    /// that never had that argument.
    fn argument<'buffer>(&self, buffer: &'buffer [u8], index: usize) -> Option<&'buffer [u8]> {
        if index >= self.argument_count {
            return None;
        }
        let (from, to) = *self.arguments.get(index)?;
        buffer.get(self.start + from..self.start + to)
    }

    /// Whether the frame carries a key, by element count alone.
    fn has_key(&self) -> bool {
        self.argument_count > KEY_ARGUMENT
    }
}

/// Reads `<marker><digits>CRLF` from the front of `line`, returning
/// `(value, bytes consumed including the terminator)`.
///
/// `None` means the line has not fully arrived, or can never be legal; both stop
/// the frame, which is the documented simplification above.
///
/// A bare `LF` is accepted because some clients emit it, and a control that
/// refused those frames would measure a stalled connection rather than a parse.
/// A non-digit is rejected rather than read as a zero, so a malformed length
/// stops the frame instead of being interpreted as a different length.
fn read_length(line: &[u8], marker: u8) -> Option<(usize, usize)> {
    if line.first() != Some(&marker) {
        return None;
    }
    let newline = line.iter().position(|byte| *byte == b'\n')?;
    // Strip an optional CR so the digits are only digits.
    let with_carriage_return = line.get(1..newline)?;
    let digits = with_carriage_return
        .strip_suffix(b"\r")
        .unwrap_or(with_carriage_return);
    if digits.is_empty() {
        return None;
    }
    let mut value = 0usize;
    for digit in digits {
        if !digit.is_ascii_digit() {
            return None;
        }
        value = value
            .checked_mul(10)?
            .checked_add(usize::from(*digit - b'0'))?;
    }
    Some((value, newline + 1))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_ladder_is_ordered_and_named() {
        assert_eq!(Control::LADDER.len(), 4);
        for (index, control) in Control::LADDER.into_iter().enumerate() {
            assert_eq!(Control::from_name(control.name()), Some(control));
            assert_eq!(
                control.builds_on(),
                index.checked_sub(1).map(|i| Control::LADDER[i])
            );
        }
        assert_eq!(Control::NoOp.builds_on(), None);
        assert_eq!(Control::from_name("nope"), None);
    }

    #[test]
    fn the_table_finds_what_it_was_given_and_misses_cleanly() {
        let mut table = Table::with_capacity(1024, 8);
        let present = format!("bench:key:{:016x}", 0);
        assert!(table.get(present.as_bytes()).is_some());
        let (hits, misses) = table.take_stats();
        assert_eq!(hits, 1);
        assert_eq!(misses, 0);

        assert!(table.get(b"bench:key:ffffffffffffffff").is_none());
        let (hits, misses) = table.take_stats();
        assert_eq!(hits, 0, "stats must not double-count");
        assert_eq!(misses, 1);
    }

    #[test]
    fn the_table_does_not_depend_on_insertion_order_for_lookups() {
        let mut table = Table::with_capacity(256, 4);
        for index in 0..64 {
            let key = format!("bench:key:{index:016x}");
            assert!(table.get(key.as_bytes()).is_some(), "missing {key}");
        }
    }

    #[test]
    fn the_parser_finds_frame_boundaries() {
        let mut parser = FrameParser::default();
        parser.push(b"*2\r\n$3\r\nGET\r\n$3\r\nfoo\r\n");
        assert!(parser.next_frame().is_some());
        assert!(parser.next_frame().is_none(), "one frame was sent");

        // Split across reads, which is what a real client does.
        let mut parser = FrameParser::default();
        parser.push(b"*2\r\n$3\r\nGET\r\n$3\r");
        assert!(parser.next_frame().is_none(), "not all here yet");
        parser.push(b"\nfoo\r\n");
        assert!(parser.next_frame().is_some());
    }

    /// `used` counts the marker, the digits and the terminator, because a
    /// caller advances a cursor by it and a count that omitted the marker would
    /// walk every subsequent argument one byte to the left.
    #[test]
    fn length_lines_parse_and_reject() {
        assert_eq!(read_length(b"$12\r\nabc", b'$'), Some((12, 5)));
        assert_eq!(read_length(b"*3\r\n", b'*'), Some((3, 4)));
        assert_eq!(read_length(b"*3\n", b'*'), Some((3, 3)), "bare LF");
        assert_eq!(read_length(b"$1x\n", b'$'), None, "non-digit");
        assert_eq!(read_length(b"$x", b'$'), None, "no terminator");
        assert_eq!(read_length(b"12", b'$'), None, "wrong marker");
        assert_eq!(read_length(b"$\r\n", b'$'), None, "no digits");
    }

    /// The key a control reads must be the real key, not the command name.
    ///
    /// A control that hashed `"GET"` instead of the key would produce a
    /// plausible number for entirely the wrong work - a table with one hot key
    /// and a near-perfect hit rate, reported as a hash cost. This is the exact
    /// failure a control exists to prevent, and the first version of this parser
    /// had it.
    #[test]
    fn the_frame_key_is_the_key_not_the_command_name() {
        let mut parser = FrameParser::default();
        parser.push(b"*2\r\n$3\r\nGET\r\n$5\r\nkey42\r\n");
        let frame = parser.next_frame().expect("complete frame");
        assert_eq!(
            frame.argument(parser.buffer(), 0),
            Some(&b"GET"[..]),
            "element 0 is the command name"
        );
        assert_eq!(
            frame.argument(parser.buffer(), KEY_ARGUMENT),
            Some(&b"key42"[..]),
            "the key is element 1; hashing element 0 would measure a hot key"
        );
    }

    /// A frame split mid-payload must not yield a short key. If it did, the
    /// table would miss on a key that exists, and the miss count would blame the
    /// table for a parser bug.
    #[test]
    fn a_truncated_payload_yields_no_key() {
        let mut parser = FrameParser::default();
        parser.push(b"*2\r\n$3\r\nGET\r\n$5\r\nkey4");
        assert!(parser.next_frame().is_none());
        parser.push(b"2\r\n");
        let frame = parser.next_frame().expect("frame completes");
        assert_eq!(
            frame.argument(parser.buffer(), KEY_ARGUMENT),
            Some(&b"key42"[..])
        );
    }

    /// Several frames in one read must each come back, in order, with the right
    /// key. A cursor that failed to advance would spin on the first frame, and a
    /// frame carrying a borrow of the parser would not even compile.
    #[test]
    fn pipelined_frames_come_back_in_order() {
        let mut parser = FrameParser::default();
        parser.push(b"*2\r\n$3\r\nGET\r\n$3\r\naaa\r\n*2\r\n$3\r\nGET\r\n$3\r\nbbb\r\n");
        let first = parser.next_frame().expect("first frame");
        let second = parser.next_frame().expect("second frame");
        assert_eq!(
            first.argument(&parser.buffer, KEY_ARGUMENT),
            Some(&b"aaa"[..])
        );
        assert_eq!(
            second.argument(&parser.buffer, KEY_ARGUMENT),
            Some(&b"bbb"[..])
        );
        assert!(parser.next_frame().is_none());
    }

    /// A `PING` has no key, and a control must answer it rather than crash.
    #[test]
    fn a_frame_without_arguments_has_no_key() {
        let mut parser = FrameParser::default();
        parser.push(b"*1\r\n$4\r\nPING\r\n");
        let frame = parser.next_frame().expect("complete frame");
        assert!(!frame.has_key(), "PING is one element: the name");
        assert_eq!(frame.argument(parser.buffer(), 0), Some(&b"PING"[..]));
        assert_eq!(
            frame.argument(parser.buffer(), KEY_ARGUMENT),
            None,
            "no key, and not an empty one"
        );
    }
}
