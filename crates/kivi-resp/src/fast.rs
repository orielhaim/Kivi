//! The direct read path: what a common command costs when it is not turned into
//! an [`Operation`](kivi_state::Operation) first.
//!
//! # Why this exists
//!
//! Measured, not assumed. `docs/perf-program.md` Part VI has the callgrind cost map
//! of a pipelined `GET` through production, at **4,087 instructions per command**,
//! against roughly 1,100 for the same parse, the same store and the same reply in
//! `kivi-wirelab` mode `c`. The 3,000-instruction difference is not spread evenly:
//!
//! | stage | Ir/cmd | what it is |
//! | --- | ---: | --- |
//! | allocator | 667 | an owned `Key`, a `Vec<Slot>`, a `Vec<Operation>`, a boxed future, a results `Vec` |
//! | engine wrapper | 549 | `handle_request` and the op-result plumbing for a local read |
//! | registry lookup | 539 | a linear scan with string compares over the whole command table |
//! | memcpy | 526 | the receive buffer copied into a second buffer, then compacted |
//! | clock | 299 | `SystemClock` read per command, through jiff |
//! | turn machinery | 223 | `DecodedTurn` with two fresh vectors, per turn |
//! | rendezvous | 179 | a same-thread crossbeam channel per command |
//! | `Operation` build | 118 | `translate` constructing a typed operation |
//!
//! None of that is *semantic* work. A `GET` needs: find the command, borrow the key,
//! probe the table, encode one bulk reply. This module is the part of production that
//! does exactly that, and it composes with the general path rather than replacing it.
//!
//! # The three properties that make it cheap
//!
//! 1. **Dispatch is a `match`, not a lookup.** [`classify`] compares the folded name
//!    against a handful of literals. The general path calls
//!    [`command::lookup`](crate::command::lookup), which is
//!    `REGISTRY.iter().find(|spec| spec.name == name)` - a linear scan with a string
//!    comparison per entry, over a registry of about sixty commands. That is the
//!    539 instructions, and no amount of caching the *result* fixes it because the
//!    scan happens before there is a result.
//!
//! 2. **The key is borrowed.** No [`kivi_state::Key`], so no allocation, and the
//!    bytes handed to the store are the bytes the parser produced.
//!
//! 3. **Execution is an ordinary function call.** [`Executor::fast_read`] returns a
//!    [`Reply`], not a future, not a channel message, not a slot index. A read that
//!    cannot be answered without suspending says so and falls back.
//!
//! # What this deliberately does not do
//!
//! It does not implement semantics. Every reply here is the same
//! [`Reply`] the general path would have produced, produced by the same
//! authoritative primitives below `Operation` - the store's own `get`, the same
//! clock, the same expiry rules. A second implementation of `GET` would be a second
//! thing to get wrong, and the whole point of the existing design is that there is
//! exactly one authority for what a command means.
//!
//! It also does not reorder anything. See [`crate::connection`] for the ordering
//! rule, which is what makes a fast prefix safe to run ahead of the general path.

use kivi_types::WallTimestamp;

use crate::translate::Reply;

/// A command this module can answer without an [`Operation`](kivi_state::Operation).
///
/// The set is small on purpose. Every entry is a pure read of one key whose reply
/// needs nothing the general path does not already provide, and every entry has to
/// be one whose answer cannot change between two points in the same turn. A write is
/// never here: a write that ran ahead of a later read would change what the read
/// sees, and the whole design of this path is that it may run ahead.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FastKind {
    /// `GET key` - the value, or nil.
    Get,
    /// `EXISTS key` - 1 or 0.
    Exists,
    /// `STRLEN key` - the value's byte length, 0 when absent.
    StrLen,
    /// `TTL key` - whole seconds remaining, or -1 / -2.
    Ttl,
    /// `PTTL key` - milliseconds remaining, or -1 / -2.
    Pttl,
}

impl FastKind {
    /// Whether this command takes exactly one key argument.
    ///
    /// Every current member does, and the check is explicit rather than assumed,
    /// because a future member that did not would otherwise be dispatched with an
    /// argument count the connection never verified. It takes no `self` because
    /// today every variant returns the same answer; when one stops doing so, giving
    /// it a `self` and a `match` is the change, and keeping the signature
    /// receiver-shaped now means that change is local.
    const fn takes_one_key() -> bool {
        true
    }
}

/// Classifies a folded command name and argument count.
///
/// `folded` is the parser's uppercase fold from
/// [`fold_command_name`](crate::frame::fold_command_name) and `name_len` how much
/// of it is the name; the rest is zero padding, which is why the length is needed
/// and the bytes alone are not enough.
///
/// `None` means "not a fast command", and the caller falls back to the general path.
/// That is the answer for every command with a side effect, every command needing
/// the general operation graph, and every command whose name is not one of these.
#[must_use]
pub fn classify(folded: &[u8], name_len: usize, argc: usize) -> Option<FastKind> {
    // `argc` is the total argument count including the command name, which is what
    // Redis and this crate both count. A wrong-arity fast command is not an error
    // here - it is a refusal, and the general path produces the exact Redis wording
    // for it, so falling back keeps one implementation of that message.
    let expected = 2;
    if argc != expected {
        return None;
    }
    let name = folded.get(..name_len)?;
    let kind = match name {
        b"GET" => FastKind::Get,
        b"EXISTS" => FastKind::Exists,
        b"STRLEN" => FastKind::StrLen,
        b"TTL" => FastKind::Ttl,
        b"PTTL" => FastKind::Pttl,
        _ => return None,
    };
    FastKind::takes_one_key().then_some(kind)
}

/// The engine's answer to a [`FastKind`], or "I cannot do this directly".
///
/// Declared here rather than on [`Executor`] so the trait file does not have to know
/// about the fast path's types, and so this module owns its own vocabulary.
pub trait FastReader {
    /// Answers `kind` for `key` without suspending, or returns `None`.
    ///
    /// `None` is not an error. It means the answer is not available synchronously -
    /// a chunked value that must be resolved, a key on a tablet this worker does not
    /// own, a command needing the general graph - and the caller must use the general
    /// path. Every `None` is a correctness-preserving fallback, never a silent wrong
    /// answer.
    fn fast_read(&self, kind: FastKind, key: &[u8], now: WallTimestamp) -> Option<Reply>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame::fold_command_name;

    /// Folds a name the way the parser would, defaulting to a zeroed array for a
    /// name the parser rejects outright.
    ///
    /// `fold_command_name` returns `None` for an empty, over-long or non-ASCII name
    /// - those are refusals, not folds - so the helper has to decide what a caller
    /// sees for them. Zeroed is the honest answer for "the parser never produced a
    /// name", and it is also the case a matching bug is most likely to hide in: a
    /// zeroed array truncated to a real length is not any command.
    fn folded(name: &[u8]) -> ([u8; 24], usize) {
        let bytes = fold_command_name(name).unwrap_or([0u8; 24]);
        (bytes, name.len())
    }

    #[test]
    fn every_fast_command_is_recognised_case_insensitively() {
        for name in [
            &b"GET"[..],
            b"get",
            b"Get",
            b"EXISTS",
            b"exists",
            b"STRLEN",
            b"strlen",
            b"TTL",
            b"ttl",
            b"PTTL",
            b"pttl",
        ] {
            let (bytes, len) = folded(name);
            assert!(
                classify(&bytes, len, 2).is_some(),
                "{name:?} should be fast"
            );
        }
    }

    #[test]
    fn a_write_is_never_fast() {
        // The safety property, tested directly: a fast prefix may run ahead of the
        // general path, so anything with a visible effect must be excluded. If a
        // future edit adds `SET` here, this fails.
        for name in [
            &b"SET"[..],
            b"DEL",
            b"SETRANGE",
            b"INCR",
            b"APPEND",
            b"GETSET",
            b"EXPIRE",
        ] {
            let (bytes, len) = folded(name);
            assert!(
                classify(&bytes, len, 2).is_none(),
                "{name:?} has a side effect and must not be fast"
            );
        }
    }

    #[test]
    fn a_non_command_is_never_fast() {
        for name in [&b"PING"[..], b"QUIT", b"HELLO", b"NOPE", b"G", b"GETX", b""] {
            let (bytes, len) = folded(name);
            assert!(classify(&bytes, len, 2).is_none(), "{name:?} is not fast");
        }
    }

    #[test]
    fn wrong_arity_falls_back_so_the_general_path_owns_the_error() {
        let (bytes, len) = folded(b"GET");
        // Zero, two and three arguments: none of them a `GET key`.
        for argc in [0usize, 1, 3, 4] {
            assert!(
                classify(&bytes, len, argc).is_none(),
                "argc {argc} must fall back"
            );
        }
        assert!(classify(&bytes, len, 2).is_some());
    }

    /// The padding trap: the fold is a fixed 24-byte array zero-padded, so comparing
    /// the whole array would make every name look wrong. A first version that
    /// matched on the array without truncating recognised nothing at all.
    #[test]
    fn the_zero_padding_does_not_hide_a_match() {
        let (bytes, len) = folded(b"GET");
        assert_eq!(bytes[3], 0, "the fold pads with NULs past the name");
        assert!(classify(&bytes, len, 2).is_some());
        // And a name that is a prefix of a fast one must not match.
        let (bytes, len) = folded(b"GEX");
        assert!(classify(&bytes, len, 2).is_none());
    }
}
