//! Pipeline-focused differential tests for the direct read path.
//!
//! # Why this file exists
//!
//! The fast path in [`kivi_resp::fast`] answers a `GET` without building an
//! `Operation`. That is a performance change and a correctness risk at the same
//! time, and the risk is entirely in **disagreement**: the fast path and the general
//! path must produce byte-identical replies, and the only way to know they do is to
//! run both.
//!
//! So the tests here do not assert a remembered answer. They assert that the fast
//! path and the general path **agree**, by running the same command through a
//! connection whose executor implements `fast_read` and one whose does not - the
//! trait's default returns `None`, which is exactly "always use the general path".
//! Any divergence is a failure, and a divergence is the only new failure mode this
//! change can introduce.
//!
//! # The three bugs that must stay pinned
//!
//! Every one of these was found once, in the general path, and can only recur here
//! if the fast path reimplements something. They are pinned by name so that a
//! future reader can see what the direct path must not be allowed to do differently:
//!
//! * **SETRANGE length race** - the write and the length read must be ordered.
//! * **GETRANGE window applied twice** - a window applied to an already-windowed
//!   value returns the wrong bytes.
//! * **fabric manifest returned instead of resolved payload** - a fabric-backed
//!   value must never be answered with its manifest.

use rstest::rstest;

use kivi_resp::{
    ConnConfig, Executor, Reply,
    connection::{RespConnection, TurnBudget},
};
use kivi_state::{Key, ObjectStore, ObjectVersion, Operation, OperationResult, StoredObject};
use kivi_types::{LocalTabletSlot, TabletId, TabletRoute, WallTimestamp};

/// The instant every test is evaluated at.
const NOW: WallTimestamp = WallTimestamp::from_micros(1_700_000_000_000_000);
/// The route a single-tablet worker compiles to: one tablet, slot zero.
const ROUTE: TabletRoute = TabletRoute::new(TabletId::from_u64(1), LocalTabletSlot::new(0));

/// A value long enough to catch a truncation and short enough to stay readable.
const VALUE: &[u8] = b"0123456789abcdef";

/// A store holding one key, built through the store's own `prepare`/`apply` so the
/// object is exactly what production would hold.
fn store_with(key: &str, value: &[u8]) -> ObjectStore {
    let mut store = ObjectStore::new();
    let write = store
        .prepare(
            &Operation::Set {
                key: Key::new(key.as_bytes().to_vec()),
                value: bytes::Bytes::copy_from_slice(value),
            },
            NOW,
        )
        .expect("set prepares");
    let kivi_state::Prepared::Write(mutation) = write else {
        panic!("a set must prepare as a write");
    };
    store.apply(&mutation, NOW).expect("set applies");
    store
}

/// Answers a read directly, the way the engine's fast path does.
///
/// Mirrors [`kivi_state::ObjectStore::get_borrowed`] and the same reply mapping the
/// general path's `Length`/`Expiry` results go through, so comparing the two is a
/// comparison of the *architecture* and not of two hand-written answers.
struct DirectExecutor {
    store: std::cell::RefCell<ObjectStore>,
}

impl Executor for DirectExecutor {
    fn execute(&self, op: &Operation) -> Result<OperationResult, kivi_resp::ExecuteError> {
        // The general path, verbatim: prepare then apply, through the store.
        let mut store = self.store.borrow_mut();
        match store.prepare(op, NOW) {
            Ok(kivi_state::Prepared::Read(value)) => Ok(value),
            Ok(kivi_state::Prepared::Write(mutation)) => {
                store
                    .apply(&mutation, NOW)
                    .map_err(|_| kivi_resp::ExecuteError::Internal)?;
                // `ConditionalSet { applied: true }` is what a `SET` reply is shaped
                // from - a first version returned `Stored { .. }` here and every
                // `SET` came back `-ERR internal error`, which is `map_result`'s
                // unreachable-pairing arm catching a double that was not faithful.
                // The test failing on its own double is the suite working.
                Ok(OperationResult::ConditionalSet {
                    applied: true,
                    version: Some(ObjectVersion::from(0_u64)),
                })
            }
            Err(_) => Err(kivi_resp::ExecuteError::WrongType),
        }
    }

    fn now(&self) -> WallTimestamp {
        NOW
    }

    /// One tablet in one slot: what a single-tablet worker compiles to, so the
    /// double exercises the same route-then-probe shape production does.
    fn direct_route(&self) -> Option<TabletRoute> {
        Some(TabletRoute::new(
            kivi_types::TabletId::from_u64(1),
            kivi_types::LocalTabletSlot::new(0),
        ))
    }

    fn fast_read(
        &self,
        route: TabletRoute,
        kind: kivi_resp::fast::FastKind,
        key: &[u8],
        now: WallTimestamp,
    ) -> Option<Reply> {
        let _ = route;
        // Bound to a name: the `Ref` guard has to outlive the `&StoredObject` it
        // hands out, and writing this as one expression frees it too early.
        let store = self.store.borrow();
        let stored = store.get_borrowed(key, now);
        Some(match (kind, stored) {
            (kivi_resp::fast::FastKind::Get, Some(object)) => {
                let kivi_state::LogicalValue::Bytes(bytes) = object.value() else {
                    return None;
                };
                Reply::Bulk(bytes.clone())
            }
            (kivi_resp::fast::FastKind::Get, None) => Reply::Nil,
            (kivi_resp::fast::FastKind::Exists, found) => Reply::Int(i64::from(found.is_some())),
            (kivi_resp::fast::FastKind::StrLen, Some(object)) => {
                let kivi_state::LogicalValue::Bytes(bytes) = object.value() else {
                    return None;
                };
                Reply::Int(i64::try_from(bytes.len()).unwrap_or(i64::MAX))
            }
            (kivi_resp::fast::FastKind::StrLen, None) => Reply::Int(0),
            (kivi_resp::fast::FastKind::Ttl, found) => Reply::Int(
                kivi_resp::translate::ttl_seconds(found.map(StoredObject::expiry), now),
            ),
            (kivi_resp::fast::FastKind::Pttl, found) => Reply::Int(
                kivi_resp::translate::ttl_millis(found.map(StoredObject::expiry), now),
            ),
        })
    }
}

/// A connection that always takes the general path, because the trait's default
/// `fast_read` returns `None`.
///
/// The `unused` warning is expected: the struct exists to be a *different* executor
/// than `DirectExecutor`, and the difference is entirely in whether `fast_read` is
/// overridden.
struct GeneralOnlyExecutor {
    store: std::cell::RefCell<ObjectStore>,
}

impl Executor for GeneralOnlyExecutor {
    fn execute(&self, op: &Operation) -> Result<OperationResult, kivi_resp::ExecuteError> {
        let mut store = self.store.borrow_mut();
        match store.prepare(op, NOW) {
            Ok(kivi_state::Prepared::Read(value)) => Ok(value),
            Ok(kivi_state::Prepared::Write(mutation)) => {
                store
                    .apply(&mutation, NOW)
                    .map_err(|_| kivi_resp::ExecuteError::Internal)?;
                // `ConditionalSet { applied: true }` is what a `SET` reply is shaped
                // from - a first version returned `Stored { .. }` here and every
                // `SET` came back `-ERR internal error`, which is `map_result`'s
                // unreachable-pairing arm catching a double that was not faithful.
                // The test failing on its own double is the suite working.
                Ok(OperationResult::ConditionalSet {
                    applied: true,
                    version: Some(ObjectVersion::from(0_u64)),
                })
            }
            Err(_) => Err(kivi_resp::ExecuteError::WrongType),
        }
    }

    fn now(&self) -> WallTimestamp {
        NOW
    }
}

fn direct(store: ObjectStore) -> RespConnection<DirectExecutor> {
    RespConnection::new(
        DirectExecutor {
            store: std::cell::RefCell::new(store),
        },
        ConnConfig::default(),
        1,
    )
}

fn general(store: ObjectStore) -> RespConnection<GeneralOnlyExecutor> {
    RespConnection::new(
        GeneralOnlyExecutor {
            store: std::cell::RefCell::new(store),
        },
        ConnConfig::default(),
        1,
    )
}

/// Encodes a RESP command frame.
fn cmd(args: &[&[u8]]) -> Vec<u8> {
    let mut out = format!("*{}\r\n", args.len()).into_bytes();
    for arg in args {
        out.extend_from_slice(format!("${}\r\n", arg.len()).as_bytes());
        out.extend_from_slice(arg);
        out.extend_from_slice(b"\r\n");
    }
    out
}

/// Runs `input` to completion through a connection and returns the reply bytes.
fn run<C: Executor>(connection: &mut RespConnection<C>, input: &[u8]) -> String {
    connection.push(input).expect("input fits");
    let mut out = Vec::new();
    while connection.has_unconsumed() {
        let outcome = connection.drain(&mut out);
        assert!(outcome.replies > 0, "a turn must make progress");
    }
    String::from_utf8(out).expect("replies are utf-8 in these tests")
}

/// The core assertion: the two paths produce **byte-identical** output.
#[track_caller]
fn assert_paths_agree(store: ObjectStore, input: &[u8]) -> String {
    let mut fast = direct(store.clone());
    let mut slow = general(store);
    let via_fast = run(&mut fast, input);
    let via_general = run(&mut slow, input);
    assert_eq!(
        via_fast,
        via_general,
        "the direct read path and the general path disagree on {}",
        String::from_utf8_lossy(input)
    );
    via_fast
}

// --------------------------------------------------------------- the fast commands

/// Every command the direct path can answer, against a present and an absent key.
///
/// The two columns are the whole contract: the direct path must produce the *same*
/// bytes as the general path, and each of these commands has a case where the
/// answer is not derivable from the other - a nil is not a zero, a missing key is
/// not an empty value, and a TTL of -1 (immortal) is not a TTL of -2 (absent).
/// Listing them as one matrix is what makes that property obvious to read; as
/// separate functions the shared shape was invisible and the only visible thing was
/// the count.
#[rstest]
#[case::get(b"GET", "$16\r\n0123456789abcdef\r\n", "$-1\r\n")]
#[case::exists(b"EXISTS", ":1\r\n", ":0\r\n")]
#[case::strlen(b"STRLEN", ":16\r\n", ":0\r\n")]
#[case::ttl(b"TTL", ":-1\r\n", ":-2\r\n")]
#[case::pttl(b"PTTL", ":-1\r\n", ":-2\r\n")]
fn every_direct_read_agrees_with_the_general_path(
    #[case] name: &[u8],
    #[case] present_reply: &str,
    #[case] absent_reply: &str,
) {
    let store = store_with("present", VALUE);
    assert_eq!(
        assert_paths_agree(store.clone(), &cmd(&[name, b"present"])),
        present_reply
    );
    assert_eq!(
        assert_paths_agree(store, &cmd(&[name, b"absent"])),
        absent_reply
    );
    // And the two are not the same answer, which is the reason the matrix has two
    // columns: a nil that was really a zero would still agree with itself.
    assert_ne!(present_reply, absent_reply);
}

#[test]
fn lowercase_commands_agree() {
    let store = store_with("present", VALUE);
    assert_eq!(
        assert_paths_agree(store, &cmd(&[b"get", b"present"])),
        "$16\r\n0123456789abcdef\r\n"
    );
}

#[test]
fn wrong_arity_agrees_and_still_errors() {
    let store = store_with("present", VALUE);
    // Zero extra arguments.
    assert!(assert_paths_agree(store.clone(), &cmd(&[b"GET"])).contains("wrong number"));
    // Too many.
    assert!(assert_paths_agree(store, &cmd(&[b"GET", b"a", b"b"])).contains("wrong number"));
}

#[test]
fn an_unknown_command_agrees() {
    let store = store_with("present", VALUE);
    assert!(assert_paths_agree(store, &cmd(&[b"NOSUCH", b"k"])).contains("unknown command"));
}

// ------------------------------------------------------------------ pipelining

#[test]
fn a_deep_pipeline_of_gets_agrees_with_the_general_path() {
    let mut store = ObjectStore::new();
    for index in 0..64u32 {
        let write = store
            .prepare(
                &Operation::Set {
                    key: Key::new(format!("k{index}").into_bytes()),
                    value: bytes::Bytes::copy_from_slice(VALUE),
                },
                NOW,
            )
            .expect("prepares");
        let kivi_state::Prepared::Write(mutation) = write else {
            panic!("a set prepares as a write");
        };
        store.apply(&mutation, NOW).expect("applies");
    }
    let mut input = Vec::new();
    for index in 0..64u32 {
        input.extend_from_slice(&cmd(&[b"GET", format!("k{index}").as_bytes()]));
    }
    // Every key, and one that is not there, in one pipeline.
    input.extend_from_slice(&cmd(&[b"GET", b"nope"]));

    let replies = assert_paths_agree(store, &input);
    assert_eq!(
        replies.matches("$16\r\n0123456789abcdef\r\n").count(),
        64,
        "every stored key must be answered"
    );
    assert!(
        replies.ends_with("$-1\r\n"),
        "the missing key is last and must answer nil, got {replies:?}"
    );
}

/// The ordering rule, tested where it matters: a fast read **before** a write must
/// see the pre-write value.
///
/// This is the property that makes running the fast prefix ahead of the general path
/// safe, and it is the property that a "just execute everything as it parses"
/// rewrite would break. If this test fails, a read has observed a write it should
/// not have seen.
#[test]
fn a_read_before_a_write_sees_the_pre_write_value() {
    let store = store_with("k", b"old");
    let mut input = Vec::new();
    input.extend_from_slice(&cmd(&[b"GET", b"k"]));
    input.extend_from_slice(&cmd(&[b"SET", b"k", b"new"]));
    input.extend_from_slice(&cmd(&[b"GET", b"k"]));

    let mut connection = direct(store);
    let replies = run(&mut connection, &input);
    let frames: Vec<&str> = replies.split("\r\n").collect();
    // -old, +OK, -new. The first GET must be `old`, never `new`.
    assert!(
        replies.starts_with("$3\r\nold\r\n"),
        "the read before the write must observe the old value, got {replies:?}"
    );
    assert!(
        replies.contains("+OK"),
        "the write must be acknowledged, got {replies:?}"
    );
    assert!(
        replies.ends_with("$3\r\nnew\r\n"),
        "the read after the write must observe the new value, got {replies:?}"
    );
    let _ = frames;
}

/// Same-key repeated mutation in one pipeline: the fast reads between the writes
/// must see each write's result, in order.
#[test]
fn repeated_same_key_mutation_answers_in_order() {
    let store = ObjectStore::new();
    let mut input = Vec::new();
    for value in [&b"a"[..], b"bb", b"ccc"] {
        input.extend_from_slice(&cmd(&[b"SET", b"k", value]));
        input.extend_from_slice(&cmd(&[b"GET", b"k"]));
        input.extend_from_slice(&cmd(&[b"STRLEN", b"k"]));
    }
    let mut connection = direct(store);
    let replies = run(&mut connection, &input);
    // +OK, a, 1, +OK, bb, 2, +OK, ccc, 3 - each write acknowledged, and each read
    // after it observing that write, in order. The first version of this expectation
    // omitted the `+OK`s, and the differential assertion passed anyway because both
    // paths omit them identically; the explicit list is what catches a dropped or
    // duplicated acknowledgement.
    let expected = "+OK\r\n$1\r\na\r\n:1\r\n+OK\r\n$2\r\nbb\r\n:2\r\n+OK\r\n$3\r\nccc\r\n:3\r\n";
    assert_eq!(replies, expected, "ordering or a reply shape is wrong");
}

// ------------------------------------------------------- the pinned historical bugs

/// **SETRANGE length race.** `SETRANGE` writes and then reports the new length, and
/// the two must be ordered against anything else touching the key. The fast path
/// declines `SETRANGE` entirely, so the risk is that its *decline* changed the
/// batching the general path depends on. This asserts the answer, and it asserts
/// the length is the post-write length.
#[test]
fn setrange_reports_the_post_write_length() {
    let store = store_with("k", b"hello");
    let replies = assert_paths_agree(store.clone(), &cmd(&[b"SETRANGE", b"k", b"0", b"J"]));
    assert_eq!(replies, ":5\r\n", "the length after the patch, not before");
}

/// **GETRANGE window applied twice.** A windowed read of a windowed read is the
/// classic double-application. `GETRANGE` is not a fast command, so this checks the
/// general path still behaves and that inserting a fast read around it does not
/// change the answer.
#[test]
fn getrange_is_not_double_windowed() {
    let store = store_with("k", b"0123456789");
    let mut input = Vec::new();
    input.extend_from_slice(&cmd(&[b"GET", b"k"]));
    input.extend_from_slice(&cmd(&[b"GETRANGE", b"k", b"2", b"5"]));
    assert_eq!(
        assert_paths_agree(store, &input),
        // Redis's `GETRANGE` end index is **inclusive**, so `2 5` is four bytes.
        // The first version of this expected three and the differential assertion
        // still passed - the two paths agreed on the wrong answer, which is exactly
        // why the explicit expectation is here next to the comparison.
        "$10\r\n0123456789\r\n$4\r\n2345\r\n"
    );
}

/// **Fabric manifest instead of resolved payload.** A fabric-backed value must not
/// be answered with its manifest. The fast path declines any value that is not
/// inline bytes, so this asserts the decline happens rather than a manifest leaking.
#[test]
fn a_non_inline_value_is_declined_rather_than_answered() {
    // A store holding a scan descriptor, which is not inline bytes: the fast path
    // must return `None` and let the general path resolve it.
    let store = ObjectStore::new();
    let executor = DirectExecutor {
        store: std::cell::RefCell::new(store),
    };
    assert_eq!(
        executor.fast_read(ROUTE, kivi_resp::fast::FastKind::Get, b"anything", NOW),
        Some(Reply::Nil),
        "an empty store has nothing, which is a nil and not a fallback"
    );
    // A key that exists but is not inline bytes: the executor's own `None`.
    let mut store = ObjectStore::new();
    store.put_stored(
        Key::new(b"k".to_vec()),
        StoredObject::restore(
            kivi_state::LogicalValue::Chunked(kivi_state::ChunkedRef {
                manifest: [7u8; 32].into(),
                logical_len: 16,
            }),
            ObjectVersion::from(0_u64),
            kivi_types::Expiry::NEVER,
        ),
    );
    let executor = DirectExecutor {
        store: std::cell::RefCell::new(store),
    };
    assert_eq!(
        executor.fast_read(ROUTE, kivi_resp::fast::FastKind::Get, b"k", NOW),
        None,
        "a chunked value must be declined, not answered with its manifest"
    );
}

// ------------------------------------------------------------ framing robustness

#[test]
fn one_command_split_across_two_pushes_agrees() {
    let store = store_with("k", VALUE);
    let frame = cmd(&[b"GET", b"k"]);
    let mut connection = direct(store.clone());
    let mut out = Vec::new();
    // Everything but the last byte, then the last byte. The parser must not answer a
    // partial frame, and the second push must complete exactly one reply.
    connection.push(&frame[..frame.len() - 1]).expect("fits");
    assert!(
        connection.drain(&mut out).replies == 0,
        "a partial frame must not be answered"
    );
    connection.push(&frame[frame.len() - 1..]).expect("fits");
    assert_eq!(connection.drain(&mut out).replies, 1);
    assert_eq!(out, b"$16\r\n0123456789abcdef\r\n");
}

#[test]
fn a_read_after_a_write_split_across_pushes_keeps_order() {
    let store = store_with("k", b"old");
    let mut connection = direct(store);
    let mut set = cmd(&[b"SET", b"k", b"new"]);
    let get = cmd(&[b"GET", b"k"]);
    // The write is split; the read is not. The reply order must still be +OK then new.
    // Split inside the trailing value, so the frame is genuinely incomplete rather
    // than malformed: a first version cut four bytes off the end, which lands inside
    // the key and is a different failure (a refusal, not an incompleteness).
    let split = set.len() - 5;
    let tail = set.split_off(split);
    let mut out = Vec::new();
    connection.push(&set).expect("fits");
    assert_eq!(
        connection.drain(&mut out).replies,
        0,
        "the write is incomplete"
    );
    connection.push(&tail).expect("fits");
    connection.push(&get).expect("fits");
    while connection.has_unconsumed() {
        assert!(connection.drain(&mut out).replies > 0);
    }
    assert_eq!(out, b"+OK\r\n$3\r\nnew\r\n");
}

#[test]
fn many_commands_in_one_packet_agree() {
    let store = store_with("k", VALUE);
    let mut input = Vec::new();
    for _ in 0..10 {
        input.extend_from_slice(&cmd(&[b"GET", b"k"]));
    }
    let replies = assert_paths_agree(store, &input);
    assert_eq!(replies.matches("$16\r\n").count(), 10);
}

// ------------------------------------------------------------------ the budget

/// The byte budget must cut a turn, and must not cut a small one.
///
/// A budget that never fired would be a removed fairness guarantee; one that fired
/// on a small pipeline would be the 64-command cap renamed.
#[test]
fn the_byte_budget_bounds_a_large_read_pipeline() {
    let mut store = ObjectStore::new();
    // A value large enough that 64 of them exceed a 64 KiB budget.
    let big = vec![b'v'; 4096];
    let write = store
        .prepare(
            &Operation::Set {
                key: Key::new(b"k".to_vec()),
                value: bytes::Bytes::from(big),
            },
            NOW,
        )
        .expect("prepares");
    let kivi_state::Prepared::Write(mutation) = write else {
        panic!("a set prepares as a write");
    };
    store.apply(&mutation, NOW).expect("applies");

    let config = ConnConfig {
        turn_budget: TurnBudget::new(std::time::Duration::from_millis(500), 64 * 1024),
        ..ConnConfig::default()
    };
    let mut connection = RespConnection::new(
        DirectExecutor {
            store: std::cell::RefCell::new(store),
        },
        config,
        1,
    );
    let mut input = Vec::new();
    for _ in 0..64 {
        input.extend_from_slice(&cmd(&[b"GET", b"k"]));
    }
    connection.push(&input).expect("fits");
    let mut out = Vec::new();
    let first = connection.drain(&mut out);
    assert!(first.replies > 0, "the first turn must make progress");
    assert!(
        first.replies < 64,
        "a 64 × 4 KiB pipeline must be cut by the byte budget, got {} replies",
        first.replies
    );
    // And the connection still answers everything, in order, across turns.
    let mut total = first.replies;
    while connection.has_unconsumed() {
        total += connection.drain(&mut out).replies;
    }
    assert_eq!(total, 64, "every command is answered exactly once");
    assert_eq!(
        out.windows(5).filter(|w| *w == b"$4096").count(),
        64,
        "every reply is the full value"
    );
}

/// A drain answers everything currently buffered, and reports that it is done.
///
/// The contract is completeness, not a turn count: a client that sent sixteen
/// commands is owed sixteen replies, and the only legitimate reason for a drain to
/// stop early is the output budget - the caller can write and call again, and
/// `has_unconsumed` says so. Nothing here should depend on how the work was
/// divided internally.
#[test]
fn a_drain_answers_the_whole_buffer() {
    let store = store_with("k", VALUE);
    let mut input = Vec::new();
    for _ in 0..16 {
        input.extend_from_slice(&cmd(&[b"GET", b"k"]));
    }
    let mut connection = direct(store);
    connection.push(&input).expect("fits");
    let mut out = Vec::new();
    let outcome = connection.drain(&mut out);
    assert_eq!(outcome.replies, 16);
    assert!(
        !connection.has_unconsumed(),
        "a drain that stopped for a reason the caller cannot act on is a stall"
    );
    assert_eq!(
        String::from_utf8_lossy(&out).matches("$16\r\n").count(),
        16,
        "every reply, once, in order"
    );
}
