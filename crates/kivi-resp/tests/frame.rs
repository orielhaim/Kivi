//! The zero-copy RESP request parser.
//!
//! The properties that matter are not "does it parse a happy path" but the
//! two the network actually exercises: a frame split across reads must wait
//! rather than be refused, and bytes that are not RESP at all must not be
//! resynchronised by guessing.

use kivi_resp::MAX_ARGS;
use kivi_resp::frame::{Limits, Parsed, fold_command_name, parse_command};

fn limits() -> Limits {
    Limits {
        max_bulk_bytes: 1024,
        max_array_elements: 64,
    }
}

fn frame(parts: &[&[u8]]) -> Vec<u8> {
    let mut out = format!("*{}\r\n", parts.len()).into_bytes();
    for part in parts {
        out.extend_from_slice(format!("${}\r\n", part.len()).as_bytes());
        out.extend_from_slice(part);
        out.extend_from_slice(b"\r\n");
    }
    out
}

#[test]
fn a_command_borrows_its_arguments_in_place() {
    let bytes = frame(&[b"GET", b"key"]);
    let (parsed, used) = parse_command(&bytes, &limits());
    assert_eq!(used, bytes.len());
    let Parsed::Command(command) = parsed else {
        panic!("expected a command, got {parsed:?}");
    };
    assert_eq!(command.name(), b"GET");
    assert_eq!(command.args(), &[b"key".as_slice()]);
    // The arguments point into the caller's buffer, not into fresh storage.
    // "*2\r\n" is 4 bytes and "$3\r\n" is 4, so the key starts at 17.
    let base = bytes.as_ptr() as usize;
    assert_eq!(command.args()[0].as_ptr() as usize - base, 17);
    assert_eq!(command.name().as_ptr() as usize - base, 8);
}

#[test]
fn binary_safe_arguments_survive_verbatim() {
    let payload = [0x00u8, 0xFF, 0x0D, 0x0A, b'*', b'$'];
    let mut bytes = frame(&[b"SET", b"k", &payload]);
    let (parsed, used) = parse_command(&bytes, &limits());
    assert_eq!(used, bytes.len());
    let Parsed::Command(command) = parsed else {
        panic!("expected a command");
    };
    assert_eq!(command.args()[1], payload.as_slice());
    // A key that looks like a frame is still just bytes.
    bytes.clear();
    bytes.extend_from_slice(&frame(&[b"GET", b"*1\r\n$4\r\nPING\r\n"]));
    let (parsed, _) = parse_command(&bytes, &limits());
    let Parsed::Command(command) = parsed else {
        panic!("expected a command");
    };
    assert_eq!(command.args()[0], b"*1\r\n$4\r\nPING\r\n");
}

#[test]
fn simple_string_elements_are_accepted() {
    let bytes = b"*2\r\n+GET\r\n$3\r\nfoo\r\n";
    let (parsed, used) = parse_command(bytes, &limits());
    assert_eq!(used, bytes.len());
    let Parsed::Command(command) = parsed else {
        panic!("expected a command, got {parsed:?}");
    };
    assert_eq!(command.name(), b"GET");
    assert_eq!(command.args(), &[b"foo".as_slice()]);
}

#[test]
fn every_truncation_waits_instead_of_refusing() {
    let full = frame(&[b"SET", b"key", b"value"]);
    for cut in 1..full.len() {
        let (parsed, used) = parse_command(&full[..cut], &limits());
        assert!(
            matches!(parsed, Parsed::Incomplete),
            "cut at {cut} should wait, got {parsed:?}"
        );
        assert_eq!(used, 0, "an incomplete frame must consume nothing");
    }
    let (parsed, used) = parse_command(&full, &limits());
    assert!(matches!(parsed, Parsed::Command(_)));
    assert_eq!(used, full.len());
}

#[test]
fn a_missing_bulk_terminator_waits_rather_than_refusing() {
    // The payload arrived but its trailing CRLF has not.
    let bytes = b"*2\r\n$3\r\nGET\r\n$3\r\nfoo";
    let (parsed, used) = parse_command(bytes, &limits());
    assert!(matches!(parsed, Parsed::Incomplete), "got {parsed:?}");
    assert_eq!(used, 0);
    let mut complete = bytes.to_vec();
    complete.extend_from_slice(b"\r\n");
    let (parsed, used) = parse_command(&complete, &limits());
    assert!(matches!(parsed, Parsed::Command(_)));
    assert_eq!(used, complete.len());
}

#[test]
fn a_wrong_terminator_is_malformed_not_incomplete() {
    let bytes = b"*2\r\n$3\r\nGET\r\n$3\r\nfooXX";
    let (parsed, _) = parse_command(bytes, &limits());
    assert!(matches!(parsed, Parsed::Malformed), "got {parsed:?}");
    assert!(parsed.is_fatal());
}

#[test]
fn a_bare_line_terminator_is_accepted_in_a_length_line() {
    // Length lines tolerate a bare LF; a payload terminator does not, since
    // the payload is opaque bytes and only CRLF is unambiguous.
    let bytes = b"*2\n$3\r\nGET\r\n$3\r\nfoo\r\n";
    let (parsed, used) = parse_command(bytes, &limits());
    assert!(matches!(parsed, Parsed::Command(_)), "got {parsed:?}");
    assert_eq!(used, bytes.len());

    let bytes = b"*2\r\n$3\r\nGET\n$3\r\nfoo\r\n";
    let (parsed, _) = parse_command(bytes, &limits());
    assert!(parsed.is_fatal(), "got {parsed:?}");
}

#[test]
fn non_string_elements_are_refused_without_closing() {
    for bytes in [
        &b"*1\r\n:1\r\n"[..],
        &b"*2\r\n$3\r\nGET\r\n:2\r\n"[..],
        &b"*1\r\n_\r\n"[..],
        &b"*1\r\n*1\r\n$1\r\na\r\n"[..],
    ] {
        let (parsed, _) = parse_command(bytes, &limits());
        assert!(
            matches!(parsed, Parsed::BadArgument),
            "{bytes:?} should be a bad argument, got {parsed:?}"
        );
        assert!(!parsed.is_fatal());
    }
}

#[test]
fn an_empty_array_is_refused_without_closing() {
    let (parsed, _) = parse_command(b"*0\r\n", &limits());
    assert!(matches!(parsed, Parsed::Empty), "got {parsed:?}");
    assert!(!parsed.is_fatal());
}

#[test]
fn a_well_formed_non_command_keeps_the_connection_open() {
    for bytes in [
        &b"+PING\r\n"[..],
        &b"-ERR already an error\r\n"[..],
        &b":42\r\n"[..],
        &b"$3\r\nfoo\r\n"[..],
    ] {
        let (parsed, used) = parse_command(bytes, &limits());
        assert!(
            matches!(parsed, Parsed::NotACommand),
            "{bytes:?} should not be a command, got {parsed:?}"
        );
        assert_eq!(used, bytes.len(), "the frame must be consumed whole");
        assert!(!parsed.is_fatal(), "Kivi answers and stays open here");
    }
}

#[test]
fn bytes_that_are_not_resp_close_the_connection() {
    for bytes in [
        &b"\xff\x00\x01"[..],
        &b"*-1\r\n"[..],
        &b"*x\r\n"[..],
        &b"*99999999999999999999999\r\n"[..],
    ] {
        let (parsed, _) = parse_command(bytes, &limits());
        assert!(parsed.is_fatal(), "{bytes:?} should close, got {parsed:?}");
    }
}

#[test]
fn too_many_arguments_is_refused_but_the_stream_stays_in_sync() {
    let mut parts: Vec<Vec<u8>> = vec![b"SET".to_vec()];
    for index in 0..MAX_ARGS + 4 {
        parts.push(format!("a{index}").into_bytes());
    }
    let refs: Vec<&[u8]> = parts.iter().map(Vec::as_slice).collect();
    let bytes = frame(&refs);
    let (parsed, used) = parse_command(&bytes, &limits());
    assert!(matches!(parsed, Parsed::TooManyArgs), "got {parsed:?}");
    assert!(!parsed.is_fatal());
    assert_eq!(used, bytes.len(), "the refused frame is skipped whole");

    // A pipelined command behind the refused one still runs.
    let mut batch = bytes.clone();
    batch.extend_from_slice(&frame(&[b"PING"]));
    let (parsed, used) = parse_command(&batch, &limits());
    assert!(matches!(parsed, Parsed::TooManyArgs));
    assert_eq!(used, bytes.len());
    let (parsed, _) = parse_command(&batch[used..], &limits());
    assert!(matches!(parsed, Parsed::Command(_)));
}

#[test]
fn a_bulk_past_the_bound_is_fatal() {
    let bounds = Limits {
        max_bulk_bytes: 4,
        max_array_elements: 64,
    };
    let bytes = frame(&[b"GET", b"12345"]);
    let (parsed, _) = parse_command(&bytes, &bounds);
    assert!(matches!(parsed, Parsed::TooLarge), "got {parsed:?}");
    assert!(parsed.is_fatal());
}

#[test]
fn a_hostile_length_line_is_refused_in_bounded_work() {
    // 4096 bytes with no terminator: the parser must give up on the line
    // rather than scan all of it.
    let mut bytes = b"*".to_vec();
    bytes.extend(std::iter::repeat_n(b'9', 4096));
    let (parsed, _) = parse_command(&bytes, &limits());
    assert!(parsed.is_fatal(), "got {parsed:?}");
}

#[test]
fn a_huge_declared_length_does_not_overflow_the_cursor() {
    let bytes = b"*2\r\n$3\r\nGET\r\n$99999999999999999999\r\nx\r\n";
    let (parsed, _) = parse_command(bytes, &limits());
    assert!(parsed.is_fatal(), "got {parsed:?}");
}

#[test]
fn folding_a_command_name_is_case_insensitive_and_bounded() {
    assert_eq!(
        fold_command_name(b"get").map(|f| f[..3].to_vec()),
        Some(b"GET".to_vec())
    );
    assert_eq!(
        fold_command_name(b"GeT").map(|f| f[..3].to_vec()),
        Some(b"GET".to_vec())
    );
    assert!(fold_command_name(b"").is_none());
    assert!(fold_command_name(b"\xff\xfe").is_none());
    // Longer than any command: refused rather than truncated.
    assert!(fold_command_name(&[b'G'; 64]).is_none());
    let longest = b"PEXPIRETIME";
    assert_eq!(
        fold_command_name(longest).map(|f| f[..longest.len()].to_vec()),
        Some(b"PEXPIRETIME".to_vec())
    );
}
