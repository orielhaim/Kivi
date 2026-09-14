//! Bounded randomized differential test: Kivi RESP vs reference Redis.
//!
//! A seeded splitmix stream generates exact-profile command sequences; the
//! same sequence runs against isolated Kivi/Redis key prefixes, comparing
//! every observable response plus periodic full known-key state. TTL reads
//! compare shape (positive/immortal/missing), never digits.
//!
//! Deterministic seed, bounded length, CI-reasonable runtime. Gating matches
//! `redis_conformance.rs`: opt in with `KIVI_LAB_REDIS_URL`, plus the lab
//! `redis-compat` feature for the Kivi side.

#![cfg(feature = "redis-compat")]

use kivi_lab::conformance::{
    Observable, cleanup_prefix, redis_opted_in, redis_url, run_scope, setup_edge_keys,
};
use kivi_lab::process::Server;
use kivi_lab::resp_client::{Reply, RespClient};

/// Small deterministic RNG (splitmix64): no extra dependencies, fixed
/// stream per seed, trivially reproducible failures.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn below(&mut self, bound: u64) -> u64 {
        if bound == 0 { 0 } else { self.next() % bound }
    }
}

/// Fixed seed: the same run everywhere, every time.
const SEED: u64 = 0xD1FF_1AB9;
/// Sequence length: bounded for CI (a larger manual stress is a different job).
const SEQUENCE: usize = 300;
/// Keyspace size (small enough for periodic full-state checks).
const KEYS: usize = 8;

/// Generates one argv (unprefixed key names; the caller prefixes position 1).
fn gen_op(rng: &mut Rng) -> Vec<Vec<u8>> {
    let key = format!("dk{}", rng.below(KEYS as u64));
    let value = match rng.below(4) {
        0 => Vec::new(),
        1 => b"v".to_vec(),
        2 => vec![
            0x00,
            0xFF,
            0x10,
            u8::try_from(rng.below(256)).unwrap_or(u8::MAX),
        ],
        _ => vec![b'x'; 64],
    };
    let owned = |parts: &[&[u8]]| parts.iter().map(|part| part.to_vec()).collect::<Vec<_>>();
    match rng.below(100) {
        0..=29 => {
            let mut argv = owned(&[b"SET", key.as_bytes()]);
            argv.push(value);
            argv
        }
        30..=49 => owned(&[b"GET", key.as_bytes()]),
        50..=57 => owned(&[b"DEL", key.as_bytes()]),
        58..=64 => owned(&[b"EXISTS", key.as_bytes()]),
        65..=72 => {
            let secs = (rng.below(300) + 1).to_string();
            owned(&[b"EXPIRE", key.as_bytes(), secs.as_bytes()])
        }
        73..=80 => owned(&[b"TTL", key.as_bytes()]),
        81..=84 => owned(&[b"PERSIST", key.as_bytes()]),
        85..=91 => {
            let offset = rng.below(12).to_string();
            let mut argv = owned(&[b"SETRANGE", key.as_bytes(), offset.as_bytes()]);
            argv.push(value);
            argv
        }
        92..=96 => {
            // GETRANGE windows: positive, negative, and empty-window mixes.
            let (start, end) = match rng.below(4) {
                0 => ("0".to_owned(), "5".to_owned()),
                1 => ("-4".to_owned(), "-1".to_owned()),
                2 => ("0".to_owned(), "-1".to_owned()),
                _ => ("3".to_owned(), "1".to_owned()),
            };
            owned(&[
                b"GETRANGE",
                key.as_bytes(),
                start.as_bytes(),
                end.as_bytes(),
            ])
        }
        _ => owned(&[b"STRLEN", key.as_bytes()]),
    }
}

/// Prefixes the key position (argv[1]) for isolation.
fn render(op: &[Vec<u8>], prefix: &str) -> Vec<Vec<u8>> {
    let mut rendered = op.to_vec();
    if let Some(key) = rendered.get_mut(1) {
        let mut prefixed = prefix.as_bytes().to_vec();
        prefixed.extend_from_slice(key);
        *key = prefixed;
    }
    rendered
}

fn round_trip(client: &mut RespClient, op: &[Vec<u8>]) -> Reply {
    let argv: Vec<&[u8]> = op.iter().map(Vec::as_slice).collect();
    client.round_trip(&argv).expect("round trip")
}

/// Compares two observables; TTL-shaped commands compare sign class only.
fn same(command: &[u8], left: &Observable, right: &Observable) -> bool {
    if command == b"TTL" {
        return sign_class(left) == sign_class(right);
    }
    left == right
}

fn sign_class(observed: &Observable) -> &'static str {
    match observed {
        Observable::Integer(value) if *value > 0 => "positive",
        Observable::Integer(value) if *value == -1 => "immortal",
        Observable::Integer(_) => "missing",
        _ => "other",
    }
}

#[test]
fn randomized_differential_matches_reference() {
    if !redis_opted_in() {
        eprintln!("SKIP differential: KIVI_LAB_REDIS_URL unset");
        return;
    }
    let server = Server::spawn_ephemeral_resp().expect("kivi with RESP spawns");
    let kivi_addr = server.resp_endpoint().expect("resp endpoint");
    let mut kivi = RespClient::connect(&kivi_addr).expect("kivi reachable");
    let mut reference = RespClient::connect(&redis_url(None)).expect("reference reachable");
    let prefix = run_scope();
    setup_edge_keys(&mut kivi, &prefix).expect("kivi setup");
    setup_edge_keys(&mut reference, &prefix).expect("reference setup");

    let mut rng = Rng(SEED);
    let mut mismatches = 0usize;
    for step in 0..SEQUENCE {
        let op = render(&gen_op(&mut rng), &prefix);
        let left = Observable::of(&round_trip(&mut kivi, &op));
        let right = Observable::of(&round_trip(&mut reference, &op));
        let command = op.first().map_or(&b""[..], Vec::as_slice);
        if !same(command, &left, &right) {
            mismatches += 1;
            eprintln!(
                "MISMATCH step {step} {:?} argv={:?}: kivi={left:?} redis={right:?}",
                String::from_utf8_lossy(command),
                op.iter()
                    .map(|arg| String::from_utf8_lossy(arg).into_owned())
                    .collect::<Vec<_>>(),
            );
        }
        // Periodic full known-key state check (GET over every key).
        if step % 50 == 49 {
            for key in 0..KEYS {
                let get = render(&[b"GET".to_vec(), format!("dk{key}").into_bytes()], &prefix);
                let left = Observable::of(&round_trip(&mut kivi, &get));
                let right = Observable::of(&round_trip(&mut reference, &get));
                if left != right {
                    mismatches += 1;
                    eprintln!("STATE MISMATCH step {step} dk{key}: kivi={left:?} redis={right:?}");
                }
            }
        }
    }
    let cleaned_kivi = cleanup_prefix(&mut kivi, &prefix).expect("kivi cleanup");
    let cleaned_ref = cleanup_prefix(&mut reference, &prefix).expect("reference cleanup");
    eprintln!(
        "differential steps={SEQUENCE} mismatches={mismatches} cleaned kivi={cleaned_kivi} redis={cleaned_ref}"
    );
    assert_eq!(
        mismatches, 0,
        "differential mismatches (see MISMATCH lines)"
    );
}
