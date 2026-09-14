//! Exact-profile conformance: Kivi RESP vs reference Redis.
//!
//! Runs [`kivi_lab::conformance::exact_vectors`] against a spawned Kivi
//! (ephemeral + RESP) and the reference Redis, then diffs normalized
//! observables. Only `Exact` commands compare — deviations and unsupported
//! forms have their own tests and never fail this harness.
//!
//! External Redis is opt-in: without `KIVI_LAB_REDIS_URL` these tests skip
//! with a notice (ordinary `cargo test` stays self-contained); a set but
//! unreachable URL fails loudly. This file additionally needs the lab
//! `redis-compat` feature (a RESP-capable Kivi binary); without it the
//! file compiles to nothing.

#![cfg(feature = "redis-compat")]

use kivi_lab::conformance::{
    cleanup_prefix, diff_vectors, redis_config, redis_opted_in, redis_url, run_scope, run_vectors,
    setup_edge_keys,
};
use kivi_lab::process::Server;
use kivi_lab::resp_client::RespClient;

/// Reference connection, or a loud failure when the opted-in URL is down.
fn reference() -> RespClient {
    // `redis_url` always yields something; without the env var this test
    // would have skipped above, so reaching here means "required".
    RespClient::connect(&redis_url(None)).expect("reference Redis reachable")
}

/// Exact-profile matrix: every vector must observe identically on both
/// sides (TTL-family replies compare shape, never digits).
#[test]
fn exact_profile_conformance() {
    if !redis_opted_in() {
        eprintln!("SKIP exact conformance: KIVI_LAB_REDIS_URL unset");
        return;
    }
    let server = Server::spawn_ephemeral_resp().expect("kivi with RESP spawns");
    let kivi_addr = server.resp_endpoint().expect("resp endpoint");
    let mut kivi = RespClient::connect(&kivi_addr).expect("kivi reachable");
    let mut reference = reference();
    let config = redis_config(&mut reference).expect("reference config readable");
    eprintln!(
        "reference version={} appendonly={} appendfsync={} save=[{}] policy={}",
        config.version, config.appendonly, config.appendfsync, config.save, config.maxmemory_policy
    );
    eprintln!("persistence section:\n{}", config.persistence_section);
    let prefix = run_scope();
    setup_edge_keys(&mut kivi, &prefix).expect("kivi setup");
    setup_edge_keys(&mut reference, &prefix).expect("reference setup");
    let left = run_vectors(&mut kivi, &prefix).expect("kivi vectors");
    let right = run_vectors(&mut reference, &prefix).expect("reference vectors");
    let mismatches = diff_vectors(&left, &right);
    for (argv, kivi_outcome, redis_outcome) in &mismatches {
        eprintln!("MISMATCH {argv:?}: kivi={kivi_outcome:?} redis={redis_outcome:?}");
    }
    let cleaned_kivi = cleanup_prefix(&mut kivi, &prefix).expect("kivi cleanup");
    let cleaned_ref = cleanup_prefix(&mut reference, &prefix).expect("reference cleanup");
    eprintln!("cleaned kivi={cleaned_kivi} redis={cleaned_ref} prefix={prefix}");
    assert!(
        mismatches.is_empty(),
        "{} exact-profile mismatches (see MISMATCH lines above)",
        mismatches.len()
    );
}
