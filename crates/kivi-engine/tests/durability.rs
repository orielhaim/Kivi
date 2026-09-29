//! Durable engine tests: embedded write → shutdown → reopen → read back.
//!
//! No processes or sockets here - these prove the engine/durability
//! contract (WAL persist, recovery replay, identity continuity) through
//! the public `LocalEngine` API. Real crash/kill coverage lives in
//! `kivi-lab/tests/durability.rs`.

mod harness;

use harness::{ROOT, durable_config, root_snapshot};
use kivi_engine::{DurabilityMode, EngineConfig, LocalEngine};
use kivi_state::Key;
use kivi_types::NodeIncarnation;

fn durable(dir: &std::path::Path) -> EngineConfig {
    durable_config(
        dir,
        kivi_engine::ChunkFabricConfig::default(),
        kivi_engine::FabricConfig::default(),
    )
}

fn start(dir: &std::path::Path) -> (LocalEngine, NodeIncarnation) {
    let config = durable(dir);
    let incarnation = match &config.durability {
        DurabilityMode::Durable(cfg) => cfg.incarnation,
        DurabilityMode::Ephemeral => unreachable!("durable config is durable"),
    };
    (
        LocalEngine::start(config).expect("durable engine starts"),
        incarnation,
    )
}

#[test]
fn embedded_write_restart_read_back_exact() {
    let scratch = tempfile::tempdir().expect("scratch");
    let (engine, first_incarnation) = start(scratch.path());
    let client = engine.client();
    client
        .set(&Key::from("a"), bytes::Bytes::from_static(b"1"))
        .expect("set");
    assert_eq!(client.counter_add(&Key::from("n"), 41).expect("add"), 41);
    engine.shutdown().expect("clean shutdown");
    // Reopen the same directory: acknowledged state must come back, the
    // incarnation must advance, and new writes must continue the log.
    let (engine, second_incarnation) = start(scratch.path());
    assert!(second_incarnation.supersedes(first_incarnation));
    let client = engine.client();
    assert_eq!(
        client.get(&Key::from("a")).expect("get"),
        Some(bytes::Bytes::from_static(b"1"))
    );
    assert_eq!(client.counter_get(&Key::from("n")).expect("read"), Some(41));
    assert_eq!(client.counter_add(&Key::from("n"), 1).expect("add"), 42);
    let durability = engine
        .admin_handle()
        .durability()
        .expect("durable admin info")
        .clone();
    assert_eq!(durability.recovery.records_replayed, 2);
    engine.shutdown().expect("clean shutdown");
}

#[test]
fn recovery_rejects_forgotten_tablet() {
    let scratch = tempfile::tempdir().expect("scratch");
    let (engine, _) = start(scratch.path());
    engine
        .client()
        .set(&Key::from("a"), bytes::Bytes::from_static(b"1"))
        .expect("set");
    engine.shutdown().expect("clean shutdown");
    // Reopen with tablet 1 sealed (Active->Fenced, unwritable): its WAL
    // history has nowhere to replay, so startup must fail rather than
    // discard it.
    let sealed = root_snapshot().seal(ROOT).expect("seal");
    let mut config = durable(scratch.path());
    config.directory = sealed;
    let error = LocalEngine::start(config).expect_err("forgotten tablet fails startup");
    let message = format!("{error:?}");
    assert!(
        message.contains("UnknownTablet") || message.contains("unknown tablet"),
        "clear error, got {message}"
    );
}
