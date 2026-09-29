//! New native operations over real loopback TCP: range reads, logical
//! lengths, and conditional stores (all three wire opcodes, both inline and
//! chunked roots, through the reactor path - never the embedded channel).

use std::sync::{Arc, Mutex};
use std::thread;

use bytes::Bytes;
use kivi_client::{ClientConfig, NativeClient};
use kivi_lab::process::Server;
use kivi_lab::testkit::{NS, far_future, fill, key, run_workload_op, start_ephemeral_client};
use kivi_lab::workload::{Workload, workload_op_for};
use kivi_state::{ExpiryPolicy, SetCondition};

#[test]
fn ranges_slice_and_measure_inline_values() {
    let (engine, client) = start_ephemeral_client();
    let key = key("rk");
    assert_eq!(client.get_range(&key, 0, 4).expect("range"), None);
    assert_eq!(client.bytes_length(&key).expect("length"), None);
    client.set(&key, Bytes::from_static(b"hello")).expect("set");
    assert_eq!(
        client.get_range(&key, 1, 3).expect("range"),
        Some(Bytes::from_static(b"ell"))
    );
    // A window entirely past the end is empty, not an error.
    assert_eq!(
        client.get_range(&key, 99, 4).expect("past end"),
        Some(Bytes::new())
    );
    assert_eq!(client.bytes_length(&key).expect("length"), Some(5));
    engine.shutdown().expect("clean shutdown");
}

#[test]
fn medium_range_patch_can_expire() {
    let (engine, client) = start_ephemeral_client();
    let key = key("medium-expire");
    let value = fill(1024, 9);
    client
        .set(&key, Bytes::copy_from_slice(&value))
        .expect("set");
    client
        .set_range(&key, 0, Bytes::from_static(b"v"))
        .expect("range patch");
    assert_eq!(
        client.get_range(&key, 0, 64).expect("range"),
        Some({
            let mut expected = value[..64].to_vec();
            expected[0] = b'v';
            Bytes::from(expected)
        })
    );
    let expiry = far_future();
    assert!(
        client
            .expire_at(&key, expiry.as_stamp().expect("dated"))
            .expect("expire")
    );
    assert_eq!(client.get_expiry(&key).expect("expiry"), Some(expiry));
    engine.shutdown().expect("clean shutdown");
}

#[test]
fn conditional_stores_evaluate_atomically_with_policies() {
    let (engine, client) = start_ephemeral_client();
    let key = key("ck");
    // Absent: NX applies, and a second NX refuses without a live version.
    assert_eq!(
        client
            .set_conditional(
                &key,
                Bytes::from_static(b"v1"),
                SetCondition::IfAbsent,
                ExpiryPolicy::Clear
            )
            .expect("nx"),
        (true, Some(1))
    );
    assert_eq!(
        client
            .set_conditional(
                &key,
                Bytes::from_static(b"v2"),
                SetCondition::IfAbsent,
                ExpiryPolicy::Clear
            )
            .expect("nx again"),
        (false, None)
    );
    // Present: XX applies, and `Keep` preserves the dated expiry.
    let expiry = far_future();
    assert!(
        client
            .expire_at(&key, expiry.as_stamp().expect("dated"))
            .expect("expire")
    );
    assert_eq!(
        client
            .set_conditional(
                &key,
                Bytes::from_static(b"v3"),
                SetCondition::IfPresent,
                ExpiryPolicy::Keep
            )
            .expect("xx keep"),
        (true, Some(3))
    );
    assert_eq!(client.get_expiry(&key).expect("expiry"), Some(expiry));
    assert_eq!(
        client.get(&key).expect("get"),
        Some(Bytes::from_static(b"v3"))
    );
    engine.shutdown().expect("clean shutdown");
}

#[test]
fn chunked_roots_slice_and_measure_without_full_reads() {
    let (engine, client) = start_ephemeral_client();
    let key = key("big-range");
    // Multi-chunk value (2.5 MiB): stages through the streaming upload.
    let big = fill(2_500_000, 0x57EA);
    let mut source = std::io::Cursor::new(big.clone());
    client
        .put_stream(&key, Some(big.len() as u64), &mut source)
        .expect("upload");
    // Logical length answers from root metadata, without a full read.
    assert_eq!(client.bytes_length(&key).expect("length"), Some(2_500_000));
    let window = client
        .get_range(&key, 1_000_000, 16)
        .expect("range")
        .expect("present");
    assert_eq!(&window[..], &big[1_000_000..1_000_016]);
    engine.shutdown().expect("clean shutdown");
}

/// A chunked root must survive a run of the compat op mix - in particular
/// repeated `set_range` patches over the same staged value.
#[test]
fn repeated_fabric_set_range_keeps_root_readable() {
    let data_dir = tempfile::tempdir().expect("data directory");
    let server = Server::spawn_auto(data_dir.path(), &[]).expect("server starts");
    let client = NativeClient::new(ClientConfig {
        seeds: vec![server.endpoint()],
        namespace: NS,
        ..ClientConfig::default()
    })
    .expect("client builds");
    let keys = vec!["repeated-fabric-range".to_owned()];
    let value = Bytes::from(fill(1024, 9));
    client
        .set(&key(&keys[0]), value.clone())
        .expect("initial set");
    for index in 0..20 {
        let operation = workload_op_for(Workload::Compat, 1024, index, &keys, "unused-counter");
        let result = run_workload_op(&client, &operation, &value);
        assert!(
            result.is_ok(),
            "operation {index} {operation:?} failed: {result:?}"
        );
        // The root stays readable after every op in the mix.
        client
            .get(&key(&keys[0]))
            .unwrap_or_else(|error| panic!("read after {index} {operation:?} failed: {error}"));
    }
    client
        .get(&key(&keys[0]))
        .expect("final get")
        .expect("final value");
    server.kill();
}

/// Four clients × 100 balanced ops against a real server: no op may end in a
/// terminal error.
#[test]
fn concurrent_balanced_operations_have_no_terminal_errors() {
    let data_dir = tempfile::tempdir().expect("data directory");
    let server = Server::spawn_auto(data_dir.path(), &[]).expect("server starts");
    let keys: Vec<String> = (0..256)
        .map(|index| format!("concurrent-balanced:g:k{index}"))
        .collect();
    let value = Bytes::from(fill(1024, 9));
    let seeder = NativeClient::new(ClientConfig {
        seeds: vec![server.endpoint()],
        namespace: NS,
        ..ClientConfig::default()
    })
    .expect("seeder builds");
    for name in &keys {
        seeder.set(&key(name), value.clone()).expect("seed key");
    }
    let errors = Arc::new(Mutex::new(Vec::new()));
    let endpoint = server.endpoint();
    let mut handles = Vec::new();
    for thread in 0..4 {
        let keys = keys.clone();
        let value = value.clone();
        let errors = Arc::clone(&errors);
        let endpoint = endpoint.clone();
        handles.push(thread::spawn(move || {
            let client = NativeClient::new(ClientConfig {
                seeds: vec![endpoint],
                namespace: NS,
                ..ClientConfig::default()
            })
            .expect("client builds");
            for index in 0..100 {
                let operation = workload_op_for(Workload::Balanced, 1024, index, &keys, "unused");
                if let Err(error) = run_workload_op(&client, &operation, &value) {
                    errors.lock().expect("errors lock").push(format!(
                        "thread={thread} index={index} operation={operation:?} error={error}"
                    ));
                }
            }
        }));
    }
    for handle in handles {
        handle.join().expect("worker joins");
    }
    let errors = errors.lock().expect("errors lock").clone();
    server.kill();
    assert!(errors.is_empty(), "{}", errors.join("\n"));
}
