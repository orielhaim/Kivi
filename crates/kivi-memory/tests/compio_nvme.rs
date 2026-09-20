//! The `NVMe` provider on the Compio reactor: explicit asynchronous
//! demotion and promotion using the existing Compio architecture (no
//! second runtime, no hidden blocking page faults).

use kivi_memory::{demote_async, promote_async};

/// Async demote followed by async promote round-trips the bytes with a
/// verified checksum on the caller's Compio runtime.
#[test]
fn compio_demote_promote_roundtrip() {
    let dir = tempfile::tempdir().expect("tempdir");
    let payload = vec![0xA5u8; 8192];
    let (offset, len, _checksum) = compio::runtime::Runtime::new()
        .expect("runtime")
        .block_on(demote_async(dir.path(), &payload))
        .expect("demote");
    assert_eq!(len, payload.len() as u64);
    let seen = compio::runtime::Runtime::new()
        .expect("runtime")
        .block_on(promote_async(dir.path(), offset, len));
    let seen = seen.expect("promote serves");
    assert_eq!(seen, payload);
}

/// Reading a torn record through the async path fails loudly instead of
/// serving corrupt bytes.
#[test]
fn compio_promote_detects_corruption() {
    let dir = tempfile::tempdir().expect("tempdir");
    let payload = b"precious bytes".to_vec();
    let (offset, len, _) = compio::runtime::Runtime::new()
        .expect("runtime")
        .block_on(demote_async(dir.path(), &payload))
        .expect("demote");
    assert_eq!(len, payload.len() as u64);
    // Corrupt one payload byte in place.
    let data_path = dir.path().join("materializations.dat");
    let mut raw = std::fs::read(&data_path).expect("read");
    let last = raw.len() - 1;
    raw[last] ^= 0xFF;
    std::fs::write(&data_path, raw).expect("clobber");
    let error = compio::runtime::Runtime::new()
        .expect("runtime")
        .block_on(promote_async(dir.path(), offset, len))
        .expect_err("corruption detected");
    assert!(matches!(
        error,
        kivi_memory::MemoryError::CorruptRepresentation { .. }
    ));
}
