//! Architectural dependency-direction checks: the RESP edge must never leak
//! into the foundations.
//!
//! ```text
//! kivi-resp
//!     ↓
//! Kivi semantic APIs
//! ```
//!
//! Never the reverse: `kivi-state`, `kivi-codec`, `kivi-checkpoint`,
//! `kivi-durability`, and `kivi-chunk` must have no dependency on
//! `redis-protocol` or `kivi-resp` in any feature configuration. These tests
//! shell out to `cargo tree` (available wherever the test suite runs) and
//! fail loudly on any leak.

use std::process::Command;

/// Foundation crates that must never depend on the RESP edge.
const FOUNDATIONS: &[&str] = &[
    "kivi-state",
    "kivi-codec",
    "kivi-checkpoint",
    "kivi-durability",
    "kivi-chunk",
];

/// Asserts `package`'s normal dependency closure contains neither
/// `redis-protocol` nor `kivi-resp`.
fn assert_no_resp_leak(package: &str) {
    let output = Command::new("cargo")
        .arg("tree")
        .arg("-p")
        .arg(package)
        .arg("-e")
        .arg("normal")
        .arg("--prefix")
        .arg("none")
        .output()
        .expect("cargo tree runs");
    assert!(
        output.status.success(),
        "cargo tree -p {package} failed: {output:?}"
    );
    let tree = String::from_utf8_lossy(&output.stdout);
    for line in tree.lines() {
        let name = line.split_whitespace().next().unwrap_or("");
        assert!(
            name != "redis-protocol" && name != "kivi-resp",
            "architectural leak: {package} depends on {name}"
        );
    }
}

#[test]
fn foundations_have_no_resp_dependency() {
    for package in FOUNDATIONS {
        assert_no_resp_leak(package);
    }
}
