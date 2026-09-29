//! Remote direct memory access, as a capability and not as a transport.
//!
//! ## What was evaluated, and why it is not a dependency
//!
//! The live option in Rust is `sideway`, which wraps `libibverbs` and
//! `librdmacm`. It is the maintained successor to the abandoned `rdma` crate;
//! `rust-ibverbs` does not exist. It was rejected as a default dependency for
//! four independent reasons, each of which would be sufficient on its own:
//!
//! 1. **Licence.** `sideway` is MPL-2.0, file-level copyleft. Depending on it
//!    puts Kivi's AGPL sources under a second copyleft with different terms,
//!    and any file touched by the binding would need review.
//! 2. **Build.** It needs `cmake` and `clang` for `bindgen`, plus
//!    `libibverbs1`, `librdmacm1` and a provider at runtime. A default Kivi
//!    binary must stay straightforward to build, and a system without
//!    `librdmacm` installed gets an RDMA stub that returns `EOPNOTSUPP` for
//!    every call rather than a clean absence.
//! 3. **Platform.** Linux only, which is fine for a feature but not for a
//!    default.
//! 4. **Maturity.** 0.x, and the FFI layer is effectively undocumented, so
//!    reviewing it means reading C headers.
//!
//! ## What RDMA would buy, and only that
//!
//! Lower latency and higher bandwidth for moving bytes between nodes, for three
//! jobs: redundancy fragment transfer, repair and reconstruction transport, and
//! a future remote Memory Fabric provider. Not the peer transport in general:
//! Kivi's peer transport carries consensus and control traffic whose cost is
//! dominated by round trips, not by bandwidth.
//!
//! ## The invariant
//!
//! RDMA is a transport optimisation. It must not bypass identity verification,
//! generation fencing, asset verification, redundancy publication, or
//! consistency semantics. Bytes that arrive over an RDMA queue are not trusted
//! because they arrived quickly: the receiver still re-derives the asset id and
//! still checks the generation. A remote write therefore costs the same
//! verification as a local one, and the only thing RDMA changes is how many
//! round trips that verification took.
//!
//! Kivi therefore treats RDMA as a capability: absent means the normal
//! transport, and the fallback is not a slower path but *the* path.

/// RDMA ports present on this machine, as `device/port` names.
///
/// Empty means the machine has none, which is the overwhelming majority of
/// deployments and is not a degraded mode: Kivi's peer transport is the
/// normal transport and always works. Presence is all that is recorded -
/// nothing in Kivi reads a port, so enumerating one any deeper would only
/// invite a caller to depend on a field Kivi has no contract for.
#[must_use]
pub fn discover_rdma() -> Vec<String> {
    #[cfg(any(target_os = "linux", target_os = "android"))]
    {
        let Ok(entries) = std::fs::read_dir("/sys/class/infiniband") else {
            return Vec::new();
        };
        let mut out: Vec<String> = entries
            .filter_map(Result::ok)
            .flat_map(|entry| {
                let name = entry.file_name().to_string_lossy().into_owned();
                let Ok(ports) = std::fs::read_dir(entry.path().join("ports")) else {
                    return Vec::new();
                };
                ports
                    .filter_map(Result::ok)
                    .filter_map(|port| port.file_name().to_str()?.parse::<u8>().ok())
                    .map(move |port| format!("{name}/p{port}"))
                    .collect::<Vec<_>>()
            })
            .collect();
        out.sort_unstable();
        out
    }
    #[cfg(not(any(target_os = "linux", target_os = "android")))]
    {
        // There is no RDMA transport on Windows or macOS. A design that needs
        // one would need a different mechanism entirely, and pretending
        // otherwise would produce a capability report that lies.
        Vec::new()
    }
}
