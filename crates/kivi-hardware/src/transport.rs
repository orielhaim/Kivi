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

/// One RDMA device the platform offers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RdmaDevice {
    /// Kernel name, for example `mlx5_0`.
    pub name: String,
    /// Port the device offers, for example `1`.
    pub port: u8,
    /// Memory node the device is closest to, when the platform says.
    pub numa_node: Option<crate::topology::NumaNodeId>,
    /// Link layer the port uses.
    pub link_layer: String,
}

/// RDMA devices present on this machine.
///
/// Empty means the machine has none, which is the overwhelming majority of
/// deployments and is not a degraded mode: Kivi's peer transport is the
/// normal transport and always works.
#[must_use]
pub fn discover_rdma() -> Vec<String> {
    discover()
        .iter()
        .map(|device| format!("{}/p{}", device.name, device.port))
        .collect()
}

/// RDMA devices present, with their detail.
#[must_use]
pub fn discover() -> Vec<RdmaDevice> {
    #[cfg(any(target_os = "linux", target_os = "android"))]
    {
        read_infiniband()
    }
    #[cfg(not(any(target_os = "linux", target_os = "android")))]
    {
        // There is no RDMA transport on Windows or macOS. A design that needs
        // one would need a different mechanism entirely, and pretending
        // otherwise would produce a capability report that lies.
        Vec::new()
    }
}

#[cfg(any(target_os = "linux", target_os = "android"))]
fn read_infiniband() -> Vec<RdmaDevice> {
    let mut devices: Vec<RdmaDevice> = Vec::new();
    let Ok(entries) = std::fs::read_dir("/sys/class/infiniband") else {
        return devices;
    };
    for entry in entries.filter_map(Result::ok) {
        let path = entry.path();
        let Ok(name) = entry.file_name().into_string() else {
            continue;
        };
        let numa_node = std::fs::read_to_string(path.join("device/numa_node"))
            .ok()
            .and_then(|raw| raw.trim().parse::<i32>().ok())
            // A negative value is the kernel's "no affinity", which is a
            // different fact from node 0.
            .filter(|node| *node >= 0)
            .map(|node| crate::topology::NumaNodeId(node.cast_unsigned()));
        let Ok(ports) = std::fs::read_dir(path.join("ports")) else {
            continue;
        };
        for port in ports.filter_map(Result::ok) {
            let Some(port) = port.file_name().to_str().and_then(|n| n.parse::<u8>().ok()) else {
                continue;
            };
            let link_layer = std::fs::read_to_string(path.join(format!("ports/{port}/link_layer")))
                .unwrap_or_default()
                .trim()
                .to_string();
            devices.push(RdmaDevice {
                name: name.clone(),
                port,
                numa_node,
                link_layer,
            });
        }
    }
    devices.sort_by(|a, b| (&a.name, a.port).cmp(&(&b.name, b.port)));
    devices
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn discovery_answers_on_every_platform() {
        // Empty is the normal answer. It must be a value, not a failure.
        let _ = discover();
        let _ = discover_rdma();
    }

    #[test]
    fn every_device_is_fully_described() {
        for device in discover() {
            assert!(!device.name.is_empty());
            assert!(device.link_layer.len() < 16, "sanity on a sysfs read");
        }
    }
}
