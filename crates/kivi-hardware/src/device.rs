//! I/O device locality.
//!
//! A device belongs on the NUMA node whose CPUs and DRAM are closest to it. A
//! worker doing heavy I/O against a device on node 1 while running on node 0
//! pays cross-socket latency on every completion and every buffer touch, which
//! is the reason this module exists.
//!
//! Linux discloses locality per PCI device through `numa_node` and exposes CXL
//! capacity as `/dev/daxN.Y` character devices. Windows exposes no PCI-locality
//! interface to an unprivileged application, so [`discover`] returns nothing
//! there rather than guessing.

#[cfg(any(target_os = "linux", target_os = "android"))]
use crate::topology::IoDevice;
#[cfg(not(any(target_os = "linux", target_os = "android")))]
use crate::topology::IoDevice;

#[cfg(any(target_os = "linux", target_os = "android"))]
mod imp {
    use std::fs;
    use std::path::Path;

    use crate::topology::{DeviceClass, IoDevice, NumaNodeId};

    /// PCI class/subclass pairs Kivi cares about.
    const NVME: u32 = 0x01_08;
    const ETHERNET: u32 = 0x02_00;
    const PROCESSING_ACCELERATOR: u32 = 0x12_00;

    fn numa_node_of(path: &Path) -> Option<NumaNodeId> {
        let raw = fs::read_to_string(path.join("numa_node")).ok()?;
        let value = raw.trim().parse::<i32>().ok()?;
        // A negative `numa_node` is the kernel's way of saying "no affinity",
        // which is a different fact from node 0 and must not become one.
        (value >= 0).then_some(NumaNodeId(value.cast_unsigned()))
    }

    fn pci_class(path: &Path) -> u32 {
        fs::read_to_string(path.join("class"))
            .ok()
            .and_then(|raw| u32::from_str_radix(raw.trim(), 16).ok())
            .unwrap_or(0)
    }

    fn pci_devices() -> Vec<IoDevice> {
        let mut out: Vec<IoDevice> = Vec::new();
        let Ok(entries) = fs::read_dir("/sys/bus/pci/devices") else {
            return out;
        };
        for entry in entries.filter_map(Result::ok) {
            let path = entry.path();
            let name = entry.file_name().into_string().unwrap_or_default();
            let class = pci_class(&path);
            // A PCI class alone cannot distinguish an RDMA NIC from a plain
            // NIC. `class/net/*/device/infiniband` is the kernel's own
            // statement that the device has RDMA character devices.
            let kind = if class == NVME {
                DeviceClass::Block
            } else if class == ETHERNET && has_infiniband(&path) {
                DeviceClass::RdmaNic
            } else if class == PROCESSING_ACCELERATOR {
                DeviceClass::DsaEngine
            } else {
                continue;
            };
            out.push(IoDevice {
                class: kind,
                name,
                numa_node: numa_node_of(&path),
                path: path.to_string_lossy().into_owned(),
            });
        }
        out
    }

    fn has_infiniband(pci_path: &Path) -> bool {
        let Ok(net) = fs::read_dir("/sys/class/net") else {
            return false;
        };
        net.filter_map(Result::ok).any(|interface| {
            fs::read_link(interface.path().join("device")).is_ok_and(|device| {
                device.starts_with(pci_path) && interface.path().join("device/infiniband").is_dir()
            })
        })
    }

    /// `/dev/daxN.Y` character devices, which is how CXL capacity reaches
    /// userspace when the kernel exposes it as Device DAX.
    pub(super) fn dax_regions() -> Vec<IoDevice> {
        let Ok(entries) = fs::read_dir("/dev") else {
            return Vec::new();
        };
        let mut out: Vec<IoDevice> = entries
            .filter_map(Result::ok)
            .filter_map(|entry| {
                let name = entry.file_name().into_string().ok()?;
                let rest = name.strip_prefix("dax")?;
                if rest.is_empty() || !rest.chars().all(|c| c.is_ascii_digit() || c == '.') {
                    return None;
                }
                Some(IoDevice {
                    class: DeviceClass::CxlRegion,
                    name: format!("/dev/{name}"),
                    numa_node: dax_numa_node(&name),
                    path: format!("/sys/devices/virtual/dax/{name}"),
                })
            })
            .collect();
        out.sort_by(|a, b| a.name.cmp(&b.name));
        out
    }

    /// A DAX region is attached to whichever node maps its pages. The kernel
    /// exposes that through the region's own sysfs node membership rather than
    /// through the region itself, so the lookup walks
    /// `/sys/devices/virtual/dax/<region>/dax*/`.
    fn dax_numa_node(region: &str) -> Option<NumaNodeId> {
        let base = std::path::Path::new("/sys/devices/virtual/dax").join(region);
        let entries = fs::read_dir(&base).ok()?;
        for entry in entries.filter_map(Result::ok) {
            let member = entry.path();
            let Ok(node) = fs::read_link(member.join("node")) else {
                continue;
            };
            let text = node.to_string_lossy();
            let digits = text.rsplit("node").next()?;
            if let Ok(id) = digits.trim_start_matches(['/', 'c']).parse::<u32>() {
                return Some(NumaNodeId(id));
            }
        }
        None
    }

    pub(super) fn discover() -> Vec<IoDevice> {
        let mut out = pci_devices();
        out.extend(dax_regions());
        out
    }
}

#[cfg(not(any(target_os = "linux", target_os = "android")))]
mod imp {
    use super::IoDevice;

    pub(super) fn discover() -> Vec<IoDevice> {
        Vec::new()
    }
}

/// Devices with a NUMA node the platform discloses.
///
/// Empty means the platform does not disclose locality. That is a valid
/// machine, not a failure: placement and the memory fabric treat undisclosed
/// locality as "unknown", never as "node 0".
#[must_use]
pub fn discover() -> Vec<IoDevice> {
    imp::discover()
}

/// CXL capacity exposed as Device DAX character devices.
#[must_use]
pub fn dax_regions() -> Vec<IoDevice> {
    #[cfg(any(target_os = "linux", target_os = "android"))]
    {
        imp::dax_regions()
    }
    #[cfg(not(any(target_os = "linux", target_os = "android")))]
    {
        Vec::new()
    }
}
