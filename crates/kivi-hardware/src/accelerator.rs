//! Optional accelerators that are evaluated, not adopted.
//!
//! Three of them were researched in full, and none of them earned a default
//! dependency. This module records the decision, the reason, and the shape the
//! integration would take if a deployment wanted one - so the next engineer
//! does not repeat the evaluation, and so the capability snapshot can name
//! them even when they are absent.
//!
//! ## Intel Data Streaming Accelerator
//!
//! DSA does bulk copy, fill, compare, and checksum off the cores, for large
//! transfers. The two candidate uses in Kivi are redundancy encode/decode
//! buffer movement and large-object staging.
//!
//! **Evaluated, not adopted.** There is no Rust support. The `dsa` crate on
//! `crates.io` is RustCrypto's *Digital Signature Algorithm* and is unrelated; no
//! `idxd` or oneDSA binding exists on crates.io or lib.rs. Intel ships oneDPL
//! and oneDSA in C++, and DPDK has an `idxd` PMD that would work against the
//! in-kernel `idxd` driver or `vfio-pci`. Adopting it would mean writing and
//! maintaining the `ioctl` FFI by hand, against hardware that exists only on
//! suitable Xeon platforms.
//!
//! **The measurement that decides it, and which has not been taken:** a DSA
//! copy occupies the same DRAM bandwidth as a core copy, so DSA only wins by
//! *freeing a core*. That is worth something on a many-socket machine running
//! several storage processes, where the freed core can be given to request
//! serving. It is worth nothing on a machine whose cores are already the
//! bottleneck for Kivi's own work. A decision here needs a DSA machine and a
//! profile showing that Kivi's cores are saturated by memory movement, which is
//! not a profile the software architecture currently produces.
//!
//! ## SPDK
//!
//! SPDK is a user-space NVMe stack that removes the kernel from the block I/O
//! path, with its own poll-mode reactors and its own huge-page requirements.
//!
//! **Evaluated, not adopted.** The three Rust efforts are all 0.x:
//!
//! * `spdk-rs` is not on `crates.io`. The `OpenEBS` project of that name is an
//!   internal component of Mayastor, requires Nix, builds SPDK from a pinned
//!   fork, and states that it is not designed for an arbitrary SPDK version.
//! * `spdk-io` 0.1.0 is a reasonable async design but its own documentation
//!   does not build, which means a default build of a crate that depends on it
//!   would be a documentation failure and a build failure for anyone without
//!   SPDK installed.
//! * `async-spdk` 0.1.0 is a 1.5 kB stub.
//! * `ironspdk` 0.1.14 is the only substantial one: it keeps SPDK's reactor
//!   execution model rather than bolting a general async runtime on top,
//!   which is the right architecture. It requires SPDK 26.01, a C SPDK build
//!   with a specific configure recipe, superuser, and VFIO or the kernel
//!   driver, and it is 0.x.
//!
//! **The measurement that decides it, and which has not been taken:** SPDK's
//! win is at NVMe queue depth, where a kernel submission is measurably more
//! expensive than a user-space one. Kivi's off-core lane is one OS thread doing
//! batched appends against a per-worker record file, and the recent
//! performance work showed the dominant costs were the number of flushes and
//! copies, not the submission path. So the submission path SPDK removes is not
//! currently the one that costs. That conclusion is only valid for the workload
//! measured, which is why the integration point is described rather than built:
//! it would be a `DurabilityProvider` implementation and a Memory Fabric
//! provider, both of which are existing seams.
//!
//! ## DPDK and kernel bypass
//!
//! **Not implemented, and deliberately so.** The performance work found the
//! major gains came from storage and durability, not from packet processing.
//! A kernel-bypass networking rewrite without evidence is out of scope: it
//! replaces a working QUIC/H3 mesh with a poll-mode one, needs huge pages,
//! needs NIC driver support, and buys nothing for a protocol whose cost is
//! round trips. Revisit only if a profile shows the kernel network stack is
//! material, and then as a provider behind the existing peer transport, not as
//! a second transport.

/// Intel DSA engines present on this machine.
#[must_use]
pub fn discover_dsa() -> Vec<String> {
    #[cfg(any(target_os = "linux", target_os = "android"))]
    {
        discover_dsa_linux()
    }
    #[cfg(not(any(target_os = "linux", target_os = "android")))]
    {
        // DSA is an Intel Xeon platform feature driven by the in-kernel `idxd`
        // driver. There is no equivalent elsewhere, and the honest answer on a
        // platform without the driver is an empty list.
        Vec::new()
    }
}

#[cfg(any(target_os = "linux", target_os = "android"))]
fn discover_dsa_linux() -> Vec<String> {
    let Ok(entries) = std::fs::read_dir("/dev/dsa") else {
        return Vec::new();
    };
    let mut out: Vec<String> = entries
        .filter_map(Result::ok)
        .map(|entry| entry.file_name().to_string_lossy().into_owned())
        .collect();
    out.sort_unstable();
    out
}

/// The evaluations this crate records, for the admin surface and for the next
/// engineer who asks why a track is not wired in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum AcceleratorTrack {
    /// SPDK user-space NVMe.
    Spdk,
    /// Intel Data Streaming Accelerator.
    IntelDsa,
    /// RDMA verbs.
    Rdma,
    /// DPDK kernel bypass.
    Dpdk,
    /// AF_XDP or userspace-interrupt networking.
    UserspaceNetworking,
}

impl AcceleratorTrack {
    /// Every track that was researched.
    pub const ALL: &'static [Self] = &[
        Self::Spdk,
        Self::IntelDsa,
        Self::Rdma,
        Self::Dpdk,
        Self::UserspaceNetworking,
    ];

    /// One-line reason for the decision, suitable for an admin response.
    #[must_use]
    pub const fn decision(self) -> &'static str {
        match self {
            Self::Spdk => {
                "evaluated, not adopted: 0.x bindings need a pinned C SPDK build and \
                 superuser, and Kivi's dominant I/O cost is flush count and copies, \
                 not submission path"
            }
            Self::IntelDsa => {
                "evaluated, not adopted: no Rust binding exists; the FFI would be \
                 hand-written against hardware that is only on suitable Xeon parts, \
                 and DSA only wins by freeing a core Kivi may already need"
            }
            Self::Rdma => {
                "capability, not dependency: sideway is MPL-2.0 and needs cmake, \
                 clang and librdmacm; RDMA optimises transport only and never \
                 bypasses identity, fencing or verification"
            }
            Self::Dpdk => {
                "not implemented: no profile shows the kernel network stack is \
                 material, and the measured gains came from storage and durability"
            }
            Self::UserspaceNetworking => {
                "not implemented: the QUIC/H3 peer mesh's cost is round trips, not \
                 per-packet work, so a poll-mode stack would not shorten them"
            }
        }
    }

    /// The classification this track carries in a default build.
    #[must_use]
    pub const fn classification(self) -> &'static str {
        match self {
            Self::Rdma => "opt-in",
            _ => "experimental",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every track must state a decision and a classification. A track with an
    /// empty decision is the failure mode this module exists to prevent: an
    /// operator asks why a capability is not wired in and gets silence.
    #[test]
    fn every_track_states_a_reason_and_a_classification() {
        for track in AcceleratorTrack::ALL {
            assert!(!track.decision().is_empty(), "{track:?}");
            assert!(
                [
                    "default",
                    "runtime-auto",
                    "opt-in",
                    "experimental",
                    "unsupported"
                ]
                .contains(&track.classification()),
                "{track:?} says {:?}",
                track.classification()
            );
        }
    }
}
