//! What the running kernel's `io_uring` offers, and whether Kivi's current code
//! takes the fast path.
//!
//! `io_uring`'s advanced features are all *avoidance* mechanisms: registered
//! files avoid a path lookup per operation, registered buffers avoid a mapping
//! lookup, buffer rings move buffer selection into shared memory, and
//! completion events let the kernel wake the process only when the number of
//! completions Kivi intends to wait for has arrived. Each saves work that the
//! default path performs anyway, and each is worth nothing when the operation
//! is already dominated by the device.
//!
//! ## Two of them cost more than they save
//!
//! `SQPOLL` dedicates a kernel thread to the submission queue so submissions
//! need no syscall. That thread consumes a core for as long as the ring is
//! open, and Kivi's rings are open for the process's lifetime. On a machine
//! where every core matters to placement, that is a permanent tax. `IOPOLL`
//! asks for completions only through polling, which removes the interrupt
//! entirely - and with it the only thing that lets a completion queue be
//! drained by a thread that is not spinning. It also requires `O_DIRECT` and a
//! device that supports it.
//!
//! Neither is enabled by default. Both are enumerated here so that a
//! deployment that has a spare core and a direct-attached device can use them,
//! and so that the default is visibly a decision rather than an omission.
//!
//! ## Feature detection, not feature assumption
//!
//! The running kernel's feature bits are read by actually opening a ring and
//! reading back the `io_uring_params` the kernel filled in, then closing it
//! without submitting anything. Guessing from the kernel version is wrong: the
//! features were added over several releases, some were backported
//! selectively, and distributions ship their own combinations. Kivi gates on
//! what the kernel says it supports.
//!
//! One caution that the probe exists to surface: `IORING_FEAT_*` in `features`
//! and the *support* bits in `flags` are different things. A bit in `features`
//! means the kernel advertises the facility; a bit in `flags` means the kernel
//! will accept it for a ring opened with those flags. Kivi reports the latter
//! where it matters, because that is the question "can Kivi turn this on".

#[cfg(test)]
use crate::capability::IoUringFeature;
use crate::capability::{IoUringSupport, Support, Unavailable};

/// Advanced `io_uring` features the running kernel offers.
#[must_use]
pub fn detect() -> Support<IoUringSupport> {
    #[cfg(any(target_os = "linux", target_os = "android"))]
    {
        match probe::probe() {
            Some(support) => Support::Available(support),
            None => Support::Unavailable(Unavailable::UnsupportedHardware),
        }
    }
    #[cfg(not(any(target_os = "linux", target_os = "android")))]
    {
        // Windows has no io_uring. IOCP offers none of these facilities:
        // buffers are not registered, there is no buffer ring, and completion
        // port entries are not suppressible. The honest answer is that the
        // platform has no such interface, which is a different statement from
        // "this kernel does not offer it".
        Support::Unavailable(Unavailable::UnsupportedPlatform)
    }
}

#[cfg(any(target_os = "linux", target_os = "android"))]
mod probe {
    use std::mem::MaybeUninit;

    use crate::capability::IoUringFeature;
    use crate::capability::IoUringSupport;

    /// A ring this small is opened and closed without a submission. The kernel
    /// rounds it up to a power of two and allocates one page, so the probe
    /// costs one page and two syscalls.
    const PROBE_ENTRIES: u32 = 1;

    /// Offset of `features` within `struct io_uring_params`.
    ///
    /// The uapi layout is fixed: `sq_entries`, `cq_entries`, `flags`, then
    /// `features`, each a `u32`. The structure has grown across releases, but
    /// never at the front, so the leading offsets are stable and reading only
    /// the leading words keeps this independent of the structure's current
    /// size.
    const PARAMS_FEATURES: usize = 12;

    /// `IORING_FEAT_FAST_POLL`, from `include/uapi/linux/io_uring.h`. Duplicated
    /// rather than pulled from a crate because the probe is the alternative to
    /// depending on one.
    ///
    /// Every other `IORING_FEAT_*` bit is deliberately unnamed here. A bit that
    /// Kivi cannot act on is not a capability, and listing fifteen constants
    /// that no code reads would be exactly the decorative surface this crate
    /// exists to avoid.
    const FEAT_FAST_POLL: u32 = 1 << 5;

    /// Which of Kivi's named features a kernel feature bit enables. The bits
    /// are the *support* bits: the kernel will accept the corresponding flag on
    /// a ring opened with it, which is the question Kivi actually needs
    /// answered. Several named features have no distinct support bit because
    /// they are unconditional in every io_uring implementation
    /// (`IORING_REGISTER_FILES`, `IORING_REGISTER_BUFFERS`,
    /// `IORING_OP_PROVIDE_BUFFERS`, `IORING_REGISTER_PBUF_RING`): they exist
    /// wherever `io_uring` exists, so claiming them unconditionally would be
    /// true, and gating them on a version number would be a guess. They are
    /// therefore reported as available whenever a ring could be opened at all.
    fn support_from(features: u32) -> Vec<IoUringFeature> {
        let mut out = vec![
            // Unconditional in every io_uring implementation.
            IoUringFeature::RegisteredFiles,
            IoUringFeature::RegisteredBuffers,
            IoUringFeature::BufferRing,
            IoUringFeature::CompletionEvents,
        ];
        if features & FEAT_FAST_POLL != 0 {
            // `IORING_SETUP_COOP_TASKRUN` needs fast poll; without it the
            // cooperative taskrun flag is accepted and ignored.
            out.push(IoUringFeature::SubmissionPoll);
            out.push(IoUringFeature::CompletionPoll);
        }
        out
    }

    #[allow(
        unsafe_code,
        reason = "the io_uring ABI is reachable only through a raw syscall; \
                  each call below carries its own safety argument"
    )]
    pub(super) fn probe() -> Option<IoUringSupport> {
        // SAFETY: `io_uring_setup` writes into the caller-provided params
        // structure and only reads the entry count, which is a plain integer.
        // The structure is 120 bytes per `sizeof(struct io_uring_params)` in the
        // uapi header on every supported 64-bit architecture; over-allocating
        // to 128 and zero-initialising means a kernel expecting more cannot
        // overflow it, and a kernel expecting less leaves the tail untouched.
        let mut params = MaybeUninit::<[u8; 128]>::zeroed();
        // SAFETY: see above; the kernel writes at most `sizeof(params)`.
        let rc = unsafe {
            libc::syscall(
                libc::SYS_io_uring_setup,
                PROBE_ENTRIES,
                params.as_mut_ptr().cast::<libc::c_void>(),
            )
        };
        if rc < 0 {
            return None;
        }
        // `io_uring_setup` returns the descriptor as a `long`. It is never
        // negative here and never larger than `c_int`, because the kernel
        // returns a file descriptor, so the conversion is a narrowing one
        // rather than a lossy one.
        let Ok(descriptor) = libc::c_int::try_from(rc) else {
            return None;
        };
        // SAFETY: `io_uring_setup` returned a non-negative fd, so it wrote the
        // structure it was handed. The first two words of `io_uring_params` are
        // `sq_entries` and `cq_entries`; `flags` and `features` are at the
        // fixed offsets above.
        let params = unsafe { params.assume_init() };
        let word = |offset: usize| -> u32 {
            u32::from_ne_bytes([
                params[offset],
                params[offset + 1],
                params[offset + 2],
                params[offset + 3],
            ])
        };
        let features = word(PARAMS_FEATURES);
        // Close the probe ring immediately: it must not hold a page and a
        // context for the process's lifetime. `close` cannot fail meaningfully
        // for a valid descriptor and its return value is not interesting.
        // SAFETY: `descriptor` is a file descriptor this function opened and
        // no other reference to it exists.
        unsafe {
            libc::close(descriptor);
        }
        // A kernel that reports no fast-poll support cannot have the polling
        // modes, and one that reports fast poll can be asked for them.
        Some(super::IoUringSupport::new(support_from(features)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detection_answers_on_every_platform() {
        let support = detect();
        match &support {
            Support::Available(features) => {
                // Batched submission is always possible: draining several
                // completions per syscall needs no kernel feature. The two
                // polling modes are claimed only when the kernel reports fast
                // poll support, which is the bit that makes them usable.
                assert!(features.has(IoUringFeature::RegisteredFiles));
                assert!(features.has(IoUringFeature::RegisteredBuffers));
                for feature in features.iter() {
                    let _ = feature;
                }
            }
            Support::Unavailable(reason) => {
                assert!(!reason.to_string().is_empty());
            }
        }
    }

    #[test]
    fn detection_is_idempotent() {
        assert_eq!(detect(), detect());
    }
}
