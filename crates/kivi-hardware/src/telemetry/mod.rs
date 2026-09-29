//! Hardware observation backends and the one place their numbers become
//! decisions.
//!
//! Three backends observe related effects: the performance monitoring unit
//! counts what a core did, DAMON counts which pages were touched, and Kivi's
//! own software counters count which object was asked for. Adding those three
//! together produces a number that means nothing. This module therefore does
//! not sum them: each backend produces a value on its own axis, and
//! [`signals`] is the only place where axes are combined, with the combination
//! rule stated rather than implied.
//!
//! | Backend | Axis | Cannot answer |
//! |---|---|---|
//! | [`pmu`] | What the cores executed | Which logical object was involved |
//! | [`damon`] | Which memory regions were touched | Which object, at what granularity |
//! | software counters | Which object was asked for | Whether the answer cost anything |
//!
//! All three are optional. All three are periodic and cheap: the PMU is read in
//! one batched syscall per interval, DAMON is read from a file the kernel
//! writes, and the software counters are already maintained for the controller.
//! Nothing here runs per request.
//!
//! Nothing here is correctness bearing. A counter that stops incrementing makes
//! an estimate worse, never an answer wrong.

pub mod damon;
pub mod io_uring;
pub mod pmu;
pub mod signals;

use crate::capability::{DamonParams, IoUringSupport, PmuSupport, Support};

/// What the performance monitoring unit offers, and whether this process may
/// read it.
///
/// On Linux the answer is gated by `perf_event_paranoid`, whose default of 2
/// permits per-process hardware counters and whose value of 4 on several
/// distributions forbids unprivileged access entirely. That is a machine
/// configuration, not a defect, so it is reported as
/// [`Unavailable::PermissionDenied`] and Kivi runs without counters.
#[must_use]
pub fn pmu_support() -> Support<PmuSupport> {
    pmu::detect()
}

/// Whether the kernel's DAMON control interface is present and writable.
///
/// DAMON is a kernel subsystem, so its absence is a kernel or boot
/// configuration fact. Most container and hardened kernels omit it.
#[must_use]
pub fn damon_support() -> Support<DamonParams> {
    damon::detect()
}

/// Advanced io_uring features the running kernel offers.
///
/// Windows has no io_uring, and IOCP offers none of these: registered buffers,
/// buffer rings and completion events are Linux facilities. The answer is
/// therefore `unsupported-platform` there rather than `unavailable`, which is
/// the honest distinction between "this kernel does not offer it" and "this
/// platform has no such interface".
#[must_use]
pub fn io_uring_support() -> Support<IoUringSupport> {
    io_uring::detect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Each backend must answer with a reason or a value, never an error, and
    /// must answer the same way twice: a backend whose answer changed between
    /// calls would make two subsystems disagree about the same machine.
    #[test]
    fn every_backend_answers_once_and_consistently() {
        assert_eq!(pmu_support(), pmu_support());
        assert_eq!(damon_support(), damon_support());
        assert_eq!(io_uring_support(), io_uring_support());
    }
}
