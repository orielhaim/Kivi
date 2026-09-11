//! CPU affinity for `DataWorker` threads, behind a project-owned boundary.
//!
//! [`AffinityMode`] describes intent (automatic, explicit, or disabled) in
//! plain `usize` core indices; `core_affinity2` types never escape this
//! module. NUMA topology awareness (`hwlocality`) is a later stage — this
//! layer pins threads to cores and nothing more.
//!
//! Affinity applies to production networked workers. Channel-only workers
//! (tests, embedded use) never pin: parallel test binaries sharing a machine
//! must not fight over cores.

use core::fmt;

/// How a worker thread binds to CPUs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AffinityMode {
    /// No pinning; the OS schedules freely. Used for tests and development.
    Disabled,
    /// Pin worker `i` to available core `i % core_count`.
    Auto,
    /// Pin worker `i` to `cores[i % cores.len()]`. Empty lists behave as
    /// [`Disabled`](Self::Disabled) rather than failing.
    Explicit(Vec<usize>),
}

impl fmt::Display for AffinityMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Disabled => write!(f, "disabled"),
            Self::Auto => write!(f, "auto"),
            Self::Explicit(cores) => {
                write!(f, "explicit[")?;
                for (index, core) in cores.iter().enumerate() {
                    if index > 0 {
                        write!(f, ",")?;
                    }
                    write!(f, "{core}")?;
                }
                write!(f, "]")
            }
        }
    }
}

/// Affinity establishment failure.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum AffinityError {
    /// The platform reported no usable cores.
    #[error("no CPU cores reported by the platform")]
    NoCores,
    /// An explicitly requested core is outside the available set.
    #[error("requested core {core} but only {available} cores exist")]
    UnknownCore {
        /// Requested core index.
        core: usize,
        /// Cores the platform reported.
        available: usize,
    },
    /// The OS rejected the pinning request.
    #[error("operating system refused to pin thread to core {core}")]
    SetFailed {
        /// Core that could not be set.
        core: usize,
    },
}

/// Pins the calling thread for `worker_index` of `worker_count` workers.
///
/// Returns the core pinned to, or `None` when pinning is disabled (or the
/// explicit list is empty). Explicitly requested but unestablishable
/// affinity fails with [`AffinityError`] — never silently ignored.
///
/// # Errors
///
/// Returns [`AffinityError`] when no cores exist, an explicit core is out of
/// range, or the OS rejects the request.
pub fn pin_current_thread(
    mode: &AffinityMode,
    worker_index: usize,
    worker_count: usize,
) -> Result<Option<usize>, AffinityError> {
    let _ = worker_count;
    match mode {
        AffinityMode::Disabled => Ok(None),
        AffinityMode::Auto => {
            let cores = core_affinity2::get_core_ids().map_err(|_| AffinityError::NoCores)?;
            if cores.is_empty() {
                return Err(AffinityError::NoCores);
            }
            let core = worker_index % cores.len();
            set_core(core)?;
            Ok(Some(core))
        }
        AffinityMode::Explicit(cores) => {
            if cores.is_empty() {
                return Ok(None);
            }
            let core = cores[worker_index % cores.len()];
            let available = core_affinity2::get_core_ids().map_err(|_| AffinityError::NoCores)?;
            if !available.iter().any(|id| usize::from(*id) == core) {
                return Err(AffinityError::UnknownCore {
                    core,
                    available: available.len(),
                });
            }
            set_core(core)?;
            Ok(Some(core))
        }
    }
}

/// Returns usable core indices, or an empty vector when the platform reports
/// none. Introspection helper for CLIs and diagnostics (never on hot paths).
#[must_use]
pub fn available_cores() -> Vec<usize> {
    core_affinity2::get_core_ids()
        .map(|ids| ids.into_iter().map(usize::from).collect())
        .unwrap_or_default()
}

fn set_core(core: usize) -> Result<(), AffinityError> {
    core_affinity2::CoreId::from(core)
        .set_affinity()
        .map_err(|_| AffinityError::SetFailed { core })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disabled_never_touches_affinity() {
        assert_eq!(pin_current_thread(&AffinityMode::Disabled, 0, 4), Ok(None));
        assert_eq!(
            pin_current_thread(&AffinityMode::Explicit(Vec::new()), 0, 4),
            Ok(None)
        );
    }

    #[test]
    fn explicit_out_of_range_fails_clearly() {
        let mode = AffinityMode::Explicit(vec![usize::MAX]);
        assert!(matches!(
            pin_current_thread(&mode, 0, 1),
            Err(AffinityError::UnknownCore { .. })
        ));
    }

    #[test]
    fn display_names_modes() {
        assert_eq!(AffinityMode::Auto.to_string(), "auto");
        assert_eq!(
            AffinityMode::Explicit(vec![0, 2]).to_string(),
            "explicit[0,2]"
        );
    }
}
