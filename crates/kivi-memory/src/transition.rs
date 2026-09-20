//! Deterministic transition state machine.
//!
//! Every representation change (DRAM ↔ compressed ↔ `NVMe`, promotion,
//! demotion, eviction staging) runs as an explicit [`Transition`] with
//! five states: `Build`, `Verify`, `Publish`, `Retire`, `Done`
//! (`Aborted` on cancellation). The machine is pure: events advance it,
//! effects (arena writes, off-core submissions, checksums) are returned
//! as [`TransitionEffect`] values the fabric executes. Deterministic
//! simulation drives the same machine with scripted effects.
//!
//! Mutation fencing: a build records the object's logical version; if the
//! object mutates before publication, the transition aborts with
//! [`MemoryError::MutatedDuringBuild`](crate::error::MemoryError::MutatedDuringBuild)
//! and the authoritative state (already mutated) stays correct.

use bytes::Bytes;

use crate::error::MemoryError;
use crate::repr::Residence;

/// Where a transition is heading.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TransitionTarget {
    /// Decoded DRAM arena residence.
    Dram,
    /// Compressed DRAM representation.
    Compressed,
    /// `NVMe` demotion (via [`OffcoreOp`](crate::offcore::OffcoreOp)).
    Nvme,
    /// Full eviction (requires a reconstruction source).
    Evicted,
}

/// State of one transition.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TransitionState {
    /// Building the new representation (compressing, staging demote
    /// bytes, reading back for promotion).
    Build,
    /// Verifying the built representation (round-trip equality,
    /// checksum, length guards).
    Verify,
    /// Publishing the new representation as primary or shadow.
    Publish,
    /// Retiring the previous representation after the safety proof.
    Retire,
    /// Finished successfully.
    Done,
    /// Aborted (cancelled or version-fenced). No state changed.
    Aborted,
}

/// Events that advance a transition.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TransitionEvent {
    /// The new representation bytes are ready for verification.
    Built {
        /// Built bytes (compressed block, demote payload, ...).
        bytes: Vec<u8>,
    },
    /// Verification succeeded.
    Verified,
    /// Verification failed; the built bytes are quarantined.
    VerifyFailed {
        /// What failed.
        detail: String,
    },
    /// Publication acknowledged by the object table.
    Published,
    /// Retirement acknowledged.
    Retired,
    /// The object mutated under the build.
    Mutated {
        /// Current logical version.
        current: u64,
    },
    /// Explicit cancellation (migration, load shed, shutdown).
    Cancel,
}

/// Side effects the fabric must execute for a transition step.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TransitionEffect {
    /// Compress these bytes and report `Built`.
    Compress {
        /// Bytes to compress.
        bytes: Bytes,
    },
    /// Decompress and verify equality against the expected bytes.
    VerifyDecompress {
        /// Expected logical bytes.
        expected: Bytes,
        /// Compressed block to verify.
        stored: Bytes,
    },
    /// Submit an `NVMe` demotion for these bytes.
    SubmitDemote {
        /// Bytes to demote.
        bytes: Vec<u8>,
    },
    /// Submit an `NVMe` promotion.
    SubmitPromote,
    /// Verify an `NVMe` record checksum and length.
    VerifyOffcore {
        /// Bytes read back.
        bytes: Vec<u8>,
        /// Expected checksum.
        checksum: u32,
    },
    /// No effect; the transition advanced internally.
    None,
}

/// One deterministic representation transition.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Transition {
    /// Which object is transitioning.
    pub object: u64,
    /// Logical version the build started from.
    pub build_version: u64,
    /// Where it is heading.
    pub target: TransitionTarget,
    /// Current state.
    pub state: TransitionState,
    /// Bytes staged during build/verify (bounded by `MEDIUM_MAX`).
    staged: Vec<u8>,
}

impl Transition {
    /// Begins a transition for an object at a logical version.
    #[must_use]
    pub const fn begin(object: u64, version: u64, target: TransitionTarget) -> Self {
        Self {
            object,
            build_version: version,
            target,
            state: TransitionState::Build,
            staged: Vec::new(),
        }
    }

    /// Current state.
    #[must_use]
    pub const fn state(&self) -> TransitionState {
        self.state
    }

    /// Advances the machine by one event, returning the effect the
    /// fabric must execute. Terminal states (`Done`, `Aborted`) ignore
    /// further events.
    ///
    /// # Errors
    ///
    /// Returns [`MemoryError::Cancelled`](crate::error::MemoryError::Cancelled)
    /// on explicit cancellation,
    /// [`MemoryError::MutatedDuringBuild`](crate::error::MemoryError::MutatedDuringBuild)
    /// when the object mutated under the build,
    /// [`MemoryError::CorruptRepresentation`](crate::error::MemoryError::CorruptRepresentation)
    /// on verification failure, or [`MemoryError::Invalid`](crate::error::MemoryError::Invalid)
    /// on an illegal event for the current state.
    pub fn step(&mut self, event: TransitionEvent) -> Result<TransitionEffect, MemoryError> {
        if matches!(self.state, TransitionState::Done | TransitionState::Aborted) {
            return Ok(TransitionEffect::None);
        }
        match event {
            TransitionEvent::Cancel => {
                self.state = TransitionState::Aborted;
                self.staged.clear();
                return Err(MemoryError::Cancelled {
                    object: self.object,
                    reason: "explicit cancel",
                });
            }
            TransitionEvent::Mutated { current } => {
                self.state = TransitionState::Aborted;
                self.staged.clear();
                return Err(MemoryError::MutatedDuringBuild {
                    object: self.object,
                    build: self.build_version,
                    current,
                });
            }
            _ => {}
        }
        match (self.state, event) {
            (TransitionState::Build, TransitionEvent::Built { bytes }) => {
                if bytes.len() as u64 > crate::MEDIUM_MAX as u64 {
                    self.state = TransitionState::Aborted;
                    return Err(MemoryError::TooLarge {
                        len: bytes.len() as u64,
                        max: crate::MEDIUM_MAX as u64,
                        context: "transition build",
                    });
                }
                self.staged = bytes;
                self.state = TransitionState::Verify;
                Ok(TransitionEffect::None)
            }
            (TransitionState::Verify, TransitionEvent::Verified) => {
                self.state = TransitionState::Publish;
                Ok(TransitionEffect::None)
            }
            (TransitionState::Verify, TransitionEvent::VerifyFailed { detail }) => {
                self.state = TransitionState::Aborted;
                self.staged.clear();
                Err(MemoryError::CorruptRepresentation {
                    object: self.object,
                    detail,
                })
            }
            (TransitionState::Publish, TransitionEvent::Published) => {
                self.state = TransitionState::Retire;
                Ok(TransitionEffect::None)
            }
            (TransitionState::Retire, TransitionEvent::Retired) => {
                self.state = TransitionState::Done;
                self.staged.clear();
                Ok(TransitionEffect::None)
            }
            (_state, _) => {
                self.state = TransitionState::Aborted;
                Err(MemoryError::Invalid {
                    detail: "illegal transition event for state".to_owned(),
                })
            }
        }
    }

    /// Takes the staged bytes (for the fabric to materialize).
    #[must_use]
    pub fn take_staged(&mut self) -> Vec<u8> {
        std::mem::take(&mut self.staged)
    }

    /// Whether the transition finished successfully.
    #[must_use]
    pub const fn is_done(&self) -> bool {
        matches!(self.state, TransitionState::Done)
    }

    /// Residence tag the transition publishes (for planner bookkeeping).
    #[must_use]
    pub const fn publishes(&self) -> &'static str {
        match self.target {
            TransitionTarget::Dram => "arena",
            TransitionTarget::Compressed => "compressed",
            TransitionTarget::Nvme => "nvme",
            TransitionTarget::Evicted => "evicted",
        }
    }

    /// Derives the first effect for a target from the current primary.
    /// Pure helper so both production and simulation start identically.
    #[must_use]
    pub fn initial_effect(target: TransitionTarget, primary: &Residence) -> TransitionEffect {
        match (target, primary) {
            (TransitionTarget::Compressed, Residence::Inline(_) | Residence::Arena { .. }) => {
                // The fabric fills the actual bytes; this documents intent.
                TransitionEffect::None
            }
            (TransitionTarget::Nvme, _) => TransitionEffect::SubmitDemote { bytes: Vec::new() },
            (TransitionTarget::Dram, Residence::Offcore { .. }) => TransitionEffect::SubmitPromote,
            _ => TransitionEffect::None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn happy_path_build_verify_publish_retire() {
        let mut transition = Transition::begin(1, 7, TransitionTarget::Compressed);
        transition
            .step(TransitionEvent::Built {
                bytes: vec![1, 2, 3],
            })
            .expect("built");
        assert_eq!(transition.state(), TransitionState::Verify);
        transition
            .step(TransitionEvent::Verified)
            .expect("verified");
        assert_eq!(transition.state(), TransitionState::Publish);
        transition
            .step(TransitionEvent::Published)
            .expect("published");
        assert_eq!(transition.state(), TransitionState::Retire);
        transition.step(TransitionEvent::Retired).expect("retired");
        assert!(transition.is_done());
    }

    #[test]
    fn mutation_during_build_aborts_without_state_change() {
        let mut transition = Transition::begin(1, 7, TransitionTarget::Nvme);
        let error = transition
            .step(TransitionEvent::Mutated { current: 8 })
            .expect_err("must abort");
        assert!(matches!(error, MemoryError::MutatedDuringBuild { .. }));
        assert_eq!(transition.state(), TransitionState::Aborted);
    }

    #[test]
    fn verify_failure_quarantines_the_build() {
        let mut transition = Transition::begin(1, 7, TransitionTarget::Compressed);
        transition
            .step(TransitionEvent::Built { bytes: vec![9] })
            .expect("built");
        let error = transition
            .step(TransitionEvent::VerifyFailed {
                detail: "round-trip mismatch".to_owned(),
            })
            .expect_err("must fail");
        assert!(matches!(error, MemoryError::CorruptRepresentation { .. }));
    }

    #[test]
    fn terminal_states_ignore_further_events() {
        let mut transition = Transition::begin(1, 7, TransitionTarget::Dram);
        transition
            .step(TransitionEvent::Cancel)
            .expect_err("cancel");
        assert_eq!(transition.state(), TransitionState::Aborted);
        let effect = transition.step(TransitionEvent::Verified).expect("ignored");
        assert_eq!(effect, TransitionEffect::None);
    }
}
