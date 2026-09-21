//! Unified consensus command envelope.
//!
//! One Kivi-owned envelope covers tablet writes, control-plane writes, and
//! Lazy-ALR read fences, so a single `OpenRaft` type configuration and a
//! single set of storage/transport/codec implementations serve every group:
//!
//! ```text
//! ConsensusCommand {
//!     Tablet(ReplicatedMutation),
//!     Control(ControlMutation),
//!     ReadSync(AlrFenceSync),
//! }
//! ```
//!
//! Tablet groups apply `Tablet` through the replicated state machine;
//! the system control group applies `Control` through replicated control
//! state; `ReadSync` advances the applied pointer on tablet groups without
//! mutating any logical state (the Lazy-ALR synchronization boundary).
//! There is exactly one `OpenRaft`↔Kivi boundary (this type plus
//! [`ReplicatedOutcome`](crate::mutation::ReplicatedOutcome)); no
//! parallel codec/conversion implementations exist.

use kivi_types::{AlrFence, CommitPosition, TabletId};

use crate::mutation::{ReplicatedMutation, ReplicatedMutationError};

/// Framing version of [`ConsensusCommand`].
pub const CONSENSUS_COMMAND_VERSION: u16 = 1;

/// Framing version of [`AlrFenceSync`].
pub const ALR_FENCE_SYNC_VERSION: u16 = 1;

/// One Lazy-ALR synchronization: an [`AlrFence`] plus the batch-former's
/// applied position at formation time.
///
/// Applying a fence advances the applied pointer and nothing else: no
/// objects change, no sessions advance, no outcomes install, no dedup
/// records. Its log slot still orders exactly like a write, so when the
/// fence (or a subsuming write ordered after formation) is locally
/// applied, every write that completed before the batch formed is
/// necessarily visible. `formation_applied` records what the former knew
/// at formation for diagnostics and subsumption proofs; apply ignores it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AlrFenceSync {
    /// Fence identity (tablet, batch, requester).
    pub fence: AlrFence,
    /// Former applied position at batch formation (same-log coordinates).
    pub formation_applied: CommitPosition,
}

impl AlrFenceSync {
    /// Builds a fence sync from its two facts.
    #[must_use]
    pub const fn new(fence: AlrFence, formation_applied: CommitPosition) -> Self {
        Self {
            fence,
            formation_applied,
        }
    }

    /// Encodes the canonical bytes: `version u16, tablet u64, batch u64,
    /// requester u64, formation u64` (fixed 34 bytes, little-endian).
    #[must_use]
    pub fn encode_to_vec(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(2 + 8 * 4);
        out.extend_from_slice(&ALR_FENCE_SYNC_VERSION.to_le_bytes());
        out.extend_from_slice(&self.fence.tablet.as_u64().to_le_bytes());
        out.extend_from_slice(&self.fence.batch.to_le_bytes());
        out.extend_from_slice(&self.fence.requester.as_u64().to_le_bytes());
        out.extend_from_slice(&self.formation_applied.as_u64().to_le_bytes());
        out
    }

    /// Decodes exactly one fence sync.
    ///
    /// # Errors
    ///
    /// Returns [`ConsensusCommandError`] on version mismatch, truncation,
    /// or trailing bytes. The tablet/positions are structural here;
    /// group-membership checks happen at apply time, never here.
    pub fn decode_exact(input: &[u8]) -> Result<Self, ConsensusCommandError> {
        use ConsensusCommandError as Fault;
        if input.len() < 2 + 8 * 4 {
            return Err(Fault::Truncated);
        }
        let version = u16::from_le_bytes(input[..2].try_into().unwrap_or([0; 2]));
        if version != ALR_FENCE_SYNC_VERSION {
            return Err(Fault::UnsupportedVersion { found: version });
        }
        let body = &input[2..];
        if body.len() != 8 * 4 {
            return Err(if body.len() < 8 * 4 {
                Fault::Truncated
            } else {
                Fault::TrailingBytes
            });
        }
        let word = |offset: usize| {
            u64::from_le_bytes(body[offset..offset + 8].try_into().unwrap_or([0; 8]))
        };
        Ok(Self {
            fence: AlrFence::new(
                TabletId::from_u64(word(0)),
                word(8),
                kivi_types::NodeId::from_u64(word(16)),
            ),
            formation_applied: CommitPosition::from_u64(word(24)),
        })
    }
}

impl core::fmt::Display for AlrFenceSync {
    /// Short identity summary (never the canonical encoding).
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "read-sync({} formation {})",
            self.fence, self.formation_applied
        )
    }
}

/// What gets proposed, committed, and applied in any group: either a
/// deterministic tablet command, a typed control-plane mutation, or a
/// Lazy-ALR read fence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConsensusCommand {
    /// Deterministic tablet write (normal data path).
    Tablet(ReplicatedMutation),
    /// Typed control-plane mutation (system control group only).
    Control(kivi_control::ControlMutation),
    /// Lazy-ALR read fence (tablet groups only): orders like a write,
    /// mutates nothing, carries no client outcome.
    ReadSync(AlrFenceSync),
}

/// Why a consensus command was rejected at decode time.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum ConsensusCommandError {
    /// Framing version this binary cannot speak.
    #[error("unsupported consensus-command version {found}")]
    UnsupportedVersion {
        /// Observed version.
        found: u16,
    },
    /// Truncated input where framing promised bytes.
    #[error("truncated consensus command")]
    Truncated,
    /// Unknown command tag.
    #[error("unknown consensus command tag {tag}")]
    BadTag {
        /// Observed tag.
        tag: u8,
    },
    /// Trailing bytes after the framed command.
    #[error("trailing bytes in consensus command")]
    TrailingBytes,
    /// The tablet body does not decode.
    #[error("undecodable tablet command: {detail}")]
    BadTablet {
        /// Human-readable cause.
        detail: String,
    },
    /// The control body does not decode.
    #[error("undecodable control mutation: {detail}")]
    BadControl {
        /// Human-readable cause.
        detail: String,
    },
    /// The read-fence body does not decode.
    #[error("undecodable read fence: {detail}")]
    BadSync {
        /// Human-readable cause.
        detail: String,
    },
}

impl ConsensusCommand {
    /// Wraps a tablet command.
    #[must_use]
    pub const fn tablet(command: ReplicatedMutation) -> Self {
        Self::Tablet(command)
    }

    /// Wraps a control mutation.
    #[must_use]
    pub const fn control(mutation: kivi_control::ControlMutation) -> Self {
        Self::Control(mutation)
    }

    /// Returns the tablet command, if this is one.
    #[must_use]
    pub const fn as_tablet(&self) -> Option<&ReplicatedMutation> {
        match self {
            Self::Tablet(command) => Some(command),
            Self::Control(_) | Self::ReadSync(_) => None,
        }
    }

    /// Returns the control mutation, if this is one.
    #[must_use]
    pub const fn as_control(&self) -> Option<&kivi_control::ControlMutation> {
        match self {
            Self::Tablet(_) | Self::ReadSync(_) => None,
            Self::Control(mutation) => Some(mutation),
        }
    }

    /// Whether this is a tablet command.
    #[must_use]
    pub const fn is_tablet(&self) -> bool {
        matches!(self, Self::Tablet(_))
    }

    /// Whether this is a control mutation.
    #[must_use]
    pub const fn is_control(&self) -> bool {
        matches!(self, Self::Control(_))
    }

    /// Whether this is a Lazy-ALR read fence.
    #[must_use]
    pub const fn is_read_sync(&self) -> bool {
        matches!(self, Self::ReadSync(_))
    }

    /// Returns the read fence, if this is one.
    #[must_use]
    pub const fn as_read_sync(&self) -> Option<&AlrFenceSync> {
        match self {
            Self::ReadSync(sync) => Some(sync),
            Self::Tablet(_) | Self::Control(_) => None,
        }
    }

    /// Encodes the canonical bytes: `version u16, tag u8,
    /// body = inner canonical bytes`.
    ///
    /// The inner bodies are self-framed (their own version/length
    /// prefixes), so no outer length prefix is needed: the tag selects
    /// the inner decoder and the inner decoder enforces exactness. Unknown
    /// tags fail loudly: an old binary never mistakes a fence for a write.
    #[must_use]
    pub fn encode_to_vec(&self) -> Vec<u8> {
        let (tag, body) = match self {
            Self::Tablet(command) => (0u8, command.encode_to_vec()),
            Self::Control(mutation) => (1u8, mutation.encode_to_vec()),
            Self::ReadSync(sync) => (2u8, sync.encode_to_vec()),
        };
        let mut out = Vec::with_capacity(2 + 1 + body.len());
        out.extend_from_slice(&CONSENSUS_COMMAND_VERSION.to_le_bytes());
        out.push(tag);
        out.extend_from_slice(&body);
        out
    }

    /// Decodes exactly one command.
    ///
    /// # Errors
    ///
    /// Returns [`ConsensusCommandError`] on version mismatch, truncation,
    /// unknown tags, undecodable bodies, or trailing bytes (enforced by
    /// the inner decoders).
    pub fn decode_exact(input: &[u8]) -> Result<Self, ConsensusCommandError> {
        use ConsensusCommandError as Fault;
        if input.len() < 2 + 1 {
            return Err(Fault::Truncated);
        }
        let version = u16::from_le_bytes(input[..2].try_into().unwrap_or([0; 2]));
        if version != CONSENSUS_COMMAND_VERSION {
            return Err(Fault::UnsupportedVersion { found: version });
        }
        let tag = input[2];
        let body = &input[3..];
        match tag {
            0 => {
                let command =
                    ReplicatedMutation::decode_exact(body).map_err(|error| match error {
                        ReplicatedMutationError::Truncated
                        | ReplicatedMutationError::Oversized { .. } => Fault::Truncated,
                        other => Fault::BadTablet {
                            detail: other.to_string(),
                        },
                    })?;
                Ok(Self::Tablet(command))
            }
            1 => {
                let mutation = kivi_control::ControlMutation::decode_exact(body).map_err(
                    |error| match error {
                        kivi_control::ControlMutationError::Truncated
                        | kivi_control::ControlMutationError::Oversized { .. } => Fault::Truncated,
                        other => Fault::BadControl {
                            detail: other.to_string(),
                        },
                    },
                )?;
                Ok(Self::Control(mutation))
            }
            2 => {
                let sync = AlrFenceSync::decode_exact(body).map_err(|error| match error {
                    ConsensusCommandError::Truncated
                    | ConsensusCommandError::UnsupportedVersion { .. } => error,
                    other => ConsensusCommandError::BadSync {
                        detail: other.to_string(),
                    },
                })?;
                Ok(Self::ReadSync(sync))
            }
            _ => Err(Fault::BadTag { tag }),
        }
    }
}

impl core::fmt::Display for ConsensusCommand {
    /// Short identity summary. The canonical encoding stays the byte
    /// format above, never this string (`OpenRaft`'s `AppData` bound
    /// requires `Display`).
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Tablet(command) => write!(f, "{command}"),
            Self::Control(_) => write!(f, "control(mutation)"),
            Self::ReadSync(sync) => write!(f, "{sync}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use kivi_state::{Key, Mutation, MutationEnvelope, ObjectVersion, OperationResult};
    use kivi_types::{
        NamespaceId, RequestIdentity, RequestSeq, SessionId, TabletAuthority, TabletEpoch,
        TabletId, WallTimestamp, WriteGuardGeneration,
    };

    use super::*;

    fn tablet_command() -> ConsensusCommand {
        let authority = TabletAuthority::new(
            TabletId::from_u64(9),
            TabletEpoch::INITIAL,
            WriteGuardGeneration::INITIAL,
        );
        let envelope = MutationEnvelope::new(
            NamespaceId::from_u64(1),
            authority,
            RequestIdentity::new(SessionId::from_u128(0x5E55), RequestSeq::from_u64(1)),
            None,
            Mutation::CounterAdd {
                key: Key::from("n"),
                delta: 41,
            },
        );
        ConsensusCommand::Tablet(ReplicatedMutation::new(
            envelope,
            OperationResult::CounterUpdated {
                value: 41,
                version: ObjectVersion::from_u64(7),
            },
            WallTimestamp::from_micros(1_000_000),
            RequestSeq::from_u64(0),
        ))
    }

    #[test]
    fn tablet_command_round_trips_through_envelope() {
        let command = tablet_command();
        let back = ConsensusCommand::decode_exact(&command.encode_to_vec()).expect("decodes");
        assert_eq!(back, command);
        assert!(back.is_tablet());
        assert!(back.as_tablet().is_some());
    }

    #[test]
    fn corrupt_envelope_fails() {
        let good = tablet_command().encode_to_vec();
        assert!(ConsensusCommand::decode_exact(&good[..2]).is_err());
        let mut versioned = good.clone();
        versioned[0] = versioned[0].wrapping_add(1);
        assert!(matches!(
            ConsensusCommand::decode_exact(&versioned),
            Err(ConsensusCommandError::UnsupportedVersion { .. })
        ));
        let mut tagged = good;
        tagged[2] = 0x7F;
        assert!(matches!(
            ConsensusCommand::decode_exact(&tagged),
            Err(ConsensusCommandError::BadTag { .. })
        ));
    }

    fn fence_command() -> ConsensusCommand {
        use kivi_types::{AlrFence, CommitPosition, NodeId};
        ConsensusCommand::ReadSync(AlrFenceSync::new(
            AlrFence::new(TabletId::from_u64(9), 17, NodeId::from_u64(2)),
            CommitPosition::from_u64(41),
        ))
    }

    #[test]
    fn fence_round_trips_and_names_its_kind() {
        let command = fence_command();
        assert!(command.is_read_sync());
        assert!(!command.is_tablet());
        assert!(!command.is_control());
        assert_eq!(
            command.as_read_sync().expect("fence").formation_applied,
            kivi_types::CommitPosition::from_u64(41)
        );
        let back = ConsensusCommand::decode_exact(&command.encode_to_vec()).expect("decodes");
        assert_eq!(back, command);
        // A fence is never mistaken for a write by an old or partial
        // decoder: distinct tag, exact body.
        assert!(back.as_tablet().is_none());
    }

    #[test]
    fn fence_rejects_short_and_versioned_bodies() {
        let good = fence_command().encode_to_vec();
        assert!(ConsensusCommand::decode_exact(&good[..10]).is_err());
        let mut versioned = good.clone();
        // Inner fence version bump: the outer tag decodes, the body
        // refuses loudly instead of misreading positions.
        versioned[3] = versioned[3].wrapping_add(1);
        assert!(ConsensusCommand::decode_exact(&versioned).is_err());
        let mut trailing = good;
        trailing.push(0x00);
        assert!(matches!(
            ConsensusCommand::decode_exact(&trailing),
            Err(ConsensusCommandError::BadSync { .. })
        ));
    }
}
