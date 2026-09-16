//! Unified consensus command envelope.
//!
//! One Kivi-owned envelope covers both tablet writes and control-plane
//! writes, so a single `OpenRaft` type configuration and a single set of
//! storage/transport/codec implementations serve every group:
//!
//! ```text
//! ConsensusCommand {
//!     Tablet(ReplicatedMutation),
//!     Control(ControlMutation),
//! }
//! ```
//!
//! Tablet groups apply `Tablet` through the replicated state machine;
//! the system control group applies `Control` through replicated control
//! state. There is exactly one `OpenRaft`↔Kivi boundary (this type plus
//! [`ReplicatedOutcome`](crate::mutation::ReplicatedOutcome)); no
//! parallel codec/conversion implementations exist.

use crate::mutation::{ReplicatedMutation, ReplicatedMutationError};

/// Framing version of [`ConsensusCommand`].
pub const CONSENSUS_COMMAND_VERSION: u16 = 1;

/// What gets proposed, committed, and applied in any group: either a
/// deterministic tablet command or a typed control-plane mutation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConsensusCommand {
    /// Deterministic tablet write (normal data path).
    Tablet(ReplicatedMutation),
    /// Typed control-plane mutation (system control group only).
    Control(kivi_control::ControlMutation),
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
            Self::Control(_) => None,
        }
    }

    /// Returns the control mutation, if this is one.
    #[must_use]
    pub const fn as_control(&self) -> Option<&kivi_control::ControlMutation> {
        match self {
            Self::Tablet(_) => None,
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

    /// Encodes the canonical bytes: `version u16, tag u8,
    /// body = inner canonical bytes`.
    ///
    /// The inner bodies are self-framed (their own version/length
    /// prefixes), so no outer length prefix is needed: the tag selects
    /// the inner decoder and the inner decoder enforces exactness.
    #[must_use]
    pub fn encode_to_vec(&self) -> Vec<u8> {
        let (tag, body) = match self {
            Self::Tablet(command) => (0u8, command.encode_to_vec()),
            Self::Control(mutation) => (1u8, mutation.encode_to_vec()),
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
        }
    }
}

#[cfg(test)]
mod tests {
    use kivi_state::{Key, Mutation, MutationEnvelope, ObjectVersion, OperationResult};
    use kivi_types::{
        NamespaceId, RequestIdentity, RequestSeq, SessionId, TabletAuthority, TabletEpoch,
        TabletId, UnixMicros, WriteGuardGeneration,
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
            UnixMicros::from_micros(1_000_000),
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
}
