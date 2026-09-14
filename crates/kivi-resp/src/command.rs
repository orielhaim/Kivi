//! Redis command registry: the single source of compatibility truth.
//!
//! Every command Kivi's RESP edge names appears here exactly once with its
//! compatibility class. Dispatch validation, `COMMAND` output, and the
//! published support table all read from [`REGISTRY`], so documentation
//! cannot drift from implementation.

/// Compatibility class of one Redis command in the Kivi RESP profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CompatClass {
    /// Exact Redis semantics (edge cases verified against official docs).
    Exact,
    /// Supported with a documented Kivi-profile deviation.
    ProfileDeviation,
    /// Recognized but explicitly unsupported (clear error, no fake).
    Unsupported,
}

impl core::fmt::Display for CompatClass {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Exact => write!(f, "exact"),
            Self::ProfileDeviation => write!(f, "profile-deviation"),
            Self::Unsupported => write!(f, "unsupported"),
        }
    }
}

/// Whether a command reads, writes, or manages the connection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CommandKind {
    /// Reads state (never enters the WAL).
    Read,
    /// May change state (enters the WAL on success).
    Write,
    /// Connection handling (no key access).
    Connection,
    /// Introspection (no key access).
    Introspection,
}

/// One command's static metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CommandSpec {
    /// Uppercase ASCII command name (`GET`, `CLIENT`, ...).
    pub name: &'static str,
    /// Minimum argument count including the command name.
    pub min_arity: usize,
    /// Maximum argument count including the command name (`None` = open).
    pub max_arity: Option<usize>,
    /// Read/write/connection classification.
    pub kind: CommandKind,
    /// Compatibility class in this profile.
    pub compat: CompatClass,
    /// One-line note (exact edge or deviation reason).
    pub notes: &'static str,
}

/// The full profile registry. Sorted by name for stable `COMMAND` output.
pub const REGISTRY: &[CommandSpec] = &[
    CommandSpec {
        name: "AUTH",
        min_arity: 2,
        max_arity: Some(3),
        kind: CommandKind::Connection,
        compat: CompatClass::Unsupported,
        notes: "no ACL system in this stage; always errors clearly, never succeeds silently",
    },
    CommandSpec {
        name: "CLIENT",
        min_arity: 2,
        max_arity: None,
        kind: CommandKind::Connection,
        compat: CompatClass::Exact,
        notes: "SETINFO/SETNAME/GETNAME accepted; other subcommands rejected",
    },
    CommandSpec {
        name: "COMMAND",
        min_arity: 1,
        max_arity: None,
        kind: CommandKind::Introspection,
        compat: CompatClass::ProfileDeviation,
        notes: "COUNT/INFO/DOCS/LIST served from this registry; full Redis metadata universe not implemented",
    },
    CommandSpec {
        name: "DECR",
        min_arity: 2,
        max_arity: Some(2),
        kind: CommandKind::Write,
        compat: CompatClass::Unsupported,
        notes: "StrictCounter is not Redis String-ints; atomic GET-parse-SET forbidden",
    },
    CommandSpec {
        name: "DECRBY",
        min_arity: 3,
        max_arity: Some(3),
        kind: CommandKind::Write,
        compat: CompatClass::Unsupported,
        notes: "StrictCounter is not Redis String-ints; atomic GET-parse-SET forbidden",
    },
    CommandSpec {
        name: "DEL",
        min_arity: 2,
        max_arity: Some(2),
        kind: CommandKind::Write,
        compat: CompatClass::Exact,
        notes: "single-key only; multi-key forms rejected, never faked",
    },
    CommandSpec {
        name: "ECHO",
        min_arity: 2,
        max_arity: Some(2),
        kind: CommandKind::Connection,
        compat: CompatClass::Exact,
        notes: "binary-safe echo",
    },
    CommandSpec {
        name: "EXISTS",
        min_arity: 2,
        max_arity: Some(2),
        kind: CommandKind::Read,
        compat: CompatClass::Exact,
        notes: "single-key only; multi-key forms rejected, never faked",
    },
    CommandSpec {
        name: "EXPIRE",
        min_arity: 3,
        max_arity: Some(3),
        kind: CommandKind::Write,
        compat: CompatClass::Exact,
        notes: "relative seconds, bare form only; NX/XX/GT/LT error clearly instead of racing",
    },
    CommandSpec {
        name: "EXPIREAT",
        min_arity: 3,
        max_arity: Some(3),
        kind: CommandKind::Write,
        compat: CompatClass::Exact,
        notes: "absolute unix seconds, bare form only; options error clearly instead of racing",
    },
    CommandSpec {
        name: "EXPIRETIME",
        min_arity: 2,
        max_arity: Some(2),
        kind: CommandKind::Read,
        compat: CompatClass::Exact,
        notes: "absolute expiry in seconds; -1 immortal, -2 missing",
    },
    CommandSpec {
        name: "GET",
        min_arity: 2,
        max_arity: Some(2),
        kind: CommandKind::Read,
        compat: CompatClass::Exact,
        notes: "missing answers nil; counters answer WRONGTYPE",
    },
    CommandSpec {
        name: "GETRANGE",
        min_arity: 4,
        max_arity: Some(4),
        kind: CommandKind::Read,
        compat: CompatClass::Exact,
        notes: "inclusive start/end, negatives from end, clamped; missing answers empty string",
    },
    CommandSpec {
        name: "HELLO",
        min_arity: 1,
        max_arity: None,
        kind: CommandKind::Connection,
        compat: CompatClass::ProfileDeviation,
        notes: "2/3 switch response encoding; AUTH always errors clearly; SETNAME honored",
    },
    CommandSpec {
        name: "INCR",
        min_arity: 2,
        max_arity: Some(2),
        kind: CommandKind::Write,
        compat: CompatClass::Unsupported,
        notes: "StrictCounter is not Redis String-ints; atomic GET-parse-SET forbidden",
    },
    CommandSpec {
        name: "INCRBY",
        min_arity: 3,
        max_arity: Some(3),
        kind: CommandKind::Write,
        compat: CompatClass::Unsupported,
        notes: "StrictCounter is not Redis String-ints; atomic GET-parse-SET forbidden",
    },
    CommandSpec {
        name: "MGET",
        min_arity: 2,
        max_arity: None,
        kind: CommandKind::Read,
        compat: CompatClass::Unsupported,
        notes: "cross-tablet atomicity not faked with a read loop",
    },
    CommandSpec {
        name: "MSET",
        min_arity: 3,
        max_arity: None,
        kind: CommandKind::Write,
        compat: CompatClass::Unsupported,
        notes: "cross-tablet atomicity not faked with a write loop",
    },
    CommandSpec {
        name: "PERSIST",
        min_arity: 2,
        max_arity: Some(2),
        kind: CommandKind::Write,
        compat: CompatClass::Exact,
        notes: "1 when an expiry was removed, 0 when absent or immortal",
    },
    CommandSpec {
        name: "PEXPIRE",
        min_arity: 3,
        max_arity: Some(3),
        kind: CommandKind::Write,
        compat: CompatClass::Exact,
        notes: "relative milliseconds, bare form only; options error clearly instead of racing",
    },
    CommandSpec {
        name: "PEXPIREAT",
        min_arity: 3,
        max_arity: Some(3),
        kind: CommandKind::Write,
        compat: CompatClass::Exact,
        notes: "absolute unix milliseconds, bare form only; options error clearly instead of racing",
    },
    CommandSpec {
        name: "PEXPIRETIME",
        min_arity: 2,
        max_arity: Some(2),
        kind: CommandKind::Read,
        compat: CompatClass::Exact,
        notes: "absolute expiry in ms; -1 immortal, -2 missing",
    },
    CommandSpec {
        name: "PING",
        min_arity: 1,
        max_arity: Some(2),
        kind: CommandKind::Connection,
        compat: CompatClass::Exact,
        notes: "PONG or echo",
    },
    CommandSpec {
        name: "PTTL",
        min_arity: 2,
        max_arity: Some(2),
        kind: CommandKind::Read,
        compat: CompatClass::Exact,
        notes: "-2 missing, -1 immortal, else ms remaining",
    },
    CommandSpec {
        name: "QUIT",
        min_arity: 1,
        max_arity: Some(1),
        kind: CommandKind::Connection,
        compat: CompatClass::Exact,
        notes: "+OK then close",
    },
    CommandSpec {
        name: "SELECT",
        min_arity: 2,
        max_arity: Some(2),
        kind: CommandKind::Connection,
        compat: CompatClass::ProfileDeviation,
        notes: "0 maps to the configured namespace; others rejected, never faked",
    },
    CommandSpec {
        name: "SET",
        min_arity: 3,
        max_arity: None,
        kind: CommandKind::Write,
        compat: CompatClass::Exact,
        notes: "NX/XX + EX/PX/EXAT/PXAT/KEEPTTL atomic; GET/IFEQ family rejected",
    },
    CommandSpec {
        name: "SETRANGE",
        min_arity: 4,
        max_arity: Some(4),
        kind: CommandKind::Write,
        compat: CompatClass::Exact,
        notes: "zero-pads, answers new length; counters WRONGTYPE",
    },
    CommandSpec {
        name: "STRLEN",
        min_arity: 2,
        max_arity: Some(2),
        kind: CommandKind::Read,
        compat: CompatClass::Exact,
        notes: "missing answers 0 without error; counters WRONGTYPE",
    },
    CommandSpec {
        name: "TTL",
        min_arity: 2,
        max_arity: Some(2),
        kind: CommandKind::Read,
        compat: CompatClass::Exact,
        notes: "-2 missing, -1 immortal, else seconds remaining",
    },
];

/// Looks up a command by uppercase ASCII name.
#[must_use]
pub fn lookup(name: &str) -> Option<&'static CommandSpec> {
    REGISTRY.iter().find(|spec| spec.name == name)
}

/// Checks arity against the spec (`true` when the count fits).
#[must_use]
pub fn arity_ok(spec: &CommandSpec, argc: usize) -> bool {
    argc >= spec.min_arity && spec.max_arity.is_none_or(|max| argc <= max)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn registry_names_are_unique_sorted_and_sane() {
        let mut seen = HashSet::new();
        let mut previous: Option<&str> = None;
        for spec in REGISTRY {
            assert!(seen.insert(spec.name), "duplicate command {}", spec.name);
            assert!(
                spec.name.bytes().all(|byte| byte.is_ascii_uppercase()),
                "command names are uppercase ASCII: {}",
                spec.name
            );
            assert!(spec.min_arity >= 1, "arity starts at the command name");
            if let Some(max) = spec.max_arity {
                assert!(max >= spec.min_arity, "max arity covers min");
            }
            assert!(!spec.notes.is_empty(), "every command documents its edge");
            if let Some(prev) = previous {
                assert!(
                    prev < spec.name,
                    "registry sorted for stable COMMAND output: {prev} before {}",
                    spec.name
                );
            }
            previous = Some(spec.name);
        }
    }

    #[test]
    fn every_spec_lookup_round_trips() {
        for spec in REGISTRY {
            assert_eq!(lookup(spec.name), Some(spec));
            assert!(arity_ok(spec, spec.min_arity));
        }
        assert_eq!(lookup("NOPE"), None);
    }
}
