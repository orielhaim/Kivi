//! Target-neutral workload model.
//!
//! Describes *what* to do, never *how* a database executes it. This module
//! must not know about `kivi-state` internals (chunk ids, manifests),
//! worker routing, or any connection struct — keys
//! are plain strings, values are described by length, and target adapters
//! in [`crate::targets`] own every protocol-specific conversion.
//!
//! Covered semantics: operation mix, key distribution (uniform / hot),
//! value sizes, client/thread count, pipeline depth, op count, and TTL
//! behavior (expiry attach + TTL read inside the `Compat` mix).

/// Shared request mix.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum Workload {
    /// Repeated GETs of a pre-populated keyspace.
    Get,
    /// SETs of values.
    Set,
    /// Counter increments (Kivi-native only; the RESP profile leaves
    /// counters unsupported, so this never runs against RESP targets).
    Counter,
    /// Mixed GET/SET/DELETE/COUNTER across the keyspace (native only).
    Mixed,
    /// RESP-profile mix: GET/SET/EXPIRE/TTL/SETRANGE across the keyspace
    /// (the exact subset both Native and RESP speak).
    Compat,
    /// GETRANGE slices over the keyspace (both targets).
    Range,
}

/// Benchmark value sizes (§15 matrix). Payload bytes are deterministic
/// ([`fill_pattern`]), so both sides store identical content.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum ValueSize {
    /// 16 bytes.
    B16,
    /// 1 KiB.
    Kb1,
    /// 64 KiB.
    Kb64,
    /// 256 KiB.
    Kb256,
    /// 1 MiB.
    Mb1,
    /// 4 MiB.
    Mb4,
    /// 64 MiB (replicated large-value range; streams via `put_stream`).
    Mb64,
}

impl ValueSize {
    /// Length in bytes.
    #[must_use]
    #[allow(clippy::len_without_is_empty)]
    pub const fn len(self) -> usize {
        match self {
            Self::B16 => 16,
            Self::Kb1 => 1024,
            Self::Kb64 => 64 * 1024,
            Self::Kb256 => 256 * 1024,
            Self::Mb1 => 1024 * 1024,
            Self::Mb4 => 4 * 1024 * 1024,
            Self::Mb64 => 64 * 1024 * 1024,
        }
    }
}

/// Deterministic pseudo-random fill (splitmix64): representative,
/// incompressible-ish content — never zeros-only, identical on both sides.
#[must_use]
pub fn fill_pattern(len: usize, seed: u64) -> Vec<u8> {
    let mut state = seed;
    let mut out = Vec::with_capacity(len);
    while out.len() < len {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        out.extend_from_slice(&z.to_le_bytes());
    }
    out.truncate(len);
    out
}

/// Shared key distribution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeySpace {
    /// One hot key shared by all threads.
    Hot,
    /// `count` distinct keys per thread.
    Uniform(usize),
}

/// Parses `hot` or `uniform:<n>` without panicking.
///
/// # Errors
///
/// Returns a description when the spec is neither form or the count is zero.
pub fn parse_key_space(spec: &str) -> Result<KeySpace, String> {
    if spec == "hot" {
        Ok(KeySpace::Hot)
    } else if let Some(count) = spec.strip_prefix("uniform:") {
        let count: usize = count
            .parse()
            .map_err(|_| format!("invalid key count in {spec:?}"))?;
        if count > 0 {
            Ok(KeySpace::Uniform(count))
        } else {
            Err(format!("key count must be nonzero in {spec:?}"))
        }
    } else {
        Err(format!("keys must be `hot` or `uniform:<n>`, got {spec:?}"))
    }
}

/// One target-agnostic workload operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WorkloadOp {
    /// Read a key.
    Get(String),
    /// Write `len` bytes.
    Set {
        /// Key to write.
        key: String,
        /// Value length in bytes.
        len: usize,
    },
    /// Add 1 to a counter (native only).
    CounterAdd(String),
    /// Remove a key.
    Delete(String),
    /// Attach a 60-second expiry.
    Expire(String),
    /// Read expiry/TTL.
    Ttl(String),
    /// Patch one byte at offset 0.
    SetRange(String),
    /// Slice bytes `[0, 64)` (clamped by the store).
    GetRange(String),
}

/// Fixed GETRANGE window length for the `Range` workload.
pub const RANGE_WINDOW: u64 = 64;

/// Plain key names for one thread's byte keyspace, under `prefix` (every
/// benchmark/conformance run isolates its keys; never bare shared names
/// against an externally supplied instance). Counter keys stay separate
/// ([`counter_key_name`]): mixing types on one key is `WrongType` by
/// design, so workloads seed and hammer the type they measure.
#[must_use]
pub fn thread_key_names(space: KeySpace, thread: usize, prefix: &str) -> Vec<String> {
    match space {
        KeySpace::Hot => vec![format!("{prefix}hot")],
        KeySpace::Uniform(count) => (0..count)
            .map(|i| format!("{prefix}t{thread}:k{i}"))
            .collect(),
    }
}

/// Dedicated per-thread counter key name (never shares with byte keys).
#[must_use]
pub fn counter_key_name(space: KeySpace, thread: usize, prefix: &str) -> String {
    match space {
        KeySpace::Hot => format!("{prefix}hot:c"),
        KeySpace::Uniform(_) => format!("{prefix}t{thread}:c"),
    }
}

/// Pure workload generation: the `i`-th op of `workload` over `keys`.
/// No I/O, no client types — identical for every target adapter.
#[must_use]
pub fn workload_op_for(
    workload: Workload,
    value_len: usize,
    i: usize,
    keys: &[String],
    counter: &str,
) -> WorkloadOp {
    let key = keys[i % keys.len()].clone();
    match workload {
        Workload::Get => WorkloadOp::Get(key),
        Workload::Set => WorkloadOp::Set {
            key,
            len: value_len,
        },
        Workload::Counter => WorkloadOp::CounterAdd(counter.to_owned()),
        Workload::Mixed => match i % 4 {
            0 => WorkloadOp::Get(key),
            1 => WorkloadOp::Set {
                key,
                len: value_len,
            },
            2 => WorkloadOp::CounterAdd(counter.to_owned()),
            _ => WorkloadOp::Delete(key),
        },
        Workload::Compat => match i % 5 {
            0 => WorkloadOp::Get(key),
            1 => WorkloadOp::Set {
                key,
                len: value_len,
            },
            2 => WorkloadOp::Expire(key),
            3 => WorkloadOp::Ttl(key),
            _ => WorkloadOp::SetRange(key),
        },
        Workload::Range => WorkloadOp::GetRange(key),
    }
}

/// Seed operations for one thread's keyspace: byte keys first (in key
/// order) at `value_len`, then the counter key. Adapters execute these
/// before measuring.
#[must_use]
pub fn seed_ops_for(
    space: KeySpace,
    thread: usize,
    workload: Workload,
    value_len: usize,
    prefix: &str,
) -> Vec<WorkloadOp> {
    let mut ops = Vec::new();
    let seed_bytes = matches!(
        workload,
        Workload::Get | Workload::Set | Workload::Mixed | Workload::Compat | Workload::Range
    );
    let seed_counters = matches!(workload, Workload::Counter | Workload::Mixed);
    if seed_bytes {
        if matches!(space, KeySpace::Hot) {
            if thread == 0 {
                ops.push(WorkloadOp::Set {
                    key: format!("{prefix}hot"),
                    len: value_len,
                });
            }
        } else {
            for key in thread_key_names(space, thread, prefix) {
                ops.push(WorkloadOp::Set {
                    key,
                    len: value_len,
                });
            }
        }
    }
    if seed_counters {
        ops.push(WorkloadOp::CounterAdd(counter_key_name(
            space, thread, prefix,
        )));
    }
    ops
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[derive(Debug, Parser)]
    struct Probe {
        #[arg(value_enum)]
        workload: Workload,
        #[arg(value_enum)]
        size: ValueSize,
    }

    #[test]
    fn enums_parse_from_cli() {
        let probe = Probe::try_parse_from(["probe", "compat", "kb64"]).expect("parses");
        assert_eq!(probe.workload, Workload::Compat);
        assert_eq!(probe.size, ValueSize::Kb64);
        assert_eq!(ValueSize::Mb4.len(), 4 * 1024 * 1024);
    }

    #[test]
    fn key_space_parser_accepts_hot_and_uniform() {
        assert_eq!(parse_key_space("hot"), Ok(KeySpace::Hot));
        assert_eq!(parse_key_space("uniform:16"), Ok(KeySpace::Uniform(16)));
        assert!(parse_key_space("uniform:0").is_err());
        assert!(parse_key_space("uniform:lots").is_err());
        assert!(parse_key_space("cold").is_err());
    }

    #[test]
    fn fill_pattern_is_deterministic_and_sized() {
        assert_eq!(fill_pattern(16, 7), fill_pattern(16, 7));
        assert_ne!(fill_pattern(16, 7), fill_pattern(16, 8));
        assert_eq!(fill_pattern(100, 1).len(), 100);
        assert!(fill_pattern(64, 1).iter().any(|byte| *byte != 0));
    }

    #[test]
    fn workload_mix_matches_legacy_pattern() {
        let keys = vec!["a".to_owned(), "b".to_owned()];
        let counter = "c";
        assert_eq!(
            workload_op_for(Workload::Get, 1, 0, &keys, counter),
            WorkloadOp::Get("a".to_owned())
        );
        assert_eq!(
            workload_op_for(Workload::Set, 1, 1, &keys, counter),
            WorkloadOp::Set {
                key: "b".to_owned(),
                len: 1
            }
        );
        assert_eq!(
            workload_op_for(Workload::Counter, 1, 7, &keys, counter),
            WorkloadOp::CounterAdd("c".to_owned())
        );
        assert_eq!(
            workload_op_for(Workload::Mixed, 1, 3, &keys, counter),
            WorkloadOp::Delete("b".to_owned())
        );
        assert_eq!(
            workload_op_for(Workload::Compat, 1, 4, &keys, counter),
            WorkloadOp::SetRange("a".to_owned())
        );
        assert_eq!(
            workload_op_for(Workload::Range, 1, 9, &keys, counter),
            WorkloadOp::GetRange("b".to_owned())
        );
    }

    #[test]
    fn seed_ops_cover_measured_types_only() {
        assert_eq!(
            seed_ops_for(KeySpace::Hot, 0, Workload::Get, 1, ""),
            vec![WorkloadOp::Set {
                key: "hot".to_owned(),
                len: 1
            }]
        );
        assert!(seed_ops_for(KeySpace::Hot, 1, Workload::Get, 1, "").is_empty());
        assert_eq!(
            seed_ops_for(KeySpace::Hot, 0, Workload::Counter, 1, ""),
            vec![WorkloadOp::CounterAdd("hot:c".to_owned())]
        );
    }

    #[test]
    fn workload_key_names_match_legacy_layout() {
        assert_eq!(
            thread_key_names(KeySpace::Hot, 3, ""),
            vec!["hot".to_owned()]
        );
        assert_eq!(
            thread_key_names(KeySpace::Uniform(2), 1, ""),
            vec!["t1:k0".to_owned(), "t1:k1".to_owned()]
        );
        assert_eq!(counter_key_name(KeySpace::Hot, 0, ""), "hot:c");
        assert_eq!(counter_key_name(KeySpace::Uniform(9), 2, ""), "t2:c");
    }

    #[test]
    fn key_prefix_isolates_runs() {
        assert_eq!(
            thread_key_names(KeySpace::Uniform(1), 0, "run-1:"),
            vec!["run-1:t0:k0".to_owned()]
        );
        assert_eq!(
            seed_ops_for(KeySpace::Hot, 0, Workload::Get, 1, "run-1:"),
            vec![WorkloadOp::Set {
                key: "run-1:hot".to_owned(),
                len: 1
            }]
        );
    }
}
