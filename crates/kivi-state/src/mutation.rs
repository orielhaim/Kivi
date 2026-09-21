//! Typed Mutation IR: the canonical replicated/persisted semantics.
//!
//! User commands are never replicated as-is. [`prepare`](crate::ObjectStore::prepare)
//! normalizes each write into a [`Mutation`] carrying everything a replica
//! needs for deterministic replay: keys, values, deltas, and expiries —
//! never wall-clock reads, randomness, or local decisions. [`CounterAdd`](Mutation::CounterAdd)
//! deterministically derives its resulting value from prior ordered state on
//! every replica identically.
//!
//! The binary form below is project-owned (little-endian, versioned tags,
//! length-prefixed blobs) and part of the durable contract. No serde,
//! bincode, or rkyv type defines it.

use kivi_codec::{CodecError, Decode, Encode, decode_byte_vec, encode_bytes};
use kivi_types::{Expiry, ManifestId};

use crate::object::{Key, ObjectType, ObjectVersion};
use crate::txn::{TxnExpect, TxnId, TxnWrite, TxnWriteKind};

/// One deterministic state transition.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Mutation {
    /// Store bytes under `key`, overwriting any type and clearing expiry.
    PutBytes {
        /// Target key.
        key: Key,
        /// Value to store.
        value: bytes::Bytes,
    },
    /// Remove `key`, reporting whether a live object existed.
    Delete {
        /// Target key.
        key: Key,
    },
    /// Add `delta` to a counter, creating it at `delta` when absent.
    CounterAdd {
        /// Target key.
        key: Key,
        /// Signed addend.
        delta: i64,
    },
    /// Attach (`Some`) or clear (`None`) an absolute expiry on a live key.
    SetExpiry {
        /// Target key.
        key: Key,
        /// New expiry (`NEVER` clears).
        expiry: Expiry,
    },
    /// Point `key` at a chunked value: the manifest addressing its
    /// immutable chunks plus the total logical length. The WAL carries
    /// this small root transition only — bulk bytes live in chunk packs
    /// whose durability precedes this record (§11, §12). Overwrites any
    /// type and clears expiry, exactly like [`PutBytes`](Self::PutBytes).
    ReplaceChunkedRoot {
        /// Target key.
        key: Key,
        /// Manifest addressing the immutable chunk sequence.
        manifest: ManifestId,
        /// Total logical bytes across the manifest.
        logical_len: u64,
    },
    /// Point `key` at a Memory Fabric value: the fabric-scoped object id
    /// plus the total logical length and the published version. The WAL
    /// carries this small root transition only — staged bytes live in the
    /// fabric whose durability precedes this record (same chunks-first
    /// rule as [`ReplaceChunkedRoot`](Self::ReplaceChunkedRoot)).
    /// Overwrites any type and clears expiry, exactly like
    /// [`PutBytes`](Self::PutBytes).
    ReplaceFabricRoot {
        /// Target key.
        key: Key,
        /// Fabric-scoped object id assigned at staging.
        fabric_id: u64,
        /// Total logical bytes of the materialized value.
        logical_len: u64,
        /// Logical version the bytes were published at (pins reads).
        version: u64,
    },
    /// Patch a byte range of `key`'s value at `offset` with `patch`,
    /// zero-padding past-the-end gaps (Redis `SETRANGE` semantics). Unlike
    /// [`PutBytes`](Self::PutBytes), a splice preserves the live expiry:
    /// partial writes touch bytes, never the TTL. The tablet layer only
    /// admits this against absent, expired, or inline bases whose patched
    /// result stays within the legacy inline bound; chunked bases are
    /// resolved and restaged by the engine first (with the preserved
    /// expiry), so apply meets no chunk reference it cannot read. Replay is
    /// deterministic: the same ordered state yields the same base, hence
    /// the same splice.
    SpliceBytes {
        /// Target key.
        key: Key,
        /// Logical byte index the patch overwrites from.
        offset: u64,
        /// Bytes written starting at `offset`.
        patch: bytes::Bytes,
    },
    /// Store bytes under `key` with an explicit expiry, produced only by
    /// conditional sets whose policy resolved to a concrete stamp at
    /// preparation time (`Clear` resolves to `NEVER`, `Keep` to the live
    /// expiry or `NEVER`, `ExpireAt` to its stamp). Overwrites any type.
    /// The expiry rides in the mutation so replay applies the same stamp
    /// without re-reading state.
    PutBytesWithExpiry {
        /// Target key.
        key: Key,
        /// Value to store.
        value: bytes::Bytes,
        /// Resolved expiry to attach.
        expiry: Expiry,
    },
    /// Point `key` at a chunked value with an explicit expiry: the chunked
    /// spelling of [`PutBytesWithExpiry`](Self::PutBytesWithExpiry) for
    /// conditional stores whose staged value exceeded the inline bound.
    ReplaceChunkedRootWithExpiry {
        /// Target key.
        key: Key,
        /// Manifest addressing the immutable chunk sequence.
        manifest: ManifestId,
        /// Total logical bytes across the manifest.
        logical_len: u64,
        /// Resolved expiry to attach.
        expiry: Expiry,
    },
    /// Point `key` at a fabric value with an explicit expiry: the fabric
    /// spelling of [`PutBytesWithExpiry`](Self::PutBytesWithExpiry) for
    /// conditional stores whose staged value exceeded the inline bound.
    ReplaceFabricRootWithExpiry {
        /// Target key.
        key: Key,
        /// Fabric-scoped object id assigned at staging.
        fabric_id: u64,
        /// Total logical bytes of the materialized value.
        logical_len: u64,
        /// Logical version the bytes were published at (pins reads).
        version: u64,
        /// Resolved expiry to attach.
        expiry: Expiry,
    },
    /// Reserve `key` for transaction `txn`: validates the OCC expectation
    /// and records a durable intent without touching user-visible state.
    /// Single-key like every other variant (`key` is the reserved key), so
    /// overlays, dirty bands, and sweeps need no multi-key contract.
    TxnPrepare {
        /// Transaction identity (idempotency key for the reservation).
        txn: TxnId,
        /// Coordinator tablet owning the decision record.
        coordinator: u64,
        /// Reserved key.
        key: Key,
        /// OCC version expectation.
        expect: TxnExpect,
        /// Prepared write (applied at finalize-commit).
        write: TxnWriteKind,
        /// Digest over the full write set (confused-deputy binding: a
        /// different digest under one `TxnId` conflicts, never overwrites).
        digest: [u8; 32],
    },
    /// Resolve one key's intent for `txn`: commit applies the prepared
    /// write, abort discards it. Idempotent: a missing intent (already
    /// finalized or never prepared here) answers finalized-not-applied.
    TxnFinalize {
        /// Transaction identity.
        txn: TxnId,
        /// Key whose intent resolves.
        key: Key,
        /// Whether to apply (`true`) or discard (`false`) the intent.
        commit: bool,
        /// Digest the intent was prepared for (must match the reservation).
        digest: [u8; 32],
    },
    /// Commit a same-tablet write set as one ordered atomic record: every
    /// write validates and applies together, or nothing does. The
    /// single-tablet cheap path persists exactly one of these per
    /// transaction (no prepare/record/finalize waves); replay applies it
    /// through the same [`commit_local`](crate::ObjectStore::commit_local)
    /// path, so replicas converge without re-planning. Always non-empty
    /// (decode rejects empty sets loudly).
    TxnCommitLocal {
        /// Transaction identity (idempotency key for the whole batch).
        txn: TxnId,
        /// Writes in request order (duplicates collapse last-wins).
        writes: Vec<TxnWrite>,
    },
    /// Add `delta` to a commutative counter, creating it at `delta` when
    /// absent. No ordinal is produced: the outcome carries the post-apply
    /// sum only for durability prediction, and the client-visible result is
    /// `Applied`.
    CommutativeAdd {
        /// Target key.
        key: Key,
        /// Signed addend.
        delta: i64,
    },
    /// Create a bounded counter with global `capacity` and full local share
    /// `[0, capacity]` held by `holder` (the executing tablet,
    /// leader-materialized at prepare like all deterministic inputs).
    BoundedCreate {
        /// Target key.
        key: Key,
        /// Global capacity (inclusive upper bound).
        capacity: u64,
        /// Tablet holding the initial full share.
        holder: u64,
    },
    /// Add `delta` to a bounded counter within locally owned rights.
    BoundedAdd {
        /// Target key.
        key: Key,
        /// Signed addend.
        delta: i64,
    },
    /// Durably move one bounded counter's escrow share to `[min, max]`.
    /// Narrowing gives rights away unilaterally; widening must pair with a
    /// matching narrow in the same atomic batch/transaction (conservation
    /// checked by the driver), so rights are never minted.
    EscrowSetShare {
        /// Target key.
        key: Key,
        /// New share lower bound (inclusive).
        min: i64,
        /// New share upper bound (inclusive).
        max: i64,
    },
    /// Create a semaphore with `capacity` total permits.
    SemaphoreCreate {
        /// Target key.
        key: Key,
        /// Total capacity.
        capacity: u64,
    },
    /// Acquire `qty` permits under `permit`: idempotent by permit id.
    SemaphoreAcquire {
        /// Target key.
        key: Key,
        /// Permit identity (stable across retries).
        permit: [u8; 16],
        /// Owner/session identity acquiring.
        owner: u64,
        /// Quantity to acquire.
        qty: u64,
    },
    /// Release the permits held under `permit`: idempotent, mints nothing.
    SemaphoreRelease {
        /// Target key.
        key: Key,
        /// Permit identity to release.
        permit: [u8; 16],
    },
    /// Grant the lease to `owner` under the resolved fencing token until
    /// the resolved expiry. Both are fixed at prepare (deterministic
    /// inputs), so replay issues the identical grant.
    LeaseAcquire {
        /// Target key.
        key: Key,
        /// Owner/session identity acquiring.
        owner: u64,
        /// Fencing token issued with this grant.
        fencing: crate::FencingToken,
        /// Logical expiry of the grant.
        expires_at: kivi_types::WallTimestamp,
    },
    /// Extend a live grant to the resolved expiry (same fencing token).
    LeaseRenew {
        /// Target key.
        key: Key,
        /// Owner/session identity renewing.
        owner: u64,
        /// Fencing token of the grant being renewed.
        fencing: crate::FencingToken,
        /// New logical expiry of the grant.
        expires_at: kivi_types::WallTimestamp,
    },
    /// Release the grant named by (`owner`, `fencing`).
    LeaseRelease {
        /// Target key.
        key: Key,
        /// Owner/session identity releasing.
        owner: u64,
        /// Fencing token of the grant being released.
        fencing: crate::FencingToken,
    },
    /// Create one shard of a sharded stream.
    StreamCreate {
        /// Key holding this shard's log.
        key: Key,
        /// Stream identity (shared by all shards of the stream).
        stream: [u8; 16],
        /// Shard index within the stream.
        shard: u32,
    },
    /// Append one entry to a shard (offset assigned deterministically at
    /// apply from the shard's `next_offset`).
    StreamAppend {
        /// Key holding this shard's log.
        key: Key,
        /// Partition key that routed this entry here.
        partition: bytes::Bytes,
        /// Entry payload bytes.
        payload: bytes::Bytes,
    },
    /// Trim a shard's retained prefix through `through` (inclusive).
    StreamTrim {
        /// Key holding this shard's log.
        key: Key,
        /// Trim retained entries with `offset <= through`.
        through: u64,
    },
}

/// Canonical wire tags. Fixed forever within framing version 1; new
/// operations take new tags, never reuse.
const TAG_PUT_BYTES: u8 = 1;
const TAG_DELETE: u8 = 2;
const TAG_COUNTER_ADD: u8 = 3;
const TAG_SET_EXPIRY: u8 = 4;
/// Chunked-root mutation tag. New tags never reuse old ones.
const TAG_REPLACE_CHUNKED_ROOT: u8 = 5;
/// Byte-range splice mutation tag. New tags never reuse old ones.
const TAG_SPLICE_BYTES: u8 = 6;
/// Conditional inline-store mutation tag. New tags never reuse old ones.
const TAG_PUT_BYTES_WITH_EXPIRY: u8 = 7;
/// Conditional chunked-root mutation tag. New tags never reuse old ones.
const TAG_REPLACE_CHUNKED_ROOT_WITH_EXPIRY: u8 = 8;
/// Transaction prepare (durable intent reservation) tag.
const TAG_TXN_PREPARE: u8 = 9;
/// Transaction per-key finalize tag.
const TAG_TXN_FINALIZE: u8 = 10;
/// Commutative-counter add tag. New tags never reuse old ones.
const TAG_COMMUTATIVE_ADD: u8 = 11;
/// Bounded-counter create tag. New tags never reuse old ones.
const TAG_BOUNDED_CREATE: u8 = 12;
/// Bounded-counter add tag. New tags never reuse old ones.
const TAG_BOUNDED_ADD: u8 = 13;
/// Escrow share-move tag. New tags never reuse old ones.
const TAG_ESCROW_SET_SHARE: u8 = 14;
/// Semaphore create tag. New tags never reuse old ones.
const TAG_SEMAPHORE_CREATE: u8 = 15;
/// Semaphore acquire tag. New tags never reuse old ones.
const TAG_SEMAPHORE_ACQUIRE: u8 = 16;
/// Semaphore release tag. New tags never reuse old ones.
const TAG_SEMAPHORE_RELEASE: u8 = 17;
/// Lease acquire tag. New tags never reuse old ones.
const TAG_LEASE_ACQUIRE: u8 = 18;
/// Lease renew tag. New tags never reuse old ones.
const TAG_LEASE_RENEW: u8 = 19;
/// Lease release tag. New tags never reuse old ones.
const TAG_LEASE_RELEASE: u8 = 20;
/// Stream-shard create tag. New tags never reuse old ones.
const TAG_STREAM_CREATE: u8 = 21;
/// Stream append tag. New tags never reuse old ones.
const TAG_STREAM_APPEND: u8 = 22;
/// Stream trim tag. New tags never reuse old ones.
const TAG_STREAM_TRIM: u8 = 23;
/// Same-tablet atomic commit tag. New tags never reuse old ones.
const TAG_TXN_COMMIT_LOCAL: u8 = 24;
/// Fabric-root mutation tag. New tags never reuse old ones.
const TAG_REPLACE_FABRIC_ROOT: u8 = 25;
/// Conditional fabric-root mutation tag. New tags never reuse old ones.
const TAG_REPLACE_FABRIC_ROOT_WITH_EXPIRY: u8 = 26;

impl Mutation {
    /// Returns the single key this mutation touches. Every mutation is
    /// single-key except [`TxnCommitLocal`](Self::TxnCommitLocal), which
    /// answers its first write's key (routing anchor; all keys share one
    /// tablet by driver construction). Callers that must cover every key
    /// (dirty bands, sweeps, overlays) use [`keys`](Self::keys) instead.
    ///
    /// # Panics
    ///
    /// Panics on [`TxnCommitLocal`](Self::TxnCommitLocal) with an empty
    /// write list: drivers never construct one (admission and the codec
    /// reject empties), so an empty commit here is a caller bug.
    #[must_use]
    pub fn key(&self) -> &Key {
        match self {
            Self::PutBytes { key, .. }
            | Self::Delete { key }
            | Self::CounterAdd { key, .. }
            | Self::SetExpiry { key, .. }
            | Self::ReplaceChunkedRoot { key, .. }
            | Self::ReplaceFabricRoot { key, .. }
            | Self::SpliceBytes { key, .. }
            | Self::PutBytesWithExpiry { key, .. }
            | Self::ReplaceChunkedRootWithExpiry { key, .. }
            | Self::ReplaceFabricRootWithExpiry { key, .. }
            | Self::TxnPrepare { key, .. }
            | Self::TxnFinalize { key, .. }
            | Self::CommutativeAdd { key, .. }
            | Self::BoundedCreate { key, .. }
            | Self::BoundedAdd { key, .. }
            | Self::EscrowSetShare { key, .. }
            | Self::SemaphoreCreate { key, .. }
            | Self::SemaphoreAcquire { key, .. }
            | Self::SemaphoreRelease { key, .. }
            | Self::LeaseAcquire { key, .. }
            | Self::LeaseRenew { key, .. }
            | Self::LeaseRelease { key, .. }
            | Self::StreamCreate { key, .. }
            | Self::StreamAppend { key, .. }
            | Self::StreamTrim { key, .. } => key,
            Self::TxnCommitLocal { writes, .. } => {
                &writes.first().expect("local commits are never empty").key
            }
        }
    }

    /// Returns every key this mutation touches, in a deterministic order
    /// (request order for local commits, deduplicated). Dirty-band
    /// tracking, sweep exclusion, and overlays iterate this; [`key`](Self::key)
    /// stays for single-key call sites (routing, per-key dedup).
    #[must_use]
    pub fn keys(&self) -> Vec<&Key> {
        match self {
            Self::TxnCommitLocal { writes, .. } => {
                let mut seen = std::collections::BTreeSet::new();
                let mut out = Vec::new();
                for write in writes {
                    if seen.insert(write.key.as_bytes()) {
                        out.push(&write.key);
                    }
                }
                out
            }
            _ => vec![self.key()],
        }
    }

    /// Maps a stored expiry to its conditional replay policy: `NEVER`
    /// always came from `PersistExpiry` (its only producer), anything else
    /// from `ExpireAt`.
    fn conditional_policy(expiry: &kivi_types::Expiry) -> crate::ops::ExpiryPolicy {
        if *expiry == kivi_types::Expiry::NEVER {
            crate::ops::ExpiryPolicy::Clear
        } else {
            crate::ops::ExpiryPolicy::ExpireAt(
                expiry
                    .as_stamp()
                    .unwrap_or(kivi_types::WallTimestamp::from_micros(0)),
            )
        }
    }

    /// Reconstructs the originating operation. Exact inverse of
    /// `prepare`'s normalization: `SetExpiry{NEVER}` always came from
    /// `PersistExpiry` (which is the only producer of `NEVER`), any other
    /// expiry came from `ExpireAt`. WAL replay uses this to share the single
    /// [`outcome_for`](crate::ops::outcome_for) mapping with the live path.
    #[must_use]
    pub fn as_operation(&self) -> crate::ops::Operation {
        use crate::ops::Operation;
        if let Some(operation) = self.as_txn_operation() {
            return operation;
        }
        if let Some(operation) = self.as_semaphore_stream_operation() {
            return operation;
        }
        if let Some(operation) = self.as_root_operation() {
            return operation;
        }
        if let Some(operation) = self.as_lease_operation() {
            return operation;
        }
        if let Some(operation) = self.as_conditional_operation() {
            return operation;
        }
        match self {
            Self::PutBytes { key, value } => Operation::Set {
                key: key.clone(),
                value: value.clone(),
            },
            Self::Delete { key } => Operation::Delete { key: key.clone() },
            Self::CounterAdd { key, delta } => Operation::CounterAdd {
                key: key.clone(),
                delta: *delta,
            },
            Self::SetExpiry { key, expiry } => {
                if *expiry == kivi_types::Expiry::NEVER {
                    Operation::PersistExpiry { key: key.clone() }
                } else {
                    Operation::ExpireAt {
                        key: key.clone(),
                        expires_at: expiry
                            .as_stamp()
                            .unwrap_or(kivi_types::WallTimestamp::from_micros(0)),
                    }
                }
            }
            Self::ReplaceChunkedRoot { .. } | Self::ReplaceFabricRoot { .. } => {
                unreachable!("root replacements dispatch above")
            }
            Self::SpliceBytes { key, offset, patch } => Operation::SetRange {
                key: key.clone(),
                offset: *offset,
                patch: patch.clone(),
            },
            Self::PutBytesWithExpiry { .. }
            | Self::ReplaceChunkedRootWithExpiry { .. }
            | Self::ReplaceFabricRootWithExpiry { .. } => {
                unreachable!("conditional mutations dispatch above")
            }
            Self::TxnPrepare { .. } | Self::TxnFinalize { .. } | Self::TxnCommitLocal { .. } => {
                unreachable!("transaction mutations dispatch above")
            }
            Self::SemaphoreCreate { .. }
            | Self::SemaphoreAcquire { .. }
            | Self::SemaphoreRelease { .. }
            | Self::StreamCreate { .. }
            | Self::StreamAppend { .. }
            | Self::StreamTrim { .. } => {
                unreachable!("semaphore/stream mutations dispatch above")
            }
            Self::CommutativeAdd { key, delta } => Operation::CommutativeAdd {
                key: key.clone(),
                delta: *delta,
            },
            Self::BoundedCreate {
                key,
                capacity,
                holder,
            } => Operation::BoundedCounterCreate {
                key: key.clone(),
                capacity: *capacity,
                holder: kivi_types::TabletId::from_u64(*holder),
            },
            Self::BoundedAdd { key, delta } => Operation::BoundedCounterAdd {
                key: key.clone(),
                delta: *delta,
            },
            Self::EscrowSetShare { key, min, max } => Operation::EscrowTransfer {
                key: key.clone(),
                new_min: *min,
                new_max: *max,
            },
            Self::LeaseAcquire { .. } | Self::LeaseRenew { .. } | Self::LeaseRelease { .. } => {
                unreachable!("lease mutations dispatch above")
            }
        }
    }

    /// Reconstructs a lease originating operation (`Some` for the three
    /// lease shapes, `None` otherwise): dispatches first in
    /// [`as_operation`](Self::as_operation) so the shared match stays
    /// reviewable.
    fn as_lease_operation(&self) -> Option<crate::ops::Operation> {
        use crate::ops::Operation;
        // Lease mutations carry prepare-resolved fencing/expiry while
        // operations carry caller TTLs; like the chunked-conditional
        // precedent, the mutation is the truth and the operation
        // is only the replay-mapping key (never used to re-derive).
        // Both grant shapes replay identically: a renew under the same
        // identity is indistinguishable from its grant.
        match self {
            Self::LeaseAcquire {
                key,
                owner,
                fencing,
                ..
            }
            | Self::LeaseRenew {
                key,
                owner,
                fencing,
                ..
            } => Some(Operation::LeaseRenew {
                key: key.clone(),
                owner: *owner,
                fencing: *fencing,
                ttl_micros: 0,
            }),
            Self::LeaseRelease {
                key,
                owner,
                fencing,
            } => Some(Operation::LeaseRelease {
                key: key.clone(),
                owner: *owner,
                fencing: *fencing,
            }),
            _ => None,
        }
    }

    /// Reconstructs a transaction originating operation (`Some` for the
    /// three 2PC shapes, `None` otherwise): dispatches first in
    /// [`as_operation`](Self::as_operation) so the shared match stays
    /// reviewable.
    fn as_txn_operation(&self) -> Option<crate::ops::Operation> {
        use crate::ops::Operation;
        match self {
            Self::TxnPrepare {
                txn,
                coordinator,
                key,
                expect,
                write,
                digest,
            } => Some(Operation::TxnPrepare {
                txn: *txn,
                coordinator: kivi_types::TabletId::from_u64(*coordinator),
                write: crate::txn::TxnWrite {
                    key: key.clone(),
                    kind: write.clone(),
                    expect: *expect,
                },
                digest: *digest,
            }),
            Self::TxnFinalize {
                txn,
                key,
                commit,
                digest,
            } => Some(Operation::TxnFinalize {
                txn: *txn,
                key: key.clone(),
                commit: *commit,
                digest: *digest,
            }),
            Self::TxnCommitLocal { txn, writes } => Some(Operation::TxnCommitLocal {
                txn: *txn,
                writes: writes.clone(),
            }),
            _ => None,
        }
    }

    /// Reconstructs a root-replacement originating operation (`Some` for
    /// the chunked/fabric shapes, `None` otherwise): dispatches first in
    /// [`as_operation`](Self::as_operation) so the shared match stays
    /// reviewable.
    fn as_root_operation(&self) -> Option<crate::ops::Operation> {
        use crate::ops::Operation;
        match self {
            Self::ReplaceChunkedRoot {
                key,
                manifest,
                logical_len,
            } => Some(Operation::SetChunked {
                key: key.clone(),
                manifest: *manifest,
                logical_len: *logical_len,
            }),
            Self::ReplaceFabricRoot {
                key,
                fabric_id,
                logical_len,
                version,
            } => Some(Operation::SetFabric {
                key: key.clone(),
                fabric_id: *fabric_id,
                logical_len: *logical_len,
                version: *version,
            }),
            _ => None,
        }
    }

    /// Reconstructs a conditional originating operation (`Some` for the
    /// two conditional shapes, `None` otherwise): dispatches first in
    /// [`as_operation`](Self::as_operation) so the shared match stays
    /// reviewable.
    fn as_conditional_operation(&self) -> Option<crate::ops::Operation> {
        use crate::ops::Operation;
        match self {
            Self::PutBytesWithExpiry { key, value, expiry } => Some(Operation::SetConditional {
                key: key.clone(),
                value: value.clone(),
                condition: crate::ops::SetCondition::Always,
                expiry: Self::conditional_policy(expiry),
            }),
            Self::ReplaceChunkedRootWithExpiry {
                key,
                manifest: _,
                logical_len: _,
                expiry,
            } => {
                // Chunked conditional roots replay through the same
                // conditional shape; the manifest itself is the mutation's
                // truth, the operation is only the replay-mapping key.
                Some(Operation::SetConditional {
                    key: key.clone(),
                    value: bytes::Bytes::new(),
                    condition: crate::ops::SetCondition::Always,
                    expiry: Self::conditional_policy(expiry),
                })
            }
            Self::ReplaceFabricRootWithExpiry {
                key,
                fabric_id: _,
                logical_len: _,
                version: _,
                expiry,
            } => {
                // Fabric conditional roots replay like chunked ones: the
                // reference is the mutation's truth.
                Some(Operation::SetConditional {
                    key: key.clone(),
                    value: bytes::Bytes::new(),
                    condition: crate::ops::SetCondition::Always,
                    expiry: Self::conditional_policy(expiry),
                })
            }
            _ => None,
        }
    }

    /// Reconstructs a semaphore/stream originating operation (`Some` for
    /// those six shapes, `None` otherwise): dispatches first in
    /// [`as_operation`](Self::as_operation) so the shared match stays
    /// reviewable.
    fn as_semaphore_stream_operation(&self) -> Option<crate::ops::Operation> {
        use crate::ops::Operation;
        match self {
            Self::SemaphoreCreate { key, capacity } => Some(Operation::SemaphoreCreate {
                key: key.clone(),
                capacity: *capacity,
            }),
            Self::SemaphoreAcquire {
                key,
                permit,
                owner,
                qty,
            } => Some(Operation::SemaphoreAcquire {
                key: key.clone(),
                permit: crate::PermitId::from_bytes(*permit),
                owner: *owner,
                qty: *qty,
            }),
            Self::SemaphoreRelease { key, permit } => Some(Operation::SemaphoreRelease {
                key: key.clone(),
                permit: crate::PermitId::from_bytes(*permit),
            }),
            Self::StreamCreate { key, stream, shard } => Some(Operation::StreamCreate {
                key: key.clone(),
                stream: *stream,
                shard: *shard,
            }),
            Self::StreamAppend {
                key,
                partition,
                payload,
            } => Some(Operation::StreamAppend {
                key: key.clone(),
                partition: partition.to_vec(),
                payload: payload.clone(),
            }),
            Self::StreamTrim { key, through } => Some(Operation::StreamTrim {
                key: key.clone(),
                through_offset: *through,
            }),
            _ => None,
        }
    }
}

impl Encode for Mutation {
    fn encoded_len(&self) -> usize {
        match self {
            Self::PutBytes { key, value } => 1 + (4 + key.len()) + (4 + value.len()),
            Self::Delete { key } => 1 + (4 + key.len()),
            // Shared key + u64 body shape (signed delta, capacity, or
            // trim cursor).
            Self::CounterAdd { key, .. }
            | Self::CommutativeAdd { key, .. }
            | Self::BoundedAdd { key, .. }
            | Self::SemaphoreCreate { key, .. }
            | Self::StreamTrim { key, .. } => 1 + (4 + key.len()) + 8,
            Self::SetExpiry { key, expiry } => 1 + (4 + key.len()) + expiry.encoded_len(),
            Self::ReplaceChunkedRoot { key, .. } => 1 + (4 + key.len()) + 32 + 8,
            Self::ReplaceFabricRoot { key, .. } => 1 + (4 + key.len()) + 8 + 8 + 8,
            Self::SpliceBytes { key, patch, .. } => 1 + (4 + key.len()) + 8 + (4 + patch.len()),
            Self::PutBytesWithExpiry { key, value, expiry } => {
                1 + (4 + key.len()) + (4 + value.len()) + expiry.encoded_len()
            }
            Self::ReplaceChunkedRootWithExpiry { key, expiry, .. } => {
                1 + (4 + key.len()) + 32 + 8 + expiry.encoded_len()
            }
            Self::ReplaceFabricRootWithExpiry { key, expiry, .. } => {
                1 + (4 + key.len()) + 8 + 8 + 8 + expiry.encoded_len()
            }
            Self::TxnPrepare {
                key, expect, write, ..
            } => 1 + 16 + 8 + (4 + key.len()) + expect.encoded_len() + write.encoded_len() + 32,
            Self::TxnFinalize { key, .. } => 1 + 16 + (4 + key.len()) + 1 + 32,
            Self::TxnCommitLocal { writes, .. } => {
                1 + 16
                    + 2
                    + writes
                        .iter()
                        .map(|write| {
                            (4 + write.key.len())
                                + write.expect.encoded_len()
                                + write.kind.encoded_len()
                        })
                        .sum::<usize>()
            }
            Self::BoundedCreate { key, .. } | Self::EscrowSetShare { key, .. } => {
                1 + (4 + key.len()) + 8 + 8
            }
            Self::SemaphoreAcquire { key, .. } => 1 + (4 + key.len()) + 16 + 8 + 8,
            Self::SemaphoreRelease { key, .. } => 1 + (4 + key.len()) + 16,
            Self::LeaseAcquire { key, .. } | Self::LeaseRenew { key, .. } => {
                1 + (4 + key.len())
                    + 8
                    + 8
                    + kivi_types::WallTimestamp::from_micros(0).encoded_len()
            }
            Self::LeaseRelease { key, .. } => 1 + (4 + key.len()) + 8 + 8,
            Self::StreamCreate { key, .. } => 1 + (4 + key.len()) + 16 + 4,
            Self::StreamAppend {
                key,
                partition,
                payload,
            } => 1 + (4 + key.len()) + (4 + partition.len()) + (4 + payload.len()),
        }
    }

    fn encode(&self, out: &mut Vec<u8>) {
        match self {
            Self::PutBytes { key, value } => {
                out.push(TAG_PUT_BYTES);
                encode_bytes(out, key.as_bytes());
                encode_bytes(out, value);
            }
            Self::Delete { key } => {
                out.push(TAG_DELETE);
                encode_bytes(out, key.as_bytes());
            }
            Self::CounterAdd { key, delta } => {
                Self::encode_key_delta(out, TAG_COUNTER_ADD, key, *delta);
            }
            Self::SetExpiry { key, expiry } => {
                out.push(TAG_SET_EXPIRY);
                encode_bytes(out, key.as_bytes());
                expiry.encode(out);
            }
            Self::ReplaceChunkedRoot { .. }
            | Self::ReplaceChunkedRootWithExpiry { .. }
            | Self::ReplaceFabricRoot { .. }
            | Self::ReplaceFabricRootWithExpiry { .. } => self.encode_root_mutation(out),
            Self::SpliceBytes { key, offset, patch } => {
                out.push(TAG_SPLICE_BYTES);
                encode_bytes(out, key.as_bytes());
                offset.encode(out);
                encode_bytes(out, patch);
            }
            Self::PutBytesWithExpiry { key, value, expiry } => {
                out.push(TAG_PUT_BYTES_WITH_EXPIRY);
                encode_bytes(out, key.as_bytes());
                encode_bytes(out, value);
                expiry.encode(out);
            }
            Self::TxnPrepare {
                txn,
                coordinator,
                key,
                expect,
                write,
                digest,
            } => Self::encode_txn_prepare(out, txn, *coordinator, key, expect, write, digest),
            Self::TxnFinalize {
                txn,
                key,
                commit,
                digest,
            } => {
                Self::encode_txn_finalize(out, txn, key, *commit, digest);
            }
            Self::TxnCommitLocal { txn, writes } => Self::encode_txn_commit_local(out, txn, writes),
            Self::CommutativeAdd { key, delta } => {
                Self::encode_key_delta(out, TAG_COMMUTATIVE_ADD, key, *delta);
            }
            Self::BoundedCreate { .. } | Self::BoundedAdd { .. } | Self::EscrowSetShare { .. } => {
                self.encode_bounded_mutation(out);
            }
            Self::SemaphoreCreate { .. }
            | Self::SemaphoreAcquire { .. }
            | Self::SemaphoreRelease { .. } => {
                self.encode_semaphore_mutation(out);
            }
            Self::LeaseAcquire {
                key,
                owner,
                fencing,
                expires_at,
            } => {
                Self::encode_lease_grant(
                    out,
                    TAG_LEASE_ACQUIRE,
                    key,
                    *owner,
                    *fencing,
                    *expires_at,
                );
            }
            Self::LeaseRenew {
                key,
                owner,
                fencing,
                expires_at,
            } => Self::encode_lease_grant(out, TAG_LEASE_RENEW, key, *owner, *fencing, *expires_at),
            Self::LeaseRelease {
                key,
                owner,
                fencing,
            } => {
                out.push(TAG_LEASE_RELEASE);
                encode_bytes(out, key.as_bytes());
                owner.encode(out);
                fencing.encode(out);
            }
            Self::StreamCreate { .. } | Self::StreamAppend { .. } | Self::StreamTrim { .. } => {
                self.encode_stream_mutation(out);
            }
        }
    }
}

impl Mutation {
    /// Encodes one `key + i64` counter-style body (strict, commutative, and
    /// bounded adds share the shape; only the tag differs).
    fn encode_key_delta(out: &mut Vec<u8>, tag: u8, key: &Key, delta: i64) {
        out.push(tag);
        encode_bytes(out, key.as_bytes());
        out.extend_from_slice(&delta.to_le_bytes());
    }

    /// Encodes one chunked-root replacement: manifest plus logical length,
    /// with the resolved expiry for conditional shapes (`None` stores no
    /// expiry, exactly like the inline `Set` twin).
    fn encode_chunked_root(
        out: &mut Vec<u8>,
        tag: u8,
        key: &Key,
        manifest: &ManifestId,
        logical_len: u64,
        expiry: Option<&Expiry>,
    ) {
        out.push(tag);
        encode_bytes(out, key.as_bytes());
        out.extend_from_slice(manifest.as_bytes());
        logical_len.encode(out);
        if let Some(expiry) = expiry {
            expiry.encode(out);
        }
    }

    /// Encodes one fabric-root replacement: staged reference triple, with
    /// the resolved expiry for conditional shapes (same contract as
    /// [`encode_chunked_root`](Self::encode_chunked_root)).
    fn encode_fabric_root(
        out: &mut Vec<u8>,
        tag: u8,
        key: &Key,
        fabric_id: u64,
        logical_len: u64,
        version: u64,
        expiry: Option<&Expiry>,
    ) {
        out.push(tag);
        encode_bytes(out, key.as_bytes());
        fabric_id.encode(out);
        logical_len.encode(out);
        version.encode(out);
        if let Some(expiry) = expiry {
            expiry.encode(out);
        }
    }

    /// Encodes one 2PC prepare: coordinator, key, expectation, write, and
    /// the write-set digest binding the reservation to its transaction.
    fn encode_txn_prepare(
        out: &mut Vec<u8>,
        txn: &crate::txn::TxnId,
        coordinator: u64,
        key: &Key,
        expect: &crate::txn::TxnExpect,
        write: &crate::txn::TxnWriteKind,
        digest: &[u8; 32],
    ) {
        out.push(TAG_TXN_PREPARE);
        txn.encode(out);
        coordinator.encode(out);
        encode_bytes(out, key.as_bytes());
        expect.encode(out);
        write.encode(out);
        out.extend_from_slice(digest);
    }

    /// Encodes one 2PC finalize: which intent resolves, how, and under
    /// which digest (mismatches finalize nothing).
    fn encode_txn_finalize(
        out: &mut Vec<u8>,
        txn: &crate::txn::TxnId,
        key: &Key,
        commit: bool,
        digest: &[u8; 32],
    ) {
        out.push(TAG_TXN_FINALIZE);
        txn.encode(out);
        encode_bytes(out, key.as_bytes());
        commit.encode(out);
        out.extend_from_slice(digest);
    }

    /// Encodes one same-tablet atomic commit: the id plus every write
    /// (key, expectation, kind) in request order.
    fn encode_txn_commit_local(
        out: &mut Vec<u8>,
        txn: &crate::txn::TxnId,
        writes: &[crate::txn::TxnWrite],
    ) {
        out.push(TAG_TXN_COMMIT_LOCAL);
        txn.encode(out);
        let count = u16::try_from(writes.len()).unwrap_or(u16::MAX);
        out.extend_from_slice(&count.to_le_bytes());
        for write in writes {
            encode_bytes(out, write.key.as_bytes());
            write.expect.encode(out);
            write.kind.encode(out);
        }
    }

    /// Encodes one lease grant shape (acquire and renew share it): owner,
    /// fencing token, and resolved expiry.
    fn encode_lease_grant(
        out: &mut Vec<u8>,
        tag: u8,
        key: &Key,
        owner: u64,
        fencing: crate::FencingToken,
        expires_at: kivi_types::WallTimestamp,
    ) {
        out.push(tag);
        encode_bytes(out, key.as_bytes());
        owner.encode(out);
        fencing.encode(out);
        expires_at.encode(out);
    }

    /// Encodes one root-replacement shape (chunked or fabric, plain or
    /// conditional): manifest or staged-reference triple, plus the
    /// resolved expiry for conditional shapes. Dispatches on the variant
    /// so the shared tag match stays reviewable.
    fn encode_root_mutation(&self, out: &mut Vec<u8>) {
        match self {
            Self::ReplaceChunkedRoot {
                key,
                manifest,
                logical_len,
            } => Self::encode_chunked_root(
                out,
                TAG_REPLACE_CHUNKED_ROOT,
                key,
                manifest,
                *logical_len,
                None,
            ),
            Self::ReplaceChunkedRootWithExpiry {
                key,
                manifest,
                logical_len,
                expiry,
            } => Self::encode_chunked_root(
                out,
                TAG_REPLACE_CHUNKED_ROOT_WITH_EXPIRY,
                key,
                manifest,
                *logical_len,
                Some(expiry),
            ),
            Self::ReplaceFabricRoot {
                key,
                fabric_id,
                logical_len,
                version,
            } => Self::encode_fabric_root(
                out,
                TAG_REPLACE_FABRIC_ROOT,
                key,
                *fabric_id,
                *logical_len,
                *version,
                None,
            ),
            Self::ReplaceFabricRootWithExpiry {
                key,
                fabric_id,
                logical_len,
                version,
                expiry,
            } => Self::encode_fabric_root(
                out,
                TAG_REPLACE_FABRIC_ROOT_WITH_EXPIRY,
                key,
                *fabric_id,
                *logical_len,
                *version,
                Some(expiry),
            ),
            _ => unreachable!("root replacements only"),
        }
    }

    /// Encodes one bounded-counter shape (create, add, or share move):
    /// dispatches on the variant so the shared tag match stays reviewable.
    fn encode_bounded_mutation(&self, out: &mut Vec<u8>) {
        match self {
            Self::BoundedCreate {
                key,
                capacity,
                holder,
            } => {
                out.push(TAG_BOUNDED_CREATE);
                encode_bytes(out, key.as_bytes());
                capacity.encode(out);
                holder.encode(out);
            }
            Self::BoundedAdd { key, delta } => {
                Self::encode_key_delta(out, TAG_BOUNDED_ADD, key, *delta);
            }
            Self::EscrowSetShare { key, min, max } => {
                out.push(TAG_ESCROW_SET_SHARE);
                encode_bytes(out, key.as_bytes());
                out.extend_from_slice(&min.to_le_bytes());
                out.extend_from_slice(&max.to_le_bytes());
            }
            _ => unreachable!("bounded shapes only"),
        }
    }

    /// Encodes one semaphore shape (create, acquire, or release):
    /// dispatches on the variant so the shared tag match stays reviewable.
    fn encode_semaphore_mutation(&self, out: &mut Vec<u8>) {
        match self {
            Self::SemaphoreCreate { key, capacity } => {
                out.push(TAG_SEMAPHORE_CREATE);
                encode_bytes(out, key.as_bytes());
                capacity.encode(out);
            }
            Self::SemaphoreAcquire {
                key,
                permit,
                owner,
                qty,
            } => {
                out.push(TAG_SEMAPHORE_ACQUIRE);
                encode_bytes(out, key.as_bytes());
                out.extend_from_slice(permit);
                owner.encode(out);
                qty.encode(out);
            }
            Self::SemaphoreRelease { key, permit } => {
                out.push(TAG_SEMAPHORE_RELEASE);
                encode_bytes(out, key.as_bytes());
                out.extend_from_slice(permit);
            }
            _ => unreachable!("semaphore shapes only"),
        }
    }

    /// Encodes one stream shape (create, append, or trim): dispatches on
    /// the variant so the shared tag match stays reviewable.
    fn encode_stream_mutation(&self, out: &mut Vec<u8>) {
        match self {
            Self::StreamCreate { key, stream, shard } => {
                out.push(TAG_STREAM_CREATE);
                encode_bytes(out, key.as_bytes());
                out.extend_from_slice(stream);
                shard.encode(out);
            }
            Self::StreamAppend {
                key,
                partition,
                payload,
            } => {
                out.push(TAG_STREAM_APPEND);
                encode_bytes(out, key.as_bytes());
                encode_bytes(out, partition);
                encode_bytes(out, payload);
            }
            Self::StreamTrim { key, through } => {
                out.push(TAG_STREAM_TRIM);
                encode_bytes(out, key.as_bytes());
                through.encode(out);
            }
            _ => unreachable!("stream shapes only"),
        }
    }
}

impl Decode for Mutation {
    #[allow(clippy::too_many_lines)]
    fn decode(input: &[u8]) -> Result<(Self, usize), CodecError> {
        let (tag, first) = u8::decode(input)?;
        match tag {
            TAG_PUT_BYTES => {
                let (key, second) = decode_byte_vec(&input[first..])?;
                let (value, third) = decode_byte_vec(&input[first + second..])?;
                Ok((
                    Self::PutBytes {
                        key: Key::from(key),
                        value: bytes::Bytes::from(value),
                    },
                    first + second + third,
                ))
            }
            TAG_DELETE => {
                let (key, second) = decode_byte_vec(&input[first..])?;
                Ok((
                    Self::Delete {
                        key: Key::from(key),
                    },
                    first + second,
                ))
            }
            TAG_COUNTER_ADD => {
                let (key, second) = decode_byte_vec(&input[first..])?;
                let (raw, third) = <[u8; 8]>::decode(&input[first + second..])?;
                Ok((
                    Self::CounterAdd {
                        key: Key::from(key),
                        delta: i64::from_le_bytes(raw),
                    },
                    first + second + third,
                ))
            }
            TAG_SET_EXPIRY => {
                let (key, second) = decode_byte_vec(&input[first..])?;
                let (expiry, third) = Expiry::decode(&input[first + second..])?;
                Ok((
                    Self::SetExpiry {
                        key: Key::from(key),
                        expiry,
                    },
                    first + second + third,
                ))
            }
            TAG_REPLACE_CHUNKED_ROOT => {
                let (key, second) = decode_byte_vec(&input[first..])?;
                let (raw, third) = <[u8; 32]>::decode(&input[first + second..])?;
                let (logical_len, fourth) = u64::decode(&input[first + second + third..])?;
                Ok((
                    Self::ReplaceChunkedRoot {
                        key: Key::from(key),
                        manifest: ManifestId::from_bytes(raw),
                        logical_len,
                    },
                    first + second + third + fourth,
                ))
            }
            TAG_SPLICE_BYTES => {
                let (key, second) = decode_byte_vec(&input[first..])?;
                let (offset, third) = u64::decode(&input[first + second..])?;
                let (patch, fourth) = decode_byte_vec(&input[first + second + third..])?;
                Ok((
                    Self::SpliceBytes {
                        key: Key::from(key),
                        offset,
                        patch: bytes::Bytes::from(patch),
                    },
                    first + second + third + fourth,
                ))
            }
            TAG_PUT_BYTES_WITH_EXPIRY => {
                let (key, second) = decode_byte_vec(&input[first..])?;
                let (value, third) = decode_byte_vec(&input[first + second..])?;
                let (expiry, fourth) = Expiry::decode(&input[first + second + third..])?;
                Ok((
                    Self::PutBytesWithExpiry {
                        key: Key::from(key),
                        value: bytes::Bytes::from(value),
                        expiry,
                    },
                    first + second + third + fourth,
                ))
            }
            TAG_REPLACE_CHUNKED_ROOT_WITH_EXPIRY => {
                let (key, second) = decode_byte_vec(&input[first..])?;
                let (raw, third) = <[u8; 32]>::decode(&input[first + second..])?;
                let (logical_len, fourth) = u64::decode(&input[first + second + third..])?;
                let (expiry, fifth) = Expiry::decode(&input[first + second + third + fourth..])?;
                Ok((
                    Self::ReplaceChunkedRootWithExpiry {
                        key: Key::from(key),
                        manifest: ManifestId::from_bytes(raw),
                        logical_len,
                        expiry,
                    },
                    first + second + third + fourth + fifth,
                ))
            }
            TAG_REPLACE_FABRIC_ROOT => {
                let (key, second) = decode_byte_vec(&input[first..])?;
                let (fabric_id, third) = u64::decode(&input[first + second..])?;
                let (logical_len, fourth) = u64::decode(&input[first + second + third..])?;
                let (version, fifth) = u64::decode(&input[first + second + third + fourth..])?;
                Ok((
                    Self::ReplaceFabricRoot {
                        key: Key::from(key),
                        fabric_id,
                        logical_len,
                        version,
                    },
                    first + second + third + fourth + fifth,
                ))
            }
            TAG_REPLACE_FABRIC_ROOT_WITH_EXPIRY => {
                let (key, second) = decode_byte_vec(&input[first..])?;
                let (fabric_id, third) = u64::decode(&input[first + second..])?;
                let (logical_len, fourth) = u64::decode(&input[first + second + third..])?;
                let (version, fifth) = u64::decode(&input[first + second + third + fourth..])?;
                let (expiry, sixth) =
                    Expiry::decode(&input[first + second + third + fourth + fifth..])?;
                Ok((
                    Self::ReplaceFabricRootWithExpiry {
                        key: Key::from(key),
                        fabric_id,
                        logical_len,
                        version,
                        expiry,
                    },
                    first + second + third + fourth + fifth + sixth,
                ))
            }
            TAG_TXN_PREPARE => {
                let (txn, second) = TxnId::decode(&input[first..])?;
                let (coordinator, third) = u64::decode(&input[first + second..])?;
                let (key, fourth) = decode_byte_vec(&input[first + second + third..])?;
                let (expect, fifth) = TxnExpect::decode(&input[first + second + third + fourth..])?;
                let (write, sixth) =
                    TxnWriteKind::decode(&input[first + second + third + fourth + fifth..])?;
                let at = first + second + third + fourth + fifth + sixth;
                let (digest, used) = <[u8; 32]>::decode(&input[at..])?;
                Ok((
                    Self::TxnPrepare {
                        txn,
                        coordinator,
                        key: Key::from(key),
                        expect,
                        write,
                        digest,
                    },
                    at + used,
                ))
            }
            TAG_TXN_FINALIZE => {
                let (txn, second) = TxnId::decode(&input[first..])?;
                let (key, third) = decode_byte_vec(&input[first + second..])?;
                let (commit, fourth) = bool::decode(&input[first + second + third..])?;
                let at = first + second + third + fourth;
                let (digest, used) = <[u8; 32]>::decode(&input[at..])?;
                Ok((
                    Self::TxnFinalize {
                        txn,
                        key: Key::from(key),
                        commit,
                        digest,
                    },
                    at + used,
                ))
            }
            TAG_TXN_COMMIT_LOCAL => {
                let (txn, mut at) = TxnId::decode(&input[first..])?;
                at += first;
                let (count_raw, used) = <[u8; 2]>::decode(&input[at..])?;
                at += used;
                let count = usize::from(u16::from_le_bytes(count_raw));
                // Empty local commits are malformed (drivers never send
                // them; `key()` has no anchor without a first write).
                if count == 0 || count > crate::txn::MAX_TXN_KEYS {
                    return Err(CodecError::InvalidTag {
                        kind: "mutation",
                        tag: TAG_TXN_COMMIT_LOCAL,
                    });
                }
                let mut writes = Vec::with_capacity(count.min(16));
                for _ in 0..count {
                    let (key, used) = decode_byte_vec(&input[at..])?;
                    at += used;
                    let (expect, used) = TxnExpect::decode(&input[at..])?;
                    at += used;
                    let (kind, used) = TxnWriteKind::decode(&input[at..])?;
                    at += used;
                    writes.push(crate::txn::TxnWrite {
                        key: Key::from(key),
                        kind,
                        expect,
                    });
                }
                Ok((Self::TxnCommitLocal { txn, writes }, at))
            }
            TAG_COMMUTATIVE_ADD => {
                let (key, second) = decode_byte_vec(&input[first..])?;
                let (raw, third) = <[u8; 8]>::decode(&input[first + second..])?;
                Ok((
                    Self::CommutativeAdd {
                        key: Key::from(key),
                        delta: i64::from_le_bytes(raw),
                    },
                    first + second + third,
                ))
            }
            TAG_BOUNDED_CREATE => {
                let (key, second) = decode_byte_vec(&input[first..])?;
                let (capacity, third) = u64::decode(&input[first + second..])?;
                let (holder, fourth) = u64::decode(&input[first + second + third..])?;
                Ok((
                    Self::BoundedCreate {
                        key: Key::from(key),
                        capacity,
                        holder,
                    },
                    first + second + third + fourth,
                ))
            }
            TAG_BOUNDED_ADD => {
                let (key, second) = decode_byte_vec(&input[first..])?;
                let (raw, third) = <[u8; 8]>::decode(&input[first + second..])?;
                Ok((
                    Self::BoundedAdd {
                        key: Key::from(key),
                        delta: i64::from_le_bytes(raw),
                    },
                    first + second + third,
                ))
            }
            TAG_ESCROW_SET_SHARE => {
                let (key, second) = decode_byte_vec(&input[first..])?;
                let (min_raw, third) = <[u8; 8]>::decode(&input[first + second..])?;
                let (max_raw, fourth) = <[u8; 8]>::decode(&input[first + second + third..])?;
                Ok((
                    Self::EscrowSetShare {
                        key: Key::from(key),
                        min: i64::from_le_bytes(min_raw),
                        max: i64::from_le_bytes(max_raw),
                    },
                    first + second + third + fourth,
                ))
            }
            TAG_SEMAPHORE_CREATE => {
                let (key, second) = decode_byte_vec(&input[first..])?;
                let (capacity, third) = u64::decode(&input[first + second..])?;
                Ok((
                    Self::SemaphoreCreate {
                        key: Key::from(key),
                        capacity,
                    },
                    first + second + third,
                ))
            }
            TAG_SEMAPHORE_ACQUIRE => {
                let (key, second) = decode_byte_vec(&input[first..])?;
                let (permit, third) = <[u8; 16]>::decode(&input[first + second..])?;
                let (owner, fourth) = u64::decode(&input[first + second + third..])?;
                let (qty, fifth) = u64::decode(&input[first + second + third + fourth..])?;
                Ok((
                    Self::SemaphoreAcquire {
                        key: Key::from(key),
                        permit,
                        owner,
                        qty,
                    },
                    first + second + third + fourth + fifth,
                ))
            }
            TAG_SEMAPHORE_RELEASE => {
                let (key, second) = decode_byte_vec(&input[first..])?;
                let (permit, third) = <[u8; 16]>::decode(&input[first + second..])?;
                Ok((
                    Self::SemaphoreRelease {
                        key: Key::from(key),
                        permit,
                    },
                    first + second + third,
                ))
            }
            TAG_LEASE_ACQUIRE => {
                let (key, second) = decode_byte_vec(&input[first..])?;
                let (owner, third) = u64::decode(&input[first + second..])?;
                let (fencing, fourth) =
                    crate::FencingToken::decode(&input[first + second + third..])?;
                let (expires_at, fifth) =
                    kivi_types::WallTimestamp::decode(&input[first + second + third + fourth..])?;
                Ok((
                    Self::LeaseAcquire {
                        key: Key::from(key),
                        owner,
                        fencing,
                        expires_at,
                    },
                    first + second + third + fourth + fifth,
                ))
            }
            TAG_LEASE_RENEW => {
                let (key, second) = decode_byte_vec(&input[first..])?;
                let (owner, third) = u64::decode(&input[first + second..])?;
                let (fencing, fourth) =
                    crate::FencingToken::decode(&input[first + second + third..])?;
                let (expires_at, fifth) =
                    kivi_types::WallTimestamp::decode(&input[first + second + third + fourth..])?;
                Ok((
                    Self::LeaseRenew {
                        key: Key::from(key),
                        owner,
                        fencing,
                        expires_at,
                    },
                    first + second + third + fourth + fifth,
                ))
            }
            TAG_LEASE_RELEASE => {
                let (key, second) = decode_byte_vec(&input[first..])?;
                let (owner, third) = u64::decode(&input[first + second..])?;
                let (fencing, fourth) =
                    crate::FencingToken::decode(&input[first + second + third..])?;
                Ok((
                    Self::LeaseRelease {
                        key: Key::from(key),
                        owner,
                        fencing,
                    },
                    first + second + third + fourth,
                ))
            }
            TAG_STREAM_CREATE => {
                let (key, second) = decode_byte_vec(&input[first..])?;
                let (stream, third) = <[u8; 16]>::decode(&input[first + second..])?;
                let (shard, fourth) = u32::decode(&input[first + second + third..])?;
                Ok((
                    Self::StreamCreate {
                        key: Key::from(key),
                        stream,
                        shard,
                    },
                    first + second + third + fourth,
                ))
            }
            TAG_STREAM_APPEND => {
                let (key, second) = decode_byte_vec(&input[first..])?;
                let (partition, third) = decode_byte_vec(&input[first + second..])?;
                let (payload, fourth) = decode_byte_vec(&input[first + second + third..])?;
                Ok((
                    Self::StreamAppend {
                        key: Key::from(key),
                        partition: bytes::Bytes::from(partition),
                        payload: bytes::Bytes::from(payload),
                    },
                    first + second + third + fourth,
                ))
            }
            TAG_STREAM_TRIM => {
                let (key, second) = decode_byte_vec(&input[first..])?;
                let (through, third) = u64::decode(&input[first + second..])?;
                Ok((
                    Self::StreamTrim {
                        key: Key::from(key),
                        through,
                    },
                    first + second + third,
                ))
            }
            other => Err(CodecError::InvalidTag {
                kind: "mutation",
                tag: other,
            }),
        }
    }
}

/// Outcome of one applied mutation.
///
/// Exhaustive by design (see [`OperationResult`](crate::OperationResult)):
/// consumers handle every outcome explicitly.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ApplyOutcome {
    /// Bytes stored with the new version.
    Put {
        /// Version after the store.
        version: ObjectVersion,
    },
    /// Key removed; `existed` names a previously live object.
    Deleted {
        /// Whether a live object was removed.
        existed: bool,
    },
    /// Counter stored with the new value and version.
    Counter {
        /// Version after the addition.
        version: ObjectVersion,
        /// Value after the addition.
        value: i64,
    },
    /// Expiry attached or cleared.
    Expiry {
        /// Whether a live object took the change.
        applied: bool,
        /// Version after the change (`None` when untouched).
        version: Option<ObjectVersion>,
    },
    /// Transaction intent reserved (prepare validated and recorded).
    TxnPrepared,
    /// Transaction prepare evaluated to a conflict (OCC mismatch or a live
    /// intent from another transaction blocks the key). Deterministic: same
    /// log on same state always agrees, so this is a normal outcome, never
    /// divergence.
    TxnConflict,
    /// One key's intent resolved: commit applied the prepared write (or the
    /// intent was already gone), abort discarded it.
    TxnFinalized {
        /// Whether a prepared write was applied (`false` for aborts and for
        /// idempotent replays with no intent left).
        applied: bool,
        /// Version after the applied write (`None` when nothing applied).
        version: Option<ObjectVersion>,
    },
    /// Commutative addition applied with the post-apply sum and version.
    /// The client-visible result drops the sum (`Applied`): this outcome
    /// exists so durability prediction and replay verify exactly.
    Commutative {
        /// Version after the addition.
        version: ObjectVersion,
        /// Sum after the addition.
        value: i64,
    },
    /// Bounded-counter addition applied with the post-apply value/version.
    Bounded {
        /// Version after the addition.
        version: ObjectVersion,
        /// Value after the addition.
        value: i64,
    },
    /// Semaphore acquisition applied (permit live, or identical retry).
    SemaphoreAcquired,
    /// Semaphore release applied.
    SemaphoreReleased {
        /// Whether a live permit was released.
        released: bool,
    },
    /// Lease grant applied (acquire or renew) with the authoritative token.
    LeaseGranted {
        /// Fencing token of the grant.
        fencing: crate::FencingToken,
        /// Logical expiry of the grant.
        expires_at: kivi_types::WallTimestamp,
    },
    /// Lease release applied.
    LeaseReleased {
        /// Whether a live holding was released.
        released: bool,
    },
    /// Stream append applied with the assigned shard-local offset.
    StreamAppended {
        /// Assigned offset.
        offset: u64,
    },
    /// Stream trim applied with the dropped entry count.
    StreamTrimmed {
        /// Entries removed.
        removed: u64,
    },
    /// Same-tablet atomic commit applied: resulting versions in request
    /// order (`None` for deletes, which carry no version). The whole set
    /// validated and applied together; verification compares this vector
    /// element-wise against the predicted one.
    TxnLocalCommitted {
        /// Resulting versions in request order.
        versions: Vec<Option<ObjectVersion>>,
    },
}

/// Deterministic apply failure. Same state plus same mutation always agrees,
/// so replicas never diverge through these paths.
///
/// Exhaustive like [`ApplyOutcome`].
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ApplyError {
    /// An object version cannot advance past `u64::MAX`.
    #[error("object version space exhausted")]
    VersionExhausted,
    /// A counter addition overflows `i64`.
    #[error("counter addition overflows i64")]
    CounterOverflow,
    /// A counter mutation met a byte string (only reachable by applying
    /// mutations against divergent state).
    #[error("mutation needs {expected}, key holds {found}")]
    TypeMismatch {
        /// Type the mutation required.
        expected: ObjectType,
        /// Type actually stored.
        found: ObjectType,
    },
    /// A splice met a chunked base the engine must have restaged first,
    /// or an offset that overflows `u64`. Only reachable by applying
    /// mutations against divergent state: admission resolves chunked
    /// bases through the lane and validates arithmetic before the WAL.
    #[error("byte-range splice cannot apply to stored state")]
    UnresolvableSplice,
    /// A transactional prepare or finalize diverged from its validated
    /// prediction (same log on same state never produces this): either a
    /// driver bug replaying a different write under one `TxnId`, or state
    /// divergence. Fail closed, never partial intent state.
    #[error("transaction mutation diverged from its prepared prediction")]
    TxnDiverged,
    /// A bounded-counter mutation would leave owned escrow rights (only
    /// reachable by applying mutations against divergent state: admission
    /// checks rights before the WAL).
    #[error("bounded counter would exceed owned escrow rights")]
    BoundedExceeded,
    /// A semaphore acquisition would exceed capacity (only reachable on
    /// divergent state: admission checks capacity before the WAL).
    #[error("semaphore capacity exhausted")]
    SemaphoreExhausted,
    /// A lease mutation met a conflicting live holder or a stale fencing
    /// token (only reachable on divergent state: admission validates the
    /// grant before the WAL).
    #[error("lease grant conflict")]
    LeaseConflict,
    /// A stream append would exceed shard bounds (only reachable on
    /// divergent state: admission bounds entries before the WAL).
    #[error("stream shard full")]
    StreamFull,
}

#[cfg(test)]
mod tests {
    use super::*;
    use kivi_types::WallTimestamp;

    #[test]
    fn mutation_codecs_round_trip() {
        let key = Key::from("user:1");
        let cases = legacy_codec_cases(&key)
            .into_iter()
            .chain(txn_codec_cases(&key))
            .chain(semantic_codec_cases(&key));
        for mutation in cases {
            assert_eq!(mutation.encoded_len(), mutation.encode_to_vec().len());
            assert_eq!(
                Mutation::decode_exact(&mutation.encode_to_vec()).expect("round trip"),
                mutation
            );
        }
    }

    fn legacy_codec_cases(key: &Key) -> Vec<Mutation> {
        vec![
            Mutation::PutBytes {
                key: key.clone(),
                value: bytes::Bytes::from_static(b"v"),
            },
            Mutation::Delete { key: key.clone() },
            Mutation::CounterAdd {
                key: key.clone(),
                delta: -41,
            },
            Mutation::SetExpiry {
                key: key.clone(),
                expiry: Expiry::at(WallTimestamp::from_micros(99)),
            },
            Mutation::SetExpiry {
                key: key.clone(),
                expiry: Expiry::NEVER,
            },
            Mutation::ReplaceChunkedRoot {
                key: key.clone(),
                manifest: ManifestId::from_bytes([0x11; 32]),
                logical_len: 3_000_000,
            },
            Mutation::SpliceBytes {
                key: key.clone(),
                offset: 7,
                patch: bytes::Bytes::from_static(b"patch"),
            },
            Mutation::SpliceBytes {
                key: key.clone(),
                offset: u64::MAX,
                patch: bytes::Bytes::new(),
            },
            Mutation::PutBytesWithExpiry {
                key: key.clone(),
                value: bytes::Bytes::from_static(b"v"),
                expiry: Expiry::NEVER,
            },
            Mutation::PutBytesWithExpiry {
                key: key.clone(),
                value: bytes::Bytes::from_static(b"v"),
                expiry: Expiry::at(WallTimestamp::from_micros(7)),
            },
            Mutation::ReplaceChunkedRootWithExpiry {
                key: key.clone(),
                manifest: ManifestId::from_bytes([0x33; 32]),
                logical_len: 11,
                expiry: Expiry::NEVER,
            },
        ]
    }

    fn txn_codec_cases(key: &Key) -> Vec<Mutation> {
        vec![
            Mutation::TxnPrepare {
                txn: crate::txn::TxnId::derive(3, 4, 0),
                coordinator: 9,
                key: key.clone(),
                expect: crate::txn::TxnExpect::Version(ObjectVersion::from_u64(2)),
                write: crate::txn::TxnWriteKind::Put(bytes::Bytes::from_static(b"v")),
                digest: [0xD1; 32],
            },
            Mutation::TxnPrepare {
                txn: crate::txn::TxnId::derive(3, 5, 0),
                coordinator: 9,
                key: key.clone(),
                expect: crate::txn::TxnExpect::Absent,
                write: crate::txn::TxnWriteKind::CounterAdd(-3),
                digest: [0xD2; 32],
            },
            Mutation::TxnFinalize {
                txn: crate::txn::TxnId::derive(3, 4, 0),
                key: key.clone(),
                commit: true,
                digest: [0xD1; 32],
            },
            Mutation::TxnCommitLocal {
                txn: crate::txn::TxnId::derive(3, 4, 0),
                writes: vec![
                    crate::txn::TxnWrite {
                        key: key.clone(),
                        kind: crate::txn::TxnWriteKind::Put(bytes::Bytes::from_static(b"v")),
                        expect: crate::txn::TxnExpect::Any,
                    },
                    crate::txn::TxnWrite {
                        key: Key::from("user:2"),
                        kind: crate::txn::TxnWriteKind::CounterAdd(3),
                        expect: crate::txn::TxnExpect::Absent,
                    },
                ],
            },
        ]
    }

    fn semantic_codec_cases(key: &Key) -> Vec<Mutation> {
        vec![
            Mutation::CommutativeAdd {
                key: key.clone(),
                delta: 7,
            },
            Mutation::BoundedCreate {
                key: key.clone(),
                capacity: 100,
                holder: 9,
            },
            Mutation::BoundedAdd {
                key: key.clone(),
                delta: -3,
            },
            Mutation::EscrowSetShare {
                key: key.clone(),
                min: 0,
                max: 60,
            },
            Mutation::SemaphoreCreate {
                key: key.clone(),
                capacity: 4,
            },
            Mutation::SemaphoreAcquire {
                key: key.clone(),
                permit: [0xA5; 16],
                owner: 11,
                qty: 2,
            },
            Mutation::SemaphoreRelease {
                key: key.clone(),
                permit: [0xA5; 16],
            },
            Mutation::LeaseAcquire {
                key: key.clone(),
                owner: 11,
                fencing: crate::FencingToken::from_u64(2),
                expires_at: kivi_types::WallTimestamp::from_micros(99),
            },
            Mutation::LeaseRenew {
                key: key.clone(),
                owner: 11,
                fencing: crate::FencingToken::from_u64(2),
                expires_at: kivi_types::WallTimestamp::from_micros(199),
            },
            Mutation::LeaseRelease {
                key: key.clone(),
                owner: 11,
                fencing: crate::FencingToken::from_u64(2),
            },
            Mutation::StreamCreate {
                key: key.clone(),
                stream: [0x5E; 16],
                shard: 3,
            },
            Mutation::StreamAppend {
                key: key.clone(),
                partition: bytes::Bytes::from_static(b"p"),
                payload: bytes::Bytes::from_static(b"e"),
            },
            Mutation::StreamTrim {
                key: key.clone(),
                through: 41,
            },
        ]
    }

    #[test]
    fn malformed_mutations_are_rejected() {
        assert_eq!(
            Mutation::decode(&[0xFF]),
            Err(CodecError::InvalidTag {
                kind: "mutation",
                tag: 0xFF
            })
        );
        let valid = Mutation::Delete {
            key: Key::from("k"),
        }
        .encode_to_vec();
        assert!(matches!(
            Mutation::decode(&valid[..valid.len() - 1]),
            Err(CodecError::Truncated { .. })
        ));
        assert!(matches!(
            Mutation::decode_exact(&[valid.clone(), vec![0]].concat()),
            Err(CodecError::TrailingBytes { .. })
        ));
    }
}
