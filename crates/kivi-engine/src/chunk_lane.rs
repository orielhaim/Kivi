//! Per-worker chunk lane: staging, durability proofs, verified reads.
//!
//! One OS thread per worker owns a [`ChunkStore`]
//! plus a bounded chunk cache. Every other context reaches it through ONE
//! bounded [`async_channel`] job queue, each style awaiting its own way:
//!
//! ```text
//! net connection task (reactor)  →  send().await + recv().await
//!     (suspends only the task; the reactor thread never blocks)
//! embedded bridge turn           →  try_send + poll replies per turn
//! plain worker thread / tests    →  blocking send/recv
//! ```
//!
//! A full queue fast-fails [`Overloaded`](kivi_chunk::ChunkError::Overloaded)
//! like every other saturated Kivi queue (§21, §64: all queues bounded).
//! Jobs carry their own single-slot reply channels, so completions never
//! reorder and a dead waiter can never wedge the lane (`try_send` +
//! disconnect-tolerant replies).
//!
//! The lane is also where the staging lifecycle (§10) is enforced: staging
//! appends volatile records, [`sync`](ChunkStore::sync) flips one barrier
//! for the whole batch (§34), and only [`durable`](kivi_chunk::ChunkStore::durable_manifest)
//! proofs ever become WAL roots — in exactly the chunks-first order §11
//! mandates.

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::thread::{self, JoinHandle};

use async_channel::{Receiver, Sender, bounded};
use bytes::Bytes;
use kivi_chunk::{ChunkError, ChunkStats, ChunkStore};
use kivi_types::{ChunkId, ManifestId, SecurityDomainId};

/// Lane job queue depth. Uploads are rare next to point ops; 64 deep
/// absorbs benchmark bursts while keeping worst-case queued bytes obvious
/// (64 × frame ceiling only if every slot holds a max-size legacy stage —
/// streaming assemblers stage per-chunk instead).
pub const CHUNK_JOB_DEPTH: usize = 64;

/// Default chunk-cache bound per lane: 64 MiB of immutable entries.
pub const DEFAULT_CHUNK_CACHE_BYTES: u64 = 64 * 1024 * 1024;

/// One staged value, proven durable: every chunk plus the manifest synced
/// in a single pack barrier, ready to become a small WAL root.
#[derive(Debug, Clone)]
pub struct StagedValue {
    /// Chunk addresses in logical order with their lengths.
    pub chunks: Vec<(ChunkId, u64)>,
    /// Manifest addressing the sequence.
    pub manifest: ManifestId,
    /// Canonical manifest bytes (for WAL-adjacent bookkeeping).
    pub canonical: Vec<u8>,
    /// Total logical bytes.
    pub logical_len: u64,
}

/// Single-slot reply channel. The lane `try_send`s exactly once; waiters
/// choose their style: `await`, per-turn polling, or blocking.
#[derive(Debug)]
pub struct ChunkReply<T> {
    rx: Receiver<T>,
}

impl<T> ChunkReply<T> {
    /// Non-blocking poll for bridge/worker turns.
    ///
    /// # Errors
    ///
    /// Returns [`async_channel::TryRecvError`] when no reply is ready or
    /// the lane is gone.
    pub fn try_recv(&self) -> Result<T, async_channel::TryRecvError> {
        self.rx.try_recv()
    }

    /// Reactor-style wait: suspends only the awaiting task.
    ///
    /// # Errors
    ///
    /// Returns [`async_channel::RecvError`] when the lane is gone.
    pub async fn recv_async(self) -> Result<T, async_channel::RecvError> {
        self.rx.recv().await
    }

    /// Plain-thread wait for channel-only workers and tests.
    ///
    /// # Errors
    ///
    /// Returns [`async_channel::RecvError`] when the lane is gone.
    pub fn recv_blocking(self) -> Result<T, async_channel::RecvError> {
        self.rx.recv_blocking()
    }
}

/// Work for the lane thread. Every variant carries its reply channel;
/// [`Shutdown`](ChunkJob::Shutdown) carries none.
#[derive(Debug)]
pub enum ChunkJob {
    /// Chunk a value, stage chunks + manifest, sync once, prove durable.
    StageValue {
        /// Full logical bytes (legacy path: already bounded by the frame
        /// ceiling; streaming stages per-chunk jobs instead).
        value: Bytes,
        /// Where the durable proof goes.
        reply: Sender<Result<StagedValue, ChunkError>>,
    },
    /// Resolve a manifest to its verified logical bytes (bounded by the
    /// caller's cap — legacy reads pass `MAX_LEGACY_VALUE_BYTES`).
    ReadValue {
        /// Manifest to resolve.
        manifest: ManifestId,
        /// Total logical bytes expected.
        logical_len: u64,
        /// Where the bytes go.
        reply: Sender<Result<Bytes, ChunkError>>,
    },
    /// Resolve one chunk (partial reads, splice paths, tests).
    ReadChunk {
        /// Chunk to resolve.
        id: ChunkId,
        /// Where the bytes go.
        reply: Sender<Result<Bytes, ChunkError>>,
    },
    /// Fetch one verified manifest for streaming reads: the server walks
    /// entries chunk by chunk instead of materializing the value, so
    /// downloads stay flat regardless of size.
    FetchManifest {
        /// Manifest to fetch.
        manifest: ManifestId,
        /// Where the verified manifest goes.
        reply: Sender<Result<kivi_chunk::ChunkManifest, ChunkError>>,
    },
    /// Stage one upload chunk with no sync barrier: streaming uploads
    /// append chunk after chunk, and a single barrier at commit proves
    /// them all together (see [`FinalizeStream`](ChunkJob::FinalizeStream)).
    StageChunk {
        /// Content address of these exact bytes.
        id: ChunkId,
        /// Chunk bytes.
        bytes: Bytes,
        /// Where completion goes.
        reply: Sender<Result<(), ChunkError>>,
    },
    /// Prove one streamed upload durable: store its manifest and sync
    /// once over every chunk staged since the last barrier. Answers the
    /// durable proof for a small WAL root, exactly like
    /// [`StageValue`](ChunkJob::StageValue).
    FinalizeStream {
        /// Chunk addresses in logical order with their lengths.
        entries: Vec<(ChunkId, u64)>,
        /// Total logical bytes across the entries.
        total_len: u64,
        /// Where the durable proof goes.
        reply: Sender<Result<StagedValue, ChunkError>>,
    },
    /// Patch a byte range of a chunked value without materializing it:
    /// prefix entries wholly below the patch are referenced untouched,
    /// the span from the first affected chunk boundary streams through a
    /// fresh grid (changed pieces stage as new chunks, unchanged pieces
    /// dedup to their existing addresses), and one sync proves the new
    /// manifest. Answers the durable proof for a small WAL root, exactly
    /// like [`StageValue`](ChunkJob::StageValue).
    SpliceRange {
        /// Manifest addressing the base value.
        manifest: ManifestId,
        /// Total logical bytes the tablet root claims (must agree with
        /// the manifest; disagreement is corruption, never a miss).
        logical_len: u64,
        /// Logical byte index the patch overwrites from.
        offset: u64,
        /// Bytes written starting at `offset` (past-the-end gaps
        /// zero-pad, exactly like inline splices).
        patch: Bytes,
        /// Where the durable proof goes.
        reply: Sender<Result<StagedValue, ChunkError>>,
    },
    /// Snapshot lane counters for the admin plane.
    Stats {
        /// Where the snapshot goes.
        reply: Sender<ChunkLaneStats>,
    },
    /// Rotate the active pack now (tests, GC-time simplification).
    Seal {
        /// Where completion goes.
        reply: Sender<Result<(), ChunkError>>,
    },
    /// Delete sealed packs by sequence (GC). Best-effort per pack.
    DeletePacks {
        /// Pack sequences to remove (never the active tail).
        seqs: Vec<u64>,
        /// Where the actually-deleted sequences go.
        reply: Sender<Result<Vec<u64>, ChunkError>>,
    },
    /// Prove one chunked root's dependencies: manifest indexed and
    /// readable, every entry's chunk indexed, lengths agreeing. Startup
    /// and recovery gate on this (§30, §31); serving reads re-verify.
    CheckRoot {
        /// Manifest to prove.
        manifest: ManifestId,
        /// Logical length the root claims.
        logical_len: u64,
        /// Where the proof goes (`Ok(())` proven).
        reply: Sender<Result<(), ChunkError>>,
    },
    /// Plan mark GC over this lane: expand live manifests, census packs,
    /// report fully-dead sealed packs. Read-only (maintenance context may
    /// block on manifest reads).
    GcMark {
        /// Manifests referenced by live tablets, retained checkpoints,
        /// and required WAL history.
        live: Vec<ManifestId>,
        /// In-flight staged chunk addresses.
        pinned_chunks: Vec<ChunkId>,
        /// In-flight staged manifest addresses.
        pinned_manifests: Vec<ManifestId>,
        /// Where the plan goes.
        reply: Sender<Result<kivi_chunk::GcReport, ChunkError>>,
    },
    /// Drain and exit (engine shutdown; workers stop submitting first).
    Shutdown,
}

/// Lane counters for the admin plane: store truth plus cache truth.
#[derive(Debug, Clone, Copy, Default)]
pub struct ChunkLaneStats {
    /// Store counters (packs, records, dedup, syncs).
    pub store: ChunkStats,
    /// Cache resident bytes.
    pub cache_bytes: u64,
    /// Cache entries.
    pub cache_entries: u64,
    /// Cache hits (chunk reads served without pack I/O).
    pub cache_hits: u64,
    /// Cache misses (chunk reads that hit pack files).
    pub cache_misses: u64,
}

/// Bounded immutable chunk cache: chunk id to verified bytes, FIFO
/// eviction under a byte cap. Entries are immutable content addresses, so
/// eviction is always safe (a miss re-reads and re-verifies). Worker-local
/// by construction: it lives on the lane thread, never shared.
#[derive(Debug, Default)]
struct ChunkCache {
    entries: HashMap<ChunkId, Bytes>,
    order: VecDeque<ChunkId>,
    bytes: u64,
    cap: u64,
    hits: u64,
    misses: u64,
}

impl ChunkCache {
    fn with_cap(cap: u64) -> Self {
        Self {
            cap: cap.max(1),
            ..Self::default()
        }
    }

    fn get(&mut self, id: &ChunkId) -> Option<Bytes> {
        if let Some(bytes) = self.entries.get(id) {
            self.hits += 1;
            Some(bytes.clone())
        } else {
            self.misses += 1;
            None
        }
    }

    fn insert(&mut self, id: ChunkId, bytes: Bytes) {
        if self.entries.contains_key(&id) {
            return;
        }
        let len = bytes.len() as u64;
        // Oversize single chunks bypass the cache (still served, just not
        // retained): the cap bounds residency, never servability.
        if len > self.cap {
            return;
        }
        while !self.order.is_empty() && self.bytes + len > self.cap {
            if let Some(old) = self.order.pop_front()
                && let Some(removed) = self.entries.remove(&old)
            {
                self.bytes -= removed.len() as u64;
            } else {
                break;
            }
        }
        self.order.push_back(id);
        self.bytes += len;
        self.entries.insert(id, bytes);
    }
}

/// Clonable handle to one worker's chunk lane. Cheap: one channel sender.
#[derive(Debug, Clone)]
pub struct ChunkLaneHandle {
    tx: Sender<ChunkJob>,
    worker: kivi_types::WorkerId,
}

impl ChunkLaneHandle {
    /// Returns the worker this lane serves (tracing, admin, GC planning).
    #[must_use]
    pub const fn worker(&self) -> kivi_types::WorkerId {
        self.worker
    }

    /// Stages one value end to end (chunk, manifest, one sync barrier) and
    /// returns the durable proof. Reactor-style wait.
    ///
    /// # Errors
    ///
    /// Returns [`ChunkError::Overloaded`] when the bounded lane queue is
    /// full or the lane is gone, or the staging failure itself.
    pub async fn stage_value_async(&self, value: Bytes) -> Result<StagedValue, ChunkError> {
        let reply = self.submit_stage(value)?;
        reply
            .recv_async()
            .await
            .map_err(|_| ChunkError::Overloaded)?
    }

    /// Blocking twin for plain worker threads and tests.
    ///
    /// # Errors
    ///
    /// Returns [`ChunkError::Overloaded`] when the bounded lane queue is
    /// full or the lane is gone, or the staging failure itself.
    pub fn stage_value_blocking(&self, value: Bytes) -> Result<StagedValue, ChunkError> {
        let reply = self.submit_stage(value)?;
        reply.recv_blocking().map_err(|_| ChunkError::Overloaded)?
    }

    /// Verifies one chunked root names durable local payload (manifest
    /// indexed, lengths agree, every chunk present). Reactor-style wait
    /// for async admission paths; transaction admission proves every
    /// chunked reference this way before a participant may say Prepared.
    ///
    /// # Errors
    ///
    /// Returns [`ChunkError::Overloaded`] when the bounded lane queue is
    /// full or the lane is gone, or the verification failure (missing or
    /// corrupt data fails loudly — a commit must never reference it).
    pub async fn check_root_async(
        &self,
        manifest: ManifestId,
        logical_len: u64,
    ) -> Result<(), ChunkError> {
        let (reply_tx, reply_rx) = bounded(1);
        self.tx
            .try_send(ChunkJob::CheckRoot {
                manifest,
                logical_len,
                reply: reply_tx,
            })
            .map_err(|_| ChunkError::Overloaded)?;
        ChunkReply { rx: reply_rx }
            .recv_async()
            .await
            .map_err(|_| ChunkError::Overloaded)?
    }

    /// Non-blocking submit for bridge turns: returns the reply to poll.
    ///
    /// # Errors
    ///
    /// Returns [`ChunkError::Overloaded`] when the bounded lane queue is full.
    pub fn submit_stage(
        &self,
        value: Bytes,
    ) -> Result<ChunkReply<Result<StagedValue, ChunkError>>, ChunkError> {
        let (reply_tx, reply_rx) = bounded(1);
        self.tx
            .try_send(ChunkJob::StageValue {
                value,
                reply: reply_tx,
            })
            .map_err(|_| ChunkError::Overloaded)?;
        Ok(ChunkReply { rx: reply_rx })
    }

    /// Resolves one manifest to verified bytes. Reactor-style wait.
    ///
    /// # Errors
    ///
    /// Returns [`ChunkError::Overloaded`] when the bounded lane queue is
    /// full or the lane is gone, or the resolution failure (missing or
    /// corrupt data fails loudly, never wrong bytes).
    pub async fn read_value_async(
        &self,
        manifest: ManifestId,
        logical_len: u64,
    ) -> Result<Bytes, ChunkError> {
        let reply = self.submit_read(manifest, logical_len)?;
        reply
            .recv_async()
            .await
            .map_err(|_| ChunkError::Overloaded)?
    }

    /// Blocking twin for plain worker threads and tests.
    ///
    /// # Errors
    ///
    /// Returns [`ChunkError::Overloaded`] when the bounded lane queue is
    /// full or the lane is gone, or the resolution failure itself.
    pub fn read_value_blocking(
        &self,
        manifest: ManifestId,
        logical_len: u64,
    ) -> Result<Bytes, ChunkError> {
        let reply = self.submit_read(manifest, logical_len)?;
        reply.recv_blocking().map_err(|_| ChunkError::Overloaded)?
    }

    /// Non-blocking submit for bridge turns.
    ///
    /// # Errors
    ///
    /// Returns [`ChunkError::Overloaded`] when the bounded lane queue is full.
    pub fn submit_read(
        &self,
        manifest: ManifestId,
        logical_len: u64,
    ) -> Result<ChunkReply<Result<Bytes, ChunkError>>, ChunkError> {
        let (reply_tx, reply_rx) = bounded(1);
        self.tx
            .try_send(ChunkJob::ReadValue {
                manifest,
                logical_len,
                reply: reply_tx,
            })
            .map_err(|_| ChunkError::Overloaded)?;
        Ok(ChunkReply { rx: reply_rx })
    }

    /// Patches a chunked value's byte range end to end (prefix reuse,
    /// streamed re-chunk, one sync barrier) and returns the durable proof.
    /// Reactor-style wait.
    ///
    /// # Errors
    ///
    /// Returns [`ChunkError::Overloaded`] when the bounded lane queue is
    /// full or the lane is gone, or the splice failure itself (corrupt
    /// base, overflowing range, entry-count bound).
    pub async fn splice_range_async(
        &self,
        manifest: ManifestId,
        logical_len: u64,
        offset: u64,
        patch: Bytes,
    ) -> Result<StagedValue, ChunkError> {
        let reply = self.submit_splice(manifest, logical_len, offset, patch)?;
        reply
            .recv_async()
            .await
            .map_err(|_| ChunkError::Overloaded)?
    }

    /// Blocking twin for plain worker threads and tests.
    ///
    /// # Errors
    ///
    /// Returns [`ChunkError::Overloaded`] when the bounded lane queue is
    /// full or the lane is gone, or the splice failure itself.
    pub fn splice_range_blocking(
        &self,
        manifest: ManifestId,
        logical_len: u64,
        offset: u64,
        patch: Bytes,
    ) -> Result<StagedValue, ChunkError> {
        let reply = self.submit_splice(manifest, logical_len, offset, patch)?;
        reply.recv_blocking().map_err(|_| ChunkError::Overloaded)?
    }

    /// Non-blocking submit for bridge turns.
    ///
    /// # Errors
    ///
    /// Returns [`ChunkError::Overloaded`] when the bounded lane queue is full.
    pub fn submit_splice(
        &self,
        manifest: ManifestId,
        logical_len: u64,
        offset: u64,
        patch: Bytes,
    ) -> Result<ChunkReply<Result<StagedValue, ChunkError>>, ChunkError> {
        let (reply_tx, reply_rx) = bounded(1);
        self.tx
            .try_send(ChunkJob::SpliceRange {
                manifest,
                logical_len,
                offset,
                patch,
                reply: reply_tx,
            })
            .map_err(|_| ChunkError::Overloaded)?;
        Ok(ChunkReply { rx: reply_rx })
    }

    /// Resolves one chunk (blocking twin covers tests; async for reactor).
    ///
    /// # Errors
    ///
    /// Returns [`ChunkError::Overloaded`] when the bounded lane queue is
    /// full or the lane is gone, or the resolution failure itself.
    pub async fn read_chunk_async(&self, id: ChunkId) -> Result<Bytes, ChunkError> {
        let (reply_tx, reply_rx) = bounded(1);
        self.tx
            .try_send(ChunkJob::ReadChunk {
                id,
                reply: reply_tx,
            })
            .map_err(|_| ChunkError::Overloaded)?;
        reply_rx.recv().await.map_err(|_| ChunkError::Overloaded)?
    }

    /// Fetches one verified manifest for streaming reads. Reactor-style wait.
    ///
    /// # Errors
    ///
    /// Returns [`ChunkError::Overloaded`] when the bounded lane queue is
    /// full or the lane is gone, or the fetch failure itself.
    pub async fn fetch_manifest_async(
        &self,
        manifest: ManifestId,
    ) -> Result<kivi_chunk::ChunkManifest, ChunkError> {
        let (reply_tx, reply_rx) = bounded(1);
        self.tx
            .try_send(ChunkJob::FetchManifest {
                manifest,
                reply: reply_tx,
            })
            .map_err(|_| ChunkError::Overloaded)?;
        reply_rx.recv().await.map_err(|_| ChunkError::Overloaded)?
    }

    /// Blocking twin for plain worker threads and tests.
    ///
    /// # Errors
    ///
    /// Returns [`ChunkError::Overloaded`] when the bounded lane queue is
    /// full or the lane is gone, or the fetch failure itself.
    pub fn fetch_manifest_blocking(
        &self,
        manifest: ManifestId,
    ) -> Result<kivi_chunk::ChunkManifest, ChunkError> {
        let (reply_tx, reply_rx) = bounded(1);
        self.tx
            .try_send(ChunkJob::FetchManifest {
                manifest,
                reply: reply_tx,
            })
            .map_err(|_| ChunkError::Overloaded)?;
        reply_rx
            .recv_blocking()
            .map_err(|_| ChunkError::Overloaded)?
    }

    /// Stages one upload chunk with no sync barrier (streaming uploads
    /// batch the barrier at commit). Reactor-style wait.
    ///
    /// # Errors
    ///
    /// Returns [`ChunkError::Overloaded`] when the bounded lane queue is
    /// full or the lane is gone, or the staging failure itself.
    pub async fn stage_chunk_async(&self, id: ChunkId, bytes: Bytes) -> Result<(), ChunkError> {
        let (reply_tx, reply_rx) = bounded(1);
        self.tx
            .try_send(ChunkJob::StageChunk {
                id,
                bytes,
                reply: reply_tx,
            })
            .map_err(|_| ChunkError::Overloaded)?;
        reply_rx.recv().await.map_err(|_| ChunkError::Overloaded)?
    }

    /// Blocking twin for plain worker threads and tests.
    ///
    /// # Errors
    ///
    /// Returns [`ChunkError::Overloaded`] when the bounded lane queue is
    /// full or the lane is gone, or the staging failure itself.
    pub fn stage_chunk_blocking(&self, id: ChunkId, bytes: Bytes) -> Result<(), ChunkError> {
        let (reply_tx, reply_rx) = bounded(1);
        self.tx
            .try_send(ChunkJob::StageChunk {
                id,
                bytes,
                reply: reply_tx,
            })
            .map_err(|_| ChunkError::Overloaded)?;
        reply_rx
            .recv_blocking()
            .map_err(|_| ChunkError::Overloaded)?
    }

    /// Proves one streamed upload durable (manifest plus one sync barrier
    /// over every chunk staged since the last barrier). Reactor-style wait.
    ///
    /// # Errors
    ///
    /// Returns [`ChunkError::Overloaded`] when the bounded lane queue is
    /// full or the lane is gone, or the finalization failure itself.
    pub async fn finalize_stream_async(
        &self,
        entries: Vec<(ChunkId, u64)>,
        total_len: u64,
    ) -> Result<StagedValue, ChunkError> {
        let (reply_tx, reply_rx) = bounded(1);
        self.tx
            .try_send(ChunkJob::FinalizeStream {
                entries,
                total_len,
                reply: reply_tx,
            })
            .map_err(|_| ChunkError::Overloaded)?;
        reply_rx.recv().await.map_err(|_| ChunkError::Overloaded)?
    }

    /// Blocking twin for plain worker threads and tests.
    ///
    /// # Errors
    ///
    /// Returns [`ChunkError::Overloaded`] when the bounded lane queue is
    /// full or the lane is gone, or the finalization failure itself.
    pub fn finalize_stream_blocking(
        &self,
        entries: Vec<(ChunkId, u64)>,
        total_len: u64,
    ) -> Result<StagedValue, ChunkError> {
        let (reply_tx, reply_rx) = bounded(1);
        self.tx
            .try_send(ChunkJob::FinalizeStream {
                entries,
                total_len,
                reply: reply_tx,
            })
            .map_err(|_| ChunkError::Overloaded)?;
        reply_rx
            .recv_blocking()
            .map_err(|_| ChunkError::Overloaded)?
    }

    /// Snapshots lane counters (blocking; admin/maintenance/test paths).
    ///
    /// # Errors
    ///
    /// Returns [`ChunkError::Overloaded`] when the bounded lane queue is
    /// full or the lane is gone.
    pub fn stats_blocking(&self) -> Result<ChunkLaneStats, ChunkError> {
        let (reply_tx, reply_rx) = bounded(1);
        self.tx
            .try_send(ChunkJob::Stats { reply: reply_tx })
            .map_err(|_| ChunkError::Overloaded)?;
        reply_rx.recv_blocking().map_err(|_| ChunkError::Overloaded)
    }

    /// Plans mark GC over the lane (blocking; checkpoint worker only).
    /// `live` merges retained-checkpoint band refs with journaled commits;
    /// pins join inside the lane job from the caller-supplied sets.
    ///
    /// # Errors
    ///
    /// Returns [`ChunkError::Overloaded`] when the bounded lane queue is
    /// full or the lane is gone, or a missing live root (corruption).
    pub fn gc_mark_blocking(
        &self,
        live: Vec<ManifestId>,
        pinned_chunks: Vec<ChunkId>,
        pinned_manifests: Vec<ManifestId>,
    ) -> Result<kivi_chunk::GcReport, ChunkError> {
        let (reply_tx, reply_rx) = bounded(1);
        self.tx
            .try_send(ChunkJob::GcMark {
                live,
                pinned_chunks,
                pinned_manifests,
                reply: reply_tx,
            })
            .map_err(|_| ChunkError::Overloaded)?;
        reply_rx
            .recv_blocking()
            .map_err(|_| ChunkError::Overloaded)?
    }

    /// Proves one chunked root's dependencies (blocking; startup and
    /// recovery gates only).
    ///
    /// # Errors
    ///
    /// Returns [`ChunkError::Overloaded`] when the bounded lane queue is
    /// full or the lane is gone, or the dependency failure (missing or
    /// corrupt data fails loudly).
    pub fn check_root_blocking(
        &self,
        manifest: ManifestId,
        logical_len: u64,
    ) -> Result<(), ChunkError> {
        let (reply_tx, reply_rx) = bounded(1);
        self.tx
            .try_send(ChunkJob::CheckRoot {
                manifest,
                logical_len,
                reply: reply_tx,
            })
            .map_err(|_| ChunkError::Overloaded)?;
        reply_rx
            .recv_blocking()
            .map_err(|_| ChunkError::Overloaded)?
    }

    /// Deletes sealed packs by sequence (blocking; checkpoint worker only).
    /// Best-effort per pack: failures keep files (disk usage, never
    /// corruption).
    ///
    /// # Errors
    ///
    /// Returns [`ChunkError::Overloaded`] when the bounded lane queue is
    /// full or the lane is gone, or refuses the active tail.
    pub fn delete_packs_blocking(&self, seqs: Vec<u64>) -> Result<Vec<u64>, ChunkError> {
        let (reply_tx, reply_rx) = bounded(1);
        self.tx
            .try_send(ChunkJob::DeletePacks {
                seqs,
                reply: reply_tx,
            })
            .map_err(|_| ChunkError::Overloaded)?;
        reply_rx
            .recv_blocking()
            .map_err(|_| ChunkError::Overloaded)?
    }

    /// Lane-thread body: owns the store and cache, serves jobs to
    /// completion, exits on [`Shutdown`](ChunkJob::Shutdown) or channel
    /// close (all handles dropped). Replies are `try_send`: a vanished
    /// waiter (worker gone) never wedges the lane.
    fn serve(mut store: ChunkStore, jobs: &Receiver<ChunkJob>, cache_cap: u64) {
        let mut cache = ChunkCache::with_cap(cache_cap);
        let domain = store.domain();
        while let Ok(job) = jobs.recv_blocking() {
            match job {
                ChunkJob::StageValue { value, reply } => {
                    let _ = reply.try_send(stage_value_job(&mut store, domain, &value));
                }
                ChunkJob::ReadValue {
                    manifest,
                    logical_len,
                    reply,
                } => {
                    let _ = reply.try_send(read_value_job(
                        &mut store,
                        &mut cache,
                        manifest,
                        logical_len,
                    ));
                }
                ChunkJob::ReadChunk { id, reply } => {
                    let _ = reply.try_send(read_chunk_cached(&mut store, &mut cache, id));
                }
                ChunkJob::FetchManifest { manifest, reply } => {
                    let _ = reply.try_send(store.read_manifest(manifest).map(|(loaded, _)| loaded));
                }
                ChunkJob::StageChunk { id, bytes, reply } => {
                    let _ = reply.try_send(store.stage_chunk(id, &bytes).map(|_| ()));
                }
                ChunkJob::FinalizeStream {
                    entries,
                    total_len,
                    reply,
                } => {
                    let _ =
                        reply.try_send(finalize_stream_job(&mut store, domain, entries, total_len));
                }
                ChunkJob::SpliceRange {
                    manifest,
                    logical_len,
                    offset,
                    patch,
                    reply,
                } => {
                    let _ = reply.try_send(splice_range_job(
                        &mut store,
                        &mut cache,
                        domain,
                        manifest,
                        logical_len,
                        offset,
                        &patch,
                    ));
                }
                ChunkJob::Stats { reply } => {
                    let _ = reply.try_send(ChunkLaneStats {
                        store: store.stats(),
                        cache_bytes: cache.bytes,
                        cache_entries: cache.entries.len() as u64,
                        cache_hits: cache.hits,
                        cache_misses: cache.misses,
                    });
                }
                ChunkJob::Seal { reply } => {
                    let _ = reply.try_send(store.seal_active());
                }
                ChunkJob::DeletePacks { seqs, reply } => {
                    let _ = reply.try_send(store.delete_packs(&seqs));
                }
                ChunkJob::CheckRoot {
                    manifest,
                    logical_len,
                    reply,
                } => {
                    let _ = reply.try_send(check_root(&store, manifest, logical_len));
                }
                ChunkJob::GcMark {
                    live,
                    pinned_chunks,
                    pinned_manifests,
                    reply,
                } => {
                    let inputs = kivi_chunk::GcInputs {
                        live_manifests: live.into_iter().collect(),
                        pinned_chunks: pinned_chunks.into_iter().collect(),
                        pinned_manifests: pinned_manifests.into_iter().collect(),
                    };
                    let _ = reply.try_send(kivi_chunk::plan_gc(&store, &inputs));
                }
                ChunkJob::Shutdown => break,
            }
        }
    }
}

/// Spawns one lane thread owning `store`. Returns the shared handle plus a
/// join guard the engine holds for shutdown ordering (workers first, then
/// lanes: no worker submits after it exits).
///
/// # Panics
///
/// Panics when the OS refuses the lane thread: resource exhaustion at
/// startup is fatal, mirroring worker spawn handling.
#[must_use]
pub fn spawn_lane(
    worker: kivi_types::WorkerId,
    store: ChunkStore,
    cache_cap: u64,
) -> (ChunkLaneHandle, ChunkLaneGuard) {
    let (tx, rx) = bounded::<ChunkJob>(CHUNK_JOB_DEPTH);
    let thread = thread::Builder::new()
        .name(format!("kivi-chunk-{}", worker.as_u64()))
        .spawn(move || ChunkLaneHandle::serve(store, &rx, cache_cap))
        .expect("chunk lane thread spawns");
    (
        ChunkLaneHandle { tx, worker },
        ChunkLaneGuard {
            thread: Some(thread),
        },
    )
}

/// Engine-held lane ownership: shuts the thread down and joins it.
#[derive(Debug)]
pub struct ChunkLaneGuard {
    thread: Option<JoinHandle<()>>,
}

impl ChunkLaneGuard {
    /// Sends [`Shutdown`](ChunkJob::Shutdown) and joins. Safe to call with
    /// jobs still queued: lane threads never touch tablet state, so
    /// abandoning queued chunk work strands no logical state (waiters fail
    /// on the severed channel).
    pub fn shutdown(&mut self, handle: &ChunkLaneHandle) {
        let _ = handle.tx.try_send(ChunkJob::Shutdown);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

/// Stages chunks + manifest + one sync barrier for one value. Pure lane
/// work: deterministic chunking, content addressing, dedup-aware appends.
fn stage_value_job(
    store: &mut ChunkStore,
    domain: SecurityDomainId,
    value: &Bytes,
) -> Result<StagedValue, ChunkError> {
    use kivi_chunk::{Chunking, build_manifest};
    use kivi_codec::integrity::chunk_id;

    let chunking = Chunking::DEFAULT;
    chunking.validate()?;
    let total = value.len() as u64;
    let mut chunks = Vec::new();
    let mut entries: Vec<kivi_chunk::ChunkEntry> = Vec::new();
    for piece_len in chunking.split_lengths(total)? {
        let len = usize::try_from(piece_len).map_err(|_| ChunkError::TooLarge {
            len: piece_len,
            max: u64::MAX,
            context: "chunk piece",
        })?;
        let start = entries
            .iter()
            .map(|entry| entry.len)
            .sum::<u64>()
            .try_into()
            .map_err(|_| ChunkError::TooLarge {
                len: total,
                max: u64::MAX,
                context: "chunk offset",
            })?;
        let piece = &value[start..start + len];
        let id = chunk_id(domain, piece);
        store.stage_chunk(id, piece)?;
        chunks.push((id, piece_len));
        entries.push(kivi_chunk::ChunkEntry { id, len: piece_len });
    }
    let (manifest, manifest_id) = build_manifest(
        domain,
        chunking,
        kivi_chunk::ChunkCodecId::NONE,
        total,
        entries,
    )?;
    let (canonical, recomputed) = manifest.canonical_with_id(domain);
    debug_assert_eq!(manifest_id, recomputed);
    store.stage_manifest(manifest_id, &canonical)?;
    // One barrier for the whole value (§34): everything staged above
    // becomes durable together, or nothing new does.
    store.sync()?;
    Ok(StagedValue {
        chunks,
        manifest: manifest_id,
        canonical,
        logical_len: total,
    })
}

/// Proves one streamed upload durable: builds its manifest over the
/// caller-staged entries, stages the manifest, and syncs once over
/// everything staged since the last barrier. Pure lane work; the caller
/// staged every entry through [`ChunkJob::StageChunk`] (dedup makes
/// re-staged bytes free), so this only adds the manifest record plus the
/// barrier.
fn finalize_stream_job(
    store: &mut ChunkStore,
    domain: SecurityDomainId,
    staged: Vec<(ChunkId, u64)>,
    total_len: u64,
) -> Result<StagedValue, ChunkError> {
    use kivi_chunk::{Chunking, build_manifest};

    let chunking = Chunking::DEFAULT;
    chunking.validate()?;
    let entries: Vec<kivi_chunk::ChunkEntry> = staged
        .iter()
        .map(|(id, len)| kivi_chunk::ChunkEntry { id: *id, len: *len })
        .collect();
    let (manifest, manifest_id) = build_manifest(
        domain,
        chunking,
        kivi_chunk::ChunkCodecId::NONE,
        total_len,
        entries,
    )?;
    let (canonical, recomputed) = manifest.canonical_with_id(domain);
    debug_assert_eq!(manifest_id, recomputed);
    store.stage_manifest(manifest_id, &canonical)?;
    // One barrier for the whole upload (§34): everything staged since
    // the last barrier becomes durable together, or nothing new does.
    store.sync()?;
    Ok(StagedValue {
        chunks: staged,
        manifest: manifest_id,
        canonical,
        logical_len: total_len,
    })
}

/// Resolves a manifest through cache + verified pack reads, checking the
/// assembled length against the root's claim.
fn read_value_job(
    store: &mut ChunkStore,
    cache: &mut ChunkCache,
    manifest: ManifestId,
    logical_len: u64,
) -> Result<Bytes, ChunkError> {
    if logical_len > kivi_chunk::policy::MAX_LEGACY_VALUE_BYTES {
        return Err(ChunkError::TooLarge {
            len: logical_len,
            max: kivi_chunk::policy::MAX_LEGACY_VALUE_BYTES,
            context: "legacy chunked read (use streaming)",
        });
    }
    let (loaded, _) = store.read_manifest(manifest)?;
    if loaded.total_len != logical_len {
        return Err(ChunkError::CorruptManifest {
            id: manifest,
            detail: format!(
                "manifest total {} disagrees with root claim {logical_len}",
                loaded.total_len
            ),
        });
    }
    // Bounded above by `MAX_LEGACY_VALUE_BYTES` (64 MiB), so the
    // conversion below cannot fail on any supported address space.
    let capacity = usize::try_from(logical_len).expect("capped length fits address space");
    let mut out = Vec::with_capacity(capacity);
    for entry in &loaded.entries {
        let bytes = read_chunk_cached(store, cache, entry.id)?;
        if bytes.len() as u64 != entry.len {
            return Err(ChunkError::CorruptChunk {
                id: entry.id,
                detail: "chunk body length disagrees with its manifest entry".to_owned(),
            });
        }
        out.extend_from_slice(&bytes);
    }
    if out.len() as u64 != logical_len {
        return Err(ChunkError::CorruptManifest {
            id: manifest,
            detail: "assembled bytes disagree with manifest total".to_owned(),
        });
    }
    Ok(Bytes::from(out))
}

/// Patches a chunked value's byte range without materializing it.
/// Prefix entries wholly below the patch stay referenced untouched (never
/// even read); the span from the first affected chunk boundary streams
/// through a fresh fixed-size grid, so changed pieces stage as new chunks
/// while unchanged pieces dedup to their existing addresses. Peak lane
/// residency is two chunks plus the patch plus one piece under fill —
/// never the value. One [`sync`](ChunkStore::sync) proves the new
/// manifest; the reply carries only the newly staged addresses (prefix
/// chunks stay covered by the old root until commit, then by the new
/// one — and every staged byte lands in the GC-immune active tail, so
/// the worker pinning on receipt closes the only exposure window).
fn splice_range_job(
    store: &mut ChunkStore,
    cache: &mut ChunkCache,
    domain: SecurityDomainId,
    manifest: ManifestId,
    logical_len: u64,
    offset: u64,
    patch: &[u8],
) -> Result<StagedValue, ChunkError> {
    use kivi_chunk::build_manifest;
    use kivi_codec::integrity::chunk_id;

    let mut plan = splice_plan(store, manifest, logical_len, offset, patch.len() as u64)?;
    // Aligned no-op fast path: an empty patch on an entry edge with no
    // gap to pad reuses every entry by reference — no chunk reads at all.
    if patch.is_empty() && offset <= logical_len && offset == plan.prefix_len {
        let mut entries = std::mem::take(&mut plan.prefix);
        entries.extend_from_slice(&plan.entries[plan.boundary..]);
        let (built, id) = build_manifest(
            domain,
            plan.chunking,
            kivi_chunk::ChunkCodecId::NONE,
            plan.new_total,
            entries,
        )?;
        let (canonical, recomputed) = built.canonical_with_id(domain);
        debug_assert_eq!(id, recomputed);
        store.stage_manifest(id, &canonical)?;
        store.sync()?;
        return Ok(StagedValue {
            chunks: Vec::new(),
            manifest: id,
            canonical,
            logical_len: plan.new_total,
        });
    }
    let mut stream = SpliceStream::open(store, cache, &plan, offset, patch, logical_len)?;
    let skip_pieces = splice_us(plan.prefix_len / plan.chunk_size, "splice prefix")?;
    let mut staged = Vec::new();
    let mut entries = std::mem::take(&mut plan.prefix);
    // Index, never `into_iter`: the plan stays borrowed whole while the
    // stream fills each piece from it.
    for index in skip_pieces..plan.pieces.len() {
        let piece_len = plan.pieces[index];
        let piece = stream.fill_piece(
            store,
            &plan,
            manifest,
            splice_us(piece_len, "splice piece")?,
        )?;
        let id = chunk_id(domain, &piece);
        store.stage_chunk(id, &piece)?;
        staged.push((id, piece_len));
        entries.push(kivi_chunk::ChunkEntry { id, len: piece_len });
    }
    let (built, id) = build_manifest(
        domain,
        plan.chunking,
        kivi_chunk::ChunkCodecId::NONE,
        plan.new_total,
        entries,
    )?;
    let (canonical, recomputed) = built.canonical_with_id(domain);
    debug_assert_eq!(id, recomputed);
    store.stage_manifest(id, &canonical)?;
    // One barrier for the splice (§34): new chunks plus the new manifest
    // become durable together, or nothing new does.
    store.sync()?;
    Ok(StagedValue {
        chunks: staged,
        manifest: id,
        canonical,
        logical_len: plan.new_total,
    })
}

/// Checked `u64` to `usize` for splice arithmetic: grid and slice
/// lengths always fit the address space on supported targets, and a
/// hostile manifest length fails here instead of truncating.
fn splice_us(value: u64, context: &'static str) -> Result<usize, ChunkError> {
    usize::try_from(value).map_err(|_| ChunkError::TooLarge {
        len: value,
        max: u64::MAX,
        context,
    })
}

/// Validated splice geometry: the base entries plus the retained prefix,
/// the fresh-grid pieces, and the patch extent. Pure data over the
/// manifest — no chunk I/O — so streaming agrees with planning by
/// construction.
struct SplicePlan {
    /// Base manifest entries in logical order.
    entries: Vec<kivi_chunk::ChunkEntry>,
    /// Grid the spliced value chunks on (the base manifest's own).
    chunking: kivi_chunk::Chunking,
    /// Logical bytes per full chunk.
    chunk_size: u64,
    /// Spliced total: the old length grown past the patch end as needed.
    new_total: u64,
    /// Fresh-grid piece lengths for `new_total`.
    pieces: Vec<u64>,
    /// Entries wholly below the patch, referenced untouched.
    prefix: Vec<kivi_chunk::ChunkEntry>,
    /// Logical bytes the prefix covers (a chunk-size multiple).
    prefix_len: u64,
    /// First non-retained entry (read and re-chunked by the stream).
    boundary: usize,
    /// `offset + patch` length (checked).
    patch_end: u64,
}

/// Validates the base manifest against the root claim and computes the
/// splice geometry: support checks, the entry-count bound, and the
/// prefix scan. No chunk I/O — failures here touch nothing.
fn splice_plan(
    store: &mut ChunkStore,
    manifest: ManifestId,
    logical_len: u64,
    offset: u64,
    patch_len: u64,
) -> Result<SplicePlan, ChunkError> {
    use kivi_chunk::Chunking;

    let (loaded, _) = store.read_manifest(manifest)?;
    if loaded.total_len != logical_len {
        return Err(ChunkError::CorruptManifest {
            id: manifest,
            detail: format!(
                "manifest total {} disagrees with root claim {logical_len}",
                loaded.total_len
            ),
        });
    }
    if loaded.chunking_version != kivi_chunk::policy::CHUNKING_V1 {
        return Err(ChunkError::Unsupported {
            detail: format!(
                "splice needs chunking V1, manifest names {}",
                loaded.chunking_version
            ),
        });
    }
    if loaded.codec != kivi_chunk::ChunkCodecId::NONE {
        return Err(ChunkError::Unsupported {
            detail: format!(
                "splice needs the NONE codec, manifest names {}",
                loaded.codec
            ),
        });
    }
    let chunk_size = loaded.chunk_size;
    if chunk_size == 0 {
        return Err(ChunkError::CorruptManifest {
            id: manifest,
            detail: "manifest names a zero chunk size".to_owned(),
        });
    }
    let chunking = Chunking {
        version: kivi_chunk::policy::CHUNKING_V1,
        chunk_size: splice_us(chunk_size, "splice chunk size")?,
    };
    chunking.validate()?;
    let patch_end = offset
        .checked_add(patch_len)
        .ok_or_else(|| ChunkError::Invalid {
            detail: "splice range overflows u64".to_owned(),
        })?;
    let new_total = logical_len.max(patch_end);
    // Entry-count bound before any chunk I/O: the spliced value chunks
    // on the same grid, so its piece count is the fresh-grid count.
    let pieces = chunking.split_lengths(new_total)?;
    if pieces.len() > kivi_chunk::MAX_CHUNKS_PER_MANIFEST {
        return Err(ChunkError::TooLarge {
            len: pieces.len() as u64,
            max: kivi_chunk::MAX_CHUNKS_PER_MANIFEST as u64,
            context: "spliced manifest entries",
        });
    }
    // Prefix scan: retain whole full-size entries fully below the patch.
    // Stops at the boundary entry (read by the stream) or a short entry
    // (only the final entry may be short; anything earlier is corruption,
    // and stopping re-chunks it instead of trusting it). The scan only
    // ever advances `prefix_len` toward `offset`, so it always terminates.
    let mut prefix = Vec::new();
    let mut prefix_len = 0u64;
    let mut boundary = 0usize;
    while boundary < loaded.entries.len() {
        let entry = loaded.entries[boundary];
        let end = prefix_len
            .checked_add(entry.len)
            .ok_or_else(|| ChunkError::Invalid {
                detail: "splice prefix offsets overflow u64".to_owned(),
            })?;
        if entry.len != chunk_size || end > offset {
            break;
        }
        prefix.push(entry);
        prefix_len = end;
        boundary += 1;
    }
    Ok(SplicePlan {
        entries: loaded.entries,
        chunking,
        chunk_size,
        new_total,
        pieces,
        prefix,
        prefix_len,
        boundary,
        patch_end,
    })
}

/// Rejects chunk bytes whose length disagrees with the manifest entry:
/// a lying pack record is corruption, never served.
fn verify_entry(entry: kivi_chunk::ChunkEntry, bytes: &Bytes) -> Result<(), ChunkError> {
    if bytes.len() as u64 != entry.len {
        return Err(ChunkError::CorruptChunk {
            id: entry.id,
            detail: "chunk body length disagrees with its manifest entry".to_owned(),
        });
    }
    Ok(())
}

/// Open region stream for [`splice_range_job`]: head-old, zero gap,
/// patch, tail-old. Reads at most the two boundary chunks (cached); the
/// tail streams from packs on demand without evicting the hot cache.
struct SpliceStream {
    /// Queued segments in positional order.
    segments: std::collections::VecDeque<SpliceSegment>,
    /// First tail entry needing a pack read.
    tail_from: usize,
    /// Overwritten head to skip in the first tail read.
    tail_skip: u64,
}

impl SpliceStream {
    /// Opens the stream: queues the boundary spans and locates the tail.
    /// Reads at most two chunks (both cached); every other failure mode
    /// reports before any staging happens.
    fn open(
        store: &mut ChunkStore,
        cache: &mut ChunkCache,
        plan: &SplicePlan,
        offset: u64,
        patch: &[u8],
        logical_len: u64,
    ) -> Result<Self, ChunkError> {
        use std::collections::VecDeque;

        let mut segments = VecDeque::new();
        let mut tail_from = plan.entries.len();
        // Boundary spans (head, same-chunk tail) come from the one
        // verified cached read, so the boundary chunk is never re-read.
        // Queued in positional order after this block, together with the
        // zero gap and the patch.
        let mut head: Option<Bytes> = None;
        let mut same_tail: Option<Bytes> = None;
        if plan.boundary < plan.entries.len() {
            let entry = plan.entries[plan.boundary];
            let cached = read_chunk_cached(store, cache, entry.id)?;
            verify_entry(entry, &cached)?;
            let entry_end = plan.prefix_len + entry.len;
            let rel = offset
                .checked_sub(plan.prefix_len)
                .ok_or_else(|| ChunkError::Invalid {
                    detail: "splice offset precedes its retained prefix".to_owned(),
                })?;
            if plan.patch_end < entry_end {
                let tail_rel = splice_us(
                    plan.patch_end.saturating_sub(plan.prefix_len),
                    "splice tail",
                )?;
                same_tail = Some(cached.slice(tail_rel..));
            }
            // Past-the-end offsets stream the whole short final entry
            // (the append case); inside offsets stream the kept head
            // *before* the patch point — the overwritten span is skipped,
            // never streamed.
            head = Some(if rel >= entry.len {
                cached
            } else {
                cached.slice(..splice_us(rel, "splice head")?)
            });
            tail_from = plan.boundary + 1;
        }
        if let Some(head) = head.filter(|bytes| !bytes.is_empty()) {
            segments.push_back(SpliceSegment::Mem(head));
        }
        // Zero gap before the patch, present only when the patch starts
        // past the old end (a second gap after the patch is impossible:
        // the patch itself covers `[offset, patch_end)`).
        let zeros = offset.saturating_sub(logical_len);
        if zeros > 0 {
            segments.push_back(SpliceSegment::Zeros(zeros));
        }
        if !patch.is_empty() {
            segments.push_back(SpliceSegment::Mem(Bytes::copy_from_slice(patch)));
        }
        if let Some(tail) = same_tail.filter(|bytes| !bytes.is_empty()) {
            segments.push_back(SpliceSegment::Mem(tail));
        }
        // First tail entry intersecting `[patch_end, logical_len)`,
        // skipping the boundary entry (handled above: `tail_from`
        // already points past it, and the `len` guard keeps the scan
        // from overriding that). `starts` accumulates from zero over
        // every entry, so partial first takes compute exactly.
        let mut starts = 0u64;
        for (index, entry) in plan.entries.iter().enumerate() {
            let end = starts + entry.len;
            if index > plan.boundary
                && end > plan.patch_end
                && starts < logical_len
                && tail_from == plan.entries.len()
            {
                tail_from = index;
            }
            starts = end;
        }
        // When the tail runs on into pack entries, those start past the
        // patch: skip the overwritten head of the first one.
        let mut tail_skip = 0u64;
        if tail_from < plan.entries.len() {
            let mut start = 0u64;
            for entry in &plan.entries[..tail_from] {
                start += entry.len;
            }
            tail_skip = plan.patch_end.saturating_sub(start);
        }
        Ok(Self {
            segments,
            tail_from,
            tail_skip,
        })
    }

    /// Cuts one grid piece from the stream: resident segments first,
    /// then on-demand tail reads (direct, never cached). At most one
    /// piece plus one chunk stay resident.
    fn fill_piece(
        &mut self,
        store: &mut ChunkStore,
        plan: &SplicePlan,
        manifest: ManifestId,
        need: usize,
    ) -> Result<Vec<u8>, ChunkError> {
        let mut piece = Vec::with_capacity(need);
        let mut remaining = need;
        while remaining > 0 {
            match self.segments.pop_front() {
                None => {
                    if self.tail_from >= plan.entries.len() {
                        return Err(ChunkError::CorruptManifest {
                            id: manifest,
                            detail: "splice stream ran dry before the grid filled".to_owned(),
                        });
                    }
                    let entry = plan.entries[self.tail_from];
                    let bytes = store.read_chunk(entry.id)?;
                    verify_entry(entry, &bytes)?;
                    self.tail_from += 1;
                    let skip = splice_us(self.tail_skip.min(bytes.len() as u64), "splice skip")?;
                    self.tail_skip = 0;
                    if skip < bytes.len() {
                        self.segments
                            .push_front(SpliceSegment::Mem(bytes.slice(skip..)));
                    }
                }
                Some(SpliceSegment::Mem(bytes)) => {
                    let take = remaining.min(bytes.len());
                    piece.extend_from_slice(&bytes[..take]);
                    remaining -= take;
                    if take < bytes.len() {
                        self.segments
                            .push_front(SpliceSegment::Mem(bytes.slice(take..)));
                    }
                }
                Some(SpliceSegment::Zeros(count)) => {
                    let take = splice_us((remaining as u64).min(count), "splice zeros")?;
                    piece.resize(piece.len() + take, 0);
                    remaining -= take;
                    if (take as u64) < count {
                        self.segments
                            .push_front(SpliceSegment::Zeros(count - take as u64));
                    }
                }
            }
        }
        Ok(piece)
    }
}

/// One region-stream segment for [`splice_range_job`]: resident bytes or
/// a virtual zero run (gaps pad without materializing).
#[derive(Debug)]
enum SpliceSegment {
    /// Resident bytes with the unconsumed remainder (slicing shares).
    Mem(Bytes),
    /// `n` zero bytes, emitted without allocation.
    Zeros(u64),
}

/// Proves one chunked root's dependencies in its owner's lane (§30,
///
/// §31): the manifest is indexed and readable, every entry's chunk is
/// indexed, and the manifest total agrees with the root's claim. Index
/// membership suffices because lane recovery verified every indexed
/// record's bytes when it opened; serving reads re-verify anyway. A
/// lying root is corruption, never a miss.
fn check_root(
    store: &ChunkStore,
    manifest: ManifestId,
    logical_len: u64,
) -> Result<(), ChunkError> {
    if !store.has_manifest(&manifest) {
        return Err(ChunkError::MissingManifest { id: manifest });
    }
    let (loaded, _) = store.read_manifest(manifest)?;
    if loaded.total_len != logical_len {
        return Err(ChunkError::CorruptManifest {
            id: manifest,
            detail: format!(
                "root claims {logical_len} bytes but its manifest holds {}",
                loaded.total_len
            ),
        });
    }
    for entry in &loaded.entries {
        if !store.has_chunk(&entry.id) {
            return Err(ChunkError::MissingChunk { id: entry.id });
        }
    }
    Ok(())
}

/// One verified chunk through the cache.
fn read_chunk_cached(
    store: &mut ChunkStore,
    cache: &mut ChunkCache,
    id: ChunkId,
) -> Result<Bytes, ChunkError> {
    if let Some(hit) = cache.get(&id) {
        return Ok(hit);
    }
    let bytes = store.read_chunk(id)?;
    cache.insert(id, bytes.clone());
    Ok(bytes)
}

/// In-flight staging pins (§40): addresses an uploader intends to stage,
/// registered *before* the first append and removed after commit or abort.
/// Global to the engine (every lane's GC consults the same set); held
/// briefly and off any hot path — point ops never touch it.
///
/// The matching RAII guard is [`PinnedUpload`]: pinning before the first
/// append is what closes the stage-then-seal-then-commit race (an uploader
/// may stage into a pack that seals before its root commits; without pins
/// a concurrent mark could propose that pack while the commit is still in
/// flight).
#[derive(Debug, Default)]
pub struct StagingPins {
    inner: std::sync::Mutex<Pins>,
}

/// The pinned address sets.
#[derive(Debug, Default)]
pub struct Pins {
    /// Chunk addresses being staged.
    pub chunks: std::collections::HashSet<ChunkId>,
    /// Manifest addresses being staged.
    pub manifests: std::collections::HashSet<ManifestId>,
}

impl StagingPins {
    /// Pins one upload's chunk addresses for the staging window.
    pub fn pin_chunks(&self, chunks: &[(ChunkId, u64)]) {
        if let Ok(mut pins) = self.inner.lock() {
            pins.chunks.extend(chunks.iter().map(|(id, _)| *id));
        }
    }

    /// Pins one upload's manifest address for the staging window.
    pub fn pin_manifest(&self, manifest: ManifestId) {
        if let Ok(mut pins) = self.inner.lock() {
            pins.manifests.insert(manifest);
        }
    }

    /// Pins one upload's addresses for the staging window.
    pub fn pin(&self, chunks: &[(ChunkId, u64)], manifest: ManifestId) {
        self.pin_chunks(chunks);
        self.pin_manifest(manifest);
    }

    /// Releases one upload's addresses after commit or abort.
    pub fn unpin(&self, chunks: &[(ChunkId, u64)], manifest: ManifestId) {
        if let Ok(mut pins) = self.inner.lock() {
            for (id, _) in chunks {
                pins.chunks.remove(id);
            }
            pins.manifests.remove(&manifest);
        }
    }

    /// Snapshots both sets for one GC plan.
    pub fn snapshot(
        &self,
    ) -> (
        Arc<std::collections::HashSet<ChunkId>>,
        Arc<std::collections::HashSet<ManifestId>>,
    ) {
        match self.inner.lock() {
            Ok(pins) => (
                Arc::new(pins.chunks.clone()),
                Arc::new(pins.manifests.clone()),
            ),
            Err(_) => (
                Arc::new(std::collections::HashSet::default()),
                Arc::new(std::collections::HashSet::default()),
            ),
        }
    }
}

/// One upload's pinned staging window. Pre-hashes the value's chunk
/// addresses (authoritative hashing still happens lane-side on append;
/// this is pins only) and pins them *before* the staging job is
/// submitted; the manifest pins once proven. Dropping releases everything,
/// so every exit path — commit, error, timeout, abort — unpins
/// automatically. Carry it until the root commits (outbox entries and
/// coordinator pendings hold one for exactly that span).
#[derive(Debug)]
pub struct PinnedUpload {
    pins: Arc<StagingPins>,
    chunks: Vec<(ChunkId, u64)>,
    manifest: Option<ManifestId>,
}

impl PinnedUpload {
    /// Opens the window: pins every chunk address about to be staged.
    /// Pure CPU (hashing); the lane re-hashes authoritatively on append.
    ///
    /// # Errors
    ///
    /// Returns [`ChunkError`] when the value's piece count exceeds the
    /// address space (unreachable for memory-resident values, whose
    /// lengths always fit).
    pub fn begin(
        pins: &Arc<StagingPins>,
        domain: SecurityDomainId,
        value: &Bytes,
    ) -> Result<Self, ChunkError> {
        use kivi_chunk::Chunking;
        use kivi_codec::integrity::chunk_id;

        let chunking = Chunking::DEFAULT;
        let mut chunks = Vec::new();
        let mut at = 0usize;
        for len in chunking.split_lengths(value.len() as u64)? {
            let len = usize::try_from(len).map_err(|_| ChunkError::TooLarge {
                len,
                max: u64::MAX,
                context: "chunk piece",
            })?;
            let piece = &value[at..at + len];
            chunks.push((chunk_id(domain, piece), len as u64));
            at += len;
        }
        pins.pin_chunks(&chunks);
        Ok(Self {
            pins: Arc::clone(pins),
            chunks,
            manifest: None,
        })
    }

    /// Pins the proven manifest once staging returns it. Call before
    /// admission: from here until commit, GC sees the full address set.
    pub fn set_manifest(&mut self, manifest: ManifestId) {
        if self.manifest.is_none() {
            self.pins.pin_manifest(manifest);
            self.manifest = Some(manifest);
        }
    }

    /// Wraps an already-staged proof (lane splice path): pins every new
    /// address plus the manifest at once, before admission. The window
    /// between the lane's sync and this pin is safe by construction —
    /// staged bytes land in the GC-immune active tail, and the old root
    /// stays live until commit — but from here the guard owns the full
    /// address set exactly like the pre-pinned upload path.
    #[must_use]
    pub fn staged(pins: &Arc<StagingPins>, staged: &StagedValue) -> Self {
        pins.pin(&staged.chunks, staged.manifest);
        Self {
            pins: Arc::clone(pins),
            chunks: staged.chunks.clone(),
            manifest: Some(staged.manifest),
        }
    }
}

impl Drop for PinnedUpload {
    fn drop(&mut self) {
        self.pins
            .unpin(&self.chunks, self.manifest.unwrap_or(ManifestId::ZERO));
    }
}

/// Splits a legacy `Set` at the representation boundary: values at or
/// below `threshold` stay inline; larger ones come back for lane staging.
/// Pure and total over operations: every non-`Set` passes through
/// untouched, so call-site behavior for all other opcodes is unchanged.
/// Conditional stores split identically, preserving their condition and
/// expiry policy into a staged conditional chunked operation.
///
/// Transactional writes pass through untouched: large inline puts stay
/// inline (bounded by the transaction byte cap), and chunked references
/// are proven — never staged — at admission (`verify_txn_chunk_refs` on
/// the worker path, the async twin on the connection path, sidecar checks
/// on the consensus path), so a later Commit can never meet a durable
/// root with unavailable payload.
pub fn split_large_set(op: kivi_state::Operation, threshold: u64) -> LargeSetSplit {
    match op {
        kivi_state::Operation::Set { key, value } if value.len() as u64 > threshold => {
            LargeSetSplit::Stage { key, value }
        }
        kivi_state::Operation::SetConditional {
            key,
            value,
            condition,
            expiry,
        } if value.len() as u64 > threshold => LargeSetSplit::StageConditional {
            key,
            value,
            condition,
            expiry,
        },
        other => LargeSetSplit::Inline(other),
    }
}

/// The tablet-visible base a `SetRange` patches against: live inline
/// bytes (cloned at peek time), a chunked root's address, a counter
/// (which rejects at prepare), or nothing (absent or expired reads as
/// empty). Built from one synchronous store peek; the prepare-time type
/// gate re-checks, so a representation flip between peek and prepare
/// answers retryable instead of splicing wrong state.
#[derive(Debug, Clone)]
pub enum RangeBase {
    /// No live object: the patch applies over empty bytes.
    Absent,
    /// Live non-bytes value (counters and semantic objects): patches reject
    /// with `WrongType` at prepare.
    NonBytes,
    /// Live inline bytes to splice in memory.
    Inline(Bytes),
    /// Live chunked root: the lane restages instead of recording a splice.
    Chunked {
        /// Manifest addressing the base value.
        manifest: ManifestId,
        /// Total logical bytes the root claims.
        logical_len: u64,
    },
}

/// Outcome of [`plan_set_range`]: where a range patch goes next.
#[derive(Debug)]
pub enum SetRangePlan {
    /// Admit the splice as-is: the base is absent, expired, counter
    /// (prepare rejects), or inline with a small result. Small results
    /// WAL as [`SpliceBytes`](kivi_state::Mutation::SpliceBytes).
    Inline(kivi_state::Operation),
    /// The patched result crosses the representation boundary: store
    /// these full bytes through the ordinary large-`Set` staging path
    /// (which re-splits at the threshold itself).
    RestageSet {
        /// Target key.
        key: kivi_state::Key,
        /// Fully spliced bytes (zeros plus patch over the base).
        value: Bytes,
    },
    /// The base is chunked: restage through the lane's prefix-reuse
    /// splice, then admit `SetChunked` in place of the patch.
    Splice {
        /// Target key.
        key: kivi_state::Key,
        /// Manifest addressing the base value.
        manifest: ManifestId,
        /// Total logical bytes the root claims.
        logical_len: u64,
        /// Logical byte index the patch overwrites from.
        offset: u64,
        /// Bytes written starting at `offset`.
        patch: Bytes,
    },
    /// The patch is not admittable: overflowing arithmetic or a result
    /// past the legacy inline bound (rewrite through a streaming upload
    /// instead). Answered `InvalidRequest` with state untouched.
    Reject {
        /// Human-readable reason (logged and returned verbatim).
        detail: String,
    },
}

/// Plans a range patch against its peeked base: small inline results
/// WAL as splices, results crossing the threshold restage as full
/// values, chunked bases restage through the lane splice, and absurd
/// ranges reject before any I/O. Pure: no lane traffic, no tablet
/// mutation — the caller stages (blocking, async, or parked) and admits.
pub fn plan_set_range(
    key: kivi_state::Key,
    offset: u64,
    patch: Bytes,
    base: RangeBase,
    threshold: u64,
) -> SetRangePlan {
    let legacy_max = kivi_chunk::policy::MAX_LEGACY_VALUE_BYTES;
    let Some(patch_end) = offset.checked_add(patch.len() as u64) else {
        return SetRangePlan::Reject {
            detail: "set-range offset plus patch length overflows u64".to_owned(),
        };
    };
    match base {
        RangeBase::NonBytes => {
            SetRangePlan::Inline(kivi_state::Operation::SetRange { key, offset, patch })
        }
        RangeBase::Chunked {
            manifest,
            logical_len,
        } => SetRangePlan::Splice {
            key,
            manifest,
            logical_len,
            offset,
            patch,
        },
        RangeBase::Absent => {
            if patch_end > legacy_max {
                return SetRangePlan::Reject {
                    detail: format!(
                        "set-range result {patch_end} exceeds the {legacy_max}-byte range bound; rewrite through a streaming upload"
                    ),
                };
            }
            if patch_end > threshold {
                let Ok(offset_usize) = usize::try_from(offset) else {
                    return SetRangePlan::Reject {
                        detail: "set-range offset does not fit the address space".to_owned(),
                    };
                };
                let mut value = vec![0u8; offset_usize];
                value.extend_from_slice(&patch);
                SetRangePlan::RestageSet {
                    key,
                    value: Bytes::from(value),
                }
            } else {
                SetRangePlan::Inline(kivi_state::Operation::SetRange { key, offset, patch })
            }
        }
        RangeBase::Inline(bytes) => {
            let result = (bytes.len() as u64).max(patch_end);
            if result > legacy_max {
                return SetRangePlan::Reject {
                    detail: format!(
                        "set-range result {result} exceeds the {legacy_max}-byte range bound; rewrite through a streaming upload"
                    ),
                };
            }
            if result > threshold {
                // Materialized once, bounded by the legacy cap above, then
                // stored through the ordinary large-`Set` staging path.
                match kivi_state::splice_inline(&bytes, offset, &patch) {
                    Some(value) => SetRangePlan::RestageSet { key, value },
                    None => SetRangePlan::Reject {
                        detail: "set-range offset does not fit the address space".to_owned(),
                    },
                }
            } else {
                SetRangePlan::Inline(kivi_state::Operation::SetRange { key, offset, patch })
            }
        }
    }
}

/// Outcome of [`split_large_set`].
#[derive(Debug)]
pub enum LargeSetSplit {
    /// Proceed down the normal (inline) path unchanged.
    Inline(kivi_state::Operation),
    /// Stage `value` on the owner's chunk lane, then admit
    /// `SetChunked` in place of the original `Set`.
    Stage {
        /// Target key.
        key: kivi_state::Key,
        /// Full logical bytes to chunk.
        value: Bytes,
    },
    /// Stage `value` on the owner's chunk lane, then admit
    /// `SetConditionalChunked` preserving the condition and policy.
    StageConditional {
        /// Target key.
        key: kivi_state::Key,
        /// Full logical bytes to chunk.
        value: Bytes,
        /// Presence condition evaluated atomically at prepare.
        condition: kivi_state::SetCondition,
        /// Expiry policy resolved at prepare.
        expiry: kivi_state::ExpiryPolicy,
    },
}
