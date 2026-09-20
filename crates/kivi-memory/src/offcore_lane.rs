//! Per-worker offcore lane: asynchronous demotion/promotion I/O.
//!
//! One OS thread per worker owns an [`NvmeProvider`](kivi_memory::NvmeProvider)
//! over `demotion.dat` and serves bounded [`OffcoreJob`]s. Every other
//! context reaches it through ONE bounded [`async_channel`] queue, each
//! style awaiting its own way:
//!
//! ```text
//! net connection task (reactor)  →  send().await + recv().await
//!     (suspends only the task; the reactor thread never blocks)
//! embedded bridge turn           →  try_send + poll replies per turn
//! plain worker thread / tests    →  blocking send/recv
//! ```
//!
//! A full queue fast-fails [`MemoryError::Overloaded`] like every other
//! saturated Kivi queue. Jobs carry their own single-slot reply channels,
//! so completions never reorder and a dead waiter can never wedge the
//! lane.
//!
//! The lane owns the only append handle to `demotion.dat` (single writer;
//! torn tails truncate at open). Worker threads may hold read-only
//! handles to the same file for direct checkpoint/migration reads —
//! concurrent reads are safe, appends never race.
//!
//! Durable admission records live elsewhere (the durability lane's
//! `materializations.dat` + fabric journal); this lane holds
//! optimization copies only.

use std::path::Path;
use std::thread::{self, JoinHandle};

use crate::{MemoryError, NvmeOptions, NvmeProvider};
use async_channel::{Receiver, Sender, bounded};

/// Lane job queue depth. Demotions are rarer than point ops; 64 deep
/// absorbs pressure bursts while keeping worst-case queued bytes obvious.
pub const OFFCORE_JOB_DEPTH: usize = 64;

/// One completed demotion: the device record locator for the new
/// off-core residence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DemotedRecord {
    /// Which fabric object was demoted.
    pub object: u64,
    /// Logical version the bytes were staged at.
    pub version: u64,
    /// Device offset of the record.
    pub offset: u64,
    /// Stored payload length.
    pub len: u64,
    /// CRC32C of the payload.
    pub checksum: u32,
}

/// One completed promotion: verified payload bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PromotedBytes {
    /// Which fabric object was promoted.
    pub object: u64,
    /// Logical version the promotion was issued for.
    pub version: u64,
    /// Verified payload bytes.
    pub bytes: Vec<u8>,
}

/// Lane statistics for observability.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct OffcoreLaneStats {
    /// Completed demotions.
    pub demotions: u64,
    /// Completed promotions.
    pub promotions: u64,
    /// Failed operations (device errors, corrupt records).
    pub errors: u64,
    /// Jobs currently queued (observed at query time).
    pub queued: usize,
}

/// One lane job.
#[derive(Debug)]
enum OffcoreJob {
    /// Append one demotion record.
    Demote {
        /// Fabric object served.
        object: u64,
        /// Logical version staged at.
        version: u64,
        /// Payload bytes.
        bytes: Vec<u8>,
        /// Single-slot reply.
        reply: Sender<Result<DemotedRecord, MemoryError>>,
    },
    /// Read and verify one demotion record.
    Promote {
        /// Fabric object served.
        object: u64,
        /// Logical version issued for.
        version: u64,
        /// Device offset of the record.
        offset: u64,
        /// Expected payload length.
        len: u64,
        /// Single-slot reply.
        reply: Sender<Result<PromotedBytes, MemoryError>>,
    },
    /// Report statistics.
    Stats {
        /// Single-slot reply.
        reply: Sender<OffcoreLaneStats>,
    },
    /// Draining stop: the lane runs every job queued ahead (FIFO), then
    /// exits. No reply: the guard joins the thread instead.
    Shutdown,
}

/// Single-slot reply channel. The lane `try_send`s exactly once; waiters
/// choose their style: `await`, per-turn polling, or blocking.
#[derive(Debug)]
pub struct OffcoreReply<T> {
    rx: Receiver<T>,
}

impl<T> OffcoreReply<T> {
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
    pub async fn recv_async(&self) -> Result<T, async_channel::RecvError> {
        self.rx.recv().await
    }

    /// Blocking wait for plain worker threads and tests.
    ///
    /// # Errors
    ///
    /// Returns [`async_channel::RecvError`] when the lane is gone.
    pub fn recv_blocking(&self) -> Result<T, async_channel::RecvError> {
        self.rx.recv_blocking()
    }
}

/// Clonable handle to one worker's offcore lane. Cheap: one sender.
#[derive(Debug, Clone)]
pub struct OffcoreLaneHandle {
    tx: Sender<OffcoreJob>,
    worker: kivi_types::WorkerId,
}

impl OffcoreLaneHandle {
    /// Returns the worker this lane serves.
    #[must_use]
    pub const fn worker(&self) -> kivi_types::WorkerId {
        self.worker
    }

    /// Submits a demotion, returning the reply handle. Reactor-style wait.
    ///
    /// # Errors
    ///
    /// Returns [`MemoryError::Overloaded`] when the bounded lane queue is
    /// full or the lane is gone.
    pub async fn demote_async(
        &self,
        object: u64,
        version: u64,
        bytes: Vec<u8>,
    ) -> Result<DemotedRecord, MemoryError> {
        let reply = self.submit_demote(object, version, bytes)?;
        reply
            .recv_async()
            .await
            .map_err(|_| MemoryError::Overloaded {
                queue: "offcore lane",
            })?
    }

    /// Blocking twin for plain worker threads and tests.
    ///
    /// # Errors
    ///
    /// Returns [`MemoryError::Overloaded`] when the bounded lane queue is
    /// full or the lane is gone.
    pub fn demote_blocking(
        &self,
        object: u64,
        version: u64,
        bytes: Vec<u8>,
    ) -> Result<DemotedRecord, MemoryError> {
        let reply = self.submit_demote(object, version, bytes)?;
        reply.recv_blocking().map_err(|_| MemoryError::Overloaded {
            queue: "offcore lane",
        })?
    }

    /// Submits a demotion without waiting (bridge parking).
    ///
    /// # Errors
    ///
    /// Returns [`MemoryError::Overloaded`] when the bounded lane queue is
    /// full or the lane is gone.
    pub fn submit_demote(
        &self,
        object: u64,
        version: u64,
        bytes: Vec<u8>,
    ) -> Result<OffcoreReply<Result<DemotedRecord, MemoryError>>, MemoryError> {
        let (reply_tx, reply_rx) = bounded(1);
        self.tx
            .try_send(OffcoreJob::Demote {
                object,
                version,
                bytes,
                reply: reply_tx,
            })
            .map_err(|_| MemoryError::Overloaded {
                queue: "offcore lane",
            })?;
        Ok(OffcoreReply { rx: reply_rx })
    }

    /// Submits a promotion, returning the reply handle. Reactor-style wait.
    ///
    /// # Errors
    ///
    /// Returns [`MemoryError::Overloaded`] when the bounded lane queue is
    /// full or the lane is gone.
    pub async fn promote_async(
        &self,
        object: u64,
        version: u64,
        offset: u64,
        len: u64,
    ) -> Result<PromotedBytes, MemoryError> {
        let reply = self.submit_promote(object, version, offset, len)?;
        reply
            .recv_async()
            .await
            .map_err(|_| MemoryError::Overloaded {
                queue: "offcore lane",
            })?
    }

    /// Blocking twin for plain worker threads and tests.
    ///
    /// # Errors
    ///
    /// Returns [`MemoryError::Overloaded`] when the bounded lane queue is
    /// full or the lane is gone.
    pub fn promote_blocking(
        &self,
        object: u64,
        version: u64,
        offset: u64,
        len: u64,
    ) -> Result<PromotedBytes, MemoryError> {
        let reply = self.submit_promote(object, version, offset, len)?;
        reply.recv_blocking().map_err(|_| MemoryError::Overloaded {
            queue: "offcore lane",
        })?
    }

    /// Submits a promotion without waiting (bridge parking).
    ///
    /// # Errors
    ///
    /// Returns [`MemoryError::Overloaded`] when the bounded lane queue is
    /// full or the lane is gone.
    pub fn submit_promote(
        &self,
        object: u64,
        version: u64,
        offset: u64,
        len: u64,
    ) -> Result<OffcoreReply<Result<PromotedBytes, MemoryError>>, MemoryError> {
        let (reply_tx, reply_rx) = bounded(1);
        self.tx
            .try_send(OffcoreJob::Promote {
                object,
                version,
                offset,
                len,
                reply: reply_tx,
            })
            .map_err(|_| MemoryError::Overloaded {
                queue: "offcore lane",
            })?;
        Ok(OffcoreReply { rx: reply_rx })
    }

    /// Reads lane statistics (blocking; admin/observability only).
    #[must_use]
    pub fn stats_blocking(&self) -> OffcoreLaneStats {
        let (reply_tx, reply_rx) = bounded(1);
        if self
            .tx
            .try_send(OffcoreJob::Stats { reply: reply_tx })
            .is_err()
        {
            return OffcoreLaneStats::default();
        }
        reply_rx.recv_blocking().unwrap_or_default()
    }
}

/// Shutdown guard for one offcore lane thread.
#[derive(Debug)]
pub struct OffcoreLaneGuard {
    tx: Sender<OffcoreJob>,
    handle: Option<JoinHandle<()>>,
}

impl OffcoreLaneGuard {
    /// Stops the lane thread after draining submitted jobs: the stop is
    /// just another queued job, so everything submitted before it runs
    /// first. Joins even when the queue is full (a saturated lane still
    /// drains; only a dead lane skips the stop).
    pub fn shutdown(mut self) {
        // Blocking send: a saturated lane still drains to this stop; a
        // dead lane fails the send at once and the join below returns.
        let _ = self.tx.send_blocking(OffcoreJob::Shutdown);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

/// Spawns one offcore lane thread over `dir/demotion.dat`.
///
/// # Errors
///
/// Returns [`MemoryError::Io`] when the directory or data file cannot be
/// opened, and [`MemoryError::Unsupported`] on newer formats.
pub fn spawn_offcore_lane(
    worker: kivi_types::WorkerId,
    dir: &Path,
    options: NvmeOptions,
) -> Result<(OffcoreLaneHandle, OffcoreLaneGuard), MemoryError> {
    let (tx, rx) = bounded::<OffcoreJob>(OFFCORE_JOB_DEPTH);
    let provider = NvmeProvider::open(&dir.join("demotion"), options)?;
    let thread = thread::Builder::new()
        .name(format!("kivi-offcore-{}", worker.as_u64()))
        .spawn(move || serve(provider, &rx))
        .map_err(|error| MemoryError::Io {
            op: "spawn offcore lane",
            path: dir.to_owned(),
            message: error.to_string(),
        })?;
    Ok((
        OffcoreLaneHandle {
            tx: tx.clone(),
            worker,
        },
        OffcoreLaneGuard {
            tx,
            handle: Some(thread),
        },
    ))
}

/// Lane main loop: exactly one job at a time, replies never block (single
/// slot + `try_send`, disconnect-tolerant).
fn serve(mut provider: NvmeProvider, rx: &Receiver<OffcoreJob>) {
    let mut stats = OffcoreLaneStats::default();
    while let Ok(job) = rx.recv_blocking() {
        match job {
            OffcoreJob::Demote {
                object,
                version,
                bytes,
                reply,
            } => {
                // Relaxed barrier: demotion records are optimization
                // copies, never referenced durably (see
                // `NvmeProvider::append_relaxed`).
                let outcome = match provider.append_relaxed(&bytes) {
                    Ok((offset, len, checksum)) => {
                        stats.demotions += 1;
                        Ok(DemotedRecord {
                            object,
                            version,
                            offset,
                            len,
                            checksum,
                        })
                    }
                    Err(error) => {
                        stats.errors += 1;
                        Err(error)
                    }
                };
                let _ = reply.try_send(outcome);
            }
            OffcoreJob::Promote {
                object,
                version,
                offset,
                len,
                reply,
            } => {
                let outcome = match provider.read_at(offset, len) {
                    Ok(bytes) => {
                        stats.promotions += 1;
                        Ok(PromotedBytes {
                            object,
                            version,
                            bytes,
                        })
                    }
                    Err(error) => {
                        stats.errors += 1;
                        Err(error)
                    }
                };
                let _ = reply.try_send(outcome);
            }
            OffcoreJob::Stats { reply } => {
                stats.queued = rx.len();
                let _ = reply.try_send(stats);
            }
            OffcoreJob::Shutdown => {
                break;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn demote_promote_roundtrip_on_lane() {
        let dir = tempfile::tempdir().expect("tempdir");
        let worker = kivi_types::WorkerId::from_u64(0);
        let (lane, guard) =
            spawn_offcore_lane(worker, dir.path(), NvmeOptions::default()).expect("lane spawns");
        let record = lane
            .demote_blocking(7, 1, b"cold bytes".to_vec())
            .expect("demote");
        assert_eq!((record.object, record.version), (7, 1));
        let promoted = lane
            .promote_blocking(7, 1, record.offset, record.len)
            .expect("promote");
        assert_eq!(promoted.bytes, b"cold bytes");
        let stats = lane.stats_blocking();
        assert_eq!((stats.demotions, stats.promotions), (1, 1));
        guard.shutdown();
    }

    #[test]
    fn promote_missing_record_fails_closed() {
        let dir = tempfile::tempdir().expect("tempdir");
        let worker = kivi_types::WorkerId::from_u64(0);
        let (lane, guard) =
            spawn_offcore_lane(worker, dir.path(), NvmeOptions::default()).expect("lane spawns");
        let error = lane
            .promote_blocking(9, 1, 0, 10)
            .expect_err("missing record fails");
        assert!(matches!(error, MemoryError::CorruptRepresentation { .. }));
        guard.shutdown();
    }
}
