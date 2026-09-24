//! Bounded background execution for redundancy work.
//!
//! [`RedundancyLane`] is mechanism only: it schedules generic closures, so
//! it serves the embedded [`crate::RedundancyFabric`], the cluster
//! [`crate::DistributedFabric`], and any future caller without knowing any
//! of them. Callers run CPU coding inside jobs (never tablet workers) with
//! network and fsync inline, then await the [`LaneJoin`] for gates or drop
//! it for best-effort work.
//!
//! Bounds are hard: queue depth, bytes inflight, and worker count are all
//! capped ([`LaneConfig`]); overflow fails fast with
//! [`crate::RedundancyError::Overloaded`], never sheds silently. Priority
//! orders service ([`LanePriority`]); the generation-fence hook skips stale
//! work before it runs; the foreground-pressure flag parks background work
//! while hot reads are active. The pattern mirrors `kivi-consensus`
//! `SidecarStore`: `async_channel::bounded` jobs drained by std worker
//! threads, `futures::channel::oneshot` replies.

use std::collections::{BTreeMap, VecDeque};
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicU64, Ordering},
};
use std::thread::JoinHandle;

use crate::RedundancyError;

/// Job priority: higher ranks run first (ties stay FIFO).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum LanePriority {
    /// Fills spare capacity; parked under foreground pressure.
    Background = 0,
    /// Ordinary bounded work.
    Normal = 1,
    /// Runs even under foreground pressure.
    Foreground = 2,
}

/// Lane bounds: queue depth, bytes inflight, worker count.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LaneConfig {
    /// Maximum queued (not yet running) jobs.
    pub depth: usize,
    /// Maximum bytes held by queued + running jobs.
    pub max_bytes_inflight: u64,
    /// Worker threads draining the queue.
    pub concurrency: usize,
}

impl LaneConfig {
    /// Conservative defaults: 64 queued jobs, 64 MiB inflight, 4 workers.
    #[must_use]
    pub const fn conservative() -> Self {
        Self {
            depth: 64,
            max_bytes_inflight: 64 * 1024 * 1024,
            concurrency: 4,
        }
    }

    /// Validates nonzero bounds.
    ///
    /// # Errors
    ///
    /// Returns [`RedundancyError::InvalidParams`] on zero bounds.
    pub const fn validate(self) -> Result<(), RedundancyError> {
        if self.depth == 0 || self.max_bytes_inflight == 0 || self.concurrency == 0 {
            return Err(RedundancyError::InvalidParams {
                detail: String::new(),
            });
        }
        Ok(())
    }
}

impl Default for LaneConfig {
    fn default() -> Self {
        Self::conservative()
    }
}

/// Value a lane job produces (mechanism-level shapes only).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LaneValue {
    /// Opaque bytes (read images, encoded layouts).
    Bytes(Vec<u8>),
    /// A count (rebuilt fragments, swept orphans).
    Number(u64),
    /// No value (barrier, best-effort retire).
    Empty,
}

/// Generation-fence hook: returns `false` when the job went stale before
/// running (stale generations skip, count, and never execute).
pub type FenceHook = Arc<dyn Fn() -> bool + Send + Sync>;

/// One queued job: declared cost, priority, fence, and the work itself.
struct Job {
    /// Static name (operators, diagnostics).
    name: &'static str,
    /// Declared bytes (admission + inflight accounting).
    bytes: u64,
    /// Service rank.
    priority: LanePriority,
    /// Staleness gate checked immediately before running.
    fence: Option<FenceHook>,
    /// The work (CPU coding, network, fsync inline).
    run: Box<dyn FnOnce() -> Result<LaneValue, RedundancyError> + Send>,
    /// Reply rendezvous.
    reply: futures::channel::oneshot::Sender<Result<LaneValue, RedundancyError>>,
}

impl std::fmt::Debug for Job {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Job")
            .field("name", &self.name)
            .field("bytes", &self.bytes)
            .field("priority", &self.priority)
            .finish_non_exhaustive()
    }
}

/// Shared scheduler state behind one lock (never held across job runs).
struct Shared {
    /// Queued jobs by priority rank (higher first, FIFO within a rank).
    queues: BTreeMap<u8, VecDeque<Job>>,
    /// Bytes held by queued + running jobs.
    inflight: u64,
}

/// Live lane counters (lock-free operator reads).
#[derive(Debug, Default)]
struct Counters {
    /// Jobs accepted.
    submitted: AtomicU64,
    /// Jobs run to completion (any outcome).
    completed: AtomicU64,
    /// Jobs skipped by a false fence hook.
    skipped_stale: AtomicU64,
    /// Submits refused on full queues or oversize declarations.
    rejected_overload: AtomicU64,
}

/// Point-in-time lane counters for operators.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct LaneStats {
    /// Jobs accepted.
    pub submitted: u64,
    /// Jobs run to completion.
    pub completed: u64,
    /// Jobs skipped stale.
    pub skipped_stale: u64,
    /// Submits refused overloaded.
    pub rejected_overload: u64,
}

/// The join handle for one submitted job: await it for gates, drop it for
/// best-effort work (the job still runs; only the reply is shed).
pub struct LaneJoin {
    /// Job name (diagnostics).
    name: &'static str,
    /// Reply rendezvous.
    reply: futures::channel::oneshot::Receiver<Result<LaneValue, RedundancyError>>,
}

impl std::fmt::Debug for LaneJoin {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LaneJoin")
            .field("name", &self.name)
            .finish_non_exhaustive()
    }
}

impl LaneJoin {
    /// Waits for the outcome, blocking the caller.
    ///
    /// # Errors
    ///
    /// Returns [`RedundancyError::Overloaded`] when the lane dropped the
    /// reply (shutdown); otherwise returns the job's own outcome (stale
    /// skips report here too).
    pub fn recv_blocking(self) -> Result<LaneValue, RedundancyError> {
        futures::executor::block_on(self.reply).map_err(|_| RedundancyError::Overloaded {
            detail: format!("lane dropped {}", self.name),
        })?
    }

    /// Waits for the outcome asynchronously.
    ///
    /// # Errors
    ///
    /// Returns [`RedundancyError::Overloaded`] when the lane dropped the
    /// reply; otherwise returns the job's own outcome.
    pub async fn recv_async(self) -> Result<LaneValue, RedundancyError> {
        self.reply.await.map_err(|_| RedundancyError::Overloaded {
            detail: format!("lane dropped {}", self.name),
        })?
    }

    /// Polls for the outcome without blocking (`None` = still queued or
    /// running).
    pub fn try_recv(&mut self) -> Option<Result<LaneValue, RedundancyError>> {
        self.reply.try_recv().ok()?
    }
}

/// Bounded executor for redundancy closures.
///
/// CPU coding runs here (never tablet workers); network and fsync run inline
/// in jobs; tablet workers only await joins for gates or drop them for
/// best-effort work.
pub struct RedundancyLane {
    /// Job intake (`None` after shutdown starts draining).
    sender: Option<async_channel::Sender<Job>>,
    /// Worker threads.
    handles: Vec<JoinHandle<()>>,
    /// Foreground-pressure flag (parks non-foreground jobs while set).
    pressure: Arc<AtomicBool>,
    /// Live counters.
    counters: Arc<Counters>,
    /// Bounds (admission checks).
    config: LaneConfig,
}

impl std::fmt::Debug for RedundancyLane {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RedundancyLane")
            .field("config", &self.config)
            .field("pressure", &self.pressure.load(Ordering::Relaxed))
            .field("stats", &self.stats())
            .finish_non_exhaustive()
    }
}

impl RedundancyLane {
    /// Opens the lane: spawns `config.concurrency` std worker threads over
    /// a bounded intake queue.
    ///
    /// # Errors
    ///
    /// Returns [`RedundancyError::InvalidParams`] on zero config bounds, or
    /// [`RedundancyError::Io`] when a worker thread cannot spawn.
    pub fn open(config: LaneConfig) -> Result<Self, RedundancyError> {
        config.validate()?;
        let (sender, receiver) = async_channel::bounded::<Job>(config.depth);
        let pressure = Arc::new(AtomicBool::new(false));
        let counters = Arc::new(Counters::default());
        let shared = Arc::new(std::sync::Mutex::new(Shared {
            queues: BTreeMap::new(),
            inflight: 0,
        }));
        let mut handles = Vec::with_capacity(config.concurrency);
        for index in 0..config.concurrency {
            let worker_receiver = receiver.clone();
            let worker_shared = Arc::clone(&shared);
            let worker_pressure = Arc::clone(&pressure);
            let worker_counters = Arc::clone(&counters);
            let max_bytes = config.max_bytes_inflight;
            let handle = std::thread::Builder::new()
                .name(format!("kivi-lane-{index}"))
                .spawn(move || {
                    serve(
                        &worker_receiver,
                        &worker_shared,
                        &worker_pressure,
                        &worker_counters,
                        max_bytes,
                    );
                })
                .map_err(|error| RedundancyError::Io {
                    detail: format!("lane thread spawn: {error}"),
                })?;
            handles.push(handle);
        }
        Ok(Self {
            sender: Some(sender),
            handles,
            pressure,
            counters,
            config,
        })
    }

    /// Submits one closure for bounded background execution.
    ///
    /// Admission is fallible on purpose: full queues and oversize
    /// declarations fail fast with [`RedundancyError::Overloaded`] (and
    /// count it) instead of shedding silently.
    ///
    /// # Errors
    ///
    /// Returns [`RedundancyError::Overloaded`] when `bytes` exceeds the
    /// inflight budget or the queue is full/shut down.
    pub fn submit(
        &self,
        name: &'static str,
        bytes: u64,
        priority: LanePriority,
        fence: Option<FenceHook>,
        run: impl FnOnce() -> Result<LaneValue, RedundancyError> + Send + 'static,
    ) -> Result<LaneJoin, RedundancyError> {
        let Some(sender) = self.sender.as_ref() else {
            return Err(RedundancyError::Overloaded {
                detail: "lane is shut down".to_owned(),
            });
        };
        if bytes > self.config.max_bytes_inflight {
            self.counters
                .rejected_overload
                .fetch_add(1, Ordering::Relaxed);
            return Err(RedundancyError::Overloaded {
                detail: format!("lane job {name} declares {bytes} bytes"),
            });
        }
        let (tx, rx) = futures::channel::oneshot::channel();
        let job = Job {
            name,
            bytes,
            priority,
            fence,
            run: Box::new(run),
            reply: tx,
        };
        if sender.try_send(job).is_err() {
            self.counters
                .rejected_overload
                .fetch_add(1, Ordering::Relaxed);
            return Err(RedundancyError::Overloaded {
                detail: format!("lane queue full for {name}"),
            });
        }
        self.counters.submitted.fetch_add(1, Ordering::Relaxed);
        Ok(LaneJoin { name, reply: rx })
    }

    /// Sets foreground pressure: while set, only
    /// [`LanePriority::Foreground`] jobs run (hot reads keep the lane).
    pub fn set_foreground_pressure(&self, pressure: bool) {
        self.pressure.store(pressure, Ordering::Relaxed);
    }

    /// Returns the foreground-pressure flag.
    #[must_use]
    pub fn foreground_pressure(&self) -> bool {
        self.pressure.load(Ordering::Relaxed)
    }

    /// Snapshots lane counters for operators.
    #[must_use]
    pub fn stats(&self) -> LaneStats {
        LaneStats {
            submitted: self.counters.submitted.load(Ordering::Relaxed),
            completed: self.counters.completed.load(Ordering::Relaxed),
            skipped_stale: self.counters.skipped_stale.load(Ordering::Relaxed),
            rejected_overload: self.counters.rejected_overload.load(Ordering::Relaxed),
        }
    }
}

impl Drop for RedundancyLane {
    fn drop(&mut self) {
        // Close intake so workers drain and exit, then join them.
        self.sender.take();
        while let Some(handle) = self.handles.pop() {
            let _ = handle.join();
        }
    }
}

/// Worker loop: stage arrivals into the shared priority queue, claim the
/// best runnable job, run it outside the lock, release its budget.
fn serve(
    receiver: &async_channel::Receiver<Job>,
    shared: &std::sync::Mutex<Shared>,
    pressure: &AtomicBool,
    counters: &Counters,
    max_bytes: u64,
) {
    loop {
        // Stage every ready arrival first (priority order is decided over
        // the full staged set, not arrival order).
        while let Ok(job) = receiver.try_recv() {
            push_job(shared, job);
        }
        let claimed = claim_job(shared, pressure.load(Ordering::Relaxed), max_bytes);
        if let Some(job) = claimed {
            run_job(job, shared, counters);
            continue;
        }
        if queue_empty(shared) {
            // Nothing staged: block for the next arrival (shutdown closes
            // the channel and ends the loop here).
            match receiver.recv_blocking() {
                Ok(job) => push_job(shared, job),
                Err(_) => break,
            }
        } else {
            // Jobs are staged but parked (budget or pressure): brief sleep,
            // then recheck — completions release budget and pressure clears
            // externally, so every park ends without new arrivals.
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
    }
}

/// Stages one arrival into its priority queue (lock poison sheds the job:
/// the submitter's join reports overload instead of hanging).
fn push_job(shared: &std::sync::Mutex<Shared>, job: Job) {
    if let Ok(mut guard) = shared.lock() {
        guard
            .queues
            .entry(job.priority as u8)
            .or_default()
            .push_back(job);
    }
}

/// Whether any job is staged.
fn queue_empty(shared: &std::sync::Mutex<Shared>) -> bool {
    shared
        .lock()
        .map_or(true, |guard| guard.queues.values().all(VecDeque::is_empty))
}

/// Claims the best runnable job: highest rank first (foreground only under
/// pressure), first fitting declaration within a rank, budget-gated.
fn claim_job(shared: &std::sync::Mutex<Shared>, pressure: bool, max_bytes: u64) -> Option<Job> {
    let mut guard = shared.lock().ok()?;
    // Ranks are `u8` discriminants; iterate high to low.
    let mut ranks: Vec<u8> = guard.queues.keys().copied().collect();
    ranks.sort_unstable_by(|a, b| b.cmp(a));
    for rank in ranks {
        if pressure && rank != LanePriority::Foreground as u8 {
            continue;
        }
        let fits = guard.queues.get(&rank).map(|queue| {
            queue.iter().position(|job| {
                guard.inflight.saturating_add(job.bytes) <= max_bytes || guard.inflight == 0
            })
        });
        if let Some(Some(position)) = fits
            && let Some(queue) = guard.queues.get_mut(&rank)
            && let Some(job) = queue.remove(position)
        {
            guard.inflight = guard.inflight.saturating_add(job.bytes);
            return Some(job);
        }
    }
    None
}

/// Runs one claimed job: fence first (stale skips count, never execute),
/// then the closure, then budget release and reply.
fn run_job(job: Job, shared: &std::sync::Mutex<Shared>, counters: &Counters) {
    let bytes = job.bytes;
    if let Some(fence) = &job.fence
        && !fence()
    {
        counters.skipped_stale.fetch_add(1, Ordering::Relaxed);
        release(shared, bytes);
        let _ = job.reply.send(Err(RedundancyError::Overloaded {
            detail: format!("lane skipped stale {}", job.name),
        }));
        return;
    }
    let Job { run, reply, .. } = job;
    let outcome = run();
    counters.completed.fetch_add(1, Ordering::Relaxed);
    release(shared, bytes);
    let _ = reply.send(outcome);
}

/// Releases one job's budget bytes.
fn release(shared: &std::sync::Mutex<Shared>, bytes: u64) {
    if let Ok(mut guard) = shared.lock() {
        guard.inflight = guard.inflight.saturating_sub(bytes);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex as StdMutex;

    #[test]
    fn bounds_reject_overflow() {
        let lane = RedundancyLane::open(LaneConfig {
            depth: 2,
            max_bytes_inflight: 1024,
            concurrency: 1,
        })
        .expect("opens");
        // Oversize declarations fail fast.
        assert!(
            lane.submit("huge", 2048, LanePriority::Normal, None, || Ok(
                LaneValue::Empty
            ))
            .is_err()
        );
        // Block the single worker, fill the queue, overflow fails. The
        // blocker signals start so the fill is deterministic (no sleep
        // racing the worker pickup).
        let gate = Arc::new(std::sync::Barrier::new(2));
        let gate_hold = Arc::clone(&gate);
        let started = Arc::new(AtomicBool::new(false));
        let started_hold = Arc::clone(&started);
        let _blocker = lane
            .submit("blocker", 1, LanePriority::Normal, None, move || {
                started_hold.store(true, Ordering::Relaxed);
                gate_hold.wait();
                Ok(LaneValue::Empty)
            })
            .expect("blocker queued");
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while !started.load(Ordering::Relaxed) {
            assert!(
                std::time::Instant::now() <= deadline,
                "lane worker never started"
            );
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        let _first = lane
            .submit("first", 1, LanePriority::Normal, None, || {
                Ok(LaneValue::Empty)
            })
            .expect("queued");
        let _second = lane
            .submit("second", 1, LanePriority::Normal, None, || {
                Ok(LaneValue::Empty)
            })
            .expect("queued");
        let error = lane
            .submit("overflow", 1, LanePriority::Normal, None, || {
                Ok(LaneValue::Empty)
            })
            .expect_err("queue full");
        assert!(matches!(error, RedundancyError::Overloaded { .. }));
        gate.wait();
        assert_eq!(lane.stats().rejected_overload, 2);
    }

    #[test]
    fn priority_orders_under_pressure() {
        let lane = RedundancyLane::open(LaneConfig {
            depth: 16,
            max_bytes_inflight: 1024,
            concurrency: 1,
        })
        .expect("opens");
        let order = Arc::new(StdMutex::new(Vec::new()));
        lane.set_foreground_pressure(true);
        let mut joins = Vec::new();
        for name in ["bg-one", "bg-two"] {
            let log = Arc::clone(&order);
            joins.push(
                lane.submit(name, 1, LanePriority::Background, None, move || {
                    log.lock().expect("unpoisoned").push(name);
                    Ok(LaneValue::Empty)
                })
                .expect("queued"),
            );
        }
        let log = Arc::clone(&order);
        let foreground = lane
            .submit("fg", 1, LanePriority::Foreground, None, move || {
                log.lock().expect("unpoisoned").push("fg");
                Ok(LaneValue::Number(7))
            })
            .expect("queued");
        // Foreground runs despite pressure; background stays parked.
        assert_eq!(
            foreground.recv_blocking().expect("foreground runs"),
            LaneValue::Number(7)
        );
        for join in &mut joins {
            assert!(join.try_recv().is_none(), "background stays parked");
        }
        lane.set_foreground_pressure(false);
        for join in joins {
            join.recv_blocking().expect("background runs");
        }
        assert_eq!(
            order.lock().expect("unpoisoned").as_slice(),
            &["fg", "bg-one", "bg-two"]
        );
    }

    #[test]
    fn cancellation_skips_before_running() {
        let lane = RedundancyLane::open(LaneConfig::conservative()).expect("opens");
        let ran = Arc::new(AtomicBool::new(false));
        let ran_hold = Arc::clone(&ran);
        let stale: FenceHook = Arc::new(|| false);
        let join = lane
            .submit("stale", 1, LanePriority::Normal, Some(stale), move || {
                ran_hold.store(true, Ordering::Relaxed);
                Ok(LaneValue::Empty)
            })
            .expect("queued");
        let outcome = join.recv_blocking().expect_err("stale skips");
        assert!(matches!(outcome, RedundancyError::Overloaded { .. }));
        assert!(!ran.load(Ordering::Relaxed), "stale work never executes");
        assert_eq!(lane.stats().skipped_stale, 1);
        // A live fence runs normally.
        let live: FenceHook = Arc::new(|| true);
        let join = lane
            .submit("live", 1, LanePriority::Normal, Some(live), || {
                Ok(LaneValue::Bytes(vec![1, 2, 3]))
            })
            .expect("queued");
        assert_eq!(
            join.recv_blocking().expect("runs"),
            LaneValue::Bytes(vec![1, 2, 3])
        );
    }

    #[test]
    fn concurrent_engineers_share_one_lane() {
        let lane = Arc::new(RedundancyLane::open(LaneConfig::conservative()).expect("opens"));
        let mut handles = Vec::new();
        for engineer in 0..8u64 {
            let lane_hold = Arc::clone(&lane);
            handles.push(std::thread::spawn(move || {
                let join = lane_hold
                    .submit("shared", 1, LanePriority::Normal, None, move || {
                        Ok(LaneValue::Number(engineer))
                    })
                    .expect("queued");
                join.recv_blocking().expect("runs")
            }));
        }
        let mut values: Vec<u64> = handles
            .into_iter()
            .map(|handle| {
                let outcome = handle.join().expect("engineer joins");
                assert!(matches!(outcome, LaneValue::Number(_)));
                let LaneValue::Number(value) = outcome else {
                    return u64::MAX;
                };
                value
            })
            .collect();
        values.sort_unstable();
        assert_eq!(values, (0..8u64).collect::<Vec<_>>());
        assert_eq!(lane.stats().submitted, 8);
        assert_eq!(lane.stats().completed, 8);
    }
}
