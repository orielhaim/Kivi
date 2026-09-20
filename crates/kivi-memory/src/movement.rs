//! Bounded movement and reclamation.
//!
//! Background compression, compaction, promotion, and demotion share one
//! [`MovementScheduler`]: a bounded queue of [`MoveRequest`]s drained
//! under a per-tick [`MovementBudget`] (bytes, operations, CPU, concurrent
//! off-core ops). Foreground tail latency always wins: an exhausted
//! budget defers work to the next tick instead of borrowing from serving.
//! Cancellation is explicit and safe: dropping a queued move changes no
//! state because publication always follows verification.

use std::collections::{HashMap, VecDeque};

use crate::error::MemoryError;
use crate::transition::TransitionTarget;

/// How much background work one tick may perform.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MovementBudget {
    /// Maximum payload bytes moved per tick.
    pub max_bytes_per_tick: u64,
    /// Maximum moves completed per tick.
    pub max_ops_per_tick: usize,
    /// Maximum background CPU nanoseconds per tick.
    pub max_cpu_ns_per_tick: u64,
    /// Maximum concurrent off-core (`NVMe`) operations.
    pub max_concurrent_offcore: usize,
    /// Bytes moved so far this tick.
    pub used_bytes: u64,
    /// Operations completed so far this tick.
    pub used_ops: usize,
    /// CPU consumed so far this tick.
    pub used_cpu_ns: u64,
}

impl MovementBudget {
    /// Creates a fresh budget with the given per-tick limits.
    #[must_use]
    pub const fn new(
        max_bytes_per_tick: u64,
        max_ops_per_tick: usize,
        max_cpu_ns_per_tick: u64,
        max_concurrent_offcore: usize,
    ) -> Self {
        Self {
            max_bytes_per_tick,
            max_ops_per_tick,
            max_cpu_ns_per_tick,
            max_concurrent_offcore,
            used_bytes: 0,
            used_ops: 0,
            used_cpu_ns: 0,
        }
    }

    /// Conservative production default: 4 MiB, 32 ops, 2 ms CPU, 8
    /// concurrent off-core ops per tick.
    #[must_use]
    pub const fn production() -> Self {
        Self::new(4 << 20, 32, 2_000_000, 8)
    }

    /// Whether a move of `bytes` costing `cpu_ns` fits the remaining
    /// budget.
    #[must_use]
    pub const fn fits(&self, bytes: u64, cpu_ns: u64) -> bool {
        self.used_bytes.saturating_add(bytes) <= self.max_bytes_per_tick
            && self.used_ops.saturating_add(1) <= self.max_ops_per_tick
            && self.used_cpu_ns.saturating_add(cpu_ns) <= self.max_cpu_ns_per_tick
    }

    /// Charges a completed move. Callers check [`fits`](Self::fits)
    /// first; overcharges saturate instead of wrapping.
    pub const fn charge(&mut self, bytes: u64, cpu_ns: u64) {
        self.used_bytes = self.used_bytes.saturating_add(bytes);
        self.used_ops = self.used_ops.saturating_add(1);
        self.used_cpu_ns = self.used_cpu_ns.saturating_add(cpu_ns);
    }

    /// Resets per-tick accounting for the next tick.
    pub const fn reset(&mut self) {
        self.used_bytes = 0;
        self.used_ops = 0;
        self.used_cpu_ns = 0;
    }
}

/// One queued background move.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MoveRequest {
    /// Which object moves.
    pub object: u64,
    /// Where it is heading.
    pub target: TransitionTarget,
    /// Payload bytes (for budget accounting).
    pub bytes: u64,
    /// Estimated CPU cost in nanoseconds.
    pub cpu_ns: u64,
    /// Tenant tag for per-tenant fairness (opaque).
    pub tenant: u64,
}

/// Bounded queue of background moves with per-tenant fairness.
#[derive(Debug)]
pub struct BoundedMoveQueue {
    capacity: usize,
    queue: VecDeque<MoveRequest>,
    tenant_bytes: HashMap<u64, u64>,
    max_bytes_per_tenant: u64,
}

impl BoundedMoveQueue {
    /// Creates a queue with operation and per-tenant byte bounds.
    #[must_use]
    pub fn new(capacity: usize, max_bytes_per_tenant: u64) -> Self {
        Self {
            capacity,
            queue: VecDeque::new(),
            tenant_bytes: HashMap::new(),
            max_bytes_per_tenant,
        }
    }

    /// Enqueues a move, or rejects with `Overloaded` when full or when
    /// the tenant's queued bytes exceed its share.
    ///
    /// # Errors
    ///
    /// Returns [`MemoryError::Overloaded`] when the queue or the
    /// tenant's byte share is exhausted.
    pub fn push(&mut self, request: MoveRequest) -> Result<(), MemoryError> {
        if self.queue.len() >= self.capacity {
            return Err(MemoryError::Overloaded {
                queue: "move queue",
            });
        }
        let tenant_bytes = self.tenant_bytes.get(&request.tenant).copied().unwrap_or(0);
        if tenant_bytes.saturating_add(request.bytes) > self.max_bytes_per_tenant {
            return Err(MemoryError::Overloaded {
                queue: "tenant move share",
            });
        }
        self.tenant_bytes
            .insert(request.tenant, tenant_bytes + request.bytes);
        self.queue.push_back(request);
        Ok(())
    }

    /// Pops the next move, releasing its tenant accounting.
    pub fn pop(&mut self) -> Option<MoveRequest> {
        let request = self.queue.pop_front()?;
        if let Some(bytes) = self.tenant_bytes.get_mut(&request.tenant) {
            *bytes = bytes.saturating_sub(request.bytes);
        }
        Some(request)
    }

    /// Cancels all queued moves for an object. Returns the count.
    pub fn cancel_for_object(&mut self, object: u64) -> usize {
        let before = self.queue.len();
        let mut kept = VecDeque::with_capacity(before);
        for request in self.queue.drain(..) {
            if request.object == object {
                if let Some(bytes) = self.tenant_bytes.get_mut(&request.tenant) {
                    *bytes = bytes.saturating_sub(request.bytes);
                }
            } else {
                kept.push_back(request);
            }
        }
        self.queue = kept;
        before - self.queue.len()
    }

    /// Queued move count.
    #[must_use]
    pub fn len(&self) -> usize {
        self.queue.len()
    }

    /// Whether the queue is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }
}

/// Drains a [`BoundedMoveQueue`] under a [`MovementBudget`], one tick at
/// a time. Pure scheduling: the caller executes the returned moves and
/// charges the budget.
#[derive(Debug)]
pub struct MovementScheduler {
    budget: MovementBudget,
}

impl MovementScheduler {
    /// Creates a scheduler with the given per-tick budget.
    #[must_use]
    pub const fn new(budget: MovementBudget) -> Self {
        Self { budget }
    }

    /// Current budget (for observability).
    #[must_use]
    pub const fn budget(&self) -> &MovementBudget {
        &self.budget
    }

    /// Resets the tick budget.
    pub const fn reset_tick(&mut self) {
        self.budget.reset();
    }

    /// Takes the next executable move: the oldest queued move fitting the
    /// remaining budget with off-core concurrency available.
    /// Non-fitting head moves do not block fitting moves behind them
    /// (bounded bypass keeps one huge object from wedging the queue).
    ///
    /// # Panics
    ///
    /// Cannot panic: `position` comes from searching the same queue it
    /// indexes, with no mutation in between.
    pub fn next_move(
        &mut self,
        queue: &mut BoundedMoveQueue,
        in_flight_offcore: usize,
    ) -> Option<MoveRequest> {
        let position = queue.queue.iter().position(|request| {
            let offcore_ok = !matches!(request.target, TransitionTarget::Nvme)
                || in_flight_offcore < self.budget.max_concurrent_offcore;
            offcore_ok && self.budget.fits(request.bytes, request.cpu_ns)
        })?;
        let request = queue.queue.remove(position).expect("position valid");
        if let Some(bytes) = queue.tenant_bytes.get_mut(&request.tenant) {
            *bytes = bytes.saturating_sub(request.bytes);
        }
        self.budget.charge(request.bytes, request.cpu_ns);
        Some(request)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn budget_bounds_background_work() {
        let mut scheduler = MovementScheduler::new(MovementBudget::new(100, 1, 1_000, 8));
        let mut queue = BoundedMoveQueue::new(16, 1 << 20);
        queue
            .push(MoveRequest {
                object: 1,
                target: TransitionTarget::Compressed,
                bytes: 60,
                cpu_ns: 100,
                tenant: 0,
            })
            .expect("push");
        queue
            .push(MoveRequest {
                object: 2,
                target: TransitionTarget::Compressed,
                bytes: 60,
                cpu_ns: 100,
                tenant: 0,
            })
            .expect("push");
        assert!(scheduler.next_move(&mut queue, 0).is_some());
        // Second move exceeds the byte budget this tick.
        assert!(scheduler.next_move(&mut queue, 0).is_none());
        scheduler.reset_tick();
        assert!(scheduler.next_move(&mut queue, 0).is_some());
    }

    #[test]
    fn cancellation_is_safe_and_targeted() {
        let mut queue = BoundedMoveQueue::new(16, 1 << 20);
        queue
            .push(MoveRequest {
                object: 1,
                target: TransitionTarget::Nvme,
                bytes: 10,
                cpu_ns: 10,
                tenant: 0,
            })
            .expect("push");
        queue
            .push(MoveRequest {
                object: 2,
                target: TransitionTarget::Nvme,
                bytes: 10,
                cpu_ns: 10,
                tenant: 0,
            })
            .expect("push");
        assert_eq!(queue.cancel_for_object(1), 1);
        assert_eq!(queue.len(), 1);
    }
}
