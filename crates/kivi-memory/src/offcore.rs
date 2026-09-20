//! Explicit asynchronous off-core operations.
//!
//! `NVMe` demotion and promotion are never hidden behind blocking page
//! faults. They are explicit [`OffcoreOp`]s enqueued on a bounded
//! [`OffcoreQueue`], executed by the Compio backend ([`crate::nvme`]) or
//! the deterministic simulator ([`crate::sim`]), and completed back to
//! the fabric. Per-object ordering follows `CacheLib` Navy's spooling rule:
//! at most one outstanding op per object; later ops for the same object
//! wait in a per-object spool instead of racing it.

/// Identifier for one off-core operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct OffcoreOpId(pub u64);

/// What an off-core operation does.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OffcoreKind {
    /// Writes a representation to `NVMe` (demotion).
    Demote,
    /// Reads a representation back from `NVMe` (promotion).
    Promote,
    /// Deletes an `NVMe` record after its representation retired.
    Delete,
}

/// Lifecycle of one off-core operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OffcoreState {
    /// Queued, not yet submitted to the device.
    Queued,
    /// Submitted to the device, awaiting completion.
    Submitted,
    /// Completed successfully.
    Completed,
    /// Failed; the fabric falls back to another representation.
    Failed,
    /// Cancelled before submission (mutation, migration, or load shed).
    Cancelled,
}

/// One explicit `NVMe` operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OffcoreOp {
    /// Operation identifier.
    pub id: OffcoreOpId,
    /// Which object it serves.
    pub object: u64,
    /// Logical version the op was issued for (fences mutation races).
    pub version: u64,
    /// What it does.
    pub kind: OffcoreKind,
    /// Payload bytes for demotion; empty for promotion/deletion.
    pub bytes: Vec<u8>,
    /// Current lifecycle state.
    pub state: OffcoreState,
}

impl OffcoreOp {
    /// Creates a queued operation.
    #[must_use]
    pub const fn new(id: OffcoreOpId, object: u64, version: u64, kind: OffcoreKind) -> Self {
        Self {
            id,
            object,
            version,
            kind,
            bytes: Vec::new(),
            state: OffcoreState::Queued,
        }
    }
}

/// Completion reported by the device backend.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OffcoreCompletion {
    /// Which operation completed.
    pub id: OffcoreOpId,
    /// Which object it served.
    pub object: u64,
    /// Logical version the op was issued for (fences mutation races:
    /// a completion for a stale version is discarded).
    pub version: u64,
    /// Bytes read back (promotion) or empty (demote/delete).
    pub bytes: Vec<u8>,
    /// Device offset of a demoted record (for the new residence).
    pub offset: Option<u64>,
    /// Stored length of a demoted record.
    pub len: Option<u64>,
    /// Content checksum of a demoted record.
    pub checksum: Option<u32>,
    /// Whether the operation succeeded.
    pub ok: bool,
    /// Failure detail when `!ok`.
    pub detail: String,
}

/// Bounded queue of off-core operations with per-object spooling.
///
/// Capacity bounds memory and device pressure (Navy admission insight:
/// `MaxConcurrentInserts`/`MaxParcelMemory` reject instead of queueing
/// without bound). At most one op per object is submitted at a time; a
/// second op for the same object spools until the first completes, which
/// keeps demote/promote/delete for one object totally ordered without a
/// global lock.
#[derive(Debug)]
pub struct OffcoreQueue {
    capacity: usize,
    max_bytes: u64,
    queued_bytes: u64,
    queued: Vec<OffcoreOp>,
    submitted: Vec<OffcoreOp>,
    next_id: u64,
}

impl OffcoreQueue {
    /// Creates a queue with operation and byte bounds.
    #[must_use]
    pub const fn new(capacity: usize, max_bytes: u64) -> Self {
        Self {
            capacity,
            max_bytes,
            queued_bytes: 0,
            queued: Vec::new(),
            submitted: Vec::new(),
            next_id: 1,
        }
    }

    /// Enqueues a demotion. Succeeds even when another op for the same
    /// object is outstanding (spooled); fails with `Overloaded` only when
    /// the queue or byte bound is exhausted.
    ///
    /// # Errors
    ///
    /// Returns [`crate::error::MemoryError::Overloaded`] when bounds are
    /// exhausted.
    pub fn enqueue_demote(
        &mut self,
        object: u64,
        version: u64,
        bytes: Vec<u8>,
    ) -> Result<OffcoreOpId, crate::error::MemoryError> {
        self.enqueue(object, version, OffcoreKind::Demote, bytes)
    }

    /// Enqueues a promotion.
    ///
    /// # Errors
    ///
    /// Returns [`crate::error::MemoryError::Overloaded`] when bounds are
    /// exhausted.
    pub fn enqueue_promote(
        &mut self,
        object: u64,
        version: u64,
    ) -> Result<OffcoreOpId, crate::error::MemoryError> {
        self.enqueue(object, version, OffcoreKind::Promote, Vec::new())
    }

    /// Enqueues a deletion.
    ///
    /// # Errors
    ///
    /// Returns [`crate::error::MemoryError::Overloaded`] when bounds are
    /// exhausted.
    pub fn enqueue_delete(
        &mut self,
        object: u64,
        version: u64,
    ) -> Result<OffcoreOpId, crate::error::MemoryError> {
        self.enqueue(object, version, OffcoreKind::Delete, Vec::new())
    }

    fn enqueue(
        &mut self,
        object: u64,
        version: u64,
        kind: OffcoreKind,
        bytes: Vec<u8>,
    ) -> Result<OffcoreOpId, crate::error::MemoryError> {
        if self.queued.len() >= self.capacity {
            return Err(crate::error::MemoryError::Overloaded {
                queue: "offcore queue",
            });
        }
        let bytes_len = bytes.len() as u64;
        if self.queued_bytes.saturating_add(bytes_len) > self.max_bytes {
            return Err(crate::error::MemoryError::Overloaded {
                queue: "offcore bytes",
            });
        }
        let id = OffcoreOpId(self.next_id);
        self.next_id += 1;
        self.queued_bytes += bytes_len;
        let mut op = OffcoreOp::new(id, object, version, kind);
        op.bytes = bytes;
        self.queued.push(op);
        Ok(id)
    }

    /// Takes the next submittable op: the oldest queued op whose object
    /// has nothing submitted. Returns `None` when empty or all head
    /// objects are busy (their turns come when submissions complete).
    pub fn take_submittable(&mut self) -> Option<OffcoreOp> {
        let position = self
            .queued
            .iter()
            .position(|op| !self.submitted.iter().any(|busy| busy.object == op.object))?;
        let mut op = self.queued.remove(position);
        self.queued_bytes = self.queued_bytes.saturating_sub(op.bytes.len() as u64);
        op.state = OffcoreState::Submitted;
        self.submitted.push(op.clone());
        Some(op)
    }

    /// Applies a completion, removing the op from the submitted set.
    /// Returns `true` when the op was known.
    pub fn complete(&mut self, id: OffcoreOpId) -> bool {
        if let Some(position) = self.submitted.iter().position(|op| op.id == id) {
            self.submitted.remove(position);
            return true;
        }
        false
    }

    /// Cancels all queued (never submitted) ops for an object, e.g. after
    /// mutation or tablet migration. Returns the cancelled count.
    pub fn cancel_for_object(&mut self, object: u64) -> usize {
        let before = self.queued.len();
        let mut kept = Vec::with_capacity(before);
        for op in self.queued.drain(..) {
            if op.object == object {
                self.queued_bytes = self.queued_bytes.saturating_sub(op.bytes.len() as u64);
            } else {
                kept.push(op);
            }
        }
        self.queued = kept;
        before - self.queued.len()
    }

    /// Number of queued (unsubmitted) operations.
    #[must_use]
    pub const fn queued_len(&self) -> usize {
        self.queued.len()
    }

    /// Number of submitted (in-flight) operations.
    #[must_use]
    pub const fn submitted_len(&self) -> usize {
        self.submitted.len()
    }

    /// Queued payload bytes.
    #[must_use]
    pub const fn queued_bytes(&self) -> u64 {
        self.queued_bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn per_object_ordering_spools_second_op() {
        let mut queue = OffcoreQueue::new(16, 1 << 20);
        queue.enqueue_demote(1, 1, vec![1, 2, 3]).expect("enqueue");
        queue.enqueue_promote(1, 1).expect("spooled");
        queue.enqueue_demote(2, 1, vec![9]).expect("enqueue");
        let first = queue.take_submittable().expect("first");
        assert_eq!(first.object, 1);
        // Object 1 is busy; object 2 goes next despite queue order.
        let second = queue.take_submittable().expect("second");
        assert_eq!(second.object, 2);
        // Nothing submittable left until object 1 completes.
        assert!(queue.take_submittable().is_none());
        queue.complete(first.id);
        let third = queue.take_submittable().expect("third");
        assert_eq!(third.object, 1);
    }

    #[test]
    fn bounds_reject_instead_of_growing() {
        let mut queue = OffcoreQueue::new(1, 8);
        queue.enqueue_demote(1, 1, vec![0; 8]).expect("fits");
        assert!(queue.enqueue_demote(2, 1, vec![0]).is_err());
    }

    #[test]
    fn cancel_drops_only_the_target_object() {
        let mut queue = OffcoreQueue::new(16, 1 << 20);
        queue.enqueue_demote(1, 1, vec![1]).expect("enqueue");
        queue.enqueue_demote(2, 1, vec![2]).expect("enqueue");
        assert_eq!(queue.cancel_for_object(1), 1);
        assert_eq!(queue.queued_len(), 1);
    }
}
