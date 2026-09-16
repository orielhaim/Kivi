//! Checkpoint capture views: plain-data snapshots of one tablet at a cut.
//!
//! The engine builds these from live tablets (committed state only —
//! never speculative or pending batches) and hands them to the background
//! builder. Everything here is owned plain data with no locks, no threads,
//! and no engine types: the checkpoint crate never depends on the engine.

use kivi_state::{Key, StoredObject, TxnIntent};
use kivi_types::{CommitPosition, NamespaceId, TabletEpoch, TabletId, WriteGuardGeneration};

use crate::dedup::SessionCheckpoint;
use crate::layout::BandDirty;

/// One tablet captured at a durable cut: everything the builder needs,
/// nothing it must borrow.
#[derive(Debug, Clone)]
pub struct TabletSnapshot {
    /// Namespace served (band mapping input).
    pub namespace: NamespaceId,
    /// Tablet captured.
    pub tablet: TabletId,
    /// Tablet epoch at capture.
    pub epoch: TabletEpoch,
    /// Write-guard generation at capture.
    pub guard: WriteGuardGeneration,
    /// Checkpoint cut: last applied commit position (durable truth).
    pub cut: CommitPosition,
    /// All committed objects (unsorted; the builder sorts into bands).
    pub objects: Vec<(Key, StoredObject)>,
    /// All session dedup states (floors plus retained outcomes).
    pub sessions: Vec<SessionCheckpoint>,
    /// All prepared transaction intents (never discarded on restart).
    pub intents: Vec<TxnIntent>,
    /// Dirty bands taken at capture (cleared on the live tablet).
    pub dirty: BandDirty,
}

impl TabletSnapshot {
    /// Counts committed objects.
    #[must_use]
    pub fn object_count(&self) -> usize {
        self.objects.len()
    }

    /// Counts retained dedup outcomes across sessions.
    #[must_use]
    pub fn outcome_count(&self) -> usize {
        self.sessions
            .iter()
            .map(|session| session.outcomes.len())
            .sum()
    }
}
