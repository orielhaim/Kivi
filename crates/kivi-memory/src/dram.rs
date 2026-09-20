//! Local DRAM materialization: arena-backed provider.
//!
//! The DRAM provider owns no bytes itself; it accounts the bytes held by
//! [`WorkerArenas`] against a capacity bound and reports pressure. Reads
//! resolve handles synchronously; writes allocate or update in the
//! behavior class the planner chose.

use bytes::Bytes;

use crate::arena::WorkerArenas;
use crate::error::MemoryError;
use crate::handle::{BehaviorClass, ObjectHandle};
use crate::provider::{ProviderCaps, ProviderId};

/// Accounting view over worker-local DRAM arenas.
#[derive(Debug)]
pub struct DramProvider {
    id: ProviderId,
    numa_node: Option<u32>,
    capacity_bytes: u64,
}

impl DramProvider {
    /// Creates a DRAM provider identity with a capacity bound.
    #[must_use]
    pub const fn new(id: ProviderId, numa_node: Option<u32>, capacity_bytes: u64) -> Self {
        Self {
            id,
            numa_node,
            capacity_bytes,
        }
    }

    /// Provider identifier.
    #[must_use]
    pub const fn id(&self) -> ProviderId {
        self.id
    }

    /// Current capabilities, calibrated with live arena pressure.
    #[must_use]
    pub fn caps(&self, arenas: &WorkerArenas) -> ProviderCaps {
        let mut caps = ProviderCaps::dram(self.numa_node, self.capacity_bytes);
        caps.free_bytes = self.capacity_bytes.saturating_sub(arenas.used_bytes());
        // Loaded latency grows with pressure: a full arena compacts and
        // relocates, which costs foreground-adjacent CPU. The penalty is
        // bounded by construction (pressure in [0, 1] yields at most
        // 60 ns), so the float-to-int conversion is exact in range.
        let pressure = arenas.pressure();
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let penalty = (pressure * pressure * 60.0) as u64;
        caps.read_latency_ns += penalty;
        caps.write_latency_ns += penalty;
        caps
    }

    /// Allocates a payload in the given behavior class.
    ///
    /// # Errors
    ///
    /// Propagates arena allocation failures.
    pub fn alloc(
        &self,
        arenas: &mut WorkerArenas,
        class: BehaviorClass,
        payload: Bytes,
    ) -> Result<ObjectHandle, MemoryError> {
        arenas.alloc(class, payload)
    }

    /// Resolves a handle synchronously.
    ///
    /// # Errors
    ///
    /// Returns [`MemoryError::StaleHandle`] for dead handles.
    pub fn get(
        &self,
        arenas: &WorkerArenas,
        class: BehaviorClass,
        handle: ObjectHandle,
    ) -> Result<Bytes, MemoryError> {
        arenas.get(class, handle)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pressure_inflates_calibrated_latency() {
        let provider = DramProvider::new(ProviderId(0), Some(0), 1_000);
        let mut arenas = WorkerArenas::new(1_000);
        let idle = provider.caps(&arenas).read_latency_ns;
        arenas
            .alloc(BehaviorClass::HotMutable, Bytes::from(vec![0u8; 900]))
            .expect("alloc");
        let loaded = provider.caps(&arenas).read_latency_ns;
        assert!(loaded > idle);
    }
}
