//! Deterministic simulated providers.
//!
//! Placement and transition state machines must not require real hardware
//! to test. [`SimProvider`] models capacity, latency, failure, and
//! pressure deterministically from an explicit seed and scripted
//! [`SimFault`]s: same seed plus same script replays the same latencies,
//! admissions, corruptions, and outages. Production providers implement
//! the same record semantics; only timing and faults differ.

use std::collections::HashMap;

use crate::error::MemoryError;
use crate::provider::{ProviderCaps, ProviderId, ProviderKind};

/// Scripted fault for deterministic failure testing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SimFault {
    /// Fail the next `count` operations with a device error.
    FailNext {
        /// How many operations fail.
        count: u64,
    },
    /// Corrupt the next `count` reads (checksum mismatch on serve).
    CorruptNext {
        /// How many reads corrupt.
        count: u64,
    },
    /// Take the provider offline until `ticks` pass (virtual time).
    OfflineUntil {
        /// Virtual tick when service resumes.
        ticks: u64,
    },
    /// Clamp free bytes to `bytes` (pressure injection).
    Pressure {
        /// Free bytes reported while active.
        bytes: u64,
    },
}

/// Deterministic provider: capacity-bounded byte store with scripted
/// latency and faults.
#[derive(Debug)]
pub struct SimProvider {
    id: ProviderId,
    kind: ProviderKind,
    capacity_bytes: u64,
    used_bytes: u64,
    base_read_ns: u64,
    base_write_ns: u64,
    durable: bool,
    store: HashMap<u64, Vec<u8>>,
    next_offset: u64,
    faults: Vec<SimFault>,
    fail_remaining: u64,
    corrupt_remaining: u64,
    offline_until: u64,
    now_ticks: u64,
    pressure_bytes: Option<u64>,
    rng_state: u64,
    ops: u64,
    errors: u64,
}

impl SimProvider {
    /// Creates a simulated provider.
    #[must_use]
    pub fn new(
        id: ProviderId,
        kind: ProviderKind,
        capacity_bytes: u64,
        base_read_ns: u64,
        base_write_ns: u64,
        durable: bool,
        seed: u64,
    ) -> Self {
        Self {
            id,
            kind,
            capacity_bytes,
            used_bytes: 0,
            base_read_ns,
            base_write_ns,
            durable,
            store: HashMap::new(),
            next_offset: 0,
            faults: Vec::new(),
            fail_remaining: 0,
            corrupt_remaining: 0,
            offline_until: 0,
            now_ticks: 0,
            pressure_bytes: None,
            rng_state: if seed == 0 { 1 } else { seed },
            ops: 0,
            errors: 0,
        }
    }

    /// DRAM-like simulated provider.
    #[must_use]
    pub fn dram(id: ProviderId, capacity_bytes: u64, seed: u64) -> Self {
        Self::new(id, ProviderKind::DRAM, capacity_bytes, 90, 90, false, seed)
    }

    /// `NVMe`-like simulated provider.
    #[must_use]
    pub fn nvme(id: ProviderId, capacity_bytes: u64, seed: u64) -> Self {
        Self::new(
            id,
            ProviderKind::NVME,
            capacity_bytes,
            25_000,
            35_000,
            true,
            seed,
        )
    }

    /// Provider identifier.
    #[must_use]
    pub const fn id(&self) -> ProviderId {
        self.id
    }

    /// Advances virtual time, expiring time-bound faults.
    pub const fn advance_to(&mut self, ticks: u64) {
        if ticks > self.now_ticks {
            self.now_ticks = ticks;
        }
    }

    /// Scripts a fault.
    pub fn inject(&mut self, fault: SimFault) {
        match fault {
            SimFault::FailNext { count } => {
                self.fail_remaining = self.fail_remaining.saturating_add(count);
            }
            SimFault::CorruptNext { count } => {
                self.corrupt_remaining = self.corrupt_remaining.saturating_add(count);
            }
            SimFault::OfflineUntil { ticks } => {
                self.offline_until = self.offline_until.max(ticks);
            }
            SimFault::Pressure { bytes } => {
                self.pressure_bytes = Some(bytes);
                self.faults.push(fault);
            }
        }
    }

    /// Clears pressure injection.
    pub fn clear_pressure(&mut self) {
        self.pressure_bytes = None;
        self.faults
            .retain(|fault| !matches!(fault, SimFault::Pressure { .. }));
    }

    fn deterministic_jitter(&mut self) -> u64 {
        self.rng_state ^= self.rng_state >> 12;
        self.rng_state ^= self.rng_state << 25;
        self.rng_state ^= self.rng_state >> 27;
        self.rng_state.wrapping_mul(0x2545_F491_4F6C_DD1D) % 100
    }

    fn check_available(&mut self) -> Result<(), MemoryError> {
        if self.now_ticks < self.offline_until {
            self.errors += 1;
            return Err(MemoryError::ProviderFailed {
                provider: "sim",
                object: u64::MAX,
                detail: "simulated offline".to_owned(),
            });
        }
        if self.fail_remaining > 0 {
            self.fail_remaining -= 1;
            self.errors += 1;
            return Err(MemoryError::ProviderFailed {
                provider: "sim",
                object: u64::MAX,
                detail: "scripted failure".to_owned(),
            });
        }
        Ok(())
    }

    /// Writes bytes, returning `(offset, latency_ns)`.
    ///
    /// # Errors
    ///
    /// Returns [`MemoryError::ProviderFailed`] on scripted faults and
    /// [`MemoryError::Overloaded`] when capacity is exhausted.
    pub fn write(&mut self, bytes: &[u8]) -> Result<(u64, u64), MemoryError> {
        self.check_available()?;
        if self.used_bytes.saturating_add(bytes.len() as u64) > self.capacity_bytes {
            return Err(MemoryError::Overloaded {
                queue: "sim capacity",
            });
        }
        let jitter = self.deterministic_jitter();
        self.ops += 1;
        let offset = self.next_offset;
        self.next_offset += 1;
        self.used_bytes += bytes.len() as u64;
        self.store.insert(offset, bytes.to_vec());
        Ok((offset, self.base_write_ns + jitter))
    }

    /// Reads bytes by offset.
    ///
    /// # Errors
    ///
    /// Returns [`MemoryError::ProviderFailed`] on scripted faults and
    /// [`MemoryError::CorruptRepresentation`] on scripted corruption or
    /// missing records.
    pub fn read(&mut self, offset: u64) -> Result<(Vec<u8>, u64), MemoryError> {
        self.check_available()?;
        let jitter = self.deterministic_jitter();
        self.ops += 1;
        let bytes = self
            .store
            .get(&offset)
            .cloned()
            .ok_or(MemoryError::CorruptRepresentation {
                object: u64::MAX,
                detail: "sim record missing".to_owned(),
            })?;
        if self.corrupt_remaining > 0 {
            self.corrupt_remaining -= 1;
            self.errors += 1;
            return Err(MemoryError::CorruptRepresentation {
                object: u64::MAX,
                detail: "scripted corruption".to_owned(),
            });
        }
        Ok((bytes, self.base_read_ns + jitter))
    }

    /// Current capabilities with pressure applied.
    #[must_use]
    pub fn caps(&self) -> ProviderCaps {
        let free = self
            .pressure_bytes
            .unwrap_or_else(|| self.capacity_bytes.saturating_sub(self.used_bytes));
        ProviderCaps {
            kind: self.kind,
            locality: crate::provider::LocalityCaps {
                numa_node: None,
                remote_penalty_ns: 0,
                node_shared: true,
            },
            durable: self.durable,
            shared_read: true,
            mutable_in_place: !self.durable,
            direct_io: false,
            fdp: false,
            read_latency_ns: self.base_read_ns,
            write_latency_ns: self.base_write_ns,
            read_bandwidth_bps: 3_000_000_000,
            write_bandwidth_bps: 2_000_000_000,
            cost_per_byte: if self.durable { 3 } else { 100 },
            decompress_ns_per_byte: 0,
            capacity_bytes: self.capacity_bytes,
            free_bytes: free.min(self.capacity_bytes),
            healthy: self.now_ticks >= self.offline_until && self.fail_remaining == 0,
        }
    }

    /// Operation count.
    #[must_use]
    pub const fn ops(&self) -> u64 {
        self.ops
    }

    /// Error count.
    #[must_use]
    pub const fn errors(&self) -> u64 {
        self.errors
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scripted_failures_are_deterministic() {
        let mut first = SimProvider::nvme(ProviderId(0), 1 << 20, 99);
        let mut second = SimProvider::nvme(ProviderId(0), 1 << 20, 99);
        first.inject(SimFault::FailNext { count: 1 });
        second.inject(SimFault::FailNext { count: 1 });
        assert!(first.write(b"x").is_err());
        assert!(second.write(b"x").is_err());
        let (_, first_latency) = first.write(b"y").expect("recovers");
        let (_, second_latency) = second.write(b"y").expect("recovers");
        assert_eq!(first_latency, second_latency);
    }

    #[test]
    fn offline_windows_reject_and_recover() {
        let mut provider = SimProvider::dram(ProviderId(0), 1 << 20, 7);
        provider.inject(SimFault::OfflineUntil { ticks: 100 });
        assert!(provider.write(b"x").is_err());
        provider.advance_to(100);
        assert!(provider.write(b"x").is_ok());
    }
}
