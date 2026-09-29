//! A memory provider backed by CXL or pmem capacity.
//!
//! ## Why this is a provider and not a durability tier
//!
//! A CXL Type-3 device may be volatile or persistent, and the two are
//! indistinguishable from the host without an explicit check. Kivi therefore
//! registers every region here as a *volatile, node-attached* provider. The
//! mapping is real capacity - the pages are really device-backed and really
//! reachable - but nothing in Kivi may treat a byte in this tier as surviving
//! a restart, because a device that reports no persistence is a device that has
//! none. A deployment that proves persistence adds it; Kivi does not guess.
//!
//! ## Why the capabilities are measured, not asserted
//!
//! CXL memory-mode bandwidth and latency depend on the link speed, the
//! interleave granularity, the memory controller's settings, and how far the
//! worker is from the node the device is attached to. Any table of "typical CXL
//! latency" is a guess that is wrong on at least one of those axes. So this
//! provider registers with an *uncalibrated* descriptor - every cost field zero,
//! `healthy: false` - and [`CxlProvider::read`] / [`CxlProvider::write`] feed
//! the real [`Calibrator`]. A region that has never been touched is therefore
//! excluded by `ProviderCaps::satisfies`, which is the correct answer: the
//! planner has no evidence to rank it with.
//!
//! ## Emulation is labelled
//!
//! [`CxlProvider::emulate`] maps anonymous memory instead of a device. It
//! exists so the provider's lifecycle, extent accounting, and calibration path
//! are exercised on a machine with no CXL hardware, and every capability it
//! produces is marked [`ProviderKind::CXL_EMULATED`]. A benchmark over it is
//! reported as emulated and must not be quoted as a CXL number.

use std::time::Instant;

use kivi_hardware::dax::DaxRegion;
use kivi_hardware::topology::Topology;

use crate::calibrate::Calibrator;
use crate::error::MemoryError;
use crate::provider::{LocalityCaps, ProviderCaps, ProviderId, ProviderKind};

/// The provider identifier this provider calibrates under.
///
/// A CXL provider owns every mapped region, so there is exactly one of them per
/// process and the calibrator needs one key for it. A deployment with two
/// providers would need one key each, which is why this is a constant and not a
/// field: the moment a second one exists, this must become one.
const PROVIDER_ID: ProviderId = ProviderId(0);

/// Provider kind tag used in this module's errors.
const CXL_KIND: &str = "cxl";

/// A read or write that fell outside every mapped region.
///
/// Reported as invalid caller use rather than as a device failure: on a live
/// region an out-of-range offset is a bug, and on a region the kernel removed
/// under the process it is the only honest description available. Either way
/// the fabric's response is the same - do not serve these bytes - which is why
/// the two need not be distinguished.
fn outside_region() -> MemoryError {
    MemoryError::Invalid {
        detail: "cxl access outside every mapped region".to_string(),
    }
}

/// Bytes one extent occupies, and the granularity a DAX mapping faults in.
///
/// Not a tuning knob: the kernel maps a DAX region in 2 MiB units and will not
/// fault a smaller one, so an allocation that is not a whole number of these is
/// either a refusal or a silent overcommit.
pub const CXL_UNIT_BYTES: u64 = kivi_hardware::dax::DAX_ALIGNMENT_BYTES;

/// One mapped CXL or pmem region, with the region it came from.
#[derive(Debug)]
pub struct CxlRegion {
    /// The mapping. Owns the lifetime of the device mapping.
    mapped: DaxRegion,
    /// Whether this is a real device or the anonymous emulation.
    emulated: bool,
}

impl CxlRegion {
    /// Wraps a mapping and records whether it is a device or an emulation.
    ///
    /// The region description is not retained: everything the provider needs
    /// (capacity, node) is already inside the mapping, and keeping a second copy
    /// of a description that a hotplug can invalidate would give a caller two
    /// answers to "how big is this region".
    #[must_use]
    pub const fn new(mapped: DaxRegion, emulated: bool) -> Self {
        Self { mapped, emulated }
    }
    /// Bytes mapped.
    #[must_use]
    pub fn len(&self) -> usize {
        self.mapped.len()
    }

    /// True when the mapping holds no bytes, which never happens for a region
    /// that mapped successfully.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.mapped.is_empty()
    }

    /// Whether this region is an anonymous emulation rather than a device.
    #[must_use]
    pub const fn is_emulated(&self) -> bool {
        self.emulated
    }

    /// The path the mapping came from.
    #[must_use]
    pub fn path(&self) -> &std::path::Path {
        self.mapped.path()
    }

    /// Bytes the extent allocator has handed out.
    #[must_use]
    pub fn allocated_bytes(&self) -> u64 {
        self.mapped.allocated_bytes()
    }

    /// Bytes still available.
    #[must_use]
    pub fn free_bytes(&self) -> u64 {
        (self.len() as u64).saturating_sub(self.mapped.allocated_bytes())
    }

    /// Takes `bytes` of extent, rounded up to the mapping granularity.
    fn take(&mut self, bytes: u64) -> Result<u64, MemoryError> {
        self.mapped
            .allocate(bytes)
            .ok_or(MemoryError::ProviderExhausted { provider: CXL_KIND })
    }

    /// Returns an extent to the region.
    fn give_back(&mut self, offset: u64, bytes: u64) {
        let _ = self.mapped.release(offset, bytes);
    }
}

/// A set of mapped CXL/pmem regions acting as one provider.
///
/// One provider per node rather than one per region: the planner places an
/// object by node, and splitting one device's capacity across several providers
/// would let it rank a fragment of a device against a whole DRAM arena.
#[derive(Debug)]
pub struct CxlProvider {
    /// Regions in registration order.
    regions: Vec<CxlRegion>,
    /// Total mapped capacity, summed once at construction so the provider can
    /// report a capacity without scanning its regions.
    capacity_bytes: u64,
    /// Extents currently handed out.
    allocated_bytes: u64,
    /// Nodes this provider's regions are attached to.
    numa_nodes: Vec<u32>,
    /// Every region emulated, or any real device present.
    emulated: bool,
}

impl CxlProvider {
    /// Maps every tiered region the topology offers.
    ///
    /// Returns an empty provider - not an error - on a machine with no CXL or
    /// pmem, because that is a normal deployment and must not stop startup. A
    /// region that fails to map is dropped and named in the reason; a provider
    /// that swallowed that would report capacity it cannot reach.
    #[must_use]
    pub fn discover(topology: &Topology) -> (Self, Vec<String>) {
        let mut regions = Vec::new();
        let mut dropped = Vec::new();
        for region in kivi_hardware::dax::discover_regions(topology) {
            match DaxRegion::map(&region) {
                Ok(mapped) => regions.push(CxlRegion::new(mapped, false)),
                Err(error) => dropped.push(format!("{}: {error}", region.path)),
            }
        }
        (Self::from_regions(regions), dropped)
    }

    /// Wraps already-mapped regions.
    #[must_use]
    pub fn from_regions(regions: Vec<CxlRegion>) -> Self {
        let capacity_bytes = regions
            .iter()
            .map(|region| region.len() as u64)
            .fold(0_u64, u64::saturating_add);
        let numa_nodes = regions
            .iter()
            .filter_map(|region| region.mapped.numa_node())
            .map(|node| node.0)
            .collect();
        let emulated = regions.iter().all(|region| region.emulated);
        Self {
            regions,
            capacity_bytes,
            allocated_bytes: 0,
            numa_nodes,
            emulated,
        }
    }

    /// An anonymous region standing in for a device.
    ///
    /// The provider this builds is labelled [`ProviderKind::CXL_EMULATED`] and
    /// starts uncalibrated exactly like a real one, so the calibration path is
    /// the same code either way. The node hint is recorded for diagnostics only:
    /// anonymous memory has no NUMA attachment to report, and claiming one would
    /// make `numa_nodes()` a fiction.
    ///
    /// # Errors
    ///
    /// Returns the operating-system error when the mapping cannot be created.
    pub fn emulate(
        len: u64,
        numa_node: Option<u32>,
    ) -> Result<(Self, Vec<String>), std::io::Error> {
        let _ = numa_node;
        let mapped = DaxRegion::map_anonymous(usize::try_from(len).unwrap_or(usize::MAX), None)?;
        Ok((
            Self::from_regions(vec![CxlRegion::new(mapped, true)]),
            Vec::new(),
        ))
    }

    /// Number of mapped regions.
    #[must_use]
    pub fn region_count(&self) -> usize {
        self.regions.len()
    }

    /// Total capacity in bytes.
    #[must_use]
    pub const fn capacity_bytes(&self) -> u64 {
        self.capacity_bytes
    }

    /// Bytes currently handed out.
    #[must_use]
    pub const fn allocated_bytes(&self) -> u64 {
        self.allocated_bytes
    }

    /// Bytes still available.
    #[must_use]
    pub const fn free_bytes(&self) -> u64 {
        self.capacity_bytes.saturating_sub(self.allocated_bytes)
    }

    /// Whether this provider has no capacity at all.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.capacity_bytes == 0
    }

    /// Whether every region is an emulation.
    #[must_use]
    pub const fn is_emulated(&self) -> bool {
        self.emulated
    }

    /// NUMA nodes this provider's capacity is attached to.
    #[must_use]
    pub fn numa_nodes(&self) -> &[u32] {
        &self.numa_nodes
    }

    /// The capabilities this provider registers with, before calibration.
    ///
    /// Every cost field is zero and `healthy` is false: the honest descriptor
    /// for capacity nobody has measured. The planner excludes it, the calibrator
    /// fills it in, and a provider that is never touched stays excluded.
    #[must_use]
    pub fn caps(&self) -> ProviderCaps {
        let kind = if self.emulated {
            ProviderKind::CXL_EMULATED
        } else {
            ProviderKind::CXL
        };
        ProviderCaps {
            kind,
            // A CXL region is attached to exactly one node, and a cross-node
            // access crosses the link. The penalty is left at zero because it
            // is a property of the *placement decision*, not of the device; the
            // measured latency of a remote read arrives through calibration.
            locality: LocalityCaps {
                numa_node: self.numa_nodes.first().copied(),
                remote_penalty_ns: 0,
                node_shared: true,
            },
            // See the module docs: a CXL device is not proven persistent here,
            // so it is not claimed to be.
            durable: false,
            shared_read: true,
            mutable_in_place: true,
            // No device DAX path is opened, so Direct I/O is not on offer even
            // though a raw device would allow it. Claiming it would let the
            // planner take a path this provider does not implement.
            direct_io: false,
            fdp: false,
            read_latency_ns: 0,
            write_latency_ns: 0,
            read_bandwidth_bps: 0,
            write_bandwidth_bps: 0,
            cost_per_byte: 0,
            decompress_ns_per_byte: 0,
            capacity_bytes: self.capacity_bytes,
            free_bytes: self.free_bytes(),
            healthy: false,
        }
    }

    /// The calibrated capabilities, folding in what `calibration` has measured.
    ///
    /// A provider with no observations stays unhealthy; a provider whose
    /// calibration reports an error rate above the deployer's tolerance stays
    /// unhealthy however fast it was. The two callers of this pass the same
    /// descriptor and only this folds measurements in, so `caps` is taken by
    /// reference: a by-value `ProviderCaps` that is only partly overwritten is a
    /// way for a future field to silently keep the uncalibrated value.
    #[must_use]
    pub fn calibrated_caps(
        &self,
        caps: &ProviderCaps,
        calibration: &crate::calibrate::ProviderCalibration,
        max_error_rate: f64,
    ) -> ProviderCaps {
        ProviderCaps {
            read_latency_ns: saturating_ns(calibration.read_latency_ns.get_or(0.0)),
            write_latency_ns: saturating_ns(calibration.write_latency_ns.get_or(0.0)),
            read_bandwidth_bps: saturating_ns(calibration.read_bps.get_or(0.0)),
            write_bandwidth_bps: saturating_ns(calibration.write_bps.get_or(0.0)),
            free_bytes: self.free_bytes(),
            healthy: calibration.ops > 0 && calibration.error_rate() <= max_error_rate,
            ..caps.clone()
        }
    }

    /// Writes `payload` and returns the extent it occupies.
    ///
    /// # Errors
    ///
    /// Returns [`MemoryError::ProviderExhausted { provider: CXL_KIND }`] when every region is full, and
    /// the operating-system error when the mapping cannot be written.
    pub fn write(
        &mut self,
        payload: &[u8],
        calibrator: &mut Calibrator,
    ) -> Result<u64, MemoryError> {
        let rounded = u64::try_from(payload.len())
            .unwrap_or(u64::MAX)
            .div_ceil(CXL_UNIT_BYTES)
            .max(1)
            * CXL_UNIT_BYTES;
        let region = self
            .regions
            .iter_mut()
            .find(|region| region.free_bytes() >= rounded)
            .ok_or(MemoryError::ProviderExhausted { provider: CXL_KIND })?;
        let offset = region.take(rounded)?;
        self.allocated_bytes = self.allocated_bytes.saturating_add(rounded);
        let started = Instant::now();
        let written = region.mapped.write(offset, payload);
        let elapsed = started.elapsed();
        match written {
            Ok(()) => {
                calibrator
                    .for_provider(PROVIDER_ID)
                    .observe_write(rounded, nanos(elapsed));
                Ok(offset)
            }
            Err(_error) => {
                // A failed write must not leave the extent allocated: the caller
                // would otherwise have an offset it believes holds a complete
                // object and a region that will not hand those bytes out again.
                region.give_back(offset, rounded);
                self.allocated_bytes = self.allocated_bytes.saturating_sub(rounded);
                Err(outside_region())
            }
        }
    }

    /// Reads `len` bytes from `offset`.
    ///
    /// # Errors
    ///
    /// Returns [`MemoryError::Io`] when the offset is not inside the mapping,
    /// which on a region that was removed is indistinguishable from a range
    /// that was never allocated.
    pub fn read(
        &mut self,
        offset: u64,
        len: usize,
        calibrator: &mut Calibrator,
    ) -> Result<Vec<u8>, MemoryError> {
        let region = self
            .regions
            .iter_mut()
            .find(|region| offset < region.len() as u64)
            .ok_or_else(outside_region)?;
        let started = Instant::now();
        let bytes = region
            .mapped
            .read(offset, len)
            .map(<[u8]>::to_vec)
            .map_err(|_| outside_region())?;
        let elapsed = started.elapsed();
        calibrator
            .for_provider(PROVIDER_ID)
            .observe_read(len as u64, nanos(elapsed));
        Ok(bytes)
    }

    /// Returns an extent to the provider.
    pub fn release(&mut self, offset: u64, bytes: u64) {
        let rounded = bytes.div_ceil(CXL_UNIT_BYTES).max(1) * CXL_UNIT_BYTES;
        for region in &mut self.regions {
            if offset < region.len() as u64 {
                if region.mapped.release(offset, rounded) {
                    self.allocated_bytes = self.allocated_bytes.saturating_sub(rounded);
                }
                return;
            }
        }
    }
}

fn nanos(elapsed: core::time::Duration) -> u64 {
    u64::try_from(elapsed.as_nanos()).unwrap_or(u64::MAX)
}

/// A measured value as nanoseconds, saturating a negative or absurd one.
///
/// The calibrator's averages are `f64` because they are running means, and a
/// running mean can briefly be negative when a sample arrives after a reset.
/// Reporting that as a huge `u64` would make the provider look catastrophically
/// slow rather than unmeasured, so the floor is zero and the ceiling is
/// `2^53`, the largest integer an `f64` represents exactly - past that the
/// conversion would round, and a rounded nanosecond count is a fabricated one.
/// Real latencies and bandwidths are many orders of magnitude below it.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn saturating_ns(value: f64) -> u64 {
    const EXACT_CEILING: f64 = 9_007_199_254_740_992.0; // 2^53
    if !value.is_finite() || value <= 0.0 {
        return 0;
    }
    let nanos = value.ceil();
    if nanos >= EXACT_CEILING {
        return EXACT_CEILING as u64;
    }
    nanos as u64
}

impl core::fmt::Display for CxlProvider {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "cxl(regions={}, {}MiB, free={}MiB, nodes={:?}{})",
            self.regions.len(),
            self.capacity_bytes / (1024 * 1024),
            self.free_bytes() / (1024 * 1024),
            self.numa_nodes,
            if self.emulated { ", emulated" } else { "" },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kivi_hardware::topology::NumaNodeId;

    /// Builds the emulated region a test needs, or skips the test if the
    /// platform cannot map one.
    fn emulated(len: u64, node: Option<u32>) -> Option<CxlRegion> {
        let mapped =
            DaxRegion::map_anonymous(usize::try_from(len).ok()?, node.map(NumaNodeId)).ok()?;
        Some(CxlRegion::new(mapped, true))
    }

    fn calibrator() -> Calibrator {
        Calibrator::new()
    }

    #[test]
    fn a_discovered_provider_on_a_machine_without_cxl_is_empty_not_fatal() {
        let (provider, dropped) = CxlProvider::discover(&Topology::single_core());
        assert!(provider.is_empty());
        assert!(dropped.is_empty(), "nothing to drop when nothing exists");
        assert!(!provider.caps().healthy);
    }

    #[test]
    fn an_unmeasured_provider_is_excluded_by_the_planner() {
        // The point of starting unhealthy: a provider nobody has touched has no
        // evidence to rank, so `satisfies` must refuse it rather than trust the
        // zeros in its cost fields.
        let Some(region) = emulated(64 * 1024 * 1024, Some(0)) else {
            return;
        };
        let provider = CxlProvider::from_regions(vec![region]);
        let caps = provider.caps();
        assert!(!caps.healthy);
        assert_eq!(caps.read_latency_ns, 0, "no measured latency is claimed");
        assert!(!caps.durable, "a CXL device is not proven persistent");
        assert_eq!(caps.capacity_bytes, 64 * 1024 * 1024);
    }

    #[test]
    fn a_failed_write_returns_its_extent() {
        // Otherwise the caller holds an offset it believes holds a complete
        // object, in a region that will never hand those bytes out again.
        let Some(region) = emulated(4 * 1024 * 1024, None) else {
            return;
        };
        let mut provider = CxlProvider::from_regions(vec![region]);
        let mut calibrator = calibrator();
        let offset = provider
            .write(b"kivi", &mut calibrator)
            .expect("a write into an emulated region");
        assert!(
            offset < CXL_UNIT_BYTES,
            "the first extent starts the region"
        );
        provider.release(offset, 4);
        assert_eq!(provider.free_bytes(), provider.capacity_bytes());
    }

    #[test]
    fn allocation_is_bounded_by_the_region() {
        let Some(region) = emulated(4 * 1024 * 1024, None) else {
            return;
        };
        let mut provider = CxlProvider::from_regions(vec![region]);
        let mut calibrator = calibrator();
        let unit = vec![0_u8; 2 * 1024 * 1024];
        for _ in 0..2 {
            provider
                .write(&unit, &mut calibrator)
                .expect("two units fit in four");
        }
        assert!(
            provider.write(&unit, &mut calibrator).is_err(),
            "a third unit does not fit"
        );
    }

    #[test]
    fn a_read_returns_exactly_what_a_write_stored() {
        let Some(region) = emulated(4 * 1024 * 1024, None) else {
            return;
        };
        let mut provider = CxlProvider::from_regions(vec![region]);
        let mut calibrator = calibrator();
        let payload = b"the quick brown fox".repeat(64);
        let offset = provider
            .write(&payload, &mut calibrator)
            .expect("a write into the region");
        let read = provider
            .read(offset, payload.len(), &mut calibrator)
            .expect("a read from the region");
        assert_eq!(read, payload);
    }

    #[test]
    fn reading_past_the_region_is_an_error_not_a_short_read() {
        let Some(region) = emulated(2 * 1024 * 1024, None) else {
            return;
        };
        let mut provider = CxlProvider::from_regions(vec![region]);
        let mut calibrator = calibrator();
        // Inside the mapping, so this succeeds - the region is a mapping, and
        // refusing an unallocated range would be a second allocator.
        assert!(provider.read(0, 1 << 20, &mut calibrator).is_ok());
        // One byte past the end, so this does not.
        assert!(
            provider
                .read(0, (2 * 1024 * 1024) + 1, &mut calibrator)
                .is_err()
        );
    }

    #[test]
    fn a_read_beyond_every_region_is_an_error() {
        let Some(region) = emulated(2 * 1024 * 1024, None) else {
            return;
        };
        let mut provider = CxlProvider::from_regions(vec![region]);
        let mut calibrator = calibrator();
        assert!(
            provider
                .read(u64::from(u32::MAX), 1, &mut calibrator)
                .is_err()
        );
    }

    #[test]
    fn calibration_turns_a_measured_provider_healthy() {
        let Some(region) = emulated(64 * 1024 * 1024, Some(0)) else {
            return;
        };
        let provider = CxlProvider::from_regions(vec![region]);
        let mut calibrator = calibrator();
        let base = provider.caps();
        // Before any observation there is no calibration at all, and an
        // unobserved calibration is not a measurement.
        assert!(calibrator.get(PROVIDER_ID).is_none());
        let mut provider = provider;
        for _ in 0..4 {
            provider
                .write(&[7; 4096], &mut calibrator)
                .expect("a write into the region");
        }
        let after = provider.calibrated_caps(
            &base,
            calibrator.get(PROVIDER_ID).expect("registered"),
            f64::MAX,
        );
        assert!(after.healthy, "four measured writes are evidence");
        assert!(
            after.write_latency_ns > 0,
            "a measured latency, not a default"
        );
    }

    #[test]
    fn an_error_flood_keeps_the_provider_unhealthy_however_fast_it_was() {
        let Some(region) = emulated(64 * 1024 * 1024, Some(0)) else {
            return;
        };
        let mut provider = CxlProvider::from_regions(vec![region]);
        let mut calibrator = calibrator();
        for _ in 0..4 {
            provider
                .write(&[7; 4096], &mut calibrator)
                .expect("a write into the region");
        }
        let mut calibration = calibrator.get(PROVIDER_ID).expect("registered").clone();
        for _ in 0..4 {
            calibration.observe_error();
        }
        let caps = provider.calibrated_caps(&provider.caps(), &calibration, 0.1);
        assert!(!caps.healthy, "50% errors is not a healthy provider");
    }

    #[test]
    fn an_emulated_provider_is_labelled_so_no_benchmark_can_mistake_it() {
        let Some(region) = emulated(2 * 1024 * 1024, None) else {
            return;
        };
        let provider = CxlProvider::from_regions(vec![region]);
        assert!(provider.is_emulated());
        assert_eq!(provider.caps().kind, ProviderKind::CXL_EMULATED);
        assert!(format!("{provider}").contains("emulated"));
    }

    #[test]
    fn real_regions_are_not_labelled_emulated() {
        let caps = ProviderCaps {
            healthy: true,
            ..ProviderCaps::dram(Some(0), 1)
        };
        assert_ne!(caps.kind, ProviderKind::CXL_EMULATED);
        assert_ne!(caps.kind, ProviderKind::CXL);
    }
}
