//! DAMON: the kernel's memory access monitor, over its own sysfs control
//! interface.
//!
//! No third-party crate wraps DAMON, and none should be added: the kernel ABI
//! is a small set of text files under `/sys/kernel/mm/damon`, and a binding
//! that hid it would only move the sysfs reads somewhere less legible.
//!
//! ## What DAMON can and cannot answer
//!
//! DAMON samples the CPU's physical address space at a configurable interval
//! and reports, per region, how often that region was accessed and how often
//! its pages were faulted. It does **not** know about Kivi objects. So it is
//! used for what it is actually good at:
//!
//! * which of Kivi's arenas are being touched at all, and how hard;
//! * whether a memory node's pages are being reached from a core on another
//!   node, which is the one thing software counters cannot see;
//! * whether the process's resident set is behaving the way the memory fabric
//!   believes it is.
//!
//! It is explicitly *not* used as a second hotness estimate to be added to the
//! software one. See [`crate::telemetry::signals`].
//!
//! ## Granularity is the hard limit
//!
//! DAMON aggregates over a region defined by three boundaries, with a minimum
//! alignment of one page for the inner boundaries and one 2 MiB-aligned
//! boundary for the outer. An arena smaller than a few megabytes is
//! indistinguishable from its neighbours, so a Kivi arena that small is mapped
//! to the region containing it rather than being measured on its own. That is
//! reported as [`DamonReport::resolved_regions`], so a caller can tell the
//! difference between "DAMON says this arena is cold" and "DAMON could not see
//! this arena".
//!
//! ## Failure is the normal case
//!
//! DAMON is absent from most hardened and container kernels, and when present
//! its control files are usually root-only. Every operation therefore reports
//! absence rather than failing, and the whole backend is inert unless the
//! caller explicitly opens a monitor.

#![allow(clippy::cast_precision_loss)] // See 	elemetry::pmu for why a count-to-64 widening is exact here.

use std::collections::BTreeMap;
use std::io;
use std::path::{Path, PathBuf};

use crate::capability::{DamonParams, Support, Unavailable};

/// Where the kernel exposes the DAMON control interface.
const DAMON_ROOT: &str = "/sys/kernel/mm/damon";

/// One of Kivi's arenas, described well enough for DAMON to resolve it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WatchedRegion {
    /// Stable name of the arena, for example `worker-0/arena-hot`.
    pub name: String,
    /// First byte of the arena's address range.
    pub start: u64,
    /// Length of the arena's address range.
    pub len: u64,
}

impl WatchedRegion {
    /// Builds a region description.
    #[must_use]
    pub fn new(name: impl Into<String>, start: u64, len: u64) -> Self {
        Self {
            name: name.into(),
            start,
            len,
        }
    }

    /// Whether the region is small enough that DAMON cannot resolve it against
    /// its neighbours. A region below 2 MiB shares at least one DAMON
    /// aggregation boundary with whatever is next to it.
    #[must_use]
    pub fn resolvable(&self) -> bool {
        self.len >= 2 * 1024 * 1024
    }
}

/// One observation of one region.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct RegionAccess {
    /// Accesses DAMON counted in the last report period, already scaled by the
    /// aggregation period DAMON itself reports.
    pub accesses: u64,
    /// Page faults DAMON counted in the last report period.
    pub faults: u64,
    /// Number of 2 MiB samples the region was resolved into. Below two, the
    /// observation is a statement about a neighbourhood, not about the arena.
    pub samples: u32,
}

impl RegionAccess {
    /// Accesses per 2 MiB sample, which is the only comparable unit: two
    /// regions of different sizes cannot be compared on raw access counts.
    #[must_use]
    pub fn accesses_per_mebibyte(&self) -> f64 {
        if self.samples == 0 {
            return 0.0;
        }
        (self.accesses.min(u64::from(u32::MAX)) as f64) / f64::from(self.samples) / 2.0
    }

    /// Whether the observation is specific enough to act on.
    #[must_use]
    pub fn is_resolved(&self) -> bool {
        self.samples >= 2
    }
}

/// What one DAMON report said.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DamonReport {
    /// Per-region observations, keyed by arena name.
    pub regions: BTreeMap<String, RegionAccess>,
    /// Regions DAMON could not resolve, because they are too small or their
    /// addresses were not registered. Recorded explicitly so a caller never
    /// reads an absent key as "cold".
    pub unresolved: Vec<String>,
    /// Whether the kernel could write a report at all.
    pub available: bool,
}

impl DamonReport {
    /// A report from a monitor that is not running.
    #[must_use]
    pub fn unavailable() -> Self {
        Self::default()
    }

    /// The observation for one arena, or `None` when DAMON could not resolve
    /// it.
    #[must_use]
    pub fn region(&self, name: &str) -> Option<&RegionAccess> {
        self.regions.get(name)
    }
}

/// A running DAMON monitor.
///
/// The monitor owns a kernel monitor directory, registers Kivi's arenas as
/// `saddr`/`eaddr` pairs, and reads the kernel's own `damon_stats` file. It
/// holds no device handle and no file descriptor, so a monitor that the kernel
/// removes simply stops producing reports.
#[derive(Debug)]
pub struct DamonMonitor {
    monitor: Option<PathBuf>,
    regions: Vec<WatchedRegion>,
}

impl DamonMonitor {
    /// Registers `regions` and starts the kernel monitor.
    ///
    /// # Errors
    ///
    /// Returns the operating-system error when the control files are not
    /// writable, which is the common case: DAMON's control interface is
    /// root-only on most kernels. A caller that cannot tolerate the error
    /// should use [`try_open`].
    pub fn open(regions: &[WatchedRegion]) -> io::Result<Self> {
        Self::try_open(regions).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::NotFound,
                "damon control interface is not available",
            )
        })
    }

    /// Registers `regions` and starts the kernel monitor, or returns `None`
    /// when the control interface is absent or not writable.
    #[must_use]
    pub fn try_open(regions: &[WatchedRegion]) -> Option<Self> {
        let root = PathBuf::from(DAMON_ROOT);
        let admin = root.join("admin");
        if !admin.is_dir() {
            return None;
        }
        let monitor = admin.join("monitor_0");
        if !monitor.is_dir() {
            return None;
        }
        let mut self_ = Self {
            monitor: Some(monitor),
            regions: Vec::new(),
        };
        for (id, region) in regions.iter().enumerate() {
            if self_.register(region, id).is_err() {
                continue;
            }
            self_.regions.push(region.clone());
        }
        if self_.regions.is_empty() {
            return None;
        }
        if self_.write("state", "start").is_err() {
            return None;
        }
        Some(self_)
    }

    /// True when the kernel monitor is actually running.
    #[must_use]
    pub fn is_active(&self) -> bool {
        self.monitor
            .as_ref()
            .and_then(|dir| std::fs::read_to_string(dir.join("state")).ok())
            .is_some_and(|state| state.trim() == "start")
    }

    /// Regions DAMON could register.
    #[must_use]
    pub fn regions(&self) -> &[WatchedRegion] {
        &self.regions
    }

    /// Reads the kernel's report.
    ///
    /// Parsing is driven by the column header the kernel prints, not by field
    /// position. The column set has changed across kernel versions, and a
    /// positional parser would silently attribute one kernel's fault count to
    /// another's access count. A report without a recognised header is reported
    /// as unavailable rather than guessed at.
    #[must_use]
    pub fn report(&self) -> DamonReport {
        let Some(dir) = &self.monitor else {
            return DamonReport::unavailable();
        };
        let Ok(raw) = std::fs::read_to_string(dir.join("damon_stats")) else {
            return DamonReport::unavailable();
        };
        let Some(rows) = parse_stats(&raw) else {
            // An unrecognised layout is a report Kivi did not read, not a
            // report that said nothing.
            return DamonReport::unavailable();
        };
        let mut report = DamonReport {
            available: true,
            ..DamonReport::default()
        };
        for region in &self.regions {
            report
                .regions
                .insert(region.name.clone(), RegionAccess::default());
        }
        for (start, access) in rows {
            match self.region_for(start) {
                Some(region) => {
                    report.regions.insert(region.name.clone(), access);
                }
                None => {
                    report.unresolved.push(format!("{start:#x}"));
                }
            }
        }
        for region in &self.regions {
            if !report.regions.contains_key(&region.name) {
                report.unresolved.push(region.name.clone());
            }
        }
        report
    }

    fn region_for(&self, start: u64) -> Option<&WatchedRegion> {
        self.regions
            .iter()
            .find(|region| region.start <= start && start < region.start + region.len)
    }

    /// Stops the kernel monitor. Kivi does this on shutdown so the kernel
    /// stops sampling a process that is going away.
    pub fn stop(&mut self) {
        if self.is_active() {
            let _ = self.write("state", "stop");
        }
        self.monitor = None;
    }

    fn register(&self, region: &WatchedRegion, id: usize) -> io::Result<()> {
        let dir = self
            .monitor
            .as_ref()
            .ok_or_else(|| io::Error::other("damon monitor not open"))?;
        // DAMON ids are 0..n and must be unique within the monitor. The kernel
        // matches a region by address, so the id only has to be stable for the
        // life of the monitor.
        std::fs::write(
            dir.join(format!("saddr{id}")),
            format!("{:#x}", region.start),
        )?;
        std::fs::write(
            dir.join(format!("eaddr{id}")),
            format!("{:#x}", region.start + region.len),
        )?;
        Ok(())
    }

    fn write(&self, name: &str, value: &str) -> io::Result<()> {
        let dir = self
            .monitor
            .as_ref()
            .ok_or_else(|| io::Error::other("damon monitor not open"))?;
        std::fs::write(dir.join(name), value)
    }
}

impl Drop for DamonMonitor {
    fn drop(&mut self) {
        self.stop();
    }
}

fn parse_count(field: &str) -> u64 {
    field.parse::<u64>().unwrap_or(0)
}

/// Parses a `damon_stats` file into `(start address, observation)` rows.
///
/// `None` when the file has no column header Kivi recognises, which is the only
/// honest answer: a layout this code does not understand is a report it did not
/// read.
///
/// The kernel writes a `#`-prefixed settings line (`#aggregated <n> <us>`) and
/// then a *bare* column header,
/// `nr_accesses_sample nr_accesses nr_faults_sample nr_faults saddr eaddr`.
/// The header is therefore identified by naming a column, not by a leading `#`,
/// and the start address is the second-to-last column rather than the first.
/// Every field is located by name; nothing here depends on a column's position.
#[must_use]
pub fn parse_stats(raw: &str) -> Option<Vec<(u64, RegionAccess)>> {
    let mut header: Option<StatsHeader<'_>> = None;
    let mut rows = Vec::new();
    for line in raw.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let fields: Vec<&str> = line.split_whitespace().collect();
        let Some(header) = &mut header else {
            // A line that names a known column *is* the header. Anything else
            // arriving before it is a scalar setting, not a data row.
            header = StatsHeader::recognise(&fields);
            continue;
        };
        if fields.len() < header.names.len() {
            continue;
        }
        let value = |name: &str| -> u64 {
            header
                .names
                .iter()
                .position(|column| *column == name)
                .and_then(|index| fields.get(index))
                .map_or(0, |field| parse_count(field))
        };
        // The kernel prints addresses as `0x`-prefixed hex.
        let Some(field) = fields.get(header.saddr) else {
            continue;
        };
        let Ok(start) = u64::from_str_radix((*field).trim_start_matches("0x"), 16) else {
            continue;
        };
        let samples = value("nr_accesses_sample")
            .max(value("nr_faults_sample"))
            .max(1);
        rows.push((
            start,
            RegionAccess {
                accesses: value("nr_accesses"),
                faults: value("nr_faults"),
                samples: u32::try_from(samples).unwrap_or(u32::MAX),
            },
        ));
    }
    header.map(|_| rows)
}

/// A recognised `damon_stats` column header, with the address column's index
/// resolved once.
///
/// Resolving `saddr` here rather than per row is what makes the parser total: a
/// header is only ever accepted when it names `saddr`, so no row can reach an
/// index that does not exist.
#[derive(Debug, Clone)]
struct StatsHeader<'a> {
    names: Vec<&'a str>,
    saddr: usize,
}

impl<'a> StatsHeader<'a> {
    /// Accepts `names` as a header when it names the columns Kivi needs.
    fn recognise(names: &[&'a str]) -> Option<Self> {
        let saddr = names.iter().position(|name| *name == "saddr")?;
        names.iter().position(|name| *name == "nr_accesses")?;
        Some(Self {
            names: names.to_vec(),
            saddr,
        })
    }
}

/// Whether the kernel's DAMON control interface exists and this process may
/// control it.
#[must_use]
pub fn detect() -> Support<DamonParams> {
    let admin = Path::new(DAMON_ROOT).join("admin");
    if !admin.is_dir() {
        return Support::Unavailable(Unavailable::NotPresent);
    }
    let Ok(entries) = std::fs::read_dir(&admin) else {
        return Support::Unavailable(Unavailable::NotPresent);
    };
    let monitor = entries.filter_map(Result::ok).find_map(|entry| {
        entry
            .file_name()
            .to_str()
            .is_some_and(|name| name.starts_with("monitor_"))
            .then(|| entry.path())
            .filter(|dir| dir.join("state").exists())
    });
    let Some(monitor) = monitor else {
        return Support::Unavailable(Unavailable::NotPresent);
    };
    // Writability is probed by *opening* the control file, not by writing to
    // it. Opening requires the same permission a write would and changes
    // nothing, which matters because a probe that wrote "start" then "stop"
    // would switch off a DAMON monitor belonging to somebody else.
    if std::fs::OpenOptions::new()
        .write(true)
        .open(monitor.join("state"))
        .is_err()
    {
        return Support::Unavailable(Unavailable::PermissionDenied);
    }
    // Read from the monitor that was actually found, not from a hardcoded
    // `monitor_0`, so a kernel offering only `monitor_1` is not reported with
    // another monitor's interval.
    let free_us = std::fs::read_to_string(monitor.join("aggr_interval"))
        .ok()
        .and_then(|raw| raw.trim().parse::<u64>().ok())
        .unwrap_or(0);
    let aggressive = Path::new(DAMON_ROOT).join("aggr_interval").exists();
    Support::Available(DamonParams {
        free_us,
        aggressive,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detection_answers_even_without_damon() {
        let support = detect();
        assert!(!support.to_string().is_empty());
    }

    /// The layout the kernel writes (`mm/damon/sysfs.c`). The settings line is
    /// `#`-prefixed; the column header is *not*, and the address columns are the
    /// last two rather than the first.
    const DOCUMENTED_STATS: &str = "\
#aggregated 100000 100
nr_accesses_sample nr_accesses nr_faults_sample nr_faults saddr eaddr
1589 11422 59 41 0x7f0000000000 0x7f0001000000
1611 12233 62 45 0x7f0001000000 0x7f0002000000
";

    #[test]
    fn the_documented_stats_layout_parses_by_column_name() {
        let rows = parse_stats(DOCUMENTED_STATS).expect("a recognised header");
        assert_eq!(rows.len(), 2, "the settings line is not a data row");
        let (start, first) = rows[0];
        assert_eq!(start, 0x7f00_0000_0000, "saddr, not the first column");
        assert_eq!(first.accesses, 11_422);
        assert_eq!(first.faults, 41);
        assert_eq!(
            first.samples, 1_589,
            "the widest sample count is the resolution"
        );
        assert!(first.is_resolved());
        let (second_start, second) = rows[1];
        assert_eq!(second_start, 0x7f00_0100_0000);
        assert_eq!(second.accesses, 12_233);
        // Accesses normalise per 2 MiB sample, so the denser region reads
        // higher despite a similar raw count.
        assert!(second.accesses_per_mebibyte() > first.accesses_per_mebibyte());
    }

    #[test]
    fn a_layout_with_no_saddr_column_is_not_guessed_at() {
        // A header this code cannot map is a report it did not read. Guessing
        // would attribute one kernel's access count to another's faults.
        let unknown = "\
#aggregated 100000 100
alpha beta gamma
1 2 3
";
        assert!(parse_stats(unknown).is_none());
    }

    #[test]
    fn a_settings_line_before_the_header_is_not_a_data_row() {
        // A bare settings line (no leading `#`) must not be mistaken for a row
        // and must not be allowed to become the header.
        let settings_first = "\
aggregated 100000 100
nr_accesses nr_faults saddr eaddr
10 2 0x1000 0x2000
";
        let rows = parse_stats(settings_first).expect("the header is recognised");
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].0, 0x1000);
    }

    #[test]
    fn a_short_data_row_is_skipped_not_read_past_its_end() {
        let ragged = "\
#aggregated 100000 100
nr_accesses nr_faults saddr eaddr
10 2 0x1000 0x2000
30 4
";
        let rows = parse_stats(ragged).expect("the header is recognised");
        assert_eq!(
            rows.len(),
            1,
            "the truncated row is not evidence about a region"
        );
        assert_eq!(rows[0].1.accesses, 10);
    }

    #[test]
    fn a_non_hex_address_row_is_skipped() {
        let bad = "\
#aggregated 100000 100
nr_accesses nr_faults saddr eaddr
10 2 not-an-address 0x2000
";
        let rows = parse_stats(bad).expect("the header is recognised");
        assert!(rows.is_empty(), "an unread address is not zero access");
    }

    #[test]
    fn a_missing_monitor_opens_as_none_never_panics() {
        // A machine with no DAMON, or an unreadable one, must not prevent
        // startup.
        let regions = [WatchedRegion::new("arena", 0x1000, 4 << 20)];
        let monitor = DamonMonitor::try_open(&regions);
        if let Some(mut monitor) = monitor {
            assert!(!monitor.regions().is_empty());
            monitor.stop();
        }
    }

    #[test]
    fn an_unavailable_report_says_so() {
        let report = DamonReport::unavailable();
        assert!(!report.available);
        assert!(report.region("arena").is_none());
        assert!(report.unresolved.is_empty());
    }

    #[test]
    fn a_region_below_the_aggregation_granularity_is_not_resolvable() {
        assert!(!WatchedRegion::new("small", 0, 512 * 1024).resolvable());
        assert!(WatchedRegion::new("large", 0, 8 * 1024 * 1024).resolvable());
    }

    #[test]
    fn unresolved_regions_read_as_absent_not_cold() {
        let access = RegionAccess {
            accesses: 1000,
            faults: 0,
            samples: 1,
        };
        assert!(!access.is_resolved());
        assert!((access.accesses_per_mebibyte() - 500.0).abs() < 1e-9);
    }

    #[test]
    fn per_mebibyte_normalises_by_size() {
        let small = RegionAccess {
            accesses: 100,
            faults: 0,
            samples: 2,
        };
        let large = RegionAccess {
            accesses: 400,
            faults: 0,
            samples: 8,
        };
        assert!((small.accesses_per_mebibyte() - large.accesses_per_mebibyte()).abs() < 1e-9);
    }
}
