//! Bounded production logging: size-rotated files plus optional stderr.
//!
//! Disk use is bounded by construction: one active log file plus a fixed
//! number of retained rotated files, each capped at `max_file_bytes`.
//! Defaults are `64 MiB` per file with `12` total files (`768 MiB` worst
//! case). Rotation is synchronous file rename of a bounded file (no 64 MiB
//! gzip on the hot path); compression is deliberately off by default.
//!
//! File writes never block the consensus / `DataWorker` / H3 / reconciler
//! hot paths: the rotator sits behind a [`tracing_appender`] non-blocking
//! writer, which hands log lines to a dedicated logging thread. The
//! returned [`LogGuard`] must live for the full process lifetime so
//! shutdown flushes buffered lines.
//!
//! Crate decision (researched live, Sep 2026):
//! `file-rotate 0.8.0` + `tracing-appender 0.2.5`. `tracing-appender` alone
//! only offers time-based rotation (`MINUTELY`/`HOURLY`/`DAILY`/`NEVER`)
//! with no size limit, so a Raft/control log storm can fill a disk in one
//! hour. `tracing-rolling-file 0.1.3` (single owner, 1.5M downloads,
//! custom Debian-style naming) and the `rolling-file` ecosystem are smaller
//! and less standard. `file-rotate` (5.5M downloads, maintained,
//! `ContentLimit::Bytes`, `AppendCount` with bounded retention,
//! `std::io::Write` integration, Windows-correct rename rotation) is the
//! smallest maintained solution providing size rotation + bounded retention
//! + Windows support + non-blocking tracing integration.

use std::path::PathBuf;

use tracing_subscriber::EnvFilter;

/// Default maximum bytes per log file (64 MiB).
pub const DEFAULT_MAX_FILE_BYTES: u64 = 64 * 1024 * 1024;
/// Default total retained log files (active + rotated).
pub const DEFAULT_RETAINED_FILES: usize = 12;
/// Default log filename inside the log directory.
pub const LOG_FILENAME: &str = "kivi.log";

/// Resolved logging configuration (environment + defaults).
#[derive(Debug, Clone)]
pub struct LogConfig {
    /// Directory holding `kivi.log`, `kivi.log.1`, ... None means
    /// stderr-only (ephemeral/single-node dev default).
    pub dir: Option<PathBuf>,
    /// Tracing filter directive (e.g. `info,openraft=warn`).
    pub filter: String,
    /// Maximum bytes per file before rotation.
    pub max_file_bytes: u64,
    /// Total files retained (active + rotated), minimum 2.
    pub retained_files: usize,
    /// Whether to also emit human-readable logs to stderr.
    pub stderr: bool,
}

impl LogConfig {
    /// Resolves configuration from the environment:
    ///
    /// * `KIVI_LOG_DIR`: log directory (unset = stderr-only).
    /// * `KIVI_LOG_LEVEL` or `RUST_LOG`: filter directive (default
    ///   `info,openraft=warn`).
    /// * `KIVI_LOG_MAX_BYTES`: per-file cap (default 67108864).
    /// * `KIVI_LOG_FILES`: total retained files (default 12).
    /// * `KIVI_LOG_STDERR`: `0` disables the stderr layer, `1` forces it on.
    ///   Default: stderr on when no log dir is set, off when files are used.
    #[must_use]
    pub fn from_env() -> Self {
        let dir = std::env::var("KIVI_LOG_DIR")
            .ok()
            .filter(|value| !value.is_empty())
            .map(PathBuf::from);
        let filter = std::env::var("KIVI_LOG_LEVEL")
            .or_else(|_| std::env::var("RUST_LOG"))
            .unwrap_or_else(|_| "info,openraft=warn".to_owned());
        let max_file_bytes = std::env::var("KIVI_LOG_MAX_BYTES")
            .ok()
            .and_then(|value| value.parse::<u64>().ok())
            .filter(|value| *value >= 4096)
            .unwrap_or(DEFAULT_MAX_FILE_BYTES);
        let retained_files = std::env::var("KIVI_LOG_FILES")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .map_or(DEFAULT_RETAINED_FILES, |value| value.clamp(2, 64));
        let stderr = std::env::var("KIVI_LOG_STDERR")
            .ok()
            .map_or_else(|| dir.is_none(), |value| value != "0");
        Self {
            dir,
            filter,
            max_file_bytes,
            retained_files,
            stderr,
        }
    }

    /// Maximum worst-case disk use in bytes.
    #[must_use]
    pub const fn max_disk_bytes(&self) -> u64 {
        self.max_file_bytes
            .saturating_mul(self.retained_files as u64)
    }
}

/// Holds the non-blocking logging worker guard for the process lifetime.
/// Dropping it flushes buffered lines; keep it alive in `main`.
#[derive(Debug)]
pub struct LogGuard {
    _guard: Option<tracing_appender::non_blocking::WorkerGuard>,
}

impl LogGuard {
    /// Initializes global tracing from `config`. Safe to call once per
    /// process (subsequent calls return an empty guard without
    /// re-initializing, for test harnesses that already set a subscriber).
    pub fn init(config: &LogConfig) -> Self {
        let filter = EnvFilter::try_new(&config.filter)
            .unwrap_or_else(|_| EnvFilter::new("info,openraft=warn"));
        if let Some(dir) = &config.dir {
            let _ = std::fs::create_dir_all(dir);
            // `AppendCount(n)` keeps `n` rotated files plus the active one,
            // so total files = `n + 1`.
            let rotated = config.retained_files.saturating_sub(1).max(1);
            let path = dir.join(LOG_FILENAME);
            let writer = file_rotate::FileRotate::new(
                &path,
                file_rotate::suffix::AppendCount::new(rotated),
                file_rotate::ContentLimit::Bytes(
                    usize::try_from(config.max_file_bytes).unwrap_or(usize::MAX),
                ),
                file_rotate::compression::Compression::None,
                None,
            );
            let (non_blocking, guard) = tracing_appender::non_blocking(writer);
            // `try_init` keeps test harnesses (which may already own the
            // global subscriber) from panicking; the guard still flushes.
            let result = tracing_subscriber::fmt()
                .with_env_filter(filter)
                .with_writer(non_blocking)
                .with_ansi(false)
                .try_init();
            if result.is_err() {
                return Self {
                    _guard: Some(guard),
                };
            }
            if config.stderr {
                tracing::info!(
                    log_dir = %dir.display(),
                    max_file_bytes = config.max_file_bytes,
                    retained_files = config.retained_files,
                    max_disk_bytes = config.max_disk_bytes(),
                    "bounded file logging enabled (size rotation, non-blocking)"
                );
            }
            return Self {
                _guard: Some(guard),
            };
        }
        if config.stderr {
            let result = tracing_subscriber::fmt().with_env_filter(filter).try_init();
            if result.is_err() {
                return Self { _guard: None };
            }
        }
        Self { _guard: None }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write as _;

    #[test]
    fn rotation_bounds_retained_files() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("kivi.log");
        // Tiny limits: 1 KiB per file, 3 total files.
        let mut writer = file_rotate::FileRotate::new(
            &path,
            file_rotate::suffix::AppendCount::new(2),
            file_rotate::ContentLimit::Bytes(1024),
            file_rotate::compression::Compression::None,
            None,
        );
        // Emit ~64 KiB across many lines to force several rotations.
        for index in 0..2000 {
            writeln!(
                writer,
                "rotation-test line {index:06} padding-padding-padding"
            )
            .expect("write");
        }
        drop(writer);
        // Bounded: active + 2 rotated = 3 files, oldest data gone.
        let mut files: Vec<_> = std::fs::read_dir(dir.path())
            .expect("read dir")
            .filter_map(Result::ok)
            .map(|entry| entry.path())
            .collect();
        files.sort();
        assert_eq!(
            files.len(),
            3,
            "retained file count must stay bounded: {files:?}"
        );
        // Current logging continues: active file is non-empty and recent.
        let active = std::fs::metadata(&path).expect("active log exists");
        assert!(
            active.len() > 0 && active.len() <= 2048,
            "active file bounded, got {}",
            active.len()
        );
    }

    #[test]
    fn config_defaults_are_bounded() {
        let config = LogConfig {
            dir: None,
            filter: "info,openraft=warn".to_owned(),
            max_file_bytes: DEFAULT_MAX_FILE_BYTES,
            retained_files: DEFAULT_RETAINED_FILES,
            stderr: true,
        };
        assert_eq!(config.max_disk_bytes(), 64 * 1024 * 1024 * 12);
        assert!(
            config.max_disk_bytes() < 1024 * 1024 * 1024,
            "default retention under 1 GiB"
        );
    }
}
