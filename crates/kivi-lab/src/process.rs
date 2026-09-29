//! One reusable race-free server process harness.
//!
//! Every system test and lab command spawns `kivi-server` the same way: the
//! server binds ephemeral ports (`--port 0`, `--admin 127.0.0.1:0`,
//! `--redis-listen 127.0.0.1:0`) and reports the actual endpoints on stdout
//! (`KIVI_READY ...`). Tests connect to the reported addresses, so no
//! probe-then-bind window exists for a parallel test to steal a port in.
//! Restarts take fresh ports: crash recovery never depends on socket
//! addresses, only on the data directory.
//!
//! [`Server::spawn_ephemeral_resp`] fails loudly when the server binary was
//! built without the RESP edge (rebuild hint included). It never silently
//! skips: a lab run that explicitly requires RESP must fail clearly.

use std::collections::{BTreeMap, VecDeque};
use std::io::{Read, Write};
use std::net::TcpStream;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use kivi_client::{ClientConfig, NativeClient};
use kivi_types::NamespaceId;

/// How long `KIVI_READY` may take (recovery replays before listeners bind).
pub const READY_TIMEOUT: Duration = Duration::from_secs(20);

/// Server mode: pure memory, or crash-recoverable on a data directory.
#[derive(Debug, Clone)]
pub enum ServerMode {
    /// `--ephemeral`: benchmarks and development.
    Ephemeral,
    /// `--data-dir <path>`: kill/restart tests reuse the same path.
    DataDir(PathBuf),
}

/// What went wrong spawning or driving a server.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum SpawnError {
    /// The server binary could not be found.
    #[error("kivi-server binary not found: {0}")]
    BinaryMissing(String),
    /// The process exited before reporting readiness (stderr tail included).
    #[error("server exited during startup: {0}\nstderr tail:\n{1}")]
    Exited(String, String),
    /// No `KIVI_READY` inside the window (stderr tail included).
    #[error("server not ready after {0:?}\nstderr tail:\n{1}")]
    TimedOut(Duration, String),
    /// A RESP endpoint was required but the binary reports none (no `redis=` in
    /// `KIVI_READY`).
    ///
    /// The edge is a default feature, so this means the binary is either stale or was
    /// built with `--no-default-features`.
    #[error(
        "server binary reports no RESP endpoint; rebuild kivi-server (the edge is a \
         default feature) or drop --no-default-features"
    )]
    RespUnavailable,
    /// A plain I/O failure driving the child.
    #[error("process I/O: {0}")]
    Io(String),
}

/// Locates the `kivi-server` binary: `KIVI_SERVER_BIN` wins when set,
/// otherwise the `target/{debug,release}/` directory anchoring the current
/// test executable (which is how `cargo test -p kivi-lab` and
/// `cargo test --workspace` lay it out).
///
/// # Errors
///
/// Returns [`SpawnError::BinaryMissing`] when nothing is found, with the
/// exact build command to run.
pub fn server_binary_path() -> Result<PathBuf, SpawnError> {
    if let Some(path) = configured_server_binary()? {
        return Ok(path);
    }
    let exe =
        std::env::current_exe().map_err(|error| SpawnError::BinaryMissing(error.to_string()))?;
    let name = server_binary_name();
    // Bins live beside the current executable (`<target>/<profile>/`);
    // integration tests live one level deeper (`<target>/<profile>/deps/`).
    let mut dir = exe.parent();
    for _ in 0..2 {
        let Some(parent) = dir else { break };
        let path = parent.join(name);
        if path.is_file() {
            return Ok(path);
        }
        dir = parent.parent();
    }
    Err(SpawnError::BinaryMissing(format!(
        "no server binary beside {}; run `cargo build --release -p kivi-server`, \
         or a debug build for tests, or set KIVI_SERVER_BIN",
        exe.display()
    )))
}

/// Locates a release `kivi-server` beside the current Cargo target directory.
///
/// `KIVI_SERVER_BIN` still takes precedence, so an operator can select a
/// different release build without changing the lab binary.
///
/// # Errors
///
/// Returns [`SpawnError::BinaryMissing`] when no configured or release binary exists.
pub fn release_server_binary_path() -> Result<PathBuf, SpawnError> {
    if let Some(path) = configured_server_binary()? {
        if !is_release_path(&path) {
            return Err(SpawnError::BinaryMissing(format!(
                "KIVI_SERVER_BIN must point to a release binary for campaign runs: {}",
                path.display()
            )));
        }
        return Ok(path);
    }
    let exe =
        std::env::current_exe().map_err(|error| SpawnError::BinaryMissing(error.to_string()))?;
    let name = server_binary_name();
    let mut dir = exe.parent();
    for _ in 0..4 {
        let Some(parent) = dir else { break };
        let path = parent.join("release").join(name);
        if path.is_file() {
            return Ok(path);
        }
        dir = parent.parent();
    }
    Err(SpawnError::BinaryMissing(format!(
        "no release server binary beside {}; run `cargo build --release -p kivi-server`, \
         or set KIVI_SERVER_BIN",
        exe.display()
    )))
}

fn configured_server_binary() -> Result<Option<PathBuf>, SpawnError> {
    let Ok(path) = std::env::var("KIVI_SERVER_BIN") else {
        return Ok(None);
    };
    let path = PathBuf::from(path);
    if path.is_file() {
        return Ok(Some(path));
    }
    Err(SpawnError::BinaryMissing(format!(
        "KIVI_SERVER_BIN={} is not a file; run `cargo build --release -p kivi-server` first",
        path.display()
    )))
}

fn server_binary_name() -> &'static str {
    if cfg!(windows) {
        "kivi-server.exe"
    } else {
        "kivi-server"
    }
}

fn is_release_path(path: &Path) -> bool {
    path.components().any(|component| {
        component
            .as_os_str()
            .to_string_lossy()
            .eq_ignore_ascii_case("release")
    })
}

/// A running server: native endpoints, admin endpoint, and an optional
/// RESP endpoint. Dropping kills the child (tests never leak processes).
pub struct Server {
    child: Child,
    native: Vec<String>,
    admin: String,
    resp: Option<String>,
    namespace: NamespaceId,
    stderr_lines: LineTail,
    /// The child's stdout. A Kivi server logs through `tracing` to stdout, so
    /// this is where the explanation for a failure actually is.
    stdout_lines: LineTail,
}

impl std::fmt::Debug for Server {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Server")
            .field("native", &self.native)
            .field("admin", &self.admin)
            .field("resp", &self.resp)
            .field("namespace", &self.namespace)
            .finish_non_exhaustive()
    }
}

impl Server {
    /// Spawns with ephemeral ports and waits for `KIVI_READY` (which the
    /// server prints after every listener binds, so recovery already ran).
    /// An early exit fails with the stderr tail - startup failure stays
    /// loud, never a silent hang.
    ///
    /// # Errors
    ///
    /// Returns [`SpawnError`] on missing binary, early exit, or timeout.
    pub fn spawn(
        mode: ServerMode,
        workers: usize,
        tablets: usize,
        extra: &[&str],
        want_resp: bool,
    ) -> Result<Self, SpawnError> {
        let binary = server_binary_path()?;
        Self::spawn_with_binary(&binary, mode, workers, tablets, extra, want_resp)
    }

    /// Spawns a specific server binary with ephemeral ports and waits for
    /// `KIVI_READY`.
    ///
    /// # Errors
    ///
    /// Returns [`SpawnError`] on missing binary, early exit, or timeout.
    pub fn spawn_with_binary(
        binary: &Path,
        mode: ServerMode,
        workers: usize,
        tablets: usize,
        extra: &[&str],
        want_resp: bool,
    ) -> Result<Self, SpawnError> {
        match probe_server_with_binary(
            binary,
            mode,
            workers,
            tablets,
            extra,
            want_resp,
            READY_TIMEOUT,
        )? {
            ProbeOutcome::Ready(server) => {
                TcpStream::connect(server.endpoint()).map_err(|error| {
                    SpawnError::Io(format!("worker accepts after ready: {error}"))
                })?;
                Ok(server)
            }
            ProbeOutcome::Exited { status, stderr, .. } => Err(SpawnError::Exited(status, stderr)),
            ProbeOutcome::TimedOut { stderr } => Err(SpawnError::TimedOut(READY_TIMEOUT, stderr)),
        }
    }

    /// Spawns an ephemeral 2-worker / 1-tablet server (the common case).
    ///
    /// # Errors
    ///
    /// Returns [`SpawnError`] on missing binary, early exit, or timeout.
    pub fn spawn_ephemeral() -> Result<Self, SpawnError> {
        Self::spawn(ServerMode::Ephemeral, 2, 1, &[], false)
    }

    /// Spawns an ephemeral server with a RESP endpoint. Fails loudly when
    /// the binary lacks the feature - never a silent skip.
    ///
    /// # Errors
    ///
    /// Returns [`SpawnError`] on missing binary, early exit, timeout, or a
    /// non-RESP binary ([`SpawnError::RespUnavailable`]).
    pub fn spawn_ephemeral_resp() -> Result<Self, SpawnError> {
        Self::spawn(ServerMode::Ephemeral, 2, 1, &[], true)
    }

    /// Spawns on a data directory (durable mode), like [`Server::spawn`].
    ///
    /// # Errors
    ///
    /// Returns [`SpawnError`] on missing binary, early exit, or timeout.
    pub fn spawn_auto(data_dir: &Path, extra: &[&str]) -> Result<Self, SpawnError> {
        Self::spawn(ServerMode::DataDir(data_dir.to_owned()), 2, 1, extra, false)
    }

    /// First native worker endpoint (`host:port`).
    #[must_use]
    pub fn endpoint(&self) -> String {
        self.native.first().cloned().unwrap_or_default()
    }

    /// All native worker endpoints in worker-index order.
    #[must_use]
    pub fn native_endpoints(&self) -> &[String] {
        &self.native
    }

    /// Admin endpoint (`host:port`).
    #[must_use]
    pub fn admin_endpoint(&self) -> String {
        self.admin.clone()
    }

    /// RESP endpoint, or a loud error when this server has none.
    ///
    /// # Errors
    ///
    /// Returns [`SpawnError::RespUnavailable`] for native-only servers.
    pub fn resp_endpoint(&self) -> Result<String, SpawnError> {
        self.resp.clone().ok_or(SpawnError::RespUnavailable)
    }

    /// Process id of the spawned server.
    #[must_use]
    pub fn pid(&self) -> u32 {
        self.child.id()
    }

    /// Lightweight process diagnostics without an additional dependency.
    #[must_use]
    pub fn diagnostics(&mut self) -> BTreeMap<String, String> {
        let mut values = BTreeMap::new();
        values.insert("pid".to_owned(), self.child.id().to_string());
        let alive = self.child.try_wait().ok().flatten().is_none();
        values.insert("alive".to_owned(), alive.to_string());
        values.insert("stderr_tail".to_owned(), snapshot(&self.stderr_lines));
        // A Kivi server logs through `tracing` to stdout, so a stderr-only tail
        // explains nothing. This is the field that makes a harness-reported
        // failure diagnosable.
        values.insert("stdout_tail".to_owned(), snapshot(&self.stdout_lines));
        #[cfg(target_os = "linux")]
        if let Ok(status) = std::fs::read_to_string(format!("/proc/{}/status", self.child.id())) {
            for line in status.lines().filter(|line| {
                line.starts_with("VmRSS:")
                    || line.starts_with("VmSize:")
                    || line.starts_with("Threads:")
            }) {
                if let Some((key, value)) = line.split_once(':') {
                    values.insert(key.to_ascii_lowercase(), value.trim().to_owned());
                }
            }
        }
        values
    }

    /// Raw HTTP GET against the admin plane without panicking.
    ///
    /// # Errors
    ///
    /// Returns a string when the admin socket cannot be reached or read.
    pub fn try_admin_get(&self, path: &str) -> Result<(u16, String), String> {
        try_admin_get_endpoint(&self.admin_endpoint(), path)
    }

    /// Raw HTTP GET against the admin plane (no HTTP client dependency).
    ///
    /// # Panics
    ///
    /// Panics on connection or I/O failure (test-harness loudness by design).
    #[must_use]
    pub fn admin_get(&self, path: &str) -> (u16, String) {
        self.try_admin_get(path).expect("admin request succeeds")
    }

    /// Extracts the first JSON number following `"key":` (admin DTOs are
    /// tiny and machine-shaped; a full JSON parser is not warranted).
    ///
    /// # Panics
    ///
    /// Panics when the endpoint errors or the key is absent/non-numeric.
    pub fn admin_number(&self, path: &str, key: &str) -> u64 {
        let (status, body) = self.admin_get(path);
        assert_eq!(status, 200, "GET {path}");
        let needle = format!("\"{key}\":");
        let start = body
            .find(&needle)
            .unwrap_or_else(|| panic!("{key} present in {path}: {body}"))
            + needle.len();
        body[start..]
            .chars()
            .take_while(char::is_ascii_digit)
            .collect::<String>()
            .parse()
            .unwrap_or_else(|_| panic!("{key} numeric in {path}: {body}"))
    }

    /// Native client seeded at worker 0.
    ///
    /// # Panics
    ///
    /// Panics when the client fails to build (unreachable seed).
    #[must_use]
    pub fn client(&self) -> NativeClient {
        NativeClient::new(ClientConfig {
            seeds: vec![self.endpoint()],
            namespace: self.namespace,
            ..ClientConfig::default()
        })
        .expect("client builds")
    }

    /// The honest crash: no shutdown handshake, no flush beyond what the
    /// WAL already persisted per acknowledged write.
    ///
    /// # Panics
    ///
    /// Panics when the kill signal itself fails.
    pub fn kill(mut self) {
        self.child.kill().expect("kill succeeds");
        let _ = self.child.wait();
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// Raw HTTP GET against an admin endpoint without panicking.
///
/// # Errors
///
/// Returns a string when the admin socket cannot be reached or read.
pub fn try_admin_get_endpoint(endpoint: &str, path: &str) -> Result<(u16, String), String> {
    let mut socket =
        TcpStream::connect(endpoint).map_err(|error| format!("admin connect: {error}"))?;
    socket
        .set_read_timeout(Some(Duration::from_secs(10)))
        .map_err(|error| format!("admin timeout: {error}"))?;
    write!(
        socket,
        "GET {path} HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n"
    )
    .map_err(|error| format!("admin write: {error}"))?;
    let mut body = String::new();
    socket
        .read_to_string(&mut body)
        .map_err(|error| format!("admin read: {error}"))?;
    let status = body
        .lines()
        .next()
        .unwrap_or_default()
        .split_whitespace()
        .nth(1)
        .unwrap_or("0")
        .parse::<u16>()
        .unwrap_or(0);
    let payload = body.split("\r\n\r\n").nth(1).unwrap_or("").to_owned();
    Ok((status, payload))
}

/// What a probed server did inside its startup window.
#[derive(Debug)]
pub enum ProbeOutcome {
    /// `KIVI_READY` parsed: serving (recovery already ran).
    Ready(Server),
    /// The process exited before reporting readiness.
    Exited {
        /// Exit status text.
        status: String,
        /// Whether the status reports success.
        success: bool,
        /// Last stderr lines for diagnostics.
        stderr: String,
    },
    /// Neither readiness nor exit inside the window (caller owns the
    /// still-running child, already killed).
    TimedOut {
        /// Last stderr lines for diagnostics.
        stderr: String,
    },
}

/// Starts a server like [`Server::spawn`] but reports early exit and
/// timeout as data instead of errors: the primitive behind refusal tests
/// (second owner of a live directory must fail) and corruption tests (a
/// broken chain must either fail loudly or serve a fallback).
///
/// # Errors
///
/// Returns [`SpawnError`] only when the binary itself cannot be started
/// or probed (missing binary, I/O failure, non-RESP binary with
/// `want_resp`).
pub fn probe_server(
    mode: ServerMode,
    workers: usize,
    tablets: usize,
    extra: &[&str],
    want_resp: bool,
    timeout: Duration,
) -> Result<ProbeOutcome, SpawnError> {
    let binary = server_binary_path()?;
    probe_server_with_binary(&binary, mode, workers, tablets, extra, want_resp, timeout)
}

fn probe_server_with_binary(
    binary: &Path,
    mode: ServerMode,
    workers: usize,
    tablets: usize,
    extra: &[&str],
    want_resp: bool,
    timeout: Duration,
) -> Result<ProbeOutcome, SpawnError> {
    // Detect RESP capability from `--help` first: failing fast here beats
    // parsing a ready line that can never contain `redis=`.
    if want_resp && !binary_supports_resp(binary)? {
        return Err(SpawnError::RespUnavailable);
    }
    let mut command = Command::new(binary);
    match mode {
        ServerMode::Ephemeral => {
            command.arg("--ephemeral");
        }
        ServerMode::DataDir(dir) => {
            command.arg("--data-dir").arg(dir);
        }
    }
    command
        .arg("--port")
        .arg("0")
        .arg("--admin")
        .arg("127.0.0.1:0")
        .arg("--workers")
        .arg(workers.to_string())
        .arg("--tablets")
        .arg(tablets.to_string());
    if want_resp {
        command.arg("--redis-listen").arg("127.0.0.1:0");
    }
    command.args(extra);
    let mut child = command
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|error| SpawnError::Io(error.to_string()))?;
    let Some(errstream) = child.stderr.take() else {
        panic!("server stderr not piped");
    };
    let stderr_lines = drain_stream(errstream, Arc::new(|_| {}));
    let outcome = wait_ready_or_exit(&mut child, timeout);
    Ok(match outcome {
        WaitOutcome::Ready {
            native,
            admin,
            resp,
            stdout,
        } => {
            if want_resp && resp.is_none() {
                let _ = child.kill();
                let _ = child.wait();
                return Err(SpawnError::RespUnavailable);
            }
            ProbeOutcome::Ready(Server {
                child,
                native,
                admin,
                resp,
                namespace: NamespaceId::from_u64(1),
                stderr_lines,
                stdout_lines: stdout,
            })
        }
        WaitOutcome::Exited { status, stdout } => {
            let success = status.success();
            ProbeOutcome::Exited {
                status: status.to_string(),
                success,
                // The server logs to stdout, so a bare stderr tail explains
                // nothing. Quote both.
                stderr: format!("{}\n--- stdout ---\n{stdout}", snapshot(&stderr_lines)),
            }
        }
        WaitOutcome::TimedOut { stdout } => ProbeOutcome::TimedOut {
            stderr: format!("{}\n--- stdout ---\n{stdout}", snapshot(&stderr_lines)),
        },
    })
}

/// Internal ready-wait outcome (child ownership stays with the caller).
pub(crate) enum WaitOutcome {
    /// `KIVI_READY` parsed: native endpoints, admin endpoint, optional RESP, and
    /// the stdout tail so far.
    Ready {
        /// Worker endpoints (single-node) or the one native endpoint (cluster).
        native: Vec<String>,
        /// Admin HTTP endpoint.
        admin: String,
        /// RESP endpoint, when the child was asked for one and has it.
        resp: Option<String>,
        /// The live tail of the child's stdout. It keeps filling after this
        /// point, and the lines that explain a later failure are the ones written
        /// *after* readiness - so this is the tail itself, not a snapshot of it.
        stdout: LineTail,
    },
    /// The process exited before reporting readiness.
    Exited {
        /// How it ended.
        status: std::process::ExitStatus,
        /// Its stdout, which is where a server logs.
        stdout: String,
    },
    /// Neither readiness nor exit inside the window (child already killed).
    TimedOut {
        /// Its stdout, which is where a server logs.
        stdout: String,
    },
}

/// Checks `--help` output for the RESP flag (fast capability probe that
/// never starts listeners).
fn binary_supports_resp(binary: &Path) -> Result<bool, SpawnError> {
    let output = Command::new(binary)
        .arg("--help")
        .output()
        .map_err(|error| SpawnError::Io(error.to_string()))?;
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    Ok(text.contains("redis-listen"))
}

/// A ring of the last [`LINE_HISTORY`] lines a child stream produced.
type LineTail = Arc<Mutex<VecDeque<String>>>;

/// How many lines of a child's output to keep for diagnostics.
const LINE_HISTORY: usize = 64;

/// Drains one of a child's output streams on its own thread until EOF.
///
/// # It never stops early
///
/// Both streams must be drained for the child's whole life. A reader that returns
/// after the line it was looking for closes the pipe, and the child's next write
/// blocks once the kernel buffer fills - which, for a server that logs every
/// consensus decision to stdout, is a matter of a few kilobytes. The process stays
/// alive, so a supervisor watching only for liveness sees a healthy child while
/// the server is wedged mid-log-write. That is the worst shape a harness bug can
/// take: it produces failures that look like the system under test.
fn drain_stream<R>(stream: R, on_line: Arc<dyn Fn(&str) + Send + Sync>) -> LineTail
where
    R: std::io::Read + Send + 'static,
{
    let lines: LineTail = Arc::new(Mutex::new(VecDeque::new()));
    let captured = Arc::clone(&lines);
    std::thread::spawn(move || {
        use std::io::BufRead as _;
        let mut reader = std::io::BufReader::new(stream);
        let mut line = String::new();
        loop {
            line.clear();
            match reader.read_line(&mut line) {
                Ok(0) | Err(_) => break,
                Ok(_) => {
                    if !line.trim().is_empty()
                        && let Ok(mut captured) = captured.lock()
                    {
                        if captured.len() == LINE_HISTORY {
                            captured.pop_front();
                        }
                        captured.push_back(line.trim_end().to_owned());
                    }
                    on_line(&line);
                }
            }
        }
    });
    lines
}

fn snapshot(lines: &LineTail) -> String {
    lines
        .lock()
        .map(|captured| captured.iter().cloned().collect::<Vec<_>>().join("\n"))
        .unwrap_or_default()
}

/// Waits for `KIVI_READY` on the child's stdout, watching for early exit
/// on a reader thread so a silent stall can never hang the harness past
/// the deadline: the reader only produces lines, the loop owns the clock.
///
/// The stdout reader started here runs until EOF, not until readiness. It used to
/// stop at `KIVI_READY`, which closed the pipe and let the server block on a log
/// write a few kilobytes later - see [`drain_stream`]. Its line tail comes back on
/// every arm so a failure can quote what the child actually said.
pub(crate) fn wait_ready_or_exit(child: &mut Child, timeout: Duration) -> WaitOutcome {
    let (tx, rx) = std::sync::mpsc::channel::<Option<(Vec<String>, String, Option<String>)>>();
    let Some(stdout) = child.stdout.take() else {
        // Internal contract: the harness always pipes stdout before this
        // call, so a missing pipe is a harness bug, never runtime noise.
        panic!("server stdout not piped");
    };
    let ready_tx = std::sync::Arc::new(std::sync::Mutex::new(Some(tx)));
    let lines = drain_stream(
        stdout,
        Arc::new({
            let ready_tx = std::sync::Arc::clone(&ready_tx);
            move |line| {
                // Parse before taking the slot. Taking it first meant the first
                // line that was not a readiness line - and a server that logs to
                // stdout emits several before it is ready - consumed the one-shot
                // sender, so `KIVI_READY` could never be seen and every spawn
                // timed out against a server that had in fact bound its ports and
                // printed them. The harness reported "no RESP endpoint" and
                // "connection refused" for processes that were serving.
                let Some(ports) = parse_ready(line) else {
                    return;
                };
                if let Ok(mut slot) = ready_tx.lock()
                    && let Some(tx) = slot.take()
                {
                    let _ = tx.send(Some(ports));
                }
            }
        }),
    );
    let deadline = Instant::now() + timeout;
    loop {
        if let Ok(Some(status)) = child.try_wait() {
            return WaitOutcome::Exited {
                status,
                stdout: snapshot(&lines),
            };
        }
        match rx.recv_timeout(Duration::from_millis(25)) {
            Ok(Some(ports)) => {
                return WaitOutcome::Ready {
                    native: ports.0,
                    admin: ports.1,
                    resp: ports.2,
                    stdout: Arc::clone(&lines),
                };
            }
            // EOF before readiness, or a quiet quantum: the child is
            // either dying (exit status collected above) or still starting;
            // loop on and let the deadline decide.
            Ok(None) | Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {}
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                if let Ok(Some(status)) = child.try_wait() {
                    return WaitOutcome::Exited {
                        status,
                        stdout: snapshot(&lines),
                    };
                }
            }
        }
        if Instant::now() > deadline {
            let _ = child.kill();
            let _ = child.wait();
            return WaitOutcome::TimedOut {
                stdout: snapshot(&lines),
            };
        }
    }
}

/// Parses readiness lines; tracing noise on the shared stdout is skipped
/// by the caller, never an error. Two shapes (both carry `admin=`, plus
/// an optional `redis=`):
///
/// ```text
/// KIVI_READY workers=<a>,<b> admin=<c> [redis=<d>]   (single-node)
/// KIVI_READY native=<a> peer=<b> admin=<c> [redis=<d>] ... (cluster)
/// ```
///
/// The returned native list is the worker list (single-node) or the
/// single native endpoint (cluster); extra cluster tokens (`peer=`,
/// `node=`, ...) are ignored here (the cluster harness reads what it
/// needs separately).
pub(crate) fn parse_ready(line: &str) -> Option<(Vec<String>, String, Option<String>)> {
    let rest = line.strip_prefix("KIVI_READY")?.trim();
    let mut workers: Option<&str> = None;
    let mut native: Option<&str> = None;
    let mut admin: Option<&str> = None;
    let mut redis: Option<&str> = None;
    for token in rest.split_whitespace() {
        if let Some(value) = token.strip_prefix("workers=") {
            workers = Some(value);
        }
        if let Some(value) = token.strip_prefix("native=") {
            native = Some(value);
        }
        if let Some(value) = token.strip_prefix("admin=") {
            admin = Some(value);
        }
        if let Some(value) = token.strip_prefix("redis=") {
            redis = Some(value);
        }
    }
    let native: Vec<String> = match (workers, native) {
        (Some(list), _) => list.split(',').map(str::to_owned).collect(),
        (None, Some(addr)) => vec![addr.to_owned()],
        (None, None) => return None,
    };
    let admin = admin?.to_owned();
    if native.is_empty() || admin.is_empty() {
        return None;
    }
    Some((native, admin, redis.map(str::to_owned)))
}
