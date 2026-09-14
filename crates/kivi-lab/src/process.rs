//! One reusable race-free server process harness.
//!
//! Every system test and lab command spawns `kivi-server` the same way: the
//! server binds ephemeral ports (`--port 0`, `--admin 127.0.0.1:0`,
//! `--redis-listen 127.0.0.1:0`) and reports the actual endpoints on stdout
//! (`KIVI_READY ...`). Tests connect to the reported addresses, so the old
//! probe-then-bind window — where a parallel test could steal a "reserved"
//! port between probe and bind — no longer exists anywhere. Restarts take
//! fresh ports: crash recovery never depends on socket addresses, only on
//! the data directory.
//!
//! [`Server::spawn_ephemeral_resp`] fails loudly when the server binary was
//! built without `redis-compat` (rebuild hint included). It never silently
//! skips: a lab run that explicitly requires RESP must fail clearly.

use std::io::{Read, Write};
use std::net::TcpStream;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
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
    /// A RESP endpoint was required but the binary was built without the
    /// `redis-compat` feature (no `redis=` in `KIVI_READY`).
    #[error(
        "server binary has no RESP endpoint; rebuild kivi-server with `--features redis-compat`"
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
    if let Ok(path) = std::env::var("KIVI_SERVER_BIN") {
        let path = PathBuf::from(&path);
        if path.is_file() {
            return Ok(path);
        }
        return Err(SpawnError::BinaryMissing(format!(
            "KIVI_SERVER_BIN={} is not a file; run `cargo build -p kivi-server` first",
            path.display()
        )));
    }
    let exe =
        std::env::current_exe().map_err(|error| SpawnError::BinaryMissing(error.to_string()))?;
    let name = if cfg!(windows) {
        "kivi-server.exe"
    } else {
        "kivi-server"
    };
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
        "no server binary beside {}; run `cargo build -p kivi-server` (add `--features redis-compat` for RESP runs) or set KIVI_SERVER_BIN",
        exe.display()
    )))
}

/// A running server: native endpoints, admin endpoint, and an optional
/// RESP endpoint. Dropping kills the child (tests never leak processes).
pub struct Server {
    child: Child,
    native: Vec<String>,
    admin: String,
    resp: Option<String>,
    namespace: NamespaceId,
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
    /// An early exit fails with the stderr tail — startup failure stays
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
        match probe_server(mode, workers, tablets, extra, want_resp, READY_TIMEOUT)? {
            ProbeOutcome::Ready(server) => {
                // Belt-and-braces: a successful connect proves the worker
                // listener serves, not just binds.
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
    /// the binary lacks the feature — never a silent skip.
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

    /// Raw HTTP GET against the admin plane (no HTTP client dependency;
    /// the responses are tiny JSON documents). Returns status + body.
    ///
    /// # Panics
    ///
    /// Panics on connection or I/O failure (test-harness loudness by design).
    #[must_use]
    pub fn admin_get(&self, path: &str) -> (u16, String) {
        let mut socket = TcpStream::connect(self.admin_endpoint()).expect("admin connect");
        socket
            .set_read_timeout(Some(Duration::from_secs(10)))
            .expect("timeout");
        write!(
            socket,
            "GET {path} HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n"
        )
        .expect("write");
        let mut body = String::new();
        socket
            .read_to_string(&mut body)
            .expect("read admin response");
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
        (status, payload)
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
    // Detect RESP capability from `--help` first: failing fast here beats
    // parsing a ready line that can never contain `redis=`.
    if want_resp && !binary_supports_resp(&binary)? {
        return Err(SpawnError::RespUnavailable);
    }
    let mut command = Command::new(&binary);
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
    let outcome = wait_ready_or_exit(&mut child, timeout);
    Ok(match outcome {
        WaitOutcome::Ready(native, admin, resp) => {
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
            })
        }
        WaitOutcome::Exited(status) => {
            let success = status.success();
            ProbeOutcome::Exited {
                status: status.to_string(),
                success,
                stderr: stderr_tail(&mut child),
            }
        }
        WaitOutcome::TimedOut => ProbeOutcome::TimedOut {
            stderr: stderr_tail(&mut child),
        },
    })
}

/// Internal ready-wait outcome (child ownership stays with the caller).
enum WaitOutcome {
    /// `KIVI_READY` parsed: native endpoints, admin endpoint, optional RESP.
    Ready(Vec<String>, String, Option<String>),
    /// The process exited before reporting readiness.
    Exited(std::process::ExitStatus),
    /// Neither readiness nor exit inside the window (child already killed).
    TimedOut,
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

/// Last lines of a dead child's stderr for failure diagnostics.
fn stderr_tail(child: &mut Child) -> String {
    let mut log = String::new();
    if let Some(stderr) = child.stderr.as_mut() {
        let _ = stderr.read_to_string(&mut log);
    }
    log.lines()
        .rev()
        .take(8)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect::<Vec<_>>()
        .join("\n")
}

/// Waits for `KIVI_READY` on the child's stdout, watching for early exit
/// on a reader thread so a silent stall can never hang the harness past
/// the deadline: the reader only produces lines, the loop owns the clock.
fn wait_ready_or_exit(child: &mut Child, timeout: Duration) -> WaitOutcome {
    let Some(stdout) = child.stdout.take() else {
        // Internal contract: the harness always pipes stdout before this
        // call, so a missing pipe is a harness bug, never runtime noise.
        panic!("server stdout not piped");
    };
    let (tx, rx) = std::sync::mpsc::channel::<Option<(Vec<String>, String, Option<String>)>>();
    std::thread::spawn(move || {
        use std::io::BufRead as _;
        let mut reader = std::io::BufReader::new(stdout);
        let mut line = String::new();
        loop {
            line.clear();
            match reader.read_line(&mut line) {
                Ok(0) | Err(_) => {
                    let _ = tx.send(None);
                    break;
                }
                Ok(_) => {
                    if let Some(ports) = parse_ready(&line) {
                        let _ = tx.send(Some(ports));
                        break;
                    }
                }
            }
        }
    });
    let deadline = Instant::now() + timeout;
    loop {
        if let Ok(Some(status)) = child.try_wait() {
            return WaitOutcome::Exited(status);
        }
        match rx.recv_timeout(Duration::from_millis(25)) {
            Ok(Some(ports)) => return WaitOutcome::Ready(ports.0, ports.1, ports.2),
            // EOF before readiness, or a quiet quantum: the child is
            // either dying (exit status collected above) or still starting;
            // loop on and let the deadline decide.
            Ok(None) | Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {}
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                if let Ok(Some(status)) = child.try_wait() {
                    return WaitOutcome::Exited(status);
                }
            }
        }
        if Instant::now() > deadline {
            let _ = child.kill();
            return WaitOutcome::TimedOut;
        }
    }
}

/// Parses `KIVI_READY workers=<a>,<b> admin=<c> [redis=<d>]`; tracing noise
/// on the shared stdout is skipped by the caller, never an error.
fn parse_ready(line: &str) -> Option<(Vec<String>, String, Option<String>)> {
    let rest = line.strip_prefix("KIVI_READY")?.trim();
    let mut workers: Option<&str> = None;
    let mut admin: Option<&str> = None;
    let mut redis: Option<&str> = None;
    for token in rest.split_whitespace() {
        if let Some(value) = token.strip_prefix("workers=") {
            workers = Some(value);
        }
        if let Some(value) = token.strip_prefix("admin=") {
            admin = Some(value);
        }
        if let Some(value) = token.strip_prefix("redis=") {
            redis = Some(value);
        }
    }
    let native: Vec<String> = workers?.split(',').map(str::to_owned).collect();
    let admin = admin?.to_owned();
    if native.is_empty() || admin.is_empty() {
        return None;
    }
    Some((native, admin, redis.map(str::to_owned)))
}
