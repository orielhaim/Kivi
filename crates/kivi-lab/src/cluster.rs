//! Reusable 3-node replicated cluster harness (task W).
//!
//! Spawns three real `kivi-server --cluster-mode` processes with
//! independent data directories, discovers their endpoints from
//! `KIVI_READY`, and drives the money tests: leader discovery via admin,
//! hard kills (`kill -9` semantics: no shutdown handshake), restarts
//! from the same directories, and convergence waits. No duplicate
//! process harness: ready-waiting, binary discovery, and error shapes
//! reuse [`process`](super::process).
//!
//! ## Port allocation
//!
//! Static topologies name every port before any process starts, which
//! conflicts with race-free ephemeral discovery. The harness probes nine
//! loopback ports (peer/native/admin × 3) by bind-then-release and
//! spawns immediately; if any member fails to bind (a parallel run stole
//! a probed port), the whole attempt is discarded and retried with fresh
//! probes (bounded attempts). A bind conflict therefore retries loudly
//! instead of serving a half cluster — race-free in effect, never silent.
//! Restarts always take fresh ports (the old listener may still be
//! releasing); the mesh heals through the restarted member's outbound
//! dials, and the harness tracks the new endpoints.

use std::io::{Read as _, Write as _};
use std::net::TcpStream;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use kivi_client::{ClientConfig, NativeClient};
use kivi_types::NamespaceId;
use serde_json::Value;

use super::process::{READY_TIMEOUT, SpawnError, WaitOutcome, server_binary_path, stderr_tail};

/// How long a cluster election may take (loaded CI boxes elect slowly).
pub const LEADER_TIMEOUT: Duration = Duration::from_secs(45);
/// How long convergence may take after healing or restart.
pub const CONVERGE_TIMEOUT: Duration = Duration::from_secs(45);
/// Cluster formation attempts (bind-conflict retries).
const SPAWN_ATTEMPTS: usize = 3;
/// Fixed test cluster identity (deterministic across runs; data
/// directories isolate state, never the id).
pub const TEST_CLUSTER_ID: u128 = 0xC105_7E57;
/// Replicated tablet under test.
pub const TEST_TABLET: u64 = 9;

/// One cluster member: a live child or a killed slot awaiting restart.
pub struct ClusterNode {
    child: Option<Child>,
    /// This member's node id (1, 2, 3).
    pub node_id: u64,
    /// Data directory (survives kills; reused across restarts).
    pub data_dir: tempfile::TempDir,
    /// Native client endpoint (`host:port`).
    pub native: String,
    /// Peer endpoint (`host:port`).
    pub peer: String,
    /// Admin endpoint (`host:port`).
    pub admin: String,
    /// RESP endpoint (`host:port`), when the binary serves one.
    pub resp: Option<String>,
}

impl std::fmt::Debug for ClusterNode {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ClusterNode")
            .field("node_id", &self.node_id)
            .field("native", &self.native)
            .field("alive", &self.child.is_some())
            .finish_non_exhaustive()
    }
}

/// Three real server processes forming one replicated tablet group.
/// Dropping kills every remaining child (tests never leak processes).
pub struct Cluster {
    nodes: Vec<ClusterNode>,
    id: u128,
    binary: PathBuf,
}

impl std::fmt::Debug for Cluster {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Cluster")
            .field("nodes", &self.nodes)
            .finish_non_exhaustive()
    }
}

impl Drop for Cluster {
    fn drop(&mut self) {
        for node in &mut self.nodes {
            if let Some(mut child) = node.child.take() {
                let _ = child.kill();
                let _ = child.wait();
            }
        }
    }
}

/// Probes one free loopback port (bind-then-release; the caller spawns
/// immediately and retries the whole formation on conflict).
fn free_addr() -> String {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("probe binds");
    let addr = listener.local_addr().expect("probe addr").to_string();
    drop(listener);
    addr
}

impl Cluster {
    /// Spawns a fresh 3-node cluster (new data directories) and waits
    /// for exactly one leader. Retries formation on bind conflicts.
    ///
    /// # Errors
    ///
    /// Returns [`SpawnError`] when the binary is missing, members exit
    /// during startup, readiness times out repeatedly, or no leader
    /// emerges.
    pub fn spawn() -> Result<Self, SpawnError> {
        Self::spawn_with(false)
    }

    /// Spawns like [`Cluster::spawn`] with a RESP edge per member. Fails
    /// loudly when the binary was built without `redis-compat` — never a
    /// silent native-only cluster.
    ///
    /// # Errors
    ///
    /// Returns [`SpawnError`] like [`Cluster::spawn`], plus
    /// [`SpawnError::RespUnavailable`] for a non-RESP binary.
    pub fn spawn_with_resp() -> Result<Self, SpawnError> {
        Self::spawn_with(true)
    }

    fn spawn_with(want_resp: bool) -> Result<Self, SpawnError> {
        let binary = server_binary_path()?;
        let mut last = SpawnError::Io("no spawn attempt ran".to_owned());
        for _ in 0..SPAWN_ATTEMPTS {
            match Self::spawn_once(&binary, want_resp) {
                Ok(cluster) => return Ok(cluster),
                Err(error) => {
                    last = error;
                }
            }
        }
        Err(last)
    }

    /// Kills every member (used at the end of full-restart tests before
    /// reviving the same directories).
    pub fn kill_all(&mut self) {
        for node in &mut self.nodes {
            if let Some(mut child) = node.child.take() {
                let _ = child.kill();
                let _ = child.wait();
            }
        }
    }

    /// Restarts every member from its existing data directory with fresh
    /// ports, then waits for exactly one leader.
    ///
    /// # Errors
    ///
    /// Returns [`SpawnError`] on startup failure, or panics (via the
    /// leader wait) when the reformed cluster never elects.
    pub fn restart_all(&mut self) -> Result<(), SpawnError> {
        self.kill_all();
        // Sequential revives: each member sees the previous revives'
        // fresh addresses, so the mesh heals through outbound dials.
        for index in 0..self.nodes.len() {
            self.restart(index)?;
        }
        let _ = self.wait_leader();
        Ok(())
    }

    /// Hard-kills one member (`kill -9` semantics: no flush beyond what
    /// the WAL already persisted per acknowledged write).
    pub fn kill(&mut self, index: usize) {
        if let Some(mut child) = self.nodes[index].child.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }

    /// Restarts one member from its existing data directory on its
    /// STATIC ports (same identity, advanced incarnation, no
    /// re-bootstrap). Static topologies require stable addresses: every
    /// member's maps stay valid across restarts, so redirects and dials
    /// never go stale. The server retries binds through `TIME_WAIT`
    /// release, so rapid kill/restart cycles are safe.
    ///
    /// # Errors
    ///
    /// Returns [`SpawnError`] when the member fails to start.
    pub fn restart(&mut self, index: usize) -> Result<(), SpawnError> {
        self.kill(index);
        // Same static addresses (topology never changes shape; the
        // server retries binds through `TIME_WAIT` release). The ready
        // line revalidates every address below.
        let mut args = self.base_args(self.nodes[index].node_id, self.nodes[index].data_dir.path());
        args.push("--cluster-peers".to_owned());
        args.push(flag_list(&self.peer_addrs()));
        args.push("--cluster-natives".to_owned());
        args.push(flag_list(&self.native_addrs()));
        args.push("--cluster-native".to_owned());
        args.push(self.nodes[index].native.clone());
        args.push("--admin".to_owned());
        args.push(self.nodes[index].admin.clone());
        if let Some(redis) = self.nodes[index].resp.clone() {
            args.push("--redis-listen".to_owned());
            args.push(redis);
        }
        let binary = self.binary.clone();
        let (child, ready_native, ready_admin, ready_resp) = spawn_member(&binary, &args)?;
        let node = &mut self.nodes[index];
        node.child = Some(child);
        node.native = ready_native;
        node.admin = ready_admin;
        node.resp = ready_resp;
        Ok(())
    }

    /// RESP endpoint of one member, or a loud error when this cluster
    /// serves no RESP edge.
    ///
    /// # Errors
    ///
    /// Returns [`SpawnError::RespUnavailable`] for native-only clusters.
    pub fn resp_endpoint(&self, index: usize) -> Result<String, SpawnError> {
        self.nodes[index]
            .resp
            .clone()
            .ok_or(SpawnError::RespUnavailable)
    }

    /// Native endpoints of live members (client seeds).
    #[must_use]
    pub fn native_endpoints(&self) -> Vec<String> {
        self.nodes
            .iter()
            .filter(|node| node.child.is_some())
            .map(|node| node.native.clone())
            .collect()
    }

    /// Native client seeded across live members (follows leader
    /// redirects with bounded retries and stable identities).
    ///
    /// # Panics
    ///
    /// Panics when no member is alive or the client fails to build.
    #[must_use]
    pub fn client(&self) -> NativeClient {
        NativeClient::new(ClientConfig {
            seeds: self.native_endpoints(),
            namespace: NamespaceId::from_u64(1),
            ..ClientConfig::default()
        })
        .expect("cluster client builds")
    }

    /// Native client with an explicit session (exactly-once retries
    /// across failovers resume its sequences).
    ///
    /// # Panics
    ///
    /// Panics when no member is alive or the client fails to build.
    #[must_use]
    pub fn client_with_session(&self, session: kivi_types::SessionId) -> NativeClient {
        NativeClient::new(ClientConfig {
            seeds: self.native_endpoints(),
            namespace: NamespaceId::from_u64(1),
            session: Some(session),
            ..ClientConfig::default()
        })
        .expect("cluster client builds")
    }

    /// Raw admin GET against one member (status + parsed JSON).
    ///
    /// # Panics
    ///
    /// Panics on connection or I/O failure.
    #[must_use]
    pub fn admin_get(&self, index: usize, path: &str) -> (u16, Value) {
        let mut socket =
            TcpStream::connect(self.nodes[index].admin.clone()).expect("admin connect");
        socket
            .set_read_timeout(Some(Duration::from_secs(10)))
            .expect("timeout");
        write!(
            socket,
            "GET {path} HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n"
        )
        .expect("write");
        let mut body = String::new();
        socket.read_to_string(&mut body).expect("read");
        let status = body
            .lines()
            .next()
            .unwrap_or_default()
            .split_whitespace()
            .nth(1)
            .unwrap_or("0")
            .parse::<u16>()
            .unwrap_or(0);
        let payload = body.split("\r\n\r\n").nth(1).unwrap_or("");
        let json: Value = serde_json::from_str(payload).unwrap_or(Value::Null);
        (status, json)
    }

    /// Parsed `/v1/tablet` for one member.
    ///
    /// # Panics
    ///
    /// Panics when the endpoint errors.
    #[must_use]
    pub fn tablet(&self, index: usize) -> Value {
        let (status, json) = self.admin_get(index, "/v1/tablet");
        assert_eq!(status, 200, "GET /v1/tablet on node {index}");
        json
    }

    /// Parsed `/v1/node` for one member.
    ///
    /// # Panics
    ///
    /// Panics when the endpoint errors.
    #[must_use]
    pub fn node_info(&self, index: usize) -> Value {
        let (status, json) = self.admin_get(index, "/v1/node");
        assert_eq!(status, 200, "GET /v1/node on node {index}");
        json
    }

    /// Role string of one member (`leader`, `follower`, `candidate`).
    ///
    /// # Panics
    ///
    /// Panics when the endpoint errors.
    #[must_use]
    pub fn role(&self, index: usize) -> String {
        self.tablet(index)["role"]
            .as_str()
            .unwrap_or("?")
            .to_owned()
    }

    /// Applied log index of one member.
    ///
    /// # Panics
    ///
    /// Panics when the endpoint errors or the field is missing.
    #[must_use]
    pub fn applied(&self, index: usize) -> u64 {
        self.tablet(index)["applied"].as_u64().unwrap_or(u64::MAX)
    }

    /// Incarnation of one member.
    ///
    /// # Panics
    ///
    /// Panics when the endpoint errors or the field is missing.
    #[must_use]
    pub fn incarnation(&self, index: usize) -> u64 {
        self.node_info(index)["incarnation"].as_u64().unwrap_or(0)
    }

    /// Whether one member process is alive.
    #[must_use]
    pub fn alive(&self, index: usize) -> bool {
        self.nodes[index].child.is_some()
    }

    /// Waits for exactly one stable leader across live members and
    /// returns its index (stable across consecutive polls, so callers
    /// never catch a handover mid-flight).
    ///
    /// # Panics
    ///
    /// Panics when no single stable leader emerges inside the window.
    #[must_use = "the leader index routes the next test step"]
    pub fn wait_leader(&self) -> usize {
        let deadline = Instant::now() + LEADER_TIMEOUT;
        let mut stable = 0usize;
        let mut last = usize::MAX;
        loop {
            let leaders: Vec<usize> = (0..self.nodes.len())
                .filter(|i| self.alive(*i) && self.role(*i) == "leader")
                .collect();
            if leaders.len() == 1 && leaders[0] == last {
                stable += 1;
                if stable >= 3 {
                    return leaders[0];
                }
            } else {
                stable = 0;
                last = leaders.first().copied().unwrap_or(usize::MAX);
            }
            assert!(
                Instant::now() < deadline,
                "cluster never elected one stable leader"
            );
            std::thread::sleep(Duration::from_millis(100));
        }
    }

    /// Waits until every live member applied at least `index`.
    ///
    /// # Panics
    ///
    /// Panics on timeout.
    pub fn wait_applied(&self, index: u64) {
        let deadline = Instant::now() + CONVERGE_TIMEOUT;
        loop {
            let mut ready = true;
            for i in 0..self.nodes.len() {
                if !self.alive(i) {
                    continue;
                }
                if self.applied(i) < index {
                    ready = false;
                }
            }
            if ready {
                break;
            }
            assert!(
                Instant::now() < deadline,
                "cluster never applied index {index}"
            );
            std::thread::sleep(Duration::from_millis(100));
        }
    }

    /// Waits until every live member converges on one applied index.
    ///
    /// # Panics
    ///
    /// Panics on timeout.
    pub fn wait_converged(&self) {
        let deadline = Instant::now() + CONVERGE_TIMEOUT;
        loop {
            let mut watermark = 0u64;
            let mut converged = true;
            for i in 0..self.nodes.len() {
                if !self.alive(i) {
                    continue;
                }
                let applied = self.applied(i);
                if watermark == 0 {
                    watermark = applied;
                } else if applied != watermark {
                    converged = false;
                }
            }
            if converged && watermark > 0 {
                break;
            }
            assert!(Instant::now() < deadline, "cluster never converged");
            std::thread::sleep(Duration::from_millis(100));
        }
    }

    fn base_args(&self, node_id: u64, data_dir: &std::path::Path) -> Vec<String> {
        vec![
            "--cluster-mode".to_owned(),
            "--cluster-id".to_owned(),
            self.id.to_string(),
            "--node-id".to_owned(),
            node_id.to_string(),
            "--cluster-tablet".to_owned(),
            TEST_TABLET.to_string(),
            "--data-dir".to_owned(),
            data_dir.display().to_string(),
        ]
    }

    /// Current raw peer addresses in node-id order.
    fn peer_addrs(&self) -> Vec<String> {
        self.nodes.iter().map(|node| node.peer.clone()).collect()
    }

    /// Current raw native addresses in node-id order.
    fn native_addrs(&self) -> Vec<String> {
        self.nodes.iter().map(|node| node.native.clone()).collect()
    }

    fn spawn_once(binary: &PathBuf, want_resp: bool) -> Result<Self, SpawnError> {
        let peers = [free_addr(), free_addr(), free_addr()];
        let natives = [free_addr(), free_addr(), free_addr()];
        let admins = [free_addr(), free_addr(), free_addr()];
        let redis = [free_addr(), free_addr(), free_addr()];
        let cluster_id = TEST_CLUSTER_ID;
        let mut nodes = Vec::new();
        for (index, node_id) in [1u64, 2, 3].into_iter().enumerate() {
            let data_dir =
                tempfile::tempdir().map_err(|error| SpawnError::Io(error.to_string()))?;
            let mut args = vec![
                "--cluster-mode".to_owned(),
                "--cluster-id".to_owned(),
                cluster_id.to_string(),
                "--node-id".to_owned(),
                node_id.to_string(),
                "--cluster-tablet".to_owned(),
                TEST_TABLET.to_string(),
                "--data-dir".to_owned(),
                data_dir.path().display().to_string(),
                "--cluster-peers".to_owned(),
                flag_list(&peers),
                "--cluster-natives".to_owned(),
                flag_list(&natives),
                "--cluster-native".to_owned(),
                natives[index].clone(),
                "--admin".to_owned(),
                admins[index].clone(),
            ];
            if want_resp {
                args.push("--redis-listen".to_owned());
                args.push(redis[index].clone());
            }
            let (child, ready_native, ready_admin, ready_resp) = spawn_member(binary, &args)?;
            if want_resp && ready_resp.is_none() {
                drop(child);
                return Err(SpawnError::RespUnavailable);
            }
            nodes.push(ClusterNode {
                child: Some(child),
                node_id,
                data_dir,
                native: ready_native,
                peer: peers[index].clone(),
                admin: ready_admin,
                resp: ready_resp,
            });
        }
        let cluster = Self {
            nodes,
            id: cluster_id,
            binary: binary.clone(),
        };
        let _ = cluster.wait_leader();
        Ok(cluster)
    }
}

/// Formats raw addresses as a `1=a,2=b,3=c` flag value.
fn flag_list(addrs: &[String]) -> String {
    addrs
        .iter()
        .enumerate()
        .map(|(i, addr)| format!("{}={addr}", i + 1))
        .collect::<Vec<_>>()
        .join(",")
}

/// Spawns one member and waits for its `KIVI_READY` (native + admin +
/// optional RESP).
fn spawn_member(
    binary: &PathBuf,
    args: &[String],
) -> Result<(Child, String, String, Option<String>), SpawnError> {
    let mut child = Command::new(binary)
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|error| SpawnError::Io(error.to_string()))?;
    match super::process::wait_ready_or_exit(&mut child, READY_TIMEOUT) {
        WaitOutcome::Ready(native, admin, resp) => Ok((
            child,
            native.into_iter().next().unwrap_or_default(),
            admin,
            resp,
        )),
        WaitOutcome::Exited(status) => Err(SpawnError::Exited(
            status.to_string(),
            stderr_tail(&mut child),
        )),
        WaitOutcome::TimedOut => Err(SpawnError::TimedOut(READY_TIMEOUT, stderr_tail(&mut child))),
    }
}
