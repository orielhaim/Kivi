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
//! loopback ports (UDP for the QUIC peer mesh, TCP for native/admin × 3)
//! by bind-then-release and spawns immediately; if any member fails to
//! bind (a parallel run stole a probed port), the whole attempt is
//! discarded and retried with fresh probes (bounded attempts). A bind
//! conflict therefore retries loudly instead of serving a half cluster —
//! race-free in effect, never silent. Restarts reuse static ports (the
//! mesh heals through the restarted member's outbound dials).

use std::io::{Read as _, Write as _};
use std::net::TcpStream;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use kivi_client::{ClientConfig, NativeClient};
use kivi_types::NamespaceId;
use serde_json::Value;

use super::process::{READY_TIMEOUT, SpawnError, server_binary_path};

/// How long a cluster election may take (loaded CI boxes elect slowly).
pub const LEADER_TIMEOUT: Duration = Duration::from_secs(45);
/// How long convergence may take after healing or restart.
pub const CONVERGE_TIMEOUT: Duration = Duration::from_secs(45);
/// Cluster formation attempts (bind-conflict retries).
const SPAWN_ATTEMPTS: usize = 3;
/// Fixed test cluster identity (deterministic across runs; data
/// directories isolate state, never the id).
pub const TEST_CLUSTER_ID: u128 = 0xC105_7E57;
/// Default tablet count for single-tablet regression clusters.
pub const TEST_TABLETS: usize = 1;
/// Default consensus worker count for test clusters.
pub const TEST_WORKERS: usize = 2;

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

/// Three real server processes forming replicated tablet groups (one
/// replica of every tablet per process at formation; membership moves
/// through the control plane afterwards).
/// Dropping kills every remaining child (tests never leak processes).
pub struct Cluster {
    nodes: Vec<ClusterNode>,
    id: u128,
    binary: PathBuf,
    tablet_count: usize,
    worker_count: usize,
    layout: String,
    extra_env: Vec<(String, String)>,
    ready_timeout: Duration,
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
        // `KIVI_LAB_KEEP_DIRS` preserves data directories (with per-member
        // server logs) for post-mortems instead of deleting them.
        let keep = std::env::var("KIVI_LAB_KEEP_DIRS").is_ok();
        for node in &mut self.nodes {
            if let Some(mut child) = node.child.take() {
                let _ = child.kill();
                let _ = child.wait();
            }
            if keep {
                node.data_dir.disable_cleanup(true);
                eprintln!("kivi-lab: kept data dir {}", node.data_dir.path().display());
            }
        }
    }
}

/// Probes one free loopback TCP port (bind-then-release; the caller
/// spawns immediately and retries the whole formation on conflict).
/// Native, admin, and RESP edges stay TCP.
fn free_addr() -> String {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("probe binds");
    let addr = listener.local_addr().expect("probe addr").to_string();
    drop(listener);
    addr
}

/// Probes one free loopback UDP port for the QUIC peer mesh. TCP probes
/// say nothing about UDP availability (separate namespaces), so peer
/// ports probe UDP while the mesh binds with retries on conflict.
fn free_udp_addr() -> String {
    let socket = std::net::UdpSocket::bind("127.0.0.1:0").expect("probe binds");
    let addr = socket.local_addr().expect("probe addr").to_string();
    drop(socket);
    addr
}

/// One member's ordered tiling view: `(tablet, range_start, range_end)`.
type Tiling = Vec<(u64, Vec<u8>, Option<Vec<u8>>)>;

impl Cluster {
    /// Spawns a fresh 3-node single-tablet cluster (new data directories)
    /// and waits for exactly one leader. Retries formation on bind
    /// conflicts.
    ///
    /// # Errors
    ///
    /// Returns [`SpawnError`] when the binary is missing, members exit
    /// during startup, readiness times out repeatedly, or no leader
    /// emerges.
    pub fn spawn() -> Result<Self, SpawnError> {
        Self::spawn_with_tablets(TEST_TABLETS, TEST_WORKERS, false)
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
        Self::spawn_with_tablets(TEST_TABLETS, TEST_WORKERS, true)
    }

    /// Spawns a fresh 3-node cluster with `tablet_count` tablets striped
    /// over `worker_count` consensus workers per node, waiting for every
    /// tablet to elect exactly one leader. Retries formation on bind
    /// conflicts.
    ///
    /// # Errors
    ///
    /// Returns [`SpawnError`] like [`Cluster::spawn`].
    pub fn spawn_with_tablets(
        tablet_count: usize,
        worker_count: usize,
        want_resp: bool,
    ) -> Result<Self, SpawnError> {
        Self::spawn_full(tablet_count, worker_count, want_resp, &[])
    }

    /// Spawns a fresh 3-node ORDERED cluster (`--layout ordered`) with
    /// `tablet_count` ordered tablets, waiting for every tablet to elect
    /// exactly one leader. Range scans, transactions, and secondary
    /// indexes run against these clusters.
    ///
    /// # Errors
    ///
    /// Returns [`SpawnError`] like [`Cluster::spawn`].
    pub fn spawn_ordered(tablet_count: usize) -> Result<Self, SpawnError> {
        Self::spawn_with_layout(tablet_count, TEST_WORKERS, "ordered")
    }

    /// Spawns a fresh 3-node cluster with an explicit namespace layout.
    ///
    /// # Errors
    ///
    /// Returns [`SpawnError`] like [`Cluster::spawn`].
    pub fn spawn_with_layout(
        tablet_count: usize,
        worker_count: usize,
        layout: &str,
    ) -> Result<Self, SpawnError> {
        Self::spawn_full_timeout_layout(
            tablet_count,
            worker_count,
            false,
            &[],
            READY_TIMEOUT,
            layout,
        )
    }

    /// Spawns like [`Cluster::spawn_with_tablets`] with an explicit
    /// readiness window per member (large formations open hundreds of
    /// groups per process; the default 20 s window suits small ones).
    /// The window persists on the cluster for later restarts.
    ///
    /// # Errors
    ///
    /// Returns [`SpawnError`] like [`Cluster::spawn`].
    pub fn spawn_with_tablets_timeout(
        tablet_count: usize,
        worker_count: usize,
        want_resp: bool,
        ready_timeout: Duration,
    ) -> Result<Self, SpawnError> {
        Self::spawn_full_timeout(tablet_count, worker_count, want_resp, &[], ready_timeout)
    }

    /// Spawns like [`Cluster::spawn_with_tablets`] with extra environment
    /// variables for every member (correctness probes such as
    /// `KIVI_DISABLE_PREFLIGHT=1`; test-only insecure peer TLS is always
    /// set).
    ///
    /// # Errors
    ///
    /// Returns [`SpawnError`] like [`Cluster::spawn`].
    pub fn spawn_full(
        tablet_count: usize,
        worker_count: usize,
        want_resp: bool,
        extra_env: &[(&str, &str)],
    ) -> Result<Self, SpawnError> {
        Self::spawn_full_timeout(
            tablet_count,
            worker_count,
            want_resp,
            extra_env,
            READY_TIMEOUT,
        )
    }

    /// Spawns like [`Cluster::spawn_full`] with an explicit readiness
    /// window per member (persisted for later restarts).
    ///
    /// # Errors
    ///
    /// Returns [`SpawnError`] like [`Cluster::spawn`].
    pub fn spawn_full_timeout(
        tablet_count: usize,
        worker_count: usize,
        want_resp: bool,
        extra_env: &[(&str, &str)],
        ready_timeout: Duration,
    ) -> Result<Self, SpawnError> {
        Self::spawn_full_timeout_layout(
            tablet_count,
            worker_count,
            want_resp,
            extra_env,
            ready_timeout,
            "hash",
        )
    }

    /// Spawns like [`Cluster::spawn_full_timeout`] with an explicit
    /// namespace layout.
    ///
    /// # Errors
    ///
    /// Returns [`SpawnError`] like [`Cluster::spawn`].
    pub fn spawn_full_timeout_layout(
        tablet_count: usize,
        worker_count: usize,
        want_resp: bool,
        extra_env: &[(&str, &str)],
        ready_timeout: Duration,
        layout: &str,
    ) -> Result<Self, SpawnError> {
        let cluster = Self::spawn_bare_layout(
            tablet_count,
            worker_count,
            want_resp,
            extra_env,
            ready_timeout,
            layout,
        )?;
        // Single-tablet clusters keep the legacy single-leader wait;
        // multi-tablet formations wait for every tablet to elect.
        if cluster.tablet_count == 1 {
            let _ = cluster.wait_leader();
        } else {
            let _ = cluster.wait_all_leaders();
        }
        Ok(cluster)
    }

    /// Spawns a fresh 3-node cluster WITHOUT waiting for elections:
    /// members are up (ready) but groups may still be campaigning. Density
    /// probes use this with their own generous convergence loops; every
    /// other spawn helper waits for stable leadership before returning.
    ///
    /// # Errors
    ///
    /// Returns [`SpawnError`] like [`Cluster::spawn`].
    pub fn spawn_bare(
        tablet_count: usize,
        worker_count: usize,
        want_resp: bool,
        extra_env: &[(&str, &str)],
        ready_timeout: Duration,
    ) -> Result<Self, SpawnError> {
        Self::spawn_bare_layout(
            tablet_count,
            worker_count,
            want_resp,
            extra_env,
            ready_timeout,
            "hash",
        )
    }

    /// Spawns like [`Cluster::spawn_bare`] with an explicit namespace
    /// layout.
    ///
    /// # Errors
    ///
    /// Returns [`SpawnError`] like [`Cluster::spawn`].
    pub fn spawn_bare_layout(
        tablet_count: usize,
        worker_count: usize,
        want_resp: bool,
        extra_env: &[(&str, &str)],
        ready_timeout: Duration,
        layout: &str,
    ) -> Result<Self, SpawnError> {
        let binary = server_binary_path()?;
        let mut last = SpawnError::Io("no spawn attempt ran".to_owned());
        for _ in 0..SPAWN_ATTEMPTS {
            match Self::spawn_once(
                &binary,
                tablet_count,
                worker_count,
                want_resp,
                extra_env,
                ready_timeout,
                layout,
            ) {
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

    /// Restarts every member from its existing data directory on its
    /// static ports, then waits for leadership everywhere (one leader for
    /// single-tablet clusters, one per tablet otherwise).
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
        if self.tablet_count == 1 {
            let _ = self.wait_leader();
        } else {
            let _ = self.wait_all_leaders();
        }
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
    /// never go stale. Killed processes release UDP ports immediately;
    /// the server also retries binds through release races, so rapid
    /// kill/restart cycles are safe.
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
        args.push("--cluster-admins".to_owned());
        args.push(flag_list(&self.admin_addrs()));
        args.push("--cluster-native".to_owned());
        args.push(self.nodes[index].native.clone());
        args.push("--admin".to_owned());
        args.push(self.nodes[index].admin.clone());
        if let Some(redis) = self.nodes[index].resp.clone() {
            args.push("--redis-listen".to_owned());
            args.push(redis);
        }
        let binary = self.binary.clone();
        let env: Vec<(&str, &str)> = self
            .extra_env
            .iter()
            .map(|(key, value)| (key.as_str(), value.as_str()))
            .collect();
        let data_dir = self.nodes[index].data_dir.path().to_owned();
        let ready_timeout = self.ready_timeout;
        let (child, ready_native, ready_admin, ready_resp) =
            spawn_member(&binary, &args, &data_dir, &env, ready_timeout)?;
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

    /// Diagnoses a dead member for failure output: exit status plus
    /// the tail of its server log (post-mortem without keeping dirs).
    fn diagnose(&self, index: usize) -> String {
        let node = &self.nodes[index];
        let status = if node.child.is_some() {
            "still spawned"
        } else {
            "no child handle"
        };
        let tail = file_tail(&node.data_dir.path().join("server-stderr.log"));
        format!(
            "member {index} (node {}): {status}; stderr tail: {tail}",
            node.node_id
        )
    }

    /// Raw admin GET against one member (status + parsed JSON).
    ///
    /// # Panics
    ///
    /// Panics on connection or I/O failure (naming the member and its
    /// exit state instead of a bare refused dial).
    #[must_use]
    pub fn admin_get(&self, index: usize, path: &str) -> (u16, Value) {
        let admin = self.nodes[index].admin.clone();
        let mut socket = TcpStream::connect(admin.clone()).unwrap_or_else(|error| {
            panic!(
                "admin connect node {index} ({admin}) {path}: {error}; {}",
                self.diagnose(index)
            )
        });
        socket
            .set_read_timeout(Some(Duration::from_secs(10)))
            .expect("timeout");
        write!(
            socket,
            "GET {path} HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n"
        )
        .unwrap_or_else(|error| panic!("admin write node {index} ({admin}) {path}: {error}"));
        let mut body = String::new();
        socket
            .read_to_string(&mut body)
            .unwrap_or_else(|error| panic!("admin read node {index} ({admin}) {path}: {error}"));
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

    /// Raw admin POST with a JSON body against one member.
    ///
    /// # Panics
    ///
    /// Panics on connection or I/O failure.
    #[must_use]
    pub fn admin_post(&self, index: usize, path: &str, body: &Value) -> (u16, Value) {
        let payload = body.to_string();
        let admin = self.nodes[index].admin.clone();
        let mut socket = TcpStream::connect(admin.clone()).unwrap_or_else(|error| {
            panic!(
                "admin connect node {index} ({admin}) {path}: {error}; {}",
                self.diagnose(index)
            )
        });
        socket
            .set_read_timeout(Some(Duration::from_secs(10)))
            .expect("timeout");
        write!(
            socket,
            "POST {path} HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{payload}",
            payload.len()
        )
        .unwrap_or_else(|error| panic!("admin write node {index} ({admin}) {path}: {error}"));
        let mut raw = String::new();
        socket
            .read_to_string(&mut raw)
            .unwrap_or_else(|error| panic!("admin read node {index} ({admin}) {path}: {error}"));
        let status = raw
            .lines()
            .next()
            .unwrap_or_default()
            .split_whitespace()
            .nth(1)
            .unwrap_or("0")
            .parse::<u16>()
            .unwrap_or(0);
        let json: Value =
            serde_json::from_str(raw.split("\r\n\r\n").nth(1).unwrap_or("")).unwrap_or(Value::Null);
        (status, json)
    }

    /// Suspends the peer link from member `index` toward node `peer`
    /// (partition hook: bounded backoff, membership unchanged).
    ///
    /// # Panics
    ///
    /// Panics when the endpoint errors.
    pub fn suspend_peer(&self, index: usize, peer: u64) {
        let (status, _) = self.admin_post(
            index,
            "/v1/peers/suspend",
            &serde_json::json!({ "node": peer }),
        );
        assert_eq!(status, 200, "suspend peer {peer} on node {index}");
    }

    /// Resumes a suspended peer link.
    ///
    /// # Panics
    ///
    /// Panics when the endpoint errors.
    pub fn resume_peer(&self, index: usize, peer: u64) {
        let (status, _) = self.admin_post(
            index,
            "/v1/peers/resume",
            &serde_json::json!({ "node": peer }),
        );
        assert_eq!(status, 200, "resume peer {peer} on node {index}");
    }

    /// Isolates member `index` from every other member (bidirectional
    /// suspends), returning the peer ids isolated. Heal with
    /// [`Cluster::heal`].
    ///
    /// # Panics
    ///
    /// Panics when any suspend fails.
    pub fn partition(&self, index: usize) {
        let victim = self.nodes[index].node_id;
        for (other, node) in self.nodes.iter().enumerate() {
            if other == index {
                continue;
            }
            self.suspend_peer(index, node.node_id);
            if self.alive(other) {
                self.suspend_peer(other, victim);
            }
        }
    }

    /// Heals every link toward/away from member `index`.
    ///
    /// # Panics
    ///
    /// Panics when any resume fails.
    pub fn heal(&self, index: usize) {
        let victim = self.nodes[index].node_id;
        for (other, node) in self.nodes.iter().enumerate() {
            if other == index {
                continue;
            }
            self.resume_peer(index, node.node_id);
            if self.alive(other) {
                self.resume_peer(other, victim);
            }
        }
    }

    /// Triggers a checkpoint snapshot + log purge on member `index`,
    /// returning the snapshot base index.
    ///
    /// # Panics
    ///
    /// Panics when the endpoint errors.
    #[must_use]
    pub fn snapshot(&self, index: usize) -> u64 {
        let (status, json) = self.admin_post(index, "/v1/snapshot", &serde_json::json!({}));
        assert_eq!(status, 200, "snapshot on node {index}: {json}");
        json["snapshot"].as_u64().unwrap_or(0)
    }

    /// Parsed `/ready` for one member (readiness reflects replica health,
    /// transport, membership, and leader knowledge — never just the TCP
    /// listener).
    ///
    /// # Panics
    ///
    /// Panics when the endpoint errors.
    #[must_use]
    pub fn ready(&self, index: usize) -> Value {
        let (status, json) = self.admin_get(index, "/ready");
        assert_eq!(status, 200, "GET /ready on node {index}");
        json
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

    /// Waits for exactly one stable leader among live members except
    /// `exclude`, returning its index. Mirrors the in-process
    /// `wait_leader_except`: the isolated minority may still believe it
    /// leads (stale term), so fencing tests must elect the majority
    /// without it.
    ///
    /// # Panics
    ///
    /// Panics when no single stable majority leader emerges inside the window.
    #[must_use = "the majority leader index routes the next test step"]
    pub fn wait_leader_except(&self, exclude: usize) -> usize {
        let deadline = Instant::now() + LEADER_TIMEOUT;
        let mut stable = 0usize;
        let mut last = usize::MAX;
        loop {
            let leaders: Vec<usize> = (0..self.nodes.len())
                .filter(|i| *i != exclude && self.alive(*i) && self.role(*i) == "leader")
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
                "majority never elected without node {exclude}"
            );
            std::thread::sleep(Duration::from_millis(100));
        }
    }

    /// Native endpoint of one member (even when partitioned; direct-dial
    /// fencing probes use this to pin a client on the isolate).
    #[must_use]
    pub fn native_endpoint(&self, index: usize) -> String {
        self.nodes[index].native.clone()
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
            "--cluster-tablets".to_owned(),
            self.tablet_count.to_string(),
            "--cluster-workers".to_owned(),
            self.worker_count.to_string(),
            "--layout".to_owned(),
            self.layout.clone(),
            "--data-dir".to_owned(),
            data_dir.display().to_string(),
        ]
    }

    /// Current peer addresses with node ids.
    fn peer_addrs(&self) -> Vec<(u64, String)> {
        self.nodes
            .iter()
            .map(|node| (node.node_id, node.peer.clone()))
            .collect()
    }

    /// Current native addresses with node ids.
    fn native_addrs(&self) -> Vec<(u64, String)> {
        self.nodes
            .iter()
            .map(|node| (node.node_id, node.native.clone()))
            .collect()
    }

    /// Current admin addresses with node ids (reconciler forwarding
    /// targets; the registry records each node's own endpoint).
    fn admin_addrs(&self) -> Vec<(u64, String)> {
        self.nodes
            .iter()
            .map(|node| (node.node_id, node.admin.clone()))
            .collect()
    }

    fn spawn_once(
        binary: &PathBuf,
        tablet_count: usize,
        worker_count: usize,
        want_resp: bool,
        extra_env: &[(&str, &str)],
        ready_timeout: Duration,
        layout: &str,
    ) -> Result<Self, SpawnError> {
        let peers = [free_udp_addr(), free_udp_addr(), free_udp_addr()];
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
                "--cluster-tablets".to_owned(),
                tablet_count.to_string(),
                "--cluster-workers".to_owned(),
                worker_count.to_string(),
                "--layout".to_owned(),
                layout.to_owned(),
                "--data-dir".to_owned(),
                data_dir.path().display().to_string(),
                "--cluster-peers".to_owned(),
                flag_list(&[
                    (1u64, peers[0].clone()),
                    (2, peers[1].clone()),
                    (3, peers[2].clone()),
                ]),
                "--cluster-natives".to_owned(),
                flag_list(&[
                    (1u64, natives[0].clone()),
                    (2, natives[1].clone()),
                    (3, natives[2].clone()),
                ]),
                "--cluster-admins".to_owned(),
                flag_list(&[
                    (1u64, admins[0].clone()),
                    (2, admins[1].clone()),
                    (3, admins[2].clone()),
                ]),
                "--cluster-native".to_owned(),
                natives[index].clone(),
                "--admin".to_owned(),
                admins[index].clone(),
            ];
            if want_resp {
                args.push("--redis-listen".to_owned());
                args.push(redis[index].clone());
            }
            let (child, ready_native, ready_admin, ready_resp) =
                spawn_member(binary, &args, data_dir.path(), extra_env, ready_timeout)?;
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
        Ok(Self {
            nodes,
            id: cluster_id,
            binary: binary.clone(),
            tablet_count,
            worker_count,
            layout: layout.to_owned(),
            extra_env: extra_env
                .iter()
                .map(|(key, value)| ((*key).to_owned(), (*value).to_owned()))
                .collect(),
            ready_timeout,
        })
    }

    /// Configured tablet count.
    #[must_use]
    pub fn tablet_count(&self) -> usize {
        self.tablet_count
    }

    /// Configured worker count.
    #[must_use]
    pub fn worker_count(&self) -> usize {
        self.worker_count
    }

    /// Parsed `/v1/tablets` for one member (per-tablet diagnostics for
    /// every local group, sorted by tablet).
    ///
    /// # Panics
    ///
    /// Panics when the endpoint errors.
    #[must_use]
    pub fn tablets_status(&self, index: usize) -> Value {
        let (status, json) = self.admin_get(index, "/v1/tablets");
        assert_eq!(status, 200, "GET /v1/tablets on node {index}");
        json
    }

    /// Parsed `/v1/tablets` for one member, or `None` when the member is
    /// temporarily unable to serve it (e.g. the 5 s admin ceiling under a
    /// 1000-group debug fan-out). Polling loops use this and retry;
    /// correctness assertions keep using [`tablets_status`](Self::tablets_status).
    #[must_use]
    pub fn tablets_status_opt(&self, index: usize) -> Option<Value> {
        let (status, json) = self.admin_get(index, "/v1/tablets");
        (status == 200).then_some(json)
    }

    /// Tablet ids served by one member (from `/v1/tablets`).
    ///
    /// # Panics
    ///
    /// Panics when the endpoint errors or tablet ids are missing.
    #[must_use]
    pub fn tablet_ids(&self, index: usize) -> Vec<u64> {
        self.tablets_status(index)
            .as_array()
            .expect("tablets array")
            .iter()
            .map(|tablet| tablet["group"].as_u64().expect("tablet group"))
            .collect()
    }

    /// Per-tablet applied indexes of one member (`tablet → applied`).
    ///
    /// # Panics
    ///
    /// Panics when the endpoint errors or fields are missing.
    #[must_use]
    pub fn tablets_applied(&self, index: usize) -> std::collections::BTreeMap<u64, u64> {
        self.tablets_status(index)
            .as_array()
            .expect("tablets array")
            .iter()
            .map(|tablet| {
                (
                    tablet["group"].as_u64().expect("tablet group"),
                    tablet["applied"].as_u64().expect("tablet applied"),
                )
            })
            .collect()
    }

    /// Per-tablet leader node ids of one member (`tablet → leader`), as
    /// last observed by that member.
    ///
    /// # Panics
    ///
    /// Panics when the endpoint errors.
    #[must_use]
    pub fn tablets_leaders(&self, index: usize) -> std::collections::BTreeMap<u64, Option<u64>> {
        self.tablets_status(index)
            .as_array()
            .expect("tablets array")
            .iter()
            .map(|tablet| {
                (
                    tablet["group"].as_u64().expect("tablet group"),
                    tablet["leader"].as_u64(),
                )
            })
            .collect()
    }

    /// Waits until every tablet has exactly one stable leader across live
    /// members (stable across consecutive polls, so callers never catch a
    /// handover mid-flight), returning `(tablet, leader_node_index)` in
    /// tablet order.
    ///
    /// # Panics
    ///
    /// Panics when any tablet never reaches one stable leader inside the
    /// window.
    #[must_use = "the leader map routes the next test step"]
    pub fn wait_all_leaders(&self) -> Vec<(u64, usize)> {
        let deadline = Instant::now() + LEADER_TIMEOUT;
        // Tablet set first (members agree on the static set). A member
        // under load may 503 its fan-out; retry for the set like any
        // other poll round.
        let tablets = loop {
            if let Some(status) = self.tablets_status_opt(self.live_index()) {
                break status
                    .as_array()
                    .expect("tablets array")
                    .iter()
                    .map(|tablet| tablet["group"].as_u64().expect("tablet group"))
                    .collect::<Vec<_>>();
            }
            assert!(Instant::now() < deadline, "tablet set never readable");
            std::thread::sleep(Duration::from_millis(200));
        };
        let mut stable = 0usize;
        let mut last: Vec<(u64, usize)> = Vec::new();
        loop {
            let mut current: Vec<(u64, usize)> = Vec::new();
            let mut ok = true;
            for tablet in &tablets {
                // Slow members answer 503 under fan-out load: skip them
                // this round instead of failing the whole wait.
                let leaders: Vec<usize> = (0..self.nodes.len())
                    .filter(|i| {
                        self.alive(*i)
                            && self.tablets_status_opt(*i).is_some_and(|status| {
                                status.as_array().is_some_and(|statuses| {
                                    statuses.iter().any(|entry| {
                                        entry["group"].as_u64() == Some(*tablet)
                                            && entry["role"].as_str() == Some("leader")
                                    })
                                })
                            })
                    })
                    .collect();
                if leaders.len() != 1 {
                    ok = false;
                    break;
                }
                current.push((*tablet, leaders[0]));
            }
            if ok && current == last {
                stable += 1;
                if stable >= 3 {
                    return current;
                }
            } else {
                stable = 0;
                last = current;
            }
            assert!(
                Instant::now() < deadline,
                "tablets never reached one stable leader each"
            );
            std::thread::sleep(Duration::from_millis(200));
        }
    }

    /// Index of one live member (formation always leaves one).
    ///
    /// # Panics
    ///
    /// Panics when no member is alive.
    #[must_use]
    pub fn live_index(&self) -> usize {
        (0..self.nodes.len())
            .find(|i| self.alive(*i))
            .expect("a live member")
    }

    /// Index of the live member leading `tablet` right now (by its own
    /// authoritative role view). Call after convergence/election waits;
    /// leadership may move at any time.
    ///
    /// # Panics
    ///
    /// Panics when no live member currently leads the tablet.
    #[must_use]
    pub fn leader_of(&self, tablet: u64) -> usize {
        (0..self.nodes.len())
            .filter(|i| self.alive(*i))
            .find(|i| {
                self.tablets_status(*i).as_array().is_some_and(|statuses| {
                    statuses.iter().any(|entry| {
                        entry["group"].as_u64() == Some(tablet)
                            && entry["role"].as_str() == Some("leader")
                    })
                })
            })
            .expect("a live leader for the tablet")
    }

    /// Waits until every live member converges per tablet: for each
    /// tablet, all live members report the same applied index.
    ///
    /// # Panics
    ///
    /// Panics on timeout.
    pub fn wait_converged_all(&self) {
        self.wait_converged_all_timeout(CONVERGE_TIMEOUT);
    }

    /// Waits like [`wait_converged_all`](Self::wait_converged_all) with
    /// an explicit deadline (rolling restarts over many tablets need
    /// longer than the default window).
    ///
    /// # Panics
    ///
    /// Panics past the deadline.
    pub fn wait_converged_all_timeout(&self, timeout: Duration) {
        let deadline = Instant::now() + timeout;
        loop {
            let mut per_tablet: std::collections::BTreeMap<u64, u64> =
                std::collections::BTreeMap::new();
            let mut converged = true;
            let mut polled = 0usize;
            for i in 0..self.nodes.len() {
                if !self.alive(i) {
                    continue;
                }
                // Slow members answer 503 under fan-out load: skip them
                // this round (a skipped member simply is not converged
                // yet as far as this round can tell).
                let Some(status) = self.tablets_status_opt(i) else {
                    converged = false;
                    continue;
                };
                polled += 1;
                for entry in status.as_array().cloned().unwrap_or_default() {
                    let tablet = entry["group"].as_u64().expect("tablet group");
                    let applied = entry["applied"].as_u64().expect("tablet applied");
                    match per_tablet.get(&tablet) {
                        None => {
                            per_tablet.insert(tablet, applied);
                        }
                        Some(watermark) if *watermark != applied => {
                            converged = false;
                        }
                        Some(_) => {}
                    }
                }
            }
            if converged && polled > 0 && !per_tablet.is_empty() {
                break;
            }
            assert!(Instant::now() < deadline, "tablets never converged");
            std::thread::sleep(Duration::from_millis(200));
        }
    }

    /// Finds `per_tablet` keys landing on each tablet of the static
    /// directory (`tablet → keys`), by hashing candidates with the
    /// production partition hash and matching the ranges served on admin
    /// `/v1/tablets`. Key strings are deterministic (`mt-key-{i:06}`), so
    /// runs agree on the same mapping.
    ///
    /// # Panics
    ///
    /// Panics when the endpoint errors, ranges are missing, or some
    /// tablet never fills inside the candidate budget.
    #[must_use]
    pub fn keys_for_tablets(
        &self,
        per_tablet: usize,
    ) -> std::collections::BTreeMap<u64, Vec<String>> {
        use kivi_state::PartitionHasher;
        let namespace = NamespaceId::from_u64(1);
        let ranges: Vec<(u64, u128, u8)> = self
            .tablets_status(self.live_index())
            .as_array()
            .expect("tablets array")
            .iter()
            .map(|tablet| {
                let group = tablet["group"].as_u64().expect("tablet group");
                let bits = tablet["range_bits"].as_str().expect("tablet range bits");
                let bits = u128::from_str_radix(bits, 16).expect("hex range bits");
                let len = tablet["range_len"].as_u64().expect("tablet range len");
                let len = u8::try_from(len).expect("range len fits u8");
                (group, bits, len)
            })
            .collect();
        assert!(!ranges.is_empty(), "cluster serves tablets");
        let route = |hash: u128| -> Option<u64> {
            ranges.iter().find_map(|(group, bits, len)| {
                let shift = 128 - *len;
                (hash >> shift == *bits >> shift).then_some(*group)
            })
        };
        let mut out: std::collections::BTreeMap<u64, Vec<String>> =
            std::collections::BTreeMap::new();
        for group in ranges.iter().map(|(group, _, _)| *group) {
            out.insert(group, Vec::new());
        }
        for i in 0..100_000u32 {
            if out.values().all(|keys| keys.len() >= per_tablet) {
                break;
            }
            let candidate = format!("mt-key-{i:06}");
            let hash = PartitionHasher::V1
                .hash(namespace, candidate.as_bytes())
                .expect("supported")
                .as_u128();
            if let Some(group) = route(hash)
                && out[&group].len() < per_tablet
            {
                out.get_mut(&group).expect("tablet bucket").push(candidate);
            }
        }
        for (group, keys) in &out {
            assert_eq!(
                keys.len(),
                per_tablet,
                "tablet {group} fills {per_tablet} keys inside the budget"
            );
        }
        out
    }

    /// Triggers a checkpoint snapshot + log purge for `tablet` on member
    /// `index`, returning the snapshot base index.
    ///
    /// # Panics
    ///
    /// Panics when the endpoint errors.
    #[must_use]
    pub fn snapshot_tablet(&self, index: usize, tablet: u64) -> u64 {
        let (status, json) = self.admin_post(
            index,
            "/v1/snapshot",
            &serde_json::json!({ "tablet": tablet }),
        );
        assert_eq!(
            status, 200,
            "snapshot tablet {tablet} on node {index}: {json}"
        );
        json["snapshot"].as_u64().unwrap_or(0)
    }

    /// Spawns one more member WITHOUT restarting existing ones and
    /// registers it with the control plane (explicit admission). The new
    /// node starts data-empty; placement arrives through migration.
    /// Returns the member index. Existing members learn its mesh address
    /// from replicated registry commits (no restarts).
    ///
    /// # Panics
    ///
    /// Panics when the member fails to start or admission is refused.
    #[must_use]
    pub fn add_node(&mut self) -> usize {
        let node_id = self
            .nodes
            .iter()
            .map(|node| node.node_id)
            .max()
            .unwrap_or(0)
            + 1;
        let peer = free_udp_addr();
        let native = free_addr();
        let admin = free_addr();
        let data_dir = tempfile::tempdir()
            .unwrap_or_else(|error| panic!("tempdir for node {node_id}: {error}"));
        let mut args = self.base_args(node_id, data_dir.path());
        let mut peers = self.peer_addrs();
        peers.push((node_id, peer.clone()));
        let mut natives = self.native_addrs();
        natives.push((node_id, native.clone()));
        let mut admin_addrs = self.admin_addrs();
        admin_addrs.push((node_id, admin.clone()));
        args.push("--cluster-peers".to_owned());
        args.push(flag_list(&peers));
        args.push("--cluster-natives".to_owned());
        args.push(flag_list(&natives));
        args.push("--cluster-admins".to_owned());
        args.push(flag_list(&admin_addrs));
        // Explicit join intent: the existing peers prove the cluster
        // already exists, so the fresh directory waits for `add_learner`
        // instead of initializing a forked control group.
        let seeds: Vec<String> = self
            .peer_addrs()
            .into_iter()
            .map(|(_, addr)| addr)
            .collect();
        if !seeds.is_empty() {
            args.push("--control-seeds".to_owned());
            args.push(seeds.join(","));
        }
        args.push("--cluster-native".to_owned());
        args.push(native.clone());
        args.push("--admin".to_owned());
        args.push(admin.clone());
        let env: Vec<(&str, &str)> = self
            .extra_env
            .iter()
            .map(|(key, value)| (key.as_str(), value.as_str()))
            .collect();
        let ready_timeout = self.ready_timeout;
        let (child, ready_native, ready_admin, _) =
            spawn_member(&self.binary, &args, data_dir.path(), &env, ready_timeout)
                .unwrap_or_else(|error| panic!("spawn node {node_id}: {error:?}"));
        self.nodes.push(ClusterNode {
            child: Some(child),
            node_id,
            data_dir,
            native: ready_native,
            peer,
            admin: ready_admin,
            resp: None,
        });
        let index = self.nodes.len() - 1;
        // Explicit admission through the control leader (retried: the
        // new member's own control learner is still catching up).
        let body = serde_json::json!({
            "node": node_id,
            "peer": self.nodes[index].peer,
            "native": self.nodes[index].native,
            "admin": self.nodes[index].admin,
        });
        self.post_control(
            "/v1/control/nodes",
            &body,
            &format!("admit node {node_id}"),
            Duration::from_secs(120),
        );
        index
    }

    /// Member index for one node id (`None` after drops).
    #[must_use]
    pub fn index_of(&self, node_id: u64) -> Option<usize> {
        self.nodes.iter().position(|node| node.node_id == node_id)
    }

    /// Number of member slots (including dead ones not yet dropped).
    #[must_use]
    pub fn member_count(&self) -> usize {
        self.nodes.len()
    }

    /// Coherent serving-ready predicate for a restarted member: the new
    /// process incarnation must advance past `incarnation_before` (same
    /// `NodeId`, new process — stale process-local state keyed only by
    /// `NodeId` is fenced), every tablet must elect one stable leader,
    /// per-tablet applied indexes must converge across live members, every
    /// live member must agree on the exact active ordered tiling at
    /// `width` (stable across polls), and the rejoined member's serving
    /// directory must reach at least `dir_before` (replay never regresses).
    /// Process-listening / `KIVI_READY` alone proves none of this: the
    /// ready line prints before the control image loads, the peer mesh
    /// heals, leaders emerge, and the reconciler converges the directory.
    /// One predicate, not test-specific wait soup; polling intervals bound
    /// the polls, semantic state bounds the wait.
    ///
    /// # Panics
    ///
    /// Panics past `timeout` or when applied indexes diverge.
    pub fn wait_member_rejoined(
        &self,
        index: usize,
        incarnation_before: u64,
        dir_before: u64,
        width: usize,
        timeout: Duration,
    ) {
        let deadline = Instant::now() + timeout;
        // New process behind the same endpoint: incarnation fencing first.
        loop {
            if self.alive(index) && self.incarnation(index) > incarnation_before {
                break;
            }
            assert!(
                Instant::now() < deadline,
                "member {index} incarnation never advanced past {incarnation_before}"
            );
            std::thread::sleep(Duration::from_millis(250));
        }
        let _ = self.wait_all_leaders();
        self.wait_converged_all();
        self.wait_tiling_agreed(width, deadline);
        // Replay floor: the rejoined directory must not regress.
        loop {
            if self.alive(index) {
                let current = self
                    .directory(index)
                    .get("version")
                    .and_then(Value::as_u64)
                    .unwrap_or(0);
                if current >= dir_before {
                    break;
                }
            }
            assert!(
                Instant::now() < deadline,
                "member {index} directory never reached {dir_before}"
            );
            std::thread::sleep(Duration::from_millis(250));
        }
        // Applied convergence across live members (wait_converged_all
        // ensures it; re-assert here so the predicate owns the proof).
        let mut per_tablet: std::collections::BTreeMap<u64, std::collections::BTreeSet<u64>> =
            std::collections::BTreeMap::new();
        for member in 0..self.nodes.len() {
            if !self.alive(member) {
                continue;
            }
            for (tablet, applied) in self.tablets_applied(member) {
                per_tablet.entry(tablet).or_default().insert(applied);
            }
        }
        for (tablet, applied_set) in &per_tablet {
            assert_eq!(
                applied_set.len(),
                1,
                "tablet {tablet} applied diverged across live members: {applied_set:?}"
            );
        }
    }

    /// Waits until every live member reports the same exact ordered tiling
    /// at `width` — first range starts empty, last is unbounded, adjacent
    /// boundaries match — stable across consecutive polls. `deadline` is
    /// absolute (callers with a multi-stage budget pass theirs through).
    ///
    /// # Panics
    ///
    /// Panics past the deadline.
    pub fn wait_tiling_agreed(&self, width: usize, deadline: Instant) {
        let mut last: Tiling = Vec::new();
        let mut stable = 0u32;
        loop {
            let mut views: Vec<Tiling> = Vec::new();
            let mut agreed = true;
            for index in 0..self.nodes.len() {
                if !self.alive(index) {
                    continue;
                }
                let mut ranges = self.ordered_ranges(index);
                ranges.sort_by(|left, right| left.1.cmp(&right.1));
                let complete = ranges.len() == width
                    && ranges[0].1.is_empty()
                    && ranges.last().expect("tiling nonempty").2.is_none()
                    && ranges
                        .windows(2)
                        .all(|pair| pair[0].2.as_deref() == Some(pair[1].1.as_slice()));
                if !complete {
                    agreed = false;
                    break;
                }
                views.push(ranges);
            }
            if agreed && !views.is_empty() && views.iter().all(|view| *view == views[0]) {
                if views[0] == last {
                    stable += 1;
                    if stable >= 2 {
                        return;
                    }
                } else {
                    stable = 0;
                    last.clone_from(&views[0]);
                }
            } else {
                stable = 0;
            }
            assert!(
                Instant::now() < deadline,
                "ordered tiling never agreed at width {width}"
            );
            std::thread::sleep(Duration::from_millis(500));
        }
    }

    /// Fetches `/v1/control` from one member (registry, placements,
    /// plans, versions).
    ///
    /// # Panics
    ///
    /// Panics when the endpoint errors.
    #[must_use]
    pub fn control(&self, index: usize) -> Value {
        let (status, json) = self.admin_get(index, "/v1/control");
        assert_eq!(status, 200, "control on node {index}: {json}");
        json
    }

    /// Waits until the registry names `count` nodes (any state).
    ///
    /// # Panics
    ///
    /// Panics past the deadline.
    pub fn wait_control_nodes(&self, count: usize, timeout: Duration) {
        let deadline = Instant::now() + timeout;
        loop {
            let mut seen = 0;
            for index in 0..self.nodes.len() {
                if self.nodes[index].child.is_none() {
                    continue;
                }
                let (_, json) = self.admin_get(index, "/v1/control");
                if let Some(nodes) = json.get("nodes").and_then(Value::as_array) {
                    seen = seen.max(nodes.len());
                }
            }
            if seen >= count {
                return;
            }
            assert!(
                Instant::now() < deadline,
                "registry never reached {count} nodes"
            );
            std::thread::sleep(Duration::from_millis(250));
        }
    }

    /// Posts to the control leader (retried across live members until
    /// one commits): control mutations only commit on the leader, and
    /// leadership moves across kills and drains.
    ///
    /// # Panics
    ///
    /// Panics past the deadline.
    fn post_control(&self, path: &str, body: &Value, what: &str, timeout: Duration) -> Value {
        let deadline = Instant::now() + timeout;
        let mut last = String::new();
        loop {
            let mut order: Vec<usize> = (0..self.nodes.len())
                .filter(|index| self.nodes[*index].child.is_some())
                .collect();
            if let Some(leader) = self.control_leader_index()
                && let Some(position) = order.iter().position(|index| *index == leader)
            {
                order.remove(position);
                order.insert(0, leader);
            }
            for index in order {
                let (status, json) = self.admin_post(index, path, body);
                if status == 200 {
                    return json;
                }
                last = format!("member {index}: {status} {json}");
            }
            assert!(
                Instant::now() < deadline,
                "{what} never committed (last: {last})"
            );
            std::thread::sleep(Duration::from_millis(500));
        }
    }

    /// Triggers a bounded rebalance through the control leader.
    ///
    /// # Panics
    ///
    /// Panics when the endpoint refuses.
    #[must_use]
    pub fn rebalance(&self) -> Value {
        self.post_control(
            "/v1/control/rebalance",
            &serde_json::json!({}),
            "rebalance",
            Duration::from_secs(120),
        )
    }

    /// Moves one tablet replica through the persisted plan pathway.
    ///
    /// # Panics
    ///
    /// Panics when the endpoint refuses.
    #[must_use]
    pub fn move_tablet(&self, tablet: u64, from: u64, to: u64) -> Value {
        self.post_control(
            "/v1/control/migrations/create",
            &serde_json::json!({ "tablet": tablet, "from": from, "to": to }),
            &format!("move tablet {tablet} {from}->{to}"),
            Duration::from_secs(120),
        )
    }

    /// Fetches live migration plans from one member.
    ///
    /// # Panics
    ///
    /// Panics when the endpoint errors.
    #[must_use]
    pub fn migrations(&self, index: usize) -> Value {
        let (status, json) = self.admin_get(index, "/v1/control/migrations");
        assert_eq!(status, 200, "migrations on node {index}: {json}");
        json
    }

    /// Splits one tablet through the persisted reconciler pathway.
    ///
    /// # Panics
    ///
    /// Panics when the endpoint refuses.
    #[must_use]
    pub fn split_tablet(&self, tablet: u64) -> Value {
        self.post_control(
            "/v1/control/splits/create",
            &serde_json::json!({ "tablet": tablet }),
            &format!("split tablet {tablet}"),
            Duration::from_secs(120),
        )
    }

    /// Merges two adjacent tablets through the persisted pathway.
    ///
    /// # Panics
    ///
    /// Panics when the endpoint refuses.
    #[must_use]
    pub fn merge_tablets(&self, left: u64, right: u64) -> Value {
        self.post_control(
            "/v1/control/merges/create",
            &serde_json::json!({ "left": left, "right": right }),
            &format!("merge tablets {left}+{right}"),
            Duration::from_secs(120),
        )
    }

    /// Splits one ORDERED tablet at an explicit key through the persisted
    /// pathway (hash parents split at their midpoint without a key).
    ///
    /// # Panics
    ///
    /// Panics when the endpoint refuses.
    #[must_use]
    pub fn split_ordered_tablet(&self, tablet: u64, split_key: &[u8]) -> Value {
        self.post_control(
            "/v1/control/splits/create",
            &serde_json::json!({ "tablet": tablet, "split_key": split_key }),
            &format!("split ordered tablet {tablet}"),
            Duration::from_secs(120),
        )
    }

    /// Ordered tablet ranges of one member: `(tablet, start, end)` with
    /// hex-decoded bounds (`None` end = `+∞`). Only `Active` tablets are
    /// returned (tombstones overlap successors by design and never route).
    ///
    /// # Panics
    ///
    /// Panics when the endpoint errors.
    #[must_use]
    pub fn ordered_ranges(&self, index: usize) -> Vec<(u64, Vec<u8>, Option<Vec<u8>>)> {
        fn unhex(text: &str) -> Vec<u8> {
            (0..text.len())
                .step_by(2)
                .map(|i| u8::from_str_radix(&text[i..i + 2], 16).expect("hex range bound"))
                .collect()
        }
        self.tablets_status(index)
            .as_array()
            .expect("tablets array")
            .iter()
            .filter(|tablet| tablet["state"].as_str() == Some("Active"))
            .filter_map(|tablet| {
                let group = tablet["group"].as_u64().expect("tablet group");
                let start = tablet["range_start"].as_str().map(unhex)?;
                let end = tablet["range_end"].as_str().map(unhex);
                Some((group, start, end))
            })
            .collect()
    }

    /// Total prepared transaction intents across all live members (stuck
    /// intent detection: quiescent clusters must report zero).
    ///
    /// # Panics
    ///
    /// Panics when an endpoint errors.
    #[must_use]
    pub fn total_intents(&self) -> usize {
        let mut total = 0;
        for index in 0..self.nodes.len() {
            if self.nodes[index].child.is_none() {
                continue;
            }
            for tablet in self
                .tablets_status(index)
                .as_array()
                .expect("tablets array")
            {
                total += usize::try_from(tablet["intent_count"].as_u64().unwrap_or(0))
                    .unwrap_or(usize::MAX);
            }
        }
        total
    }

    /// Fetches split plans from one member.
    ///
    /// # Panics
    ///
    /// Panics when the endpoint errors.
    #[must_use]
    pub fn splits(&self, index: usize) -> Value {
        let (status, json) = self.admin_get(index, "/v1/control/splits");
        assert_eq!(status, 200, "splits on node {index}: {json}");
        json
    }

    /// Fetches merge plans from one member.
    ///
    /// # Panics
    ///
    /// Panics when the endpoint errors.
    #[must_use]
    pub fn merges(&self, index: usize) -> Value {
        let (status, json) = self.admin_get(index, "/v1/control/merges");
        assert_eq!(status, 200, "merges on node {index}: {json}");
        json
    }

    /// Waits until no live split plans remain and none failed.
    ///
    /// # Panics
    ///
    /// Panics past the deadline or when a plan fails.
    pub fn wait_splits_done(&self, timeout: Duration) {
        self.wait_plans_done("/v1/control/splits", "split", timeout);
    }

    /// Waits until no live merge plans remain and none failed.
    ///
    /// # Panics
    ///
    /// Panics past the deadline or when a plan fails.
    pub fn wait_merges_done(&self, timeout: Duration) {
        self.wait_plans_done("/v1/control/merges", "merge", timeout);
    }

    /// Shared waiter for split/merge plan lists (`key` selects the array:
    /// `"splits"` or `"merges"`).
    ///
    /// # Panics
    ///
    /// Panics past the deadline or when a plan fails.
    fn wait_plans_done(&self, path: &str, what: &str, timeout: Duration) {
        let key = path.rsplit('/').next().unwrap_or(what);
        let deadline = Instant::now() + timeout;
        loop {
            // Quiescence requires EVERY live member at zero: control
            // commits replicate asynchronously, so one caught-up member
            // reporting zero must not mask a lagging member still
            // retiring (that early return lets the next split/merge fire
            // while the previous plan is still live elsewhere, breaking
            // the one-live-topology-plan invariant and width counts).
            let mut live = 0usize;
            let mut failed = 0;
            let mut probed = false;
            let mut detail = String::new();
            for index in 0..self.nodes.len() {
                if self.nodes[index].child.is_none() {
                    continue;
                }
                let (_, json) = self.admin_get(index, path);
                let Some(plans) = json.get(key).and_then(Value::as_array) else {
                    continue;
                };
                probed = true;
                detail = format!("{json}");
                let node_live = plans
                    .iter()
                    .filter(|plan| {
                        !matches!(
                            plan.get("phase").and_then(Value::as_str),
                            Some("Completed" | "Failed")
                        )
                    })
                    .count();
                failed += plans
                    .iter()
                    .filter(|plan| {
                        matches!(plan.get("phase").and_then(Value::as_str), Some("Failed"))
                    })
                    .count();
                live = live.max(node_live);
            }
            assert!(probed, "no live member serves {what} plans");
            assert_eq!(failed, 0, "{what} plans failed");
            if live == 0 {
                return;
            }
            eprintln!("wait {what}: {detail}");
            assert!(Instant::now() < deadline, "{what} plans never completed");
            std::thread::sleep(Duration::from_millis(500));
        }
    }

    /// Fetches repair status from one member.
    ///
    /// # Panics
    ///
    /// Panics when the endpoint errors.
    #[must_use]
    pub fn repair_status(&self, index: usize) -> Value {
        let (status, json) = self.admin_get(index, "/v1/control/repair");
        assert_eq!(status, 200, "repair on node {index}: {json}");
        json
    }

    /// Fetches topology overview from one member.
    ///
    /// # Panics
    ///
    /// Panics when the endpoint errors.
    #[must_use]
    pub fn topology(&self, index: usize) -> Value {
        let (status, json) = self.admin_get(index, "/v1/control/topology");
        assert_eq!(status, 200, "topology on node {index}: {json}");
        json
    }

    /// Fetches the live reconciler policy from one member.
    ///
    /// # Panics
    ///
    /// Panics when the endpoint errors.
    #[must_use]
    pub fn policy(&self, index: usize) -> Value {
        let (status, json) = self.admin_get(index, "/v1/control/policy");
        assert_eq!(status, 200, "policy on node {index}: {json}");
        json
    }

    /// Retunes the live reconciler policy on every live member (e.g.,
    /// enable auto-split). Broadcast (not leader-only): the policy is a
    /// local live control, and leadership may move — every member holds
    /// the same tuning so whoever leads acts on it.
    ///
    /// # Panics
    ///
    /// Panics when no live member accepts the update.
    #[must_use]
    pub fn set_policy(&self, body: &Value) -> Value {
        let mut last = serde_json::json!({});
        let mut accepted = false;
        for index in 0..self.nodes.len() {
            if self.nodes[index].child.is_none() {
                continue;
            }
            let (status, json) = self.admin_post(index, "/v1/control/policy/update", body);
            if status == 200 {
                accepted = true;
                last = json;
            }
        }
        assert!(accepted, "no live member accepted the policy update");
        last
    }

    /// Waits until the directory version on every live member reaches at
    /// least `version` (split/merge cutover convergence).
    ///
    /// # Panics
    ///
    /// Panics past the deadline.
    pub fn wait_dir_version(&self, version: u64, timeout: Duration) {
        let deadline = Instant::now() + timeout;
        loop {
            let mut ready = true;
            for index in 0..self.nodes.len() {
                if self.nodes[index].child.is_none() {
                    continue;
                }
                let info = self.node_info(index);
                let current = info.get("dir_version").and_then(Value::as_u64).unwrap_or(0);
                if current < version {
                    ready = false;
                    break;
                }
            }
            if ready {
                return;
            }
            assert!(
                Instant::now() < deadline,
                "directory never reached version {version}"
            );
            std::thread::sleep(Duration::from_millis(250));
        }
    }

    /// Parsed `/v1/directory` for one member: serving directory version,
    /// namespace, coverage flag, and every tablet with state/range/successors.
    /// Semantic convergence predicate: equal versions + equal active tilings
    /// across members means routing agreement (see `wait_dir_version` for
    /// the version floor and the acceptance `wait_tiling_agreed` for exact
    /// tiling equality).
    ///
    /// # Panics
    ///
    /// Panics when the endpoint errors.
    #[must_use]
    pub fn directory(&self, index: usize) -> Value {
        let (status, json) = self.admin_get(index, "/v1/directory");
        assert_eq!(status, 200, "GET /v1/directory on node {index}");
        json
    }

    /// Waits until no live (non-terminal) migration plans remain on ANY
    /// live member, and none failed. Like
    /// [`wait_plans_done`](Self::wait_splits_done), quiescence is
    /// all-members: a single caught-up member at zero never masks a
    /// lagging one still converging.
    ///
    /// # Panics
    ///
    /// Panics past the deadline or when a plan fails.
    pub fn wait_migrations_done(&self, timeout: Duration) {
        let deadline = Instant::now() + timeout;
        let mut ticks = 0u32;
        loop {
            let mut live = 0usize;
            let mut failed = 0;
            let mut probed = false;
            let mut detail = String::new();
            for index in 0..self.nodes.len() {
                if self.nodes[index].child.is_none() {
                    continue;
                }
                let (_, json) = self.admin_get(index, "/v1/control/migrations");
                let Some(plans) = json.get("migrations").and_then(Value::as_array) else {
                    continue;
                };
                probed = true;
                detail = format!("{json}");
                let node_live = plans
                    .iter()
                    .filter(|plan| {
                        !matches!(
                            plan.get("phase").and_then(Value::as_str),
                            Some("Completed" | "Failed")
                        )
                    })
                    .count();
                failed += plans
                    .iter()
                    .filter(|plan| {
                        matches!(plan.get("phase").and_then(Value::as_str), Some("Failed"))
                    })
                    .count();
                live = live.max(node_live);
            }
            assert!(probed, "no live member serves migrations");
            assert_eq!(failed, 0, "migration plans failed");
            if live == 0 {
                return;
            }
            ticks += 1;
            if ticks % 20 == 1 {
                eprintln!("wait migrations: {detail}");
            }
            assert!(Instant::now() < deadline, "migrations never completed");
            std::thread::sleep(Duration::from_millis(500));
        }
    }

    /// Starts draining one node (replicas migrate away, then `Drained`).
    ///
    /// # Panics
    ///
    /// Panics when the endpoint refuses.
    #[must_use]
    pub fn drain_node(&self, id: u64) -> Value {
        self.post_control(
            &format!("/v1/control/nodes/{id}/drain"),
            &serde_json::json!({}),
            &format!("drain node {id}"),
            Duration::from_secs(180),
        )
    }

    /// Waits until the registry reports `node_id` in `state`
    /// (`"Active"`, `"Draining"`, `"Drained"`, ...), with progress
    /// heartbeats (tablets still desiring the node, live plans).
    ///
    /// # Panics
    ///
    /// Panics past the deadline.
    pub fn wait_node_state(&self, node_id: u64, state: &str, timeout: Duration) {
        let deadline = Instant::now() + timeout;
        let mut ticks = 0u32;
        // Deep membership dumps are expensive (one RPC per stuck plan);
        // heartbeat every poll, deep dump roughly every minute.
        let mut last_deep: Option<Instant> = None;
        loop {
            let alive: Vec<u64> = (0..self.nodes.len())
                .filter(|index| self.nodes[*index].child.is_some())
                .map(|index| self.nodes[index].node_id)
                .collect();
            for index in 0..self.nodes.len() {
                if self.nodes[index].child.is_none() {
                    continue;
                }
                let (_, json) = self.admin_get(index, "/v1/control");
                let reached = json
                    .get("nodes")
                    .and_then(Value::as_array)
                    .is_some_and(|nodes| {
                        nodes.iter().any(|node| {
                            node.get("node").and_then(Value::as_u64) == Some(node_id)
                                && node.get("state").and_then(Value::as_str) == Some(state)
                        })
                    });
                if reached {
                    return;
                }
                ticks += 1;
                if ticks % 20 == 1 {
                    Self::drain_heartbeat(self, node_id, state, &alive, &json);
                }
            }
            if last_deep.is_none_or(|dumped| dumped.elapsed() > Duration::from_secs(60)) {
                last_deep = Some(Instant::now());
                self.dump_stuck_plans(node_id);
            }
            assert!(
                Instant::now() < deadline,
                "node {node_id} never reached {state}"
            );
            std::thread::sleep(Duration::from_millis(500));
        }
    }

    /// Drain progress heartbeat: tablets still desiring the node, live
    /// plans by phase, stuck plan identities, liveness, and the control
    /// leader (which member's reconciler is driving).
    fn drain_heartbeat(&self, node_id: u64, state: &str, alive: &[u64], json: &Value) {
        let placements =
            json.get("placements")
                .and_then(Value::as_array)
                .map_or(usize::MAX, |placements| {
                    placements
                        .iter()
                        .filter(|placement| {
                            placement
                                .get("replicas")
                                .and_then(Value::as_array)
                                .is_some_and(|replicas| {
                                    replicas
                                        .iter()
                                        .any(|replica| replica.as_u64() == Some(node_id))
                                })
                        })
                        .count()
                });
        let phases = json
            .get("migrations")
            .and_then(Value::as_array)
            .map(|plans| {
                let mut histogram: std::collections::BTreeMap<String, usize> =
                    std::collections::BTreeMap::new();
                for plan in plans {
                    let phase = plan
                        .get("phase")
                        .and_then(Value::as_str)
                        .unwrap_or("?")
                        .to_owned();
                    if phase != "Completed" && phase != "Failed" {
                        *histogram.entry(phase).or_default() += 1;
                    }
                }
                histogram
            })
            .unwrap_or_default();
        let live: usize = phases.values().sum();
        let stuck: Vec<String> = json
            .get("migrations")
            .and_then(Value::as_array)
            .map(|plans| {
                plans
                    .iter()
                    .filter(|plan| {
                        !matches!(
                            plan.get("phase").and_then(Value::as_str),
                            Some("Completed" | "Failed")
                        )
                    })
                    .map(|plan| {
                        format!(
                            "{}:{}:{}->{}",
                            plan.get("id")
                                .map_or_else(|| String::from("?"), ToString::to_string),
                            plan.get("tablet")
                                .map_or_else(|| String::from("?"), ToString::to_string),
                            plan.get("from")
                                .map_or_else(|| String::from("?"), ToString::to_string),
                            plan.get("to")
                                .map_or_else(|| String::from("?"), ToString::to_string),
                        )
                    })
                    .collect()
            })
            .unwrap_or_default();
        let leader = self.control_leader_index().map_or_else(
            || "none".to_owned(),
            |leader| self.nodes[leader].node_id.to_string(),
        );
        let placement_version = json
            .get("placement_version")
            .map_or_else(String::new, ToString::to_string);
        let generation = json
            .get("generation")
            .map_or_else(String::new, ToString::to_string);
        eprintln!(
            "wait node {node_id} -> {state}: alive={alive:?} control_leader={leader} placement_v={placement_version} gen={generation} {placements} desire it, {live} live plans {phases:?} stuck={stuck:?}"
        );
    }

    /// Deep drain-debug dump: for the first few non-terminal plans
    /// touching `node_id`, log desired vs actual membership plus the
    /// observed leader, so a stall names its exact gate (catch-up lag,
    /// joint membership, leadership transfer, or retirement). Bounded
    /// and best-effort: never panics, never blocks the waiter.
    fn dump_stuck_plans(&self, node_id: u64) {
        let Some(index) = (0..self.nodes.len()).find(|index| self.nodes[*index].child.is_some())
        else {
            return;
        };
        let (_, control) = self.admin_get(index, "/v1/control");
        let plans: Vec<(u64, u64, String, u64, u64)> = control
            .get("migrations")
            .and_then(Value::as_array)
            .map(|plans| {
                plans
                    .iter()
                    .filter(|plan| {
                        !matches!(
                            plan.get("phase").and_then(Value::as_str),
                            Some("Completed" | "Failed")
                        ) && (plan.get("from").and_then(Value::as_u64) == Some(node_id)
                            || plan.get("to").and_then(Value::as_u64) == Some(node_id))
                    })
                    .take(4)
                    .map(|plan| {
                        (
                            plan.get("id").and_then(Value::as_u64).unwrap_or(0),
                            plan.get("tablet").and_then(Value::as_u64).unwrap_or(0),
                            plan.get("phase")
                                .and_then(Value::as_str)
                                .unwrap_or("?")
                                .to_owned(),
                            plan.get("from").and_then(Value::as_u64).unwrap_or(0),
                            plan.get("to").and_then(Value::as_u64).unwrap_or(0),
                        )
                    })
                    .collect()
            })
            .unwrap_or_default();
        for (id, tablet, phase, from, to) in plans {
            let (status, membership) =
                self.admin_get(index, &format!("/v1/tablets/{tablet}/membership"));
            eprintln!(
                "DRAIN-DEBUG plan={id} tablet={tablet} phase={phase} {from}->{to} membership_status={status} membership={membership}"
            );
        }
    }

    /// Removes a safely drained node (fails loudly otherwise).
    ///
    /// # Panics
    ///
    /// Panics when the endpoint refuses.
    #[must_use]
    pub fn remove_node(&self, id: u64) -> Value {
        self.post_control(
            &format!("/v1/control/nodes/{id}/remove"),
            &serde_json::json!({}),
            &format!("remove node {id}"),
            Duration::from_secs(180),
        )
    }

    /// Fetches one tablet's observed membership through one member.
    ///
    /// # Panics
    ///
    /// Panics when the member hosts no such group.
    #[must_use]
    pub fn tablet_membership(&self, index: usize, tablet: u64) -> Value {
        let (status, json) = self.admin_get(index, &format!("/v1/tablets/{tablet}/membership"));
        assert_eq!(
            status, 200,
            "membership tablet {tablet} on node {index}: {json}"
        );
        json
    }

    /// Waits until some live member observes exactly `voters` for
    /// `tablet` (sorted comparison).
    ///
    /// # Panics
    ///
    /// Panics past the deadline.
    pub fn wait_tablet_voters(&self, tablet: u64, voters: &[u64], timeout: Duration) {
        let mut want = voters.to_vec();
        want.sort_unstable();
        let deadline = Instant::now() + timeout;
        let mut ticks = 0u32;
        loop {
            for index in 0..self.nodes.len() {
                if self.nodes[index].child.is_none() {
                    continue;
                }
                let (_, json) = self.admin_get(index, &format!("/v1/tablets/{tablet}/membership"));
                let mut seen: Vec<u64> = json
                    .get("voters")
                    .and_then(Value::as_array)
                    .map(|ids| ids.iter().filter_map(Value::as_u64).collect())
                    .unwrap_or_default();
                seen.sort_unstable();
                if seen == want {
                    return;
                }
            }
            ticks += 1;
            if ticks % 40 == 1 {
                // Heartbeat for CI forensics: plan phases plus every
                // live member's observed voters/learners/leader.
                let mut plans = String::new();
                for index in 0..self.nodes.len() {
                    if self.nodes[index].child.is_none() {
                        continue;
                    }
                    let (_, control) = self.admin_get(index, "/v1/control/migrations");
                    plans = format!("{control}");
                    break;
                }
                let mut views = Vec::new();
                for index in 0..self.nodes.len() {
                    if self.nodes[index].child.is_none() {
                        continue;
                    }
                    let (status, json) =
                        self.admin_get(index, &format!("/v1/tablets/{tablet}/membership"));
                    views.push(format!("m{index}:{status}:{json}"));
                }
                eprintln!(
                    "wait voters {want:?} for tablet {tablet}: plans={plans} views=[{}]",
                    views.join(" ")
                );
            }
            assert!(
                Instant::now() < deadline,
                "tablet {tablet} voters never became {want:?}"
            );
            std::thread::sleep(Duration::from_millis(250));
        }
    }
    /// Finds the control leader's member index through per-member
    /// control-membership observations (`None` while campaigning).
    #[must_use]
    pub fn control_leader_index(&self) -> Option<usize> {
        const CONTROL_TABLET: u64 = u64::MAX;
        for index in 0..self.nodes.len() {
            if self.nodes[index].child.is_none() {
                continue;
            }
            let (status, json) =
                self.admin_get(index, &format!("/v1/tablets/{CONTROL_TABLET}/membership"));
            if status != 200 {
                continue;
            }
            let leader = json.get("leader").and_then(Value::as_u64);
            if leader == Some(self.nodes[index].node_id) {
                return Some(index);
            }
        }
        None
    }

    /// Drops one member slot entirely (kill + forget): later
    /// `restart_all` skips it, modeling a removed node that never comes
    /// back. Mesh flags for restarts rebuild from the remaining slots.
    pub fn drop_node(&mut self, index: usize) {
        if index < self.nodes.len() {
            if let Some(mut child) = self.nodes[index].child.take() {
                let _ = child.kill();
                let _ = child.wait();
            }
            if std::env::var("KIVI_LAB_KEEP_DIRS").is_ok() {
                self.nodes[index].data_dir.disable_cleanup(true);
                eprintln!(
                    "kivi-lab: kept dropped data dir {}",
                    self.nodes[index].data_dir.path().display()
                );
            }
            self.nodes.remove(index);
        }
    }

    /// Replaces the control voter set through joint consensus on the
    /// control leader (control-plane membership itself, §31), waiting
    /// for the uniform successor — not just the joint commit. New
    /// voters must already be control learners (the reconciler admits
    /// `Active` data nodes automatically).
    ///
    /// # Panics
    ///
    /// Panics when the uniform set never lands.
    pub fn set_control_voters(&self, voters: &[u64]) {
        const CONTROL_TABLET: u64 = u64::MAX;
        let mut want: Vec<u64> = voters.to_vec();
        want.sort_unstable();
        let body = serde_json::json!({ "voters": voters, "retain": false });
        let deadline = Instant::now() + Duration::from_secs(180);
        loop {
            // Re-propose while joint: the committed joint config needs
            // its uniform successor re-proposed explicitly.
            if let Some(leader) = self.control_leader_index() {
                let _ = self.admin_post(
                    leader,
                    &format!("/v1/tablets/{CONTROL_TABLET}/members"),
                    &body,
                );
            }
            for index in 0..self.nodes.len() {
                if self.nodes[index].child.is_none() {
                    continue;
                }
                let (status, json) =
                    self.admin_get(index, &format!("/v1/tablets/{CONTROL_TABLET}/membership"));
                if status != 200 {
                    continue;
                }
                let joint = json.get("joint").and_then(Value::as_bool).unwrap_or(true);
                let mut seen: Vec<u64> = json
                    .get("voters")
                    .and_then(Value::as_array)
                    .map(|ids| ids.iter().filter_map(Value::as_u64).collect())
                    .unwrap_or_default();
                seen.sort_unstable();
                if !joint && seen == want {
                    return;
                }
            }
            assert!(
                Instant::now() < deadline,
                "control membership never became uniform {want:?}"
            );
            std::thread::sleep(Duration::from_millis(500));
        }
    }

    /// Desired replica counts per node from the registry placements.
    ///
    /// # Panics
    ///
    /// Panics when no live member serves control state.
    #[must_use]
    pub fn replica_counts(&self) -> std::collections::HashMap<u64, usize> {
        for index in 0..self.nodes.len() {
            if self.nodes[index].child.is_none() {
                continue;
            }
            let (_, json) = self.admin_get(index, "/v1/control");
            let Some(placements) = json.get("placements").and_then(Value::as_array) else {
                continue;
            };
            let mut counts = std::collections::HashMap::new();
            for placement in placements {
                let replicas: Vec<u64> = placement
                    .get("replicas")
                    .and_then(Value::as_array)
                    .map(|ids| ids.iter().filter_map(Value::as_u64).collect())
                    .unwrap_or_default();
                for replica in replicas {
                    *counts.entry(replica).or_insert(0) += 1;
                }
            }
            return counts;
        }
        panic!("no live member serves control state");
    }
}

/// Formats `(id, address)` pairs as a `1=a,2=b,3=c` flag value.
/// Node ids ride explicitly (never positionally) so removed nodes leave
/// no ambiguity for later restarts and joins.
fn flag_list(addrs: &[(u64, String)]) -> String {
    let mut sorted = addrs.to_vec();
    sorted.sort_by_key(|(id, _)| *id);
    sorted
        .iter()
        .map(|(id, addr)| format!("{id}={addr}"))
        .collect::<Vec<_>>()
        .join(",")
}

/// Spawns one member and waits for its `KIVI_READY` (native + admin +
/// optional RESP). Test-only insecure peer TLS: fresh data directories
/// and ephemeral ports per run make static certificate pins impractical,
/// and all traffic stays on loopback.
///
/// Server output goes to per-member log files in the data directory
/// (appended across restarts): pipes would need continuous draining and
/// truncation would lose the failure story, while files keep every
/// member's full story beside its data for post-mortems.
///
/// Tracing itself is bounded inside the server (`KIVI_LOG_DIR` with
/// size rotation): the stderr redirect here only carries `KIVI_READY`
/// plus panics, so kept directories (`KIVI_LAB_KEEP_DIRS=1`, including
/// the `disable_cleanup` path on `drop_node`) stay bounded during long
/// stress tests instead of growing forever.
fn spawn_member(
    binary: &PathBuf,
    args: &[String],
    data_dir: &std::path::Path,
    extra_env: &[(&str, &str)],
    ready_timeout: Duration,
) -> Result<(Child, String, String, Option<String>), SpawnError> {
    use std::fs::OpenOptions;
    let stdout_log = data_dir.join("server-stdout.log");
    let stderr_log = data_dir.join("server-stderr.log");
    let stdout_file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&stdout_log)
        .map_err(|error| SpawnError::Io(error.to_string()))?;
    let stderr_file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&stderr_log)
        .map_err(|error| SpawnError::Io(error.to_string()))?;
    let mut command = Command::new(binary);
    command
        .args(args)
        .env("KIVI_INSECURE_PEER_TLS", "1")
        .stdin(Stdio::null())
        .stdout(Stdio::from(stdout_file))
        .stderr(Stdio::from(stderr_file));
    // Bounded server tracing per member (size-rotated, retained beside
    // the data dir for post-mortems). Explicit `extra_env` wins so tests
    // can tighten limits (e.g., the rotation test) or raise verbosity
    // (`KIVI_LOG_LEVEL`) without changing the harness.
    let has_log_dir = extra_env.iter().any(|(key, _)| *key == "KIVI_LOG_DIR");
    if !has_log_dir {
        command.env("KIVI_LOG_DIR", data_dir.join("logs"));
    }
    let has_log_files = extra_env.iter().any(|(key, _)| *key == "KIVI_LOG_FILES");
    if !has_log_files {
        command.env("KIVI_LOG_FILES", "8");
    }
    let has_log_bytes = extra_env
        .iter()
        .any(|(key, _)| *key == "KIVI_LOG_MAX_BYTES");
    if !has_log_bytes {
        command.env("KIVI_LOG_MAX_BYTES", "8388608");
    }
    for (key, value) in extra_env {
        command.env(key, value);
    }
    let mut child = command
        .spawn()
        .map_err(|error| SpawnError::Io(error.to_string()))?;
    match wait_ready_file(&mut child, &stdout_log, ready_timeout) {
        WaitOutcome::Ready(native, admin, resp) => Ok((
            child,
            native.into_iter().next().unwrap_or_default(),
            admin,
            resp,
        )),
        WaitOutcome::Exited(status) => Err(SpawnError::Exited(
            status.to_string(),
            file_tail(&stderr_log),
        )),
        WaitOutcome::TimedOut => Err(SpawnError::TimedOut(ready_timeout, file_tail(&stderr_log))),
    }
}

/// Last lines of a log file for failure diagnostics (bounded tail, never
/// the whole file).
fn file_tail(path: &std::path::Path) -> String {
    let text = std::fs::read_to_string(path).unwrap_or_default();
    text.lines()
        .rev()
        .take(8)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect::<Vec<_>>()
        .join("\n")
}

/// Internal ready-wait outcome over a tailed log file (child ownership
/// stays with the caller).
enum WaitOutcome {
    /// `KIVI_READY` parsed: native endpoints, admin endpoint, optional RESP.
    Ready(Vec<String>, String, Option<String>),
    /// The process exited before reporting readiness.
    Exited(std::process::ExitStatus),
    /// Neither readiness nor exit inside the window (child already killed).
    TimedOut,
}

/// Waits for `KIVI_READY` by tailing the member's stdout log file: reads
/// only appended bytes per quantum (the clock owns the deadline, file
/// growth never blocks the harness). A silent stall can never hang past
/// the deadline; early exit reports immediately.
fn wait_ready_file(
    child: &mut Child,
    stdout_log: &std::path::Path,
    timeout: Duration,
) -> WaitOutcome {
    use std::io::{Read as _, Seek as _};
    let mut offset = 0u64;
    let mut pending = String::new();
    let deadline = Instant::now() + timeout;
    loop {
        if let Ok(Some(status)) = child.try_wait() {
            return WaitOutcome::Exited(status);
        }
        if let Ok(mut file) = std::fs::File::open(stdout_log)
            && file.seek(std::io::SeekFrom::Start(offset)).is_ok()
        {
            let mut fresh = String::new();
            if file.read_to_string(&mut fresh).is_ok() {
                pending.push_str(&fresh);
                // Scan complete lines for readiness (tracing noise is
                // skipped); the trailing partial line stays buffered.
                let mut consumed = 0usize;
                let mut found = None;
                while let Some(end) = pending[consumed..].find('\n') {
                    let line = &pending[consumed..consumed + end];
                    if found.is_none() {
                        found = super::process::parse_ready(line);
                    }
                    consumed += end + 1;
                }
                pending.drain(..consumed);
                offset += consumed as u64;
                if let Some(ports) = found {
                    return WaitOutcome::Ready(ports.0, ports.1, ports.2);
                }
            }
        }
        if Instant::now() > deadline {
            let _ = child.kill();
            return WaitOutcome::TimedOut;
        }
        std::thread::sleep(Duration::from_millis(25));
    }
}
