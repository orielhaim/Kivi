#!/usr/bin/env python3
"""The external latency laboratory's harness.

# Why this is Python and not a shell script

An earlier version was bash, and every `awk` program in it had to survive two
layers of shell quoting to reach `awk`. None of them did: the program was silently
dropped, the command returned its input unchanged, and a `$(cat /proc/.../schedstat)`
arithmetic step failed with `operand expected (error token is "-  ")`. The failure
mode was silent, which is the worst kind - the script kept running and printed a
number. Nothing that measures the world should have a quoting layer between it and
the numbers.

`/proc/<pid>/task/*/schedstat` also has a trap worth naming: summing only the
process's own task directory shows the idle main thread, because a compio runtime
runs its work on threads the process did not create as tasks. The previous campaign
discovered this. `thread_group()` here globs every task and sums them, and reports
the thread count so a run that measured one thread out of nine is visible.

# What it does

* **Preflight.** Kills stray Kivi, Redis and control processes; waits for the load
  average to fall. A run that started on top of a leftover benchmark is not a slow
  run, it is a different measurement.
* **Runs the matrix** of control servers plus real Kivi and real Redis, from one
  client, at the requested pipeline depths and client counts.
* **Accounts for the gap** with per-thread scheduler time, and optionally with
  `strace -c` syscall counts.
* **Refuses to report** a run whose replies failed verification, because a
  benchmark against a server answering errors is not a measurement.

Every figure is the minimum over repeats. The mean of a distribution with a heavy
right tail reports the tail; the minimum reports the floor, and the floor is the
number that says what the system can do.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

# `USER_HZ` for `/proc/<pid>/stat`'s utime/stime, which are in clock ticks. The
# kernel's default is 100 and this host matches; read it rather than assume, because
# every CPU number in the report is a division by it.
try:
    USER_HZ = os.sysconf("SC_CLK_TCK")
except (AttributeError, ValueError, OSError):
    USER_HZ = 100

REPO = Path("/mnt/d/Projects/Kivi")
LAB = REPO / "target/release"
WORK = Path("/home/oriel/wl")

# Ports are fixed and far apart so a stale process from a previous run cannot be
# mistaken for this one's. The preflight kills by name and by port owner.
REDIS_PORT = 7311
KIVI_RESP_PORT = 7312
KIVI_NATIVE_PORT = 7313
REDIS_BIN = Path("/home/oriel/bench/install/8.10.1/bin/redis-server")
REDIS_VERSION = "8.10.1"

# Names the preflight kills. The patterns are deliberately narrow: a `pkill -f`
# pattern that also matches the harness's own command line kills the harness, which
# looks exactly like a hang.
STRAY_PATTERNS = [
    r"kivi-server --ephemeral",
    r"kivi-wirelab-server",
    r"kivi-server --",
    r"redis-server 127\.0\.0\.1:73",
    r"redis-server \*:73",
]


# --------------------------------------------------------------------- processes


def pgrep_all(pattern: str) -> list[int]:
    """Every pid whose command line matches `pattern`, excluding this process."""
    try:
        out = subprocess.run(
            ["pgrep", "-f", pattern], capture_output=True, text=True, check=False
        ).stdout
    except FileNotFoundError:
        return []
    mine = {os.getpid(), os.getppid()}
    found = []
    for line in out.split():
        try:
            pid = int(line)
        except ValueError:
            continue
        if pid not in mine:
            found.append(pid)
    return found


def kill_tree(pid: int) -> None:
    try:
        os.kill(pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


def preflight(verbose: bool = True) -> list[str]:
    """Kill strays and wait for the machine to be quiet.

    Returns the patterns that had to be killed, so a report can say the run started
    from a clean machine rather than implying it.
    """
    killed: list[str] = []
    for pattern in STRAY_PATTERNS:
        pids = pgrep_all(pattern)
        if pids:
            killed.append(f"{pattern} ({len(pids)})")
            for pid in pids:
                kill_tree(pid)
    if killed and verbose:
        print(f"preflight: killed {', '.join(killed)}")
    # Wait for the load average to fall. One second of load average 1.0 is one
    # second of work still queued; starting a measurement into it is how a 12 us
    # result becomes a 24 us result and gets believed.
    for _ in range(60):
        load = os.getloadavg()[0]
        if load < 0.15:
            break
        time.sleep(1.0)
    load = os.getloadavg()[0]
    if verbose:
        print(f"preflight: load average {load:.2f} ({'clean' if load < 0.15 else 'BUSY'})")
    return killed


# -------------------------------------------------------------------- schedstat


def thread_group(pid: int) -> dict[str, int]:
    """CPU accounting for a whole process, from two independent sources.

    Three fields per task come from `/proc/<pid>/task/*/schedstat`: nanoseconds on
    CPU, nanoseconds waiting to run, and the number of timeslices. Summing all tasks
    matters - a compio runtime and a thread-per-connection server both put work on
    threads the process did not create as tasks of the initial thread.

    **But a task-level sum is only valid if the task existed at both samples.** A
    thread-per-connection server creates its worker when a client connects and
    destroys it when the client goes away, so a before/after pair taken outside the
    measured window misses it entirely: an early probe of the blocking control read
    0.2 ms of on-CPU time against 9,000 served requests, while `/proc/<pid>/stat`
    for the same process said 60 ms. The 0.2 ms was a real reading of a real task
    and it was the wrong task.

    So `utime_ticks` and `stime_ticks` from `/proc/<pid>/stat` are reported
    alongside, and they cover every thread in the group without needing the thread to
    have been alive at sample time. When the two disagree, the `/proc/stat` figure
    is the one to believe, and the disagreement itself is the finding.

    `run_ns` and `runs` are therefore reported as a *lower bound* on the task-level
    view, and `sampler_threads` records the widest task set seen while the run was in
    flight, so a reader can see whether the lower bound is tight.
    """
    run_ns = wait_ns = runs = 0
    tasks = 0
    try:
        entries = sorted(Path(f"/proc/{pid}/task").iterdir(), key=lambda p: int(p.name))
    except (OSError, ValueError):
        entries = []
    for task in entries:
        try:
            fields = (task / "schedstat").read_text().split()
        except OSError:
            continue
        if len(fields) < 3:
            continue
        run_ns += int(fields[0])
        wait_ns += int(fields[1])
        runs += int(fields[2])
        tasks += 1

    utime_ticks = stime_ticks = 0
    try:
        # Fields after the comm field; `utime` is #14 and `stime` #15 overall, which
        # is index 11 and 12 after the state and the parenthesised comm.
        stat = Path(f"/proc/{pid}/stat").read_text()
        tail = stat[stat.rindex(")") + 1:].split()
        utime_ticks, stime_ticks = int(tail[11]), int(tail[12])
    except (OSError, ValueError, IndexError):
        pass

    return {
        "run_ns": run_ns,
        "wait_ns": wait_ns,
        "runs": runs,
        "threads": tasks,
        "utime_ticks": utime_ticks,
        "stime_ticks": stime_ticks,
    }


def sample_threads_during(pid: int, stop: "threading.Event", widest: list[int]) -> None:
    """Watches a process's task count until `stop` is set.

    A connection thread that lives for the duration of one client run can be missed
    entirely by a before/after pair if the client is faster than the gap between
    samples. Polling every few milliseconds while the run is in flight catches it,
    and the widest task set seen is reported so the reader can tell a tight lower
    bound from a loose one.
    """
    while not stop.is_set():
        try:
            count = len(list(Path(f"/proc/{pid}/task").iterdir()))
        except OSError:
            count = 0
        if count > widest[0]:
            widest[0] = count
        stop.wait(0.004)


# ------------------------------------------------------------------ measurements


@dataclass
class Sample:
    """One measurement of one target at one shape."""

    target: str
    dialect: str
    cmd: str
    payload: int
    clients: int
    pipeline: int
    blocks: int
    repeats: int
    ns_per_op: float
    ns_per_first: float
    ops_per_sec: float
    verified_bad: int
    thread_runs: int = 0
    server_run_ns: int = 0
    server_wait_ns: int = 0
    server_threads: int = 0
    sampler_threads: int = 0
    cpu_user_ns: int = 0
    cpu_sys_ns: int = 0
    strace_calls: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.verified_bad == 0 and self.ns_per_op > 0

    @property
    def cpu_ns(self) -> int:
        return self.cpu_user_ns + self.cpu_sys_ns

    @property
    def cpu_ns_per_op(self) -> float:
        ops = self.blocks * self.repeats * self.clients * self.pipeline
        return self.cpu_ns / ops if ops else 0.0


NS_TOTAL = re.compile(r"ns/op \(total wall time\)\s*\|\s*([0-9.]+)")
NS_FIRST = re.compile(r"ns/op \(first reply of each block\)\s*\|\s*([0-9.]+)")
OPS_SEC = re.compile(r"ops/s\s*\|\s*([0-9.]+)")
BAD = re.compile(r"replies that failed verification\s*\|\s*([0-9]+)")

# A repeat table row: `| 3 | 12345 | 6789 |`
REPEAT_ROW = re.compile(r"^\|\s*(\d+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|", re.M)


def run_client(
    binary: Path,
    args: list[str],
    timeout: float = 900.0,
) -> tuple[str, int]:
    """Runs the client, returning its stdout and its exit status."""
    proc = subprocess.run(
        [str(binary), *args], capture_output=True, text=True, timeout=timeout, check=False
    )
    return proc.stdout + proc.stderr, proc.returncode


def parse_client_output(text: str) -> dict[str, float]:
    """Pulls the headline figures out of the client's report.

    The client already reports the minimum over its own repeats, so the harness
    does not have to re-derive one from the repeat table - but the repeat table is
    kept so a run whose spread is wild can be spotted.
    """
    out: dict[str, float] = {}
    for name, pattern in (
        ("ns", NS_TOTAL),
        ("first", NS_FIRST),
        ("ops", OPS_SEC),
        ("bad", BAD),
    ):
        match = pattern.search(text)
        if match:
            out[name] = float(match.group(1))
    out["repeats"] = float(len(REPEAT_ROW.findall(text)))
    return out


def read_port(log: Path, timeout: float = 20.0) -> int | None:
    """Waits for a server's startup line and returns its port.

    Only the wirelab servers print `listen=`; Redis logs in its own format and never
    prints an address at all, so a target with a known port in advance is probed by
    connecting to it instead. Both paths are here because the alternative - requiring
    every server to be started by this harness - means the reference server is
    measured under conditions this harness chose for it.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            for line in log.read_text(errors="replace").splitlines():
                if line.startswith("listen="):
                    return int(line.split(":")[-1])
        except OSError:
            pass
        time.sleep(0.2)
    return None


def port_is_open(port: int, timeout: float = 0.3) -> bool:
    """Whether something is accepting connections on `port`."""
    import socket

    try:
        with socket.create_connection(("127.0.0.1", port), timeout=timeout):
            return True
    except OSError:
        return False


# ----------------------------------------------------------------------- targets


class Target:
    """A server the client can be pointed at.

    `port` may be set in advance for a server this harness does not start the way it
    starts its own - Redis, which announces its port only in its config and never on
    stdout. In that case readiness is a connect probe.
    """

    def __init__(self, name: str, dialect: str, command: list[str] | None, port: int | None):
        self.name = name
        self.dialect = dialect
        self.command = command
        self.port = port
        self.pid: int | None = None
        self.log: Path | None = None

    def start(self) -> bool:
        if self.command is None:
            return self.port is not None
        self.log = WORK / f"srv-{self.name}.txt"
        log = self.log.open("w")
        # `start_new_session` detaches the child from this process group, so it
        # inherits none of our stdio and the harness's own output never waits on a
        # server that is still running.
        self.pid = subprocess.Popen(
            self.command, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
            start_new_session=True,
        ).pid
        if self.port is not None:
            deadline = time.time() + 20.0
            while time.time() < deadline:
                if port_is_open(self.port):
                    return True
                time.sleep(0.2)
            return False
        self.port = read_port(self.log)
        return self.port is not None

    def stop(self) -> None:
        if self.pid is not None:
            kill_tree(self.pid)
            self.pid = None
        elif self.port is not None:
            # A target started outside this harness still has to be stopped by it, or
            # the next run's preflight kills a stranger's server.
            for pattern in (f"redis-server 127.0.0.1:{self.port}",):
                for pid in pgrep_all(pattern):
                    kill_tree(pid)

    def pids(self) -> list[int]:
        if self.pid is not None:
            return [self.pid]
        # For a target this harness did not start, find it by port so the scheduler
        # accounting has something to read.
        for pattern in (f"redis-server 127.0.0.1:{self.port}",):
            found = pgrep_all(pattern)
            if found:
                return found[:1]
        return []


def control_targets() -> list[Target]:
    """The five control servers, in the order the subtraction reads."""
    return [
        Target("a", "a", [str(LAB / "kivi-wirelab-server"), "--mode", "a"], None),
        Target("ap", "ap", [str(LAB / "kivi-wirelab-server"), "--mode", "ap"], None),
        Target("b", "b", [str(LAB / "kivi-wirelab-server"), "--mode", "b"], None),
        Target("c", "c", [str(LAB / "kivi-wirelab-server"), "--mode", "c"], None),
        Target("epoll", "epoll", [str(LAB / "kivi-wirelab-server"), "--mode", "epoll"], None),
    ]


def redis_target() -> Target:
    conf = WORK / "redis-wirelab.conf"
    conf.write_text(
        "\n".join(
            [
                f"bind 127.0.0.1",
                f"port {REDIS_PORT}",
                "save \"\"",
                "appendonly no",
                "daemonize no",
                "protected-mode no",
                "maxmemory-policy noeviction",
                # A pipeline of 256 must not be split by the query buffer, or the
                # harness would be measuring Kivi's buffering against a different
                # server's.
                "proto-max-bulk-len 64mb",
                "client-output-buffer-limit normal 0 0 0",
                "latency-monitor-threshold 0",
                "dir " + str(WORK),
            ]
        )
        + "\n"
    )
    return Target(
        "redis",
        "redis",
        [str(REDIS_BIN), str(conf)],
        REDIS_PORT,
    )


def kivi_target() -> Target:
    # The RESP edge, not the native one. `kivi-server` speaks its own protocol on
    # `--port` and RESP on `--redis-listen`; pointing this at the native port made the
    # target fail to start silently, and the first depth run reported only two of the
    # three rows with nothing saying the third was missing.
    #
    # Readiness is a connect probe on a port this harness chose, because
    # `kivi-server` announces its endpoints in a `KIVI_READY` line and not in the
    # `listen=` shape the wirelab servers print.
    return Target(
        "d",
        "d",
        [
            str(LAB / "kivi-server"),
            "--ephemeral",
            "--port", str(KIVI_NATIVE_PORT),
            "--admin", "127.0.0.1:0",
            "--workers", "1",
            "--tablets", "1",
            "--redis-listen", f"127.0.0.1:{KIVI_RESP_PORT}",
            "--no-checkpoint",
            "--placement", "none",
        ],
        KIVI_RESP_PORT,
    )


TARGET_DESCRIPTIONS = {
    "a": "compio, raw echo, one persistent buffer - the runtime floor",
    "ap": "compio, raw echo, per-read Vec + copy - Kivi's buffer pattern",
    "b": "compio, RESP parse, fixed bulk reply - no engine",
    "c": "compio, RESP parse, ObjectStore lookup - the local table",
    "epoll": "blocking threads, RESP parse, fixed reply - no async runtime",
    "d": "kivi-server, full engine",
    "redis": f"redis-server {REDIS_VERSION}",
}


# ------------------------------------------------------------------- measurement


def measure(
    target: Target,
    *,
    pipeline: int,
    clients: int,
    cmd: str,
    payload: int,
    blocks: int,
    repeats: int,
    strace: bool,
) -> Sample:
    """Measures one target at one shape, with scheduler accounting around it."""
    args = [
        "--target", target.dialect,
        "--addr", f"127.0.0.1:{target.port}",
        "--pipeline", str(pipeline),
        "--clients", str(clients),
        "--cmd", cmd,
        "--payload", str(payload),
        "--blocks", str(blocks),
        "--repeats", str(repeats),
    ]
    pids = target.pids()
    before = [thread_group(p) for p in pids]
    widest = [0]
    stop = threading.Event()
    sampler = None
    if pids:
        sampler = threading.Thread(
            target=sample_threads_during, args=(pids[0], stop, widest), daemon=True
        )
        sampler.start()
    strace_proc = None
    strace_path = WORK / f"strace-{target.name}-{pipeline}-{clients}.txt"
    if strace and pids:
        # `ptrace_scope` is 1 on this host, so strace may not attach to a sibling -
        # it can only trace its own descendants. The wirelab server and the client
        # are both children of the harness, so they are siblings, and an
        # attach-to-pid fails with `Operation not permitted`. Under
        # `kernel.yama.ptrace_scope = 0` it would work; without root it does not, and
        # a first pass produced zero-length summaries that read as "no syscalls".
        # So syscall counting is unavailable on this host by policy, not by choice,
        # and the report says so rather than printing a zero.
        strace_proc = None
    text, status = run_client(LAB / "kivi-wirelab-client", args)
    stop.set()
    if sampler is not None:
        sampler.join(timeout=2.0)
    if strace_proc is not None:
        time.sleep(0.3)
        strace_proc.send_signal(signal.SIGINT)
        try:
            strace_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            strace_proc.kill()
            strace_proc.wait()
    after = [thread_group(p) for p in pids]

    figures = parse_client_output(text)
    sample = Sample(
        target=target.name,
        dialect=target.dialect,
        cmd=cmd,
        payload=payload,
        clients=clients,
        pipeline=pipeline,
        blocks=blocks,
        repeats=repeats,
        ns_per_op=figures.get("ns", 0.0),
        ns_per_first=figures.get("first", 0.0),
        ops_per_sec=figures.get("ops", 0.0),
        verified_bad=int(figures.get("bad", 1)),
    )
    if status != 0:
        sample.notes.append(f"client exit {status}")
    if not sample.ok:
        # Keep the client's own report. A rejected run whose output is discarded is a
        # rejected run nobody can diagnose, and the first Redis run failed
        # verification for a reason that took a guess to find.
        dump = WORK / f"rejected-{target.name}-{pipeline}-{clients}.txt"
        dump.write_text(text)
        sample.notes.append(f"client output in {dump}")
    if figures.get("repeats", 0) < repeats:
        sample.notes.append(
            f"client reported {figures.get('repeats', 0):.0f} of {repeats} repeats"
        )
    if before and after:
        sample.thread_runs = sum(a["runs"] - b["runs"] for a, b in zip(after, before))
        sample.server_run_ns = sum(a["run_ns"] - b["run_ns"] for a, b in zip(after, before))
        sample.server_wait_ns = sum(a["wait_ns"] - b["wait_ns"] for a, b in zip(after, before))
        sample.server_threads = sum(a["threads"] for a in after)
        # The process-wide figures. These cover every thread in the group, including
        # any that existed only while the run was in flight.
        sample.cpu_user_ns = sum(
            (a["utime_ticks"] - b["utime_ticks"]) * (10**9 // USER_HZ)
            for a, b in zip(after, before)
        )
        sample.cpu_sys_ns = sum(
            (a["stime_ticks"] - b["stime_ticks"]) * (10**9 // USER_HZ)
            for a, b in zip(after, before)
        )
        if sample.cpu_user_ns < 0 or sample.cpu_sys_ns < 0:
            sample.notes.append("cpu ticks went backwards; sample discarded")
    sample.sampler_threads = widest[0]
    if strace_path.exists() and strace_path.stat().st_size > 0:
        sample.strace_calls = strace_total_calls(strace_path)
    elif strace:
        sample.notes.append(
            "syscall counts unavailable: kernel.yama.ptrace_scope=1 forbids strace "
            "attaching to a sibling process"
        )
    return sample


def strace_total_calls(path: Path) -> int:
    """Total syscalls from an `strace -c` summary.

    The summary's last line is `N total`; parsing the per-syscall rows would
    silently drop anything with an unusual name, and a count that is quietly
    missing rows is worse than no count.
    """
    try:
        for line in reversed(path.read_text(errors="replace").splitlines()):
            if "total" in line:
                match = re.search(r"(\d+)\s+total", line)
                if match:
                    return int(match.group(1))
    except OSError:
        pass
    return 0


# ------------------------------------------------------------------------ output


def table(samples: list[Sample], *, title: str, note: str = "", timing: bool = True) -> str:
    """Renders one table.

    `timing=False` omits every latency column. That is not a formatting choice:
    `strace -c -f -p` attaches with ptrace and adds a stop-and-copy on every
    syscall, which on a server that issues a handful of syscalls per request inflates
    the request itself. A first pass measured mode `a` at 36,103 ns with `strace`
    attached and 23,332 ns without it - a 55% inflation on the row that was supposed
    to be the floor. Counting syscalls and timing latency in one pass produces two
    numbers and neither is usable, so the syscall pass is run separately and reports
    counts only.
    """
    lines = [f"## {title}", ""]
    if note:
        lines += [note, ""]
    if timing:
        lines += [
            "| target | what it is | c | p | ns/op | ops/s | CPU ns/op | user | kernel | slices/op | threads |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    else:
        lines += [
            "| target | what it is | c | p | syscalls in run | ops in run | syscalls/op |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    for s in samples:
        bad = "ok" if s.verified_bad == 0 else f"**{s.verified_bad} BAD**"
        what = TARGET_DESCRIPTIONS.get(s.target, s.target)
        ops = s.blocks * s.repeats * s.clients * s.pipeline
        if timing:
            lines.append(
                f"| `{s.target}` | {what} | {s.clients} | {s.pipeline} | {s.ns_per_op:.0f} "
                f"| {s.ops_per_sec:.0f} | {s.cpu_ns_per_op:.0f} | {s.cpu_user_ns / 1e6:.0f} ms "
                f"| {s.cpu_sys_ns / 1e6:.0f} ms | {s.thread_runs / ops:.2f} "
                f"| {max(s.sampler_threads, s.server_threads)} |"
            )
        else:
            per_op = s.strace_calls / ops if ops else 0.0
            lines.append(
                f"| `{s.target}` | {what} | {s.clients} | {s.pipeline} | {s.strace_calls} "
                f"| {ops} | {per_op:.2f} |"
            )
    return "\n".join(lines) + "\n"


def sweep_event_interval(
    intervals: list[int], *, blocks: int, repeats: int, pipeline: int, out: Path
) -> list[Sample]:
    """Measures mode `a` - a raw echo - at several `event_interval` settings.

    This is the cheapest available test of "is compio sleeping when it should be
    working". `event_interval` is how many scheduler ticks pass between polls of the
    io_uring ring for external events; compio's default is 61, chosen for a runtime
    that keeps the ring busy. A raw echo server with one request per turn is the
    opposite of that runtime, so a completion that arrives inside the interval waits
    for it to expire.

    The control is deliberately mode `a`. Anything that parses RESP or touches the
    store would put work between the runtime and the ring, and the question here is
    purely about the runtime's own request lifecycle.
    """
    samples: list[Sample] = []
    for interval in intervals:
        target = Target(
            f"a@event_interval={interval}",
            "a",
            [
                str(LAB / "kivi-wirelab-server"),
                "--mode", "a",
                "--event-interval", str(interval),
            ],
            None,
        )
        if not target.start():
            print(f"  event_interval={interval}: failed to start", flush=True)
            continue
        sample = measure(
            target, pipeline=pipeline, clients=1, cmd="get", payload=16,
            blocks=blocks, repeats=repeats, strace=False,
        )
        samples.append(sample)
        target.stop()
        print(
            f"  event_interval={interval:<5} {sample.ns_per_op:>9.0f} ns/op  "
            f"cpu={sample.cpu_ns_per_op:>7.0f} ns/op  first={sample.ns_per_first:>9.0f}  "
            f"bad={sample.verified_bad}",
            flush=True,
        )
    return samples


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shapes", default="1:1",
                        help="comma-separated clients:pipeline shapes, e.g. 1:1,1:16,4:16,16:1")
    parser.add_argument("--cmd", default="get", choices=["get", "set"])
    parser.add_argument("--payload", type=int, default=16)
    parser.add_argument("--blocks", type=int, default=4000)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--targets", default="a,ap,b,c,epoll,d,redis")
    parser.add_argument("--strace", action="store_true")
    parser.add_argument(
        "--sweep-event-interval", default="",
        help="comma-separated event_interval values to measure mode `a` at, e.g. 1,4,16,61",
    )
    parser.add_argument("--out", default=str(WORK / "report.md"))
    args = parser.parse_args()

    WORK.mkdir(parents=True, exist_ok=True)
    if args.sweep_event_interval:
        preflight()
        intervals = [int(x) for x in args.sweep_event_interval.split(",") if x.strip()]
        samples = sweep_event_interval(
            intervals, blocks=args.blocks, repeats=args.repeats,
            pipeline=1, out=Path(args.out),
        )
        report = [
            "# Wirelab: compio `event_interval` sweep",
            "",
            "Mode `a`: a raw TCP echo with no protocol, no parser and no store. Any",
            "difference between these rows is the runtime's own request lifecycle, and",
            "nothing else. compio's default is 61.",
            "",
            "| event_interval | ns/op | ns/first | CPU ns/op | kernel ms |",
            "| ---: | ---: | ---: | ---: | ---: |",
        ]
        for s in samples:
            report.append(
                f"| {s.target.split('=')[-1]} | {s.ns_per_op:.0f} | {s.ns_per_first:.0f} "
                f"| {s.cpu_ns_per_op:.0f} | {s.cpu_sys_ns / 1e6:.0f} |"
            )
        Path(args.out).write_text("\n".join(report) + "\n")
        print(f"\nwrote {args.out}")
        return 0
    shapes = []
    for item in args.shapes.split(","):
        clients, _, pipeline = item.partition(":")
        shapes.append((int(clients), int(pipeline or 1)))

    registry: dict[str, Target] = {t.name: t for t in control_targets()}
    registry["redis"] = redis_target()
    registry["d"] = kivi_target()
    wanted = [name.strip() for name in args.targets.split(",") if name.strip()]

    preflight()
    started: list[Target] = []
    for name in wanted:
        target = registry.get(name)
        if target is None:
            print(f"unknown target {name!r}", file=sys.stderr)
            continue
        if not target.start():
            print(f"target {name} failed to start; see {target.log}", file=sys.stderr)
            if target.log and target.log.exists():
                print(target.log.read_text(errors="replace")[:2000], file=sys.stderr)
            continue
        started.append(target)
        print(f"started {name} on port {target.port}", flush=True)

    samples: list[Sample] = []
    try:
        for clients, pipeline in shapes:
            for target in started:
                sample = measure(
                    target,
                    pipeline=pipeline, clients=clients, cmd=args.cmd,
                    payload=args.payload, blocks=args.blocks, repeats=args.repeats,
                    strace=args.strace,
                )
                samples.append(sample)
                flag = "" if sample.ok else "  <-- REJECTED"
                if args.strace:
                    ops = sample.blocks * sample.repeats * sample.pipeline
                    per_op = sample.strace_calls / ops if ops else 0.0
                    print(
                        f"  {target.name:<6} c={clients} p={pipeline:<4} cpu={sample.cpu_ns_per_op:>7.0f}"
                        f"syscalls={sample.strace_calls:<9} ({per_op:5.2f}/op)  "
                        f"sched_runs={sample.thread_runs}{flag}",
                        flush=True,
                    )
                else:
                    print(
                        f"  {target.name:<6} c={clients} p={pipeline:<4} cpu={sample.cpu_ns_per_op:>7.0f}"
                        f"{sample.ns_per_op:>9.0f} ns/op  first={sample.ns_per_first:>9.0f}  "
                        f"bad={sample.verified_bad}{flag}",
                        flush=True,
                    )
    finally:
        for target in started:
            target.stop()

    report = [
        "# Wirelab external report",
        "",
        "- host: Intel Core Ultra 9 285K, WSL2, 24 cores, no SMT",
        f"- client: `kivi-wirelab-client`, one generator for every target",
        f"- redis: {REDIS_VERSION}, built from source, no Docker",
        f"- shape: cmd={args.cmd} payload={args.payload}B blocks={args.blocks} "
        f"repeats={args.repeats}, min reported",
        "",
    ]
    if args.strace:
        report.append(
            table(
                samples,
                title="syscalls and scheduler runs (timing omitted: ptrace inflates it)",
                timing=False,
            )
        )
    else:
        report.append(table(samples, title="all targets, all shapes"))
    report.append(
        "The control rows are subtraction, not alternatives: `a` is what any compio\n"
        "TCP server costs on this host with no protocol at all, and every row above\n"
        "it is that floor plus one named layer. `ap` against `a` prices Kivi's\n"
        "per-read allocation and copy. `b` against `ap` prices RESP. `c` against `b`\n"
        "prices the local table - the same probe `read_path.rs` measures at 6.7 ns.\n"
        "`epoll` against `b` prices the async runtime. `d` against `c` is Kivi's own\n"
        "architecture: routing, durability, admission and the general engine."
    )
    Path(args.out).write_text("\n".join(report) + "\n")
    (WORK / "samples.json").write_text(json.dumps([asdict(s) for s in samples], indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
