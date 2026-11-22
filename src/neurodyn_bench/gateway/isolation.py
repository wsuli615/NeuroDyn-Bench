"""Rootless offline network isolation (seccomp-bpf) and syscall path auditing for Track B sandboxes.

Provides defense-in-depth for running untrusted / open coding agents without requiring sudo:
  1. Kernel-level network isolation (`seccomp-bpf`): blocks `socket`/`socketpair` for all
     non-AF_UNIX domains (e.g., AF_INET, AF_INET6, AF_PACKET) with EPERM across all forked
     and exec'd child processes, while permitting AF_UNIX so the sandbox talks to the host
     TrackBSandboxServer over `.gateway.sock`.
  2. Environment scrubbing: strips API keys, tokens, SSH/proxy env vars, and sets HOME/TMPDIR
     inside the isolated sandbox workspace.
  3. OS-level syscall auditing (`strace`): records every file open and socket attempt; flags
     any attempt to access forbidden host paths (private/, ground_truth.yaml, heldout_*,
     validation_cert.yaml, build_v0_tasks.py) or any succeeded outbound network socket as a
     hard `security_violation` in `trace.jsonl` (causing Evaluator immediate invalidation).
  4. Optional Docker backend (`--network none --read-only`) for Docker-enabled evaluation hosts.
"""

from __future__ import annotations

import ctypes
import errno
import os
import platform
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from neurodyn_bench.gateway.sandbox import TrackBSandboxServer

# Linux prctl & seccomp constants (x86_64)
_PR_SET_NO_NEW_PRIVS = 38
_PR_SET_SECCOMP = 22
_SECCOMP_MODE_FILTER = 2

_BPF_LD = 0x00
_BPF_W = 0x00
_BPF_ABS = 0x20
_BPF_JMP = 0x05
_BPF_JEQ = 0x10
_BPF_K = 0x00
_BPF_RET = 0x06

_AUDIT_ARCH_X86_64 = 0xC000003E
_SYS_SOCKET_X86_64 = 41
_SYS_SOCKETPAIR_X86_64 = 53
_AF_UNIX = 1

_SECCOMP_RET_ALLOW = 0x7FFF0000
_SECCOMP_RET_ERRNO = 0x00050000

_FORBIDDEN_BASENAMES = (
    "ground_truth.yaml",
    "heldout_conditions.tsv",
    "heldout_observables.tsv",
    "validation_cert.yaml",
    "ask_human_oracle.yaml",
    "build_v0_tasks.py",
    "certify_v0_tasks.py",
)

_STRACE_PATH_RE = re.compile(r'"([^"]+)"')
_STRACE_SOCKET_RE = re.compile(r"socket\((AF_[A-Z0-9_]+),.*\)\s*=\s*(-?\d+)")


class _SockFilter(ctypes.Structure):
    _fields_ = [
        ("code", ctypes.c_uint16),
        ("jt", ctypes.c_uint8),
        ("jf", ctypes.c_uint8),
        ("k", ctypes.c_uint32),
    ]


class _SockFprog(ctypes.Structure):
    _fields_ = [
        ("len", ctypes.c_uint16),
        ("filter", ctypes.POINTER(_SockFilter)),
    ]


def install_offline_seccomp_filter() -> None:
    """Install a kernel seccomp-bpf filter allowing AF_UNIX while blocking all other sockets.

    Inherited automatically across fork() and execve(); cannot be removed once installed.
    """
    if sys.platform != "linux" or platform.machine() != "x86_64":
        raise RuntimeError(
            f"seccomp-bpf offline filter currently supports linux/x86_64 only "
            f"(got {sys.platform}/{platform.machine()})"
        )

    eperm_ret = _SECCOMP_RET_ERRNO | errno.EPERM
    insns = [
        # 0: Load architecture from seccomp_data.arch (offset 4)
        _SockFilter(_BPF_LD | _BPF_W | _BPF_ABS, 0, 0, 4),
        # 1: Require AUDIT_ARCH_X86_64 (block 32-bit compat syscall bypass)
        _SockFilter(_BPF_JMP | _BPF_JEQ | _BPF_K, 1, 0, _AUDIT_ARCH_X86_64),
        # 2: Deny non-x86_64 arch
        _SockFilter(_BPF_RET | _BPF_K, 0, 0, eperm_ret),
        # 3: Load syscall number (offset 0)
        _SockFilter(_BPF_LD | _BPF_W | _BPF_ABS, 0, 0, 0),
        # 4: If nr == SYS_socket (41) -> jump to domain check (6); else check socketpair (5)
        _SockFilter(_BPF_JMP | _BPF_JEQ | _BPF_K, 1, 0, _SYS_SOCKET_X86_64),
        # 5: If nr == SYS_socketpair (53) -> domain check (6); else ALLOW (9)
        _SockFilter(_BPF_JMP | _BPF_JEQ | _BPF_K, 0, 3, _SYS_SOCKETPAIR_X86_64),
        # 6: Load args[0] (socket domain, offset 16)
        _SockFilter(_BPF_LD | _BPF_W | _BPF_ABS, 0, 0, 16),
        # 7: If domain == AF_UNIX (1) -> ALLOW (9); else EPERM (8)
        _SockFilter(_BPF_JMP | _BPF_JEQ | _BPF_K, 1, 0, _AF_UNIX),
        # 8: Return EPERM
        _SockFilter(_BPF_RET | _BPF_K, 0, 0, eperm_ret),
        # 9: Return ALLOW
        _SockFilter(_BPF_RET | _BPF_K, 0, 0, _SECCOMP_RET_ALLOW),
    ]

    arr = (_SockFilter * len(insns))(*insns)
    prog = _SockFprog(len(insns), arr)
    libc = ctypes.CDLL("libc.so.6", use_errno=True)

    if libc.prctl(_PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
        err = ctypes.get_errno()
        raise OSError(err, f"prctl(PR_SET_NO_NEW_PRIVS) failed: {os.strerror(err)}")
    if libc.prctl(_PR_SET_SECCOMP, _SECCOMP_MODE_FILTER, ctypes.byref(prog), 0, 0) != 0:
        err = ctypes.get_errno()
        raise OSError(err, f"prctl(PR_SET_SECCOMP) failed: {os.strerror(err)}")


def build_sanitized_env(
    workspace_dir: Path,
    socket_path: Path,
    extra_env: dict[str, str] | None = None,
) -> dict[str, str]:
    """Construct a minimal, secret-free environment for sandboxed agent execution."""
    tmp_dir = workspace_dir / ".tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": str(workspace_dir),
        "TMPDIR": str(tmp_dir),
        "PYTHONPATH": str(workspace_dir),
        "NEURODYN_SOCKET_PATH": str(socket_path),
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMBA_NUM_THREADS": "1",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    if extra_env:
        env.update(extra_env)
    return env


@dataclass
class IsolationAuditResult:
    """Result of executing a command inside an isolated Track B sandbox."""

    returncode: int
    stdout: str
    stderr: str
    timed_out: bool = False
    seccomp_active: bool = False
    strace_active: bool = False
    path_violations: list[str] = field(default_factory=list)
    network_attempts: list[str] = field(default_factory=list)
    network_escapes_succeeded: list[str] = field(default_factory=list)

    @property
    def security_violated(self) -> bool:
        return bool(self.path_violations or self.network_escapes_succeeded)

    def to_dict(self) -> dict:
        return {
            "returncode": self.returncode,
            "timed_out": self.timed_out,
            "seccomp_active": self.seccomp_active,
            "strace_active": self.strace_active,
            "security_violated": self.security_violated,
            "path_violations": self.path_violations[:20],
            "network_attempts": self.network_attempts[:20],
            "network_escapes_succeeded": self.network_escapes_succeeded[:20],
        }


def audit_strace_log(
    strace_log: Path,
    workspace_dir: Path,
    forbidden_roots: list[Path],
) -> tuple[list[str], list[str], list[str]]:
    """Inspect strace output for forbidden file accesses and non-AF_UNIX socket attempts."""
    path_violations: list[str] = []
    network_attempts: list[str] = []
    network_escapes: list[str] = []

    if not strace_log.exists():
        return path_violations, network_attempts, network_escapes

    ws_real = str(workspace_dir.resolve())
    forb_reals = [str(p.resolve()) for p in forbidden_roots]

    seen_paths: set[str] = set()
    for line in strace_log.read_text(encoding="utf-8", errors="replace").splitlines():
        m_sock = _STRACE_SOCKET_RE.search(line)
        if m_sock:
            domain = m_sock.group(1)
            ret_fd = int(m_sock.group(2))
            if domain != "AF_UNIX":
                entry = f"{domain} (ret={ret_fd})"
                network_attempts.append(entry)
                if ret_fd >= 0:
                    network_escapes.append(entry)
            continue

        if any(tok in line for tok in ("openat(", "open(", "newfstatat(", "stat(", "readlink(")):
            for raw_p in _STRACE_PATH_RE.findall(line):
                if raw_p in seen_paths:
                    continue
                # Resolve relative paths against workspace_dir
                p_obj = Path(raw_p) if os.path.isabs(raw_p) else (workspace_dir / raw_p)
                try:
                    resolved = str(p_obj.resolve())
                except OSError:
                    resolved = raw_p

                # Paths strictly inside the exported sandbox workspace are allowed
                if resolved == ws_real or resolved.startswith(ws_real + os.sep):
                    continue

                is_forbidden = False
                if "/private/" in resolved or resolved.endswith("/private") or any(resolved.endswith(os.sep + bn) or raw_p.endswith(bn) for bn in _FORBIDDEN_BASENAMES) or any(resolved == fr or resolved.startswith(fr + os.sep) for fr in forb_reals):
                    is_forbidden = True

                if is_forbidden:
                    seen_paths.add(raw_p)
                    path_violations.append(resolved)

    return path_violations, network_attempts, network_escapes


def run_isolated_sandbox_command(
    server: TrackBSandboxServer,
    command: list[str],
    timeout_s: float = 300.0,
    backend: Literal["seccomp", "docker"] = "seccomp",
    enable_strace_audit: bool = True,
    forbidden_roots: list[Path] | None = None,
    docker_image: str = "python:3.12-slim",
    extra_env: dict[str, str] | None = None,
) -> IsolationAuditResult:
    """Execute a Track B agent command under offline network isolation and path auditing.

    Automatically records any detected security violation (forbidden path read or escaped
    outbound network socket) into `server.gateway`'s `trace.jsonl`, which triggers
    immediate invalidation in `evaluate_submission()`.
    """
    ws = server.workspace_dir.resolve()
    repo_root = Path(__file__).resolve().parents[3]
    default_forbidden = [
        repo_root / "tasks",
        repo_root / "scripts",
        repo_root / "outputs",
        repo_root / "tests",
    ]
    forb_roots = list(forbidden_roots) if forbidden_roots is not None else default_forbidden

    if backend == "docker":
        docker_bin = shutil.which("docker")
        if not docker_bin:
            raise RuntimeError("Docker backend requested but 'docker' binary not found.")
        docker_cmd = [
            docker_bin,
            "run",
            "--rm",
            "--network",
            "none",
            "--read-only",
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,size=512m",
            "-v",
            f"{ws}:/workspace:rw",
            "-w",
            "/workspace",
            "-e",
            "NEURODYN_SOCKET_PATH=/workspace/.gateway.sock",
            "-e",
            "PYTHONPATH=/workspace",
            "-e",
            "OMP_NUM_THREADS=1",
            "-e",
            "OPENBLAS_NUM_THREADS=1",
            docker_image,
            *command,
        ]
        try:
            proc = subprocess.run(
                docker_cmd, capture_output=True, text=True, timeout=timeout_s, check=False
            )
            return IsolationAuditResult(
                returncode=proc.returncode,
                stdout=proc.stdout,
                stderr=proc.stderr,
                seccomp_active=True,
                strace_active=False,
            )
        except subprocess.TimeoutExpired as exc:
            return IsolationAuditResult(
                returncode=-1,
                stdout=(exc.stdout or "") if isinstance(exc.stdout, str) else "",
                stderr=(exc.stderr or "") if isinstance(exc.stderr, str) else "",
                timed_out=True,
                seccomp_active=True,
                strace_active=False,
            )

    # Rootless Linux seccomp-bpf + strace audit backend
    env = build_sanitized_env(ws, server.socket_path, extra_env=extra_env)
    strace_bin = shutil.which("strace") if enable_strace_audit else None
    strace_log = ws.parent / f".strace_{ws.name}.log"

    if strace_bin:
        run_cmd = [
            strace_bin,
            "-f",
            "-e",
            "trace=openat,open,newfstatat,stat,readlink,socket,connect",
            "-o",
            str(strace_log),
            "--",
            *command,
        ]
    else:
        run_cmd = list(command)

    timed_out = False
    try:
        proc = subprocess.run(
            run_cmd,
            cwd=str(ws),
            env=env,
            preexec_fn=install_offline_seccomp_filter,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
        rc, out, err = proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        rc = -1
        out = (exc.stdout or "") if isinstance(exc.stdout, str) else ""
        err = (exc.stderr or "") if isinstance(exc.stderr, str) else ""

    path_viols, net_attempts, net_escapes = ([], [], [])
    if strace_bin and strace_log.exists():
        path_viols, net_attempts, net_escapes = audit_strace_log(strace_log, ws, forb_roots)
        try:
            strace_log.unlink()
        except OSError:
            pass

    audit = IsolationAuditResult(
        returncode=rc,
        stdout=out,
        stderr=err,
        timed_out=timed_out,
        seccomp_active=True,
        strace_active=bool(strace_bin),
        path_violations=path_viols,
        network_attempts=net_attempts,
        network_escapes_succeeded=net_escapes,
    )

    if audit.security_violated or audit.network_attempts:
        server.gateway._append_trace(
            {
                "event": "security_violation" if audit.security_violated else "security_audit_notice",
                "round_id": server.gateway._round_counter,
                "audit": audit.to_dict(),
            }
        )

    return audit
