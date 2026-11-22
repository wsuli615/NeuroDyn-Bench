"""Track B Open Python Sandbox: Unix-socket gateway server & isolated workspace exporter.

Materializes a clean agent workspace containing ONLY rendered public/ files and a
lightweight `neurodyn_client.py` proxy. All simulator calls cross a Unix domain socket
into `TrackBSandboxServer` in the host process, ensuring physical isolation of private/
ground truths and tamper-proof BudgetLedger accounting.
"""

from __future__ import annotations

import json
import shutil
import socket
import threading
from pathlib import Path
from typing import Any, Self

import yaml

from neurodyn_bench.gateway.gateway import SimulatorGateway
from neurodyn_bench.optim.budget import BudgetExhaustedError
from neurodyn_bench.tasks.loader import _write_tsv
from neurodyn_bench.tasks.schema import DisclosureTier, FinalSubmission

_CLIENT_MODULE_TEMPLATE = '''"""Client proxy for NeuroDyn-Bench Track B Open Python Sandbox.

Usage inside the isolated workspace:
    from neurodyn_client import SandboxClient
    client = SandboxClient()
    info = client.task_info()
    res = client.evaluate({"J_ee": 1.2, ...}, seed=0)
    client.submit_solution(theta_hat)
"""

from __future__ import annotations

import json
import os
import socket
from pathlib import Path
from typing import Any


class BudgetExhaustedError(RuntimeError):
    """Raised when the hard simulation budget is exhausted."""


class SandboxClient:
    def __init__(self, socket_path: str | None = None) -> None:
        self.socket_path = socket_path or os.environ.get(
            "NEURODYN_SOCKET_PATH", str(Path(__file__).resolve().parent / ".gateway.sock")
        )

    def _rpc(self, method: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        req = json.dumps({"method": method, "payload": payload or {}}) + "\\n"
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.connect(self.socket_path)
            sock.sendall(req.encode("utf-8"))
            chunks: list[bytes] = []
            while True:
                data = sock.recv(65536)
                if not data:
                    break
                chunks.append(data)
                if b"\\n" in data:
                    break
        resp = json.loads(b"".join(chunks).decode("utf-8").strip())
        if not resp.get("ok", False):
            err_type = resp.get("error_type", "RuntimeError")
            msg = resp.get("error", "Unknown RPC error")
            if err_type == "BudgetExhaustedError":
                raise BudgetExhaustedError(msg)
            raise RuntimeError(f"{err_type}: {msg}")
        return resp["result"]

    def task_info(self) -> dict[str, Any]:
        return self._rpc("task_info")

    def budget(self) -> dict[str, Any]:
        return self._rpc("budget")

    def evaluate(
        self,
        params: dict[str, float],
        seed: int = 0,
        condition_id: str = "baseline",
        duration_scale: float = 1.0,
        tool_tag: str = "sandbox_eval",
    ) -> dict[str, Any]:
        return self._rpc(
            "evaluate",
            {
                "params": params,
                "seed": seed,
                "condition_id": condition_id,
                "duration_scale": duration_scale,
                "tool_tag": tool_tag,
            },
        )

    def evaluate_repeats(
        self,
        params: dict[str, float],
        seeds: list[int],
        condition_id: str = "baseline",
        duration_scale: float = 1.0,
        tool_tag: str = "sandbox_repeat_eval",
    ) -> dict[str, Any]:
        return self._rpc(
            "evaluate_repeats",
            {
                "params": params,
                "seeds": seeds,
                "condition_id": condition_id,
                "duration_scale": duration_scale,
                "tool_tag": tool_tag,
            },
        )

    def log_note(self, tool_name: str, summary: dict[str, Any], hypothesis: str = "") -> dict[str, Any]:
        return self._rpc(
            "record_tool_event",
            {"tool_name": tool_name, "summary": summary, "hypothesis": hypothesis},
        )

    def submit_solution(self, theta_hat: dict[str, float], notes: str = "") -> dict[str, Any]:
        return self._rpc("submit_solution", {"theta_hat": theta_hat, "notes": notes})

    def abstain_insufficient_evidence(
        self,
        subtype: str,
        evidence_summary: dict[str, Any],
        notes: str = "",
    ) -> dict[str, Any]:
        return self._rpc(
            "abstain",
            {"subtype": subtype, "evidence_summary": evidence_summary, "notes": notes},
        )
'''


def export_track_b_sandbox(
    gateway: SimulatorGateway,
    workspace_dir: str | Path,
) -> Path:
    """Create a physically isolated Track B workspace containing ONLY rendered public/ files."""
    ws = Path(workspace_dir)
    pub_out = ws / "public"
    if pub_out.exists():
        shutil.rmtree(pub_out)
    pub_out.mkdir(parents=True, exist_ok=True)

    bundle = gateway.bundle
    (pub_out / "task.yaml").write_text(
        yaml.safe_dump(bundle.metadata.model_dump(), sort_keys=False), encoding="utf-8"
    )
    (pub_out / "budget.yaml").write_text(
        yaml.safe_dump(bundle.budget.model_dump(), sort_keys=False), encoding="utf-8"
    )
    (pub_out / "simulator_config.yaml").write_text(
        yaml.safe_dump(bundle.simulator_config.model_dump(), sort_keys=False), encoding="utf-8"
    )
    _write_tsv(
        pub_out / "parameters.tsv",
        [
            "name",
            "unit",
            "default_lb",
            "default_ub",
            "phys_hard_lb",
            "phys_hard_ub",
            "scale",
            "nominal",
            "active",
            "description",
        ],
        [p.model_dump() for p in bundle.parameters],
    )
    _write_tsv(
        pub_out / "observables.tsv",
        ["obs_id", "condition_id", "target_mean", "target_sd", "weight", "tolerance_tau"],
        [o.model_dump() for o in bundle.observables],
    )
    _write_tsv(
        pub_out / "conditions.tsv",
        ["condition_id", "param_overrides_json", "description"],
        [
            {
                "condition_id": c.condition_id,
                "param_overrides_json": json.dumps(c.param_overrides, sort_keys=True),
                "description": c.description,
            }
            for c in bundle.conditions
        ],
    )

    # Copy public target reference numpy arrays if present (e.g., FastDMF target_fc.npy)
    if bundle.public_dir:
        src_pub = Path(bundle.public_dir)
        for fname in (bundle.simulator_config.target_fc_file, bundle.simulator_config.target_fcd_file):
            if fname and (src_pub / fname).exists():
                shutil.copy2(src_pub / fname, pub_out / fname)

    (ws / "neurodyn_client.py").write_text(_CLIENT_MODULE_TEMPLATE, encoding="utf-8")
    return ws


class TrackBSandboxServer:
    """Host-side Unix domain socket server bridging Track B subprocesses to SimulatorGateway."""

    def __init__(
        self,
        gateway: SimulatorGateway,
        workspace_dir: str | Path,
        socket_path: str | Path | None = None,
    ) -> None:
        self.gateway = gateway
        self.workspace_dir = export_track_b_sandbox(gateway, workspace_dir)
        self.socket_path = Path(socket_path) if socket_path else (self.workspace_dir / ".gateway.sock")
        self.submission_path = self.workspace_dir / "final_submission.json"
        self._sock: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()

    def _handle_method(self, method: str, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            if method == "task_info":
                return self.gateway.public_task_info()
            if method == "budget":
                return self.gateway.ledger.summary()
            if method == "evaluate":
                rec = self.gateway.evaluate(
                    params=payload["params"],
                    seed=int(payload.get("seed", 0)),
                    condition_id=str(payload.get("condition_id", "baseline")),
                    duration_scale=float(payload.get("duration_scale", 1.0)),
                    tool_tag=str(payload.get("tool_tag", "sandbox_eval")),
                )
                return rec.to_dict()
            if method == "evaluate_repeats":
                return self.gateway.evaluate_repeats(
                    params=payload["params"],
                    seeds=[int(s) for s in payload["seeds"]],
                    condition_id=str(payload.get("condition_id", "baseline")),
                    duration_scale=float(payload.get("duration_scale", 1.0)),
                    tool_tag=str(payload.get("tool_tag", "sandbox_repeat_eval")),
                )
            if method == "record_tool_event":
                self.gateway.record_tool_event(
                    tool_name=str(payload["tool_name"]),
                    summary=dict(payload.get("summary", {})),
                    hypothesis=str(payload.get("hypothesis", "")),
                )
                return {"recorded": True}
            if method == "submit_solution":
                sub = FinalSubmission(
                    decision="submit_solution",
                    theta_hat={k: float(v) for k, v in payload["theta_hat"].items()},
                    notes=str(payload.get("notes", "")),
                )
                self.gateway.save_submission(sub, self.submission_path)
                return {"saved": str(self.submission_path), "decision": sub.decision}
            if method == "abstain":
                sub = FinalSubmission(
                    decision="abstain_insufficient_evidence",
                    abstention_subtype=payload["subtype"],
                    evidence_summary=dict(payload.get("evidence_summary", {})),
                    notes=str(payload.get("notes", "")),
                )
                self.gateway.save_submission(sub, self.submission_path)
                return {"saved": str(self.submission_path), "decision": sub.decision}
            raise ValueError(f"Unknown RPC method: '{method}'")

    def _serve_loop(self) -> None:
        assert self._sock is not None
        while not self._stop.is_set():
            try:
                conn, _ = self._sock.accept()
            except TimeoutError:
                continue
            except OSError:
                break
            with conn:
                chunks: list[bytes] = []
                while True:
                    data = conn.recv(65536)
                    if not data:
                        break
                    chunks.append(data)
                    if b"\n" in data:
                        break
                if not chunks:
                    continue
                try:
                    req = json.loads(b"".join(chunks).decode("utf-8").strip())
                    result = self._handle_method(req["method"], req.get("payload", {}))
                    resp = {"ok": True, "result": result}
                except BudgetExhaustedError as exc:
                    resp = {"ok": False, "error_type": "BudgetExhaustedError", "error": str(exc)}
                except Exception as exc:  # noqa: BLE001
                    resp = {"ok": False, "error_type": type(exc).__name__, "error": str(exc)}
                conn.sendall((json.dumps(resp) + "\n").encode("utf-8"))

    def start(self) -> None:
        if self.socket_path.exists():
            self.socket_path.unlink()
        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._sock.bind(str(self.socket_path))
        self._sock.listen(16)
        self._sock.settimeout(0.2)
        self._stop.clear()
        self._thread = threading.Thread(target=self._serve_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._sock is not None:
            self._sock.close()
        if self.socket_path.exists():
            self.socket_path.unlink()

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(self, *args: object) -> None:
        self.stop()


class TrackBSandboxClient:
    """In-process Python convenience client matching the workspace neurodyn_client.py API."""

    def __init__(self, socket_path: str | Path) -> None:
        self.socket_path = str(socket_path)

    def _rpc(self, method: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        req = json.dumps({"method": method, "payload": payload or {}}) + "\n"
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.connect(self.socket_path)
            sock.sendall(req.encode("utf-8"))
            chunks: list[bytes] = []
            while True:
                data = sock.recv(65536)
                if not data:
                    break
                chunks.append(data)
                if b"\n" in data:
                    break
        resp = json.loads(b"".join(chunks).decode("utf-8").strip())
        if not resp.get("ok", False):
            if resp.get("error_type") == "BudgetExhaustedError":
                raise BudgetExhaustedError(resp.get("error", "Budget exhausted"))
            raise RuntimeError(f"{resp.get('error_type')}: {resp.get('error')}")
        return resp["result"]

    def task_info(self) -> dict[str, Any]:
        return self._rpc("task_info")

    def budget(self) -> dict[str, Any]:
        return self._rpc("budget")

    def evaluate(
        self,
        params: dict[str, float],
        seed: int = 0,
        condition_id: str = "baseline",
        duration_scale: float = 1.0,
    ) -> dict[str, Any]:
        return self._rpc(
            "evaluate",
            {
                "params": params,
                "seed": seed,
                "condition_id": condition_id,
                "duration_scale": duration_scale,
            },
        )

    def submit_solution(self, theta_hat: dict[str, float], notes: str = "") -> dict[str, Any]:
        return self._rpc("submit_solution", {"theta_hat": theta_hat, "notes": notes})

    def abstain_insufficient_evidence(
        self, subtype: str, evidence_summary: dict[str, Any], notes: str = ""
    ) -> dict[str, Any]:
        return self._rpc(
            "abstain",
            {"subtype": subtype, "evidence_summary": evidence_summary, "notes": notes},
        )


def create_sandbox_session(
    task_dir: str | Path,
    workspace_dir: str | Path,
    disclosure: DisclosureTier = "full",
    trace_path: str | Path | None = None,
) -> TrackBSandboxServer:
    """Helper to initialize a SimulatorGateway and Track B Unix-socket sandbox server."""
    gw = SimulatorGateway.from_task_dir(
        task_dir=task_dir,
        disclosure=disclosure,
        trace_path=trace_path or (Path(workspace_dir) / "trace.jsonl"),
    )
    return TrackBSandboxServer(gateway=gw, workspace_dir=workspace_dir)
