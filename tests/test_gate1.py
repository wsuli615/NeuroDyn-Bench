"""Gate 1 unit and integration tests: isolation, 3-tier disclosure, dual-track budget, Bayesian tools, and Evaluator."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from neurodyn_bench.eval.evaluator import evaluate_submission
from neurodyn_bench.eval.metrics import (
    compute_cost_sensitivity_scan,
    compute_headline_metrics,
)
from neurodyn_bench.gateway.gateway import SimulatorGateway
from neurodyn_bench.gateway.isolation import run_isolated_sandbox_command
from neurodyn_bench.gateway.sandbox import TrackBSandboxServer
from neurodyn_bench.tasks.loader import load_private_bundle, load_public_task
from neurodyn_bench.tasks.schema import FinalSubmission
from neurodyn_bench.tools.registry import ToolRegistry

REPO_ROOT = Path(__file__).resolve().parent.parent
DEV_TASKS = REPO_ROOT / "tasks" / "dev"


def test_public_private_isolation_and_3tier_disclosure() -> None:
    t03 = DEV_TASKS / "T03_small_ei_bound_semantics"
    assert (t03 / "public" / "task.yaml").exists()

    full_bundle, full_map = load_public_task(t03, disclosure="full")
    names_bundle, names_map = load_public_task(t03, disclosure="names_only")
    blind_bundle, blind_map = load_public_task(t03, disclosure="blinded")

    assert full_map == {}
    assert names_map == {}
    assert len(blind_map) == len(full_bundle.parameters)

    # Ensure zero private keys leak into public serialization
    dumped = json.dumps(full_bundle.model_dump())
    assert "ground_truth_params" not in dumped
    assert "heldout_conditions" not in dumped
    assert "validation_cert" not in dumped

    # names_only strips parameter descriptions
    assert all(p.description == "" for p in names_bundle.parameters)
    # blinded anonymizes names to p0..pN and strips units/descriptions
    assert [p.name for p in blind_bundle.parameters] == [f"p{i}" for i in range(len(full_bundle.parameters))]
    assert all(p.unit == "" and p.description == "" for p in blind_bundle.parameters)

    # Verify blinded gateway evaluates identically to unblinded gateway on same seed
    gw_full = SimulatorGateway.from_task_dir(t03, disclosure="full", max_units_override=5.0)
    gw_blind = SimulatorGateway.from_task_dir(t03, disclosure="blinded", max_units_override=5.0)
    r_full = gw_full.evaluate(gw_full.bundle.nominal_dict(), seed=3, duration_scale=0.2)
    r_blind = gw_blind.evaluate(gw_blind.bundle.nominal_dict(), seed=3, duration_scale=0.2)
    assert r_full.feasible == r_blind.feasible
    assert r_full.rms_z == pytest.approx(r_blind.rms_z, rel=1e-12)


def test_track_b_sandbox_unix_socket_isolation_and_budget(tmp_path: Path) -> None:
    t05 = DEV_TASKS / "T05_small_ei_high_variance"
    trace_path = tmp_path / "trace.jsonl"
    gw = SimulatorGateway.from_task_dir(
        t05, disclosure="blinded", trace_path=trace_path, max_units_override=2.0
    )
    ws_dir = tmp_path / "agent_ws"

    with TrackBSandboxServer(gw, ws_dir) as srv:
        # Physical isolation: no private/ folder inside sandbox workspace
        assert not (ws_dir / "private").exists()
        assert (ws_dir / "public" / "task.yaml").exists()
        assert (ws_dir / "neurodyn_client.py").exists()

        # Run a separate Python subprocess simulating an open coding agent
        agent_script = (
            "from neurodyn_client import SandboxClient, BudgetExhaustedError\n"
            "c = SandboxClient()\n"
            "info = c.task_info()\n"
            "assert info['disclosure'] == 'blinded'\n"
            "nom = {p['name']: p['nominal'] for p in info['parameters']}\n"
            "r1 = c.evaluate(nom, seed=1, duration_scale=1.0)\n"
            "assert r1['used_units'] == 1.0\n"
            "r1_cached = c.evaluate(nom, seed=1, duration_scale=1.0)\n"
            "assert r1_cached['cache_hit'] is True and r1_cached['used_units'] == 1.0\n"
            "r2 = c.evaluate(nom, seed=2, duration_scale=1.0)\n"
            "assert r2['used_units'] == 2.0\n"
            "hit_cap = False\n"
            "try:\n"
            "    c.evaluate(nom, seed=3, duration_scale=1.0)\n"
            "except BudgetExhaustedError:\n"
            "    hit_cap = True\n"
            "assert hit_cap\n"
            "c.submit_solution(nom, notes='sandbox test')\n"
        )
        proc = subprocess.run(
            [sys.executable, "-c", agent_script],
            cwd=str(ws_dir),
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 0, f"Sandbox agent failed:\nSTDOUT:{proc.stdout}\nSTDERR:{proc.stderr}"
        assert srv.submission_path.exists()
        assert trace_path.exists()
        lines = [json.loads(ln) for ln in trace_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        assert any(ln.get("event") == "final_submission" for ln in lines)


def test_track_a_bayesian_and_diagnostic_tools_budget_accounting(tmp_path: Path) -> None:
    t05 = DEV_TASKS / "T05_small_ei_high_variance"
    trace_path = tmp_path / "tools_trace.jsonl"
    gw = SimulatorGateway.from_task_dir(t05, disclosure="full", trace_path=trace_path, max_units_override=35.0)
    reg = ToolRegistry(gw)

    tool_names = {t["name"] for t in reg.list_tools()}
    assert {"run_bo", "run_cmaes", "run_abc_smc", "run_laplace_fisher", "run_ppc_audit", "boundary_slack", "repeat_variance"}.issubset(tool_names)

    # 1. Short feasibility scan (4 x short_probe_scale)
    scan = reg.call("short_feasibility_scan", n_samples=4, seed=1)
    assert scan["units_spent"] == pytest.approx(4.0 * gw.bundle.budget.short_probe_scale, abs=0.01)

    # 2. Small BO run (6 evals)
    bo = reg.call("run_bo", n_evals=6, seed=2)
    assert bo["n_trials_completed"] == 6
    assert bo["units_spent"] == pytest.approx(6.0, abs=0.01)

    # 3. Lightweight Bayesian ABC-SMC reusing history + 4 extra particles
    abc = reg.call("run_abc_smc", n_particles=8, n_rounds=1, reuse_history=True, seed=3)
    assert abc["status"] == "ok"
    assert "posterior_contraction" in abc

    # 4. Local Laplace-Fisher eigenspectrum (d=4 -> 5 units)
    fisher = reg.call("run_laplace_fisher", seed=4)
    assert fisher["status"] == "ok"
    assert len(fisher["fisher_eigenvalues"]) == 4

    # 5. PPC audit (2 seeds)
    ppc = reg.call("run_ppc_audit", n_seeds_per_candidate=2, seed_start=9100)
    assert "coverage_rate" in ppc and "per_observable" in ppc

    # 6. Zero-cost history diagnostics
    tradeoff = reg.call("feature_tradeoff_audit")
    ms_id = reg.call("multistart_identifiability")
    assert "tradeoff_detected" in tradeoff
    assert "degeneracy_detected" in ms_id

    # Verify hard-bound violation detection
    bad = gw.bundle.nominal_dict()
    bad["g_ei"] = 999.0
    rec_bad = gw.evaluate(bad, seed=0)
    assert rec_bad.hard_bound_violation is True
    assert rec_bad.feasible is False

    # Verify all tool events logged to trace.jsonl
    trace_lines = [json.loads(ln) for ln in trace_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    logged_tools = {ln["tool_name"] for ln in trace_lines if ln.get("event") == "tool"}
    assert {"short_feasibility_scan", "run_bo", "run_abc_smc", "run_laplace_fisher", "run_ppc_audit"}.issubset(logged_tools)


def test_evaluator_anti_gaming_and_metrics(tmp_path: Path) -> None:
    t11 = DEV_TASKS / "T11_small_ei_cbn_vs_sbn_3b"
    t03 = DEV_TASKS / "T03_small_ei_bound_semantics"

    # Correct 3b abstention with evidence
    sub_3b_ok = FinalSubmission(
        decision="abstain_insufficient_evidence",
        abstention_subtype="3b_structure_unreachable",
        evidence_summary={"tradeoff_detected": True, "conflicting_pairs": [["fr_e", "pct_sh"]]},
    )
    rep_3b_ok = evaluate_submission(t11, sub_3b_ok)
    assert rep_3b_ok.verified_pass is True
    assert rep_3b_ok.abstention_correct is True

    # Wrong subtype (3a instead of 3b)
    sub_3b_wrong = FinalSubmission(
        decision="abstain_insufficient_evidence",
        abstention_subtype="3a_non_identifiable",
        evidence_summary={"degeneracy_detected": True},
    )
    rep_3b_wrong = evaluate_submission(t11, sub_3b_wrong)
    assert rep_3b_wrong.verified_pass is False
    assert rep_3b_wrong.abstention_subtype_correct is False

    # False abstention on solvable Type 2 (T03)
    rep_false_abs = evaluate_submission(t03, sub_3b_ok)
    assert rep_false_abs.verified_pass is False
    assert rep_false_abs.false_abstention is True

    # Hard-bound violation on submit
    pub_t03, _ = load_public_task(t03)
    bad_theta = pub_t03.nominal_dict()
    bad_theta["tau_e"] = 0.01  # below phys_hard_lb = 3.0
    rep_hb = evaluate_submission(t03, FinalSubmission(decision="submit_solution", theta_hat=bad_theta))
    assert rep_hb.verified_pass is False
    assert rep_hb.hard_bound_violated_on_submit is True

    # Headline metrics & cost-sensitivity scan sanity
    hm = compute_headline_metrics([rep_3b_ok.to_dict(), rep_false_abs.to_dict()])
    assert hm["abstention_precision"] == pytest.approx(0.5)
    assert hm["abstention_recall"] == pytest.approx(1.0)
    assert hm["false_abstention_rate_on_solvable"] == pytest.approx(1.0)

    scan = compute_cost_sensitivity_scan(used_units=100.0, llm_wall_time_s=20.0)
    assert len(scan) == 6
    assert scan[0]["llm_share"] > scan[-1]["llm_share"]


def test_offline_seccomp_blocks_inet_but_allows_unix_gateway(tmp_path: Path) -> None:
    t05 = DEV_TASKS / "T05_small_ei_high_variance"
    trace_path = tmp_path / "offline_trace.jsonl"
    gw = SimulatorGateway.from_task_dir(
        t05, disclosure="full", trace_path=trace_path, max_units_override=3.0
    )
    ws_dir = tmp_path / "offline_ws"

    with TrackBSandboxServer(gw, ws_dir) as srv:
        probe_code = (
            "import os, socket, subprocess, sys\n"
            "from neurodyn_client import SandboxClient\n"
            "# 1. Sensitive host env vars must be stripped\n"
            "assert 'GITHUB_TOKEN' not in os.environ and 'OPENAI_API_KEY' not in os.environ\n"
            "# 2. IPv4 and IPv6 sockets must be blocked at kernel level (EPERM)\n"
            "for fam in (socket.AF_INET, socket.AF_INET6):\n"
            "    blocked = False\n"
            "    try:\n"
            "        s = socket.socket(fam, socket.SOCK_STREAM)\n"
            "        s.close()\n"
            "    except PermissionError:\n"
            "        blocked = True\n"
            "    assert blocked, f'Expected {fam} to be blocked by seccomp-bpf'\n"
            "# 3. Nested child subprocess must inherit seccomp-bpf filter\n"
            "child = subprocess.run(\n"
            "    [sys.executable, '-c', 'import socket; socket.socket(socket.AF_INET, socket.SOCK_STREAM)'],\n"
            "    capture_output=True, text=True,\n"
            ")\n"
            "assert child.returncode != 0 and 'PermissionError' in child.stderr\n"
            "# 4. AF_UNIX gateway proxy must still work normally\n"
            "c = SandboxClient()\n"
            "info = c.task_info()\n"
            "nom = {p['name']: p['nominal'] for p in info['parameters']}\n"
            "r = c.evaluate(nom, seed=1, duration_scale=1.0)\n"
            "assert r['used_units'] == 1.0\n"
            "c.submit_solution(nom, notes='offline seccomp test')\n"
        )
        audit = run_isolated_sandbox_command(
            srv,
            [sys.executable, "-c", probe_code],
            timeout_s=60.0,
            backend="seccomp",
            enable_strace_audit=True,
        )
        assert audit.returncode == 0, f"Offline sandbox failed:\nSTDOUT:{audit.stdout}\nSTDERR:{audit.stderr}"
        assert audit.seccomp_active is True
        assert audit.network_escapes_succeeded == []
        assert audit.path_violations == []
        assert audit.security_violated is False


def test_strace_catches_private_peek_and_evaluator_invalidates(tmp_path: Path) -> None:
    t02 = DEV_TASKS / "T02_small_ei_routine"
    priv = load_private_bundle(t02)
    gt_file = (t02 / "private" / "ground_truth.yaml").resolve()
    trace_path = tmp_path / "cheat_trace.jsonl"
    gw = SimulatorGateway.from_task_dir(
        t02, disclosure="full", trace_path=trace_path, max_units_override=5.0
    )
    ws_dir = tmp_path / "cheat_ws"

    with TrackBSandboxServer(gw, ws_dir) as srv:
        # Simulate a cheating agent that peeks at private/ground_truth.yaml and submits exact GT
        cheat_code = (
            "from pathlib import Path\n"
            "from neurodyn_client import SandboxClient\n"
            f"_ = Path({str(gt_file)!r}).read_text()\n"
            f"SandboxClient().submit_solution({priv.ground_truth_params!r}, notes='cheated')\n"
        )
        audit = run_isolated_sandbox_command(
            srv,
            [sys.executable, "-c", cheat_code],
            timeout_s=60.0,
            backend="seccomp",
            enable_strace_audit=True,
        )
        assert audit.returncode == 0
        assert audit.strace_active is True
        assert audit.security_violated is True
        assert any("ground_truth.yaml" in p for p in audit.path_violations)

        # Even though submitted theta_hat is exact ground truth, Evaluator MUST invalidate the run
        rep = evaluate_submission(t02, srv.submission_path, trace_path=trace_path)
        assert rep.verified_pass is False
        assert rep.security_violated is True
        assert rep.process_audit["n_security_violations"] >= 1
