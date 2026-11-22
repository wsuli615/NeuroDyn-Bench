"""Claude CLI LLM agents for Track A (G3 General Tool, G4 Diagnostic) and Track B (Offline Sandbox).

Invokes the locally authenticated `claude` CLI in strict non-interactive, toolless, restricted
mode (`--print --tools "" --restricted --no-session-persistence --output-format json`), ensuring
the LLM itself has zero shell, filesystem, or web-search access while logging exact token,
thinking-token, USD cost, and API latency accounting per Rule 3.2.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from neurodyn_bench.gateway.gateway import SimulatorGateway
from neurodyn_bench.gateway.isolation import run_isolated_sandbox_command
from neurodyn_bench.gateway.sandbox import TrackBSandboxServer
from neurodyn_bench.tasks.schema import DisclosureTier, FinalSubmission
from neurodyn_bench.tools.registry import ToolRegistry

_TRACK_A_ACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "action_type": {
            "type": "string",
            "enum": ["call_tool", "submit_solution", "abstain_insufficient_evidence"],
        },
        "hypothesis": {"type": "string"},
        "evidence_refs": {"type": "array", "items": {"type": "string"}},
        "tool_name": {"type": "string"},
        "tool_args_json": {
            "type": "string",
            "description": "JSON-encoded object of keyword arguments for tool_name (e.g. '{\"n_evals\": 40}').",
        },
        "theta_hat_json": {
            "type": "string",
            "description": "JSON-encoded object mapping parameter names to float values (for submit_solution).",
        },
        "abstention_subtype": {
            "type": "string",
            "enum": ["3a_non_identifiable", "3b_structure_unreachable", ""],
        },
        "evidence_summary_json": {
            "type": "string",
            "description": "JSON-encoded quantitative evidence object supporting abstention or submission.",
        },
        "notes": {"type": "string"},
    },
    "required": ["action_type", "hypothesis", "notes"],
}

_TRACK_B_CODE_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "python_code": {
            "type": "string",
            "description": "Complete executable Python script using `from neurodyn_client import SandboxClient`.",
        },
    },
    "required": ["reasoning", "python_code"],
}


@dataclass
class LLMUsageTracker:
    """Separate LLM token, cost, and latency accounting (Rule 3.2)."""

    model_alias: str
    effort: str
    canonical_model: str = ""
    n_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    thinking_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    total_cost_usd: float = 0.0
    total_api_wall_s: float = 0.0
    web_search_requests: int = 0
    per_round: list[dict[str, Any]] = field(default_factory=list)

    def record(self, raw_json: dict[str, Any], wall_s: float) -> dict[str, Any]:
        self.n_calls += 1
        usage = raw_json.get("usage", {})
        in_tok = int(usage.get("input_tokens", 0))
        out_tok = int(usage.get("output_tokens", 0))
        think_tok = int(usage.get("output_tokens_details", {}).get("thinking_tokens", 0))
        c_read = int(usage.get("cache_read_input_tokens", 0))
        c_create = int(usage.get("cache_creation_input_tokens", 0))
        cost = float(raw_json.get("total_cost_usd", 0.0))
        web_req = int(usage.get("server_tool_use", {}).get("web_search_requests", 0))
        api_s = float(raw_json.get("duration_api_ms", wall_s * 1000.0)) / 1000.0

        model_usage = raw_json.get("modelUsage", {})
        if model_usage and not self.canonical_model:
            first_key = next(iter(model_usage))
            self.canonical_model = str(model_usage[first_key].get("canonicalModel", first_key))

        self.input_tokens += in_tok
        self.output_tokens += out_tok
        self.thinking_tokens += think_tok
        self.cache_read_tokens += c_read
        self.cache_creation_tokens += c_create
        self.total_cost_usd += cost
        self.total_api_wall_s += api_s
        self.web_search_requests += web_req

        entry = {
            "call_idx": self.n_calls,
            "canonical_model": self.canonical_model or self.model_alias,
            "effort": self.effort,
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "thinking_tokens": think_tok,
            "cache_read_tokens": c_read,
            "cost_usd": round(cost, 6),
            "api_wall_s": round(api_s, 3),
            "web_search_requests": web_req,
        }
        self.per_round.append(entry)
        return entry

    def summary(self) -> dict[str, Any]:
        return {
            "model_alias": self.model_alias,
            "canonical_model": self.canonical_model or self.model_alias,
            "effort": self.effort,
            "n_calls": self.n_calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "thinking_tokens": self.thinking_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_creation_tokens": self.cache_creation_tokens,
            "total_cost_usd": round(self.total_cost_usd, 6),
            "total_api_wall_s": round(self.total_api_wall_s, 3),
            "web_search_requests": self.web_search_requests,
        }


def call_claude_structured(
    prompt: str,
    system_prompt: str,
    json_schema: dict[str, Any],
    tracker: LLMUsageTracker,
    model: str = "opus",
    effort: str = "max",
    timeout_s: float = 900.0,
) -> dict[str, Any]:
    """Invoke local `claude` CLI in strict offline/toolless structured-JSON mode."""
    cmd = [
        "claude",
        "-p",
        "--model",
        model,
        "--effort",
        effort,
        "--output-format",
        "json",
        "--tools",
        "",
        "--no-session-persistence",
        "--restricted",
        "--system-prompt",
        system_prompt,
        "--json-schema",
        json.dumps(json_schema),
        prompt,
    ]
    t0 = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd="/tmp",  # Never execute inside the repository directory
        capture_output=True,
        text=True,
        timeout=timeout_s,
        check=False,
    )
    wall_s = time.perf_counter() - t0
    if proc.returncode != 0:
        raise RuntimeError(
            f"claude CLI failed (rc={proc.returncode}): stderr={proc.stderr[:500]} stdout={proc.stdout[:500]}"
        )
    raw = json.loads(proc.stdout)
    if raw.get("is_error"):
        raise RuntimeError(f"claude CLI returned error: {raw.get('result')}")

    tracker.record(raw, wall_s)
    if "structured_output" in raw and isinstance(raw["structured_output"], dict):
        return raw["structured_output"]
    return json.loads(raw["result"])


def call_claude_text(
    prompt: str,
    system_prompt: str,
    tracker: LLMUsageTracker,
    model: str = "opus",
    effort: str = "max",
    timeout_s: float = 600.0,
) -> str:
    """Invoke local `claude` CLI in offline/toolless plain-text mode (ideal for Track B code)."""
    cmd = [
        "claude",
        "-p",
        "--model",
        model,
        "--effort",
        effort,
        "--output-format",
        "json",
        "--tools",
        "",
        "--no-session-persistence",
        "--restricted",
        "--system-prompt",
        system_prompt,
        prompt,
    ]
    t0 = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd="/tmp",
        capture_output=True,
        text=True,
        timeout=timeout_s,
        check=False,
    )
    wall_s = time.perf_counter() - t0
    if proc.returncode != 0:
        raise RuntimeError(
            f"claude CLI failed (rc={proc.returncode}): stderr={proc.stderr[:500]} stdout={proc.stdout[:500]}"
        )
    raw = json.loads(proc.stdout)
    if raw.get("is_error"):
        raise RuntimeError(f"claude CLI returned error: {raw.get('result')}")
    tracker.record(raw, wall_s)
    return str(raw.get("result", ""))


def _extract_python_code(text: str) -> str:
    """Extract the largest fenced Python block (or raw text) from an LLM response."""
    import re

    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if blocks:
        return max(blocks, key=len).strip() + "\n"
    return text.strip() + "\n"


def _safe_parse_json_obj(text: str | None) -> dict[str, Any]:
    if not text or not text.strip():
        return {}
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {"value": parsed}
    except json.JSONDecodeError:
        return {}


def _build_diagnostic_dashboard(gateway: SimulatorGateway) -> dict[str, Any]:
    """Compute a structured diagnostic summary strictly from public gateway.history (for G4)."""
    hist = gateway.history
    n_tot = len(hist)
    feas = [r for r in hist if r.feasible and r.condition_id == "baseline" and math.isfinite(r.rms_z)]
    infeas = [r for r in hist if not r.feasible]
    fail_counts: dict[str, int] = {}
    for r in infeas:
        reason = r.failure_reason or "unknown"
        fail_counts[reason] = fail_counts.get(reason, 0) + 1

    best = gateway.best_feasible_record()
    active_rows = gateway.bundle.active_parameters()
    boundary_flags = []
    if best is not None:
        for row in active_rows:
            val = best.params[row.name]
            span = max(row.default_ub - row.default_lb, 1e-12)
            if (val - row.default_lb) / span <= 0.05:
                boundary_flags.append(
                    {
                        "param": row.name,
                        "near": "default_lb",
                        "val": round(val, 5),
                        "default_lb": row.default_lb,
                        "phys_hard_lb": row.phys_hard_lb,
                        "legal_slack": row.default_lb > row.phys_hard_lb + 1e-9,
                    }
                )
            elif (row.default_ub - val) / span <= 0.05:
                boundary_flags.append(
                    {
                        "param": row.name,
                        "near": "default_ub",
                        "val": round(val, 5),
                        "default_ub": row.default_ub,
                        "phys_hard_ub": row.phys_hard_ub,
                        "legal_slack": row.default_ub < row.phys_hard_ub - 1e-9,
                    }
                )

    low_loss = [r for r in feas if r.rms_z <= 2.0]
    recent_best = [round(r.rms_z, 3) for r in feas[-8:]]
    return {
        "n_evals_total": n_tot,
        "n_feasible": len(feas),
        "feasible_fraction": round(len(feas) / max(n_tot, 1), 3),
        "failure_breakdown": fail_counts,
        "n_low_loss_rms_le_2": len(low_loss),
        "best_rms_z": round(best.rms_z, 4) if best else None,
        "best_max_abs_z": round(best.max_abs_z, 4) if best else None,
        "best_z_scores": {k: round(v, 3) for k, v in best.z_scores.items()} if best else {},
        "best_active_params": {r.name: round(best.params[r.name], 6) for r in active_rows}
        if best
        else None,
        "boundary_saturation_vs_default_box": boundary_flags,
        "recent_feasible_rms_z": recent_best,
        "budget": gateway.ledger.summary(),
    }


def _coerce_tool_kwargs(tool_name: str, raw_kwargs: dict[str, Any]) -> dict[str, Any]:
    """Sanitize LLM-provided JSON arguments (e.g., list->tuple bounds) for ToolRegistry."""
    kwargs = dict(raw_kwargs)
    if "bounds" in kwargs and isinstance(kwargs["bounds"], dict):
        clean_bounds = {}
        for k, v in kwargs["bounds"].items():
            if isinstance(v, (list, tuple)) and len(v) == 2:
                clean_bounds[k] = (float(v[0]), float(v[1]))
        kwargs["bounds"] = clean_bounds
    if "candidates" in kwargs and isinstance(kwargs["candidates"], dict):
        kwargs["candidates"] = [kwargs["candidates"]]
    # Prevent unexpected keys from crashing the loop
    allowed_by_tool: dict[str, set[str]] = {
        "run_bo": {
            "n_evals",
            "active_params",
            "bounds",
            "scales",
            "base_params",
            "n_repeats",
            "short_screen",
            "adaptive_repeat_threshold",
            "seed",
        },
        "run_cmaes": {
            "n_evals",
            "active_params",
            "bounds",
            "scales",
            "base_params",
            "init_params",
            "n_repeats",
            "short_screen",
            "adaptive_repeat_threshold",
            "popsize",
            "sigma0",
            "seed",
        },
        "run_multistart_local": {
            "n_starts",
            "evals_per_start",
            "active_params",
            "bounds",
            "scales",
            "base_params",
            "seed",
        },
        "run_abc_smc": {
            "n_particles",
            "n_rounds",
            "active_params",
            "bounds",
            "scales",
            "base_params",
            "reuse_history",
            "temperature",
            "seed",
        },
        "run_laplace_fisher": {
            "center_params",
            "active_params",
            "bounds",
            "scales",
            "step_frac",
            "central_diff",
            "seed",
        },
        "run_ppc_audit": {"candidates", "n_seeds_per_candidate", "seed_start"},
        "boundary_slack": {
            "candidate_params",
            "active_params",
            "bounds",
            "edge_tol",
            "probe_outside",
            "seed",
        },
        "repeat_variance": {"candidate_params", "n_repeats", "seed_start"},
        "short_feasibility_scan": {"n_samples", "active_params", "bounds", "scales", "seed"},
        "sensitivity_screen": {
            "center_params",
            "active_params",
            "bounds",
            "step_frac",
            "use_short_probe",
            "seed",
        },
        "feature_tradeoff_audit": set(),
        "multistart_identifiability": {"active_params", "bounds", "rms_z_threshold", "min_param_sep"},
    }
    if tool_name in allowed_by_tool:
        allowed = allowed_by_tool[tool_name]
        kwargs = {k: v for k, v in kwargs.items() if k in allowed}
    return kwargs


def _run_track_a_claude(
    gateway: SimulatorGateway,
    mode: str,
    model: str = "opus",
    effort: str = "max",
    max_rounds: int = 10,
    seed: int = 0,
) -> tuple[FinalSubmission, LLMUsageTracker]:
    """Shared Track A multi-round loop for G3 (general tool) and G4 (diagnostic protocol)."""
    registry = ToolRegistry(gateway)
    tracker = LLMUsageTracker(model_alias=model, effort=effort)
    pub_info = gateway.public_task_info()

    if mode == "g3":
        system_prompt = (
            "You are a scientific parameter calibration agent. You are given a fixed-structure "
            "neural dynamics calibration problem and a set of optional tools billed against a hard "
            "simulation budget. At each round, choose either `call_tool`, `submit_solution` "
            "(providing `theta_hat_json` with active parameter values), or "
            "`abstain_insufficient_evidence` (providing `abstention_subtype` as either "
            "'3a_non_identifiable' or '3b_structure_unreachable' plus `evidence_summary_json`). "
            "Respect physiological hard bounds (`phys_hard_lb`, `phys_hard_ub`) and never exceed "
            "the remaining simulation budget. Output valid JSON matching the required schema."
        )
    else:
        system_prompt = (
            "You are a disciplined Diagnostic Calibration Agent for mechanistic neural dynamics models.\n"
            "Core protocol (Symptom -> Discriminative Probe -> Action / Honest Abstention):\n"
            "1. Do NOT waste budget on unnecessary probes if the task is well-posed (e.g. <= 6 active "
            "parameters, clean bounds, low noise); start with an efficient optimizer (CMA-ES or BO) "
            "and verify stability before submitting.\n"
            "2. If active dimension is high (d > 8) or positive parameters span wide scales with linear "
            "default scale, run cheap `sensitivity_screen(use_short_probe=True)` first to freeze sloppy "
            "dimensions and switch wide positive parameters to `log` scale.\n"
            "3. When optimization stalls at high error (best_max_abs_z > 2.0), NEVER guess blindly or "
            "abstain prematurely. First run cheap discriminative probes:\n"
            "   - `boundary_slack`: distinguish arbitrary `default_lb/ub` (expandable within `phys_hard`) "
            "from immovable `phys_hard_lb/ub`.\n"
            "   - `repeat_variance`: detect if trial-to-trial stochasticity swamps single-eval scores "
            "(fix with `n_repeats` / `adaptive_repeat_threshold`).\n"
            "   - `feature_tradeoff_audit` + `run_ppc_audit` + wide search over `phys_hard` bounds: "
            "abstain as `3b_structure_unreachable` ONLY if features remain irreconcilably conflicting "
            "after checking legal bound expansion and repeat noise.\n"
            "4. When training error is LOW (best_max_abs_z <= 2.0), check whether the solution could be "
            "underdetermined / degenerate (especially when #observables <= #active_params or summary "
            "statistics are heavily compressed): run zero-cost `multistart_identifiability` and/or "
            "budget-aware Bayesian tools (`run_abc_smc(reuse_history=True)` for posterior contraction, "
            "`run_laplace_fisher` for Fisher rank deficiency). If multiple well-separated parameter sets "
            "fit equally well and parameters cannot be uniquely identified for unseen perturbations, "
            "abstain honestly as `3a_non_identifiable` with the tool metrics in `evidence_summary_json`.\n"
            "5. Before `submit_solution` on solvable tasks, confirm top candidate stability with fresh "
            "repeats (`run_ppc_audit` or `repeat_variance`) if budget permits.\n"
            "Always state an explicit `hypothesis` and cite `evidence_refs` from prior tool outputs."
        )

    round_log: list[dict[str, Any]] = []
    tools_catalog = registry.list_tools()

    for rnd in range(1, max_rounds + 1):
        rem = gateway.ledger.remaining_units
        state_payload: dict[str, Any] = {
            "round": rnd,
            "max_rounds": max_rounds,
            "task": {
                "task_id": pub_info["task_id"],
                "title": pub_info["title"],
                "model_type": pub_info["model_type"],
                "disclosure": pub_info["disclosure"],
                "prompt": pub_info["prompt"],
                "parameters": [p for p in pub_info["parameters"] if p["active"]],
                "fixed_parameters": [
                    {"name": p["name"], "nominal": p["nominal"], "unit": p["unit"]}
                    for p in pub_info["parameters"]
                    if not p["active"]
                ],
                "observables": pub_info["observables"],
                "conditions": pub_info["conditions"],
                "short_probe_scale": pub_info["short_probe_scale"],
            },
            "budget": gateway.ledger.summary(),
            "available_tools": tools_catalog,
            "prior_rounds": round_log[-6:],
        }

        if mode == "g4":
            state_payload["diagnostic_dashboard"] = _build_diagnostic_dashboard(gateway)
        else:
            # G3 receives raw best record + simple evaluation count only (no diagnostic dashboard)
            best = gateway.best_feasible_record()
            state_payload["raw_progress"] = {
                "n_evals": len(gateway.history),
                "best_rms_z": round(best.rms_z, 4) if best else None,
                "best_z_scores": {k: round(v, 3) for k, v in best.z_scores.items()} if best else {},
                "best_params": {
                    r.name: round(best.params[r.name], 6)
                    for r in gateway.bundle.active_parameters()
                }
                if best
                else None,
            }

        if rem < 1.0 or rnd == max_rounds:
            state_payload["instruction_override"] = (
                "FINAL ROUND OR BUDGET NEARLY EXHAUSTED: You MUST now choose either "
                "`submit_solution` or `abstain_insufficient_evidence` (zero-cost analysis tools "
                "like `feature_tradeoff_audit`, `multistart_identifiability`, or "
                "`run_abc_smc(n_particles=0, n_rounds=1, reuse_history=True)` are still allowed if round < max_rounds)."
            )

        user_prompt = (
            "Current calibration state (JSON):\n"
            + json.dumps(state_payload, indent=2, ensure_ascii=False)
            + "\n\nChoose your next action."
        )

        action = call_claude_structured(
            prompt=user_prompt,
            system_prompt=system_prompt,
            json_schema=_TRACK_A_ACTION_SCHEMA,
            tracker=tracker,
            model=model,
            effort=effort,
        )
        gateway._append_trace(
            {
                "event": "llm_call",
                "mode": mode,
                "round_id": rnd,
                "llm_usage": tracker.per_round[-1],
                "action": action,
            }
        )

        act_type = action["action_type"]
        hyp = str(action.get("hypothesis", ""))

        if act_type == "submit_solution":
            theta_raw = _safe_parse_json_obj(action.get("theta_hat_json"))
            best = gateway.best_feasible_record()
            theta = gateway.bundle.nominal_dict()
            if best is not None:
                theta.update(best.params)
            for k, v in theta_raw.items():
                if k in theta:
                    try:
                        theta[k] = float(v)
                    except (TypeError, ValueError):
                        pass
            ev_sum = _safe_parse_json_obj(action.get("evidence_summary_json"))
            ev_sum["llm_usage"] = tracker.summary()
            return (
                FinalSubmission(
                    decision="submit_solution",
                    theta_hat=theta,
                    evidence_summary=ev_sum,
                    notes=str(action.get("notes", f"{mode} claude submission")),
                ),
                tracker,
            )

        if act_type == "abstain_insufficient_evidence":
            subtype = action.get("abstention_subtype") or "3a_non_identifiable"
            if subtype not in ("3a_non_identifiable", "3b_structure_unreachable"):
                subtype = "3a_non_identifiable"
            ev_sum = _safe_parse_json_obj(action.get("evidence_summary_json"))
            if not ev_sum:
                ev_sum = {"hypothesis": hyp, "evidence_refs": action.get("evidence_refs", [])}
            ev_sum["llm_usage"] = tracker.summary()
            return (
                FinalSubmission(
                    decision="abstain_insufficient_evidence",
                    abstention_subtype=subtype,  # type: ignore[arg-type]
                    evidence_summary=ev_sum,
                    notes=str(action.get("notes", f"{mode} claude abstention")),
                ),
                tracker,
            )

        # Tool call
        tname = str(action.get("tool_name", "")).strip()
        raw_args = _safe_parse_json_obj(action.get("tool_args_json"))
        clean_args = _coerce_tool_kwargs(tname, raw_args)
        # Ensure deterministic distinct seeds if LLM omits seed
        if "seed" not in clean_args and tname in (
            "run_bo",
            "run_cmaes",
            "run_multistart_local",
            "run_abc_smc",
            "run_laplace_fisher",
            "boundary_slack",
            "short_feasibility_scan",
            "sensitivity_screen",
        ):
            clean_args["seed"] = int(seed) * 100 + rnd * 13

        try:
            tool_out = registry.call(tname, hypothesis=hyp, **clean_args)
            # Strip full_best_params bulk if large; keep compact summary for LLM context
            compact_out = {k: v for k, v in tool_out.items() if k != "full_best_params"}
        except Exception as exc:  # noqa: BLE001
            compact_out = {"error": f"{type(exc).__name__}: {exc}"}

        round_log.append(
            {
                "round": rnd,
                "tool_name": tname,
                "args": clean_args,
                "hypothesis": hyp,
                "result": compact_out,
            }
        )

    # Fallback if max_rounds reached without explicit terminal action
    best = gateway.best_feasible_record()
    theta = best.params if best is not None else gateway.bundle.nominal_dict()
    return (
        FinalSubmission(
            decision="submit_solution",
            theta_hat=theta,
            evidence_summary={"llm_usage": tracker.summary(), "fallback_max_rounds": True},
            notes=f"{mode} fallback submit after max_rounds={max_rounds}",
        ),
        tracker,
    )


def run_g3_claude(
    gateway: SimulatorGateway,
    model: str = "opus",
    effort: str = "max",
    max_rounds: int = 10,
    seed: int = 0,
) -> tuple[FinalSubmission, LLMUsageTracker]:
    """G3 General Tool Agent powered by Claude CLI (no structured diagnostic protocol)."""
    return _run_track_a_claude(
        gateway, mode="g3", model=model, effort=effort, max_rounds=max_rounds, seed=seed
    )


def run_g4_claude(
    gateway: SimulatorGateway,
    model: str = "opus",
    effort: str = "max",
    max_rounds: int = 12,
    seed: int = 0,
) -> tuple[FinalSubmission, LLMUsageTracker]:
    """G4 Diagnostic Calibration Agent powered by Claude CLI (structured dashboard + protocol)."""
    return _run_track_a_claude(
        gateway, mode="g4", model=model, effort=effort, max_rounds=max_rounds, seed=seed
    )


def run_track_b_claude(
    task_dir: str | Path,
    workspace_dir: str | Path,
    model: str = "opus",
    effort: str = "max",
    max_rounds: int = 4,
    disclosure: DisclosureTier = "full",
    trace_path: str | Path | None = None,
    timeout_per_script_s: float = 600.0,
) -> tuple[FinalSubmission, LLMUsageTracker, dict[str, Any]]:
    """Track B Open Offline Python Sandbox Agent powered by Claude CLI + seccomp-bpf/strace."""
    ws = Path(workspace_dir)
    tpath = Path(trace_path) if trace_path else (ws / "trace.jsonl")
    gw = SimulatorGateway.from_task_dir(task_dir, disclosure=disclosure, trace_path=tpath)
    tracker = LLMUsageTracker(model_alias=model, effort=effort)
    pub_info = gw.public_task_info()

    system_prompt = (
        "You are an autonomous computational neuroscience coding agent operating in an OFFLINE, "
        "sandboxed Python workspace.\n"
        "Rules & Environment:\n"
        "- Outbound network access is blocked at the kernel level (seccomp-bpf); only `SandboxClient` "
        "over local AF_UNIX works.\n"
        "- Reading files outside the sandbox directory is monitored by strace and IMMEDIATELY invalidates the run.\n"
        "- Installed libraries: `numpy`, `scipy`, `scikit-learn`, `cma`, `optuna`, `pyyaml`.\n"
        "- Import `from neurodyn_client import SandboxClient, BudgetExhaustedError`.\n"
        "- Client methods:\n"
        "  * `c.task_info()` -> public metadata, active/fixed parameters with default & phys_hard bounds, observables, budget\n"
        "  * `c.budget()` -> current used_units and remaining_units\n"
        "  * `c.evaluate(params_dict, seed=0, condition_id='baseline', duration_scale=1.0)` -> returns dict with `feasible`, `observables`, `z_scores`, `rms_z`, `max_abs_z`, `used_units`, `remaining_units`\n"
        "  * `c.evaluate_repeats(params_dict, seeds=[...], duration_scale=1.0)` -> repeated seeds summary\n"
        "  * `c.log_note(tool_name, summary_dict, hypothesis='')` -> log your diagnostic/Bayesian audit findings into trace.jsonl\n"
        "  * `c.submit_solution(theta_hat_dict, notes='')`\n"
        "  * `c.abstain_insufficient_evidence(subtype, evidence_summary_dict, notes='')` where subtype is `'3a_non_identifiable'` or `'3b_structure_unreachable'`.\n"
        "Design your own calibration, diagnostic, and uncertainty/Bayesian analysis script freely. "
        "Keep the script focused and concise (~80-150 lines) and print key intermediate findings to stdout. "
        "Verify candidate generalization across fresh seeds; if parameters are non-identifiable (3a) "
        "or the fixed architecture cannot match all observables (3b), log quantitative evidence via "
        "`c.log_note(...)` and abstain cleanly."
    )

    last_feedback: dict[str, Any] = {}
    audit_history: list[dict[str, Any]] = []

    with TrackBSandboxServer(gw, ws) as srv:
        for rnd in range(1, max_rounds + 1):
            payload = {
                "round": rnd,
                "max_rounds": max_rounds,
                "task": {
                    "task_id": pub_info["task_id"],
                    "title": pub_info["title"],
                    "model_type": pub_info["model_type"],
                    "disclosure": pub_info["disclosure"],
                    "prompt": pub_info["prompt"],
                    "parameters": [p for p in pub_info["parameters"] if p["active"]],
                    "fixed_parameters": [
                        {"name": p["name"], "nominal": p["nominal"], "unit": p["unit"]}
                        for p in pub_info["parameters"]
                        if not p["active"]
                    ],
                    "observables": pub_info["observables"],
                    "short_probe_scale": pub_info["short_probe_scale"],
                },
                "budget": gw.ledger.summary(),
                "previous_execution": last_feedback,
            }
            user_prompt = (
                "Write a self-contained Python script `solve.py` to solve (or diagnose & abstain on) "
                "this task using `SandboxClient`. Output a brief rationale followed by a single "
                "```python ... ``` code block.\n\n"
                + json.dumps(payload, indent=2, ensure_ascii=False)
            )
            raw_reply = call_claude_text(
                prompt=user_prompt,
                system_prompt=system_prompt,
                tracker=tracker,
                model=model,
                effort=effort,
            )
            code_str = _extract_python_code(raw_reply)
            gw._append_trace(
                {
                    "event": "llm_call",
                    "mode": "track_b",
                    "round_id": rnd,
                    "llm_usage": tracker.per_round[-1],
                    "code_chars": len(code_str),
                }
            )
            script_path = srv.workspace_dir / f"solve_r{rnd}.py"
            script_path.write_text(code_str, encoding="utf-8")

            audit = run_isolated_sandbox_command(
                srv,
                [sys.executable, str(script_path.name)],
                timeout_s=timeout_per_script_s,
                backend="seccomp",
                enable_strace_audit=True,
            )
            audit_history.append(audit.to_dict())

            if srv.submission_path.exists():
                raw_sub = json.loads(srv.submission_path.read_text(encoding="utf-8"))
                sub = FinalSubmission(
                    **{k: v for k, v in raw_sub.items() if k in FinalSubmission.model_fields}
                )
                return sub, tracker, {"audits": audit_history, "submission_path": str(srv.submission_path)}

            last_feedback = {
                "returncode": audit.returncode,
                "timed_out": audit.timed_out,
                "stdout_tail": audit.stdout[-2500:],
                "stderr_tail": audit.stderr[-1500:],
                "remaining_budget": gw.ledger.summary(),
            }
            if gw.exhausted:
                break

        # Fallback if no submission written
        best = gw.best_feasible_record()
        theta = best.params if best is not None else gw.bundle.nominal_dict()
        sub = FinalSubmission(
            decision="submit_solution",
            theta_hat=theta,
            evidence_summary={"llm_usage": tracker.summary(), "fallback": True},
            notes="Track B fallback submit",
        )
        gw.save_submission(sub, srv.submission_path)
        return sub, tracker, {"audits": audit_history, "submission_path": str(srv.submission_path)}
