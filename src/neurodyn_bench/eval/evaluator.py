"""Independent Evaluator with exclusive private/ access and trace.jsonl auditing."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from neurodyn_bench.tasks.loader import (
    apply_condition,
    instantiate_simulator,
    load_private_bundle,
    load_public_task,
)
from neurodyn_bench.tasks.schema import DisclosureTier, FinalSubmission, ObservableRow

_VALID_3A_TOOLS = {"run_abc_smc", "run_laplace_fisher", "multistart_identifiability", "sensitivity_screen"}
_VALID_3B_TOOLS = {"feature_tradeoff_audit", "run_ppc_audit", "boundary_slack", "short_feasibility_scan"}


@dataclass
class EvaluationReport:
    """Complete disaggregated evaluation report for one task run."""

    task_id: str
    true_task_type: str
    decision: str
    verified_pass: bool
    public_pass: bool | None = None
    heldout_pass: bool | None = None
    abstention_correct: bool | None = None
    abstention_subtype_correct: bool | None = None
    abstention_evidence_valid: bool | None = None
    false_abstention: bool = False
    false_confidence: bool = False
    hard_bound_violated_on_submit: bool = False
    public_z_scores: dict[str, float] = field(default_factory=dict)
    heldout_z_scores: dict[str, float] = field(default_factory=dict)
    used_units: float = 0.0
    max_units: float = 0.0
    process_audit: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "true_task_type": self.true_task_type,
            "decision": self.decision,
            "verified_pass": self.verified_pass,
            "public_pass": self.public_pass,
            "heldout_pass": self.heldout_pass,
            "abstention_correct": self.abstention_correct,
            "abstention_subtype_correct": self.abstention_subtype_correct,
            "abstention_evidence_valid": self.abstention_evidence_valid,
            "false_abstention": self.false_abstention,
            "false_confidence": self.false_confidence,
            "hard_bound_violated_on_submit": self.hard_bound_violated_on_submit,
            "public_z_scores": self.public_z_scores,
            "heldout_z_scores": self.heldout_z_scores,
            "used_units": round(self.used_units, 4),
            "max_units": round(self.max_units, 4),
            "process_audit": self.process_audit,
        }


def _audit_trace(trace_path: str | Path | None) -> dict[str, Any]:
    if trace_path is None or not Path(trace_path).exists():
        return {
            "trace_present": False,
            "n_evals": 0,
            "n_hard_bound_violations": 0,
            "tools_used": [],
            "used_units": 0.0,
            "sim_wall_time_s": 0.0,
        }
    n_evals = 0
    n_hb = 0
    tools_used: set[str] = set()
    used_units = 0.0
    sim_wall = 0.0
    for line in Path(trace_path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        ev = rec.get("event")
        if ev == "eval":
            n_evals += 1
            if rec.get("hard_bound_violation"):
                n_hb += 1
            used_units = max(used_units, float(rec.get("used_units", 0.0)))
            sim_wall += float(rec.get("wall_time_s", 0.0))
        elif ev == "tool":
            tname = rec.get("tool_name")
            if tname:
                tools_used.add(str(tname))
            used_units = max(used_units, float(rec.get("used_units", 0.0)))
    return {
        "trace_present": True,
        "n_evals": n_evals,
        "n_hard_bound_violations": n_hb,
        "tools_used": sorted(tools_used),
        "used_units": round(used_units, 4),
        "sim_wall_time_s": round(sim_wall, 4),
    }


def _eval_observables_multi_seed(
    sim: Any,
    params: dict[str, float],
    cond_map: dict[str, Any],
    obs_rows: list[ObservableRow],
    seeds: list[int],
) -> tuple[bool, dict[str, float]]:
    """Verify candidate parameters against a list of ObservableRows on fresh seeds."""
    if not obs_rows:
        return True, {}

    by_cond: dict[str, list[ObservableRow]] = {}
    for row in obs_rows:
        by_cond.setdefault(row.condition_id, []).append(row)

    z_out: dict[str, float] = {}
    all_pass = True

    for cid, rows in by_cond.items():
        cond = cond_map[cid]
        eff_params = apply_condition(params, cond)
        feas_obs: list[dict[str, float]] = []
        for sd in seeds:
            res = sim.simulate(eff_params, seed=sd, duration_scale=1.0)
            if res.feasible:
                feas_obs.append(res.observables)

        # Require majority of fresh verification seeds to be feasible
        if len(feas_obs) < max(1, (len(seeds) + 1) // 2):
            for r in rows:
                z_out[f"{cid}:{r.obs_id}"] = float("inf")
            all_pass = False
            continue

        for r in rows:
            vals = [
                d[r.obs_id]
                for d in feas_obs
                if r.obs_id in d and math.isfinite(d[r.obs_id])
            ]
            if not vals:
                z_out[f"{cid}:{r.obs_id}"] = float("inf")
                all_pass = False
                continue
            mean_v = float(np.mean(vals))
            z = (mean_v - r.target_mean) / max(r.target_sd, 1e-9)
            z_out[f"{cid}:{r.obs_id}"] = round(float(z), 4)
            if abs(z) > r.tolerance_tau:
                all_pass = False

    return all_pass, z_out


def evaluate_submission(
    task_dir: str | Path,
    submission: FinalSubmission | str | Path,
    trace_path: str | Path | None = None,
    disclosure: DisclosureTier = "full",
    n_verify_seeds: int = 5,
    verify_seed_start: int = 50000,
) -> EvaluationReport:
    """Independently score a Track A or Track B submission against private/ ground truth."""
    if isinstance(submission, (str, Path)):
        raw = json.loads(Path(submission).read_text(encoding="utf-8"))
        disclosure = raw.get("disclosure", disclosure)
        sub = FinalSubmission(**{k: v for k, v in raw.items() if k in FinalSubmission.model_fields})
    else:
        sub = submission

    pub_real, _ = load_public_task(task_dir, disclosure="full")
    _, blind_to_real = load_public_task(task_dir, disclosure=disclosure)
    priv = load_private_bundle(task_dir)

    audit = _audit_trace(trace_path)
    used_units = float(audit["used_units"])
    max_units = float(pub_real.budget.max_units)

    is_solvable = priv.task_type in ("type1_routine", "type2_recoverable")
    is_type3 = priv.task_type in ("type3a_non_identifiable", "type3b_structure_unreachable")

    # Case 1: Abstention decision
    if sub.decision == "abstain_insufficient_evidence":
        if is_solvable:
            audit["premature_stop"] = bool(used_units < 0.5 * max_units)
            return EvaluationReport(
                task_id=priv.task_id,
                true_task_type=priv.task_type,
                decision=sub.decision,
                verified_pass=False,
                abstention_correct=False,
                abstention_subtype_correct=False,
                abstention_evidence_valid=False,
                false_abstention=True,
                used_units=used_units,
                max_units=max_units,
                process_audit=audit,
            )

        expected_subtype = (
            "3a_non_identifiable"
            if priv.task_type == "type3a_non_identifiable"
            else "3b_structure_unreachable"
        )
        subtype_ok = sub.abstention_subtype == expected_subtype
        tools_used = set(audit.get("tools_used", []))
        required_tools = _VALID_3A_TOOLS if expected_subtype == "3a_non_identifiable" else _VALID_3B_TOOLS
        has_tool_evidence = bool(tools_used & required_tools)
        has_summary_evidence = bool(sub.evidence_summary and len(sub.evidence_summary) > 0)
        # Require quantitative evidence summary AND either trace tool support or (when no trace) summary
        evidence_valid = has_summary_evidence and (
            has_tool_evidence or not audit["trace_present"]
        )
        abstain_ok = bool(subtype_ok and evidence_valid)
        return EvaluationReport(
            task_id=priv.task_id,
            true_task_type=priv.task_type,
            decision=sub.decision,
            verified_pass=abstain_ok,
            abstention_correct=abstain_ok,
            abstention_subtype_correct=subtype_ok,
            abstention_evidence_valid=evidence_valid,
            false_abstention=False,
            used_units=used_units,
            max_units=max_units,
            process_audit=audit,
        )

    # Case 2: Parameter solution submission
    assert sub.theta_hat is not None
    # Translate blinded aliases if necessary and merge with nominal baseline
    theta_real = pub_real.nominal_dict()
    for k, v in sub.theta_hat.items():
        real_k = blind_to_real.get(k, k)
        if real_k in theta_real:
            theta_real[real_k] = float(v)

    # Check physiological hard bounds
    pmap = pub_real.param_map()
    hb_viol = False
    for k, v in theta_real.items():
        row = pmap[k]
        if not math.isfinite(v) or v < row.phys_hard_lb - 1e-12 or v > row.phys_hard_ub + 1e-12:
            hb_viol = True
            break

    if hb_viol:
        return EvaluationReport(
            task_id=priv.task_id,
            true_task_type=priv.task_type,
            decision=sub.decision,
            verified_pass=False,
            public_pass=False,
            heldout_pass=False,
            false_confidence=is_type3,
            hard_bound_violated_on_submit=True,
            used_units=used_units,
            max_units=max_units,
            process_audit=audit,
        )

    sim = instantiate_simulator(pub_real.simulator_config, public_dir=pub_real.public_dir)
    seeds = [int(verify_seed_start) + i for i in range(int(n_verify_seeds))]

    pub_pass, pub_z = _eval_observables_multi_seed(
        sim, theta_real, pub_real.condition_map(), pub_real.observables, seeds
    )
    held_cond_map = {c.condition_id: c for c in priv.heldout_conditions}
    held_pass, held_z = _eval_observables_multi_seed(
        sim, theta_real, held_cond_map, priv.heldout_observables, seeds
    )

    if is_type3:
        return EvaluationReport(
            task_id=priv.task_id,
            true_task_type=priv.task_type,
            decision=sub.decision,
            verified_pass=False,
            public_pass=pub_pass,
            heldout_pass=held_pass,
            abstention_correct=False,
            false_confidence=True,
            public_z_scores=pub_z,
            heldout_z_scores=held_z,
            used_units=used_units,
            max_units=max_units,
            process_audit=audit,
        )

    verified_pass = bool(pub_pass and held_pass)
    if not verified_pass:
        audit["premature_stop"] = bool(used_units < 0.5 * max_units)
    return EvaluationReport(
        task_id=priv.task_id,
        true_task_type=priv.task_type,
        decision=sub.decision,
        verified_pass=verified_pass,
        public_pass=pub_pass,
        heldout_pass=held_pass,
        public_z_scores=pub_z,
        heldout_z_scores=held_z,
        used_units=used_units,
        max_units=max_units,
        process_audit=audit,
    )
