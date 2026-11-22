"""Anti-gaming paired headline metrics and SimulCost-style cost-sensitivity scan."""

from __future__ import annotations

from typing import Any

import numpy as np


def compute_headline_metrics(reports: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute disaggregated headline metrics preventing never-stop and always-abstain gaming."""
    t1 = [r for r in reports if r["true_task_type"] == "type1_routine"]
    t2 = [r for r in reports if r["true_task_type"] == "type2_recoverable"]
    t3a = [r for r in reports if r["true_task_type"] == "type3a_non_identifiable"]
    t3b = [r for r in reports if r["true_task_type"] == "type3b_structure_unreachable"]
    solvable = t1 + t2
    type3 = t3a + t3b

    def _rate(items: list[dict[str, Any]], key: str = "verified_pass") -> float | None:
        if not items:
            return None
        return round(float(np.mean([1.0 if r.get(key) else 0.0 for r in items])), 4)

    abstained_all = [r for r in reports if r["decision"] == "abstain_insufficient_evidence"]
    tp_abstain = sum(1 for r in type3 if r.get("abstention_correct"))
    precision = (tp_abstain / len(abstained_all)) if abstained_all else None
    recall = (tp_abstain / len(type3)) if type3 else None
    if precision is not None and recall is not None and (precision + recall) > 0:
        f1 = 2.0 * precision * recall / (precision + recall)
    else:
        f1 = 0.0 if (type3 and abstained_all) else None

    return {
        "n_total": len(reports),
        "n_type1": len(t1),
        "n_type2": len(t2),
        "n_type3a": len(t3a),
        "n_type3b": len(t3b),
        "verified_pass_rate_type1": _rate(t1),
        "verified_pass_rate_type2": _rate(t2),
        "verified_pass_rate_solvable": _rate(solvable),
        "abstention_recall_type3a": _rate(t3a, "abstention_correct"),
        "abstention_recall_type3b": _rate(t3b, "abstention_correct"),
        "abstention_precision": round(precision, 4) if precision is not None else None,
        "abstention_recall": round(recall, 4) if recall is not None else None,
        "abstention_f1": round(f1, 4) if f1 is not None else None,
        "false_abstention_rate_on_solvable": _rate(solvable, "false_abstention"),
        "false_confidence_rate_on_type3": _rate(type3, "false_confidence"),
        "mean_used_units": round(float(np.mean([r.get("used_units", 0.0) for r in reports])), 2)
        if reports
        else 0.0,
    }


def compute_cost_sensitivity_scan(
    used_units: float,
    llm_wall_time_s: float = 0.0,
    sim_cost_seconds_grid: list[float] | None = None,
) -> list[dict[str, float]]:
    """Project total effective wall-clock time across physical simulator costs (Rule 3.2)."""
    grid = sim_cost_seconds_grid or [0.1, 1.0, 3.0, 10.0, 60.0, 300.0]
    rows = []
    for sim_s in grid:
        sim_total = float(used_units) * float(sim_s)
        total = sim_total + float(llm_wall_time_s)
        rows.append(
            {
                "sim_cost_s": float(sim_s),
                "sim_total_s": round(sim_total, 2),
                "llm_overhead_s": round(float(llm_wall_time_s), 2),
                "total_wall_s": round(total, 2),
                "llm_share": round(float(llm_wall_time_s) / max(total, 1e-9), 4),
            }
        )
    return rows
