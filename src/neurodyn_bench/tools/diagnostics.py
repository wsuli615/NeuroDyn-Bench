"""Optional lightweight discriminative diagnostic probes billed via SimulatorGateway."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from neurodyn_bench.gateway.gateway import SimulatorGateway
from neurodyn_bench.optim.budget import BudgetExhaustedError
from neurodyn_bench.tools.optimizers import (
    params_to_unit,
    resolve_search_space,
    unit_to_params,
)


def boundary_slack(
    gateway: SimulatorGateway,
    candidate_params: dict[str, float] | None = None,
    active_params: list[str] | None = None,
    bounds: dict[str, tuple[float, float]] | None = None,
    edge_tol: float = 0.05,
    probe_outside: bool = True,
    seed: int = 7000,
    hypothesis: str = "",
) -> dict[str, Any]:
    """Detect boundary saturation and distinguish arbitrary default bounds from phys_hard limits."""
    rows, eff_bounds, eff_scales = resolve_search_space(gateway, active_params, bounds, None)
    best_rec = gateway.best_feasible_record()
    cand = dict(candidate_params or (best_rec.params if best_rec else gateway.bundle.nominal_dict()))
    base_rms = best_rec.rms_z if best_rec else float("inf")
    units_before = gateway.ledger.used_units

    u = params_to_unit(cand, rows, eff_bounds, eff_scales)
    saturated: list[dict[str, Any]] = []
    expandable_recommendations: dict[str, tuple[float, float]] = {}

    for i, row in enumerate(rows):
        lb, ub = eff_bounds[row.name]
        side = None
        if u[i] <= edge_tol:
            side = "lower"
            has_slack = lb > row.phys_hard_lb + 1e-9
            new_lb = max(row.phys_hard_lb, lb - 0.5 * (ub - lb))
            new_ub = ub
            probe_val = max(row.phys_hard_lb, lb - 0.25 * (ub - lb))
        elif u[i] >= 1.0 - edge_tol:
            side = "upper"
            has_slack = ub < row.phys_hard_ub - 1e-9
            new_lb = lb
            new_ub = min(row.phys_hard_ub, ub + 0.5 * (ub - lb))
            probe_val = min(row.phys_hard_ub, ub + 0.25 * (ub - lb))
        else:
            continue

        entry: dict[str, Any] = {
            "param": row.name,
            "side": side,
            "value": round(cand[row.name], 6),
            "current_bound": [lb, ub],
            "phys_hard_bound": [row.phys_hard_lb, row.phys_hard_ub],
            "status": "expandable_legal" if has_slack else "at_phys_hard_bound",
            "outside_probe_improved": None,
        }

        if has_slack and probe_outside and gateway.ledger.can_afford(1.0):
            test_p = dict(cand)
            test_p[row.name] = probe_val
            try:
                rec = gateway.evaluate(
                    test_p, seed=int(seed) + i, duration_scale=1.0, tool_tag="boundary_slack"
                )
                entry["outside_probe_rms_z"] = (
                    round(rec.rms_z, 4) if (rec.feasible and math.isfinite(rec.rms_z)) else None
                )
                improved = bool(rec.feasible and rec.rms_z < base_rms)
                entry["outside_probe_improved"] = improved
                if improved or base_rms > 2.0:
                    expandable_recommendations[row.name] = (round(new_lb, 6), round(new_ub, 6))
            except BudgetExhaustedError:
                pass
        elif has_slack:
            expandable_recommendations[row.name] = (round(new_lb, 6), round(new_ub, 6))

        saturated.append(entry)

    summary = {
        "base_rms_z": round(base_rms, 4) if math.isfinite(base_rms) else None,
        "saturated_params": saturated,
        "recommended_expanded_bounds": expandable_recommendations,
        "units_spent": round(gateway.ledger.used_units - units_before, 4),
        "remaining_units": round(gateway.ledger.remaining_units, 4),
    }
    gateway.record_tool_event("boundary_slack", summary, hypothesis=hypothesis)
    return summary


def repeat_variance(
    gateway: SimulatorGateway,
    candidate_params: dict[str, float] | None = None,
    n_repeats: int = 5,
    seed_start: int = 7100,
    hypothesis: str = "",
) -> dict[str, Any]:
    """Quantify stochastic repeat variance at a candidate parameter point."""
    best_rec = gateway.best_feasible_record()
    cand = dict(candidate_params or (best_rec.params if best_rec else gateway.bundle.nominal_dict()))
    units_before = gateway.ledger.used_units

    seeds = [int(seed_start) + i for i in range(int(n_repeats))]
    agg = gateway.evaluate_repeats(cand, seeds=seeds, duration_scale=1.0, tool_tag="repeat_variance")

    obs_rows = [o for o in gateway.bundle.observables if o.condition_id == "baseline"]
    z_sds: dict[str, float] = {}
    for o in obs_rows:
        sd_val = agg["sd_observables"].get(o.obs_id, float("nan"))
        if math.isfinite(sd_val):
            z_sds[o.obs_id] = round(sd_val / max(o.target_sd, 1e-9), 4)

    max_z_sd = max(z_sds.values()) if z_sds else float("nan")
    summary = {
        "n_completed": agg["n_completed"],
        "n_feasible": agg["n_feasible"],
        "feasible_rate": round(agg["feasible_rate"], 3),
        "mean_rms_z": round(agg["rms_z"], 4) if math.isfinite(agg["rms_z"]) else None,
        "max_abs_z": round(agg["max_abs_z"], 4) if math.isfinite(agg["max_abs_z"]) else None,
        "repeat_z_sd": z_sds,
        "max_repeat_z_sd": round(max_z_sd, 4) if math.isfinite(max_z_sd) else None,
        "variance_dominated": bool(math.isfinite(max_z_sd) and max_z_sd >= 0.45),
        "units_spent": round(gateway.ledger.used_units - units_before, 4),
        "remaining_units": round(gateway.ledger.remaining_units, 4),
    }
    gateway.record_tool_event("repeat_variance", summary, hypothesis=hypothesis)
    return summary


def short_feasibility_scan(
    gateway: SimulatorGateway,
    n_samples: int = 12,
    active_params: list[str] | None = None,
    bounds: dict[str, tuple[float, float]] | None = None,
    scales: dict[str, str] | None = None,
    seed: int = 7200,
    hypothesis: str = "",
) -> dict[str, Any]:
    """Cheap short-horizon Latin-hypercube scan mapping feasible vs. silent/runaway regions."""
    rows, eff_bounds, eff_scales = resolve_search_space(gateway, active_params, bounds, scales)
    base = gateway.bundle.nominal_dict()
    d = len(rows)
    rng = np.random.default_rng(int(seed))
    short_scale = gateway.bundle.budget.short_probe_scale
    units_before = gateway.ledger.used_units

    lhs = np.zeros((int(n_samples), d), dtype=np.float64)
    for j in range(d):
        perm = rng.permutation(int(n_samples))
        lhs[:, j] = (perm + rng.random(int(n_samples))) / int(n_samples)

    feasible_points: list[dict[str, float]] = []
    failure_counts: dict[str, int] = {}
    n_done = 0

    for i in range(int(n_samples)):
        if not gateway.ledger.can_afford(short_scale):
            break
        cand = unit_to_params(lhs[i], rows, eff_bounds, eff_scales, base)
        try:
            rec = gateway.evaluate(
                cand,
                seed=int(seed) + i,
                duration_scale=short_scale,
                tool_tag="short_feasibility_scan",
            )
        except BudgetExhaustedError:
            break
        n_done += 1
        if rec.feasible:
            feasible_points.append(cand)
        else:
            reason = rec.failure_reason or "unknown"
            failure_counts[reason] = failure_counts.get(reason, 0) + 1

    feas_box: dict[str, list[float]] = {}
    if feasible_points:
        for row in rows:
            vals = [p[row.name] for p in feasible_points]
            feas_box[row.name] = [round(min(vals), 5), round(max(vals), 5)]

    feas_frac = len(feasible_points) / max(n_done, 1)
    summary = {
        "n_samples": n_done,
        "short_probe_scale": short_scale,
        "feasible_count": len(feasible_points),
        "feasible_fraction": round(feas_frac, 3),
        "infeasible_zone_dominant": bool(feas_frac < 0.4),
        "failure_breakdown": failure_counts,
        "feasible_bounding_box": feas_box,
        "units_spent": round(gateway.ledger.used_units - units_before, 4),
        "remaining_units": round(gateway.ledger.remaining_units, 4),
    }
    gateway.record_tool_event("short_feasibility_scan", summary, hypothesis=hypothesis)
    return summary


def sensitivity_screen(
    gateway: SimulatorGateway,
    center_params: dict[str, float] | None = None,
    active_params: list[str] | None = None,
    bounds: dict[str, tuple[float, float]] | None = None,
    step_frac: float = 0.20,
    use_short_probe: bool = True,
    seed: int = 7300,
    hypothesis: str = "",
) -> dict[str, Any]:
    """Cheap one-at-a-time sensitivity screen separating stiff parameters from sloppy dims."""
    rows, eff_bounds, eff_scales = resolve_search_space(gateway, active_params, bounds, None)
    best_rec = gateway.best_feasible_record()
    center = dict(center_params or (best_rec.params if best_rec else gateway.bundle.nominal_dict()))
    scale_dur = gateway.bundle.budget.short_probe_scale if use_short_probe else 1.0
    units_before = gateway.ledger.used_units

    obs_rows = [o for o in gateway.bundle.observables if o.condition_id == "baseline"]
    u0 = params_to_unit(center, rows, eff_bounds, eff_scales)

    try:
        base_rec = gateway.evaluate(
            center, seed=int(seed), duration_scale=scale_dur, tool_tag="sensitivity_screen"
        )
    except BudgetExhaustedError:
        return {"status": "budget_exhausted", "remaining_units": gateway.ledger.remaining_units}

    if not base_rec.feasible:
        summary = {
            "status": "infeasible_center",
            "units_spent": round(gateway.ledger.used_units - units_before, 4),
        }
        gateway.record_tool_event("sensitivity_screen", summary, hypothesis=hypothesis)
        return summary

    z0 = np.array([base_rec.z_scores[o.obs_id] for o in obs_rows], dtype=np.float64)
    scores: dict[str, float] = {}

    for j, row in enumerate(rows):
        if not gateway.ledger.can_afford(scale_dur):
            break
        u_p = u0.copy()
        du = step_frac if u0[j] + step_frac <= 1.0 else -step_frac
        u_p[j] = float(np.clip(u0[j] + du, 0.0, 1.0))
        cand = unit_to_params(u_p, rows, eff_bounds, eff_scales, center)
        try:
            rec = gateway.evaluate(
                cand, seed=int(seed), duration_scale=scale_dur, tool_tag="sensitivity_screen"
            )
        except BudgetExhaustedError:
            break
        if rec.feasible:
            zp = np.array([rec.z_scores[o.obs_id] for o in obs_rows], dtype=np.float64)
            scores[row.name] = round(float(np.linalg.norm(zp - z0) / max(abs(du), 1e-6)), 4)
        else:
            # Perturbing this parameter crossed a bifurcation -> strongly sensitive!
            scores[row.name] = 999.0

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    max_s = max([v for _, v in ranked if v < 900.0], default=1.0)
    stiff = [k for k, v in ranked if v >= 0.12 * max(max_s, 1e-9)]
    sloppy = [k for k, v in ranked if v < 0.12 * max(max_s, 1e-9)]

    summary = {
        "status": "ok",
        "duration_scale": scale_dur,
        "sensitivity_ranking": [{"param": k, "score": v} for k, v in ranked],
        "stiff_params": stiff,
        "sloppy_params": sloppy,
        "units_spent": round(gateway.ledger.used_units - units_before, 4),
        "remaining_units": round(gateway.ledger.remaining_units, 4),
    }
    gateway.record_tool_event("sensitivity_screen", summary, hypothesis=hypothesis)
    return summary


def feature_tradeoff_audit(
    gateway: SimulatorGateway,
    hypothesis: str = "",
) -> dict[str, Any]:
    """Audit recorded evaluations for irreconcilable per-observable trade-offs (Type 3b signal)."""
    obs_rows = [o for o in gateway.bundle.observables if o.condition_id == "baseline"]
    obs_ids = [o.obs_id for o in obs_rows]
    feas_recs = [
        r
        for r in gateway.history
        if r.feasible
        and r.condition_id == "baseline"
        and r.duration_scale >= 0.99
        and math.isfinite(r.rms_z)
        and all(k in r.z_scores and math.isfinite(r.z_scores[k]) for k in obs_ids)
    ]
    if len(feas_recs) < 4:
        summary = {
            "status": "insufficient_history",
            "n_feasible_records": len(feas_recs),
        }
        gateway.record_tool_event("feature_tradeoff_audit", summary, hypothesis=hypothesis)
        return summary

    z_mat = np.array([[abs(r.z_scores[k]) for k in obs_ids] for r in feas_recs], dtype=np.float64)
    min_per_obs = {obs_ids[j]: round(float(np.min(z_mat[:, j])), 3) for j in range(len(obs_ids))}
    best_joint_idx = int(np.argmin(np.max(z_mat, axis=1)))
    best_joint_z = {obs_ids[j]: round(float(z_mat[best_joint_idx, j]), 3) for j in range(len(obs_ids))}
    best_joint_max_z = float(np.max(z_mat[best_joint_idx]))

    # Check pairwise negative correlation of |z_i| vs |z_j| (improving one worsens the other)
    conflicting_pairs: list[dict[str, Any]] = []
    for i in range(len(obs_ids)):
        for j in range(i + 1, len(obs_ids)):
            if np.std(z_mat[:, i]) > 1e-6 and np.std(z_mat[:, j]) > 1e-6:
                r_ij = float(np.corrcoef(z_mat[:, i], z_mat[:, j])[0, 1])
                if r_ij <= -0.25:
                    conflicting_pairs.append(
                        {"pair": [obs_ids[i], obs_ids[j]], "abs_z_corr": round(r_ij, 3)}
                    )
    conflicting_pairs.sort(key=lambda item: item["abs_z_corr"])

    unmet_at_joint_best = [
        o.obs_id for o in obs_rows if best_joint_z[o.obs_id] > o.tolerance_tau
    ]
    individually_reachable_but_jointly_unmet = [
        k for k in unmet_at_joint_best if min_per_obs[k] <= 2.5
    ]

    summary = {
        "status": "ok",
        "n_feasible_records": len(feas_recs),
        "min_abs_z_per_observable": min_per_obs,
        "best_joint_abs_z": best_joint_z,
        "best_joint_max_abs_z": round(best_joint_max_z, 3),
        "unmet_at_joint_best": unmet_at_joint_best,
        "individually_reachable_but_jointly_unmet": individually_reachable_but_jointly_unmet,
        "conflicting_pairs": conflicting_pairs[:5],
        "tradeoff_detected": bool(
            best_joint_max_z > 2.5
            and (len(conflicting_pairs) > 0 or len(individually_reachable_but_jointly_unmet) > 0)
        ),
    }
    gateway.record_tool_event("feature_tradeoff_audit", summary, hypothesis=hypothesis)
    return summary


def multistart_identifiability(
    gateway: SimulatorGateway,
    active_params: list[str] | None = None,
    bounds: dict[str, tuple[float, float]] | None = None,
    rms_z_threshold: float = 2.0,
    min_param_sep: float = 0.25,
    hypothesis: str = "",
) -> dict[str, Any]:
    """Audit parameter-space dispersion among all low-loss feasible solutions in history (Type 3a)."""
    rows, eff_bounds, eff_scales = resolve_search_space(gateway, active_params, bounds, None)
    good = [
        r
        for r in gateway.history
        if r.feasible
        and r.condition_id == "baseline"
        and r.duration_scale >= 0.99
        and r.rms_z <= rms_z_threshold
    ]
    if len(good) < 2:
        summary = {
            "status": "insufficient_low_loss_solutions",
            "n_low_loss": len(good),
            "rms_z_threshold": rms_z_threshold,
            "degeneracy_detected": False,
        }
        gateway.record_tool_event("multistart_identifiability", summary, hypothesis=hypothesis)
        return summary

    u_mat = np.vstack([params_to_unit(r.params, rows, eff_bounds, eff_scales) for r in good])
    max_linf = 0.0
    max_l2 = 0.0
    furthest_pair = (0, 1)
    for i in range(len(good)):
        for j in range(i + 1, len(good)):
            diff = np.abs(u_mat[i] - u_mat[j])
            linf = float(np.max(diff))
            l2 = float(np.linalg.norm(diff) / math.sqrt(len(rows)))
            if linf > max_linf:
                max_linf = linf
                max_l2 = l2
                furthest_pair = (i, j)

    per_param_span = {
        rows[k].name: round(float(np.max(u_mat[:, k]) - np.min(u_mat[:, k])), 4)
        for k in range(len(rows))
    }
    summary = {
        "status": "ok",
        "n_low_loss_solutions": len(good),
        "rms_z_threshold": rms_z_threshold,
        "max_normalized_linf_separation": round(max_linf, 4),
        "max_normalized_rms_l2_separation": round(max_l2, 4),
        "per_param_normalized_span": per_param_span,
        "example_disparate_pair": [
            {r.name: round(good[furthest_pair[0]].params[r.name], 5) for r in rows},
            {r.name: round(good[furthest_pair[1]].params[r.name], 5) for r in rows},
        ],
        "degeneracy_detected": bool(max_linf >= min_param_sep),
    }
    gateway.record_tool_event("multistart_identifiability", summary, hypothesis=hypothesis)
    return summary
