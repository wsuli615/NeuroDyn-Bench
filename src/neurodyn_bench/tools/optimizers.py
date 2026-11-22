"""Optional inner-loop numerical optimizers billed strictly through SimulatorGateway."""

from __future__ import annotations

import math
from typing import Any

import cma
import numpy as np
import optuna
from scipy.optimize import minimize

from neurodyn_bench.gateway.gateway import SimulatorGateway
from neurodyn_bench.optim.budget import BudgetExhaustedError
from neurodyn_bench.tasks.schema import ParameterRow

optuna.logging.set_verbosity(optuna.logging.WARNING)


def resolve_search_space(
    gateway: SimulatorGateway,
    active_params: list[str] | None = None,
    bounds: dict[str, tuple[float, float]] | None = None,
    scales: dict[str, str] | None = None,
) -> tuple[list[ParameterRow], dict[str, tuple[float, float]], dict[str, str]]:
    """Resolve active parameter rows, search bounds (clipped to phys_hard), and scales."""
    pmap = gateway.bundle.param_map()
    if active_params is None:
        rows = gateway.bundle.active_parameters()
    else:
        rows = [pmap[name] for name in active_params]

    eff_bounds: dict[str, tuple[float, float]] = {}
    eff_scales: dict[str, str] = {}
    for row in rows:
        if bounds and row.name in bounds:
            lb, ub = float(bounds[row.name][0]), float(bounds[row.name][1])
            lb = max(lb, row.phys_hard_lb)
            ub = min(ub, row.phys_hard_ub)
        else:
            lb, ub = row.default_lb, row.default_ub
        if lb >= ub:
            raise ValueError(f"Invalid bounds for {row.name}: [{lb}, {ub}]")
        sc = scales.get(row.name, row.scale) if scales else row.scale
        if sc == "log" and lb <= 0.0:
            sc = "linear"
        eff_bounds[row.name] = (lb, ub)
        eff_scales[row.name] = sc
    return rows, eff_bounds, eff_scales


def unit_to_params(
    u: np.ndarray,
    rows: list[ParameterRow],
    eff_bounds: dict[str, tuple[float, float]],
    eff_scales: dict[str, str],
    base_params: dict[str, float],
) -> dict[str, float]:
    out = dict(base_params)
    for i, row in enumerate(rows):
        ui = float(np.clip(u[i], 0.0, 1.0))
        lb, ub = eff_bounds[row.name]
        if eff_scales[row.name] == "log":
            out[row.name] = float(math.exp(math.log(lb) + ui * (math.log(ub) - math.log(lb))))
        else:
            out[row.name] = float(lb + ui * (ub - lb))
    return out


def params_to_unit(
    params: dict[str, float],
    rows: list[ParameterRow],
    eff_bounds: dict[str, tuple[float, float]],
    eff_scales: dict[str, str],
) -> np.ndarray:
    u = np.zeros(len(rows), dtype=np.float64)
    for i, row in enumerate(rows):
        val = float(params[row.name])
        lb, ub = eff_bounds[row.name]
        if eff_scales[row.name] == "log":
            u[i] = (math.log(max(val, lb)) - math.log(lb)) / max(math.log(ub) - math.log(lb), 1e-12)
        else:
            u[i] = (val - lb) / max(ub - lb, 1e-12)
    return np.clip(u, 0.0, 1.0)


def _eval_candidate(
    gateway: SimulatorGateway,
    cand: dict[str, float],
    seed: int,
    n_repeats: int,
    short_screen: bool,
    tool_tag: str,
    adaptive_repeat_threshold: float | None = None,
) -> tuple[float, float, bool]:
    """Evaluate one candidate with optional short-probe screen and repeats."""
    short_scale = gateway.bundle.budget.short_probe_scale
    if short_screen and short_scale < 1.0:
        if not gateway.ledger.can_afford(short_scale):
            raise BudgetExhaustedError("Cannot afford short screen probe")
        probe = gateway.evaluate(
            cand, seed=seed, duration_scale=short_scale, tool_tag=f"{tool_tag}:screen"
        )
        if not probe.feasible:
            return 1e4, float("inf"), False

    if n_repeats <= 1 and adaptive_repeat_threshold is None:
        rec = gateway.evaluate(cand, seed=seed, duration_scale=1.0, tool_tag=tool_tag)
        if not rec.feasible or not math.isfinite(rec.rms_z):
            return 1e4, float("inf"), False
        return rec.rms_z, rec.max_abs_z, True

    # First evaluation
    rec0 = gateway.evaluate(cand, seed=seed, duration_scale=1.0, tool_tag=tool_tag)
    if not rec0.feasible or not math.isfinite(rec0.rms_z):
        return 1e4, float("inf"), False

    target_repeats = n_repeats
    if adaptive_repeat_threshold is not None and rec0.rms_z <= adaptive_repeat_threshold:
        target_repeats = max(n_repeats, 3)

    if target_repeats <= 1 or not gateway.ledger.can_afford(1.0):
        return rec0.rms_z, rec0.max_abs_z, True

    seeds = [seed + 10000 * r for r in range(target_repeats)]
    agg = gateway.evaluate_repeats(cand, seeds=seeds, duration_scale=1.0, tool_tag=tool_tag)
    if agg["n_feasible"] == 0 or not math.isfinite(agg["rms_z"]):
        return 1e4, float("inf"), False
    return float(agg["rms_z"]), float(agg["max_abs_z"]), True


def _detect_boundary_hits(
    best_params: dict[str, float] | None,
    rows: list[ParameterRow],
    eff_bounds: dict[str, tuple[float, float]],
    eff_scales: dict[str, str],
    tol: float = 0.04,
) -> list[dict[str, Any]]:
    if best_params is None:
        return []
    u = params_to_unit(best_params, rows, eff_bounds, eff_scales)
    hits = []
    for i, row in enumerate(rows):
        lb, ub = eff_bounds[row.name]
        if u[i] <= tol:
            hits.append(
                {
                    "param": row.name,
                    "side": "lower",
                    "value": round(best_params[row.name], 6),
                    "bound": lb,
                    "phys_hard_bound": row.phys_hard_lb,
                    "has_legal_slack": lb > row.phys_hard_lb + 1e-9,
                }
            )
        elif u[i] >= 1.0 - tol:
            hits.append(
                {
                    "param": row.name,
                    "side": "upper",
                    "value": round(best_params[row.name], 6),
                    "bound": ub,
                    "phys_hard_bound": row.phys_hard_ub,
                    "has_legal_slack": ub < row.phys_hard_ub - 1e-9,
                }
            )
    return hits


def run_bo(
    gateway: SimulatorGateway,
    n_evals: int = 30,
    active_params: list[str] | None = None,
    bounds: dict[str, tuple[float, float]] | None = None,
    scales: dict[str, str] | None = None,
    base_params: dict[str, float] | None = None,
    n_repeats: int = 1,
    short_screen: bool = False,
    adaptive_repeat_threshold: float | None = None,
    seed: int = 0,
    hypothesis: str = "",
) -> dict[str, Any]:
    """Run budgeted Bayesian Optimization (Optuna TPE) over selected parameters."""
    rows, eff_bounds, eff_scales = resolve_search_space(gateway, active_params, bounds, scales)
    base = dict(base_params or gateway.bundle.nominal_dict())
    units_before = gateway.ledger.used_units

    sampler = optuna.samplers.TPESampler(seed=int(seed), n_startup_trials=min(8, max(3, len(rows) + 1)))
    study = optuna.create_study(direction="minimize", sampler=sampler)

    best_loss = float("inf")
    best_max_z = float("inf")
    best_theta: dict[str, float] | None = None
    n_feas = 0
    n_infeas = 0

    for step in range(int(n_evals)):
        if not gateway.ledger.can_afford(1.0):
            break
        trial = study.ask()
        u = np.array([trial.suggest_float(f"u_{i}", 0.0, 1.0) for i in range(len(rows))])
        cand = unit_to_params(u, rows, eff_bounds, eff_scales, base)
        try:
            loss, max_z, feas = _eval_candidate(
                gateway,
                cand,
                seed=int(seed) * 1000 + step,
                n_repeats=n_repeats,
                short_screen=short_screen,
                tool_tag="run_bo",
                adaptive_repeat_threshold=adaptive_repeat_threshold,
            )
        except BudgetExhaustedError:
            break
        study.tell(trial, loss)
        if feas:
            n_feas += 1
            if loss < best_loss:
                best_loss = loss
                best_max_z = max_z
                best_theta = cand
        else:
            n_infeas += 1

    summary = {
        "optimizer": "bo_tpe",
        "active_params": [r.name for r in rows],
        "n_trials_completed": n_feas + n_infeas,
        "n_feasible": n_feas,
        "n_infeasible": n_infeas,
        "best_rms_z": round(best_loss, 4) if math.isfinite(best_loss) else None,
        "best_max_abs_z": round(best_max_z, 4) if math.isfinite(best_max_z) else None,
        "best_params": {k: round(best_theta[k], 6) for k in [r.name for r in rows]}
        if best_theta
        else None,
        "full_best_params": best_theta,
        "boundary_hits": _detect_boundary_hits(best_theta, rows, eff_bounds, eff_scales),
        "units_spent": round(gateway.ledger.used_units - units_before, 4),
        "remaining_units": round(gateway.ledger.remaining_units, 4),
    }
    gateway.record_tool_event("run_bo", summary, hypothesis=hypothesis)
    return summary


def run_cmaes(
    gateway: SimulatorGateway,
    n_evals: int = 40,
    active_params: list[str] | None = None,
    bounds: dict[str, tuple[float, float]] | None = None,
    scales: dict[str, str] | None = None,
    base_params: dict[str, float] | None = None,
    init_params: dict[str, float] | None = None,
    n_repeats: int = 1,
    short_screen: bool = False,
    adaptive_repeat_threshold: float | None = None,
    popsize: int | None = None,
    sigma0: float = 0.25,
    seed: int = 0,
    hypothesis: str = "",
) -> dict[str, Any]:
    """Run budgeted CMA-ES over selected parameters."""
    rows, eff_bounds, eff_scales = resolve_search_space(gateway, active_params, bounds, scales)
    base = dict(base_params or gateway.bundle.nominal_dict())
    units_before = gateway.ledger.used_units

    if len(rows) < 2:
        # CMA-ES requires dim >= 2; delegate 1D cleanly to BO
        return run_bo(
            gateway,
            n_evals=n_evals,
            active_params=active_params,
            bounds=bounds,
            scales=scales,
            base_params=base_params,
            n_repeats=n_repeats,
            short_screen=short_screen,
            adaptive_repeat_threshold=adaptive_repeat_threshold,
            seed=seed,
            hypothesis=hypothesis,
        )

    if init_params is not None:
        x0 = params_to_unit(init_params, rows, eff_bounds, eff_scales)
        x0 = np.clip(x0, 0.05, 0.95)
    else:
        x0 = np.full(len(rows), 0.5, dtype=np.float64)

    eff_pop = popsize or max(6, min(12, 4 + int(3 * math.log(len(rows) + 1))))
    es = cma.CMAEvolutionStrategy(
        x0,
        float(sigma0),
        {"bounds": [0.0, 1.0], "popsize": eff_pop, "seed": int(seed) + 1, "verbose": -9},
    )

    best_loss = float("inf")
    best_max_z = float("inf")
    best_theta: dict[str, float] | None = None
    n_feas = 0
    n_infeas = 0
    step = 0

    while not es.stop() and step < int(n_evals) and gateway.ledger.can_afford(1.0):
        xs = es.ask()
        losses: list[float] = []
        aborted = False
        for x in xs:
            if step >= int(n_evals) or not gateway.ledger.can_afford(1.0):
                aborted = True
                break
            cand = unit_to_params(np.asarray(x), rows, eff_bounds, eff_scales, base)
            try:
                loss, max_z, feas = _eval_candidate(
                    gateway,
                    cand,
                    seed=int(seed) * 1000 + step,
                    n_repeats=n_repeats,
                    short_screen=short_screen,
                    tool_tag="run_cmaes",
                    adaptive_repeat_threshold=adaptive_repeat_threshold,
                )
            except BudgetExhaustedError:
                aborted = True
                break
            step += 1
            losses.append(loss)
            if feas:
                n_feas += 1
                if loss < best_loss:
                    best_loss = loss
                    best_max_z = max_z
                    best_theta = cand
            else:
                n_infeas += 1
        if not aborted and len(losses) == len(xs):
            es.tell(xs, losses)
        else:
            break

    summary = {
        "optimizer": "cmaes",
        "active_params": [r.name for r in rows],
        "n_trials_completed": n_feas + n_infeas,
        "n_feasible": n_feas,
        "n_infeasible": n_infeas,
        "best_rms_z": round(best_loss, 4) if math.isfinite(best_loss) else None,
        "best_max_abs_z": round(best_max_z, 4) if math.isfinite(best_max_z) else None,
        "best_params": {k: round(best_theta[k], 6) for k in [r.name for r in rows]}
        if best_theta
        else None,
        "full_best_params": best_theta,
        "boundary_hits": _detect_boundary_hits(best_theta, rows, eff_bounds, eff_scales),
        "units_spent": round(gateway.ledger.used_units - units_before, 4),
        "remaining_units": round(gateway.ledger.remaining_units, 4),
    }
    gateway.record_tool_event("run_cmaes", summary, hypothesis=hypothesis)
    return summary


def run_multistart_local(
    gateway: SimulatorGateway,
    n_starts: int = 3,
    evals_per_start: int = 15,
    active_params: list[str] | None = None,
    bounds: dict[str, tuple[float, float]] | None = None,
    scales: dict[str, str] | None = None,
    base_params: dict[str, float] | None = None,
    seed: int = 0,
    hypothesis: str = "",
) -> dict[str, Any]:
    """Run budgeted multi-start Nelder-Mead local refinement, returning all converged optima."""
    rows, eff_bounds, eff_scales = resolve_search_space(gateway, active_params, bounds, scales)
    base = dict(base_params or gateway.bundle.nominal_dict())
    rng = np.random.default_rng(int(seed))
    units_before = gateway.ledger.used_units

    starts_summary: list[dict[str, Any]] = []
    best_overall: dict[str, float] | None = None
    best_loss = float("inf")

    for s_idx in range(int(n_starts)):
        if not gateway.ledger.can_afford(1.0):
            break
        u0 = rng.uniform(0.15, 0.85, size=len(rows))
        local_best = {"loss": float("inf"), "params": None, "evals": 0}

        def obj(u_vec: np.ndarray, start_i: int = s_idx, tracker: dict = local_best) -> float:
            if tracker["evals"] >= int(evals_per_start) or not gateway.ledger.can_afford(1.0):
                return 1e4
            cand = unit_to_params(u_vec, rows, eff_bounds, eff_scales, base)
            try:
                rec = gateway.evaluate(
                    cand,
                    seed=int(seed) * 100 + start_i * 31 + tracker["evals"],
                    duration_scale=1.0,
                    tool_tag="run_multistart_local",
                )
            except BudgetExhaustedError:
                return 1e4
            tracker["evals"] += 1
            loss = rec.rms_z if (rec.feasible and math.isfinite(rec.rms_z)) else 1e4
            if loss < tracker["loss"]:
                tracker["loss"] = loss
                tracker["params"] = cand
            return loss

        minimize(
            obj,
            u0,
            method="Nelder-Mead",
            options={"maxfev": int(evals_per_start), "xatol": 1e-2, "fatol": 1e-2},
        )
        if local_best["params"] is not None:
            starts_summary.append(
                {
                    "start_idx": s_idx,
                    "best_rms_z": round(local_best["loss"], 4),
                    "params": {r.name: round(local_best["params"][r.name], 6) for r in rows},
                }
            )
            if local_best["loss"] < best_loss:
                best_loss = local_best["loss"]
                best_overall = local_best["params"]

    summary = {
        "optimizer": "multistart_nelder_mead",
        "active_params": [r.name for r in rows],
        "n_starts_completed": len(starts_summary),
        "starts": starts_summary,
        "best_rms_z": round(best_loss, 4) if math.isfinite(best_loss) else None,
        "full_best_params": best_overall,
        "units_spent": round(gateway.ledger.used_units - units_before, 4),
        "remaining_units": round(gateway.ledger.remaining_units, 4),
    }
    gateway.record_tool_event("run_multistart_local", summary, hypothesis=hypothesis)
    return summary
