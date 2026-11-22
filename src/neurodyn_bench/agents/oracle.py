"""Task-specific Oracle runner used solely for offline certification (Rule 2.3)."""

from __future__ import annotations

import math
from typing import Any

from neurodyn_bench.gateway.gateway import SimulatorGateway
from neurodyn_bench.tasks.schema import FinalSubmission, PrivateTaskBundle
from neurodyn_bench.tools.optimizers import run_bo, run_cmaes


def run_oracle(
    gateway: SimulatorGateway,
    private_bundle: PrivateTaskBundle,
    seed: int = 0,
) -> FinalSubmission:
    """Run the Oracle remediation strategy for a solvable Type 1/2 task within task budget.

    Never exposed to evaluated agents; called exclusively by `scripts/certify_v0_tasks.py`
    to certify Rule 2.3 (Oracle >= 80% independent verification pass rate).
    """
    trap: dict[str, Any] = private_bundle.trap_metadata
    bundle = gateway.bundle
    pmap = bundle.param_map()

    oracle_active: list[str] = trap.get(
        "oracle_active_params", [r.name for r in bundle.active_parameters()]
    )
    oracle_bounds: dict[str, tuple[float, float]] = {}
    oracle_scales: dict[str, str] = {}

    raw_bounds = trap.get("oracle_bounds", {})
    raw_scales = trap.get("oracle_scales", {})
    for name in oracle_active:
        row = pmap[name]
        if name in raw_bounds:
            oracle_bounds[name] = (float(raw_bounds[name][0]), float(raw_bounds[name][1]))
        else:
            oracle_bounds[name] = (row.default_lb, row.default_ub)
        oracle_scales[name] = str(raw_scales.get(name, row.scale))

    adaptive_thresh = trap.get("oracle_adaptive_repeat_threshold")
    n_repeats = int(trap.get("oracle_n_repeats", 1))
    short_screen = bool(trap.get("oracle_short_screen", False))
    opt_choice = str(trap.get("oracle_optimizer", "cmaes"))

    # Reserve final 9 units to repeat-verify top-3 distinct candidates before final submission
    total = int(gateway.ledger.remaining_units)
    reserve_verify = min(9, max(0, total // 10))
    search_budget = max(12, total - reserve_verify)

    if opt_choice == "cmaes":
        run_cmaes(
            gateway,
            n_evals=search_budget,
            active_params=oracle_active,
            bounds=oracle_bounds,
            scales=oracle_scales,
            n_repeats=n_repeats,
            short_screen=short_screen,
            adaptive_repeat_threshold=adaptive_thresh,
            seed=seed,
            hypothesis="Oracle certification pure CMA-ES in remediated search space",
        )
    else:
        # 50% BO exploration -> 50% CMA-ES refinement in the remediated subspace/bounds/scales
        n_bo = max(12, int(0.50 * search_budget))
        bo_res = run_bo(
            gateway,
            n_evals=n_bo,
            active_params=oracle_active,
            bounds=oracle_bounds,
            scales=oracle_scales,
            n_repeats=1,
            short_screen=short_screen,
            adaptive_repeat_threshold=adaptive_thresh,
            seed=seed,
            hypothesis="Oracle certification BO exploration",
        )
        init_p = bo_res["full_best_params"] or bundle.nominal_dict()
        rem_search = max(0, int(gateway.ledger.remaining_units) - reserve_verify)

        if rem_search >= 8:
            run_cmaes(
                gateway,
                n_evals=rem_search,
                active_params=oracle_active,
                bounds=oracle_bounds,
                scales=oracle_scales,
                init_params=init_p,
                n_repeats=n_repeats,
                short_screen=short_screen,
                adaptive_repeat_threshold=adaptive_thresh,
                seed=seed + 50,
                hypothesis="Oracle certification CMA-ES refinement",
            )

    # Select top-3 distinct feasible candidates and verify each on 3 fresh seeds
    feas_hist = sorted(
        [r for r in gateway.history if r.feasible and r.condition_id == "baseline" and math.isfinite(r.rms_z)],
        key=lambda r: r.rms_z,
    )
    top_cands: list[dict[str, float]] = []
    for rec in feas_hist:
        if all(
            any(abs(rec.params[k] - prev[k]) > 1e-4 for k in oracle_active)
            for prev in top_cands
        ):
            top_cands.append(rec.params)
        if len(top_cands) >= 3:
            break

    if not top_cands:
        return FinalSubmission(
            decision="submit_solution",
            theta_hat=bundle.nominal_dict(),
            notes="Oracle fallback nominal",
        )

    best_verified_theta = top_cands[0]
    best_verified_loss = float("inf")
    if gateway.ledger.remaining_units >= 3.0:
        for c_i, cand in enumerate(top_cands):
            if not gateway.ledger.can_afford(3.0):
                break
            agg = gateway.evaluate_repeats(
                cand,
                seeds=[95000 + seed * 100 + c_i * 10 + k for k in range(3)],
                duration_scale=1.0,
                tool_tag="oracle_verify_top",
            )
            if agg["n_feasible"] > 0 and agg["rms_z"] < best_verified_loss:
                best_verified_loss = float(agg["rms_z"])
                best_verified_theta = cand

    return FinalSubmission(
        decision="submit_solution",
        theta_hat=best_verified_theta,
        notes=f"Oracle BO+CMA-ES + top-3 repeat verify (verified_rms_z={best_verified_loss:.3f})",
    )
