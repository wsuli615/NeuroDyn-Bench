"""Optional budget-aware Bayesian inference and uncertainty quantification tools.

All simulations executed by these tools are strictly billed through SimulatorGateway:
  - run_abc_smc: Lightweight SMC-ABC / importance-weighted posterior approximation
    reporting posterior quantiles, pairwise correlations, and posterior contraction.
  - run_laplace_fisher: Local finite-difference Gauss-Newton Fisher eigenspectrum
    reporting stiff vs. sloppy directions and effective identifiability rank.
  - run_ppc_audit: Posterior Predictive Check across fresh seeds checking target
    coverage and systematic feature trade-offs.
"""

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


def run_abc_smc(
    gateway: SimulatorGateway,
    n_particles: int = 20,
    n_rounds: int = 2,
    active_params: list[str] | None = None,
    bounds: dict[str, tuple[float, float]] | None = None,
    scales: dict[str, str] | None = None,
    base_params: dict[str, float] | None = None,
    reuse_history: bool = True,
    temperature: float = 1.0,
    seed: int = 0,
    hypothesis: str = "",
) -> dict[str, Any]:
    """Lightweight budget-aware SMC-ABC posterior approximation.

    Estimates approximate posterior quantiles, pairwise parameter correlations
    (catching banana-shaped compensation manifolds), and posterior contraction
    `1 - Var_post / Var_prior` (in normalized [0, 1] prior coordinates, where
    prior variance = 1/12).
    """
    rows, eff_bounds, eff_scales = resolve_search_space(gateway, active_params, bounds, scales)
    base = dict(base_params or gateway.bundle.nominal_dict())
    d = len(rows)
    rng = np.random.default_rng(int(seed))
    units_before = gateway.ledger.used_units

    particles_u: list[np.ndarray] = []
    particles_loss: list[float] = []

    # Optionally seed from already-billed feasible history points (0 extra sim cost)
    if reuse_history:
        for rec in gateway.history:
            if rec.feasible and rec.condition_id == "baseline" and math.isfinite(rec.rms_z):
                u = params_to_unit(rec.params, rows, eff_bounds, eff_scales)
                particles_u.append(u)
                particles_loss.append(float(rec.rms_z))

    # Round 0: stratified Latin-hypercube draws from uniform box prior
    n_init = max(0, int(n_particles) - len(particles_u))
    if n_init > 0:
        lhs = np.zeros((n_init, d), dtype=np.float64)
        for j in range(d):
            perm = rng.permutation(n_init)
            lhs[:, j] = (perm + rng.random(n_init)) / n_init
        for idx in range(n_init):
            if not gateway.ledger.can_afford(1.0):
                break
            cand = unit_to_params(lhs[idx], rows, eff_bounds, eff_scales, base)
            try:
                rec = gateway.evaluate(
                    cand, seed=int(seed) * 500 + idx, duration_scale=1.0, tool_tag="run_abc_smc"
                )
            except BudgetExhaustedError:
                break
            if rec.feasible and math.isfinite(rec.rms_z):
                particles_u.append(lhs[idx])
                particles_loss.append(float(rec.rms_z))

    # Subsequent SMC perturbation rounds focusing around top weighted particles
    for r_idx in range(1, int(n_rounds)):
        if len(particles_u) < 3 or not gateway.ledger.can_afford(1.0):
            break
        u_arr = np.vstack(particles_u)
        l_arr = np.asarray(particles_loss, dtype=np.float64)
        l_min = float(np.min(l_arr))
        log_w = -0.5 * ((l_arr - l_min) ** 2) / max(float(temperature), 1e-3)
        w = np.exp(log_w - np.max(log_w))
        w = w / np.sum(w)
        # Empirical kernel SD per dimension
        w_mean = np.sum(u_arr * w[:, None], axis=0)
        w_sd = np.sqrt(np.sum(((u_arr - w_mean) ** 2) * w[:, None], axis=0))
        step_sd = np.clip(0.5 * w_sd, 0.03, 0.25)

        n_prop = min(int(n_particles), int(gateway.ledger.remaining_units))
        for p_i in range(n_prop):
            if not gateway.ledger.can_afford(1.0):
                break
            parent_idx = int(rng.choice(len(particles_u), p=w))
            prop_u = np.clip(u_arr[parent_idx] + rng.normal(0.0, step_sd), 0.0, 1.0)
            cand = unit_to_params(prop_u, rows, eff_bounds, eff_scales, base)
            try:
                rec = gateway.evaluate(
                    cand,
                    seed=int(seed) * 500 + r_idx * 100 + p_i,
                    duration_scale=1.0,
                    tool_tag="run_abc_smc",
                )
            except BudgetExhaustedError:
                break
            if rec.feasible and math.isfinite(rec.rms_z):
                particles_u.append(prop_u)
                particles_loss.append(float(rec.rms_z))

    if len(particles_u) < 3:
        summary = {
            "status": "insufficient_feasible_particles",
            "n_feasible_particles": len(particles_u),
            "units_spent": round(gateway.ledger.used_units - units_before, 4),
            "remaining_units": round(gateway.ledger.remaining_units, 4),
        }
        gateway.record_tool_event("run_abc_smc", summary, hypothesis=hypothesis)
        return summary

    u_mat = np.vstack(particles_u)
    l_vec = np.asarray(particles_loss, dtype=np.float64)
    # Likelihood weighting on absolute rms_z^2 (captures actual data constraint, not just relative rank)
    n_obs = max(len(gateway.bundle.observables), 1)
    log_w = -0.5 * n_obs * (l_vec**2) / max(float(temperature), 1e-3)
    log_w -= np.max(log_w)
    weights = np.exp(log_w)
    weights /= np.sum(weights)
    ess = float(1.0 / np.sum(weights**2))

    # Convert particles back to physical coordinates for interpretable quantiles
    phys_mat = np.zeros_like(u_mat)
    for idx in range(u_mat.shape[0]):
        c = unit_to_params(u_mat[idx], rows, eff_bounds, eff_scales, base)
        for j, row in enumerate(rows):
            phys_mat[idx, j] = c[row.name]

    prior_var_u = 1.0 / 12.0  # variance of Uniform(0, 1)
    contraction: dict[str, float] = {}
    post_summary: dict[str, dict[str, float]] = {}

    # Also compute unweighted spread among all particles with rms_z <= 2.0 (if >=2 exist)
    accepted_mask = l_vec <= 2.0
    n_accepted = int(np.sum(accepted_mask))

    for j, row in enumerate(rows):
        mean_u = float(np.sum(weights * u_mat[:, j]))
        var_u = float(np.sum(weights * (u_mat[:, j] - mean_u) ** 2))
        if n_accepted >= 2:
            acc_var_u = float(np.var(u_mat[accepted_mask, j], ddof=1))
            eff_var_u = max(var_u, acc_var_u)
        else:
            eff_var_u = var_u
        contraction[row.name] = round(float(1.0 - eff_var_u / prior_var_u), 4)

        order = np.argsort(phys_mat[:, j])
        cdf = np.cumsum(weights[order])
        q05 = float(phys_mat[order[min(np.searchsorted(cdf, 0.05), len(order) - 1)], j])
        q50 = float(phys_mat[order[min(np.searchsorted(cdf, 0.50), len(order) - 1)], j])
        q95 = float(phys_mat[order[min(np.searchsorted(cdf, 0.95), len(order) - 1)], j])
        post_mean = float(np.sum(weights * phys_mat[:, j]))
        post_sd = float(math.sqrt(max(np.sum(weights * (phys_mat[:, j] - post_mean) ** 2), 0.0)))
        post_summary[row.name] = {
            "mean": round(post_mean, 5),
            "sd": round(post_sd, 5),
            "q05": round(q05, 5),
            "q50": round(q50, 5),
            "q95": round(q95, 5),
            "contraction": contraction[row.name],
        }

    # Weighted correlation matrix in unit space to detect compensation ridges
    strong_corrs: list[dict[str, Any]] = []
    if d >= 2 and u_mat.shape[0] >= 4:
        mu = np.sum(u_mat * weights[:, None], axis=0)
        centered = u_mat - mu
        cov = (centered.T * weights) @ centered
        stds = np.sqrt(np.maximum(np.diag(cov), 1e-12))
        corr = cov / np.outer(stds, stds)
        for i in range(d):
            for j in range(i + 1, d):
                r_ij = float(corr[i, j])
                if math.isfinite(r_ij) and abs(r_ij) >= 0.45:
                    strong_corrs.append(
                        {"pair": [rows[i].name, rows[j].name], "corr": round(r_ij, 3)}
                    )
        strong_corrs.sort(key=lambda item: abs(item["corr"]), reverse=True)

    weakly_constrained = [name for name, c_val in contraction.items() if c_val < 0.45]
    summary = {
        "status": "ok",
        "n_feasible_particles": int(u_mat.shape[0]),
        "n_accepted_rms_le_2": n_accepted,
        "ess": round(ess, 2),
        "best_rms_z": round(float(np.min(l_vec)), 4),
        "posterior": post_summary,
        "posterior_contraction": contraction,
        "weakly_constrained_params": weakly_constrained,
        "strong_pairwise_correlations": strong_corrs[:6],
        "non_identifiability_flag": bool(
            n_accepted >= 2 and len(weakly_constrained) >= max(1, d // 2)
        ),
        "units_spent": round(gateway.ledger.used_units - units_before, 4),
        "remaining_units": round(gateway.ledger.remaining_units, 4),
    }
    gateway.record_tool_event("run_abc_smc", summary, hypothesis=hypothesis)
    return summary


def run_laplace_fisher(
    gateway: SimulatorGateway,
    center_params: dict[str, float] | None = None,
    active_params: list[str] | None = None,
    bounds: dict[str, tuple[float, float]] | None = None,
    scales: dict[str, str] | None = None,
    step_frac: float = 0.05,
    central_diff: bool = False,
    seed: int = 8100,
    hypothesis: str = "",
) -> dict[str, Any]:
    """Local finite-difference Gauss-Newton Fisher information eigenspectrum around a candidate."""
    rows, eff_bounds, eff_scales = resolve_search_space(gateway, active_params, bounds, scales)
    best_rec = gateway.best_feasible_record()
    center = dict(center_params or (best_rec.params if best_rec else gateway.bundle.nominal_dict()))
    d = len(rows)
    obs_rows = [o for o in gateway.bundle.observables if o.condition_id == "baseline"]
    m = len(obs_rows)
    units_before = gateway.ledger.used_units

    u0 = params_to_unit(center, rows, eff_bounds, eff_scales)
    try:
        base_rec = gateway.evaluate(center, seed=int(seed), duration_scale=1.0, tool_tag="run_laplace_fisher")
    except BudgetExhaustedError:
        return {"status": "budget_exhausted", "remaining_units": gateway.ledger.remaining_units}

    if not base_rec.feasible:
        summary = {
            "status": "infeasible_center",
            "failure_reason": base_rec.failure_reason,
            "units_spent": round(gateway.ledger.used_units - units_before, 4),
        }
        gateway.record_tool_event("run_laplace_fisher", summary, hypothesis=hypothesis)
        return summary

    z0 = np.array([base_rec.z_scores[o.obs_id] for o in obs_rows], dtype=np.float64)
    jac = np.zeros((m, d), dtype=np.float64)

    for j in range(d):
        if not gateway.ledger.can_afford(1.0):
            break
        u_plus = u0.copy()
        du = float(step_frac) if u0[j] + step_frac <= 1.0 else -float(step_frac)
        u_plus[j] = float(np.clip(u0[j] + du, 0.0, 1.0))
        cand_plus = unit_to_params(u_plus, rows, eff_bounds, eff_scales, center)
        try:
            rec_p = gateway.evaluate(
                cand_plus, seed=int(seed), duration_scale=1.0, tool_tag="run_laplace_fisher"
            )
        except BudgetExhaustedError:
            break

        if central_diff and gateway.ledger.can_afford(1.0):
            u_minus = u0.copy()
            u_minus[j] = float(np.clip(u0[j] - du, 0.0, 1.0))
            cand_minus = unit_to_params(u_minus, rows, eff_bounds, eff_scales, center)
            try:
                rec_m = gateway.evaluate(
                    cand_minus, seed=int(seed), duration_scale=1.0, tool_tag="run_laplace_fisher"
                )
            except BudgetExhaustedError:
                rec_m = None
            if rec_p.feasible and rec_m is not None and rec_m.feasible:
                zp = np.array([rec_p.z_scores[o.obs_id] for o in obs_rows], dtype=np.float64)
                zm = np.array([rec_m.z_scores[o.obs_id] for o in obs_rows], dtype=np.float64)
                jac[:, j] = (zp - zm) / max(abs(u_plus[j] - u_minus[j]), 1e-9)
                continue

        if rec_p.feasible:
            zp = np.array([rec_p.z_scores[o.obs_id] for o in obs_rows], dtype=np.float64)
            jac[:, j] = (zp - z0) / max(u_plus[j] - u0[j], 1e-9) if du > 0 else (zp - z0) / min(u_plus[j] - u0[j], -1e-9)

    fim = jac.T @ jac
    eigvals, eigvecs = np.linalg.eigh(fim)
    order = np.argsort(eigvals)[::-1]
    eigvals = np.maximum(eigvals[order], 0.0)
    eigvecs = eigvecs[:, order]

    top_eig = float(eigvals[0]) if len(eigvals) > 0 else 0.0
    eff_rank = int(np.sum(eigvals > 0.02 * max(top_eig, 1e-12))) if top_eig > 1e-12 else 0
    cond_num = float(top_eig / max(float(eigvals[-1]), 1e-12)) if top_eig > 0.0 else float("inf")

    # Per-parameter diagonal Fisher sensitivity (column norm of Jacobian)
    param_sens = {rows[j].name: round(float(np.linalg.norm(jac[:, j])), 4) for j in range(d)}
    max_sens = max(param_sens.values()) if param_sens else 0.0
    stiff = [k for k, v in param_sens.items() if v >= 0.2 * max(max_sens, 1e-12)]
    sloppy = [k for k, v in param_sens.items() if v < 0.2 * max(max_sens, 1e-12)]

    summary = {
        "status": "ok",
        "n_params": d,
        "n_observables": m,
        "center_rms_z": round(base_rec.rms_z, 4),
        "fisher_eigenvalues": [round(float(v), 4) for v in eigvals],
        "effective_rank": eff_rank,
        "rank_deficient": bool(eff_rank < d),
        "condition_number": round(cond_num, 2) if math.isfinite(cond_num) else None,
        "param_sensitivity": param_sens,
        "stiff_params": stiff,
        "sloppy_params": sloppy,
        "units_spent": round(gateway.ledger.used_units - units_before, 4),
        "remaining_units": round(gateway.ledger.remaining_units, 4),
    }
    gateway.record_tool_event("run_laplace_fisher", summary, hypothesis=hypothesis)
    return summary


def run_ppc_audit(
    gateway: SimulatorGateway,
    candidates: list[dict[str, float]] | None = None,
    n_seeds_per_candidate: int = 3,
    seed_start: int = 9000,
    hypothesis: str = "",
) -> dict[str, Any]:
    """Posterior Predictive Check (PPC): forward-simulate candidates on fresh seeds vs. targets."""
    units_before = gateway.ledger.used_units
    if not candidates:
        best = gateway.best_feasible_record()
        candidates = [best.params] if best is not None else [gateway.bundle.nominal_dict()]

    obs_rows = [o for o in gateway.bundle.observables if o.condition_id == "baseline"]
    collected: dict[str, list[float]] = {o.obs_id: [] for o in obs_rows}
    n_feas = 0
    n_tot = 0

    for c_i, cand in enumerate(candidates):
        for s_i in range(int(n_seeds_per_candidate)):
            if not gateway.ledger.can_afford(1.0):
                break
            try:
                rec = gateway.evaluate(
                    cand,
                    seed=int(seed_start) + c_i * 50 + s_i,
                    duration_scale=1.0,
                    tool_tag="run_ppc_audit",
                )
            except BudgetExhaustedError:
                break
            n_tot += 1
            if rec.feasible:
                n_feas += 1
                for o in obs_rows:
                    val = rec.observables.get(o.obs_id, float("nan"))
                    if math.isfinite(val):
                        collected[o.obs_id].append(float(val))

    per_obs: dict[str, dict[str, Any]] = {}
    covered_count = 0
    unmet_observables: list[str] = []

    for o in obs_rows:
        vals = collected[o.obs_id]
        if not vals:
            per_obs[o.obs_id] = {"covered": False, "mean_z": None}
            unmet_observables.append(o.obs_id)
            continue
        p_mean = float(np.mean(vals))
        p_sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        mean_z = (p_mean - o.target_mean) / max(o.target_sd, 1e-9)
        tol_covered = abs(mean_z) <= o.tolerance_tau
        interval_covered = (p_mean - 2.0 * p_sd <= o.target_mean <= p_mean + 2.0 * p_sd)
        covered = bool(tol_covered or interval_covered)
        if covered:
            covered_count += 1
        else:
            unmet_observables.append(o.obs_id)
        per_obs[o.obs_id] = {
            "pred_mean": round(p_mean, 4),
            "pred_sd": round(p_sd, 4),
            "target_mean": round(o.target_mean, 4),
            "target_sd": round(o.target_sd, 4),
            "mean_z": round(mean_z, 3),
            "covered": covered,
        }

    coverage_rate = covered_count / max(len(obs_rows), 1)
    summary = {
        "n_candidates": len(candidates),
        "n_simulations": n_tot,
        "n_feasible": n_feas,
        "coverage_rate": round(coverage_rate, 3),
        "unmet_observables": unmet_observables,
        "per_observable": per_obs,
        "ppc_pass": bool(coverage_rate >= 1.0 - 1e-9),
        "units_spent": round(gateway.ledger.used_units - units_before, 4),
        "remaining_units": round(gateway.ledger.remaining_units, 4),
    }
    gateway.record_tool_event("run_ppc_audit", summary, hypothesis=hypothesis)
    return summary
