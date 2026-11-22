"""G2 Deterministic Expert-Rule Baseline (fixed-threshold if-else diagnostic state machine)."""

from __future__ import annotations

import math

from neurodyn_bench.gateway.gateway import SimulatorGateway
from neurodyn_bench.tasks.schema import FinalSubmission
from neurodyn_bench.tools.bayesian import run_abc_smc, run_laplace_fisher, run_ppc_audit
from neurodyn_bench.tools.diagnostics import (
    boundary_slack,
    feature_tradeoff_audit,
    multistart_identifiability,
    repeat_variance,
    sensitivity_screen,
)
from neurodyn_bench.tools.optimizers import run_bo, run_cmaes


def run_g2_rules(gateway: SimulatorGateway, seed: int = 0) -> FinalSubmission:
    """Deterministic expert-rule baseline using Track A tools with fixed heuristic thresholds.

    Stages:
      1. If dim > 8, run cheap sensitivity_screen to select stiff subset & infer log scale on wide ratios.
      2. Run initial BO pilot (~35% budget).
      3. Inspect boundary_slack and repeat_variance; expand legal saturated bounds and enable adaptive repeats if noisy.
      4. Run follow-up CMA-ES (~45% budget).
      5. Audit residual & identifiability:
         - High residual + trade-off + failed PPC -> abstain 3b_structure_unreachable;
         - Low residual + underdetermined (rank-deficient Fisher or high multistart dispersion / low ABC contraction) -> abstain 3a_non_identifiable;
         - Otherwise submit best verified candidate.
    """
    bundle = gateway.bundle
    total_budget = bundle.budget.max_units
    active_rows = bundle.active_parameters()
    active_names = [r.name for r in active_rows]
    bounds = {r.name: (r.default_lb, r.default_ub) for r in active_rows}
    scales = {r.name: r.scale for r in active_rows}

    # Rule 1: High-dimensional sensitivity screening (d > 8)
    if len(active_names) > 8 and gateway.ledger.remaining_units > 25.0:
        sens = sensitivity_screen(
            gateway,
            active_params=active_names,
            bounds=bounds,
            use_short_probe=True,
            seed=seed + 10,
            hypothesis="G2 rule: screen high-D space to freeze sloppy parameters",
        )
        stiff = sens.get("stiff_params") or []
        if 2 <= len(stiff) < len(active_names):
            active_names = stiff[:6]
        # Switch wide positive-ratio parameters to log scale
        pmap = bundle.param_map()
        for name in active_names:
            row = pmap[name]
            if row.default_lb > 0 and (row.default_ub / row.default_lb) >= 3.0:
                scales[name] = "log"

    # Rule 2: Initial BO exploration (~35% budget)
    pilot_n = max(12, min(int(0.35 * total_budget), int(gateway.ledger.remaining_units - 15)))
    bo_res = run_bo(
        gateway,
        n_evals=pilot_n,
        active_params=active_names,
        bounds=bounds,
        scales=scales,
        seed=seed + 100,
        hypothesis="G2 rule: initial BO exploration",
    )

    best_theta = bo_res["full_best_params"] or bundle.nominal_dict()
    best_rms = bo_res["best_rms_z"] if bo_res["best_rms_z"] is not None else float("inf")

    # Rule 3: Boundary & variance diagnostics if not yet clearly converged
    adaptive_thresh: float | None = None
    if best_rms > 1.2 and gateway.ledger.remaining_units > 20.0:
        bslack = boundary_slack(
            gateway,
            candidate_params=best_theta,
            active_params=active_names,
            bounds=bounds,
            probe_outside=True,
            seed=seed + 200,
            hypothesis="G2 rule: check legal boundary slack",
        )
        for p_name, new_b in bslack.get("recommended_expanded_bounds", {}).items():
            bounds[p_name] = (float(new_b[0]), float(new_b[1]))

        if gateway.ledger.remaining_units > 18.0:
            rvar = repeat_variance(
                gateway,
                candidate_params=best_theta,
                n_repeats=4,
                seed_start=seed + 300,
                hypothesis="G2 rule: check repeat stochastic variance",
            )
            if rvar.get("variance_dominated"):
                adaptive_thresh = 2.5

    # Rule 4: Main follow-up search with CMA-ES (~40% remaining budget, saving ~15-20% for 3a/3b audit)
    reserve_for_audit = min(22.0, 0.18 * total_budget)
    search_units = max(0, int(gateway.ledger.remaining_units - reserve_for_audit))
    if search_units >= 8:
        cma_res = run_cmaes(
            gateway,
            n_evals=search_units,
            active_params=active_names,
            bounds=bounds,
            scales=scales,
            init_params=best_theta,
            adaptive_repeat_threshold=adaptive_thresh,
            seed=seed + 400,
            hypothesis="G2 rule: follow-up CMA-ES after bound/variance adjustment",
        )
        if cma_res["full_best_params"] is not None and (
            cma_res["best_rms_z"] is not None and cma_res["best_rms_z"] < best_rms
        ):
            best_theta = cma_res["full_best_params"]
            best_rms = cma_res["best_rms_z"]

    # Rule 5: Terminal 3b vs. 3a vs. submit check
    tradeoff = feature_tradeoff_audit(gateway, hypothesis="G2 rule: check Type 3b feature conflict")
    ppc = run_ppc_audit(
        gateway,
        candidates=[best_theta],
        n_seeds_per_candidate=3,
        seed_start=seed + 500,
        hypothesis="G2 rule: PPC validation of best candidate",
    )

    best_joint_max_z = tradeoff.get("best_joint_max_abs_z")
    if (
        best_joint_max_z is not None
        and best_joint_max_z > 3.0
        and not ppc.get("ppc_pass", False)
        and tradeoff.get("tradeoff_detected", False)
    ):
        return FinalSubmission(
            decision="abstain_insufficient_evidence",
            abstention_subtype="3b_structure_unreachable",
            evidence_summary={
                "feature_tradeoff_audit": tradeoff,
                "ppc_audit": ppc,
            },
            notes="G2 rule: high residual plateau with persistent feature trade-off and failed PPC.",
        )

    # Check Type 3a when fit is good (low residual)
    if math.isfinite(best_rms) and best_rms <= 2.2:
        ms_id = multistart_identifiability(
            gateway,
            active_params=active_names,
            bounds=bounds,
            rms_z_threshold=2.2,
            min_param_sep=0.35,
            hypothesis="G2 rule: check multi-solution degeneracy",
        )
        fisher = None
        if gateway.ledger.remaining_units >= len(active_names) + 1:
            fisher = run_laplace_fisher(
                gateway,
                center_params=best_theta,
                active_params=active_names,
                bounds=bounds,
                scales=scales,
                seed=seed + 600,
                hypothesis="G2 rule: check local Fisher rank deficiency",
            )
        abc = run_abc_smc(
            gateway,
            n_particles=0,  # zero extra sims: reuse history only
            n_rounds=1,
            active_params=active_names,
            bounds=bounds,
            scales=scales,
            reuse_history=True,
            seed=seed + 700,
            hypothesis="G2 rule: check posterior contraction on collected history",
        )

        # Conservative deterministic 3a trigger: underdetermined observables (m < d) OR both degeneracy + weak contraction
        n_obs = len(bundle.observables)
        underdetermined_obs = n_obs < len(active_names)
        degen = bool(ms_id.get("degeneracy_detected"))
        rank_def = bool(fisher and fisher.get("rank_deficient"))
        weak_post = bool(abc.get("non_identifiability_flag"))

        if (underdetermined_obs and (degen or rank_def)) or (degen and rank_def and weak_post):
            return FinalSubmission(
                decision="abstain_insufficient_evidence",
                abstention_subtype="3a_non_identifiable",
                evidence_summary={
                    "multistart_identifiability": ms_id,
                    "laplace_fisher": fisher,
                    "abc_smc": abc,
                },
                notes="G2 rule: low training error with rank-deficient Fisher / disparate equivalent solutions.",
            )

    return FinalSubmission(
        decision="submit_solution",
        theta_hat=best_theta,
        evidence_summary={"ppc_audit": ppc},
        notes=f"G2 rule submission (best_rms_z={best_rms})",
    )
