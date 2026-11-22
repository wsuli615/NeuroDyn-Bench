"""G1 Raw Optimizer Baselines: out-of-the-box BO and CMA-ES on default public settings."""

from __future__ import annotations

from neurodyn_bench.gateway.gateway import SimulatorGateway
from neurodyn_bench.tasks.schema import FinalSubmission
from neurodyn_bench.tools.optimizers import run_bo, run_cmaes


def run_g1_bo(gateway: SimulatorGateway, seed: int = 0) -> FinalSubmission:
    """Run default Optuna TPE BO using public/ default bounds and scales; never abstains."""
    n_evals = int(gateway.ledger.remaining_units)
    res = run_bo(
        gateway,
        n_evals=n_evals,
        n_repeats=gateway.bundle.budget.default_n_repeats,
        short_screen=False,
        seed=seed,
        hypothesis="G1 default BO baseline",
    )
    theta = res["full_best_params"] or gateway.bundle.nominal_dict()
    return FinalSubmission(
        decision="submit_solution",
        theta_hat=theta,
        notes=f"G1 default BO (best_rms_z={res['best_rms_z']})",
    )


def run_g1_cmaes(gateway: SimulatorGateway, seed: int = 0) -> FinalSubmission:
    """Run default CMA-ES using public/ default bounds and scales; never abstains."""
    n_evals = int(gateway.ledger.remaining_units)
    res = run_cmaes(
        gateway,
        n_evals=n_evals,
        n_repeats=gateway.bundle.budget.default_n_repeats,
        short_screen=False,
        seed=seed,
        hypothesis="G1 default CMA-ES baseline",
    )
    theta = res["full_best_params"] or gateway.bundle.nominal_dict()
    return FinalSubmission(
        decision="submit_solution",
        theta_hat=theta,
        notes=f"G1 default CMA-ES (best_rms_z={res['best_rms_z']})",
    )
