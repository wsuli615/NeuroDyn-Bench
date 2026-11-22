"""Self-describing Track A optional tool registry for autonomous agent tool selection."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from neurodyn_bench.gateway.gateway import SimulatorGateway
from neurodyn_bench.tools.bayesian import run_abc_smc, run_laplace_fisher, run_ppc_audit
from neurodyn_bench.tools.diagnostics import (
    boundary_slack,
    feature_tradeoff_audit,
    multistart_identifiability,
    repeat_variance,
    sensitivity_screen,
    short_feasibility_scan,
)
from neurodyn_bench.tools.optimizers import run_bo, run_cmaes, run_multistart_local


class ToolRegistry:
    """Registry allowing Track A controllers/LLMs to discover and invoke optional tools."""

    def __init__(self, gateway: SimulatorGateway) -> None:
        self.gateway = gateway
        self._specs: dict[str, dict[str, Any]] = {
            "run_bo": {
                "category": "optimizer",
                "fn": run_bo,
                "cost_hint": "n_evals * n_repeats simulation units (less if short_screen rejects infeasible points)",
                "description": "Bayesian Optimization (Optuna TPE) supporting custom active subsets, bounds, log/linear scales, short screening, and adaptive repeats.",
            },
            "run_cmaes": {
                "category": "optimizer",
                "fn": run_cmaes,
                "cost_hint": "n_evals * n_repeats simulation units",
                "description": "CMA-ES evolutionary search supporting active parameter subsets, custom bounds/scales, warm starts, and adaptive repeats.",
            },
            "run_multistart_local": {
                "category": "optimizer",
                "fn": run_multistart_local,
                "cost_hint": "up to n_starts * evals_per_start simulation units",
                "description": "Multi-start local Nelder-Mead refinement returning distinct local optima.",
            },
            "run_abc_smc": {
                "category": "bayesian",
                "fn": run_abc_smc,
                "cost_hint": "0 to n_particles * n_rounds units (reuses feasible history when reuse_history=True)",
                "description": "Lightweight SMC-ABC posterior approximation reporting quantiles, pairwise correlations, and posterior contraction (identifies Type 3a degeneracy).",
            },
            "run_laplace_fisher": {
                "category": "bayesian",
                "fn": run_laplace_fisher,
                "cost_hint": "d + 1 units (forward diff) or 2d + 1 units (central diff)",
                "description": "Local finite-difference Gauss-Newton Fisher eigenspectrum distinguishing stiff vs. sloppy parameters and rank deficiency.",
            },
            "run_ppc_audit": {
                "category": "bayesian",
                "fn": run_ppc_audit,
                "cost_hint": "len(candidates) * n_seeds_per_candidate units",
                "description": "Posterior Predictive Check on fresh seeds verifying target observable coverage and systematic residual mismatches.",
            },
            "boundary_slack": {
                "category": "diagnostic",
                "fn": boundary_slack,
                "cost_hint": "0 to #saturated_expandable_params units",
                "description": "Checks whether best solution sits on default search bounds vs. physiological hard bounds, with optional outside probe.",
            },
            "repeat_variance": {
                "category": "diagnostic",
                "fn": repeat_variance,
                "cost_hint": "n_repeats units (default 5)",
                "description": "Runs repeated seeds at one candidate to measure trial-to-trial stochastic noise vs. residual error.",
            },
            "short_feasibility_scan": {
                "category": "diagnostic",
                "fn": short_feasibility_scan,
                "cost_hint": "n_samples * short_probe_scale units (e.g. 2.4-3.0u for 12 samples)",
                "description": "Cheap short-horizon Latin-hypercube scan mapping feasible vs. silent/runaway phase-space fraction.",
            },
            "sensitivity_screen": {
                "category": "diagnostic",
                "fn": sensitivity_screen,
                "cost_hint": "(d + 1) * short_probe_scale units (when use_short_probe=True)",
                "description": "Cheap one-at-a-time sensitivity screening separating stiff parameters from sloppy/redundant dimensions.",
            },
            "feature_tradeoff_audit": {
                "category": "diagnostic",
                "fn": feature_tradeoff_audit,
                "cost_hint": "0 units (analyzes recorded gateway history)",
                "description": "Detects irreconcilable pairwise observable trade-offs (Type 3b fixed-structure unreachability signal).",
            },
            "multistart_identifiability": {
                "category": "diagnostic",
                "fn": multistart_identifiability,
                "cost_hint": "0 units (analyzes recorded low-loss gateway history)",
                "description": "Measures parameter-space dispersion among low-loss feasible fits (Type 3a degeneracy signal).",
            },
        }

    def list_tools(self) -> list[dict[str, str]]:
        return [
            {
                "name": name,
                "category": spec["category"],
                "cost_hint": spec["cost_hint"],
                "description": spec["description"],
            }
            for name, spec in self._specs.items()
        ]

    def call(self, tool_name: str, **kwargs: Any) -> dict[str, Any]:
        if tool_name not in self._specs:
            raise KeyError(f"Unknown tool: '{tool_name}'. Available: {sorted(self._specs)}")
        fn: Callable[..., dict[str, Any]] = self._specs[tool_name]["fn"]
        return fn(self.gateway, **kwargs)
