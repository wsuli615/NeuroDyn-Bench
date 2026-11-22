"""Track A optional modular toolbox: optimizers, lightweight Bayesian inference, and diagnostics."""

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
from neurodyn_bench.tools.registry import ToolRegistry

__all__ = [
    "ToolRegistry",
    "boundary_slack",
    "feature_tradeoff_audit",
    "multistart_identifiability",
    "repeat_variance",
    "run_abc_smc",
    "run_bo",
    "run_cmaes",
    "run_laplace_fisher",
    "run_multistart_local",
    "run_ppc_audit",
    "sensitivity_screen",
    "short_feasibility_scan",
]
