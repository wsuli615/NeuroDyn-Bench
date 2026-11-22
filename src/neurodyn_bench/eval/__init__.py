"""Independent Evaluator and anti-gaming paired metrics."""

from neurodyn_bench.eval.evaluator import EvaluationReport, evaluate_submission
from neurodyn_bench.eval.metrics import (
    compute_cost_sensitivity_scan,
    compute_headline_metrics,
)

__all__ = [
    "EvaluationReport",
    "compute_cost_sensitivity_scan",
    "compute_headline_metrics",
    "evaluate_submission",
]
