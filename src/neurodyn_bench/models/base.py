"""Core simulator interfaces and standardized result dataclasses."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ParameterSpec:
    """Biophysical parameter specification with dual bounds (default vs. physiological hard)."""

    name: str
    unit: str
    default_lb: float
    default_ub: float
    phys_hard_lb: float
    phys_hard_ub: float
    scale: str = "linear"  # "linear" or "log"
    nominal: float | None = None
    description: str = ""


@dataclass
class SimulationResult:
    """Standardized output of any neural dynamics simulator evaluation."""

    observables: dict[str, float]
    feasible: bool
    failure_reason: str | None = None
    wall_time_s: float = 0.0
    sim_units: float = 1.0
    seed: int = 0
    vector_observables: dict[str, list[float]] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)


class BaseSimulator(ABC):
    """Abstract interface for fixed-structure neural dynamics simulators."""

    name: str

    @abstractmethod
    def parameter_specs(self) -> list[ParameterSpec]:
        """Return ordered parameter specifications for this simulator."""

    @abstractmethod
    def simulate(
        self,
        params: dict[str, float],
        seed: int = 0,
        duration_scale: float = 1.0,
    ) -> SimulationResult:
        """Run one forward simulation and return summary statistics.

        Args:
            params: Mapping from parameter name to float value.
            seed: Deterministic RNG seed for reproducible stochastic trajectories.
            duration_scale: Fraction of full simulation horizon (e.g. 0.2 for a
                short feasibility probe, 1.0 for a full simulation).
        """
