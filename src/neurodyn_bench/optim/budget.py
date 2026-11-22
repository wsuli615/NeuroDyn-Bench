"""Unified simulation budget ledger with hard cutoff and exact-key cache exemption."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from neurodyn_bench.models.base import SimulationResult


class BudgetExhaustedError(RuntimeError):
    """Raised when a simulator call would exceed the hard simulation budget."""


@dataclass
class BudgetLedger:
    """Accounts for every simulation, repeat, diagnostic probe, and failed run."""

    max_units: float
    used_units: float = 0.0
    n_calls: int = 0
    n_cache_hits: int = 0
    n_failed: int = 0
    total_wall_time_s: float = 0.0
    _cache: dict[tuple, SimulationResult] = field(default_factory=dict, repr=False)

    @property
    def remaining_units(self) -> float:
        return max(0.0, self.max_units - self.used_units)

    def can_afford(self, units: float) -> bool:
        return (self.used_units + units) <= (self.max_units + 1e-9)

    @staticmethod
    def _make_key(
        simulator_name: str,
        params: dict[str, float],
        seed: int,
        duration_scale: float,
    ) -> tuple:
        ordered = tuple((k, round(float(v), 12)) for k, v in sorted(params.items()))
        return (simulator_name, ordered, int(seed), round(float(duration_scale), 6))

    def execute(
        self,
        simulator_name: str,
        params: dict[str, float],
        seed: int,
        duration_scale: float,
        fn: Callable[[], SimulationResult],
    ) -> SimulationResult:
        """Run simulator with hard budget enforcement and exact-match caching."""
        key = self._make_key(simulator_name, params, seed, duration_scale)
        if key in self._cache:
            self.n_cache_hits += 1
            return self._cache[key]

        units = float(duration_scale)
        if not self.can_afford(units):
            raise BudgetExhaustedError(
                f"Budget exhausted: requested {units:.3f} units, "
                f"remaining {self.remaining_units:.3f} / {self.max_units:.3f}."
            )

        result = fn()
        self.used_units += result.sim_units
        self.n_calls += 1
        self.total_wall_time_s += result.wall_time_s
        if not result.feasible:
            self.n_failed += 1
        self._cache[key] = result
        return result

    def summary(self) -> dict[str, float | int]:
        return {
            "max_units": self.max_units,
            "used_units": round(self.used_units, 4),
            "remaining_units": round(self.remaining_units, 4),
            "n_calls": self.n_calls,
            "n_cache_hits": self.n_cache_hits,
            "n_failed": self.n_failed,
            "total_wall_time_s": round(self.total_wall_time_s, 4),
        }
