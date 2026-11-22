"""Unified SimulatorGateway enforcing hard budgets, physiological bounds, and trace.jsonl auditing."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from neurodyn_bench.models.base import BaseSimulator, SimulationResult
from neurodyn_bench.optim.budget import BudgetExhaustedError, BudgetLedger
from neurodyn_bench.tasks.loader import (
    apply_condition,
    instantiate_simulator,
    load_public_task,
)
from neurodyn_bench.tasks.schema import (
    DisclosureTier,
    FinalSubmission,
    PublicTaskBundle,
)


@dataclass
class EvalRecord:
    """Standardized evaluation result returned by SimulatorGateway."""

    params: dict[str, float]
    condition_id: str
    seed: int
    duration_scale: float
    feasible: bool
    failure_reason: str | None
    observables: dict[str, float]
    z_scores: dict[str, float]
    rms_z: float
    max_abs_z: float
    wall_time_s: float
    sim_units: float
    used_units: float
    remaining_units: float
    cache_hit: bool = False
    hard_bound_violation: bool = False
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "params": self.params,
            "condition_id": self.condition_id,
            "seed": self.seed,
            "duration_scale": self.duration_scale,
            "feasible": self.feasible,
            "failure_reason": self.failure_reason,
            "observables": {
                k: (float(v) if math.isfinite(v) else None) for k, v in self.observables.items()
            },
            "z_scores": {
                k: (float(v) if math.isfinite(v) else None) for k, v in self.z_scores.items()
            },
            "rms_z": float(self.rms_z) if math.isfinite(self.rms_z) else None,
            "max_abs_z": float(self.max_abs_z) if math.isfinite(self.max_abs_z) else None,
            "wall_time_s": round(self.wall_time_s, 4),
            "sim_units": self.sim_units,
            "used_units": round(self.used_units, 4),
            "remaining_units": round(self.remaining_units, 4),
            "cache_hit": self.cache_hit,
            "hard_bound_violation": self.hard_bound_violation,
        }


class SimulatorGateway:
    """Single source of truth for simulator execution, hard budgeting, and trace auditing."""

    def __init__(
        self,
        public_bundle: PublicTaskBundle,
        simulator: BaseSimulator,
        trace_path: str | Path | None = None,
        blind_to_real: dict[str, str] | None = None,
        max_units_override: float | None = None,
    ) -> None:
        self.bundle = public_bundle
        self.simulator = simulator
        self.blind_to_real = dict(blind_to_real or {})
        self.real_to_blind = {v: k for k, v in self.blind_to_real.items()}
        max_units = (
            float(max_units_override)
            if max_units_override is not None
            else float(public_bundle.budget.max_units)
        )
        self.ledger = BudgetLedger(max_units=max_units)
        self.trace_path = Path(trace_path) if trace_path is not None else None
        if self.trace_path is not None:
            self.trace_path.parent.mkdir(parents=True, exist_ok=True)
            self.trace_path.write_text("", encoding="utf-8")

        self._param_map = public_bundle.param_map()
        self._cond_map = public_bundle.condition_map()
        self._obs_by_cond: dict[str, list] = {}
        for o in public_bundle.observables:
            self._obs_by_cond.setdefault(o.condition_id, []).append(o)

        self.history: list[EvalRecord] = []
        self.tool_events: list[dict[str, Any]] = []
        self._round_counter = 0

    @classmethod
    def from_task_dir(
        cls,
        task_dir: str | Path,
        disclosure: DisclosureTier = "full",
        trace_path: str | Path | None = None,
        max_units_override: float | None = None,
    ) -> SimulatorGateway:
        bundle, blind_map = load_public_task(task_dir, disclosure=disclosure)
        sim = instantiate_simulator(bundle.simulator_config, public_dir=bundle.public_dir)
        return cls(
            public_bundle=bundle,
            simulator=sim,
            trace_path=trace_path,
            blind_to_real=blind_map,
            max_units_override=max_units_override,
        )

    def _append_trace(self, record: dict[str, Any]) -> None:
        if self.trace_path is None:
            return
        with self.trace_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def to_real_params(self, public_params: dict[str, float]) -> dict[str, float]:
        """Merge public parameter overrides onto nominal vector and translate blinded aliases."""
        merged = self.bundle.nominal_dict()
        for k, v in public_params.items():
            if k in merged:
                merged[k] = float(v)
            elif k in self.real_to_blind:
                merged[self.real_to_blind[k]] = float(v)
            else:
                raise KeyError(f"Unknown parameter name: '{k}'")

        if not self.blind_to_real:
            return merged
        return {self.blind_to_real.get(k, k): v for k, v in merged.items()}

    def check_hard_bounds(self, public_params: dict[str, float]) -> list[str]:
        """Return list of parameter names violating physiological hard bounds."""
        violations: list[str] = []
        for k, v in public_params.items():
            row = self._param_map.get(k)
            if row is None:
                continue
            val = float(v)
            if not math.isfinite(val) or val < row.phys_hard_lb - 1e-12 or val > row.phys_hard_ub + 1e-12:
                violations.append(k)
        return violations

    def compute_objective(
        self, observables: dict[str, float], condition_id: str = "baseline"
    ) -> tuple[dict[str, float], float, float]:
        """Compute per-observable z-scores, weighted RMS z-loss, and max |z|."""
        target_rows = self._obs_by_cond.get(condition_id, [])
        if not target_rows:
            return {}, float("nan"), float("nan")

        z_scores: dict[str, float] = {}
        w_sq_sum = 0.0
        w_sum = 0.0
        max_abs_z = 0.0
        for row in target_rows:
            val = observables.get(row.obs_id, float("nan"))
            if not math.isfinite(val):
                z_scores[row.obs_id] = float("nan")
                return z_scores, float("inf"), float("inf")
            z = (val - row.target_mean) / max(row.target_sd, 1e-9)
            z_scores[row.obs_id] = float(z)
            w_sq_sum += row.weight * (z * z)
            w_sum += row.weight
            max_abs_z = max(max_abs_z, abs(z))

        rms_z = math.sqrt(w_sq_sum / max(w_sum, 1e-12))
        return z_scores, float(rms_z), float(max_abs_z)

    def evaluate(
        self,
        params: dict[str, float],
        seed: int = 0,
        condition_id: str = "baseline",
        duration_scale: float = 1.0,
        tool_tag: str = "direct_eval",
    ) -> EvalRecord:
        """Execute one budgeted forward simulation with hard-bound checks and trace logging."""
        if condition_id not in self._cond_map:
            raise KeyError(f"Unknown public condition_id: '{condition_id}'")
        if duration_scale <= 0.0 or duration_scale > 1.0 + 1e-9:
            raise ValueError(f"duration_scale must be in (0, 1], got {duration_scale}")

        full_pub = self.bundle.nominal_dict()
        for k, v in params.items():
            if k not in full_pub:
                raise KeyError(f"Unknown public parameter: '{k}'")
            full_pub[k] = float(v)

        violations = self.check_hard_bounds(full_pub)
        if violations:
            rec = EvalRecord(
                params=full_pub,
                condition_id=condition_id,
                seed=int(seed),
                duration_scale=float(duration_scale),
                feasible=False,
                failure_reason=f"hard_bound_violation:{','.join(violations)}",
                observables={},
                z_scores={},
                rms_z=float("inf"),
                max_abs_z=float("inf"),
                wall_time_s=0.0,
                sim_units=0.0,
                used_units=self.ledger.used_units,
                remaining_units=self.ledger.remaining_units,
                cache_hit=False,
                hard_bound_violation=True,
            )
            self.history.append(rec)
            self._append_trace(
                {
                    "event": "eval",
                    "round_id": self._round_counter,
                    "tool_tag": tool_tag,
                    "timestamp": time.time(),
                    **rec.to_dict(),
                }
            )
            return rec

        real_params = self.to_real_params(full_pub)
        cond_row = self._cond_map[condition_id]
        eff_params = apply_condition(real_params, cond_row)

        cache_before = self.ledger.n_cache_hits
        sim_key = f"{self.simulator.name}:{condition_id}"
        sim_res: SimulationResult = self.ledger.execute(
            simulator_name=sim_key,
            params=eff_params,
            seed=int(seed),
            duration_scale=float(duration_scale),
            fn=lambda: self.simulator.simulate(
                eff_params, seed=int(seed), duration_scale=float(duration_scale)
            ),
        )
        cache_hit = self.ledger.n_cache_hits > cache_before

        if sim_res.feasible:
            z_scores, rms_z, max_abs_z = self.compute_objective(sim_res.observables, condition_id)
        else:
            z_scores, rms_z, max_abs_z = {}, float("inf"), float("inf")

        rec = EvalRecord(
            params=full_pub,
            condition_id=condition_id,
            seed=int(seed),
            duration_scale=float(duration_scale),
            feasible=sim_res.feasible,
            failure_reason=sim_res.failure_reason,
            observables=sim_res.observables,
            z_scores=z_scores,
            rms_z=rms_z,
            max_abs_z=max_abs_z,
            wall_time_s=sim_res.wall_time_s,
            sim_units=0.0 if cache_hit else sim_res.sim_units,
            used_units=self.ledger.used_units,
            remaining_units=self.ledger.remaining_units,
            cache_hit=cache_hit,
            hard_bound_violation=False,
            meta=sim_res.meta,
        )
        self.history.append(rec)
        self._append_trace(
            {
                "event": "eval",
                "round_id": self._round_counter,
                "tool_tag": tool_tag,
                "timestamp": time.time(),
                **rec.to_dict(),
            }
        )
        return rec

    def evaluate_repeats(
        self,
        params: dict[str, float],
        seeds: list[int],
        condition_id: str = "baseline",
        duration_scale: float = 1.0,
        tool_tag: str = "repeat_eval",
    ) -> dict[str, Any]:
        """Evaluate the same parameter vector across multiple seeds under unified budget."""
        records: list[EvalRecord] = []
        for sd in seeds:
            if not self.ledger.can_afford(duration_scale):
                break
            records.append(
                self.evaluate(
                    params,
                    seed=sd,
                    condition_id=condition_id,
                    duration_scale=duration_scale,
                    tool_tag=tool_tag,
                )
            )
        feasible_recs = [r for r in records if r.feasible]
        n_feas = len(feasible_recs)
        if n_feas == 0:
            return {
                "n_requested": len(seeds),
                "n_completed": len(records),
                "n_feasible": 0,
                "feasible_rate": 0.0,
                "mean_observables": {},
                "sd_observables": {},
                "z_scores": {},
                "rms_z": float("inf"),
                "max_abs_z": float("inf"),
                "remaining_units": self.ledger.remaining_units,
            }

        obs_keys = sorted(feasible_recs[0].observables.keys())
        mean_obs: dict[str, float] = {}
        sd_obs: dict[str, float] = {}
        for k in obs_keys:
            vals = [
                r.observables[k]
                for r in feasible_recs
                if k in r.observables and math.isfinite(r.observables[k])
            ]
            if vals:
                mean_obs[k] = float(np.mean(vals))
                sd_obs[k] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0

        z_scores, rms_z, max_abs_z = self.compute_objective(mean_obs, condition_id)
        return {
            "n_requested": len(seeds),
            "n_completed": len(records),
            "n_feasible": n_feas,
            "feasible_rate": n_feas / max(len(records), 1),
            "mean_observables": mean_obs,
            "sd_observables": sd_obs,
            "z_scores": z_scores,
            "rms_z": rms_z,
            "max_abs_z": max_abs_z,
            "remaining_units": self.ledger.remaining_units,
        }

    def record_tool_event(
        self,
        tool_name: str,
        summary: dict[str, Any],
        hypothesis: str = "",
    ) -> None:
        """Append a structured high-level tool or diagnostic event to trace.jsonl."""
        self._round_counter += 1
        entry = {
            "event": "tool",
            "round_id": self._round_counter,
            "tool_name": tool_name,
            "hypothesis": hypothesis,
            "used_units": round(self.ledger.used_units, 4),
            "remaining_units": round(self.ledger.remaining_units, 4),
            "timestamp": time.time(),
            "summary": summary,
        }
        self.tool_events.append(entry)
        self._append_trace(entry)

    def best_feasible_record(self, condition_id: str = "baseline") -> EvalRecord | None:
        """Return the lowest-rms_z feasible evaluation seen so far."""
        cands = [
            r
            for r in self.history
            if r.feasible and r.condition_id == condition_id and math.isfinite(r.rms_z)
        ]
        if not cands:
            return None
        return min(cands, key=lambda r: r.rms_z)

    def public_task_info(self) -> dict[str, Any]:
        """Serialize public task metadata and current budget state for agents."""
        return {
            "task_id": self.bundle.metadata.task_id,
            "title": self.bundle.metadata.title,
            "model_type": self.bundle.metadata.model_type,
            "disclosure": self.bundle.metadata.disclosure,
            "prompt": self.bundle.metadata.prompt,
            "parameters": [p.model_dump() for p in self.bundle.parameters],
            "observables": [o.model_dump() for o in self.bundle.observables],
            "conditions": [c.model_dump() for c in self.bundle.conditions],
            "budget": self.ledger.summary(),
            "short_probe_scale": self.bundle.budget.short_probe_scale,
        }

    def save_submission(
        self,
        submission: FinalSubmission,
        output_path: str | Path,
    ) -> Path:
        """Persist final submission and record terminal event in trace.jsonl."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = submission.model_dump()
        payload["disclosure"] = self.bundle.metadata.disclosure
        payload["budget_summary"] = self.ledger.summary()
        out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        self._append_trace(
            {
                "event": "final_submission",
                "round_id": self._round_counter + 1,
                "timestamp": time.time(),
                "submission": payload,
            }
        )
        return out

    @property
    def exhausted(self) -> bool:
        return not self.ledger.can_afford(min(self.bundle.budget.short_probe_scale, 1.0))


__all__ = ["BudgetExhaustedError", "EvalRecord", "SimulatorGateway"]
