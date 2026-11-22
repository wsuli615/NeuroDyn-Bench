"""Build the Gate 1 v0.1 NeuroDyn-Bench task suite (8 core tasks + 4 clean paired controls).

Generates independent randomized `dev` and `test` instances for:
  Small E-I (6 core + 3 clean controls):
    - T02_small_ei_routine            (Type 1 Routine)
    - T03_small_ei_bound_semantics    (Type 2 Dual-Bound Semantics) + _clean
    - T05_small_ei_high_variance      (Type 2 High Stochastic Variance) + _clean
    - T08_small_ei_ambiguous_stall    (Type 2 Multi-Cause Ambiguous Control for T11) + _clean
    - T11_small_ei_cbn_vs_sbn_3b      (Type 3b Fixed-Structure Unreachable, SNOPS prototype)
    - T13_small_ei_underconstrained_3a (Type 3a Single-Condition Non-Identifiable)
  FastDMF Whole-Brain (2 core + 1 clean control):
    - T04_fastdmf_highdim_scale       (Type 2 16D Sloppiness + Linear Scale Mismatch) + _clean
    - T14_fastdmf_fc_compression_3a   (Type 3a Static-FC Summary Compression Degeneracy)

Usage:
    .venv/bin/python scripts/build_v0_tasks.py --splits dev test
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")

import argparse
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from neurodyn_bench.models.fastdmf_env import FastDMFSimulator, compute_fc_and_fcd
from neurodyn_bench.models.small_ei import SmallEINetworkSimulator
from neurodyn_bench.tasks.loader import (
    apply_condition,
    instantiate_simulator,
    save_task_bundle,
)
from neurodyn_bench.tasks.schema import (
    BudgetSpec,
    ConditionRow,
    ObservableRow,
    ParameterRow,
    PrivateTaskBundle,
    PublicTaskBundle,
    SimulatorConfig,
    TaskMetadata,
)


def _sim_worker(args: tuple[dict[str, Any], dict[str, float], int]) -> dict[str, Any]:
    sim_cfg_dict, params, seed = args
    sim = instantiate_simulator(SimulatorConfig(**sim_cfg_dict))
    res = sim.simulate(params, seed=seed, duration_scale=1.0)
    return {
        "feasible": res.feasible,
        "failure_reason": res.failure_reason,
        "observables": res.observables,
    }


def _fastdmf_raw_worker(args: tuple[dict[str, Any], dict[str, float], int]) -> dict[str, Any]:
    sim_cfg_dict, params, seed = args
    sim = FastDMFSimulator(**sim_cfg_dict.get("kwargs", {}))
    rb, bold, _ = sim.simulate_raw(params, seed=seed, duration_scale=1.0)
    fc, fcd = compute_fc_and_fcd(bold, burn_in_trs=sim.burn_in_trs)
    res = sim.simulate(params, seed=seed, duration_scale=1.0)
    return {
        "feasible": res.feasible,
        "observables": res.observables,
        "fc": fc,
        "fcd": fcd,
        "mean_rate": float(np.mean(rb)),
    }


def _collect_target_stats(
    sim_cfg: SimulatorConfig,
    gt_params: dict[str, float],
    obs_ids: list[str],
    condition: ConditionRow,
    seeds: list[int],
    ex: ProcessPoolExecutor,
    sd_floor_rel: float = 0.04,
    sd_floor_abs: dict[str, float] | None = None,
    tolerance_tau: float = 2.0,
) -> list[ObservableRow]:
    eff_p = apply_condition(gt_params, condition)
    runs = list(ex.map(_sim_worker, [(sim_cfg.model_dump(), eff_p, s) for s in seeds]))
    feas = [r for r in runs if r["feasible"]]
    if len(feas) < max(3, len(seeds) // 2):
        raise RuntimeError(
            f"Ground truth infeasible for condition '{condition.condition_id}': "
            f"{len(feas)}/{len(seeds)} feasible. Failures: {[r['failure_reason'] for r in runs if not r['feasible']]}"
        )

    abs_floors = sd_floor_abs or {
        "fr_e": 0.4,
        "isi_cv": 0.03,
        "ff": 0.04,
        "rsc": 0.006,
        "pct_sh": 1.2,
        "dsh": 0.4,
        "es_top1": 0.05,
        "mean_rate_hz": 0.08,
        "max_rate_hz": 0.15,
        "bold_std_mean": 0.0001,
        "fc_mean": 0.015,
        "fc_std": 0.01,
        "fc_corr": 0.04,
        "fcd_mean": 0.025,
        "fcd_std": 0.02,
        "fcd_ks": 0.04,
    }

    rows: list[ObservableRow] = []
    for oid in obs_ids:
        vals = [
            r["observables"][oid]
            for r in feas
            if oid in r["observables"] and math.isfinite(r["observables"][oid])
        ]
        if not vals:
            raise RuntimeError(f"Observable '{oid}' missing/NaN in ground truth runs")
        m = float(np.mean(vals))
        emp_sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        sd = max(emp_sd, abs(m) * sd_floor_rel, abs_floors.get(oid, 1e-3))
        rows.append(
            ObservableRow(
                obs_id=oid,
                condition_id=condition.condition_id,
                target_mean=round(m, 5),
                target_sd=round(sd, 5),
                weight=1.0,
                tolerance_tau=tolerance_tau,
            )
        )
    return rows


def _base_small_ei_param_rows(active_set: set[str]) -> list[ParameterRow]:
    sim = SmallEINetworkSimulator(topology="cbn")
    rows = []
    for spec in sim.parameter_specs():
        rows.append(
            ParameterRow(
                name=spec.name,
                unit=spec.unit,
                default_lb=spec.default_lb,
                default_ub=spec.default_ub,
                phys_hard_lb=spec.phys_hard_lb,
                phys_hard_ub=spec.phys_hard_ub,
                scale=spec.scale,  # type: ignore[arg-type]
                nominal=float(spec.nominal or 1.0),
                active=spec.name in active_set,
                description=spec.description,
            )
        )
    return rows


def _base_fastdmf_param_rows(
    active_set: set[str], n_hetero_groups: int = 0
) -> list[ParameterRow]:
    sim = FastDMFSimulator(n_hetero_groups=n_hetero_groups)
    rows = []
    for spec in sim.parameter_specs():
        rows.append(
            ParameterRow(
                name=spec.name,
                unit=spec.unit,
                default_lb=spec.default_lb,
                default_ub=spec.default_ub,
                phys_hard_lb=spec.phys_hard_lb,
                phys_hard_ub=spec.phys_hard_ub,
                scale=spec.scale,  # type: ignore[arg-type]
                nominal=float(spec.nominal or 1.0),
                active=spec.name in active_set,
                description=spec.description,
            )
        )
    return rows


def _override_param_rows(
    rows: list[ParameterRow], updates: dict[str, dict[str, Any]]
) -> list[ParameterRow]:
    out = []
    for r in rows:
        if r.name in updates:
            out.append(r.model_copy(update=updates[r.name]))
        else:
            out.append(r)
    return out


def build_split(
    split: str,
    out_root: Path,
    ex: ProcessPoolExecutor,
    models: set[str] | None = None,
) -> list[str]:
    """Generate all 8 core tasks (+ 4 *_clean paired controls) for one split (dev or test)."""
    allowed = models or {"small_ei", "fastdmf"}
    is_dev = split == "dev"
    seed_offset = 10000 if is_dev else 20000
    gt_seeds = [seed_offset + i for i in range(8)]
    held_seeds = [seed_offset + 500 + i for i in range(8)]
    split_dir = out_root / split
    built_ids: list[str] = []

    ei_nom = SmallEINetworkSimulator(topology="cbn").default_params()
    baseline_cond = ConditionRow(condition_id="baseline", param_overrides={}, description="Baseline resting/background condition")

    if "small_ei" in allowed:
        _build_small_ei_split(split, is_dev, seed_offset, gt_seeds, held_seeds, split_dir, ei_nom, baseline_cond, ex, built_ids)
    if "fastdmf" in allowed:
        _build_fastdmf_split(split, is_dev, seed_offset, gt_seeds, held_seeds, split_dir, baseline_cond, ex, built_ids)
    return built_ids


def _build_small_ei_split(
    split: str,
    is_dev: bool,
    seed_offset: int,
    gt_seeds: list[int],
    held_seeds: list[int],
    split_dir: Path,
    ei_nom: dict[str, float],
    baseline_cond: ConditionRow,
    ex: ProcessPoolExecutor,
    built_ids: list[str],
) -> None:

    # ------------------------------------------------------------------
    # T02: Small E-I Routine Calibration (Type 1)
    # ------------------------------------------------------------------
    t02_id = "T02_small_ei_routine"
    t02_gt = {
        **ei_nom,
        "J_ee": 1.48 if is_dev else 1.65,
        "J_ie": 1.95 if is_dev else 1.75,
        "g_ei": 5.85 if is_dev else 6.25,
        "tau_i": 9.8 if is_dev else 11.2,
        "nu_ext_e": 6.0 if is_dev else 6.2,
    }
    t02_sim_cfg = SimulatorConfig(model_type="small_ei", kwargs={"topology": "cbn", "duration_ms": 10000.0})
    t02_held_cond = ConditionRow(
        condition_id="high_drive",
        param_overrides={"nu_ext_e_mult": 1.20},
        description="Counterfactual +20% excitatory external drive",
    )
    t02_obs = _collect_target_stats(
        t02_sim_cfg, t02_gt, ["fr_e", "fr_i", "isi_cv", "ff", "rsc", "pct_sh"], baseline_cond, gt_seeds, ex
    )
    t02_held_obs = _collect_target_stats(
        t02_sim_cfg, t02_gt, ["fr_e", "fr_i", "isi_cv", "ff"], t02_held_cond, held_seeds, ex, tolerance_tau=2.5
    )
    t02_params = _override_param_rows(
        _base_small_ei_param_rows({"J_ee", "J_ie", "g_ei", "tau_i"}),
        {"nu_ext_e": {"nominal": t02_gt["nu_ext_e"], "active": False}},
    )
    save_task_bundle(
        split_dir / t02_id,
        PublicTaskBundle(
            metadata=TaskMetadata(
                task_id=t02_id,
                title="Small E-I Routine Balanced-Regime Calibration",
                split=split,  # type: ignore[arg-type]
                model_type="small_ei",
                syllabus_module="M1/M2",
                prompt=(
                    "Calibrate the 4 active intrinsic synaptic parameters (J_ee, J_ie, g_ei, tau_i) "
                    "of a fixed 1,000-neuron Classical Balanced Network (CBN) under known external drive "
                    "to match baseline population statistics within budget. Verify reproducibility before submitting."
                ),
            ),
            parameters=t02_params,
            observables=t02_obs,
            conditions=[baseline_cond],
            budget=BudgetSpec(max_units=120.0, short_probe_scale=0.2, default_n_repeats=1),
            simulator_config=t02_sim_cfg,
        ),
        PrivateTaskBundle(
            task_id=t02_id,
            task_type="type1_routine",
            ground_truth_params=t02_gt,
            gt_simulator_config=t02_sim_cfg,
            heldout_conditions=[t02_held_cond],
            heldout_observables=t02_held_obs,
            trap_metadata={
                "oracle_active_params": ["J_ee", "J_ie", "g_ei", "tau_i"],
                "oracle_optimizer": "cmaes",
            },
        ),
    )
    built_ids.append(t02_id)
    print(f"  Built {split}/{t02_id}", flush=True)

    # ------------------------------------------------------------------
    # T03: Small E-I Bound Truncation + Physiological Hard Bound (Type 2) + _clean
    # ------------------------------------------------------------------
    t03_id = "T03_small_ei_bound_semantics"
    t03_gt = {
        **ei_nom,
        "J_ee": 1.40 if is_dev else 1.55,
        "g_ei": 10.4 if is_dev else 11.2,
        "tau_e": 3.6 if is_dev else 3.4,
        "nu_ext_e": 8.2 if is_dev else 8.8,
    }
    t03_sim_cfg = SimulatorConfig(model_type="small_ei", kwargs={"topology": "cbn", "duration_ms": 10000.0})
    t03_held_cond = ConditionRow(
        condition_id="high_drive",
        param_overrides={"nu_ext_e_mult": 1.25},
        description="Counterfactual +25% excitatory external drive",
    )
    t03_obs = _collect_target_stats(
        t03_sim_cfg, t03_gt, ["fr_e", "fr_i", "isi_cv", "ff", "rsc", "pct_sh"], baseline_cond, gt_seeds, ex
    )
    t03_held_obs = _collect_target_stats(
        t03_sim_cfg, t03_gt, ["fr_e", "fr_i", "isi_cv", "ff"], t03_held_cond, held_seeds, ex, tolerance_tau=2.5
    )
    t03_active = {"J_ee", "g_ei", "tau_e", "nu_ext_e"}
    # Trapped default box: g_ei capped at 5.8 (far below GT ~10.4-11.2, legally expandable to 20.0);
    # tau_e default_lb=3.0 equals phys_hard_lb=3.0 (AMPA biophysical floor, must not be violated!).
    t03_params_trap = _override_param_rows(
        _base_small_ei_param_rows(t03_active),
        {
            "g_ei": {"default_lb": 2.0, "default_ub": 5.8, "phys_hard_ub": 20.0, "nominal": 4.2},
            "tau_e": {"default_lb": 3.0, "default_ub": 10.0, "phys_hard_lb": 3.0, "scale": "log", "nominal": 5.5},
            "nu_ext_e": {"default_lb": 3.5, "default_ub": 11.5, "phys_hard_ub": 25.0, "nominal": 6.5},
        },
    )
    t03_params_clean = _override_param_rows(
        _base_small_ei_param_rows(t03_active),
        {
            "g_ei": {"default_lb": 2.0, "default_ub": 14.0, "phys_hard_ub": 20.0, "nominal": 8.0},
            "tau_e": {"default_lb": 3.0, "default_ub": 10.0, "phys_hard_lb": 3.0, "scale": "log", "nominal": 5.5},
            "nu_ext_e": {"default_lb": 3.5, "default_ub": 11.5, "phys_hard_ub": 25.0, "nominal": 6.5},
        },
    )
    for variant_id, p_rows, is_clean in [
        (t03_id, t03_params_trap, False),
        (f"{t03_id}_clean", t03_params_clean, True),
    ]:
        save_task_bundle(
            split_dir / variant_id,
            PublicTaskBundle(
                metadata=TaskMetadata(
                    task_id=variant_id,
                    title="Small E-I Dual-Bound Semantics Calibration" + (" (Clean Control)" if is_clean else ""),
                    split=split,  # type: ignore[arg-type]
                    model_type="small_ei",
                    syllabus_module="M1",
                    paired_with=f"{t03_id}_clean" if not is_clean else t03_id,
                    is_clean_control=is_clean,
                    prompt=(
                        "Calibrate the 4 active parameters of a fixed CBN spiking network to match baseline "
                        "observables. Initial default bounds reflect legacy guesses and may be expanded "
                        "only within the biophysical hard bounds (phys_hard_lb, phys_hard_ub)."
                    ),
                ),
                parameters=p_rows,
                observables=t03_obs,
                conditions=[baseline_cond],
                budget=BudgetSpec(max_units=160.0, short_probe_scale=0.2, default_n_repeats=1),
                simulator_config=t03_sim_cfg,
            ),
            PrivateTaskBundle(
                task_id=variant_id,
                task_type="type1_routine" if is_clean else "type2_recoverable",
                ground_truth_params=t03_gt,
                gt_simulator_config=t03_sim_cfg,
                heldout_conditions=[t03_held_cond],
                heldout_observables=t03_held_obs,
                trap_metadata={
                    "trap_type": "none" if is_clean else "bound_truncation_with_hard_bound",
                    "oracle_active_params": ["J_ee", "g_ei", "tau_e", "nu_ext_e"],
                    "oracle_bounds": {"g_ei": [2.0, 14.0], "tau_e": [3.0, 10.0], "nu_ext_e": [3.5, 11.5]},
                    "oracle_scales": {"tau_e": "log"},
                },
            ),
        )
        built_ids.append(variant_id)
        print(f"  Built {split}/{variant_id}", flush=True)

    # ------------------------------------------------------------------
    # T05: Small E-I High Stochastic Variance (Type 2) + _clean
    # ------------------------------------------------------------------
    t05_id = "T05_small_ei_high_variance"
    t05_gt = {
        **ei_nom,
        "J_ee": 1.55 if is_dev else 1.40,
        "J_ie": 1.85 if is_dev else 2.10,
        "g_ei": 6.10 if is_dev else 5.65,
        "tau_i": 10.8 if is_dev else 9.2,
        "nu_ext_e": 6.0 if is_dev else 5.9,
    }
    # Short 1200ms simulation window creates large single-trial variance in ff, rsc, pct_sh, isi_cv;
    # compute unbiased target_mean from 16 independent seeds AT the exact same horizon, with target_sd
    # representing multi-repeat precision (~emp_sd / 1.6) so 1-repeat G1 chases noise while repeat-averaging passes.
    t05_sim_trap = SimulatorConfig(model_type="small_ei", kwargs={"topology": "cbn", "duration_ms": 1200.0})
    t05_sim_clean = SimulatorConfig(model_type="small_ei", kwargs={"topology": "cbn", "duration_ms": 10000.0})
    t05_held_cond = ConditionRow(
        condition_id="high_drive",
        param_overrides={"nu_ext_e_mult": 1.20},
        description="Counterfactual +20% external drive",
    )
    t05_many_seeds = [seed_offset + 200 + i for i in range(16)]
    t05_obs_trap_raw = _collect_target_stats(
        t05_sim_trap,
        t05_gt,
        ["fr_e", "fr_i", "isi_cv", "ff", "rsc", "pct_sh"],
        baseline_cond,
        t05_many_seeds,
        ex,
        sd_floor_rel=0.035,
        sd_floor_abs={"fr_e": 0.35, "fr_i": 0.35, "isi_cv": 0.03, "ff": 0.04, "rsc": 0.008, "pct_sh": 1.5},
    )
    t05_obs_trap = [
        o.model_copy(
            update={
                "target_sd": round(max(o.target_sd / 1.5, 0.006 if o.obs_id == "rsc" else 0.03), 5)
            }
        )
        if o.obs_id in ("ff", "rsc", "pct_sh", "isi_cv")
        else o
        for o in t05_obs_trap_raw
    ]
    t05_obs_clean = _collect_target_stats(
        t05_sim_clean,
        t05_gt,
        ["fr_e", "fr_i", "isi_cv", "ff", "rsc", "pct_sh"],
        baseline_cond,
        gt_seeds,
        ex,
    )
    t05_held_obs_trap = _collect_target_stats(
        t05_sim_trap, t05_gt, ["fr_e", "fr_i", "isi_cv", "ff"], t05_held_cond, held_seeds, ex, tolerance_tau=2.5
    )
    t05_held_obs_clean = _collect_target_stats(
        t05_sim_clean, t05_gt, ["fr_e", "fr_i", "isi_cv", "ff"], t05_held_cond, held_seeds, ex, tolerance_tau=2.5
    )
    t05_params = _override_param_rows(
        _base_small_ei_param_rows({"J_ee", "J_ie", "g_ei", "tau_i"}),
        {"nu_ext_e": {"nominal": t05_gt["nu_ext_e"], "active": False}},
    )
    for variant_id, scfg, obs_list, held_list, is_clean in [
        (t05_id, t05_sim_trap, t05_obs_trap, t05_held_obs_trap, False),
        (f"{t05_id}_clean", t05_sim_clean, t05_obs_clean, t05_held_obs_clean, True),
    ]:
        save_task_bundle(
            split_dir / variant_id,
            PublicTaskBundle(
                metadata=TaskMetadata(
                    task_id=variant_id,
                    title="Small E-I High Stochastic Variance Calibration" + (" (Clean Control)" if is_clean else ""),
                    split=split,  # type: ignore[arg-type]
                    model_type="small_ei",
                    syllabus_module="M2",
                    paired_with=f"{t05_id}_clean" if not is_clean else t05_id,
                    is_clean_control=is_clean,
                    prompt=(
                        "Calibrate the 4 active intrinsic synaptic parameters (J_ee, J_ie, g_ei, tau_i) "
                        "of a stochastic CBN spiking network. Single-trial evaluations exhibit substantial "
                        "seed-to-seed variance; ensure submitted parameters generalize across fresh random seeds."
                    ),
                ),
                parameters=t05_params,
                observables=obs_list,
                conditions=[baseline_cond],
                budget=BudgetSpec(max_units=200.0, short_probe_scale=0.35, default_n_repeats=1),
                simulator_config=scfg,
            ),
            PrivateTaskBundle(
                task_id=variant_id,
                task_type="type1_routine" if is_clean else "type2_recoverable",
                ground_truth_params=t05_gt,
                gt_simulator_config=scfg,
                heldout_conditions=[t05_held_cond],
                heldout_observables=held_list,
                trap_metadata={
                    "trap_type": "none" if is_clean else "high_stochastic_variance",
                    "oracle_active_params": ["J_ee", "J_ie", "g_ei", "tau_i"],
                    "oracle_optimizer": "bo_cmaes",
                    "oracle_adaptive_repeat_threshold": 2.5 if not is_clean else None,
                    "oracle_n_repeats": 4 if not is_clean else 1,
                },
            ),
        )
        built_ids.append(variant_id)
        print(f"  Built {split}/{variant_id}", flush=True)

    # ------------------------------------------------------------------
    # T08: Small E-I Multi-Cause Ambiguous Stall (Type 2, 1:1 control paired with T11) + _clean
    # ------------------------------------------------------------------
    t08_id = "T08_small_ei_ambiguous_stall"
    t08_gt = {
        **ei_nom,
        "J_ee": 1.45 if is_dev else 1.60,
        "J_ie": 1.90 if is_dev else 2.15,
        "g_ei": 9.4 if is_dev else 10.2,
        "tau_i": 14.5 if is_dev else 15.8,
        "nu_ext_e": 7.5 if is_dev else 8.0,
    }
    t08_sim_trap = SimulatorConfig(model_type="small_ei", kwargs={"topology": "cbn", "duration_ms": 3000.0})
    t08_sim_clean = SimulatorConfig(model_type="small_ei", kwargs={"topology": "cbn", "duration_ms": 10000.0})
    t08_held_cond = ConditionRow(
        condition_id="high_drive",
        param_overrides={"nu_ext_e_mult": 1.20},
        description="Counterfactual +20% external drive",
    )
    t08_obs_trap = _collect_target_stats(
        t08_sim_trap, t08_gt, ["fr_e", "fr_i", "isi_cv", "ff", "rsc", "pct_sh"], baseline_cond, t05_many_seeds, ex
    )
    t08_obs_clean = _collect_target_stats(
        t08_sim_clean, t08_gt, ["fr_e", "fr_i", "isi_cv", "ff", "rsc", "pct_sh"], baseline_cond, gt_seeds, ex
    )
    t08_held_obs_trap = _collect_target_stats(
        t08_sim_trap, t08_gt, ["fr_e", "fr_i", "isi_cv", "ff"], t08_held_cond, held_seeds, ex, tolerance_tau=2.5
    )
    t08_held_obs_clean = _collect_target_stats(
        t08_sim_clean, t08_gt, ["fr_e", "fr_i", "isi_cv", "ff"], t08_held_cond, held_seeds, ex, tolerance_tau=2.5
    )
    t08_active = {"J_ee", "J_ie", "g_ei", "tau_i"}
    # Compound trap: narrow g_ei (<= 6.0) + narrow tau_i (<= 10.0) + elevated variance (3000ms)
    t08_params_trap = _override_param_rows(
        _base_small_ei_param_rows(t08_active),
        {
            "g_ei": {"default_lb": 2.0, "default_ub": 6.0, "phys_hard_ub": 20.0, "nominal": 4.5},
            "tau_i": {"default_lb": 4.0, "default_ub": 10.0, "phys_hard_ub": 80.0, "scale": "linear", "nominal": 7.0},
            "nu_ext_e": {"nominal": t08_gt["nu_ext_e"], "active": False},
        },
    )
    t08_params_clean = _override_param_rows(
        _base_small_ei_param_rows(t08_active),
        {
            "g_ei": {"default_lb": 2.0, "default_ub": 14.0, "phys_hard_ub": 20.0, "nominal": 8.0},
            "tau_i": {"default_lb": 4.0, "default_ub": 25.0, "phys_hard_ub": 80.0, "scale": "log", "nominal": 10.0},
            "nu_ext_e": {"nominal": t08_gt["nu_ext_e"], "active": False},
        },
    )
    for variant_id, p_rows, scfg, obs_list, held_list, is_clean in [
        (t08_id, t08_params_trap, t08_sim_trap, t08_obs_trap, t08_held_obs_trap, False),
        (f"{t08_id}_clean", t08_params_clean, t08_sim_clean, t08_obs_clean, t08_held_obs_clean, True),
    ]:
        save_task_bundle(
            split_dir / variant_id,
            PublicTaskBundle(
                metadata=TaskMetadata(
                    task_id=variant_id,
                    title="Small E-I Multi-Cause Ambiguous Stall" + (" (Clean Control)" if is_clean else ""),
                    split=split,  # type: ignore[arg-type]
                    model_type="small_ei",
                    syllabus_module="M2/M3",
                    paired_with="T11_small_ei_cbn_vs_sbn_3b" if not is_clean else t08_id,
                    is_clean_control=is_clean,
                    prompt=(
                        "Calibrate the 4 active intrinsic synaptic parameters (J_ee, J_ie, g_ei, tau_i) "
                        "of a fixed CBN spiking network to match baseline population statistics. "
                        "If optimization stalls at high error, diagnose whether the cause is recoverable "
                        "setup/noise issues or fixed-structure unreachability before acting."
                    ),
                ),
                parameters=p_rows,
                observables=obs_list,
                conditions=[baseline_cond],
                budget=BudgetSpec(max_units=200.0, short_probe_scale=0.25, default_n_repeats=1),
                simulator_config=scfg,
            ),
            PrivateTaskBundle(
                task_id=variant_id,
                task_type="type1_routine" if is_clean else "type2_recoverable",
                ground_truth_params=t08_gt,
                gt_simulator_config=scfg,
                heldout_conditions=[t08_held_cond],
                heldout_observables=held_list,
                trap_metadata={
                    "trap_type": "none" if is_clean else "compound_bound_and_variance_ambiguity",
                    "discrimination_path": [
                        "boundary_slack detects legal slack on g_ei and tau_i",
                        "repeat_variance detects elevated stochastic SD under 3000ms window",
                        "expanding bounds + adaptive repeats resolves stall (not Type 3b)",
                    ],
                    "oracle_active_params": ["J_ee", "J_ie", "g_ei", "tau_i"],
                    "oracle_bounds": {"g_ei": [2.0, 14.0], "tau_i": [4.0, 25.0]},
                    "oracle_scales": {"tau_i": "log"},
                    "oracle_adaptive_repeat_threshold": 2.2 if not is_clean else None,
                },
            ),
        )
        built_ids.append(variant_id)
        print(f"  Built {split}/{variant_id}", flush=True)

    # ------------------------------------------------------------------
    # T11: Small E-I CBN fitting Spatial SBN (Type 3b Structurally Unreachable, SNOPS prototype)
    # ------------------------------------------------------------------
    t11_id = "T11_small_ei_cbn_vs_sbn_3b"
    t11_gt = {
        **ei_nom,
        "J_ee": 1.30 if is_dev else 1.45,
        "g_ei": 5.3 if is_dev else 5.6,
        "tau_i": 8.5 if is_dev else 9.2,
        "nu_ext_e": 5.4 if is_dev else 5.7,
    }
    t11_gt_cfg = SimulatorConfig(
        model_type="small_ei",
        kwargs={
            "topology": "sbn",
            "duration_ms": 10000.0,
            "spatial_input_strength": 6.0 if is_dev else 5.5,
            "n_spatial_modes": 4,
            "spatial_mode_tau_ms": 80.0,
            "spatial_mode_sigma": 0.22,
        },
    )
    t11_pub_cfg = SimulatorConfig(
        model_type="small_ei",
        kwargs={"topology": "cbn", "duration_ms": 10000.0, "spatial_input_strength": 0.0},
    )
    t11_obs = _collect_target_stats(
        t11_gt_cfg, t11_gt, ["fr_e", "ff", "rsc", "pct_sh", "dsh"], baseline_cond, gt_seeds, ex
    )
    t11_active = {"J_ee", "J_ie", "g_ei", "tau_e", "tau_i", "nu_ext_e"}
    save_task_bundle(
        split_dir / t11_id,
        PublicTaskBundle(
            metadata=TaskMetadata(
                task_id=t11_id,
                title="Small E-I Population Covariability Calibration Under Fixed CBN Topology",
                split=split,  # type: ignore[arg-type]
                model_type="small_ei",
                syllabus_module="M3",
                paired_with=t08_id,
                prompt=(
                    "Calibrate the 6 active parameters of a fixed random Classical Balanced Network "
                    "(CBN) to jointly reproduce all 5 single-neuron, pairwise, and Factor Analysis "
                    "population statistics (fr_e, ff, rsc, pct_sh, dsh). If the fixed architecture "
                    "cannot reconcile all observables within physiological bounds, abstain as "
                    "3b_structure_unreachable with quantitative trade-off evidence."
                ),
            ),
            parameters=_base_small_ei_param_rows(t11_active),
            observables=t11_obs,
            conditions=[baseline_cond],
            budget=BudgetSpec(max_units=200.0, short_probe_scale=0.2, default_n_repeats=1),
            simulator_config=t11_pub_cfg,
        ),
        PrivateTaskBundle(
            task_id=t11_id,
            task_type="type3b_structure_unreachable",
            ground_truth_params=t11_gt,
            gt_simulator_config=t11_gt_cfg,
            heldout_conditions=[],
            heldout_observables=[],
            trap_metadata={
                "mismatch_mechanism": "spatial_balanced_network_with_4_correlated_spatial_modes_vs_random_cbn",
                "conflicting_features": ["fr_e/ff", "rsc/pct_sh/dsh"],
            },
        ),
    )
    built_ids.append(t11_id)
    print(f"  Built {split}/{t11_id}", flush=True)

    # ------------------------------------------------------------------
    # T13: Small E-I Single-Condition Under-Constrained Degeneracy (Type 3a)
    # ------------------------------------------------------------------
    t13_id = "T13_small_ei_underconstrained_3a"
    t13_gt = {
        **ei_nom,
        "J_ee": 1.50 if is_dev else 1.72,
        "g_ei": 6.0 if is_dev else 6.6,
        "tau_i": 10.2 if is_dev else 11.8,
        "nu_ext_e": 5.85 if is_dev else 6.15,
    }
    t13_sim_cfg = SimulatorConfig(model_type="small_ei", kwargs={"topology": "cbn", "duration_ms": 10000.0})
    t13_held_cond = ConditionRow(
        condition_id="counterfactual_high_drive_weak_inh",
        param_overrides={"nu_ext_e_mult": 1.35, "g_ei_mult": 0.78},
        description="Counterfactual +35% drive with 22% inhibitory blockade",
    )
    # Expose ONLY 2 scalar baseline observables (fr_e, isi_cv) for 4 strongly coupled parameters!
    t13_obs = _collect_target_stats(
        t13_sim_cfg, t13_gt, ["fr_e", "isi_cv"], baseline_cond, gt_seeds, ex
    )
    t13_held_obs = _collect_target_stats(
        t13_sim_cfg, t13_gt, ["fr_e", "isi_cv", "ff"], t13_held_cond, held_seeds, ex, tolerance_tau=2.0
    )
    save_task_bundle(
        split_dir / t13_id,
        PublicTaskBundle(
            metadata=TaskMetadata(
                task_id=t13_id,
                title="Small E-I Single-Condition Inverse Parameter Identification",
                split=split,  # type: ignore[arg-type]
                model_type="small_ei",
                syllabus_module="M3/M4",
                paired_with=t02_id,
                prompt=(
                    "Infer the 4 active synaptic and external-drive parameters (J_ee, g_ei, tau_i, nu_ext_e) "
                    "of a fixed CBN network from the provided single-condition summary observations so that "
                    "the calibrated model reliably predicts unseen perturbation conditions. If multiple "
                    "disparate parameter sets fit equally well and parameters cannot be uniquely identified, "
                    "abstain as 3a_non_identifiable with quantitative uncertainty/degeneracy evidence."
                ),
            ),
            parameters=_base_small_ei_param_rows({"J_ee", "g_ei", "tau_i", "nu_ext_e"}),
            observables=t13_obs,
            conditions=[baseline_cond],
            budget=BudgetSpec(max_units=160.0, short_probe_scale=0.2, default_n_repeats=1),
            simulator_config=t13_sim_cfg,
        ),
        PrivateTaskBundle(
            task_id=t13_id,
            task_type="type3a_non_identifiable",
            ground_truth_params=t13_gt,
            gt_simulator_config=t13_sim_cfg,
            heldout_conditions=[t13_held_cond],
            heldout_observables=t13_held_obs,
            trap_metadata={
                "non_identifiability_cause": "4_coupled_parameters_underconstrained_by_2_single_condition_observables",
                "recommended_next_data": "multi-condition external drive sweep or inhibitory perturbation recordings",
            },
        ),
    )
    built_ids.append(t13_id)
    print(f"  Built {split}/{t13_id}", flush=True)


def _build_fastdmf_split(
    split: str,
    is_dev: bool,
    seed_offset: int,
    gt_seeds: list[int],
    held_seeds: list[int],
    split_dir: Path,
    baseline_cond: ConditionRow,
    ex: ProcessPoolExecutor,
    built_ids: list[str],
) -> None:
    # ------------------------------------------------------------------
    # T04: FastDMF 16D Scale Mismatch + Sloppy Regional Parameters (Type 2) + _clean
    # ------------------------------------------------------------------
    t04_id = "T04_fastdmf_highdim_scale"
    dmf_16 = FastDMFSimulator(n_hetero_groups=8, nb_steps=160000)
    dmf16_nom = dmf_16.default_params()
    t04_gt = {
        **dmf16_nom,
        "G": 2.95 if is_dev else 2.55,
        "w": 1.54 if is_dev else 1.28,
        "sigma": 0.0042 if is_dev else 0.0036,
        "fic_alpha": 0.80 if is_dev else 0.70,
    }
    t04_sim_cfg = SimulatorConfig(
        model_type="fastdmf", kwargs={"n_hetero_groups": 8, "nb_steps": 160000, "c_scale": 0.2}
    )
    t04_raw_runs = list(
        ex.map(_fastdmf_raw_worker, [(t04_sim_cfg.model_dump(), t04_gt, s) for s in gt_seeds[:6]])
    )
    feas_raw = [r for r in t04_raw_runs if r["feasible"]]
    if len(feas_raw) < 4:
        raise RuntimeError(f"T04 FastDMF GT infeasible: {len(feas_raw)}/6")
    t04_target_fc = np.mean([r["fc"] for r in feas_raw], axis=0)
    t04_target_fcd = np.concatenate([r["fcd"] for r in feas_raw if r["fcd"].size > 0])

    # Measure GT observable targets WITH reference FC/FCD attached
    t04_tmp_pub = split_dir / t04_id / "public"
    t04_tmp_pub.mkdir(parents=True, exist_ok=True)
    np.save(t04_tmp_pub / "target_fc.npy", t04_target_fc)
    np.save(t04_tmp_pub / "target_fcd_triu.npy", t04_target_fcd)
    t04_sim_with_ref = t04_sim_cfg.model_copy(
        update={"target_fc_file": "target_fc.npy", "target_fcd_file": "target_fcd_triu.npy"}
    )

    # Evaluate GT on 6 fresh seeds for exact target_mean/target_sd including fc_corr and fcd_ks
    dmf_eval_t04 = instantiate_simulator(t04_sim_with_ref, public_dir=t04_tmp_pub)
    t04_eval_runs = [dmf_eval_t04.simulate(t04_gt, seed=seed_offset + 50 + i) for i in range(6)]
    t04_obs_ids = ["mean_rate_hz", "max_rate_hz", "bold_std_mean", "fc_mean", "fc_std", "fc_corr", "fcd_ks"]
    t04_obs: list[ObservableRow] = []
    for oid in t04_obs_ids:
        vals = [r.observables[oid] for r in t04_eval_runs if r.feasible and oid in r.observables]
        m = float(np.mean(vals))
        emp_sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        floors = {
            "mean_rate_hz": 0.10,
            "max_rate_hz": 0.20,
            "bold_std_mean": 0.00012,
            "fc_mean": 0.02,
            "fc_std": 0.012,
            "fc_corr": 0.05,
            "fcd_ks": 0.06,
        }
        sd = max(emp_sd, abs(m) * 0.05, floors[oid])
        t04_obs.append(
            ObservableRow(
                obs_id=oid,
                condition_id="baseline",
                target_mean=round(m, 5),
                target_sd=round(sd, 5),
                weight=1.0,
                tolerance_tau=2.2,
            )
        )

    t04_held_cond = ConditionRow(
        condition_id="coupled_up",
        param_overrides={"G_mult": 1.15},
        description="Counterfactual +15% global coupling perturbation",
    )
    t04_held_obs = _collect_target_stats(
        t04_sim_cfg, t04_gt, ["mean_rate_hz", "bold_std_mean", "fc_mean"], t04_held_cond, held_seeds[:5], ex, tolerance_tau=2.5
    )

    all_16_names = {s.name for s in dmf_16.parameter_specs()}
    # Trapped 16D: all 16 params active, sigma/taon/taog set to linear scale with wide upper range
    t04_params_trap = _override_param_rows(
        _base_fastdmf_param_rows(all_16_names, n_hetero_groups=8),
        {
            "sigma": {"default_lb": 0.001, "default_ub": 0.014, "scale": "linear", "nominal": 0.0075},
            "taon": {"scale": "linear"},
            "taog": {"scale": "linear"},
        },
    )
    t04_params_clean = _override_param_rows(
        _base_fastdmf_param_rows({"G", "w", "sigma", "fic_alpha"}, n_hetero_groups=8),
        {
            "sigma": {"default_lb": 0.001, "default_ub": 0.012, "scale": "log", "nominal": 0.006},
        },
    )
    for variant_id, p_rows, is_clean in [
        (t04_id, t04_params_trap, False),
        (f"{t04_id}_clean", t04_params_clean, True),
    ]:
        save_task_bundle(
            split_dir / variant_id,
            PublicTaskBundle(
                metadata=TaskMetadata(
                    task_id=variant_id,
                    title="FastDMF 16D Whole-Brain Scale & Sloppy-Subspace Calibration" + (" (Clean Control)" if is_clean else ""),
                    split=split,  # type: ignore[arg-type]
                    model_type="fastdmf",
                    syllabus_module="M1",
                    paired_with=f"{t04_id}_clean" if not is_clean else t04_id,
                    is_clean_control=is_clean,
                    prompt=(
                        "Calibrate a 100-ROI Schaefer whole-brain Dynamic Mean Field (FastDMF) model "
                        "with 16 exposed global and regional parameters to match resting-state firing rates, "
                        "BOLD amplitude, static FC, and FCD within a tight 140-unit simulation budget. "
                        "Screen out sloppy/insensitive dimensions and choose appropriate parameter scales."
                    ),
                ),
                parameters=p_rows,
                observables=t04_obs,
                conditions=[baseline_cond],
                budget=BudgetSpec(max_units=140.0, short_probe_scale=0.25, default_n_repeats=1),
                simulator_config=t04_sim_with_ref,
            ),
            PrivateTaskBundle(
                task_id=variant_id,
                task_type="type1_routine" if is_clean else "type2_recoverable",
                ground_truth_params=t04_gt,
                gt_simulator_config=t04_sim_cfg,
                heldout_conditions=[t04_held_cond],
                heldout_observables=t04_held_obs,
                trap_metadata={
                    "trap_type": "none" if is_clean else "highdim_sloppiness_and_linear_scale_mismatch",
                    "oracle_active_params": ["G", "w", "sigma", "fic_alpha"],
                    "oracle_optimizer": "bo_cmaes",
                    "oracle_scales": {"sigma": "log"},
                    "oracle_bounds": {"sigma": [0.001, 0.012]},
                },
            ),
            target_fc=t04_target_fc,
            target_fcd_triu=t04_target_fcd,
        )
        built_ids.append(variant_id)
        print(f"  Built {split}/{variant_id}", flush=True)

    # ------------------------------------------------------------------
    # T14: FastDMF Static-FC Compression Non-Identifiability (Type 3a)
    # ------------------------------------------------------------------
    t14_id = "T14_fastdmf_fc_compression_3a"
    dmf_0 = FastDMFSimulator(n_hetero_groups=0, nb_steps=160000)
    dmf0_nom = dmf_0.default_params()
    t14_gt = {
        **dmf0_nom,
        "G": 2.65 if is_dev else 2.35,
        "w": 1.48 if is_dev else 1.38,
        "JN": 0.16 if is_dev else 0.14,
        "sigma": 0.0065 if is_dev else 0.0060,
        "fic_alpha": 0.78 if is_dev else 0.72,
    }
    t14_sim_cfg = SimulatorConfig(
        model_type="fastdmf", kwargs={"n_hetero_groups": 0, "nb_steps": 160000, "c_scale": 0.2}
    )
    t14_raw_runs = list(
        ex.map(_fastdmf_raw_worker, [(t14_sim_cfg.model_dump(), t14_gt, s) for s in gt_seeds[:6]])
    )
    t14_feas_raw = [r for r in t14_raw_runs if r["feasible"]]
    t14_target_fc = np.mean([r["fc"] for r in t14_feas_raw], axis=0)
    t14_target_fcd = np.concatenate([r["fcd"] for r in t14_feas_raw if r["fcd"].size > 0])

    t14_tmp_pub = split_dir / t14_id / "public"
    t14_tmp_pub.mkdir(parents=True, exist_ok=True)
    np.save(t14_tmp_pub / "target_fc.npy", t14_target_fc)
    np.save(t14_tmp_pub / "target_fcd_triu.npy", t14_target_fcd)
    t14_sim_with_ref = t14_sim_cfg.model_copy(
        update={"target_fc_file": "target_fc.npy", "target_fcd_file": "target_fcd_triu.npy"}
    )
    dmf_eval_t14 = instantiate_simulator(t14_sim_with_ref, public_dir=t14_tmp_pub)
    t14_eval_runs = [dmf_eval_t14.simulate(t14_gt, seed=seed_offset + 80 + i) for i in range(6)]

    # Expose ONLY static FC summaries (fc_mean, fc_std, fc_corr) for 5 coupled parameters -> classic 3a compression degeneracy!
    t14_obs: list[ObservableRow] = []
    for oid, floor_val in [("fc_mean", 0.02), ("fc_std", 0.012), ("fc_corr", 0.05)]:
        vals = [r.observables[oid] for r in t14_eval_runs if r.feasible and oid in r.observables]
        m = float(np.mean(vals))
        emp_sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        sd = max(emp_sd, abs(m) * 0.05, floor_val)
        t14_obs.append(
            ObservableRow(
                obs_id=oid,
                condition_id="baseline",
                target_mean=round(m, 5),
                target_sd=round(sd, 5),
                weight=1.0,
                tolerance_tau=2.0,
            )
        )

    t14_held_cond = ConditionRow(
        condition_id="coupling_perturb_1.25x",
        param_overrides={"G_mult": 1.25},
        description="Counterfactual +25% global coupling perturbation exposing firing-rate / FCD divergence",
    )
    t14_held_obs = _collect_target_stats(
        t14_sim_cfg, t14_gt, ["mean_rate_hz", "max_rate_hz", "bold_std_mean"], t14_held_cond, held_seeds[:5], ex, tolerance_tau=2.0
    )
    save_task_bundle(
        split_dir / t14_id,
        PublicTaskBundle(
            metadata=TaskMetadata(
                task_id=t14_id,
                title="FastDMF Whole-Brain Parameter Recovery from Static FC Summary Only",
                split=split,  # type: ignore[arg-type]
                model_type="fastdmf",
                syllabus_module="M3/M4",
                paired_with=t04_id,
                prompt=(
                    "Infer the 5 coupled biophysical parameters (G, w, JN, sigma, fic_alpha) of a "
                    "100-ROI FastDMF model from static resting-state FC summary statistics alone so that "
                    "the recovered biophysical parameters generalize under global-coupling perturbations. "
                    "If static FC compression leaves parameters degenerate/non-identifiable, abstain as "
                    "3a_non_identifiable with quantitative posterior/Fisher/multi-start evidence."
                ),
            ),
            parameters=_base_fastdmf_param_rows({"G", "w", "JN", "sigma", "fic_alpha"}, n_hetero_groups=0),
            observables=t14_obs,
            conditions=[baseline_cond],
            budget=BudgetSpec(max_units=140.0, short_probe_scale=0.25, default_n_repeats=1),
            simulator_config=t14_sim_with_ref,
        ),
        PrivateTaskBundle(
            task_id=t14_id,
            task_type="type3a_non_identifiable",
            ground_truth_params=t14_gt,
            gt_simulator_config=t14_sim_cfg,
            heldout_conditions=[t14_held_cond],
            heldout_observables=t14_held_obs,
            trap_metadata={
                "non_identifiability_cause": "static_fc_summary_compression_over_5_coupled_whole_brain_params",
                "recommended_next_data": "empirical firing-rate/EEG constraints, FCD dynamics, or perturbation conditions",
            },
        ),
        target_fc=t14_target_fc,
        target_fcd_triu=t14_target_fcd,
    )
    built_ids.append(t14_id)
    print(f"  Built {split}/{t14_id}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Gate 1 v0.1 NeuroDyn-Bench task bundles.")
    parser.add_argument("--splits", nargs="+", default=["dev", "test"], choices=["dev", "test"])
    parser.add_argument("--models", nargs="+", default=["small_ei", "fastdmf"], choices=["small_ei", "fastdmf"])
    parser.add_argument("--out", default="tasks", help="Output root directory")
    parser.add_argument("--workers", type=int, default=16, help="Parallel simulation workers")
    args = parser.parse_args()

    out_root = Path(args.out)
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for sp in args.splits:
            print(f"=== Building split: {sp} (models={args.models}) ===", flush=True)
            ids = build_split(sp, out_root, ex, models=set(args.models))
            print(f"Completed {sp}: {len(ids)} task bundles -> {ids}", flush=True)


if __name__ == "__main__":
    main()
