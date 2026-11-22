"""Gate 0b: Minimal reproduction and performance/variance profiling for FastDMF whole-brain model.

Measures:
  1. Regime sanity checks (low G, nominal working point ~2-3 Hz, high G, broken FIC runaway)
  2. Multi-seed stochastic variance across 10 seeds (FC, FCD, firing rates, BOLD SD)
  3. Single-simulation runtime (160s fMRI = 80 TRs vs 40s short probe = 0.25 units)
  4. Multiprocessing scaling on dgx2felis (1 vs 8 vs 16 vs 32 workers)
  5. Minimal G1 baseline calibration (CMA-ES on 4D global twin experiment under BudgetLedger)
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cma
import numpy as np

from neurodyn_bench.models.base import SimulationResult
from neurodyn_bench.models.fastdmf_env import FastDMFSimulator, compute_fc_and_fcd
from neurodyn_bench.optim.budget import BudgetLedger

_WORKER_SIM: FastDMFSimulator | None = None
_WORKER_TWIN_SIM: FastDMFSimulator | None = None


def _eval_worker(args: tuple[dict[str, float], int, float]) -> dict:
    global _WORKER_SIM
    params, seed, duration_scale = args
    if _WORKER_SIM is None:
        _WORKER_SIM = FastDMFSimulator()
    res = _WORKER_SIM.simulate(params, seed=seed, duration_scale=duration_scale)
    return {
        "feasible": res.feasible,
        "failure_reason": res.failure_reason,
        "wall_time_s": res.wall_time_s,
        "sim_units": res.sim_units,
        "seed": res.seed,
        "observables": res.observables,
    }


def _init_twin_worker(target_fc: np.ndarray, target_fcd: np.ndarray) -> None:
    global _WORKER_TWIN_SIM
    _WORKER_TWIN_SIM = FastDMFSimulator(target_fc=target_fc, target_fcd_triu=target_fcd)


def _eval_twin_worker(args: tuple[dict[str, float], int, float]) -> dict:
    assert _WORKER_TWIN_SIM is not None
    params, seed, duration_scale = args
    res = _WORKER_TWIN_SIM.simulate(params, seed=seed, duration_scale=duration_scale)
    return {
        "feasible": res.feasible,
        "failure_reason": res.failure_reason,
        "wall_time_s": res.wall_time_s,
        "sim_units": res.sim_units,
        "seed": res.seed,
        "observables": res.observables,
    }


def main() -> None:
    print("=== Gate 0b: FastDMF Whole-Brain Dynamic Mean Field (Schaefer-100 HCP) ===", flush=True)
    sim = FastDMFSimulator()
    nom = sim.default_params()

    # 1. Regime check (serial for exact single-core timing)
    regimes = {
        "low_G_0.5": {**nom, "G": 0.5},
        "nominal_G_2.2": nom,
        "high_G_3.5": {**nom, "G": 3.5},
        "broken_fic_runaway": {**nom, "G": 3.5, "w": 2.2, "fic_alpha": 0.12},
    }
    regime_summary = {}
    for name, params in regimes.items():
        r = sim.simulate(params, seed=7, duration_scale=1.0)
        regime_summary[name] = {
            "feasible": r.feasible,
            "failure_reason": r.failure_reason,
            "wall_time_s": round(r.wall_time_s, 4),
            "observables": {
                k: round(v, 4) if np.isfinite(v) else None for k, v in r.observables.items()
            },
        }
        print(
            f"  [{name}] feasible={r.feasible} ({r.failure_reason}), t={r.wall_time_s:.3f}s, "
            f"obs={regime_summary[name]['observables']}",
            flush=True,
        )

    # 2. Single-core baseline timings (3 full + 3 short serial) + 10-seed variance (parallel)
    serial_full_times = [sim.simulate(nom, seed=s, duration_scale=1.0).wall_time_s for s in range(3)]
    serial_short_times = [sim.simulate(nom, seed=s, duration_scale=0.25).wall_time_s for s in range(3)]

    with ProcessPoolExecutor(max_workers=10) as ex:
        full_runs = list(ex.map(_eval_worker, [(nom, s, 1.0) for s in range(10)]))

    obs_keys = ["mean_rate_hz", "max_rate_hz", "bold_std_mean", "fc_mean", "fc_std", "fcd_mean", "fcd_std"]
    variance_table = {}
    for k in obs_keys:
        vals = [
            r["observables"][k]
            for r in full_runs
            if r["feasible"] and k in r["observables"] and np.isfinite(r["observables"][k])
        ]
        if vals:
            mean_v = float(np.mean(vals))
            sd_v = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            cv_v = abs(sd_v / mean_v) if abs(mean_v) > 1e-9 else float("nan")
            variance_table[k] = {
                "mean": round(mean_v, 4),
                "sd": round(sd_v, 4),
                "rel_cv": round(cv_v, 4) if np.isfinite(cv_v) else None,
            }

    print(
        f"\nSingle-core full sim (160s BOLD, 100 ROIs): median={np.median(serial_full_times):.3f}s, "
        f"mean={np.mean(serial_full_times):.3f}s",
        flush=True,
    )
    print(
        f"Single-core short probe (40s BOLD, 0.25u): median={np.median(serial_short_times):.3f}s, "
        f"mean={np.mean(serial_short_times):.3f}s",
        flush=True,
    )
    print("Multi-seed stochasticity (10 seeds, nominal G=2.2):", flush=True)
    for k, stats in variance_table.items():
        print(
            f"  {k:16s}: mean={stats['mean']:8.4f} ± {stats['sd']:7.4f} (CV={stats['rel_cv']})",
            flush=True,
        )

    # 3. CPU parallel scaling (32 sims across 8, 16, 32 workers)
    scaling = {
        "1_worker_single_core": {
            "median_s_per_sim": round(float(np.median(serial_full_times)), 3),
            "throughput_sims_per_s": round(1.0 / float(np.median(serial_full_times)), 3),
        }
    }
    for n_workers in (8, 16, 32):
        tasks = [(nom, s, 1.0) for s in range(32)]
        t0 = time.perf_counter()
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            _ = list(ex.map(_eval_worker, tasks))
        dt = time.perf_counter() - t0
        scaling[f"{n_workers}_workers"] = {
            "total_wall_s_for_32_sims": round(dt, 3),
            "throughput_sims_per_s": round(32.0 / dt, 2),
        }
        print(f"Parallel scaling ({n_workers:2d} workers, 32 sims): {dt:.2f}s ({32.0/dt:.2f} sims/s)", flush=True)

    # 4. Minimal twin-experiment G1 baseline calibration (CMA-ES, budget=80u, parallel popsize=8)
    gt_params = {
        **nom,
        "G": 2.65,
        "w": 1.48,
        "sigma": 0.0065,
        "fic_alpha": 0.78,
    }
    gt_fcs = []
    gt_fcds = []
    gt_rates = []
    for s in range(5):
        rb, bold, _ = sim.simulate_raw(gt_params, seed=200 + s, duration_scale=1.0)
        fc, fcd = compute_fc_and_fcd(bold, burn_in_trs=sim.burn_in_trs)
        gt_fcs.append(fc)
        gt_fcds.append(fcd)
        gt_rates.append(float(np.mean(rb)))

    target_fc = np.mean(gt_fcs, axis=0)
    target_fcd = np.concatenate(gt_fcds)
    target_rate = float(np.mean(gt_rates))

    active_names = ["G", "w", "sigma", "fic_alpha"]
    specs = {s.name: s for s in sim.parameter_specs()}
    lb = np.array([specs[n].default_lb for n in active_names])
    ub = np.array([specs[n].default_ub for n in active_names])

    with ProcessPoolExecutor(
        max_workers=8, initializer=_init_twin_worker, initargs=(target_fc, target_fcd)
    ) as ex:
        gt_floor = list(ex.map(_eval_twin_worker, [(gt_params, 300 + s, 1.0) for s in range(5)]))
        feas_floor = [r for r in gt_floor if r["feasible"] and "fc_corr" in r["observables"]]
        assert feas_floor, f"Ground truth failed feasibility check: {gt_floor}"
        floor_fc_corr = float(np.mean([r["observables"]["fc_corr"] for r in feas_floor]))
        floor_fcd_ks = float(np.mean([r["observables"]["fcd_ks"] for r in feas_floor]))
        print(
            f"\nTwin GT noise floor ({len(feas_floor)}/5 feasible fresh seeds): "
            f"fc_corr={floor_fc_corr:.4f}, fcd_ks={floor_fcd_ks:.4f}, rate={target_rate:.2f} Hz",
            flush=True,
        )

        ledger = BudgetLedger(max_units=80.0)
        x0 = np.full(len(active_names), 0.5)
        es = cma.CMAEvolutionStrategy(
            x0,
            0.25,
            {"bounds": [0.0, 1.0], "popsize": 8, "seed": 2026, "verbose": -9},
        )

        best_loss = float("inf")
        best_theta = None
        t_opt0 = time.perf_counter()
        eval_idx = 0
        while not es.stop() and ledger.can_afford(8.0):
            xs = es.ask()
            batch_cands = []
            batch_tasks = []
            for x in xs:
                cand = {**nom}
                for i, name in enumerate(active_names):
                    cand[name] = float(lb[i] + x[i] * (ub[i] - lb[i]))
                batch_cands.append((cand, 1000 + eval_idx))
                batch_tasks.append((cand, 1000 + eval_idx, 1.0))
                eval_idx += 1

            batch_res = list(ex.map(_eval_twin_worker, batch_tasks))
            losses = []
            for (cand, sd), r_dict in zip(batch_cands, batch_res):
                sim_res = SimulationResult(
                    observables=r_dict["observables"],
                    feasible=r_dict["feasible"],
                    failure_reason=r_dict["failure_reason"],
                    wall_time_s=r_dict["wall_time_s"],
                    sim_units=r_dict["sim_units"],
                    seed=sd,
                )
                ledger.execute(sim.name, cand, sd, 1.0, lambda sr=sim_res: sr)

                if not sim_res.feasible:
                    loss = 1e4
                else:
                    rate_pen = abs(sim_res.observables["mean_rate_hz"] - target_rate) / max(
                        target_rate, 1.0
                    )
                    loss = float(
                        (1.0 - sim_res.observables["fc_corr"])
                        + sim_res.observables["fcd_ks"]
                        + 0.5 * rate_pen
                    )
                losses.append(loss)
                if loss < best_loss:
                    best_loss = loss
                    best_theta = cand
            es.tell(xs, losses)
        opt_wall = time.perf_counter() - t_opt0

        verify = {}
        if best_theta is not None:
            v_runs = list(ex.map(_eval_twin_worker, [(best_theta, 5000 + s, 1.0) for s in range(5)]))
            verify = {
                "fc_corr_mean": round(
                    float(np.mean([r["observables"]["fc_corr"] for r in v_runs if r["feasible"]])), 4
                ),
                "fcd_ks_mean": round(
                    float(np.mean([r["observables"]["fcd_ks"] for r in v_runs if r["feasible"]])), 4
                ),
                "mean_rate_hz": round(
                    float(np.mean([r["observables"]["mean_rate_hz"] for r in v_runs if r["feasible"]])), 4
                ),
            }

    print(
        f"G1 CMA-ES baseline (budget=80u): best_loss={best_loss:.4f}, wall={opt_wall:.2f}s",
        flush=True,
    )
    print(f"  Ledger: {ledger.summary()}", flush=True)
    print(f"  Independent verification (5 fresh seeds): {verify}", flush=True)

    out_dir = Path("outputs/gate0")
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "environment": "fastdmf_schaefer100",
        "n_rois": sim.n_rois,
        "c_scale": sim.c_scale,
        "nb_steps": sim.nb_steps,
        "regimes": regime_summary,
        "full_sim_median_s": round(float(np.median(serial_full_times)), 4),
        "full_sim_mean_s": round(float(np.mean(serial_full_times)), 4),
        "short_probe_median_s": round(float(np.median(serial_short_times)), 4),
        "nominal_10seed_stats": variance_table,
        "parallel_scaling": scaling,
        "baseline_cmaes_twin": {
            "active_params": active_names,
            "gt_params": {k: gt_params[k] for k in active_names},
            "gt_noise_floor": {
                "fc_corr": round(floor_fc_corr, 4),
                "fcd_ks": round(floor_fcd_ks, 4),
                "mean_rate_hz": round(target_rate, 4),
            },
            "best_loss": round(best_loss, 4),
            "best_theta": {k: round(best_theta[k], 4) for k in active_names} if best_theta else None,
            "verify_5seeds": verify,
            "opt_wall_s": round(opt_wall, 3),
            "ledger": ledger.summary(),
        },
    }
    (out_dir / "fastdmf_gate0.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved Gate 0b report to {out_dir / 'fastdmf_gate0.json'}", flush=True)


if __name__ == "__main__":
    main()
