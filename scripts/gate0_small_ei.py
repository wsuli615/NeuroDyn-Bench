"""Gate 0a: Minimal reproduction and performance/variance profiling for Small E-I spiking network.

Measures:
  1. Regime sanity checks (silent, AI balanced, runaway, CBN vs SBN covariability)
  2. Multi-seed stochastic variance across 10 seeds (SNOPS 6 statistics + ISI CV)
  3. Single-simulation runtime & short-screen probe runtime (duration_scale=0.2 vs 1.0)
  4. Multiprocessing scaling on dgx2felis (1 vs 8 vs 32 workers)
  5. Minimal G1 baseline calibration (CMA-ES on 4D twin experiment under BudgetLedger)
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")

import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cma
import numpy as np

from neurodyn_bench.models.small_ei import SmallEINetworkSimulator
from neurodyn_bench.optim.budget import BudgetLedger

_WORKER_SIMS: dict[str, SmallEINetworkSimulator] = {}


def _get_worker_sim(topology: str) -> SmallEINetworkSimulator:
    if topology not in _WORKER_SIMS:
        _WORKER_SIMS[topology] = SmallEINetworkSimulator(topology=topology)
    return _WORKER_SIMS[topology]


def _eval_worker(args: tuple[dict[str, float], int, float, str]) -> dict:
    params, seed, duration_scale, topology = args
    sim = _get_worker_sim(topology)
    res = sim.simulate(params, seed=seed, duration_scale=duration_scale)
    return {
        "feasible": res.feasible,
        "failure_reason": res.failure_reason,
        "wall_time_s": res.wall_time_s,
        "sim_units": res.sim_units,
        "seed": res.seed,
        "observables": res.observables,
        "es_len": len(res.vector_observables.get("es", [])),
    }


def main() -> None:
    print("=== Gate 0a: Small E-I Spiking Network (CBN & SBN) ===", flush=True)
    cbn = SmallEINetworkSimulator(topology="cbn")
    sbn = SmallEINetworkSimulator(topology="sbn")

    # Warm up Numba JIT cache
    t_jit0 = time.perf_counter()
    _ = cbn.simulate(cbn.default_params(), seed=0, duration_scale=0.2)
    t_jit = time.perf_counter() - t_jit0
    print(f"Numba JIT warm-up: {t_jit:.2f} s", flush=True)

    # 1. Regime sanity checks (serial for exact single-core timing)
    regimes = {
        "cbn_ai_nominal": (cbn, cbn.default_params()),
        "cbn_silent": (cbn, {**cbn.default_params(), "nu_ext_e": 1.2, "g_ei": 9.0}),
        "cbn_runaway": (cbn, {**cbn.default_params(), "g_ei": 1.2, "J_ee": 2.2}),
        "sbn_nominal": (sbn, sbn.default_params()),
    }
    regime_summary = {}
    for name, (sim, params) in regimes.items():
        r = sim.simulate(params, seed=7, duration_scale=1.0)
        regime_summary[name] = {
            "feasible": r.feasible,
            "failure_reason": r.failure_reason,
            "wall_time_s": round(r.wall_time_s, 4),
            "observables": {
                k: round(v, 4) if np.isfinite(v) else None for k, v in r.observables.items()
            },
            "es_len": len(r.vector_observables.get("es", [])),
        }
        print(
            f"  [{name}] feasible={r.feasible} ({r.failure_reason}), t={r.wall_time_s:.3f}s, "
            f"obs={regime_summary[name]['observables']}",
            flush=True,
        )

    # 2. Single-core baseline timings (5 full + 5 short serial) + 10-seed variance (parallel)
    nom = cbn.default_params()
    serial_full_times = [cbn.simulate(nom, seed=s, duration_scale=1.0).wall_time_s for s in range(5)]
    serial_short_times = [cbn.simulate(nom, seed=s, duration_scale=0.2).wall_time_s for s in range(5)]

    with ProcessPoolExecutor(max_workers=10) as ex:
        cbn_10 = list(ex.map(_eval_worker, [(nom, s, 1.0, "cbn") for s in range(10)]))
        sbn_10 = list(ex.map(_eval_worker, [(nom, s, 1.0, "sbn") for s in range(10)]))

    obs_keys = ["fr_e", "fr_i", "isi_cv", "ff", "rsc", "pct_sh", "dsh", "es_top1", "pop_psd_peak_hz"]

    def _summarize(runs: list[dict]) -> dict:
        out = {}
        for k in obs_keys:
            vals = [
                r["observables"][k]
                for r in runs
                if r["feasible"] and k in r["observables"] and np.isfinite(r["observables"][k])
            ]
            if vals:
                m = float(np.mean(vals))
                sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
                cv = abs(sd / m) if abs(m) > 1e-9 else float("nan")
                out[k] = {
                    "mean": round(m, 4),
                    "sd": round(sd, 4),
                    "rel_cv": round(cv, 4) if np.isfinite(cv) else None,
                }
        return out

    cbn_var = _summarize(cbn_10)
    sbn_var = _summarize(sbn_10)

    print(
        f"\nSingle-core full sim (10s eval, 1000 LIF): median={np.median(serial_full_times):.3f}s, "
        f"mean={np.mean(serial_full_times):.3f}s",
        flush=True,
    )
    print(
        f"Single-core short probe (2s eval, 0.2u): median={np.median(serial_short_times):.3f}s, "
        f"mean={np.mean(serial_short_times):.3f}s",
        flush=True,
    )
    print("Multi-seed stochasticity (10 seeds, CBN vs SBN):", flush=True)
    for k in ("fr_e", "isi_cv", "ff", "rsc", "pct_sh", "dsh", "es_top1"):
        c_s = cbn_var.get(k, {})
        s_s = sbn_var.get(k, {})
        print(
            f"  {k:10s} | CBN: {c_s.get('mean', 'nan'):>7} ± {c_s.get('sd', 'nan'):<6} "
            f"| SBN: {s_s.get('mean', 'nan'):>7} ± {s_s.get('sd', 'nan'):<6}",
            flush=True,
        )

    # 3. CPU parallel scaling benchmark (16 sims across 1, 8, 16, 32 workers)
    # Note: 1-worker extrapolated cleanly from 5 serial sims to save time; 8/16/32 measured live with 32 sims
    scaling = {
        "1_worker_single_core": {
            "median_s_per_sim": round(float(np.median(serial_full_times)), 3),
            "throughput_sims_per_s": round(1.0 / float(np.median(serial_full_times)), 2),
        }
    }
    for n_workers in (8, 16, 32):
        tasks = [(nom, s, 1.0, "cbn") for s in range(32)]
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            # Pre-warm worker pool JIT import with 1 tiny probe per worker? Better: measure steady batch
            t0 = time.perf_counter()
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
        "J_ee": 1.55,
        "g_ei": 6.1,
        "tau_i": 10.5,
        "nu_ext_e": 6.0,
    }
    with ProcessPoolExecutor(max_workers=8) as ex:
        gt_runs = list(ex.map(_eval_worker, [(gt_params, 100 + s, 1.0, "cbn") for s in range(5)]))

    target_keys = ["fr_e", "ff", "rsc", "pct_sh", "isi_cv"]
    target_mean = {k: float(np.mean([r["observables"][k] for r in gt_runs])) for k in target_keys}
    target_sd = {
        k: max(
            float(np.std([r["observables"][k] for r in gt_runs], ddof=1)),
            abs(target_mean[k]) * 0.05,
            1e-3,
        )
        for k in target_keys
    }

    active_names = ["J_ee", "g_ei", "tau_i", "nu_ext_e"]
    specs = {s.name: s for s in cbn.parameter_specs()}
    lb = np.array([specs[n].default_lb for n in active_names])
    ub = np.array([specs[n].default_ub for n in active_names])

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
    with ProcessPoolExecutor(max_workers=8) as ex:
        while not es.stop() and ledger.can_afford(8.0):
            xs = es.ask()
            batch_cands = []
            batch_tasks = []
            for x in xs:
                cand = {**nom}
                for i, name in enumerate(active_names):
                    cand[name] = float(lb[i] + x[i] * (ub[i] - lb[i]))
                batch_cands.append((cand, 1000 + eval_idx))
                batch_tasks.append((cand, 1000 + eval_idx, 1.0, "cbn"))
                eval_idx += 1

            batch_res = list(ex.map(_eval_worker, batch_tasks))
            losses = []
            for (cand, sd), r_dict in zip(batch_cands, batch_res):
                # Account every evaluation in the unified BudgetLedger
                from neurodyn_bench.models.base import SimulationResult

                sim_res = SimulationResult(
                    observables=r_dict["observables"],
                    feasible=r_dict["feasible"],
                    failure_reason=r_dict["failure_reason"],
                    wall_time_s=r_dict["wall_time_s"],
                    sim_units=r_dict["sim_units"],
                    seed=sd,
                )
                ledger.execute(cbn.name, cand, sd, 1.0, lambda sr=sim_res: sr)

                if not sim_res.feasible:
                    loss = 1e4
                else:
                    z_sq = [
                        ((sim_res.observables[k] - target_mean[k]) / target_sd[k]) ** 2
                        for k in target_keys
                    ]
                    loss = float(np.sqrt(np.mean(z_sq)))
                losses.append(loss)
                if loss < best_loss:
                    best_loss = loss
                    best_theta = cand
            es.tell(xs, losses)
    opt_wall = time.perf_counter() - t_opt0

    verify_z = {}
    if best_theta is not None:
        with ProcessPoolExecutor(max_workers=5) as ex:
            v_runs = list(ex.map(_eval_worker, [(best_theta, 5000 + s, 1.0, "cbn") for s in range(5)]))
        for k in target_keys:
            vals = [r["observables"][k] for r in v_runs if r["feasible"]]
            verify_z[k] = round(abs(float(np.mean(vals)) - target_mean[k]) / target_sd[k], 3)

    print(
        f"\nG1 CMA-ES baseline (budget=80u): best_rms_z={best_loss:.3f}, wall={opt_wall:.2f}s",
        flush=True,
    )
    print(f"  Ledger: {ledger.summary()}", flush=True)
    print(f"  Independent verification |z| (5 fresh seeds): {verify_z}", flush=True)

    out_dir = Path("outputs/gate0")
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "environment": "small_ei",
        "jit_warmup_s": round(t_jit, 3),
        "regimes": regime_summary,
        "full_sim_median_s": round(float(np.median(serial_full_times)), 4),
        "full_sim_mean_s": round(float(np.mean(serial_full_times)), 4),
        "short_probe_median_s": round(float(np.median(serial_short_times)), 4),
        "cbn_10seed_stats": cbn_var,
        "sbn_10seed_stats": sbn_var,
        "parallel_scaling": scaling,
        "baseline_cmaes_twin": {
            "active_params": active_names,
            "gt_params": {k: gt_params[k] for k in active_names},
            "target_mean": {k: round(v, 4) for k, v in target_mean.items()},
            "target_sd": {k: round(v, 4) for k, v in target_sd.items()},
            "best_rms_z": round(best_loss, 4),
            "best_theta": {k: round(best_theta[k], 4) for k in active_names} if best_theta else None,
            "verify_abs_z_5seeds": verify_z,
            "opt_wall_s": round(opt_wall, 3),
            "ledger": ledger.summary(),
        },
    }
    (out_dir / "small_ei_gate0.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved Gate 0a report to {out_dir / 'small_ei_gate0.json'}", flush=True)


if __name__ == "__main__":
    main()
