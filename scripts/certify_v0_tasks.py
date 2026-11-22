"""Offline task certification (Rule 2.3) and G1/G2 reverse screening (Rule 2.1) on dgx2felis.

Pools all controller seeds across all tasks concurrently onto one worker pool (starting
heavy FastDMF tasks first) so all 8 Gate 1 tasks certify in parallel with live progress.

Writes `<task_dir>/private/validation_cert.yaml` and `outputs/gate1/<split>_certification.json`.

Usage:
    .venv/bin/python scripts/certify_v0_tasks.py --split dev --workers 64
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")

import argparse
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from neurodyn_bench.agents.g1_optimizer import run_g1_bo, run_g1_cmaes
from neurodyn_bench.agents.g2_rules import run_g2_rules
from neurodyn_bench.agents.oracle import run_oracle
from neurodyn_bench.eval.evaluator import evaluate_submission
from neurodyn_bench.eval.metrics import compute_headline_metrics
from neurodyn_bench.gateway.gateway import SimulatorGateway
from neurodyn_bench.tasks.loader import (
    apply_condition,
    instantiate_simulator,
    load_private_bundle,
    load_public_task,
)
from neurodyn_bench.tools.optimizers import (
    params_to_unit,
    resolve_search_space,
    unit_to_params,
)


def _run_controller_seed(args: tuple[str, str, int, str]) -> dict[str, Any]:
    task_dir, method, seed, trace_dir = args
    t0 = time.perf_counter()
    trace_path = Path(trace_dir) / f"{Path(task_dir).name}_{method}_s{seed}.jsonl"
    gw = SimulatorGateway.from_task_dir(task_dir, disclosure="full", trace_path=trace_path)
    priv = load_private_bundle(task_dir)

    if method == "oracle":
        sub = run_oracle(gw, priv, seed=seed)
    elif method == "g1_bo":
        sub = run_g1_bo(gw, seed=seed)
    elif method == "g1_cmaes":
        sub = run_g1_cmaes(gw, seed=seed)
    elif method == "g2_rules":
        sub = run_g2_rules(gw, seed=seed)
    else:
        raise ValueError(f"Unknown method: {method}")

    report = evaluate_submission(
        task_dir=task_dir,
        submission=sub,
        trace_path=trace_path,
        n_verify_seeds=5,
        verify_seed_start=50000 + seed * 10,
    )
    out = report.to_dict()
    out["method"] = method
    out["seed"] = seed
    out["submission_notes"] = sub.notes
    out["wall_s"] = round(time.perf_counter() - t0, 2)
    return out


def _eval_batch_worker(args: tuple[str, list[dict[str, float]], int]) -> list[dict[str, Any]]:
    task_dir, param_batch, seed_base = args
    pub, _ = load_public_task(task_dir, disclosure="full")
    sim = instantiate_simulator(pub.simulator_config, public_dir=pub.public_dir)
    obs_rows = [o for o in pub.observables if o.condition_id == "baseline"]
    out = []
    for i, cand in enumerate(param_batch):
        res = sim.simulate(cand, seed=seed_base + i, duration_scale=1.0)
        if not res.feasible:
            out.append(
                {
                    "feasible": False,
                    "failure_reason": res.failure_reason,
                    "abs_z": {},
                    "max_abs_z": float("inf"),
                }
            )
            continue
        abs_z = {}
        for o in obs_rows:
            v = res.observables.get(o.obs_id, float("nan"))
            abs_z[o.obs_id] = (
                abs(v - o.target_mean) / max(o.target_sd, 1e-9)
                if math.isfinite(v)
                else float("inf")
            )
        out.append(
            {
                "feasible": True,
                "failure_reason": None,
                "abs_z": abs_z,
                "max_abs_z": max(abs_z.values()) if abs_z else float("inf"),
                "observables": {
                    o.obs_id: round(res.observables[o.obs_id], 4)
                    for o in obs_rows
                    if o.obs_id in res.observables
                },
                "params": cand,
            }
        )
    return out


def certify_type3b_10x(
    task_dir: Path,
    ex: ProcessPoolExecutor,
    multiplier: float = 10.0,
) -> dict[str, Any]:
    """Rule 2.3 Type 3b offline proof: >= 10x budget global search over full phys_hard bounds."""
    pub, _ = load_public_task(task_dir, disclosure="full")
    total_evals = round(pub.budget.max_units * multiplier)
    gw = SimulatorGateway.from_task_dir(task_dir, max_units_override=float(total_evals))
    active_rows = pub.active_parameters()
    hard_bounds = {r.name: (r.phys_hard_lb, r.phys_hard_ub) for r in active_rows}
    rows, eff_bounds, eff_scales = resolve_search_space(
        gw, [r.name for r in active_rows], hard_bounds, None
    )
    base = pub.nominal_dict()
    d = len(rows)
    rng = np.random.default_rng(20261122)

    n_global = total_evals // 2
    lhs = np.zeros((n_global, d), dtype=np.float64)
    for j in range(d):
        perm = rng.permutation(n_global)
        lhs[:, j] = (perm + rng.random(n_global)) / n_global
    global_cands = [unit_to_params(lhs[i], rows, eff_bounds, eff_scales, base) for i in range(n_global)]

    chunk_size = max(1, n_global // 48)
    chunks = [
        (str(task_dir), global_cands[i : i + chunk_size], 80000 + i)
        for i in range(0, n_global, chunk_size)
    ]
    all_res: list[dict[str, Any]] = []
    for batch_out in ex.map(_eval_batch_worker, chunks):
        all_res.extend(batch_out)

    feas_res = [r for r in all_res if r["feasible"] and math.isfinite(r["max_abs_z"])]
    n_refine = total_evals - n_global
    n_rounds = 5
    per_round = max(48, n_refine // n_rounds)
    for r_idx in range(n_rounds):
        feas_res.sort(key=lambda item: item["max_abs_z"])
        elites = feas_res[: min(20, len(feas_res))]
        if not elites:
            break
        elite_u = np.vstack([params_to_unit(e["params"], rows, eff_bounds, eff_scales) for e in elites])
        mean_u = np.mean(elite_u, axis=0)
        sd_u = np.clip(np.std(elite_u, axis=0) * (0.85**r_idx), 0.03, 0.25)
        prop_cands = []
        for _ in range(per_round):
            parent = elite_u[int(rng.integers(0, len(elites)))]
            u_new = np.clip(0.5 * parent + 0.5 * mean_u + rng.normal(0.0, sd_u), 0.0, 1.0)
            prop_cands.append(unit_to_params(u_new, rows, eff_bounds, eff_scales, base))
        c_sz = max(1, len(prop_cands) // 48)
        r_chunks = [
            (str(task_dir), prop_cands[i : i + c_sz], 90000 + r_idx * 2000 + i)
            for i in range(0, len(prop_cands), c_sz)
        ]
        for batch_out in ex.map(_eval_batch_worker, r_chunks):
            all_res.extend(batch_out)
            feas_res.extend([x for x in batch_out if x["feasible"] and math.isfinite(x["max_abs_z"])])

    obs_ids = [o.obs_id for o in pub.observables if o.condition_id == "baseline"]
    if not feas_res:
        return {"certified_unreachable": True, "total_evals_run": len(all_res), "n_feasible": 0}

    feas_res.sort(key=lambda item: item["max_abs_z"])
    best_joint = feas_res[0]
    min_per_obs = {k: round(min(r["abs_z"][k] for r in feas_res), 3) for k in obs_ids}
    best_joint_z = {k: round(best_joint["abs_z"][k], 3) for k in obs_ids}
    unreachable = bool(best_joint["max_abs_z"] > 3.0)
    return {
        "certified_unreachable": unreachable,
        "budget_multiplier": multiplier,
        "total_evals_run": len(all_res),
        "n_feasible": len(feas_res),
        "min_abs_z_per_observable_individually": min_per_obs,
        "best_joint_abs_z": best_joint_z,
        "best_joint_max_abs_z": round(float(best_joint["max_abs_z"]), 3),
        "best_joint_observables": best_joint["observables"],
    }


def certify_type3a_degeneracy(
    task_dir: Path,
    ex: ProcessPoolExecutor,
    trace_dir: Path | None = None,
) -> dict[str, Any]:
    """Rule 2.3 Type 3a offline proof: >= 2 disparate public-passing solutions diverge on counterfactual."""
    pub, _ = load_public_task(task_dir, disclosure="full")
    priv = load_private_bundle(task_dir)
    active_rows = pub.active_parameters()
    gw_tmp = SimulatorGateway.from_task_dir(task_dir, max_units_override=400.0)
    rows, eff_bounds, eff_scales = resolve_search_space(
        gw_tmp, [r.name for r in active_rows], None, None
    )
    base = pub.nominal_dict()
    d = len(rows)

    # Collect low-loss candidates discovered by multi-seed G1 BO/CMA-ES runs (if available)
    cand_pool: list[dict[str, Any]] = []
    if trace_dir is not None and trace_dir.exists():
        for tp in sorted(trace_dir.glob(f"{task_dir.name}_g1_*_s*.jsonl")):
            for line in tp.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                if (
                    rec.get("event") == "eval"
                    and rec.get("feasible")
                    and rec.get("condition_id") == "baseline"
                    and rec.get("max_abs_z") is not None
                    and float(rec["max_abs_z"]) <= 1.8
                ):
                    cand_pool.append(
                        {"params": rec["params"], "max_abs_z": float(rec["max_abs_z"])}
                    )

    # Deduplicate and keep up to 40 diverse low-loss candidates for fresh multi-seed verification
    if len(cand_pool) < 4:
        rng = np.random.default_rng(20263000)
        n_samples = 128
        lhs = np.zeros((n_samples, d), dtype=np.float64)
        for j in range(d):
            perm = rng.permutation(n_samples)
            lhs[:, j] = (perm + rng.random(n_samples)) / n_samples
        cands = [unit_to_params(lhs[i], rows, eff_bounds, eff_scales, base) for i in range(n_samples)]
        c_sz = max(1, n_samples // 32)
        chunks = [(str(task_dir), cands[i : i + c_sz], 70000 + i) for i in range(0, n_samples, c_sz)]
        for b_out in ex.map(_eval_batch_worker, chunks):
            for r in b_out:
                if r["feasible"] and math.isfinite(r["max_abs_z"]):
                    cand_pool.append({"params": r["params"], "max_abs_z": float(r["max_abs_z"])})

    cand_pool.sort(key=lambda x: x["max_abs_z"])
    # Greedily pick up to 24 well-spread candidates from cand_pool (farthest-point sampling)
    selected: list[dict[str, Any]] = []
    if cand_pool:
        selected.append(cand_pool[0])
        u_all = [params_to_unit(c["params"], rows, eff_bounds, eff_scales) for c in cand_pool]
        u_sel = [u_all[0]]
        while len(selected) < min(24, len(cand_pool)):
            best_idx = -1
            best_min_dist = -1.0
            for idx, u_vec in enumerate(u_all):
                if cand_pool[idx]["max_abs_z"] > 1.8 and len(selected) >= 2:
                    continue
                md = min(float(np.max(np.abs(u_vec - us))) for us in u_sel)
                if md > best_min_dist:
                    best_min_dist = md
                    best_idx = idx
            if best_idx < 0 or best_min_dist < 1e-4:
                break
            selected.append(cand_pool[best_idx])
            u_sel.append(u_all[best_idx])

    # Independently verify each selected candidate on 3 fresh baseline seeds in parallel
    obs_rows = [o for o in pub.observables if o.condition_id == "baseline"]
    verify_chunks = [
        (str(task_dir), [c["params"] for _ in range(3)], 75000 + i * 10)
        for i, c in enumerate(selected)
    ]
    verified_passing: list[dict[str, Any]] = []
    for cand, v_runs in zip(selected, ex.map(_eval_batch_worker, verify_chunks)):
        feas = [r for r in v_runs if r["feasible"]]
        if len(feas) < 2:
            continue
        # Check mean observable z across the 3 fresh seeds
        z_fresh = {}
        for o in obs_rows:
            vals = [r["observables"][o.obs_id] for r in feas if o.obs_id in r["observables"]]
            if not vals:
                z_fresh[o.obs_id] = float("inf")
            else:
                z_fresh[o.obs_id] = abs(float(np.mean(vals)) - o.target_mean) / max(o.target_sd, 1e-9)
        fresh_max_z = max(z_fresh.values()) if z_fresh else float("inf")
        if fresh_max_z <= 2.0:
            verified_passing.append(
                {"params": cand["params"], "fresh_pub_max_abs_z": fresh_max_z, "z_fresh": z_fresh}
            )

    if len(verified_passing) < 2:
        return {
            "certified_non_identifiable": False,
            "n_candidates_pooled": len(cand_pool),
            "n_public_passing_solutions": len(verified_passing),
            "max_normalized_linf_separation": 0.0,
            "counterfactual_max_abs_z": 0.0,
        }

    u_mat = np.vstack(
        [params_to_unit(r["params"], rows, eff_bounds, eff_scales) for r in verified_passing]
    )
    best_pair = (0, 1)
    max_sep = 0.0
    for i in range(len(verified_passing)):
        for j in range(i + 1, len(verified_passing)):
            sep = float(np.max(np.abs(u_mat[i] - u_mat[j])))
            if sep > max_sep:
                max_sep = sep
                best_pair = (i, j)

    p_a = verified_passing[best_pair[0]]["params"]
    p_b = verified_passing[best_pair[1]]["params"]
    sim = instantiate_simulator(pub.simulator_config, public_dir=pub.public_dir)
    held_cond = priv.heldout_conditions[0]
    held_obs = priv.heldout_observables

    def _eval_held(p_vec: dict[str, float]) -> dict[str, Any]:
        eff = apply_condition(p_vec, held_cond)
        runs = [sim.simulate(eff, seed=65000 + k) for k in range(3)]
        feas = [r for r in runs if r.feasible]
        if not feas:
            return {"feasible": False, "z": {o.obs_id: 999.0 for o in held_obs}}
        z_map = {}
        for o in held_obs:
            vals = [r.observables[o.obs_id] for r in feas if o.obs_id in r.observables]
            m = float(np.mean(vals)) if vals else float("nan")
            z_map[o.obs_id] = (
                round((m - o.target_mean) / max(o.target_sd, 1e-9), 3) if math.isfinite(m) else 999.0
            )
        return {"feasible": True, "z": z_map}

    held_a = _eval_held(p_a)
    held_b = _eval_held(p_b)
    max_cf_z_diff = max(abs(held_a["z"][o.obs_id] - held_b["z"][o.obs_id]) for o in held_obs)
    max_cf_abs_z = max(
        max(abs(held_a["z"][o.obs_id]), abs(held_b["z"][o.obs_id])) for o in held_obs
    )
    certified = bool(len(verified_passing) >= 2 and max_sep >= 0.30 and max_cf_abs_z > 2.0)

    return {
        "certified_non_identifiable": certified,
        "n_candidates_pooled": len(cand_pool),
        "n_public_passing_solutions": len(verified_passing),
        "max_normalized_linf_separation": round(max_sep, 4),
        "pair_a_public_max_abs_z": round(float(verified_passing[best_pair[0]]["fresh_pub_max_abs_z"]), 3),
        "pair_b_public_max_abs_z": round(float(verified_passing[best_pair[1]]["fresh_pub_max_abs_z"]), 3),
        "pair_a_params": {r.name: round(p_a[r.name], 5) for r in rows},
        "pair_b_params": {r.name: round(p_b[r.name], 5) for r in rows},
        "pair_a_counterfactual": held_a,
        "pair_b_counterfactual": held_b,
        "counterfactual_max_abs_z": round(float(max_cf_abs_z), 3),
        "counterfactual_max_z_divergence": round(float(max_cf_z_diff), 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline certification & G1/G2 reverse screening.")
    parser.add_argument("--split", default="dev", choices=["dev", "test"])
    parser.add_argument("--tasks-root", default="tasks")
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--n-seeds", type=int, default=5, help="Seeds for Oracle and G1 baselines")
    parser.add_argument("--n-g2-seeds", type=int, default=3, help="Seeds for G2 expert rules")
    parser.add_argument("--include-clean", action="store_true", help="Also certify *_clean controls")
    parser.add_argument("--task-filter", nargs="*", default=None, help="Optional subset of task IDs")
    args = parser.parse_args()

    t_global = time.perf_counter()
    split_dir = Path(args.tasks_root) / args.split
    task_dirs = sorted([p for p in split_dir.iterdir() if p.is_dir() and (p / "public").exists()])
    if not args.include_clean:
        task_dirs = [p for p in task_dirs if not p.name.endswith("_clean")]
    if args.task_filter:
        allow = set(args.task_filter)
        task_dirs = [p for p in task_dirs if p.name in allow]

    # Put heavier FastDMF tasks first so long-pole jobs start immediately at t=0
    task_dirs.sort(key=lambda p: (0 if "fastdmf" in p.name else 1, p.name))

    trace_dir = Path("outputs/gate1/traces") / args.split
    trace_dir.mkdir(parents=True, exist_ok=True)

    priv_by_id = {p.name: load_private_bundle(p) for p in task_dirs}
    jobs: list[tuple[str, str, int, str]] = []
    for tdir in task_dirs:
        ttype = priv_by_id[tdir.name].task_type
        if ttype in ("type1_routine", "type2_recoverable"):
            for s in range(args.n_seeds):
                jobs.append((str(tdir), "oracle", s, str(trace_dir)))
        for s in range(args.n_seeds):
            jobs.append((str(tdir), "g1_bo", s, str(trace_dir)))
            jobs.append((str(tdir), "g1_cmaes", s, str(trace_dir)))
        for s in range(args.n_g2_seeds):
            jobs.append((str(tdir), "g2_rules", s, str(trace_dir)))

    print(
        f"=== Gate 1 Certification ({args.split}): {len(task_dirs)} tasks, {len(jobs)} controller jobs on {args.workers} workers ===",
        flush=True,
    )

    by_task_method: dict[str, dict[str, list[dict[str, Any]]]] = {p.name: {} for p in task_dirs}
    all_reports_by_method: dict[str, list[dict[str, Any]]] = {
        "oracle": [],
        "g1_bo": [],
        "g1_cmaes": [],
        "g2_rules": [],
    }
    proofs: dict[str, dict[str, Any]] = {}

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        # Phase A: Run all controller jobs across all tasks concurrently
        futs = [ex.submit(_run_controller_seed, job) for job in jobs]
        for done_cnt, fut in enumerate(as_completed(futs), start=1):
            res = fut.result()
            tid = res["task_id"]
            m = res["method"]
            by_task_method[tid].setdefault(m, []).append(res)
            all_reports_by_method[m].append(res)
            print(
                f"  [{done_cnt:3d}/{len(jobs)}] {tid:34s} {m:8s} s{res['seed']} -> "
                f"pass={res['verified_pass']!s:5s} dec={res['decision'][:7]} "
                f"u={res['used_units']:5.1f} t={res['wall_s']:6.1f}s",
                flush=True,
            )

        # Phase B: Offline Type 3a and Type 3b 10x proofs using full worker pool
        for tdir in task_dirs:
            priv = priv_by_id[tdir.name]
            if priv.task_type == "type3a_non_identifiable":
                print(f"\nRunning Type 3a multi-start + counterfactual proof for {tdir.name}...", flush=True)
                proofs[tdir.name] = {"type3a_proof": certify_type3a_degeneracy(tdir, ex, trace_dir=trace_dir)}
                p3a = proofs[tdir.name]["type3a_proof"]
                print(
                    f"  => {tdir.name} 3a certified={p3a['certified_non_identifiable']}, "
                    f"n_pub_pass={p3a['n_public_passing_solutions']}, "
                    f"max_sep={p3a['max_normalized_linf_separation']}, "
                    f"cf_max_abs_z={p3a['counterfactual_max_abs_z']}",
                    flush=True,
                )
            elif priv.task_type == "type3b_structure_unreachable":
                print(f"\nRunning Type 3b 10x-budget global proof for {tdir.name}...", flush=True)
                proofs[tdir.name] = {"type3b_10x_proof": certify_type3b_10x(tdir, ex, multiplier=10.0)}
                p3b = proofs[tdir.name]["type3b_10x_proof"]
                print(
                    f"  => {tdir.name} 3b certified={p3b['certified_unreachable']}, "
                    f"evals={p3b['total_evals_run']}, best_joint_max_abs_z={p3b.get('best_joint_max_abs_z')}, "
                    f"indiv_min={p3b.get('min_abs_z_per_observable_individually')}",
                    flush=True,
                )

    # Assemble and write validation_cert.yaml per task
    summary_all: dict[str, Any] = {"split": args.split, "tasks": {}, "headline": {}}
    for tdir in sorted(task_dirs, key=lambda p: p.name):
        priv = priv_by_id[tdir.name]
        tid = priv.task_id
        ttype = priv.task_type
        cert: dict[str, Any] = {
            "task_id": tid,
            "task_type": ttype,
            "certified_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        for m_name, m_runs in sorted(by_task_method[tid].items()):
            m_runs_sorted = sorted(m_runs, key=lambda r: r["seed"])
            pass_rate = float(np.mean([1.0 if r["verified_pass"] else 0.0 for r in m_runs_sorted]))
            mean_u = float(np.mean([r["used_units"] for r in m_runs_sorted]))
            cert[m_name] = {
                "n_seeds": len(m_runs_sorted),
                "verified_pass_rate": round(pass_rate, 4),
                "mean_used_units": round(mean_u, 2),
                "decisions": [r["decision"] for r in m_runs_sorted],
                "per_seed_pass": [bool(r["verified_pass"]) for r in m_runs_sorted],
            }
        cert.update(proofs.get(tid, {}))

        if ttype == "type1_routine":
            qa_ok = cert["oracle"]["verified_pass_rate"] >= 0.8
        elif ttype == "type2_recoverable":
            g1_max = max(cert["g1_bo"]["verified_pass_rate"], cert["g1_cmaes"]["verified_pass_rate"])
            qa_ok = (cert["oracle"]["verified_pass_rate"] >= 0.8) and (g1_max <= 0.6)
        elif ttype == "type3a_non_identifiable":
            qa_ok = bool(cert["type3a_proof"]["certified_non_identifiable"])
        else:
            qa_ok = bool(cert["type3b_10x_proof"]["certified_unreachable"])

        cert["qa_gate_passed"] = qa_ok
        (tdir / "private" / "validation_cert.yaml").write_text(
            yaml.safe_dump(cert, sort_keys=False), encoding="utf-8"
        )
        summary_all["tasks"][tid] = cert

    for m_name, reps in all_reports_by_method.items():
        if reps:
            summary_all["headline"][m_name] = compute_headline_metrics(reps)

    summary_all["total_wall_s"] = round(time.perf_counter() - t_global, 2)
    out_path = Path("outputs/gate1") / f"{args.split}_certification.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary_all, indent=2), encoding="utf-8")

    print(f"\n=== Certification Summary ({args.split}, total wall={summary_all['total_wall_s']}s) ===", flush=True)
    for tid, cert in summary_all["tasks"].items():
        o_r = cert.get("oracle", {}).get("verified_pass_rate", "-")
        bo_r = cert.get("g1_bo", {}).get("verified_pass_rate", "-")
        cma_r = cert.get("g1_cmaes", {}).get("verified_pass_rate", "-")
        g2_r = cert.get("g2_rules", {}).get("verified_pass_rate", "-")
        print(
            f"  {tid:34s} | {cert['task_type']:27s} | Oracle={o_r!s:4s} G1_BO={bo_r!s:4s} "
            f"G1_CMA={cma_r!s:4s} G2={g2_r!s:4s} | QA={cert['qa_gate_passed']}",
            flush=True,
        )
    print("\n=== Headline Metrics ===", flush=True)
    for m_name, hm in summary_all["headline"].items():
        print(f"  {m_name:10s}: {hm}", flush=True)
    print(f"Saved certification summary to {out_path}", flush=True)


if __name__ == "__main__":
    main()
