"""Gate 2 LLM Agent Benchmark Runner (Claude Opus 5.5 max via local CLI subscription).

Evaluates Track A (g3_claude, g4_claude) and/or Track B (track_b_claude under seccomp-bpf +
strace offline isolation) on NeuroDyn-Bench tasks, reporting disaggregated headline metrics,
separate LLM token/cost/latency accounting, and SimulCost-style cost sensitivity scans.

Usage:
    .venv/bin/python scripts/run_gate2_llm.py --split dev --methods g3_claude g4_claude --model opus --effort max
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from neurodyn_bench.agents.llm_claude import (
    run_g3_claude,
    run_g4_claude,
    run_track_b_claude,
)
from neurodyn_bench.eval.evaluator import evaluate_submission
from neurodyn_bench.eval.metrics import (
    compute_cost_sensitivity_scan,
    compute_headline_metrics,
)
from neurodyn_bench.gateway.gateway import SimulatorGateway


def _run_one(
    args: tuple[str, str, int, str, str, str, str],
) -> dict[str, Any]:
    task_dir, method, seed, model, effort, disclosure, out_dir = args
    t0 = time.perf_counter()
    tid = Path(task_dir).name
    run_root = Path(out_dir) / f"{tid}_{method}_s{seed}"
    run_root.mkdir(parents=True, exist_ok=True)
    trace_path = run_root / "trace.jsonl"
    sub_path = run_root / "final_submission.json"

    if method in ("g3_claude", "g4_claude"):
        gw = SimulatorGateway.from_task_dir(task_dir, disclosure=disclosure, trace_path=trace_path)  # type: ignore[arg-type]
        if method == "g3_claude":
            sub, tracker = run_g3_claude(gw, model=model, effort=effort, seed=seed)
        else:
            sub, tracker = run_g4_claude(gw, model=model, effort=effort, seed=seed)
        gw.save_submission(sub, sub_path)
        extra_meta: dict[str, Any] = {}
    elif method == "track_b_claude":
        sub, tracker, extra_meta = run_track_b_claude(
            task_dir=task_dir,
            workspace_dir=run_root / "sandbox_ws",
            model=model,
            effort=effort,
            disclosure=disclosure,  # type: ignore[arg-type]
            trace_path=trace_path,
        )
        sub_path = Path(extra_meta["submission_path"])
    else:
        raise ValueError(f"Unsupported method: {method}")

    rep = evaluate_submission(
        task_dir=task_dir,
        submission=sub_path,
        trace_path=trace_path,
        disclosure=disclosure,  # type: ignore[arg-type]
        n_verify_seeds=5,
        verify_seed_start=50000 + seed * 10,
    )
    out = rep.to_dict()
    out["method"] = method
    out["seed"] = seed
    out["disclosure"] = disclosure
    out["llm_usage"] = tracker.summary()
    out["cost_sensitivity_scan"] = compute_cost_sensitivity_scan(
        used_units=rep.used_units,
        llm_wall_time_s=tracker.total_api_wall_s,
    )
    out["total_wall_s"] = round(time.perf_counter() - t0, 2)
    out["notes"] = sub.notes
    if extra_meta:
        out["sandbox_meta"] = extra_meta
    (run_root / "eval_report.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Gate 2 LLM agent evaluation via Claude CLI.")
    parser.add_argument("--split", default="dev", choices=["dev", "test"])
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["g3_claude", "g4_claude"],
        choices=["g3_claude", "g4_claude", "track_b_claude"],
    )
    parser.add_argument("--model", default="opus")
    parser.add_argument("--effort", default="max", choices=["low", "medium", "high", "xhigh", "max"])
    parser.add_argument("--disclosure", default="full", choices=["full", "names_only", "blinded"])
    parser.add_argument("--n-seeds", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--task-filter", nargs="*", default=None)
    parser.add_argument("--out-dir", default="outputs/gate2")
    args = parser.parse_args()

    split_dir = Path("tasks") / args.split
    task_dirs = sorted(
        [
            p
            for p in split_dir.iterdir()
            if p.is_dir() and (p / "public").exists() and not p.name.endswith("_clean")
        ]
    )
    if args.task_filter:
        allow = set(args.task_filter)
        task_dirs = [p for p in task_dirs if p.name in allow]

    run_dir = Path(args.out_dir) / f"{args.split}_{args.model}_{args.effort}_{args.disclosure}"
    run_dir.mkdir(parents=True, exist_ok=True)

    jobs = [
        (str(tdir), m, s, args.model, args.effort, args.disclosure, str(run_dir))
        for tdir in task_dirs
        for m in args.methods
        for s in range(args.n_seeds)
    ]
    print(
        f"=== Gate 2 LLM Eval ({args.split}, model={args.model}, effort={args.effort}, "
        f"disclosure={args.disclosure}): {len(jobs)} jobs on {args.workers} workers ===",
        flush=True,
    )

    by_method: dict[str, list[dict[str, Any]]] = {m: [] for m in args.methods}
    errors: list[dict[str, str]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        fut_to_job = {ex.submit(_run_one, job): job for job in jobs}
        for idx, fut in enumerate(as_completed(fut_to_job), start=1):
            job = fut_to_job[fut]
            try:
                res = fut.result()
            except Exception as exc:  # noqa: BLE001
                err_rec = {
                    "task_id": Path(job[0]).name,
                    "method": job[1],
                    "seed": str(job[2]),
                    "error": f"{type(exc).__name__}: {exc}",
                }
                errors.append(err_rec)
                print(
                    f"  [{idx:2d}/{len(jobs)}] {err_rec['task_id']:34s} {err_rec['method']:14s} "
                    f"s{err_rec['seed']} -> ERROR: {err_rec['error']}",
                    flush=True,
                )
                continue
            m = res["method"]
            by_method[m].append(res)
            u = res["llm_usage"]
            print(
                f"  [{idx:2d}/{len(jobs)}] {res['task_id']:34s} {m:14s} s{res['seed']} -> "
                f"pass={res['verified_pass']!s:5s} dec={res['decision'][:7]} "
                f"sim_u={res['used_units']:5.1f} llm_calls={u['n_calls']} "
                f"think_tok={u['thinking_tokens']} cost=${u['total_cost_usd']:.3f} "
                f"wall={res['total_wall_s']:.1f}s",
                flush=True,
            )

    summary = {
        "config": vars(args),
        "headline": {m: compute_headline_metrics(reps) for m, reps in by_method.items() if reps},
        "results": by_method,
        "errors": errors,
    }
    out_json = run_dir / "summary.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n=== Gate 2 Headline Summary ===", flush=True)
    for m, hm in summary["headline"].items():
        print(f"  {m:14s}: {hm}", flush=True)
    print(f"Saved summary to {out_json}", flush=True)


if __name__ == "__main__":
    main()
