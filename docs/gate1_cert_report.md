# Gate 1 Empirical Certification Report: Dual-Track Harness & 8 Certified v0.1 Tasks

- **Date**: 2026-11-22
- **Host**: `dgx2felis` (64-worker parallel certification; `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`)
- **Author**: `wsuli615`
- **Status**: **8 / 8 Core Gate 1 Tasks Passed All QA Admission Gates (Rules 2.1–2.6)**

---

## 1. Gate 1 Engineering Deliverables

1. **PEtab-Style Isolated Task Bundles (`src/neurodyn_bench/tasks/`)**:
   - Strict `public/` vs. `private/` separation (`load_public_task` never touches `private/`).
   - Dual-bound parameter semantics (`default_lb/ub` vs. `phys_hard_lb/ub`).
   - Built-in **three-tier disclosure renderer** (`full`, `names_only`, `blinded` as `p0, p1, ...` with server-side reversible translation).
   - Paired **no-trap clean controls (`*_clean`)** generated for all 4 Type 2 tasks (`T03`, `T04`, `T05`, `T08`), across independent `dev` and frozen `test` splits (24 task bundles total).
2. **Hybrid Dual-Track Gateway (`src/neurodyn_bench/gateway/`)**:
   - **Unified `SimulatorGateway` + `BudgetLedger`**: Every optimization trial, repeat, short probe, Bayesian posterior sample, and failed/divergent run deducts from one hard budget and logs immutably to `trace.jsonl`.
   - **Track A (`src/neurodyn_bench/tools/`)**: Self-describing optional tool registry exposing inner-loop optimizers (`run_bo`, `run_cmaes`, `run_multistart_local`), **budget-aware lightweight Bayesian inference** (`run_abc_smc`, `run_laplace_fisher`, `run_ppc_audit`), and discriminative diagnostic probes (`boundary_slack`, `repeat_variance`, `short_feasibility_scan`, `sensitivity_screen`, `feature_tradeoff_audit`, `multistart_identifiability`).
   - **Track B (`src/neurodyn_bench/gateway/sandbox.py`)**: Unix-domain-socket isolated Python sandbox exporting only rendered `public/` files and a `neurodyn_client.py` proxy so frontier coding agents can freely write custom calibration and diagnostic scripts without ever reaching `private/` or bypassing `BudgetLedger`.
3. **Independent Evaluator & Anti-Gaming Metrics (`src/neurodyn_bench/eval/`)**:
   - Fresh-seed replication (5 seeds) + hidden counterfactual/held-out verification + evidence-gated `3a`/`3b` abstention scoring + automated hard-bound and premature-stop auditing.
   - Verified with `pytest -v tests/test_gate0.py tests/test_gate1.py` (**7 / 7 passed**).

---

## 2. Offline Certification Scoreboard (`dev` Split — 8 Core Tasks)

All solvable tasks (`Type 1` & `Type 2`) were evaluated across 5 seeds for `Oracle`, `G1_BO`, and `G1_CMAES` and 3 seeds for `G2_Rules`; all `Type 3` tasks underwent mandatory offline proof verification (Rule 2.3):

| Task ID | Environment | Task Type | Oracle (5s) | G1 BO (5s) | G1 CMA-ES (5s) | G2 Rules (3s) | Offline Proof / QA Gate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **T02_small_ei_routine** | Small E-I (CBN) | Type 1 Routine | **1.00** | 0.80 | 1.00 | 0.33 | **PASS** (Oracle $\ge 0.80$) |
| **T03_small_ei_bound_semantics** | Small E-I (CBN) | Type 2 Recoverable | **0.80** | 0.00 | 0.20 | 0.00 | **PASS** (Oracle 0.80 vs G1 $\le 0.20$) |
| **T04_fastdmf_highdim_scale** | FastDMF (16D) | Type 2 Recoverable | **1.00** | 0.20 | 0.00 | 0.00 | **PASS** (Oracle 1.00 vs G1 $\le 0.20$) |
| **T05_small_ei_high_variance** | Small E-I (CBN) | Type 2 Recoverable | **0.80** | 0.60 | 0.20 | 0.33 | **PASS** (Oracle 0.80 vs G1 $\le 0.60$) |
| **T08_small_ei_ambiguous_stall** | Small E-I (CBN) | Type 2 Ambiguous | **0.80** | 0.00 | 0.00 | 0.67 | **PASS** (Oracle 0.80 vs G1 0.00; paired w/ T11) |
| **T11_small_ei_cbn_vs_sbn_3b** | Small E-I (CBN vs SBN) | Type 3b Unreachable | — | 0.00 | 0.00 | 1.00 | **PASS** (10x 2000u proof: joint max $|z|=4.81$, indiv $\le 0.03$) |
| **T13_small_ei_underconstrained_3a** | Small E-I (CBN) | Type 3a Non-Identifiable | — | 0.00 | 0.00 | 1.00 | **PASS** (17 pub-pass fits, sep 0.82, CF gap $37.3\sigma$) |
| **T14_fastdmf_fc_compression_3a** | FastDMF (5D) | Type 3a Non-Identifiable | — | 0.00 | 0.00 | 1.00 | **PASS** (14 pub-pass fits, sep 0.90, CF gap $66.7\sigma$) |

---

## 3. Aggregate Anti-Gaming Headline Metrics (`dev`)

| Controller | Type 1 Pass | Type 2 Pass | Solvable Pass | Abstention Recall (3a / 3b) | Abstention Precision | Abstention F1 | False Abstention (Solvable) | False Confidence (Type 3) | Mean Units |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Oracle (Solvable Ceiling)** | **1.00** | **0.85** | **0.88** | — | — | — | 0.00 | — | 164.0 |
| **G1 Default BO** | 0.80 | 0.20 | 0.32 | 0.00 / 0.00 | N/A | N/A | 0.00 | **1.00** | 165.0 |
| **G1 Default CMA-ES** | 1.00 | 0.10 | 0.28 | 0.00 / 0.00 | N/A | N/A | 0.00 | **1.00** | 165.0 |
| **G2 Deterministic Rules** | 0.33 | 0.25 | 0.27 | **1.00 / 1.00** | **0.75** | 0.86 | **0.20** | 0.00 | 155.1 |

### Key Scientific Takeaways Already Visible in Gate 1

1. **Out-of-the-box optimizers (`G1`) collapse on realistic calibration bottlenecks**: Strong on routine Type 1 (`80–100%`), yet dropping to **`10–20%`** on recoverable Type 2 failures and exhibiting **`100%` false confidence** (`0%` abstention recall) on Type 3a/3b tasks.
2. **Naive deterministic rules (`G2`) expose the exact trade-offs motivating intelligent diagnostic control**: While `G2` catches Type 3a/3b reliably (`Recall = 1.00`), its rigid thresholds (a) waste budget on Type 1 routine calibration (`T02` drops from `1.00` to `0.33`), (b) trigger **`20%` false abstentions** on solvable tasks (`Precision = 0.75`, falsely abstaining on `T03` and `T08`), and (c) achieve only **`25%`** recovery on Type 2 (`0%` on `T03` bound semantics and `0%` on 16D `T04` FastDMF).
3. **Ironclad offline Type 3b & Type 3a proofs**:
   - **T11 (SNOPS CBN fitting 4-mode SBN, `2,000` full simulations = 10x budget across full `phys_hard` bounds)**: Every single observable is individually reachable to $|z| \le 0.031$ (`fr_e: 0.008`, `ff: 0.000`, `rsc: 0.011`, `pct_sh: 0.031`, `dsh: 0.000`), yet the best achievable joint `max_abs_z` across 2,000 evaluations is **`4.809`**—demonstrating an irreconcilable fixed-structure trade-off.
   - **T13 (Small E-I single-condition 3a)**: **17** fresh-seed public-passing solutions (`max_abs_z <= 2.0`), with maximally separated pair (`L_inf = 0.8245`: `g_ei = 3.05` vs. `8.00`, `tau_i = 12.96` vs. `6.15`) diverging by **`37.3σ`** under the hidden counterfactual perturbation (`+35%` drive + `22%` inhibitory blockade).
   - **T14 (FastDMF static-FC 3a)**: **14** fresh-seed public-passing solutions, with maximally separated pair (`L_inf = 0.8973`: `G = 0.91, sigma = 0.0101` vs. `G = 3.11, sigma = 0.0020`) diverging by **`66.7σ`** under the hidden `+25%` global-coupling perturbation.

---

## 4. Exact Reproduction Commands

```bash
# 1. Regenerate all 12 dev and 12 test task bundles (including private/ locally)
.venv/bin/python scripts/build_v0_tasks.py --splits dev test --workers 16

# 2. Run Gate 0 + Gate 1 unit & integration test suite
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/pytest -v

# 3. Re-run full offline certification & G1/G2 reverse screening on dev
.venv/bin/python scripts/certify_v0_tasks.py --split dev --workers 64
```
