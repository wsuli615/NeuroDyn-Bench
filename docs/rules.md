# Evaluation Rules, Exam Syllabus, and Quality Gates

This document serves as the normative exam specification and QA rulebook for **NeuroDyn-Bench**. All task admission, budget accounting, anti-leakage controls, scoring formulas, and statistical reporting strictly follow these rules.

---

## 1. Exam Syllabus

The syllabus is derived backwards from real decisions and failure modes encountered by computational neuroscientists during model calibration.

### 1.1 Four Competency Modules (Reported Per-Module)

| Module | Core Capabilities Tested | Representative Challenges | Reference Weight |
| --- | --- | --- | --- |
| **M1. Task Setup** | Parse parameter semantics & dual bounds, screen sensitive dimensions, set ranges & log/linear scales, normalize multi-feature objectives, select optimizer | Ground truth outside default bounds; log-scale time constants set linearly; redundant/sloppy parameters; imbalanced feature scales | 30% |
| **M2. Iterative Control** | Interpret residuals & convergence, handle divergent/silent regimes, manage stochasticity, allocate remaining budget, decide stopping | >70% infeasible phase volume; high trial-to-trial spiking noise; multi-cause ambiguous stalls; tight-budget trade-offs | 25% |
| **M3. Process Diagnosis & Abstention** | Separate setup issues from insufficient evidence; distinguish `3a` (non-identifiable) from `3b` (structurally unreachable); avoid false abstention | Missing biophysical mechanism (spatial connectivity / adaptation) causing hard trade-offs paired 1:1 with solvable controls | 25% |
| **M4. Result Reliability** | Fresh-seed replication, held-out & counterfactual generalization, degeneracy / uncertainty quantification, recommending next experiments | Under-constrained single-condition fits that collapse under novel stimulation or coupling perturbation | 20% |

> **Rule 1.1 (Disaggregated Scoreboard)**: Every report must disaggregate scores across M1–M4 and Type 1 / Type 2 / Type 3a / Type 3b. Composite totals are secondary so strong Type 1 scores never mask poor abstention or over-abstention.

### 1.2 Hard Scope Exclusions

- No synthesizing or editing simulator differential equations (`BMD` / `ModelSMC` / `unsatz`);
- No active experimental design over new stimulation protocols (`NeuronBench`);
- No writing numerical simulator engines from scratch.

### 1.3 Two Task Formats

- **End-to-End Interactive Tasks (Primary Benchmark)**: Multi-round calibration under a hard simulation budget, ending in `submit_solution` or `abstain_insufficient_evidence`, scored by the isolated Evaluator.
- **Snapshot Decision Items (Diagnostic Micro-Eval, Gate 1.5+)**: Frozen mid-run stall snapshots asking for the optimal next action; ground-truth action utilities are computed empirically offline by executing every candidate action for an identical small follow-up budget.

---

## 2. Six Benchmark Admission Gates (QA Red Lines)

No task may enter the benchmark suite unless it passes all six rules:

### Rule 2.1: HLE-Style Reverse Screening

Except for a small set of Type 1 sanity checks, every Type 2 and Type 3 candidate task must be screened against **default out-of-the-box optimizers (G1: BO / CMA-ES)** (and G3 general agents in Gate 2):

- If default G1/G3 easily solves a Type 2 task (e.g., $>70\%$ seed pass rate), downgrade or reject it;
- If the **Oracle** configuration fails to pass reliably within budget on a Type 1/2 task, fix or reject the task.

### Rule 2.2: Single-Variable Attributability and Paired Controls (Level 1)

- Controlled Level 1 tasks isolate a primary failure mode and export a **same-ground-truth clean baseline (`*_clean`) paired with the trapped task**, directly measuring the net difficulty and net diagnostic recovery gain.
- Dedicated **multi-cause ambiguous tasks** (e.g., T08) are included to test discrimination over simple threshold rules, with an explicit verifiable discrimination path recorded in `validation_cert.yaml`.

### Rule 2.3: Mandatory Offline Label Certification (`validation_cert.yaml`)

- **Type 1 & Type 2 (Solvable)**: Must certify offline that an Oracle configuration passes independent verification across multiple random seeds (pass rate $\ge 80\%$) within the task budget.
- **Type 3a (Non-Identifiable / Degenerate / Representation Loss)**: Must certify offline that at least two well-separated parameter sets (or a low-rank Fisher / low-contraction posterior manifold) satisfy all public training observables while diverging substantially in parameter values and on hidden counterfactual conditions.
- **Type 3b (Fixed-Structure Unreachable)**: Must certify offline via **$\ge 10\times$ task budget** multi-algorithm global search (CMA-ES + BO over full `phys_hard` bounds) that the target remains unreachable, with explicit documentation of conflicting observable pairs.
- **1:1 Pairing Rule**: Every Type 3 (abstain) task is paired with a superficially similar, verified-solvable Type 2 control task to prevent "always abstain" gaming.

### Rule 2.4: Cross-Model Coverage and Published Anchors

- Each core failure class (bounds, noise, sloppiness, 3a, 3b) spans at least two models or scales across v1.0.
- Retain published anchor problems (SNOPS CBN/SBN, FastDMF HCP, Neuroptimus AdEx) to cross-check baseline sanity.

### Rule 2.5: Anti-Memorization and Anti-Leakage Discipline

- **Randomized ground truths**: Sample synthetic ground truths randomly inside valid physiological regions, strictly avoiding textbook/default values (`TuneBench` principle).
- **Three-tier disclosure**: Support `full`, `names_only`, and `blinded` (`p0, p1, ...`) views to test genuine physical reasoning vs. memorization.
- **Strict `dev` / `test` & `private/` isolation**: Lock prompts, rules, and thresholds on `dev` before running closed evaluation on `test`; `private/` is physically isolated from evaluated systems at all times.

### Rule 2.6: Realistic Synthetic Roughness

Following `CtDB`, synthetic tasks must never provide noiseless full-state trajectories—they must include stochastic trial-to-trial variability, partial summary observables, and realistic silent/divergent phase regions.

---

## 3. Unified Budget and Cost Accounting

### Rule 3.1: All-Inclusive Simulation Budget

- Each task sets a hard ceiling $B$ in effective simulation units (`duration_scale = 1.0` for a full run; short probes billed proportionally, e.g. `0.2u` or `0.25u`).
- **Everything counts**: optimization calls, repeated seeds ($R$ repeats = $R$ units), short feasibility scans, sensitivity screens, Bayesian ABC/Laplace/PPC probes, and failed/divergent/NaN runs.
- Only exact `(simulator, params, condition, seed, duration_scale)` cache hits are exempt.
- Reaching $B$ triggers hard refusal (`BudgetExhaustedError`).

### Rule 3.2: Separate LLM Cost + Physical Cost-Sensitivity Scan

- Report LLM call counts, input/output tokens, estimated API USD cost, and wall-clock time separately from simulation units.
- Following `SimulCost`, sweep physical simulation duration across `[0.1, 1.0, 3.0, 10.0, 60.0, 300.0]` seconds to identify the break-even cost regime for LLM diagnostic overhead.

---

## 4. Anti-Gaming Scoring Metrics

### 4.1 Three Co-Primary Headline Metrics

1. **Verified Pass Rate (Type 1 & Type 2)**:
   - Fraction of solvable tasks where submitted $\hat{\theta}$ passes **both** fresh-seed replication on public observables ($|z_k| = |f_k(\hat{\theta}) - f_k^{\text{target}}| / \sigma_k \le \tau$, default $\tau = 2.0$) **and** hidden held-out/counterfactual acceptance criteria within budget $B$.
2. **Abstention Precision / Recall / F1 (Type 2 vs. Type 3 Paired)**:
   - **Recall**: Fraction of Type 3 (`3a` + `3b`) tasks where the system correctly triggers `abstain_insufficient_evidence`, identifies the exact subtype (`3a` vs. `3b`), and cites valid tool-generated evidence recorded in `trace.jsonl`.
   - **Precision**: Fraction of all abstentions that truly belong to Type 3 (false abstention on Type 2 directly penalizes precision).
3. **Diagnostic Action Lift (RQ1 & RQ2)**:
   - Action divergence rate between G4 (diagnostic) and G3 (no diagnostic protocol), plus net Verified Pass Rate and Abstention F1 gain over G3 and G2.

### 4.2 Secondary & Process Failure Metrics

- **Anytime Efficiency Curve**: Verified Pass Rate vs. simulation units consumed.
- **Parameter Recovery Error**: Reported **only** along certified stiff/identifiable directions, never on sloppy/degenerate directions.
- **Automated Process Failure Audit** (from `trace.jsonl`):
  1. Premature stopping;
  2. Out-of-hard-bounds or malformed proposals;
  3. Diagnosis-action misalignment;
  4. Long-horizon degradation (plateau/regression onset round).

---

## 5. Statistical Rigor and Human-Cost Honesty

- **Rule 5.1 (Multi-Seed Confidence Intervals)**: Run $\ge 10$ independent seeds per method x task on Level 1 ($\ge 5$ on heavy Level 2 tasks), reporting median, IQR, and 95% bootstrap CIs. Repeat runs estimate variance only and never count as additional tasks.
- **Rule 5.2 (Human Intervention Boundaries)**: Use fixed rule baselines and structured `ask_human` gap items as reproducible proxies. **Never make quantitative claims about human expert time reduction** without actual controlled human expert timing logs.

---

## 6. Reproducible Audit Trail (`trace.jsonl`)

Every run emits an immutable `trace.jsonl` recording `round_id`, `used_units`, `remaining_units`, `tool_name` / `action_type`, parameters, feasibility, wall-clock time, structured diagnostic summary, and final evaluator breakdown. Unreproducible results are excluded from official tables.
