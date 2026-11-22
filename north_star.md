# North Star: NeuroDyn-Bench

> **One-sentence North Star**: Build the first executable, cost-aware diagnostic calibration benchmark for computational neuroscience centered on structured workflow diagnosis and selective abstention under insufficient evidence (*Benchmark-First*), and rigorously test whether closing the loop from diagnostics to outer-loop actions improves calibration reliability and scientific honesty under fixed budgets.

---

## 1. The Fundamental Problem

Mechanistic neural dynamics modeling expresses hypotheses about neurons, local circuits, and whole-brain networks as stochastic nonlinear differential equations or spiking systems. Calibrating biophysical parameters (synaptic weights, excitation-inhibition ratios, synaptic time constants, external drive, feedback inhibition) to match experimental recordings faces four compounding bottlenecks:

1. **Increasing biophysical realism exacerbates inverse ill-posedness**: Parameter dimensionality grows from single cells (3–16D) to spiking circuits (8–11D), neural mass models (20+D), and heterogeneous whole-brain networks (tens to 100+D). Strong nonlinearities, bifurcations, stochasticity, sloppiness, parameter degeneracy, practical/structural non-identifiability, and summary-statistic compression mean that low training error does **not** guarantee parameter recovery or counterfactual validity.
2. **Numerical optimizers search, but cannot diagnose**: Bayesian optimization (BO / SNOPS / FastDMF), CMA-ES, data assimilation (EnKF / HDA), and simulation-based inference (SBI / VBI) handle inner-loop search, yet search-space setup, stall diagnosis, budget reallocation, and recognizing when data cannot identify parameters still rely heavily on tacit human expertise.
3. **Forty years of diagnostics stop at the reporting layer**: Posterior predictive checks (PPC), Bayesian workflow, simulation-based calibration (SBC), SNOPS engineering signals, and hierarchical validity audits describe *what* failed without closing the loop into *what calibration action to execute next*.
4. **Frontier LLM agents plateau over long horizons and refuse to abstain**: Recent benchmarks (AgentHPOBench, HydroAgent, SimulCost, Humanity's Last Exam) show that frontier LLMs degrade after 3–4 sequential interventions, mis-act on ambiguous logs, neglect simulation cost, and submit overconfident answers when problems are underdetermined or unsolvable.

**NeuroDyn-Bench** sits at the intersection of these threads: turning statistical and domain diagnostics into computable signals coupled to an executable outer-loop action space, while elevating **principled abstention under insufficient evidence** to a first-class evaluation criterion.

---

## 2. Hard Scope Boundaries

To prevent being subsumed by equation-discovery agents (`BMD`, `ModelSMC`, `unsatz`) or pure inner-loop optimizer suites (`SNOPS`, `Neuroptimus`), four boundaries are locked:

| Dimension | NeuroDyn-Bench (This Project) | Neighboring Work (Boundary) |
| --- | --- | --- |
| **Search target** | Parameter calibration, workflow setup, and process decisions under a **fixed model structure** | `BMD` / `ModelSMC` / `unsatz` search over equations or structural forms |
| **Diagnostic target** | Why the calibration process stalled, what to modify next, and whether to stop | `SNOPS` diagnoses model structural flexibility; classic PPC stops at reports |
| **Primary output** | Executable outer-loop actions (expand bounds, select subset, increase repeats, check `dt`, run discriminative probe, or abstain `3a`/`3b`) | Optimizers always output a point estimate or static posterior |
| **Core claim** | Structured diagnostics independently alter actions and improve verified reliability and honesty | Not a generic claim that LLMs guess points better than BO |

**Explicitly Out of Scope (Phase 0 / v1.0)**:

- Synthesizing new differential equations or modifying simulator source code (for structural mismatch, diagnosing unreachability and conflicting observables suffices);
- Active experimental design over novel stimulation protocols (`NeuronBench` territory);
- Fine-tuning custom LLMs in Phase 0 (first establish rock-solid benchmarks, protocols, and baselines with off-the-shelf frontier models).

---

## 3. Benchmark-First "Last Exam" Strategy

Frontier foundation models refresh every six months; prompt-engineered wrappers are rapidly absorbed, whereas a hard, executable, leak-resistant benchmark grounded in genuine scientific bottlenecks defines progress for years.

Inspired by **Humanity's Last Exam (HLE)** while correcting its static-QA limitations:

- **Three practices we adopt**:
  1. **Reverse screening**: Candidate hard tasks must defeat default optimizers and unguided frontier agents before admission, while remaining reliably solvable (or provably certifiable as `3a`/`3b`) under an Oracle or sound diagnostic sequence.
  2. **Calibration and scientific honesty as headline metrics**: Penalize both blind overconfidence (submitting degenerate/bad fits on Type 3) and cowardly evasion (abstaining on solvable Type 2).
  3. **Frozen private splits and rolling community updates**: Strictly separate `dev` from frozen `test`, and expand v1.0 via real failure cases contributed by computational neuroscience labs.
- **Two practices we reject**:
  1. **No static trivia QA**: Every task is an executable, budgeted, interactive simulation problem scored automatically on fresh seeds and hidden held-out/counterfactual conditions.
  2. **Depth and attributability over sheer item count**: 30–50 deeply vetted cross-scale tasks (starting with 10–15 benchmark prototypes in Phase 0) constitute a top-tier scientific agent benchmark.

---

## 4. Three Task Types and Three Exam Levels

### 4.1 Three Task Types

| Task Type | Core Skill Tested | Subtypes / Failure Modes | Evaluation Goal |
| --- | --- | --- | --- |
| **Type 1: Routine Calibration** | Do no harm | Well-posed calibration with clean bounds and scales | Match mature baselines efficiently without wasting diagnostic budget |
| **Type 2: Recoverable Failures** | Diagnose accurately & act effectively | Bound truncation + hard-bound semantics, scale mismatch, sloppy high-D redundancy, stochastic variance, infeasible/divergent phase regions, multi-cause ambiguous stalls | Prove diagnostics alter actions and significantly raise independent verified pass rate within fixed budget |
| **Type 3: Evidence-Insufficient / Underdetermined** | Know when to stop & explain why | **3a. Non-identifiable / degenerate / representation loss**: low training error, but multiple disparate solutions diverge on counterfactuals.<br/>**3b. Structurally unreachable**: error plateaus high with irreconcilable feature trade-offs under fixed structure. | Accurately abstain with subtype + tool evidence while maintaining high precision against paired Type 2 controls |

### 4.2 Three Exam Levels

- **Level 1 (Controlled Diagnostic Arena)**: Twin-experiment tasks on `Small E-I` spiking networks (8–10D, CBN/SBN), `AdEx` anchor neurons, and low-to-mid dimensional `FastDMF`, with paired clean vs. pitfall variants and ambiguous multi-cause traps for exact attribution and ablation.
- **Level 2 (Frontier Real-World Bottlenecks)**: High-dimensional heterogeneous `FastDMF`, SNOPS-scale multi-electrode covariability targets, degenerate circuits, and real stalled calibration cases supplied by domain collaborators.
- **Level 3 (Scientific Honesty & Counterfactual Arena)**: 1:1 paired `3a`/`3b` vs. hard-but-solvable Type 2 tasks, scored under hidden novel stimuli, held-out subjects, or counterfactual interventions (e.g., E/I balance shift, coupling perturbation).

---

## 5. Research Questions

- **RQ1 (Diagnosis-to-Action Mapping)**: Which calibration failures caused by setup errors, stochasticity, numerical instability, degeneracy, and summary compression can be identified via structured diagnostic probes, and does mapping those signals to outer-loop actions independently improve verified outcomes?
- **RQ2 (Capability Frontier & Baselines)**: Under strictly unified simulation budgets, what real gains do diagnostic agents achieve over raw optimizers (G1), deterministic expert rules (G2), and general tool agents (G3) on ambiguous, physically constrained, and long-horizon tasks—and at what single-simulation cost does LLM overhead become economical?
- **RQ3 (Reliability & Scientific Honesty)**: Do low-loss fits generalize to fresh seeds, held-out data, and counterfactual interventions, and can agents reliably distinguish `3a` (non-identifiable) and `3b` (structurally unreachable) from solvable Type 2 traps with low false-abstention rates?

---

## 6. Stage-Gate Roadmap and Stop Criteria

Progress is governed by empirical evidence rather than assuming LLMs must win:

- **Phase 0 (Prototype Exam & Feasibility Validation, 6-week pilot)**:
  - Deliverables: Dual-environment minimal reproductions + `dgx2felis` cost profiling (Gate 0); dual-track harness + 10–15 certified prototype tasks (Gate 1); closed evaluation of G1–G4 (Gate 2); go/no-go decision (Gate 3).
  - **Honesty / Pivot Rule (ADR-009)**: If empirical evaluation shows that raw optimizers or concise deterministic expert rules already solve most real failure modes, we transparently narrow agent claims and center the paper on the benchmark, empirical failure taxonomy, identifiability/unreachability auditing, and lightweight rule automation.
- **Phase 1 (NeuroDyn-Bench v1.0 Release)**: 30–50 cross-scale tasks (`dev` + frozen `test`), comprehensive leaderboard across frontier models, classical optimizers, rules, and diagnostic agents, plus optional small-scale human expert timing logs.
- **Phase 2 (Rolling Community Benchmark)**: Standardized task submission protocol for computational neuroscience labs, rolling private test splits, and integration with amortized SBI posterior diagnostics.

---

## 7. Definition of Success

One year forward, **NeuroDyn-Bench** succeeds if:

1. **AI-for-Science and computational neuroscience teams use it** as the standard executable exam for testing whether agents can reliably calibrate mechanistic neural models without fabricating overconfident fits;
2. **Domain modelers adopt its diagnostic and audit toolkit** to pinpoint why parameter fits stall and catch underdetermined regimes early;
3. **Its taxonomy, `3a`/`3b` honesty criteria, and leak-proof evaluation protocol remain durable** across generations of foundation models.
