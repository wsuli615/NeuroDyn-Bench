# Narrative and Positioning Guide

This document defines how to communicate **NeuroDyn-Bench** clearly across four distinct audiences (neuroscience collaborators, AI/ML reviewers, computational neuroscience reviewers, and CS/AI teammates) while preempting the six sharpest reviewer critiques.

---

## 1. Unified One-Minute Pitch

> Mechanistic neural dynamics models require calibrating biophysical parameters—synaptic weights, excitation-inhibition ratios, time constants, and coupling strengths—against noisy experimental recordings. This inverse problem is severely ill-posed: high-dimensional, computationally expensive, stochastic, and prone to parameter degeneracy and non-identifiability, where low training error does not imply accurate parameters and stalled fits conflate setup mistakes with fundamental lack of identifiability or structural mismatch.
>
> Classical Bayesian optimization, data assimilation, and SBI address *how to search*, while forty years of statistical and domain diagnostics address *how to report what is wrong*. Yet closing the loop from diagnostic evidence to outer-loop workflow actions—including knowing when to stop—remains manual. Meanwhile, 2026 agent benchmarks show that frontier LLMs degrade over long horizons on ambiguous logs and overconfidently submit fabricated fits when evidence is insufficient.
>
> **NeuroDyn-Bench** establishes the first executable, cost-aware calibration and diagnosis exam for computational neuroscience, testing whether coupling structured diagnostic evidence to outer-loop search and selective abstention (`3a` non-identifiable vs. `3b` structurally unreachable) genuinely improves independent validation reliability and scientific honesty under fixed budgets.

---

## 2. Audience-Specific Framing

### 2.1 For Neuroscience Collaborators

- **Emphasize scientific rigor and workflow relief, not LLM hype**: We do not replace your mechanistic model with a black box and we do not let AI alter your equations. Your model structure stays fixed; our system diagnoses why calibration stalls, adjusts the search workflow, and warns you with auditable numerical evidence when the available data cannot uniquely identify parameters.
- **Concrete ask**: One real data sample, 1–3 genuine stalled modeling cases, and joint acceptance criteria (see [`collaborator_brief.md`](collaborator_brief.md)).

### 2.2 For AI / Agent Reviewers

- **Targets three verified frontier-agent weaknesses simultaneously**: Long-horizon degradation on noisy logs (`AgentHPOBench`), premature stopping / out-of-bounds proposals / diagnosis-action misalignment (`HydroAgent`), and ignored physical simulation costs (`SimulCost`).
- **Selective abstention as a first-class metric**: Analogous to calibration error in `Humanity's Last Exam`, Type 3a (non-identifiable) and Type 3b (structurally unreachable) tasks paired 1:1 with hard-but-solvable Type 2 controls defeat both "never stop" and "always abstain" gaming.
- **Four-group controls eliminate wrapper illusions**: Explicit comparison across G1 (raw optimizer), G2 (expert rules), G3 (general tool agent without diagnostic protocol), and G4 (diagnostic agent), plus Track B open sandbox evaluation for frontier coding agents.

### 2.3 For Computational Neuroscience Reviewers

- **Grounded in inverse-problem pathology**: Explicitly embraces `sloppiness` (Gutenkunst 2007), `degeneracy` (Prinz 2004, Marder 2006), practical `non-identifiability` (Hartoyo 2019, Wischnewski 2025), and summary-representation information loss (Cai 2026). We never treat synthetic parameter drift on sloppy directions as optimizer failure; grading relies on independent replication, held-out/counterfactual predictions, and honest identifiability reporting.
- **Crisp boundary with adjacent tools**: See Section 3.

### 2.4 For CS / AI Engineering Teammates

- **One-sentence translation**: A neural dynamics model is a noisy, non-differentiable, expensive simulator with physically constrained parameters; calibration is its ill-posed black-box inverse problem; NeuroDyn-Bench builds the leak-proof exam harness and optional diagnostic/inference outer-loop tools around those simulators. See [`primer_for_cs.md`](primer_for_cs.md).

---

## 3. One-Sentence Boundary Table vs. Neighboring Work

| Neighboring Work | What It Does | Exact Distinction from NeuroDyn-Bench |
| --- | --- | --- |
| **Neuroptimus / sbibm / PEtab** | Fixes parameters, bounds, observables, and losses to benchmark inner-loop optimizers/inference | Skips task setup, stall diagnosis, workflow adaptation, and abstention; we test outer-loop decision making |
| **SNOPS (Wu et al. 2024)** | Uses feasibility/variance/trade-off heuristics inside BO for spiking networks and assesses model structural flexibility | SNOPS uses a fixed workflow and diagnoses model expressivity; we map diagnostic probes to dynamic workflow actions and stopping decisions |
| **BMD / ModelSMC / unsatz** | Uses LLMs to propose/rewrite model equations while keeping the fitter fixed | They ask *which equation is right*; we fix the structure and ask *how to calibrate, recover from stalls, and know when evidence is insufficient* |
| **NeuronBench / MDA (Murphy 2026)** | Active experimental design (choosing current/pharmacological interventions) to discover unknown channels | Focuses on active experiment selection and mechanism discovery; we focus on calibration and process diagnosis under given observations |
| **AgentHPOBench / TuneBench / HydroAgent / SimulCost** | ML HPO or power/hydrologic/physics simulation calibration benchmarks | Outside neuroscience and lack paired **Type 2 recoverable vs. Type 3a non-identifiable vs. Type 3b unreachable** abstention protocols |
| **Humanity's Last Exam (HLE)** | Static expert QA exam using reverse screening and calibration scoring | We adopt reverse screening, honesty scoring, and rolling private splits, but build executable budgeted simulations with automated counterfactual verification |

---

## 4. Six Anticipated Reviewer Critiques and Standard Responses

1. **"Can't you just hardcode a 5-row diagnostic table as if-else rules instead of using an LLM?"**
   - That is exactly why **G2 (Expert Rules)** is a mandatory control group. Simple single-cause failures should be solved cleanly by rules. Our hard tasks target multi-cause ambiguous stalls (requiring cheap discriminative probes before acting), biophysical hard-bound vs. arbitrary-bound semantics, and dynamic trade-offs under remaining budget. Under **ADR-009**, if rules empirically match agents across the suite, we will report that boundary honestly.
2. **"If a small simulator runs in seconds, isn't calling an LLM slower and more expensive?"**
   - Following `SimulCost`, we bill all simulations, repeats, probes, and failures into one simulation-unit budget, report LLM tokens/latency separately, and sweep physical simulation cost from `0.1 s` to `300 s` (SNOPS scale) to identify the exact break-even regime where diagnostic reasoning pays off.
3. **"In synthetic tasks, how do you ensure parameter recovery error isn't simply non-identifiability?"**
   - Every task requires offline `validation_cert.yaml` certification before admission. Type 1/2 tasks are verified solvable via Oracle ($\ge 80\%$ pass rate) and graded on observable fit + held-out/counterfactual prediction (only reporting parameter error along stiff, identifiable directions). Intrinsically underdetermined regimes are labeled **Type 3a**, where abstaining is the correct answer.
4. **"Can a small team realistically build a 'Last Exam' benchmark in a 6-week pilot?"**
   - Depth, reverse screening, and executable verification—not 2,500 static MCQs—define a scientific exam. Phase 0 builds 10–15 rigorously certified prototype tasks (starting with 8 core cross-environment tasks in Gate 1) to establish hard empirical failure rates, before scaling to 30–50 tasks in v1.0.
5. **"Haven't LLMs memorized classic HH, Brunel, Allen, and FastDMF defaults from pretraining?"**
   - Four defenses: (a) randomized ground truths strictly away from published defaults (`TuneBench` rule); (b) three-tier disclosure ablation (`full` / `names_only` / `blinded` as `p0, p1, ...`); (c) unpublished collaborator data and held-out counterfactuals; (d) strict `dev` vs. frozen/rolling `test` separation.
6. **"Why claim reduced human debugging effort without a large-scale user study in Phase 0?"**
   - **Rule 5.2 / ADR-009** explicitly forbids quantitative claims about human time reduction until human expert logs are collected. Phase 0 uses fixed expert-rule baselines and structured `ask_human` gap items as reproducible proxies.

---

## 5. Recommended Presentation Flow

Always present the project in this order:
1. **Real scientific bottleneck**: Ill-posed neural dynamics inverse problems and manual calibration debugging cost;
2. **Sharp research gap**: Diagnostics stuck at reporting + LLM long-horizon degradation and refusal to abstain;
3. **Executable carrier**: **NeuroDyn-Bench** (3 task types x reverse screening x unified hard budget x counterfactual verification);
4. **Honest empirical evaluation**: Four-group controls + open sandbox testing, letting data determine where diagnostic agents genuinely earn their cost.
