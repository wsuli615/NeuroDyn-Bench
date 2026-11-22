# Design Evolution, Architecture Decision Records (ADRs), and Open Questions

This log records how **NeuroDyn-Bench** converged from an initial tuning-agent idea to a benchmark-first diagnostic calibration platform, locking key architectural decisions (ADRs) and tracking remaining open questions (OQs).

---

## 1. Design Evolution Timeline

1. **Origin (2026-11-10)**: Motivated by manual expert debugging in neural dynamics parameter calibration and the difficulty of separating setup errors from model mismatch.
2. **Benchmark-First Pivot**: Surveyed 12 neighboring 2024–2026 benchmarks (`Neuroptimus`, `sbibm`, `PEtab`, `TuneBench`, `HydroAgent`, `SimulCost`, `BBOWP-Bench`, `AgentHPOBench`, `NeuronBench`, `ModelSMC`, `unsatz`, `CtDB`) and confirmed no existing suite tests setup + multi-round diagnosis-to-action + principled abstention (`3a`/`3b`) + spiking/whole-brain scales under unified costs.
3. **Cross-Disciplinary Alignment**: Authored [`primer_for_cs.md`](primer_for_cs.md) mapping single-neuron, SNN (CBN/SBN + SNOPS 6 statistics), whole-brain (`FastDMF`), and inverse-problem pathology into CS/optimization terminology.
4. **Formal Proposal Integration (2026-11-21)**: Locked hard scope boundaries against `BMD`/`ModelSMC` and `SNOPS`; split Type 3 cleanly into `3a` (non-identifiable/degenerate) vs. `3b` (fixed-structure unreachable); added anti-rule-tie ambiguity tasks and paired anti-gaming metrics.
5. **Dual-Track Benchmark Upgrade & Gate 0 Completion (2026-11-22)**: Upgraded to an HLE-inspired executable exam (`NeuroDyn-Bench`), completed Gate 0 empirical profiling of `Small E-I` and canonical C++ `FastDMF` on `dgx2felis` (resolving OQ-04), and adopted a **Hybrid Dual-Track** harness (Track A optional toolbox with lightweight Bayesian inference + Track B open Python sandbox) so evaluated AI systems autonomously choose their own methods.

---

## 2. Architecture Decision Records (ADRs)

| ID | Decision | Rationale | Scope |
| --- | --- | --- | --- |
| **ADR-001** | Benchmark-First positioning named `NeuroDyn-Bench` | Foundation models evolve rapidly; a hard, executable, leak-resistant benchmark has durable impact and anchors method iteration | Repo scope, paper framing |
| **ADR-002** | Fixed model structure; focus on calibration & process diagnosis | Clean separation from equation-discovery agents (`BMD`/`ModelSMC`/`unsatz`) and passive reporting (`SNOPS`/PPC) | Task scope, action space |
| **ADR-003** | Outer-loop/inner-loop split + Dual-Track autonomous method choice | LLMs underperform numerical optimizers on mid/late point guessing (`SimulCost`, `arXiv:2610.22379`); benchmark exposes optional optimizers, lightweight Bayesian inference, and diagnostic tools (Track A) plus open Python sandbox (Track B) under one `BudgetLedger` | Gateway, toolbox, sandbox |
| **ADR-004** | First-class selective abstention with mandatory `3a` vs. `3b` split | `3a` (low error, degenerate parameters) and `3b` (high error, feature trade-off) exhibit opposite symptoms and require distinct probes and grading | Syllabus, tools, evaluator |
| **ADR-005** | HLE-style reverse screening + 1:1 Type 2/3 pairing + offline `validation_cert.yaml` | Prevents trivial tasks, blocks "always abstain" / "never abstain" gaming, and proves labels mathematically/empirically | QA gates, cert scripts |
| **ADR-006** | Strict three-way physical separation (`public/` vs. `private/`) & all-inclusive `BudgetLedger` | Prevents data leakage and unbilled probes; LLM costs reported separately alongside physical cost-sensitivity sweeps | `tasks/`, `gateway/`, `eval/` |
| **ADR-007** | Four control groups (G1–G4) + multi-cause ambiguity & physical hard-bound tasks | Separates optimizer, rule heuristic, general tool use, and diagnostic protocol gains; prevents trivial if-else ties | Baselines, Type 2 design |
| **ADR-008** | Off-the-shelf frontier models + single agent in Phase 0 (no custom training) | Prioritizes interpretability, reproducibility, and clean attribution before investing in RL/SFT | Agents |
| **ADR-009** | Strict empirical honesty boundary | No quantitative claims on human time savings without human trials; narrow method claims candidly if deterministic rules suffice | Metrics, Gate 3 decision |
| **ADR-010** | English-only repository artifacts (`wsuli615` attribution) | Ensures clean international open-source and publication readiness; raw founding Chinese session logs retained under `handoff/` | All active repo files |

---

## 3. Open Questions Tracker (OQs)

| ID | Open Question | Owner | Status / Blocking Impact |
| --- | --- | --- | --- |
| **OQ-01** | Which real stalled modeling case serves as the first collaborator capstone (and finalizes Small E-I vs. FastDMF primary/secondary priority)? | Collaborators + PI | Pending (`docs/collaborator_brief.md` ready); blocks T09/T15 |
| **OQ-02** | Who delivers real data samples, model code, and preprocessing scripts, and on what timeline? | Collaborators | Pending; blocks Level 2/3 real cases |
| **OQ-03** | Exact joint acceptance criteria (observables, $\tau$, held-out validation protocol) for real cases | Collaborators + Team | Pending; blocks T09/T15 evaluator lock |
| **OQ-04** | Single-sim runtime, stochastic variance, and parallel scaling of Small E-I and FastDMF on `dgx2felis` | `wsuli615` | **Resolved (2026-11-22)** — see [`gate0_report.md`](gate0_report.md) and [`configs/budget.yaml`](../configs/budget.yaml) |
| **OQ-05** | Select 2–3 frontier LLM backbones and API budget cap for G3/G4 screening | PI + Team | Scheduled for Gate 2 |
| **OQ-06** | Whether Phase 0 includes small-scale human expert timing logs on 3–5 representative tasks | Collaborators + PI | Determines whether human-time quantitative claims are reported |
| **OQ-07** | Private test split hosting and v1.0 rolling update protocol | Engineering | Currently isolated via `.gitignore` (`private/`); formal hosting for v1.0 |
