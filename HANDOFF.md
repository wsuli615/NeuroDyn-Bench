# HANDOFF: NeuroDyn-Bench Project Onboarding Guide

- **Handoff Date**: 2026-11-21 (updated 2026-11-22 on `dgx2felis`)
- **Origin Session**: `bc-52ccf2c0-d67f-4850-9265-bdfa94c8b874`
- **Repository**: `https://github.com/wsuli615/NeuroDyn-Bench`
- **Compute & Development Directory**: `/home/ydeng/Code/NeuroDyn-Bench` on `dgx2felis` (256 CPU, 1.0 TiB RAM, 8 x A100-40GB)
- **Maintainer / Attribution**: `wsuli615`

This guide summarizes the complete project context, locked design decisions (ADRs), open questions (OQs), empirical Gate 0 status, and immediate Gate 1 execution plan.

---

## 1. Mission & Positioning

**NeuroDyn-Bench** is an executable, cost-aware, benchmark-first platform for evaluating and building diagnostic calibration agents for mechanistic neural dynamics models. Its distinguishing core combines:

- **Fixed-structure outer-loop calibration** (strictly separated from equation/structure discovery like `BMD`/`ModelSMC`/`unsatz` and passive reporting like classic PPC/`SNOPS`);
- **Unified hard simulation budgeting** (`BudgetLedger` billing every optimization evaluation, repeat, short probe, Bayesian sample, and failed/divergent run; LLM tokens/latency tracked separately with cost-sensitivity scans);
- **First-class scientific honesty**: Hard split between **Type 3a** (non-identifiable / degenerate / summary information loss) and **Type 3b** (fixed-structure unreachable with feature trade-offs), paired 1:1 against hard-but-solvable **Type 2** tasks.

See [`north_star.md`](north_star.md), [`docs/narrative.md`](docs/narrative.md), [`docs/architecture.md`](docs/architecture.md), and [`docs/rules.md`](docs/rules.md) for complete specifications.

---

## 2. Architecture & Dual-Track Evaluation Design

To avoid forcing evaluated AI systems into a single fixed algorithm, the harness enforces **one physical isolation and budgeting gateway** while supporting **two evaluation tracks**:

- **Gateway (`SimulatorGateway` + `BudgetLedger`)**: Single entry point for all simulator calls (`Small E-I`, `FastDMF`), enforcing physiological hard bounds, exact-key cache exemption, short-probe cost conversion, and append-only `trace.jsonl` logging.
- **Track A (Structured Optional Toolbox)**: Modular tool registry exposing inner-loop optimizers (BO, CMA-ES, multistart), **optional lightweight Bayesian inference tools** (ABC-SMC, Laplace-Fisher eigenspectrum, PPC audit), and discriminative diagnostic probes—ideal for clean G1/G2/G3/G4 ablations.
- **Track B (Open Python Sandbox)**: Exposes only `public/` files, a controlled `SimulatorGateway` client proxy, and pre-installed scientific computing libraries so frontier coding agents freely search, compose solvers, and write custom calibration/diagnostic scripts.
- **Independent Evaluator**: Isolated process holding exclusive access to `private/` (`ground_truth.yaml`, `heldout_conditions.tsv`, `heldout_observables.tsv`, `validation_cert.yaml`).

---

## 3. Key Decisions (ADRs) & Open Questions (OQs)

### Locked ADRs (details in [`docs/discussion_log.md`](docs/discussion_log.md))

- **ADR-001**: Benchmark-First positioning under the name `NeuroDyn-Bench`.
- **ADR-002**: Fixed model structure; focus on parameter calibration and workflow diagnosis (not equation discovery).
- **ADR-003**: Outer-loop / inner-loop separation; agents emit structured workflow actions (or compose solvers in Track B) rather than replacing optimizers with raw point guesses.
- **ADR-004**: Selective abstention is a first-class action with mandatory split between `3a_non_identifiable` and `3b_structure_unreachable`.
- **ADR-005**: HLE-style reverse screening + 1:1 Type 2 vs. Type 3 pairing + mandatory offline `validation_cert.yaml` certification.
- **ADR-006**: Strict three-way physical separation (`public/` vs. `private/`), with all probes/repeats/failures billed to one `BudgetLedger`.
- **ADR-007**: Four standard control groups (G1/G2/G3/G4) + deliberate multi-cause ambiguity and physical hard-bound tasks so simple if-else rules cannot trivially tie diagnostic agents.
- **ADR-008**: Off-the-shelf frontier models + single agent in Phase 0 (no custom LLM training).
- **ADR-009**: Strict empirical honesty—no quantitative claim on expert time savings without human timing trials; narrow agent claims if simple rules suffice.

### Open Questions Status

- **OQ-01 ~ OQ-03** (*Pending domain collaborators*): First real stalled modeling case (deciding primary vs. secondary environment), code/data handoff owner & schedule, and joint acceptance criteria (brief ready in [`docs/collaborator_brief.md`](docs/collaborator_brief.md)).
- **OQ-04** (*Resolved 2026-11-22*): Completed `dgx2felis` empirical profiling for `Small E-I` (3.09 s/sim, 8.68 sims/s on 32 workers) and `FastDMF` (11.19 s/sim, 2.48 sims/s on 32 workers); documented in [`docs/gate0_report.md`](docs/gate0_report.md) and [`configs/budget.yaml`](configs/budget.yaml).
- **OQ-05 ~ OQ-07** (*Scheduled for Gate 2 / v1.0*): Frontier LLM selection and API cap for G3/G4 screening, optional Phase 0 expert timing logs, and rolling private test-split hosting.

---

## 4. Current Status & Immediate Roadmap

- **Gate 0 (Completed)**:
  - Repo initialized and pushed to `wsuli615/NeuroDyn-Bench`;
  - `Small E-I` (Numba LIF + 6 SNOPS statistics) and canonical C++ `FastDMF` (`pybind11` + Schaefer-100 HCP, bit-exact to upstream `10^-15`) verified with `tests/test_gate0.py` and `docs/gate0_report.md`.
- **Gate 1 (In Progress)**:
  - Build `tasks/`, `gateway/`, `tools/` (including optional Bayesian inference tools), `eval/`, and `agents/` (G1/G2/Oracle);
  - Generate and offline-certify the first 8 cross-environment sample tasks (`T02, T03, T05, T08, T11, T13` on Small E-I; `T04, T14` on FastDMF), plus `*_clean` paired no-trap controls for `T03/T04/T05/T08` and 3-tier disclosure rendering (`full` / `names_only` / `blinded`).
- **Gate 2 → Gate 3**: Run closed multi-seed G1–G4 benchmark evaluation and decide v1.0 expansion vs. focused scope.

---

## 5. Historical Archive

Raw original Chinese discussion logs and figures from the founding session are preserved for provenance in:

- `handoff/chat_history/chat_history.md`
- `handoff/attachments/02_task_types_table.png`
- `handoff/attachments/03_research_landscape_mindmap.jpg`
