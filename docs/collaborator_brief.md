# Neuroscience Collaborator Alignment Brief

> **To our domain collaborators**: This short brief is designed for a 30-minute alignment before locking our first real-data capstone tasks (`T09`, `T15`). We aim to ground **NeuroDyn-Bench** directly in real calibration bottlenecks from active computational neuroscience labs rather than toy synthetic setups alone.

---

## 1. Project Scope in One Minute

- **Your model structure stays untouched**: We do **not** replace mechanistic equations with black-box surrogates or let AI rewrite your ODEs/spiking equations.
- **What we build and test**: An executable calibration & process-diagnosis benchmark plus diagnostic outer-loop workflows that detect *why* an optimizer stalled (bounds too narrow, high stochastic variance, coarse time step, parameter compensation / non-identifiability, or fixed-structure feature conflict), adapt the search strategy, and **stop honestly with auditable evidence** when the data cannot constrain the parameters.

---

## 2. Three Decisions Needed Up Front (OQ-01 ~ OQ-03)

1. **First real modeling bottleneck (OQ-01)**: We have validated two candidate environments on our server—**Small E-I spiking networks (8–10D, CBN/SBN)** and **FastDMF whole-brain neural mass models (Schaefer-100 HCP, 8–100+D)**. Which matches your lab's most pressing calibration bottleneck right now (or should we integrate a specific lab model)?
2. **Data, code, and preprocessing handoff (OQ-02)**: Who will provide the sample data, model code, and feature-extraction scripts, under what privacy/sharing terms (public `dev` vs. internal-only `private/` evaluation), and on what timeline?
3. **Joint acceptance criteria (OQ-03)**: Which summary statistics must be matched, what error tolerances count as success, and what held-out subjects, stimuli, or interventions can serve as independent validation?

---

## 3. Minimal Starter Materials Requested

Starting with **one** real case is enough to proceed; 2–4 additional cases can follow during the pilot:

1. **One real data sample**: Recording modality (patch-clamp, multi-electrode array, LFP, EEG, fMRI), experimental condition, and preprocessing / summary-statistic script.
2. **One concrete stalled calibration case**: Target phenomenon, what currently fails, existing model code, and any historical fitting attempts or debugging notes.
3. **Biophysical dual bounds**: Distinguishing arbitrary initial search bounds (safely expandable) from strict physiological hard limits (must never be violated).

---

## 4. Real Modeling Case Intake Template (~5 Minutes per Case)

```markdown
### Case ID: CASE-01 (<Short Title>)

1. Model & Scale:
   - Model name / repo or script location:
   - Free parameters (name, unit, default search range, physiological hard bounds):

2. Data & Target Observables:
   - Modality & sharing tier (public / private-eval only):
   - Target summary statistics (and empirical trial-to-trial SD or acceptable tolerance):

3. Observed Calibration Bottleneck:
   - Optimizer / budget used previously:
   - Failure symptoms (high plateau, boundary saturation, run-to-run drift, runaway/silent runs):

4. Manual Debugging History & Verdict (Highest Value):
   - What competing hypotheses did you test?
   - Root cause (recoverable workflow issue / 3a underdetermined data / 3b missing mechanism)?
   - If recovered, what change fixed it, and roughly how much human debugging time did it take?

5. Acceptance & Independent Validation:
   - Numeric pass thresholds:
   - Available held-out subjects, stimuli, or intervention conditions:
```

---

## 5. Optional Human Expert Timing Reference (OQ-06)

If lab bandwidth allows later in Phase 0, recording active decision time and intervention counts from 1–2 modelers on 3–5 representative tasks will provide a grounded human reference. Until such logs exist, all project reports strictly refrain from quantitative claims about human time savings (Rule 5.2 / ADR-009).
