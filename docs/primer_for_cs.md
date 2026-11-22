# Neural Dynamics Primer for CS and AI Contributors

A 10-minute technical primer mapping biophysical neural dynamics concepts, multi-scale models, and inverse-problem pathologies onto machine learning and black-box optimization concepts.

---

## 1. One-Sentence Definition

Computational neural dynamics models biological neurons, local circuits, or whole-brain networks as stochastic nonlinear dynamical systems (or hybrid discrete-event spiking systems) and calibrates physically meaningful parameters to match experimental summary statistics:

$$\frac{dx}{dt} = f(x, u; \theta) + \xi(t), \qquad y = h(x) + \epsilon$$

From a CS perspective, a neural dynamics model is a **non-differentiable, stochastic, expensive black-box simulator whose parameters carry physical units and hard biological bounds**. Calibration is the ill-posed inverse problem of inferring $\theta$ (or certifying that $\theta$ cannot be uniquely inferred) from partial summary observations $y$.

---

## 2. Three Biophysical Scales

### 2.1 Single Neurons (3–16 Parameters)

The cell membrane behaves like a leaky RC circuit with membrane potential $V$ (resting $\approx -70\text{ mV}$). When input drives $V$ across threshold ($\approx -50\text{ mV}$), the neuron emits a ~1 ms discrete electrical pulse (an action potential / spike).

- **Hodgkin–Huxley (HH)**: Detailed ion-channel gating ODEs reproducing full spike waveforms.
- **LIF / AdEx**: Threshold-and-reset abstractions; AdEx adds exponential spike initiation and spike-frequency adaptation (firing slows under sustained current).
- **Observables**: Patch-clamp current-clamp features (F-I curve, adaptation index, first-spike latency, ISI CV).

### 2.2 Spiking Neural Networks (8–11 Core Parameters — Candidate Environment 1)

Networks of $N$ excitatory (E) and inhibitory (I) neurons (typically 4:1 cortical ratio) obeying **Dale's law** (all outgoing synapses from one neuron share the same sign), coupled via sparse adjacency matrix $W$ with synaptic time constants ($\tau_E \approx 2\text{–}10\text{ ms}$, $\tau_I \approx 5\text{–}20\text{ ms}$) and conduction delays:

- **E/I Balance**: Large opposing excitatory and inhibitory currents roughly cancel, leaving fluctuations that drive cortical-like **Asynchronous Irregular (AI)** firing.
- **SNOPS Paradigm (Wu et al. 2024)**:
  - **CBN (Classical Balanced Network)**: Uniform random Erdős–Rényi connectivity.
  - **SBN (Spatial Balanced Network)**: Neurons embedded on a 2D periodic sheet with distance-dependent connectivity widths $(\sigma_E, \sigma_I)$.
  - **Six Population Activity Statistics**: Mean firing rate (`fr_e`), single-neuron Fano factor (`ff`), pairwise spike-count correlation (`rsc`), and three Factor Analysis covariability metrics—percent shared variance (`%sh`), shared dimensionality (`dsh`), and shared eigenspectrum (`es`). SNOPS showed that CBN cannot jointly match all six macaque V4 statistics while SBN can—the canonical prototype for our **Type 3b (structurally unreachable)** tasks.

### 2.3 Neural Mass and Whole-Brain Models (8–100+ Parameters — Candidate Environment 2)

- **Wilson–Cowan / Dynamic Mean Field (DMF)**: Instead of individual spikes, each brain region is represented by coupled E-I population rate ODEs.
- **FastDMF (Herzog et al. 2024; Deco et al. 2014)**: Couples 100 cortical regions (Schaefer-100 HCP structural connectome $C$) via global coupling $G$, local excitatory recurrence $w$, and analytic Feedback Inhibition Control ($J_i = \alpha G S_i + 1$) to maintain ~1–3 Hz resting rates, passing firing rates through a Balloon–Windkessel hemodynamic filter to generate simulated resting-state fMRI BOLD time series evaluated via static **Functional Connectivity (`FC`)** and sliding-window **Functional Connectivity Dynamics (`FCD`)**.

---

## 3. Why Neural Dynamics Calibration Is Harder Than Standard ML HPO

1. **Bifurcations and massive infeasible regions**: Small parameter shifts cross phase transitions into total silence (`0 Hz`) or epileptic runaway (`150–400+ Hz`), breaking smooth surrogates.
2. **Multi-order parameter scales**: Synaptic and membrane time constants span decades; linear search boxes stall.
3. **High evaluation cost and stochastic variance**: Poisson drive and finite-window sampling produce huge trial-to-trial variance (e.g., `rsc` CV $\approx 0.98$ in Gate 0a), causing single-eval optimizers to chase noise.
4. **Partial, non-differentiable summary observations**: Discrete spikes and macroscopic BOLD summaries discard microscopic trajectories.
5. **Sloppiness, Degeneracy, and Non-Identifiability**:
   - **Sloppiness** (Gutenkunst 2007): Fisher eigenvalues span orders of magnitude—only a few "stiff" combinations control observations while most "sloppy" directions drift freely.
   - **Degeneracy** (Prinz 2004): Distinct parameter vectors produce identical baseline observables yet diverge under perturbations.
   - **Non-Identifiability & Summary Loss** (Hartoyo 2019; Wischnewski 2025; Cai 2026): Low training loss does **not** guarantee unique parameters; stopping honestly (`3a`) beats fabricating a single point.
6. **Structural incompleteness (`3b`)**: Fixed equations may be fundamentally incapable of simultaneously matching all recorded population statistics.

---

## 4. Quick Glossary

| Term | Concise Meaning |
| --- | --- |
| **Firing rate (`fr`)** | Spikes per second per neuron (Hz) |
| **ISI CV** | Inter-spike interval SD / mean (~1 for Poisson irregular firing) |
| **Fano factor (`ff`)** | Spike-count variance / mean across non-overlapping time windows |
| **Spike-count correlation (`rsc`)** | Mean pairwise Pearson correlation of window spike counts across neuron pairs |
| **`%sh` / `dsh` / `es`** | Factor Analysis percent shared variance, 95%-variance shared dimensionality, and shared eigenspectrum |
| **Dale's Law** | Each neuron makes exclusively excitatory or exclusively inhibitory outgoing synapses |
| **CBN / SBN** | Classical (random) vs. Spatial (2D distance-dependent) Balanced Network |
| **FIC** | Feedback Inhibition Control stabilizing whole-brain regional firing rates near ~1–3 Hz |
| **FC / FCD** | Static BOLD functional connectivity matrix / sliding-window dynamic FC distribution |
| **Sloppiness / Degeneracy** | Extreme sensitivity anisotropy / multiple distinct parameter sets yielding equivalent outputs |
| **Twin Experiment** | Synthetic data generated from a known ground-truth parameter vector $\theta^*$ for controlled evaluation |
