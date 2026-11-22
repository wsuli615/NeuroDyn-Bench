# Gate 0 Empirical Report: Dual-Environment Minimal Reproductions on `dgx2felis`

- **Date**: 2026-11-22
- **Host**: `dgx2felis` (256 CPU cores, 1.0 TiB RAM, 8 x NVIDIA A100-SXM4-40GB, `g++ 13.3`, Python 3.12 + `uv`)
- **Author**: `wsuli615`
- **Resolves**: **OQ-04** (producing [`configs/budget.yaml`](../configs/budget.yaml))

---

## 1. Reproduction Summary and Verification

Both candidate environments and the unified `BudgetLedger` ([`src/neurodyn_bench/optim/budget.py`](../src/neurodyn_bench/optim/budget.py)) have been implemented, unit-tested (`pytest -v tests/test_gate0.py`: 3/3 passed), and empirically benchmarked:

| Environment | Implementation | Verification Highlights |
| --- | --- | --- |
| **Gate 0a: Small E-I Spiking Network** | [`src/neurodyn_bench/models/small_ei.py`](../src/neurodyn_bench/models/small_ei.py) (1,000 LIF neurons 800E/200I, Dale's law, conduction delays, exponential synapses, Numba exact Poisson CDF acceleration, CBN & SBN topologies, all 6 SNOPS population statistics) | Clean separation across silent, AI balanced, and runaway regimes; 10-seed stochasticity and G1 CMA-ES twin calibration verified |
| **Gate 0b: FastDMF Whole-Brain Model** | [`src/neurodyn_bench/models/_fastdmf_cpp.cpp`](../src/neurodyn_bench/models/_fastdmf_cpp.cpp) + [`fastdmf_env.py`](../src/neurodyn_bench/models/fastdmf_env.py) (native `pybind11` binding to canonical `concog/fastdmf` `DMF.hpp` + Eigen + Schaefer-100 HCP connectome + parallel Balloon–Windkessel BOLD) | Bit-level match to upstream `rates_deco.csv` ($\le 8.4 \times 10^{-15}$) and `bold_deco.csv` ($\le 6.5 \times 10^{-17}$); resting FC/FCD and G1 CMA-ES verified |

---

## 2. Runtime and Parallel Scaling on `dgx2felis`

> **HPC Note**: Pinning `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1` per worker process is mandatory on 256-core servers to avoid BLAS thread oversubscription during multi-process evaluation.

| Metric | Small E-I (1,000 LIF, 10 s eval + SNOPS 6 stats) | FastDMF (100 ROIs, 160 s BOLD, TR = 2 s) |
| --- | --- | --- |
| Single-core full sim median (`1.0u`) | **3.088 s** (mean 3.090 s) | **11.188 s** (mean 11.198 s) |
| Single-core short probe median | **0.804 s** (`0.2u`, 3.84x faster) | **2.813 s** (`0.25u`, 3.98x faster) |
| 8-worker throughput (32 sims wall) | **2.56 sims/s** (12.50 s, 7.9x) | **0.70 sims/s** (45.71 s, 7.8x) |
| 16-worker throughput (32 sims wall) | **4.92 sims/s** (6.50 s, 15.2x) | **1.40 sims/s** (22.93 s, 15.6x) |
| 32-worker throughput (32 sims wall) | **8.68 sims/s** (3.69 s, **26.8x**) | **2.48 sims/s** (12.91 s, **27.7x**) |

**Feasibility Implications**:
- At 32 workers, a 200-unit Small E-I run takes ~23 s (and a 10x offline Type 3b certification run of 2,000 units takes ~3.8 min).
- At 32 workers, a 160-unit FastDMF run takes ~65 s (and a 10x offline certification run of 1,600 units takes ~11 min).

---

## 3. Regimes and 10-Seed Stochastic Variance

### 3.1 Small E-I (CBN vs. SBN)

- **Regimes**: `cbn_ai_nominal` (`feasible=True`, `fr_e=12.70 Hz`, `fr_i=14.76 Hz`, `isi_cv=0.658`, `ff=0.571`, `rsc=0.008`, `pct_sh=5.55%`, `pop_psd_peak_hz=30.8 Hz`); `cbn_silent` (`feasible=False`, `fr_e=0.0 Hz`); `cbn_runaway` (`feasible=False`, `fr_e=283.9 Hz`).
- **10-Seed Variance**:

| Observable | CBN (Mean ± SD, CV) | SBN (Mean ± SD, CV) | Diagnostic Relevance |
| --- | --- | --- | --- |
| `fr_e` (Hz) | `12.667 ± 0.085` (0.007) | `13.536 ± 0.121` (0.009) | Stable under single eval |
| `isi_cv` | `0.652 ± 0.019` (0.029) | `0.657 ± 0.014` (0.021) | Low variance |
| `ff` | `0.560 ± 0.020` (0.036) | `0.542 ± 0.013` (0.024) | Moderate variance |
| `rsc` | `0.0051 ± 0.0050` (**0.977**) | `0.0068 ± 0.0041` (**0.607**) | Extreme trial noise; single-eval misleads optimizers |
| `pct_sh` (%) | `5.10 ± 0.38` (0.074) | `5.94 ± 1.57` (**0.264**) | High covariability variance; rewards adaptive repeats |

### 3.2 FastDMF (Schaefer-100 HCP)

- **Connectome Normalization & Bifurcation Finding**: Under upstream `dmf_api.py`'s `C / C.max()` scaling with `sigma = 0.01` (`dt = 0.1 ms`), noise pushes the Wong–Wang dynamics across the saddle-node bifurcation into 113–478 Hz runaway. Adopting Deco (2014) connectome scaling (`C = 0.2 * C / C.max()`, `sigma = 0.007`) stabilizes all 100 ROIs at **1.2–3.0 Hz** (`nominal_G_2.2`: `mean_rate_hz = 1.792 ± 0.020 Hz`, `fc_mean = 0.132 ± 0.012`, `fcd_mean = 0.355 ± 0.022`), while degraded FIC (`fic_alpha = 0.12`) cleanly triggers `rate_runaway` (`137.4 Hz`, max `322.9 Hz`).

---

## 4. Minimal G1 CMA-ES Twin Calibration and Spontaneous Parameter Degeneracy

Both environments ran a 4D G1 CMA-ES baseline under `BudgetLedger(max_units=80.0)` (`popsize=8`, 10 generations, 8 workers):

1. **Small E-I** (`J_ee, g_ei, tau_i, nu_ext_e`; 33.8 s wall, 80/80 units, 4 infeasible runs): All 5 observables passed independent 5-seed verification (`|z|` = `0.339, 1.852, 0.197, 0.605, 1.247 <= 2.0`), yet recovered parameters `(1.96, 5.37, 8.69, 5.79)` drifted noticeably from ground truth `(1.55, 6.10, 10.50, 6.00)` along an E-I compensation manifold.
2. **FastDMF** (`G, w, sigma, fic_alpha`; 113.5 s wall, 80/80 units, 7 runaway failures): Achieved `fc_corr = 0.6042` (matching/exceeding the 5-seed GT floor `0.5707`) and `mean_rate_hz = 1.31 Hz` (target `1.44 Hz`), yet recovered `(3.13, 1.20, 0.0054, 0.63)` drifted ~18–19% from ground truth `(2.65, 1.48, 0.0065, 0.78)` along the global-coupling / local-recurrence / FIC compensation manifold.

**Takeaway for Gate 1**: Both simulators naturally exhibit sloppiness/degeneracy, sharp bifurcation boundaries, and high stochastic variance—directly motivating hidden counterfactual verification and Type 3a/3b selective abstention.

---

## 5. Reproduction Commands

```bash
.venv/bin/python scripts/build_fastdmf.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/pytest -v tests/test_gate0.py
.venv/bin/python scripts/gate0_small_ei.py
.venv/bin/python scripts/gate0_fastdmf.py
```
