# Gate 0 实证报告：双候选环境最小复现与 `dgx2felis` 算力标定

- **日期**：2026-11-22
- **机器**：`dgx2felis`（256 CPU 核 / 1.0 TiB 内存 / 8 × NVIDIA A100-SXM4-40GB / `g++ 13.3` / `Python 3.12` + `uv`）
- **归档人**：`wsuli615`
- **解决开放问题**：**OQ-04**（产出 [`configs/budget.yaml`](../configs/budget.yaml)）

---

## 1. 复现概览与验证结论

本阶段在 `dgx2felis` 上独立搭建并跑通了两大候选仿真环境及统一预算账本（`BudgetLedger`），全部 3 项自动化单元测试通过（`pytest tests/test_gate0.py`）：

| 环境 | 核心实现 | 关键验证结果 |
| --- | --- | --- |
| **Gate 0a：Small E-I 脉冲网络** | `src/neurodyn_bench/models/small_ei.py`（1000 个 LIF 神经元 800E/200I、Dale 定律、传导延迟、指数突触、Numba 精确 Poisson CDF 加速，支持 CBN / SBN 双拓扑 + **SNOPS 6 项群体统计量**） | 清晰分离沉默态 / 异步不规则态 / 失控爆发态；10 种子复现与 G1 CMA-ES 孪生校准全部跑通 |
| **Gate 0b：FastDMF 全脑神经质量模型** | `src/neurodyn_bench/models/_fastdmf_cpp.cpp` + `fastdmf_env.py`（通过 `pybind11` 直连上游 `concog/fastdmf` 规范 C++ 核心 `DMF.hpp` + Eigen + Schaefer-100 HCP 连接组 + 并行 Balloon-Windkessel BOLD） | 与上游确定性参考 `rates_deco.csv` / `bold_deco.csv` 误差达机器精度（$\le 8.4 \times 10^{-15}$ / $6.5 \times 10^{-17}$）；跑通静息态 FC/FCD 与 G1 CMA-ES 校准 |

---

## 2. 实测性能与并行扩展性（支撑 `configs/budget.yaml`）

> **关键工程发现**：在 256 核 `dgx2felis` 上做多进程仿真并行时，必须为每个 worker 显式设置 `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`，否则 NumPy / scikit-learn 内部 BLAS 线程争用会严重拖慢吞吐。

| 指标 | Small E-I（1000 LIF, 10 s 评估 + SNOPS 6 统计量） | FastDMF（100 ROI, 160 s BOLD, TR=2 s） |
| --- | --- | --- |
| 单核完整仿真中位耗时（`1.0u`） | **3.088 s**（均值 3.090 s） | **11.188 s**（均值 11.198 s） |
| 单核短探针中位耗时 | **0.804 s**（`0.2u`，加速 3.84×） | **2.813 s**（`0.25u`，加速 3.98×） |
| 8 进程吞吐（32 次仿真墙钟） | **2.56 sims/s**（12.50 s，7.9×） | **0.70 sims/s**（45.71 s，7.8×） |
| 16 进程吞吐（32 次仿真墙钟） | **4.92 sims/s**（6.50 s，15.2×） | **1.40 sims/s**（22.93 s，15.6×） |
| 32 进程吞吐（32 次仿真墙钟） | **8.68 sims/s**（3.69 s，**26.8×**） | **2.48 sims/s**（12.91 s，**27.7×**） |

**对评测可行性的直接意义**：
- 以 32 核并行计算，单次 200 单位预算的 Small E-I 任务仅需约 **23 秒**，10 种子全量约 **4 分钟**；即使执行 **10× 预算（2000 单位）**的 Type 3b 离线不可达校验，也仅需约 **3.8 分钟**。
- 单次 160 单位预算的 FastDMF 全脑任务在 32 核下约 **65 秒**，5 种子约 **5.4 分钟**；10× 预算离线校验约 **11 分钟**。`dgx2felis` 足以轻松支撑 Phase 0 的全部反向过筛与多随机种子封闭评测。

---

## 3. 相区检查与 10 种子随机波动分析

### 3.1 Small E-I（CBN 与 SBN）

- **相区分离**：
  - `cbn_ai_nominal`：`feasible=True`，`fr_e=12.70 Hz`，`fr_i=14.76 Hz`，`isi_cv=0.658`，`ff=0.571`，`rsc=0.008`，`pct_sh=5.55%`，`pop_psd_peak_hz=30.8 Hz`；
  - `cbn_silent`（低外输入+高抑制）：`feasible=False (silent)`，`fr_e=0.0 Hz`；
  - `cbn_runaway`（低抑制 `g_ei=1.2`）：`feasible=False (runaway)`，`fr_e=283.9 Hz`。
- **10 种子随机方差（CBN vs. SBN）**：

| 统计量 | CBN（均值 ± SD，CV） | SBN（均值 ± SD，CV） | 诊断启示 |
| --- | --- | --- | --- |
| `fr_e` (Hz) | `12.667 ± 0.085`（0.007） | `13.536 ± 0.121`（0.009） | 均值极稳，单次评估即可靠 |
| `isi_cv` | `0.652 ± 0.019`（0.029） | `0.657 ± 0.014`（0.021） | 较稳 |
| `ff` (Fano) | `0.560 ± 0.020`（0.036） | `0.542 ± 0.013`（0.024） | 中等波动 |
| `rsc` | `0.0051 ± 0.0050`（**0.977**） | `0.0068 ± 0.0041`（**0.607**） | 试次方差极大，单次评估极易误导优化器 |
| `pct_sh` (%) | `5.10 ± 0.38`（0.074） | `5.94 ± 1.57`（**0.264**） | 空间网络群体协变波动更强，需复测压方差 |

### 3.2 FastDMF（Schaefer-100 HCP）

- **关键模型物理发现（连接组尺度与分岔边界）**：
  - 上游 `dmf_api.py` 默认 `C / C.max()` 归一化下，当 `sigma = 0.01`（以 ms 计的 `dt=0.1` 放大噪声 $\sqrt{1000}$ 倍）时，网络在 `G=2.0` 处直接越过鞍结分岔进入 113–478 Hz 爆发上支；
  - 采用 Deco (2014) 规范尺度 `C = 0.2 * C / C.max()` 与 `sigma = 0.007` 后，100 个脑区稳定落在 **1.2–3.0 Hz** 静息态工作点（`nominal_G_2.2`：`mean_rate_hz = 1.79 ± 0.02 Hz`、`fc_mean = 0.132 ± 0.012`、`fcd_mean = 0.355 ± 0.022`），而削弱反馈抑制（`broken_fic_runaway`, `fic_alpha=0.12`）则稳定触发 `rate_runaway`（`137.4 Hz`，峰值 `322.9 Hz`）。
  - 在尝试 `G=2.75, w=1.60, sigma=0.0085, fic_alpha=0.72` 时，部分随机种子触发了间歇性爆发——直观暴露了全脑模型在临界工作点附近的陡峭分岔边界（非常适合构造 T06/T08 类考题）。

---

## 4. 最小 G1 基线校准与“自发参数补偿”现象

两个环境均在 `BudgetLedger(max_units=80.0)` 约束下运行了最小 4 维孪生实验（G1 基线：CMA-ES，`popsize=8`，10 代，8 核并行），结果意外但高度贴合项目理论地基：

1. **Small E-I 4 维孪生实验**（活动参数 `J_ee, g_ei, tau_i, nu_ext_e`，耗时 33.8 s，计费 80.0/80.0 单位，含 4 次失败仿真）：
   - 独立 5 新种子验证 **5 项统计量全部达标**（`|z|`：`fr_e=0.339, ff=1.852, rsc=0.197, pct_sh=0.605, isi_cv=1.247`，均 $\le 2.0$）；
   - 但反演参数为 `(J_ee=1.96, g_ei=5.37, tau_i=8.69, nu_ext_e=5.79)`，与真值 `(1.55, 6.10, 10.50, 6.00)` 存在明显偏移——强兴奋连接被更快抑制与略低外输入补偿。
2. **FastDMF 4 维孪生实验**（活动参数 `G, w, sigma, fic_alpha`，耗时 113.5 s，计费 80.0/80.0 单位，含 7 次爆发失败）：
   - 独立 5 新种子验证 `fc_corr = 0.6042`（与真值单次噪声下界 `0.5707` 相当甚至略高）、`fcd_ks = 0.2487`、`mean_rate_hz = 1.31 Hz`（目标 `1.44 Hz`）；
   - 但反演参数 `(G=3.13, w=1.20, sigma=0.0054, fic_alpha=0.63)` 与真值 `(2.65, 1.48, 0.0065, 0.78)` 沿全局耦合–局部复发–FIC 补偿流形显著漂移。

**对 Gate 1 出题的核心启示**：两个候选环境均天然具备 **sloppiness / degeneracy（参数补偿）**、**陡峭分岔无效区** 与 **高试次随机方差**，无需人为造假就能构造出极其真实的 Type 2（可恢复困难）与 Type 3a/3b（不可辨识 / 结构不可达）考题；同时这也印证了 `docs/rules.md` 的硬性规定——绝不能只看训练误差，必须引入**隐藏新条件/反事实干预验证**和**离线可辨识性审计**。

---

## 5. 复现命令

```bash
# 1. 编译 FastDMF 规范 C++ pybind11 扩展（自动拉取 concog/fastdmf 并拷贝 Schaefer-100 连接组）
.venv/bin/python scripts/build_fastdmf.py

# 2. 运行单元测试（含 BudgetLedger 硬截断/缓存、Small E-I、FastDMF 上游精度核对）
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/pytest -v tests/test_gate0.py

# 3. 复现 Gate 0 双环境性能、方差与基线校准基准
.venv/bin/python scripts/gate0_small_ei.py
.venv/bin/python scripts/gate0_fastdmf.py
```
