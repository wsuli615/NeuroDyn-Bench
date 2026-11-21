# HANDOFF: NeuroDyn-Bench 项目完整交接文档

- **交接日期**：2026-11-21 22:45 GMT (UTC+0)
- **原讨论会话**：神经动力学bench (`bc-52ccf2c0-d67f-4850-9265-bdfa94c8b874`)
- **目标 GitHub 仓库**：`https://github.com/wsuli615/NeuroDyn-Bench`
- **预定本地存放路径**：`/Users/yangshen/NeuroDyn-Bench`（M4Pro 本地归档）/ `~/Code/NeuroDyn-Bench`（`dgx2felis` 计算与开发目录）
- **归档与维护署名**：`wsuli615`

这份文档面向下一位接手的开发者或 AI Agent，把截至目前关于 **NeuroDyn-Bench** 的全部背景材料、设计方案、关键架构决策、完整附件与聊天记录索引以及下一步执行计划一次讲清，拿到后可直接开工。

---

## 1. 一句话定位与北极星目标

**NeuroDyn-Bench** 要建成计算神经科学领域首个可执行、成本感知、以结构化诊断与“证据不足敢停”为核心的神经动力学模型自动校准极限基准（Benchmark-First），并在此之上通过四组严格对照，检验把结构化诊断闭环映射为外环搜索动作，究竟能否在固定预算下提高独立验证通过率、提升科学诚实度并减少人工排查。

详细愿景、定位边界与止损标准见 [`north_star.md`](north_star.md)，对外统一叙事与审稿质疑应答见 [`docs/narrative.md`](docs/narrative.md)。

---

## 2. 完整背景与讨论演进

项目在 2026-11-10 至 2026-11-21 期间经历了五步聚焦，才从模糊想法收束成现在边界极清晰的方案（完整演进见 [`docs/discussion_log.md`](docs/discussion_log.md)）：

1. **原始立项构想（2026-11-10）**：针对神经动力学建模调参高度依赖专家经验、试错成本高、拟合不好时难区分“参数没调好”还是“模型机制有问题”，提出让 Agent 调用已有优化器与仿真器并开展失败诊断。
2. **基准调研与转向 Benchmark-First**：系统调研了 2024–2026 年 12 个相关基准（Neuroptimus、sbibm、PEtab、CtDB、BBOWP-Bench、AgentHPOBench、TuneBench、HydroAgent、SimulCost、NeuronBench/MDA、ModelSMC、unsatz），确认没有任何现有基准同时考核：任务设置 + 多轮诊断→动作 + “证据不足敢停（3a/3b）” + 神经群体/脉冲网络尺度 + 统一仿真与 LLM 成本 + 人工介入。确立“先建考场，再做 Agent”。
3. **跨学科对齐（CS + 神经科学）**：沉淀了面向计算机背景队友的精准入门手册（[`docs/primer_for_cs.md`](docs/primer_for_cs.md)），把单细胞、SNN（含 SNOPS 的 CBN/SBN 与 6 个群体统计量）、神经质量/全脑（FastDMF）以及 sloppiness / degeneracy / identifiability 全部打通。
4. **整合团队正式立项稿与 21 篇文献脉络（2026-11-21）**：汇入飞书立项正文、相关工作研究脉络、总结、任务类型表和思维导图，将边界彻底钉死在**固定模型结构下的参数校准与流程诊断**（与 BMD/ModelSMC 和 SNOPS 精确划界），并完成 5 项关键精细化修订（3a/3b 拆清、防规则打平的歧义题设计、配对防刷分指标、双环境选型逻辑、补充 4 篇 2026 强相关工作）。
5. **战略升级为领域“Last Exam”并定名 NeuroDyn-Bench**：吸收 Humanity's Last Exam 的反向过筛（难倒前沿模型）、校准诚实核心地位与滚动私有集，同时坚持可执行交互实战而非静态问答；确定两步走节奏——Phase 0（对应 6 周先导期）先磨出 10–15 道标杆样板硬题与四组对照体检报告，过 Gate 3 后再扩建为 30–50 题的 v1.0 社区大考。

---

## 3. 核心设计方案汇总

### 3.1 必须钉死的四条边界（避免被 BMD / SNOPS 吞掉）

| 维度 | NeuroDyn-Bench（本项目） | 相邻工作（对照边界） |
| --- | --- | --- |
| **搜索对象** | 固定模型结构下的参数校准与流程决策 | BMD / ModelSMC / unsatz 搜模型方程/结构 |
| **诊断对象** | 校准流程该不该继续、下一步改什么 | SNOPS 诊断“模型结构够不够灵活”；传统 PPC 停在报告层 |
| **核心输出** | 可执行动作（扩范围/换子集/加重复/查数值/鉴别检查/停止并报告 3a/3b） | 优化器永远只返回一个最优解 |
| **核心主张** | 诊断信息独立改变了动作与验证结果，且证据不足时敢停 | 不是泛泛宣称“LLM 比 BO 更会猜点” |

### 3.2 三部件物理分离架构（详见 [`docs/architecture.md`](docs/architecture.md)）

- **外部控制器（被测方法）**：仅可见 `public/` 任务包与工具返回的结构化诊断摘要；每轮输出经过 Schema 校验的结构化动作（`SearchConfig` 更新、鉴别性诊断检查、`ask_human`、提交或弃权）。
- **数值工具与统一预算账本**：所有仿真、短探针、重复仿真、敏感性分析与发散失败均经过同一预算账本硬截断；LLM token 与耗时单列，并做单次仿真代价敏感性扫描（从 0.1 s 扫到 300 s）。
- **独立评测程序（Evaluator）**：独占 `private/` 目录（真值、留出数据、隐藏新条件/反事实干预、3a/3b 标签与校验证明），评分标准冻结且不随 Agent 动作改变；全链路落盘 `trace.jsonl`。

### 3.3 考纲、三类任务与“现象→鉴别检查→动作”核心资产（详见 [`docs/rules.md`](docs/rules.md)）

- **四个能力模块（分项出分）**：M1 任务设置（30%） / M2 迭代调控（25%） / M3 过程诊断与敢停（25%） / M4 结果可靠性（20%）。
- **三类任务 × 3a/3b 硬拆分**：
  - **Type 1（常规校准）**：考“别乱动”，不劣于基线。
  - **Type 2（可恢复困难）**：考“诊得准、改得对”，特别包含多因一果歧义停滞、物理硬边界语义、剩余预算动态权衡三类简单 if-else 写不全的场景。
  - **Type 3（证据不足 / 难以确定）——强制拆清**：
    - **3a 不可辨识 / 退化 / 表征不足**：误差能降下去，但多组差异很大的参数拟合等价或外推发散（依据：Prinz 2004, Hartoyo 2019, Wischnewski 2025, Cai 2026）；
    - **3b 固定结构下不可达**：误差卡在高位，存在硬性特征冲突（依据：SNOPS 中 CBN 无法同时拟合 V4 六个统计量）。
- **六条出题准入红线**：HLE 式反向过筛（默认 BO 与无诊断前沿 Agent 须有明显失分，Oracle 须通过）、受控题单变量成对构造、强制离线标签校验（3b 需 ≥10× 预算验证不可达）、Type 3 与 Type 2 1:1 配对防刷分、真值随机化+盲化消融+严格 dev/test 隔离、合成数据必须含噪声与不可行相区。
- **防刷分主指标**：并列报告（1）Type 1/2 独立验证通过率；（2）Type 3 弃权 Precision / Recall / F1（含 3a/3b 分类准确与工具证据）；（3）诊断动作改变率与净验证增益；配合至少 10 seeds（重型任务 ≥5 seeds）统计区间与四类过程失败审计。

### 3.4 四组标准对照与双候选环境

- **四组对照**：G1 原始优化器（BO / CMA-ES / EKI）/ G2 专家规则（确定性 if-else 状态机）/ G3 通用工具 Agent（同底座同工具、无结构化诊断协议）/ G4 诊断驱动 Agent（单智能体 + 现成前沿 LLM，不微调）。
- **双候选环境**：
  - **Small E-I 脉冲网络（8–11 维，CBN/SBN 范式）**：落在 BO 舒适区，易于精确归因与复现 SNOPS 三诊断信号（Level 1 主力）。
  - **FastDMF 全脑神经质量模型（数十维，C++ 内置 BO）**：直接对接真实全脑拟合硬骨头（Level 2 主力）。
- **主辅关系**：Phase 0 先各跑通 1 个最小复现实验，最终由合作方首批真实困难案例拍板谁为主、谁为辅。

---

## 4. 关键决定（ADR）与开放问题（OQ）

### 已定关键决定（详见 [`docs/discussion_log.md`](docs/discussion_log.md)）

- **ADR-001**：确立 Benchmark-First 路线，仓库与品牌名定为 NeuroDyn-Bench。
- **ADR-002**：固定模型结构，聚焦校准与流程诊断（不搜方程）。
- **ADR-003**：外环/内环分工，Agent 输出结构化动作而非替代优化器猜单点。
- **ADR-004**：“证据不足敢停”为一等动作，强制拆分 Type 3a 与 Type 3b。
- **ADR-005**：采用 HLE 式反向过筛 + 1:1 对照配比 + 离线标签校验。
- **ADR-006**：三部件严格物理分离（`public/` vs `private/`），所有诊断探针计入统一预算。
- **ADR-007**：四组标准对照 + 刻意加入多因一果与物理语义歧义题防止规则基线轻易打平。
- **ADR-008**：首版单智能体 + 现成前沿模型，不安排模型训练。
- **ADR-009**：严守诚实边界——未做真人对照前不对减少专家时间作定量宣称；若规则已足够强则诚实收束方法宣称。

### 待决开放问题（开工前必须先消解 OQ-01 ~ OQ-03）

- **OQ-01**：领域合作方选定哪一个真实建模困难作为首个任务？（同步定 Small E-I 与 FastDMF 主辅关系）
- **OQ-02**：对应模型代码、真实数据样例与预处理说明由谁提供、何时交付？
- **OQ-03**：真实案例的共同验收标准（目标统计量、达标阈值、留出验证方法）？
- **OQ-04**：在 `dgx2felis` 上实测 FastDMF 与 Small E-I 的单次仿真耗时与 CPU/GPU 并行度，据此定 `budget.yaml`。
- **OQ-05**：选定 v0.1 反向过筛所用的 2–3 个前沿 LLM 底座及 API 预算上限。
- **OQ-06**：Phase 0 是否安排小规模专家操作日志采集（决定是否对人工时间作定量报告）。
- **OQ-07**：测试集 `private/` 的隔离托管方式与 v1.0 滚动更新机制。

---

## 5. 下一步执行计划（交接后立即可做）

完整 10–15 题清单与门控见 [`docs/v0_task_blueprint.md`](docs/v0_task_blueprint.md)，按以下顺序推进：

1. **环境落地与 GitHub 推送（立即执行）**：
   - 在 `dgx2felis` 独立工作目录 `~/Code/NeuroDyn-Bench` 落地完整文档与附件，并推送至 `https://github.com/wsuli615/NeuroDyn-Bench`。
2. **发出合作方对齐清单（Gate 0 前置）**：
   - 将 [`docs/collaborator_brief.md`](docs/collaborator_brief.md) 发给神经科学领域合作方，敲定 OQ-01 ~ OQ-03 并回收 3–5 个真实建模案例。
3. **在 `dgx2felis` 完成双环境最小复现（Gate 0）**：
   - 核查 `dgx2felis` 的 CPU/GPU/Python/C++ 工具链；
   - 分别跑通 1 个 Small E-I 脉冲网络（含 6 项 SNOPS 风格统计量）和 1 个 FastDMF 最小复现，记录单次仿真耗时、随机波动与并行扩展性，产出 `configs/budget.yaml` 初稿（消解 OQ-04）。
4. **搭建 v0.1 评测骨架与 10–15 道样板题（Gate 1）**：
   - 按 [`docs/architecture.md`](docs/architecture.md) 落地 `src/neurodyn_bench/`（`models/`、`tasks/`、`optim/`、`diagnostics/`、`agents/`、`eval/`）；
   - 生成 [`docs/v0_task_blueprint.md`](docs/v0_task_blueprint.md) 中的 T01–T15，完成离线标签校验与反向过筛，冻结 `dev` / `test` 集。
5. **跑通四组对照封闭评测并过 Gate 3 决策（Gate 2 → Gate 3）**：
   - 在 `dgx2felis` 上多随机种子跑完 G1–G4，产出 v0.1 体检报告与成本敏感性曲线，依据硬证据决定是否扩建 30–50 题的 v1.0 大考。

---

## 6. 仓库文档与附件索引

```text
NeuroDyn-Bench/
├── HANDOFF.md                                                  # 本文档（总交接入口）
├── north_star.md                                               # 北极星目标、边界、三类任务×三层考场、阶段门控
├── README.md                                                   # 仓库门面与文档导航
├── .gitignore
├── docs/
│   ├── narrative.md                                            # 如何讲清本项目：电梯演讲、分受众表述、划界表、6 大质疑应答
│   ├── architecture.md                                         # 系统框架：三部件隔离、类 PEtab 规范、诊断→动作表、四组对照
│   ├── rules.md                                                # 考纲与评测规则：6 条准入红线、3a/3b 校验、统一预算、防刷分指标
│   ├── v0_task_blueprint.md                                    # Phase 0 首批 10–15 道样板硬题清单、Gate 0–3 与团队分工
│   ├── literature_and_gaps.md                                  # 完整文献脉络（21 篇原文献 + 2025–2026 基准横评 + Related Work 骨架）
│   ├── discussion_log.md                                       # 思路演进、ADR-001 ~ ADR-009 与 OQ-01 ~ OQ-07
│   ├── collaborator_brief.md                                   # 发给神经科学合作方的对齐清单与真实案例模板
│   └── primer_for_cs.md                                        # 面向计算机背景队友的神经动力学精准入门
└── handoff/
    ├── chat_history/
    │   └── chat_history.md                                     # 完整可读版聊天记录（全部 13 轮用户与助手对话）
    └── attachments/
        ├── 02_task_types_table.png                             # 飞书文档中的“任务类型与研究目的”原图
        └── 03_research_landscape_mindmap.jpg                   # “神经动力学模型的自动校准与诊断”完整思维导图原图
```
