# 完整聊天记录（NeuroDyn-Bench 讨论全程）

- **原对话链接**：`https://cursor.com/agents/bc-52ccf2c0-d67f-4850-9265-bdfa94c8b874`
- **会话 ID（`bcId`）**：`bc-52ccf2c0-d67f-4850-9265-bdfa94c8b874`
- **归档时间**：2026-11-21 22:45 GMT (UTC+0)
- **附件图片**：
  - `handoff/attachments/02_task_types_table.png`（任务类型与研究目的表）
  - `handoff/attachments/03_research_landscape_mindmap.jpg`（神经动力学模型的自动校准与诊断思维导图）

---

## Turn 1 — User

这是我们新开的一个合作项目，说说你的思路

### Assistant

（查看初始立项截图与环境，拟定初步研究方案。）

---

## Turn 2 — User

说中文，我觉得要做一个好的agent 首先得有好的benchmark来测试，那么有哪些合适的benchmark呢？分别是怎么出题的？现在的benchmark有什么不足呢？还是说根本就没有合适的benchmark?

### Assistant

现有的 benchmark 都不能直接拿来测这个 Agent。相关的不少，光 2026 年就新出了好几个，但每个只覆盖一部分。立项书最核心的三件事，没有哪个 benchmark 同时考：

- 任务设置：调哪些参数、范围多大、匹配哪些观测。
- 拟合失败时，分清是“参数没调好”还是“机制有问题”。
- 人工介入次数。

网络和群体层面的题（E/I 平衡、突触时间常数）也几乎是空白。所以需要自建，但出题方法和题源可以大量借鉴。

#### 现有 benchmark，分四类

**一、神经元参数优化：领域最贴近，但只比优化器**
- **Neuroptimus（PLoS Comput Biol 2024）**：6 道单细胞题（HH 到详细 CA1 锥体细胞，3–12 参），20 多种算法各跑 10 次、每次 1 万次评估；2 道孪生实验 + 其余真实记录或复杂模型；CMA-ES 和粒子群最稳。
- **sbibm（AISTATS 2021）**：10 个玩具任务，在 $10^3$ 到 $10^5$ 预算下比后验精度。

**二、系统生物学的参数估计：方法最成熟**
- **Benchmark-Models-PEtab**：20 个 ODE 模型配真实数据；其 PEtab 参数表/条件表/观测函数格式可直接借鉴。

**三、LLM Agent 校准仿真器：2026 年新出现，形式最接近**
- **TuneBench（电力系统）**：Agent 15 次仿真 vs GA 150 次；故意偏离默认参数防背题。
- **HydroAgent（水文模型）**：9 个前沿 LLM 几乎未达人类专家水平；三类失败：过早停止、提出越界参数、诊断对了却改错了参数。
- **SimulCost（ICML 2026）**：解析成本定义；LLM 多轮成功率 66–81%，但比传统扫描慢 1.5–2.7 倍。
- **物理描述 prompt 消融（arXiv:2610.22379）**：物理知识只在前几次仿真有用（5 次低 21.9 倍，15 次无差别），换种子中位数差 9.8 倍。
- **BBOWP-Bench**：唯一专门考“任务设置”的基准（易/中/难三档描述）。

**四、机制发现与模型误设：对应“诊断”部分**
- **NeuronBench（Kevin Murphy, 2026）**：6 种神秘神经元，考主动实验设计与干预预测。
- **ModelSMC（Macke 组, ICML 2026）**：由 LLM 提候选模型；可借鉴“删掉已知模型机制看能否发现”。
- **unsatz**：Agent 写模型结构，拟合器固定。
- **CtDB**：提醒合成数据要加噪声与试次间变异。

#### 共同的不足
1. 任务设置都是给定的；
2. 没有带标签的“拟合不了”的题；
3. 不计人工介入；
4. 成本口径不统一（只数仿真次数，不算 LLM token 与耗时）；
5. 网络和群体层面是空白；
6. 背答案与统计方差问题。

#### 我们怎么出题
核心是孪生实验 + 受控注入问题：标准题、设置陷阱题（三档信息 + 单个陷阱）、机制失配题（大预算离线验证 + 1:1 可达对照题）、欠约束题、真实数据题（第二阶段）。确立“先 benchmark、后 Agent”的路线。

---

## Turn 3 — User

那么你觉得 我们该怎么选题 或者说考纲该怎么定？

### Assistant

核心建议：考纲从研究员调参时要做的决定与真实失败事故倒推，模型只是载体。

- **四模块考纲草案**：设置（30%）/ 迭代（25%）/ 诊断（25%）/ 可靠性（20%），外加贯穿全题的仿真次数、统一成本、人工介入次数；分项出分。
- **五条出题原则**：一题只埋一个坑且成对出；按默认配置 vs 神谕配置区分度筛题；诊断题与对照题 1:1；约 10% 题目考“该问才问”；每个考点至少跨两个模型，保留锚题。
- **两种题目形态**：小题（卡壳快照决策，离线跑完所有候选动作自动定标）+ 大题（完整多轮交互调参）。
- **明确不考**：不考从零写方程/发现结构、不考主动设计新实验、不考写仿真器代码。

---

## Turn 4 — User

团队里还有几个计算机背景的，你需要精准地讲解一下神经动力学是什么 做什么

### Assistant

从计算机视角给出神经动力学系统入门：带物理意义与硬约束的随机模拟器及其病态反问题，覆盖单神经元（HH/LIF/AdEx）、脉冲网络（E/I 平衡、Dale 定律、Brunel 相图）、群体与全脑（Wilson–Cowan、神经质量模型、全脑连接组），解释状态空间/不动点/极限环/吸引子/混沌/分岔与六大难调根源（跨数量级、大片无效区、分岔不连续、随机性、不可微与部分观测、简并/不可辨识、模型缺机制），并附常用术语表。（详见 `docs/primer_for_cs.md`）

---

## Turn 5 — User

给这个repo取个名字 你觉得叫什么合适？

### Assistant

核查 GitHub、arXiv、PyPI 命名冲突后（排除已占用的 NeuroTune/NeuronBench/NeuroBench/TuneBench 等），推荐 `neurodyn-calib` / `NeuroDyn-Bench` / `neurocalib` / `calibneuro` / `ndcalib`。

---

## Turn 6 — User

你能看到这个页面吗：https://v4czwcgckj.feishu.cn/wiki/DG3CwuX5PiiS0KkRqjCcFn3Bnqg

### Assistant

无法访问需登录的飞书链接，建议粘贴正文与图表。

---

## Turn 7 — User

粘贴飞书正式立项正文（一、研究背景与目标至八、预期成果与近期讨论重点）、两张图（`02_task_types_table.png`、`03_research_landscape_mindmap.jpg`）及 `相关工作研究脉络.docx`、`总结.docx`。

### Assistant

完整审阅后提出 5 项关键精细化修订：
1. **Type 3 强制拆分为 3a（不可辨识/退化/表征不足）与 3b（固定结构下不可达）**；
2. **防“专家规则 if-else 直接打平 LLM Agent”**：刻意设计多因一果歧义停滞、依赖物理硬边界语义的决策、残余预算动态权衡；
3. **主指标防刷分（不肯停 vs 乱停）**：Type 2/3 配对测试，并列报告验证通过率 + 弃权 Precision/Recall + 诊断动作改变率与净验证增益；
4. **FastDMF vs 小型 E-I 网络主辅选型**：优先由合作方真实案例拍板；若尚未定死，小型 E-I（8–11 维，BO 舒适区）做受控归因主环境，FastDMF 做高维硬骨头/跨尺度第二环境；
5. **补充 4 篇 2026 年强相关工作**：HydroAgent (`2605.17792`)、SimulCost (ICML 2026)、Physics-labeled prompting (`2610.22379`)、ModelSMC / NeuronBench (`2608.09696`)。

---

## Turn 8 — User

我觉得可以大胆一点，ai这么强，做成 像last exam一样的bench会不会更好一些

### Assistant

高度赞同转向 **Benchmark-First 领域极限大考**，明确对 Humanity's Last Exam（HLE）的“三学两不学”：
- **三学**：反向过筛机制（难倒前沿通用 Agent 与开箱优化器）、把“诚实校准/敢停”推到头版、滚动隐藏集与社区征题；
- **两不学**：不做静态问答而做可执行交互实战大考、不拼千题规模而拼 30–50 道跨尺度硬题深度（Phase 0 先磨 10–15 道标杆样板）。
- 将三类任务升级为 **Level 1（受控诊断） / Level 2（前沿真实硬骨头） / Level 3（科学诚实与反事实）** 三层考场。

---

## Turn 9 — User

先定框架 规则和目标，把我们的目标north_star.md，我们的讨论，目前的思路全都捋清楚，思考现在怎么说清这个项目比较合适，推送https://github.com/wsuli615/NeuroDyn-Bench

### Assistant

定名 **NeuroDyn-Bench**，整理落盘 `north_star.md`、`README.md` 与 `docs/` 全套 8 份核心设计文档（叙事、架构、规则、Phase 0 题单蓝图、文献横评、讨论与 ADR、合作方清单、CS 入门），并说明云端 GitHub App 权限限制。

---

## Turn 10 — User

你要在dgx-f里做这个项目

### Assistant

说明云端 VM 会话无法直接跨机操作 `dgx2felis`，建议在 `dgx2felis` 独立创建 `~/Code/NeuroDyn-Bench` 目录接力。

---

## Turn 11 — User

好了，你单独开目录吧

### Assistant

再次确认权限边界，给出在 `dgx2felis` 另开会话接力的两种同步方式。

---

## Turn 12 — User

我们对话的位置在哪里，输出来

### Assistant

输出对话 ID `bc-52ccf2c0-d67f-4850-9265-bdfa94c8b874`、云端路径 `/workspace` 及全部 11 份文件清单。

---

## Turn 13 — User

整理 `HANDOFF.md`、完整聊天记录与附件并交接至 `dgx2felis` (`~/Code/NeuroDyn-Bench`)。
