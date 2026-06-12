# 对齐研究生态系统——MATS、Redwood、Apollo、METR

> 五个组织定义了 2026 年实验室外的对齐研究层。MATS（ML Alignment & Theory Scholars）：自 2021 年底以来 527+ 位研究人员，180+ 篇论文，10K+ 引用，h-index 47；2024 年夏季批正式注册为 501(c)(3)，约 90 名学者和 40 名导师；80% 的 2025 年前校友从事安全/安全工作，200+ 人分布在 Anthropic、DeepMind、OpenAI、UK AISI、RAND、Redwood、METR、Apollo。Redwood Research：由 Buck Shlegeris 创立的���用对齐实验室；引入了 AI 控制（第 10 课）；与 UK AISI 在控制安全案例上合作。Apollo Research：为前沿实验室进行部署前策划评估；撰写了上下文策划（第 8 课）和 Towards Safety Cases for AI Scheming。METR（Model Evaluation and Threat Research）：基于任务的能力评估、自主任务时间范围研究；"Common Elements of Frontier AI Safety Policies"比较实验室框架。Eleos AI Research：模型福利部署前评估（第 19 课）；进行了 Claude Opus 4 福利评估。

**Type:** Learn
**Languages:** none
**Prerequisites:** Phase 18 · 01-27（Phase 18 前期课程）
**Time:** ~45 分钟

## 学习目标

- 识别非实验室对齐研究生态系统的五个组织及其核心产出。
- 描述 MATS 的规模（学者、论文、h-index）及其作为人才管道的作用。
- 描述 Redwood 的 AI 控制议程及其与 UK AISI 的合作。
- 描述 METR 的基于任务评估方法。

## 问题

前沿实验室（第 18 课）内部进行安全评估并选择性地发布结果。实验室外的生态系统是评估得到验证、新型失败模式被首先发现、以及人才得到培训的地方。理解生态系统有助于解释哪些研究发现被谁信任。

## 概念

### MATS（ML Alignment & Theory Scholars）

2021 年底开始。研究指导计划；学者花费 10-12 周与一位资深研究人员合作解决特定的对齐问题。

规模（2026）：
- 自成立以来 527+ 位研究人员。
- 发表 180+ 篇论文。
- 10K+ 引用。
- h-index 47。
- 2024 年夏季：90 位学者 + 40 位导师；注册为 501(c)(3)。

职业成果：约 80% 的 2025 年前校友从事安全/安全工作。200+ 人分布在 Anthropic、DeepMind、OpenAI、UK AISI、RAND、Redwood、METR、Apollo。

### Redwood Research

应用对齐实验室。由 Buck Shlegeris 创立。引入了 AI 控制议程（第 10 课）。与 UK AISI 在控制安全案例上合作。为 DeepMind 和 Anthropic 的评估设计提供建议。

经典论文：Greenblatt、Shlegeris 等人，"AI Control"（arXiv:2312.06942，ICML 2024）；对齐伪装（Greenblatt、Denison、Wright 等人，arXiv:2412.14093，与 Anthropic 合作）。

风格：特定威胁模型、最坏情况对手、可进行压力测试的具体协议。

### Apollo Research

为前沿实验室进行部署前策划评估。撰写了上下文策划（第 8 课，arXiv:2412.04984）。与 2025 年 OpenAI 反策划训练合作。产出 Towards Safety Cases for AI Scheming（2024）。

风格：欺骗可能涌现的代理环境评估；三支柱分解（错误对齐、目标导向性、情境意识）。

### METR（Model Evaluation and Threat Research）

基于任务的能力评估。自主任务完成时间范围研究。"Common Elements of Frontier AI Safety Policies"（metr.org/common-elements，2025）比较实验室框架。

与 Apollo 共同撰写了 AI 策划安全案例草图。

风格：长时间���任务评估、实证能力测量、框架综合。

### Eleos AI Research

模型福利部署前评估。进行了系统卡第 5.3 节记录的 Claude Opus 4 福利评估。为第 19 课的福利相关声明提供外部方法论检查。

### 流程

MATS 培训研究人员。毕业生进入 Anthropic、DeepMind、OpenAI（实验室安全团队）或 Redwood、Apollo、METR、Eleos（外部评估）。外部评估者与实验室以及 UK AISI / CAISI 合作。出版物反馈到生态系统，为下一批 MATS 学员服务。

### 为什么这一层很重要

单一来源的评估不可靠：实验室评估自身模型存在结构性利益冲突。外部评估者可以提出和验证实验室可能低估的失败模式。2024 年的休眠代理论文（第 7 课）是 Anthropic + Redwood；对齐伪装是 Anthropic + Redwood；上下文策划是 Apollo；反策划是 Apollo + OpenAI。多组织结构就是质量控制。

### 在 Phase 18 中的位置

第 7-11 课引用了 Redwood 和 Apollo 的工作；第 18 课引用了 METR 的框架比较；第 19 课引用了 Eleos。第 28 课是对 Phase 其余部分所依赖的生态系统的明确组织映射。

## 使用它

无代码。阅读 METR 的"Common Elements of Frontier AI Safety Policies"，作为外部综合如何为实验室内部政策工作增值的示例。

## 交付物

本课程产出 `outputs/skill-ecosystem-map.md`。给定一个对齐声明或评估，它会识别组织、发表场所和方法论风格，并对照已知的对等组织进行交叉核对。

## 练习

1. 从第 7-15 课中选择一篇论文，确定涉及的组织。将作者与 MATS 校友和当前生态系统关联进行交叉核对。
2. 阅读 METR 的"Common Elements of Frontier AI Safety Policies。"确定他们强调的三个实验室间趋同点和两个最大分歧点。
3. MATS 的职业成果约 80% 在安全/安全领域。论证这种选择压力是适应性的（培训了领域）还是有偏见的（过滤了异端立场）。
4. Redwood 和 Apollo 都从事控制/策划工作但风格不同。选择一个失败模式，描述每个组织会如何调查它。
5. Eleos AI 是唯一纯模型福利的组织。设计一个专注于不同福利相关问题（认知自由、机器人具身等）的假设性第二组织，并阐明其方法。

## 关键术语

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| MATS | "指导计划" | ML Alignment & Theory Scholars；自 2021 年以来 527+ 位研究人员 |
| Redwood Research | "控制实验室" | 应用对齐；AI 控制作者；UK AISI 合作伙伴 |
| Apollo Research | "策划评估" | 为前沿实验室进行部署前策划评估 |
| METR | "任务时间范围评估" | 基于任务的能力评估；框架综合 |
| Eleos AI | "福利实验室" | 模型福利部署前评估 |
| Talent pipeline | "MATS -> 实验室" | MATS 毕业生流向 Anthropic、DM、OpenAI、Redwood、Apollo、METR |
| External evaluation | "非实验室检查" | 非模型生产者的评估；增加可信度 |

## 延伸阅读

- [MATS (ML Alignment & Theory Scholars)](https://www.matsprogram.org/)——指导计划
- [Redwood Research](https://www.redwoodresearch.org/)——AI 控制论文
- [Apollo Research](https://www.apolloresearch.ai/)——策划评估
- [METR — Common Elements of Frontier AI Safety Policies](https://metr.org/blog/2025-03-26-common-elements-of-frontier-ai-safety-policies/)——框架比较
- [Eleos AI Research](https://www.eleosai.org/research)——模型福利方法论
