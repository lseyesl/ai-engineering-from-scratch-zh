# LLM 功能的 A/B 测试——GrowthBook、Statsig 与"凭感觉"问题

> 传统的 A/B 测试并非为具有非确定性的 LLM 而设计。关键区别：评估（evals）回答"模型能完成这个任务吗？"A/B 测试回答"用户在意吗？"两者都是必需的；仅凭感觉就上线已经过时了。2026 年需要测试的内容：提示工程（措辞）、模型选择（GPT-4 vs GPT-3.5 vs 开源；准确率 vs 成本 vs 延迟）、生成参数（temperature、top-p）。真实案例：聊天机器人奖励模型变体实现了 +70% 对话长度和 +30% 留存率；Nextdoor AI 主题行实验在奖励函数优化后实现了 +1% CTR；Khan Academy 的 Khanmigo 在延迟与数学准确率之间进行迭代优化。平台格局：**Statsig**（2025 年 9 月被 OpenAI 以 $1.1B 收购）——序贯检验（sequential testing）、CUPED、一站式方案。**GrowthBook**——开源、数据仓库原生、贝叶斯 + 频率派 + 序贯引擎、CUPED、SRM 检查、Benjamini-Hochberg + Bonferroni 校正。你基于对仓库 SQL 的偏好以及"被 OpenAI 收购"对你的组织是否重要来选择。

**Type:** Learn
**Languages:** Python（stdlib，简易序贯检验模拟器）
**Prerequisites:** Phase 17 · 13 (Observability)、Phase 17 · 20 (Progressive Deployment)
**Time:** ~60 分钟

## 学习目标

- 区分评估（"模型能否完成任务"）与 A/B 测试（"用户是否在意"）。
- 列举三个可测试的轴（提示、模型、参数）并为每个轴选择指标。
- 解释 CUPED、序贯检验和 Benjamini-Hochberg 多重比较校正。
- 基于仓库 SQL 偏好和公司对收购的态度选择 Statsig 或 GrowthBook。

## 问题

你手动调优了一个系统提示。感觉更好了。你发布出去。转化率因噪声而变化。你责怪指标。或者你发布了一个新模型，转化率没有变化——是模型退化了还是变化太小无法检测？你不知道，因为你没有做 A/B 测试就发布了。

评估回答模型是否能在标记集上完成任务。它们不回答用户是否更喜欢输出。只有受控的在线实验才能回答这个问题，而且只有当实验有足够的统计功效、控制了非确定性并校正了多重比较时才行。

## 概念

### 评估 vs A/B 测试

**评估**——离线、标记集、裁判（评分标准或 LLM-as-judge 或人工）。回答："在此固定分布上输出是否正确/有用/安全？"

**A/B 测试**——在线、真实用户、随机化。回答："新变体是否改变了重要的用户级指标？"

两者都是必需的。评估在暴露之前发现回归；A/B 在之后确认产品影响。

### 测试什么

1. **提示工程**——措辞、系统提示结构、示例。指标：任务成功率、用户留存、每请求成本。
2. **模型选择**——GPT-4 vs GPT-3.5-Turbo vs Llama-OSS。指标：准确率（任务）+ 每请求成本 + P99 延迟。多目标。
3. **生成参数**——temperature、top-p、max_tokens。指标：任务特定（输出多样性 vs 确定性）。

### CUPED——方差缩减

利用实验前数据控制实验后比较前的周期前方差。典型方差缩减：30-70%。有效样本量免费增加。

实现：Statsig 和 GrowthBook 都实现了。

### 序贯检验

经典 A/B 假设固定样本量。序贯检验（"边看边决定"）在反复查看时控制假阳性率。始终有效的序贯程序（mSPRT、Howard 的置信序列）允许在明确胜出时提前停止。

### 多重比较校正

在 95% 置信水平下运行 20 个 A/B 测试，按概率会产生一个假阳性。Bonferroni 校正每个测试的 α；Benjamini-Hochberg 控制错误发现率。GrowthBook 实现了两者。

### SRM——样本比例失配

分配哈希将用户随机分配到变体。如果 50/50 分派变成了 47/53，说明出了问题——SRM 检查会标记它。两个平台都实现了。

### Statsig vs GrowthBook

**Statsig**：
- 2025 年 9 月被 OpenAI 以 $1.1B 收购。托管，SaaS。
- 序贯检验、CUPED、保留人群。
- 一站式：功能标志 + 实验 + 可观测性。
- 最适合：团队想要捆绑产品，不关心 OpenAI 所有权。

**GrowthBook**：
- 开源（MIT）；数据仓库原生（直接从 Snowflake/BigQuery/Redshift 读取）。
- 多引擎：贝叶斯、频率派、序贯。
- CUPED、SRM、Bonferroni、BH 校正。
- 自托管或托管云。
- 最适合：数据仓库 SQL 团队，数据团队控制指标层，想要开源。

### 非确定性使统计功效复杂化

相同的提示产生不同的输出。传统功效计算假设 IID 观测。在 LLM 非确定性下，有效样本量低于名义值。将所需样本量乘以约 1.3-1.5 倍作为安全余量。

### 真实案例结果

- 聊天机器人奖励模型变体：+70% 对话长度，+30% 留存率。
- Nextdoor 主题行：奖励函数优化后 +1% CTR。
- Khan Academy Khanmigo：迭代式延迟与数学准确率权衡。

### 反模式：凭感觉上线

每个资深工程师都能说出一个因为"感觉更好"而未经 A/B 测试就上线的功能。其中大多数在数月内退化了产品指标而团队未察觉。A/B 测试是一种强制机制。

### 你应该记住的数字

- Statsig 被 OpenAI 收购：$1.1B，2025 年 9 月。
- GrowthBook：开源 MIT；贝叶斯 + 频率派 + 序贯。
- CUPED 方差缩减：30-70%。
- LLM 非确定性 → +30-50% 样本量缓冲。

## 使用它

`code/main.py` 模拟一个带有固定边界和序贯边界的序贯 A/B 测试。展示序贯方法如何让你提前停止。

## 交付物

本课程产出 `outputs/skill-ab-plan.md`。给定功能变更、工作负载、基线，选择平台、门控、样本量。

## 练习

1. 运行 `code/main.py`。对于预期提升 5%、基线转化率 3%，要达到 80% 统计功效需要多少样本量？
2. 为受医疗监管的本地客户选择 Statsig 或 GrowthBook。
3. 设计一个测试 GPT-4 vs GPT-3.5 在每解决工单成本上的 A/B 测试。主要指标、护栏指标、次要指标是什么？
4. 你的金丝雀通过，但 A/B 显示转化率 -1.2%。你会发布吗？写出升级标准。
5. 对具有 60% 前后方差的实验期应用 CUPED。计算有效样本量的提升。

## 关键术语

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Eval | "离线测试" | 模型能力的标记集评估 |
| A/B test | "实验" | 对用户进行的实时随机比较 |
| CUPED | "方差缩减" | 利用实验前数据回归减少方差 |
| Sequential test | "可窥探的测试" | 允许提前停止的始终有效程序 |
| Multiple comparison | "族系错误" | 运行多个测试会增加假阳性 |
| Bonferroni | "严格校正" | 将 α 除以测试数 |
| Benjamini-Hochberg | "BH FDR" | 错误发现率控制，较不保守 |
| SRM | "错误分派" | 样本比例失配；分配错误 |
| Statsig | "OpenAI 旗下" | 商业一站式平台，2025 年被收购 |
| GrowthBook | "那个开源的" | MIT 数据仓库原生平台 |
| mSPRT | "序贯概率比检验" | 经典序贯程序 |

## 延伸阅读

- [GrowthBook — How to A/B Test AI](https://blog.growthbook.io/how-to-a-b-test-ai-a-practical-guide/)
- [Statsig — Beyond Prompts: Data-Driven LLM Optimization](https://www.statsig.com/blog/llm-optimization-online-experimentation)
- [Statsig vs GrowthBook comparison](https://www.statsig.com/perspectives/ab-testing-feature-flags-comparison-tools)
- [Deng et al. — CUPED](https://www.exp-platform.com/Documents/2013-02-CUPED-ImprovingSensitivityOfControlledExperiments.pdf)
- [Howard — Confidence Sequences](https://arxiv.org/abs/1810.08240)
