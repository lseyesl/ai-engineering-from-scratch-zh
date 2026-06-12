# Anthropic 负责任的扩展政策 v3.0 (Anthropic Responsible Scaling Policy v3.0)

> RSP v3.0 于 2026 年 2 月 24 日生效，取代了 2023 年的政策。采用两级缓解措施：Anthropic 单方面将采取的行动 vs 作为行业范围建议提出的行动（包括 RAND SL-4 安全标准）。新增前沿安全路线图（Frontier Safety Roadmaps）和风险报告（Risk Reports）作为常设文件，而非一次性交付物。取消了 2023 年的暂停承诺。引入了 AI R&D-4 阈值：一旦跨越，Anthropic 必须发布一份肯定性案例（affirmative case），识别错位风险及缓解措施。Claude Opus 4.6 未跨越该阈值。Anthropic 在 v3.0 公告中指出"自信地排除这种可能性正变得越来越困难"。SaferAI 将 2023 年 RSP 评为 2.2 分；他们将 v3.0 降级为 1.9 分，使 Anthropic 与 OpenAI 和 DeepMind 一同被归入"薄弱"RSP 类别。定性阈值取代了 2023 年的定量承诺；取消暂停条款是最严重的倒退。

**类型：** 学习 (Learn)
**语言：** Python（标准库，RSP 阈值决策引擎）
**前置要求：** 阶段 15 · 06（AAR），阶段 15 · 07（RSI）
**时间：** ~45 分钟

## 问题 (The Problem)

前沿实验室发布的扩展政策，部分是技术文档，部分是治理文档，部分是对监管机构的信号。RSP v3.0 是 Anthropic 当前的文档。仔细阅读它很重要，不是因为遵守它是强制性的（它不是），而是因为其框架塑造了实验室如何看待灾难性风险，以及他们如何向公众传达权衡取舍。

v3.0 与 v2.0 的差异是有用的分析单位。新增了什么：前沿安全路线图、风险报告、AI R&D-4 阈值。移除了什么：2023 年的暂停承诺。重新框架了什么：两级缓解计划，分为 Anthropic 单方面行动和行业建议。外部评审——SaferAI——将评分从 2.2（v2）降级为 1.9（v3.0）。这就是一个扩展政策如何在看起来更精致的同时变得不那么严谨。

## 概念 (The Concept)

### 两级缓解计划 (The two-tier mitigation schedule)

- **Anthropic 单方面行动 (Anthropic unilateral actions)**：无论其他实验室做什么，Anthropic 都将采取的行动。超过某个阈值停止训练、特定的安全措施、特定的部署关卡。
- **行业范围建议 (Industry-wide recommendations)**：Anthropic 认为行业应集体采取的行动。包括 RAND SL-4 安全标准。这些不是 Anthropic 方面的承诺；它们是政策倡导。

两级结构在 v2 中不存在。这意味着读者需要查看每个承诺位于哪一列。位于"行业范围建议"列的安全措施不是 Anthropic 的承诺；而是 Anthropic 的希望。

### AI R&D-4 阈值 (The AI R&D-4 threshold)

这是 RSP v3.0 指出的下一个重要能力水平阈值。具体来说：一个能够以有竞争力的成本自动化相当大一部分 AI 研究的模型。一旦 Anthropic 认为某个模型跨越了该阈值，他们必须在继续扩展之前发布一份肯定性案例，识别错位风险和缓解措施。

根据 v3.0 公告，Claude Opus 4.6 未跨越该阈值。文档补充道："自信地排除这种可能性正变得越来越困难。"这种措辞很重要；它承认该阈值已经足够接近，是一个现实关切，而非推测性的极限。

第 6 课（自动化对齐研究）和第 7 课（递归自我改进）直接关系到这个阈值。跨越研究质量门槛的自动化对齐研究人员是 AI R&D-4 阈值正在逼近的证据。

### 前沿安全路线图和风险报告 (Frontier Safety Roadmaps and Risk Reports)

v3.0 将两种工件类型提升为常设文件：

- **前沿安全路线图 (Frontier Safety Roadmap)**：前瞻性文档，描述计划中的安全工作、能力预期和缓解研究。
- **风险报告 (Risk Report)**：发布后针对特定模型的回顾性文档，描述观察到的能力和残余风险。

两者都是公开的。两者都按声明的节奏更新。其价值在于：读者可以追踪 Anthropic 在路线图中说他们将要做的事情与他们在风险报告中报告的内容之间的对比。

### 取消暂停承诺 (Removing the pause clause)

2023 年的 RSP 包含一个明确的暂停承诺：如果模型跨越了特定的能力阈值，训练将暂停，直到缓解措施到位。v3.0 用一个更软性的表述（发布肯定性案例，如果缓解措施充分则继续）取代了明确的暂停。SaferAI 和其他分析人士直接指出这是新文档中最严重的倒退。

支持这一变化的政策论据：2023 年的定量阈值最终被证明是 2026 年时代的能力基准无法达到的，因为基准本身被重新标定了。反对的论据：扩展政策中的暂停条款是一种承诺机制；移除它就移除了政策的可信度。

### SaferAI 的降级 (SaferAI's downgrade)

SaferAI 是一个独立的组织，负责对 RSP 风格的文件进行评级。他们的公开评级：2023 年 Anthropic RSP 得分为 2.2（在 4.0 为当前最佳 RSP、1.0 为名义上的评分标准中）。v3.0 得分为 1.9。这将 Anthropic 从"中等"移到了"薄弱"，与 OpenAI 和 DeepMind 一起被归入薄弱类别。

根据 SaferAI 的降级因素：
- 定性阈值取代了定量阈值。
- 暂停承诺被移除。
- AI R&D-4 阈值的缓解措施被描述为"肯定性案例"而非具体措施。
- 审查机制依赖于 Anthropic 的安全咨询小组（Safety Advisory Group），独立监督有限。

### 本课不是什么 (What this lesson is not)

这不是一堂关于合规的课。RSP v3.0 不是法规；没有任何东西强迫 Anthropic 遵守它。本课是关于以应有的具体性和怀疑态度阅读该文档。扩展政策是前沿实验室就灾难性风险姿态发出的主要公开信号。善于阅读这些政策是一项实用技能，适用于任何工作依赖于前沿能力的人。

## 使用它 (Use It)

`code/main.py` 实现了一个小型决策引擎，它镜像了 RSP 阈值评估的形状：给定一个候选模型和一组能力测量值，返回 AI R&D-4 阈值是否被跨越、所需的肯定性案例部分以及部署是否可以继续。它故意保持简单；重点在于使文档的逻辑变得明确。

## 交付它 (Ship It)

`outputs/skill-scaling-policy-review.md` 以 v3.0 为参考，审查扩展政策（Anthropic、OpenAI、DeepMind 或内部政策）：两级结构、阈值、暂停承诺、独立审查。

## 练习 (Exercises)

1. 运行 `code/main.py`。输入三个不同能力水平的合成模型。确认阈值评估器的行为符合预期，并生成正确的肯定性案例模板。

2. 完整阅读 RSP v3.0（32 页）。找出所有位于"行业范围建议"层级中的承诺。其中哪些承诺在 v2 中本应是"Anthropic 单方面"的？

3. 阅读 SaferAI 的 RSP 评级方法。应用他们的评分标准到 v3.0 文档，重现他们的 1.9 分。哪个评分项对降级影响最大？

4. 2023 年的暂停承诺被移除了。提出一个替代承诺，既能保持政策的可信度，又能承认 2026 年基准重新标定问题。

5. 比较 RSP v3.0 与 OpenAI 准备框架 v2（第 20 课）。找出一个 v3.0 更强的领域。再找出一个准备框架更强的领域。

## 关键术语 (Key Terms)

| 术语 (Term) | 人们说的 (What people say) | 实际含义 (What it actually means) |
|---|---|---|
| RSP | "Anthropic 的扩展政策" | 负责任的扩展政策；v3.0 于 2026 年 2 月 24 日生效 |
| AI R&D-4 | "研究自动化阈值" | 以有竞争力的成本自动化大量 AI 研究的能力 |
| 肯定性案例 (Affirmative case) | "安全理由" | 已发布的论证，说明风险已被识别且缓解措施充分 |
| 前沿安全路线图 (Frontier Safety Roadmap) | "前瞻计划" | 关于计划中的安全工作和预期能力的常设文件 |
| 风险报告 (Risk Report) | "模型回顾" | 发布后关于观察到的能力和残余风险的常设文件 |
| 两级缓解 (Two-tier mitigation) | "单方面 vs 行业" | Anthropic 承诺与行业建议，分开列出 |
| 暂停承诺 (Pause commitment) | "2023 年条款" | 明确承诺暂停训练；在 v3.0 中被移除 |
| SaferAI 评级 (SaferAI rating) | "独立 RSP 评分" | 第三方评分标准；v3.0 得分为 1.9（v2 为 2.2） |

## 延伸阅读 (Further Reading)

- [Anthropic — Responsible Scaling Policy v3.0](https://anthropic.com/responsible-scaling-policy/rsp-v3-0) — 完整的 32 页政策。
- [Anthropic — RSP v3.0 announcement](https://www.anthropic.com/news/responsible-scaling-policy-v3) — 与 v2 的变更摘要。
- [Anthropic — Frontier Safety Roadmap](https://www.anthropic.com/research/frontier-safety) — 从 RSP v3.0 链接的常设文件。
- [Anthropic — Risk Report: Claude Opus 4.6](https://www.anthropic.com/research/risk-report-claude-opus-4-6) — 对当前前沿模型的回顾。
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — 将 AI R&D-4 与测量的自主性联系起来。