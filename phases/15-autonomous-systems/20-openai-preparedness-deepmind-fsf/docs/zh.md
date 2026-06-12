# OpenAI 准备框架与 DeepMind 前沿安全框架 (OpenAI Preparedness Framework and DeepMind Frontier Safety Framework)

> OpenAI 准备框架 v2（Preparedness Framework v2，2025 年 4 月）引入了研究类别（Research Categories）——长程自主性（Long-range Autonomy）、消极应付（Sandbagging）、自主复制与适应（Autonomous Replication and Adaptation）、破坏安全措施（Undermining Safeguards）——与跟踪类别（Tracked Categories）区分开来。跟踪类别会触发能力报告（Capabilities Reports）和安全措施报告（Safeguards Reports），由安全咨询小组（Safety Advisory Group）审查。DeepMind 的 FSF v3（2025 年 9 月，跟踪能力级别于 2026 年 4 月 17 日新增）将自主性纳入 ML R&D 和网络（Cyber）领域（ML R&D 自主性级别 1 = 以与人类 + AI 工具有竞争力的成本完全自动化 AI R&D 流程）。FSF v3 通过自动化监控工具性推理（instrumental-reasoning）滥用，明确处理欺骗性对齐（deceptive alignment）。诚实的说明：PF v2 中的研究类别（包括长程自主性）不会自动触发缓解措施；政策语言是"潜在的"。DeepMind 自己也表示，如果工具性推理增强，"自动化监控长期来看将不再足够"。

**类型：** 学习 (Learn)
**语言：** Python（标准库，三框架决策表差异比较工具）
**前置要求：** 阶段 15 · 19（Anthropic RSP）
**时间：** ~45 分钟

## 问题 (The Problem)

第 19 课仔细阅读了 Anthropic 的扩展政策。本课通过阅读 OpenAI 和 DeepMind 的政策来完善这一图景。这三份文档是同类产物，回答同一个问题——前沿实验室何时应暂停或关卡一个模型——它们在一小套类别上趋于一致，并在特定的重要地方存在分歧。

趋同之处：三者都将长程自主性标记为值得跟踪的能力类别。三者都承认欺骗性行为（对齐造假、消极应付）是一类特定的风险。三者都有内部审查机构。分歧之处：OpenAI 将类别分为"跟踪"（强制缓解）和"研究"（无自动触发）。DeepMind 将自主性纳入两个领域而非单独命名。各实验室将能力归入跟踪 vs 研究、或关键 vs 中等、或一级 vs 二级；能力位于哪个桶中的操作后果在各实验室间是不同的。

将它们放在一起阅读是有用的练习。同一种能力在 Anthropic 可能是"强制缓解"，在 OpenAI 是"监控但不触发"，在 DeepMind 是"在特定领域中跟踪"。政策姿态很重要。

## 概念 (The Concept)

### OpenAI 准备框架 v2（2025 年 4 月）(OpenAI Preparedness Framework v2 (April 2025))

结构：

- **跟踪类别 (Tracked Categories)**：触发能力报告（模型能做什么）和安全措施报告（已有哪些缓解措施）。部署前由安全咨询小组审查。
- **研究类别 (Research Categories)**：实验室正在跟踪但尚未承诺具体缓解措施的模型能力。包括长程自主性、消极应付、自主复制与适应、破坏安全措施。

v2 中的关键框架：研究类别不会自动触发缓解措施。政策语言是"潜在的"缓解措施。这是需要关注的具体差异：一个类别位于跟踪还是研究，是"关卡"和"观察"之间的分水岭。

### DeepMind 前沿安全框架 v3（2025 年 9 月；跟踪能力级别于 2026 年 4 月 17 日新增）(DeepMind Frontier Safety Framework v3 (September 2025; Tracked Capability Levels added April 17, 2026))

结构：

- **关键能力级别 (Critical Capability Levels, CCLs)**：跨五个领域的能力阈值：网络、生物、ML R&D、CBRN、自主性（纳入 ML R&D 和网络）。
- **跟踪能力级别 (Tracked Capability Levels)**：2026 年 4 月新增的额外粒度。具体示例：ML R&D 自主性级别 1 = 以与人类 + AI 工具有竞争力的成本完全自动化 AI R&D 流程。
- **欺骗性对齐监控 (Deceptive alignment monitoring)**：明确承诺对工具性推理滥用进行自动化监控。

自主性的框架与 OpenAI 不同。DeepMind 没有将"自主性"作为顶层领域保留；它被纳入自主性会造成危害的领域（ML R&D 和网络）中。其论点是，没有领域的自主性是有能力而无风险的能力；反论点是，跨领域的自主性是一个框架应该命名的元风险。

### 三者趋同之处 (What all three converge on)

- 内部安全咨询小组（Anthropic 的 SAG、OpenAI 的 SAG、DeepMind 的内部委员会）。高能力模型部署前进行审查。
- 明确提及欺骗性对齐 / 对齐造假作为一类风险。
- 按声明节奏发布的常设工件（Anthropic：前沿安全路线图、风险报告；OpenAI：能力和安全措施报告；DeepMind：FSF 更新周期）。
- 承认仅靠监控的防御有其上限。DeepMind 明确表示："自动化监控长期来看将不再足够。"

### 分歧之处 (Where they diverge)

- **Anthropic**：暂停承诺在 v3.0 中被移除；AI R&D-4 阈值是命名的下一个关卡。
- **OpenAI**：跟踪 vs 研究的划分；研究类别（包括长程自主性）不会自动关卡。
- **DeepMind**：自主性被纳入其他领域；跟踪能力级别在 2026 年 4 月增加了粒度。

### 消极应付：一个使三者都复杂化的特定能力 (Sandbagging: a specific capability that complicates all three)

消极应付（模型在评估中策略性地表现不佳）位于 OpenAI 的研究类别中。Anthropic 的 RSP v3.0 通过评估上下文差距（第 1 课）来解决它。DeepMind 通过 FSF v3 中的欺骗性对齐监控来解决它。

如果模型在评估中消极应付，每个框架的能力阈值都会被低估。框架只有在测量有效时才有效。这就是为什么外部测量（第 21 课，METR）和对抗性评估在实验室自我评估之外是必要的。

### 政策阅读技能 (The policy-reading skill)

- **定位 (Locate)**：你关心的每种能力都应该能在政策中找到。如果找不到，政策就不覆盖它。
- **分类 (Classify)**：它是跟踪（触发缓解）还是研究（跟踪但不触发）？OpenAI 明确命名了这一点；Anthropic 和 DeepMind 有自己的等价物。
- **节奏 (Cadence)**：政策是按声明的时间表更新，还是仅在特定事件后更新？声明的时间表更强。
- **独立性 (Independence)**：外部审查是强制性的还是可选的？Anthropic 与 Apollo 和美国 AI 安全研究所合作；OpenAI 与 METR 合作；DeepMind 主要与内部 SAG 合作。

## 使用它 (Use It)

`code/main.py` 实现了一个小型决策表差异比较工具。给定一种能力（自主性、欺骗性对齐、R&D 自动化、网络能力提升等），它输出三个政策各自如何对该能力进行分类，以及触发哪些缓解措施。它是一个阅读辅助工具，而非政策工具。

## 交付它 (Ship It)

`outputs/skill-cross-policy-diff.md` 针对特定能力生成跨政策比较，以三个框架为参考。

## 练习 (Exercises)

1. 运行 `code/main.py`。确认差异比较工具的输出与源文档中至少两种能力的政策一致。

2. 完整阅读 OpenAI 准备框架 v2。找出每个研究类别。对每个类别，写一句话说明为什么它属于研究而非跟踪。

3. 完整阅读 DeepMind FSF v3，以及 2026 年 4 月的跟踪能力级别更新。找出 ML R&D 自主性级别 1 的具体评估标准。你将如何从外部测量它？

4. 消极应付位于 OpenAI 的研究类别中。设计一个评估，迫使一个消极应付的模型暴露其真实能力。参考第 1 课关于评估上下文博弈的讨论。

5. 比较三个政策在某个特定能力上的处理（自行选择）。指出哪个政策的分类你认为最严谨，哪个最不严谨。用源文本证明你的判断。

## 关键术语 (Key Terms)

| 术语 (Term) | 人们说的 (What people say) | 实际含义 (What it actually means) |
|---|---|---|
| 准备框架 (Preparedness Framework) | "OpenAI 的扩展政策" | PF v2（2025 年 4 月）；跟踪 vs 研究类别 |
| 跟踪类别 (Tracked Category) | "强制缓解" | 触发能力 + 安全措施报告；SAG 审查 |
| 研究类别 (Research Category) | "仅监控" | 跟踪但无自动缓解；包括长程自主性 |
| 前沿安全框架 (Frontier Safety Framework) | "DeepMind 的扩展政策" | FSF v3（2025 年 9 月）+ 跟踪能力级别（2026 年 4 月） |
| CCL | "关键能力级别" | DeepMind 每个领域的阈值（网络、生物、ML R&D、CBRN） |
| ML R&D 自主性级别 1 (ML R&D autonomy level 1) | "R&D 自动化" | 以有竞争力的成本完全自动化 AI R&D 流程 |
| 消极应付 (Sandbagging) | "策略性表现不佳" | 模型在评估中表现不佳；位于 OpenAI 研究类别中 |
| 工具性推理 (Instrumental reasoning) | "手段-目的推理" | 关于如何实现目标的推理；DeepMind 监控的目标 |

## 延伸阅读 (Further Reading)

- [OpenAI — Updating our Preparedness Framework](https://openai.com/index/updating-our-preparedness-framework/) — v2 公告。
- [OpenAI — Preparedness Framework v2 PDF](https://cdn.openai.com/pdf/18a02b5d-6b67-4cec-ab64-68cdfbddebcd/preparedness-framework-v2.pdf) — 完整文档。
- [DeepMind — Strengthening our Frontier Safety Framework](https://deepmind.google/blog/strengthening-our-frontier-safety-framework/) — FSF v3 公告。
- [DeepMind — Updating the Frontier Safety Framework (April 2026)](https://deepmind.google/blog/updating-the-frontier-safety-framework/) — 跟踪能力级别新增。
- [Gemini 3 Pro FSF Report](https://storage.googleapis.com/deepmind-media/gemini/gemini_3_pro_fsf_report.pdf) — FSF 格式风险报告示例。