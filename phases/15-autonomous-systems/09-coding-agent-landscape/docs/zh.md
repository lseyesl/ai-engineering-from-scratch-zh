# 自主编码代理全景（2026）

> SWE-bench Verified 在不到三年内从 4% 提升到 80.9%。同一款 Claude Sonnet 4.5 在 SWE-agent v1 上得分为 43.2%，在 Cline 自主模式下得分为 59.8% —— 模型周围的脚手架（scaffolding）现在与模型本身同等重要。OpenHands（前身为 OpenDevin）是最活跃的 MIT 许可平台，其 CodeAct 循环直接在沙箱中执行 Python 动作，而非使用 JSON 工具调用。这些头条数字背后隐藏着一个方法论问题：500 个 SWE-bench Verified 任务中有 161 个仅需 1-2 行修改，而 SWE-bench Pro（10 行以上的任务）对同样的前沿模型得分仅为 23-59%。

**类型：** 学习
**语言：** Python（stdlib，CodeAct 与 JSON 工具调用对比）
**前置要求：** 第 14 阶段 · 07（工具使用），第 15 阶段 · 01（长周期代理）
**时间：** ~45 分钟

## 问题

"哪个编码代理最好"是个错误的问题。正确的问题是：在与我的工作匹配的任务分布上，使用我将在生产环境中运行的脚手架，我能获得怎样的端到端可靠性？

从 2022 年到 2026 年，业界认识到脚手架 —— 检索层、规划器、沙箱、编辑-验证循环、反馈格式 —— 是承重结构。Claude Sonnet 4.5 在 SWE-agent v1 上得分为 43.2%；同一个模型在 Cline 的自主脚手架内得分为 59.8%。16.6 个百分点的差异，相同的权重。基础模型是一个组件；循环才是产品。

伴随的问题是基准测试饱和掩盖了退化。SWE-bench Verified 已接近饱和，简单任务的尾部（500 个任务中有 161 个需要 ≤2 行修改）拉高了顶级得分。真实世界的质量更适合用 SWE-bench Pro（10 行以上修改）这样的分布来衡量，在那里同样的领先者得分仍只有 23-59%。

## 概念

### SWE-bench，一段话

SWE-bench（Jimenez 等人）选取真实的 GitHub issue 及其 ground-truth 补丁，要求代理生成一个能让测试套件通过的补丁。SWE-bench Verified（OpenAI，2024）是一个人工筛选的 500 任务子集，移除了模糊和有缺陷的任务。SWE-bench Pro 是更难的后续版本 —— 需要 10 行以上修改的任务，当前前沿代理的得分在 23-59% 之间。

### 2022 → 2026 曲线实际说明了什么

- **2022**：研究模型在原始 SWE-bench 上约 4%。
- **2024**：GPT-4 + Devin 风格脚手架约 14%；SWE-agent 约 12%。
- **2025**：Claude 3.5/3.7 Sonnet 在 Aider 和 SWE-agent 内推至 40-55% 范围。
- **2026**：Claude Sonnet 4.5 及前沿竞品在 SWE-bench Verified 上达到 70-80%+。Epoch AI 的排行榜实时追踪这一数据。

斜率来自三个叠加因素：更好的基础模型、更好的脚手架（CodeAct、反思、验证器循环），以及更好的基准测试（Verified 去除了噪声）。

### CodeAct 与 JSON 工具调用

OpenHands（All-Hands-AI，arXiv:2407.16741，前身为 OpenDevin）采取了一个特定的架构赌注：模型不是发出由宿主解码并执行的 JSON 工具调用，而是发出 Python 代码，由 Jupyter 风格的内核在沙箱中运行。代理可以在一个动作内循环处理文件、链式调用工具并捕获自己的异常。

权衡：

- **JSON 工具调用**：每个动作是一个回合；易于审计；组合性有限；默认安全，因为每次调用都经过显式验证器。
- **CodeAct**：一个动作可以是一个完整的程序；可组合；需要加固的沙箱（OpenHands 使用 Docker 隔离）；失败模式包括沙箱运行时允许的任何行为。

两种架构都已投入生产。CodeAct 在开放平台（OpenHands、smolagents）中占主导地位。JSON 工具调用在托管服务（Anthropic Managed Agents、OpenAI Assistants）中仍占主导地位，由提供商控制执行器。

### 2026 年全景中的脚手架

| 脚手架 | 许可 | 执行模型 | 显著特性 |
|---|---|---|---|
| OpenHands (OpenDevin) | MIT | Docker 中的 CodeAct | 最活跃的开放平台；事件流可重放 |
| SWE-agent | MIT | 代理-计算机接口 (ACI) | 首个端到端 SWE-bench 脚手架 |
| Aider | Apache-2 | 本地仓库中通过 diff 编辑 | 最小脚手架，回归稳定性强 |
| Cline | Apache-2 | 带工具策略的 VS Code 代理 | Sonnet 4.5 上得分最高的开放脚手架 |
| Devin (Cognition) | 专有 | 托管 VM + 规划器 | 首个"AI 软件工程师"产品类别 |
| Claude Code | 专有 | 权限模式 + 例程 | 第 10 课详细介绍代理循环 |

### 为什么脚手架占主导

一次编码运行是一个长周期轨迹（第 1 课）。可靠性在步骤间复合。脚手架带来分数提升的三个地方：

1. **检索**：找到要读取的正确文件是隐性瓶颈。SWE-agent 的 ACI、OpenHands 的文件索引和 Aider 的仓库地图都在解决这个问题。
2. **验证器循环**：运行测试、读取堆栈跟踪并重试，在 SWE-bench 上带来 10 分以上的差异。
3. **失败隔离**：出错时回滚的沙箱可防止复合损害。同一个模型有和没有验证器循环，看起来像两个不同的产品。

### 基准测试饱和与真实分布

OpenHands 作者和 Epoch AI 都指出 SWE-bench Verified 有一个简单尾部：500 个任务中有 161 个仅需 1-2 行修改。高分部分由这个尾部驱动。SWE-bench Pro 限制为 10 行以上的修改，即使对前沿系统也返回 23-59% 的得分。你的生产分布几乎肯定更接近 Pro 而非 Verified。

选择代理的含义：运行你自己 bug 积压中类似 Pro 的子集。重要的分数是在代表你交付内容的任务上的得分。

## 使用它

`code/main.py` 在一个固定的迷你任务分布上比较两个玩具代理脚手架：

1. 一个 **JSON 工具调用** 脚手架，每个回合执行一个动作。
2. 一个 **CodeAct** 脚手架，每个动作可以发出一个小型 Python 片段。

两者都使用存根"模型"（确定性规则），因此比较将脚手架与模型质量隔离开来。输出显示 CodeAct 脚手架以更少的回合解决更多任务，代价是每个动作的爆炸半径更大。

## 交付它

`outputs/skill-scaffold-audit.md` 帮助你在采用前审计一个提议的编码代理脚手架：检索质量、验证器存在性、沙箱隔离，以及基准测试到分布的匹配度。

## 练习

1. 运行 `code/main.py`。在相同的任务集上，每个脚手架需要多少回合？每个动作的爆炸半径是多少？

2. 阅读 OpenHands 论文（arXiv:2407.16741）。论文论证 CodeAct 在复杂任务上优于 JSON 工具调用。找出论文承认的一个失败模式，并用一句话说明该模式何时会在生产中占主导。

3. 从你的 bug 积压中挑选一个需要跨两个文件进行 10 行以上修改的任务。估算前沿模型在 (a) JSON 工具调用和 (b) CodeAct 下的端到端成功概率。说明差距的原因。

4. SWE-bench Verified 有 161 个单文件、1-2 行的任务。构建一个排除它们的分数。排行榜会如何重新排序？

5. 阅读"Introducing SWE-bench Verified"（OpenAI）。解释用于移除模糊任务的具体方法论，并指出筛选会遗漏的一个类别。

## 关键术语

| 术语 | 人们说的 | 实际含义 |
|---|---|---|
| SWE-bench | "编码基准测试" | 带有 ground-truth 补丁和测试套件的真实 GitHub issue |
| SWE-bench Verified | "清洗后的子集" | 500 个人工筛选任务，存在简单尾部 |
| SWE-bench Pro | "更难的子集" | 10 行以上修改；前沿得分 23-59% |
| CodeAct | "代码即动作" | 代理发出 Python；Jupyter 风格内核在沙箱中执行 |
| JSON 工具调用 | "函数调用" | 每个动作是一个结构化的 JSON 载荷，执行前经过验证 |
| 脚手架 (Scaffold) | "代理框架" | 围绕基础模型的检索 + 规划器 + 执行器 + 验证器循环 |
| ACI (代理-计算机接口) | "SWE-agent 的格式" | 为 LLM 人体工学而非人类 shell 设计的命令集 |
| 验证器循环 (Verifier loop) | "测试并重试" | 运行测试、读取输出、修改补丁；最大的非模型可靠性增益 |

## 延伸阅读

- [Jimenez et al. — SWE-bench](https://www.swebench.com/) —— 原始基准测试和方法论。
- [OpenAI — Introducing SWE-bench Verified](https://openai.com/index/introducing-swe-bench-verified/) —— 筛选子集的构建方式。
- [Wang et al. — OpenHands: An Open Platform for AI Software Developers](https://arxiv.org/abs/2407.16741) —— CodeAct 架构和事件流设计。
- [Epoch AI — SWE-bench leaderboard](https://epoch.ai/benchmarks) —— 实时追踪的分数。
- [Anthropic — Measuring agent autonomy](https://www.anthropic.com/research/measuring-agent-autonomy) —— 长周期编码代理可靠性框架。