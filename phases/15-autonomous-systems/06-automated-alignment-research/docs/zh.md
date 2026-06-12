# 自动化对齐研究（Anthropic AAR）

> Anthropic 在独立的沙箱中运行了多组并行的 Claude Opus 4.6 自主对齐研究人员（Autonomous Alignment Researchers），通过一个共享论坛进行协调，该论坛的日志存储在所有沙箱之外（因此智能体无法删除自己的记录）。在弱到强训练（weak-to-strong training）问题上，AAR 的表现超过了人类研究人员。Anthropic 自己的总结指出，预设工作流（prescribed workflows）常常限制 AAR 的灵活性并降低性能。自动化对齐研究是压缩步骤，它将时间线压缩到 RSP 旨在检测的确切错位风险。

**类型：** 学习
**语言：** Python（标准库，并行研究论坛模拟器）
**前置要求：** Phase 15 · 05（AI Scientist v2），Phase 15 · 04（DGM）
**时间：** ~60 分钟

## 问题

对齐研究在人类研究人员时间上成本高昂。可扩展监督（scalable oversight）、奖励规范（reward specification）或弱到强训练等问题需要每次迭代耗时数周的实验。随着前沿能力的进步，对齐工作量的增长速度超过了合格研究人员的供应。

自动化对齐研究（Automated Alignment Research，AAR）提出了一个问题：那些能力正在超越对齐的前沿模型，是否能够帮助缩小这一差距。Anthropic 2026 年关于 AAR 运行的弱到强训练研究报告（alignment.anthropic.com/2026/automated-w2s-researcher/）是该类部署系统的首批公开结果之一。

结果是真正积极的：AAR 在研究问题上比人类基线表现更好。这个结果也在一个系统中集中了本阶段发展的每一个担忧。如果对齐研究可以被自动化，那么那些损害安全措施的部分也可以。RSP 中关于 AI 研发能力的阈值正是考虑到这个循环而制定的。

## 概念

### 架构（根据公开描述）

- **并行智能体。** 多个 Claude Opus 4.6 实例作为 AAR 运行，每个在独立的沙箱中。沙箱限制文件系统、网络和进程面。
- **共享论坛。** 发现结果发布到一个共享论坛，其存储后端位于所有沙箱之外。智能体可以读取论坛；智能体不能从自己的沙箱内删除或编辑论坛记录。
- **任务分配。** 一个人类设计的任务队列将研究问题分配给 AAR。AAR 提出分解方案、运行实验、撰写发现结果。
- **弱到强训练焦点。** 已发表的研究针对弱到强训练——使用较弱的监督模型来训练较强的学生模型。AAR 在此任务上优于人类研究人员。

### 为什么沙箱外日志很重要

日志完整性属性是受控实验与可能被破坏的实验之间的区别。如果 AAR 可以编辑自己的日志，它们就可以掩盖失败或可疑的实验。Anthropic 直接指出了这一点：该属性是信任研究输出的必要条件，并且验证起来并不简单。

在基础设施术语中，这类似于追加写入日志（append-only logs），通过直写（write-through）到外部存储（例如 S3 对象锁定、不可变的 Datadog 日志）。智能体没有修改过去记录的凭证。这与 2026 年企业安全团队用于创建抗勒索软件审计追踪的模式相同。

### 预设工作流的权衡

Anthropic 的报告指出，预设工作流常常限制 AAR 的灵活性并降低性能。换句话说：如果人类设计者逐步编写研究计划，AAR 的表现不如被给予高层目标和分解自由时好。这与跨智能体研究（LangGraph、Microsoft 的 Agent Framework）的结果一致——过度指定的脚手架会损害有能力的智能体。

安全含义是微妙的。提高 AAR 性能的灵活性，也正是允许未经授权的分解、新颖工具使用和目标漂移的灵活性。这种权衡是一个设计空间决策：多少灵活性足以获得超越人类的能力，而又不给智能体留出在分布外泛化目标的空间？

### 压缩风险

RSP v3.0（第 19 课）引入了一个 AI 研发能力阈值：以与人类加 AI 工具相比有竞争力的成本完全自动化 AI 研发流程的能力。DeepMind 的 FSF v3 包含一个类似的 ML 研发自主性级别。两个框架都将此阈值视为触发加强控制的临界点。

AAR 距离该阈值还差一步：它自动化了流程的一部分（针对特定、范围明确的任务的对齐研究），但不是端到端的能力开发循环。时间线问题是差距关闭的速度有多快。

压缩的时间线是复合失败（compounding-failure）的担忧。如果对齐研究和能力研究以相似的速度复合，错位风险面至少与能力增长一样快。如果能力复合更快（历史趋势），差距就会扩大。这是支持 AAR 作为有条件的好事的论点：每个额外的对齐结果都会缩小差距，但前提是研究过程是可信的。

### AAR 不能替代什么

人类研究人员设置任务队列、审查结果并持有宪法权威。AAR 加速了流程的中间部分，而不是两端。Anthropic 发布的输出既包括 AAR 的贡献，也包括人类研究人员关于发布什么、撤回什么以及改进什么的判断。

这与第 15 课中应用于研究本身的"提议然后提交"（propose-then-commit）模式相匹配：AAR 提议；人类提交。

## 使用它

`code/main.py` 模拟一个多 AAR 研究论坛。三个独立的"研究智能体"并行运行，将发现结果发布到一个追加写入论坛，并带有可选的篡改尝试。

观察：

- 沙箱外日志如何捕获日志篡改尝试。
- 带有自由分解的任务分配如何在更困难的问题上优于固定工作流。
- 如果日志不是防篡改的，一个被攻破的 AAR 如何污染论坛的信号。

## 交付它

`outputs/skill-aar-deployment-review.md` 是一个自动化对齐研究流程的部署前审查清单。

## 练习

1. 运行 `code/main.py`。比较"固定工作流"和"自由分解"设置。自由分解总是赢吗？还是存在某些问题类别，固定工作流具有竞争力？

2. 修改模拟器，使一个智能体尝试日志篡改。确认追加写入日志检测到了它。写一段话描述检测签名在日志中的确切样子。

3. 阅读 Anthropic 的弱到强 AAR 报告。确定 AAR 在哪个具体子任务上击败了人类研究人员。是什么使它适合自动化？

4. 设计一个任务队列分配策略，平衡 AAR 的灵活性（更好的结果）与预设工作流约束（更容易审计）。描述你将如何对两者进行 A/B 测试。

5. 阅读 RSP v3.0 的 AI R&D-4 阈值。用一段话描述你认为什么会跨越该阈值而 AAR 目前还没有。

## 关键术语

| 术语 | 人们说的 | 实际含义 |
|---|---|---|
| AAR | "自动化对齐研究员" | 在对齐问题上自主运行的 Claude Opus 4.6 实例 |
| 弱到强训练（Weak-to-strong training） | "用较弱的监督者训练较强的模型" | AAR 优于人类的经典可扩展监督基准 |
| 共享论坛（Shared forum） | "智能体发布发现结果的地方" | 追加写入、沙箱外存储 |
| 沙箱外日志（Out-of-sandbox log） | "智能体不能编辑自己的记录" | 防篡改直写到外部存储 |
| 预设工作流（Prescribed workflow） | "人类设计者的逐步计划" | 限制 AAR；通常性能低于自由分解 |
| 自由分解（Free decomposition） | "智能体决定如何分解任务" | 能力更强，更难审计 |
| AI 研发阈值（AI R&D threshold） | "RSP/FSF 能力级别" | 以有竞争力的成本完全自动化研发流程 |
| 压缩时间线（Compressed timeline） | "对齐与能力的竞赛" | 如果能力复合快于对齐，错位风险增长 |

## 延伸阅读

- [Anthropic — Automated Weak-to-Strong Researcher](https://alignment.anthropic.com/2026/automated-w2s-researcher/) — 主要来源。
- [Anthropic Responsible Scaling Policy v3.0](https://anthropic.com/responsible-scaling-policy/rsp-v3-0) — AI 研发阈值框架。
- [Anthropic — Measuring AI agent autonomy](https://www.anthropic.com/research/measuring-agent-autonomy) — 更广泛的智能体自主性框架。
- [DeepMind Frontier Safety Framework v3](https://deepmind.google/blog/strengthening-our-frontier-safety-framework/) — 与 RSP 平行的 ML 研发自主性级别。
- [Burns 等人（2023）. Weak-to-Strong Generalization（OpenAI）](https://openai.com/index/weak-to-strong-generalization/) — AAR 所攻击的底层问题。