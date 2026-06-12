# Self-Refine 与 CRITIC：迭代输出改进

> Self-Refine（Madaan 等人，2023）使用一个 LLM 扮演三个角色——生成、反馈、优化——在一个循环中。7 个任务上平均绝对增益 +20。CRITIC（Gou 等人，2023）通过将验证路由到外部工具来强化反馈步骤。2026 年，这种模式在每个框架中都以"评估器-优化器"（Anthropic）或护栏循环（OpenAI Agents SDK）的形式提供。

**类型：** Build
**语言：** Python（标准库）
**前置知识：** Phase 14 · 01（智能体循环），Phase 14 · 03（Reflexion）
**时间：** ~60 分钟

## 学习目标

- 陈述 Self-Refine 的三个提示（生成、反馈、优化）并解释历史对优化提示的重要性。
- 解释 CRITIC 的关键洞察：LLM 在没有外部基础时无法可靠地自我验证。
- 实现一个标准库 Self-Refine 循环，包含历史和可选的外部验证器。
- 将这种模式映射到 Anthropic 的"评估器-优化器"工作流和 OpenAI Agents SDK 的输出护栏。

## 问题

智能体产生了一个几乎正确的答案。可能一行代码有语法错误。可能摘要太长了。可能计划遗漏了一个边缘情况。你想要的是：智能体批评自己的输出，然后修复它。

Self-Refine 展示了这在单一模型上有效，不需要训练数据，不需要 RL。但有一个问题：LLM 在硬事实上不擅长自我验证。CRITIC 指出了修复方法——将验证步骤路由到外部工具（搜索、代码解释器、计算器、测试运行器）。

这两篇论文共同定义了 2026 年迭代改进的默认方式：生成、验证（尽可能使用外部工具）、优化、验证通过时停止。

## 概念

### Self-Refine（Madaan 等人，NeurIPS 2023）

一个 LLM，三个角色：

```
generate(task)            -> output_0
feedback(task, output_0)  -> critique_0
refine(task, output_0, critique_0, history) -> output_1
feedback(task, output_1)  -> critique_1
refine(task, output_1, critique_1, history) -> output_2
...
stop when feedback says "no issues" or budget exhausted.
```

关键细节：`refine` 看到完整历史——所有先前输出和批评——因此它不会重复错误。论文进行了消融：去掉历史，质量急剧下降。

头条：在 7 个任务（数学、代码、简写、对话）上平均绝对改进 +20，包括 GPT-4。无训练、无外部工具、单一模型。

### CRITIC（Gou 等人，arXiv:2305.11738，v4 2024 年 2 月）

Self-Refine 的弱点：反馈步骤是 LLM 自我评分。对于事实声明，这是不可靠的（幻觉通常对产生它的模型来说看起来很有说服力）。CRITIC 用 `verify(task, output, tools)` 替换了 `feedback(task, output)`，其中 `tools` 包括：

- 用于事实声明的搜索引擎。
- 用于代码正确性的代码解释器。
- 用于算术的计算器。
- 领域特定的验证器（单元测试、类型检查器、linter）。

验证器产生基于工具结果的结构化批评。然后优化器基于此批评进行条件优化。

头条：CRITIC 在事实任务上优于 Self-Refine，因为批评是基于基础的。在没有外部验证器的任务上（创意写作、格式化），CRITIC 退化为 Self-Refine。

### 停止条件

两种常见形态：

1. **验证器通过。** 外部测试返回成功。可用时首选（单元测试、类型检查器、护栏断言）。
2. **未发出反馈。** 模型说"输出很好。"更便宜但不可靠；与最大迭代上限配对。

2026 年默认：结合两者。"如果验证器通过，或者模型说很好且迭代次数 >= 2，或者迭代次数 >= max_iterations，则停止。"

### 评估器-优化器（Anthropic，2024）

Anthropic 2024 年 12 月的帖子将其命名为五种工作流模式之一。两个角色：

- 评估器：评分输出并产生批评。
- 优化器：根据批评修订输出。

循环直到评估器通过。这是 Anthropic 框架下的 Self-Refine/CRITIC。Anthropic 添加的关键工程细节：评估器和优化器提示应该显著不同，这样模型就不会只是橡皮图章。

### OpenAI Agents SDK 输出护栏

OpenAI Agents SDK 将这种模式作为"输出护栏"提供。护栏是在智能体最终输出上运行的验证器。如果护栏触发（引发 `OutputGuardrailTripwireTriggered`），输出被拒绝，智能体可以重试。护栏可以调用工具（CRITIC 风格）或作为纯函数（Self-Refine 风格）。

### 2026 年陷阱

- **橡皮图章循环。** 相同模型用相同提示风格做生成和批评，趋同于"看起来不错。"使用结构不同的提示，或使用更小的廉价模型进行批评。
- **过度优化。** 每次优化传递增加延迟和 token。预算 1-3 次传递；之后升级为人类审查。
- **琐碎任务上的 CRITIC。** 如果没有外部验证器，CRITIC 退化为 Self-Refine；不要为存根验证器付出延迟代价。

## 构建

`code/main.py` 在玩具任务上实现 Self-Refine 和 CRITIC：根据给定主题生成简短的项目符号列表。验证器检查格式（3 个项目符号，每个少于 60 字符）。CRITIC 添加了一个外部"事实验证器"，惩罚已知的幻觉。

组件：

- `generate`——脚本化生成器。
- `feedback`——LLM 风格的自我批评。
- `verify_external`——CRITIC 风格基于基础的验证器。
- `refine`——根据历史重写输出。
- 停止条件——验证器通过或最多 4 次迭代。

运行：

```
python3 code/main.py
```

比较 Self-Refine 和 CRITIC 的运行。CRITIC 捕获了 Self-Refine 忽略的事实错误，因为外部验证器拥有自我批评者不具备的基础。

## 使用

Anthropic 的评估器-优化器是这种模式在 Claude 友好的语言下的表达。OpenAI Agents SDK 的输出护栏是 CRITIC 形状的（护栏可以调用工具）。LangGraph 提供类似于 Self-Refine 的反思节点。Google 的 Gemini 2.5 Computer Use 添加了每个步骤的安全评估器，这是一个 CRITIC 变体：每个行动在提交前都经过验证。

## 交付

`outputs/skill-refine-loop.md` 根据任务形状、验证器可用性和迭代预算配置评估器-优化器循环。为生成器、评估器/验证器和优化器发出提示，以及一个停止策略。

## 练习

1. 使用 max_iterations=1 运行玩具。CRITIC 还有帮助吗？
2. 将外部验证器替换为嘈杂的验证器（随机 30% 误报）。循环会做什么？这是 2026 年大多数护栏栈的现实。
3. 实现"生成器-批评器在不同模型上"的变体：大模型生成，小模型批评。它胜过同模型方案吗？
4. 阅读 CRITIC 第 3 节（arXiv:2305.11738 v4）。说出三种验证工具类别，并为每个给出一个例子。
5. 将 OpenAI Agents SDK 的 `output_guardrails` 映射到 CRITIC 的验证器角色。SDK 做错了什么，又做对了什么？

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------------|------------------------|
| Self-Refine | "自我修复的 LLM" | 一个模型中的生成 -> 反馈 -> 优化循环，包含历史 |
| CRITIC | "基于工具的验证" | 用外部验证器替换反馈（搜索、代码、计算、测试） |
| 评估器-优化器 | "Anthropic 工作流模式" | 两个角色——评估器评分，优化器修订——循环到收敛 |
| 输出护栏 | "事后检查" | OpenAI Agents SDK 验证器，在智能体产生输出后运行 |
| 验证步骤 | "批评阶段" | 承重的决定：基于基础还是自我评分 |
| 优化历史 | "模型已经尝试过的" | 预置到优化提示中的先前输出 + 批评；去掉则质量崩溃 |
| 橡皮图章循环 | "自我一致失败" | 相同提示的批评返回"看起来不错"；用结构不同的提示修复 |
| 停止条件 | "收敛测试" | 验证器通过或没有反馈且达到迭代上限；永远不要单条件 |

## 延伸阅读

- [Madaan 等人，Self-Refine（arXiv:2303.17651）](https://arxiv.org/abs/2303.17651) — 经典论文
- [Gou 等人，CRITIC（arXiv:2305.11738）](https://arxiv.org/abs/2305.11738) — 基于工具的验证
- [Anthropic，构建有效的智能体](https://www.anthropic.com/research/building-effective-agents) — 评估器-优化器工作流模式
- [OpenAI Agents SDK 文档](https://openai.github.io/openai-agents-python/) — 作为 CRITIC 形状验证器的输出护栏
