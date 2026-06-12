# 代理框架权衡——LangGraph vs CrewAI vs AutoGen vs Agno

> 每个框架都在兜售同一个演示（研究代理构建报告），并隐藏同一个错误（状态模式与编排层冲突）。选择其抽象与问题形状匹配的框架；其他一切都是你需要写两遍的胶水代码。

**类型：** Learn
**语言：** Python
**前置要求：** 阶段 11 · 09（函数调用），阶段 11 · 16（LangGraph）
**预计时间：** ~45 分钟

## 问题

你有一个需要不止一次 LLM 调用的任务。可能是一个研究工作流（规划、搜索、总结、引用）。可能是一个代码审查流水线（解析 diff、评审、修补、验证）。可能是一个多轮助手，预订航班、写邮件和提交费用报告。你选择一个框架。

三天后，你发现框架的抽象泄漏了。CrewAI 给你角色，但当"研究员"需要将结构化计划交给"写手"时，它与你抗争。AutoGen 给你代理间的聊天，但没有一等公民的状态，所以你的检查点是一个对话日志的 pickle。LangGraph 给你状态图，但迫使你在知道代理会做什么之前命名每一个转换。Agno 给你一个单代理抽象，当你尝试扩展到三个并发工作者时它就会崩溃。

解决办法不是"选最好的框架"。而是将框架的核心抽象与你的问题形状相匹配。本课程绘制了这张地图。

## 概念

![代理框架矩阵：核心抽象 vs 问题形状](../assets/framework-matrix.svg)

四个框架主导 2026 年格局。它们的核心抽象并不相同。

| 框架 | 核心抽象 | 最适合 | 最不适合 |
|-----------|------------------|----------|-----------|
| **LangGraph** | `StateGraph`——类型化状态、节点、条件边、检查点器。 | 具有显式状态和人机循环中断的工作流；需要时间旅行调试的生产代理。 | 松散的、角色驱动的头脑风暴，拓扑结构未知。 |
| **CrewAI** | `Crew`——角色（目标、背景故事）、任务、流程（顺序或分层）。 | 具有短线性/分层计划的角色扮演或角色驱动工作流。 | 任何超越团队轮次历史的状态性事务；复杂的分支。 |
| **AutoGen** | `ConversableAgent` 对——两个或更多代理轮流发言直到退出条件。 | 多代理*对话*（老师-学生、提议者-批评者、行为者-评审者），思维从聊天中涌现。 | 具有已知 DAG 的确定性工作流；任何需要跨重启持久状态的事务。 |
| **Agno** | `Agent`——单个 LLM + 工具 + 记忆，可组合成团队。 | 快速构建的单代理和轻量级团队；强大的多模态能力和内置存储驱动。 | 深度的、显式分支的图形与自定义 reducer。 |

### "抽象"实际上意味着

框架的核心抽象是你在白板上画的东西，当你推销架构时。

- **LangGraph** → 你画一个图。节点是步骤，边是转换，每个点的状态对象是类型化的。思维模型是一个状态机。
- **CrewAI** → 你画一个组织结构图。每个角色有一个工作描述，经理分配任务。思维模型是一个小型专家团队。
- **AutoGen** → 你画一个 Slack 私信。两个代理互相发消息；如果你需要主持人，第三个加入。思维模型是聊天。
- **Agno** → 你画一个带有工具的框。把框放在一起成为一个团队。思维模型是"开箱即用的代理"。

### 状态问题

状态是大多数框架选择在生产中崩溃的地方。

- **LangGraph.** 类型化状态（`TypedDict` 或 Pydantic 模型），每个字段 reducer，一等公民检查点器（SQLite/Postgres/Redis）。恢复、中断和时间旅行都是免费的。*(见阶段 11 · 16。)*
- **CrewAI.** 状态通过 `context` 字段在任务之间作为字符串流动，或通过 `output_pydantic` 结构化。没有开箱即用的持久性团队存储；如果团队必须在重启后存活，你需要自己添加。
- **AutoGen.** 状态是聊天历史和你定义的任何 `context`。对话日志持久化；任意工作流状态不会持久化，除非你编写适配器。
- **Agno.** 内置存储驱动（SQLite, Postgres, Mongo, Redis, DynamoDB），通过 `storage=` 附加到 `Agent`——对话会话和用户记忆自动持久化。不是完整的图形检查点器；而是会话存储。

### 分支问题

每个非平凡的代理都会分支。谁决定分支很重要。

- **LangGraph**——你决定，通过条件边。路由是一个带有命名分支的 Python 函数。分支在编译后的图中是一等公民；检查点器记录了采取哪个分支。
- **CrewAI**——经理在分层模式下决定；在顺序模式下你在构建时决定。路由在任务列表中隐式；在经理的提示之外没有一等公民的"if"。
- **AutoGen**——代理通过聊天决定。分支从谁下一个发言中涌现。`GroupChatManager` 选择下一个发言者；你可以手写 `speaker_selection_method`，但默认是 LLM 驱动的。
- **Agno**——代理通过接下来调用哪个工具来决定。团队有协调者/路由器/协作者模式；除此之外的分支是开发者的责任。

### 可观测性问题

- **LangGraph**——通过 LangSmith 或任何 OTel 导出器的 OpenTelemetry。每个节点转换都是一个跟踪跨度；检查点兼作可重放的跟踪。LangSmith 是第一方选项；Langfuse/Phoenix 也有适配器。
- **CrewAI**——自 2025 年底起有一等公民的 OpenTelemetry；与 Langfuse、Phoenix、Opik、AgentOps 集成。
- **AutoGen**——通过 `autogen-core` 的 OpenTelemetry 集成；AgentOps 和 Opik 有连接器。跟踪粒度是按代理消息，而不是按节点。
- **Agno**——内置 `monitoring=True` 标志加上 OpenTelemetry 导出器；与 Langfuse 紧密集成，用于会话跟踪。

### 成本和延迟

所有四个框架都增加了每次调用的开销（框架逻辑、验证、序列化）。粗略的开销递增顺序：Agno ≈ LangGraph < CrewAI ≈ AutoGen。差异主要在于框架做多少额外的 LLM 路由。CrewAI 的分层经理花费 token 决定谁下一个发言；AutoGen 的 `GroupChatManager` 也是如此。LangGraph 只在你写 `llm.invoke` 的地方花费 token。Agno 的单代理路径很薄。

当每次运行的成本很重要时，优先选择显式路由（LangGraph 边、AutoGen `speaker_selection_method`）而不是 LLM 选择的路由。

### 互操作性

- **LangGraph** ↔ **LangChain** 工具、检索器、LLM。一等公民 MCP 适配器（作为 MCP 服务器导入的工具）。
- **CrewAI** ↔ 工具继承自 `BaseTool`；LangChain 工具、LlamaIndex 工具和 MCP 工具都可以适配。通过 `allow_delegation=True` 进行团队间委派。
- **AutoGen** → `FunctionTool` 包装任何 Python 可调用对象；有 MCP 适配器。紧密耦合到 AG2 生态系统以实现代理到代理的模式。
- **Agno** → `@tool` 装饰器或 BaseTool 子类；MCP 适配器；工具可以在代理和团队之间共享。

## 技能

> 你能用一句话解释为什么给定框架适合给定的代理问题。

构建前检查清单：

1. **画出形状。** 这是一个图（类型化状态、命名转换）？角色扮演（专家交接工作）？聊天（代理聊到完）？带工具的单一代理？
2. **决定谁分支。** 开发者决定的分支 → LangGraph。经理代理决定 → CrewAI 分层。聊天涌现 → AutoGen。工具调用决定 → Agno。
3. **检查状态预算。** 你需要从检查点恢复吗？时间旅行？运行中人机中断？如果是，LangGraph 是默认选择；Agno 会话覆盖对话范围的状态。
4. **检查成本预算。** LLM 选择的路由每次轮次花费额外 token。如果代理每天运行数千次，优先选择显式路由。
5. **预算框架开销。** 每个框架都是另一个依赖。如果任务只是两次 LLM 调用和一个工具，写 30 行普通 Python；没有框架比任何框架更快。

拒绝在你画出图、组织结构图、聊天或代理框之前选择框架。拒绝选择一个强迫你与它的状态模型对抗的框架。

## 决策矩阵

| 问题形状 | 首选框架 | 原因 |
|---------------|---------------------|-----|
| 具有类型化状态、人工审批、长时间运行的工作流 DAG | LangGraph | 一等公民状态、检查点器、中断、时间旅行。 |
| 具有不同角色的研究/写作流水线 | CrewAI（顺序）或 LangGraph 子图 | 在 CrewAI 中每个角色是一个任务，表达起来很简单；当分支变得复杂时用 LangGraph 扩展。 |
| 提议者-批评者或老师-学生对话 | AutoGen | 双代理聊天是它的原生形态。 |
| 带工具、会话、记忆的单代理 | Agno | 最薄的设置，内置存储和记忆。 |
| 数千个并行扇出带 reducer | LangGraph + `Send` | 唯一一个有一等公民并行分发 API 的。 |
| 快速原型，不绑定框架 | 普通 Python + 提供商 SDK | 没有框架是最快的框架。 |

## 练习

1. **简单。** 使用同一任务——"研究 Anthropic 总部，写一篇 200 字简报，引用来源"——在 LangGraph（四个节点：规划、搜索、写作、引用）和 CrewAI（三个角色：研究员、写手、编辑）中实现。报告每次运行的 token 成本和代码行数。
2. **中等。** 在 AutoGen（研究员 ↔ 写手聊天，编辑通过 `GroupChat` 加入）和 Agno（一个带有 `search_tools` 和 `write_tools` 的单代理，加上会话存储）中构建同一任务。对四个实现进行排名：(a) 每次运行成本，(b) 崩溃后恢复能力，(c) 在写步骤之前注入人工审批的能力。
3. **困难。** 构建一个决策树脚本 `pick_framework.py`，接受简短的问题描述（JSON：`{has_typed_state, has_roles, has_dialogue, has_parallel_fanout, needs_resume}`）并返回带有一句话说明的推荐。在你设计的六个案例上验证它。

## 关键术语

| 术语 | 人们说的 | 实际含义 |
|------|-----------------|-----------------------|
| 编排 | "代理如何协调" | 决定哪个节点/角色/代理接下来运行。 |
| 持久状态 | "重启后恢复" | 在进程死亡后存活的状态，附加到检查点或会话存储。 |
| LLM 选择的路由 | "让模型决定" | 规划器 LLM 每轮选择下一步；灵活但每次决策花费 token。 |
| 显式路由 | "开发者决定" | Python 函数或静态边选择下一步；便宜且可审计。 |
| Crew | "CrewAI 团队" | 角色 + 任务 + 流程（顺序或分层）绑定为一个可运行对象。 |
| GroupChat | "AutoGen 的多代理聊天" | N 个代理之间的受控对话，带有发言者选择器。 |
| Team (Agno) | "多代理 Agno" | 一组代理上的路由/协调/协作模式。 |
| StateGraph | "LangGraph 的图" | 类型化状态、节点、条件边、检查点器抽象。 |

## 延伸阅读

- [LangGraph 文档](https://langchain-ai.github.io/langgraph/)——StateGraph、检查点器、中断、时间旅行。
- [CrewAI 文档](https://docs.crewai.com/)——Crews、Flows、Agents、Tasks、Processes。
- [AutoGen 文档](https://microsoft.github.io/autogen/)——ConversableAgent、GroupChat、teams、tools。
- [Agno 文档](https://docs.agno.com/)——Agent、Team、Workflow、storage、memory。
- [Anthropic——构建有效代理（2024 年 12 月）](https://www.anthropic.com/research/building-effective-agents)——模式库（提示链、路由、并行化、编排器-工作者、评估器-优化器）框架无关。
- [Yao 等人, "ReAct: Synergizing Reasoning and Acting" (ICLR 2023)](https://arxiv.org/abs/2210.03629)——每个框架都会美化这个循环。
- [Wu 等人, "AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation" (2023)](https://arxiv.org/abs/2308.08155)——AutoGen 的设计论文。
- [Park 等人, "Generative Agents: Interactive Simulacra of Human Behavior" (UIST 2023)](https://arxiv.org/abs/2304.03442)——CrewAI 风格的角色扮演栈所构建的角色扮演基础。
- 阶段 11 · 16 (LangGraph)——本课程作为基准的框架。
- 阶段 11 · 19 (Reflexion)——一个清晰地映射到 LangGraph 但不那么适合 CrewAI 的模式。
- Phase 11 · 22 (Production observability) — how to instrument whichever framework you pick.
