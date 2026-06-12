# 多智能体基元模型

> 2026 年发布的每个多智能体框架——AutoGen、LangGraph、CrewAI、OpenAI Agents SDK、Microsoft Agent Framework——都是一个四维设计空间中的一个点。四个基元，仅此而已：智能体、交接、共享状态、编排器。本课程从零构建它们，在一个玩具系统上运行所有四种，然后将每个主要框架映射到同一坐标轴上，使你能够用一段话读懂任何新版本。

**类型：** Learn
**语言：** Python（标准库）
**前置知识：** Phase 14（智能体工程），Phase 16 · 01（为什么需要多智能体）
**时间：** ~60 分钟

## 问题

每六个月就有一个新的多智能体框架发布。2023 年的 AutoGen。2024 年的 CrewAI。2024 年的 LangGraph 和 OpenAI Swarm。2025 年 4 月的 Google ADK。2026 年 2 月的 Microsoft Agent Framework RC。每篇新闻稿都宣称自己是"正确的抽象"。

如果你试图一个一个地学习它们，你会精疲力尽。API 看起来不同。文档对"智能体"的定义不一致。一个框架称其共享内存为"黑板"，另一个称其为"消息池"，第三个称其为"StateGraph"。你开始怀疑这个领域只是在原地打转。

并非如此。在营销的表象之下，四个基元是稳定的。学会它们一次，就能用一段话读懂每个新框架。

## 概念

### 四个基元

1. **智能体（Agent）** ——一个系统提示加上一个工具列表。无状态；每次运行都从其系统提示和当前消息历史开始。
2. **交接（Handoff）** ——控制权从一个智能体到另一个智能体的结构化转移。从机制上讲，是一个返回新智能体的工具调用，或一个跟随条件的图边。
3. **共享状态（Shared State）** ——多个智能体可以读取（有时写入）的任何数据结构。消息池、黑板、键值存储、向量记忆。
4. **编排器（Orchestrator）** ——决定谁下一个发言的组件。选项：显式图（确定性）、LLM 发言者选择器（软性）、上一个发言者的交接调用（OpenAI Swarm）、或队列上的调度器（群体架构）。

这就是整个设计空间。每个框架为每个轴选择默认值；其余的都是表面语法。

### 每个 2026 框架如何映射

| 框架 | 智能体 | 交接 | 共享状态 | 编排器 |
|---|---|---|---|---|
| OpenAI Swarm / Agents SDK | `Agent(instructions, tools)` | 工具返回 Agent | 调用者的问题 | LLM 的下一个交接调用 |
| AutoGen v0.4 / AG2 | `ConversableAgent` | GroupChat 上的发言者选择器 | 消息池 | 选择器函数（LLM 或轮询） |
| CrewAI | `Agent(role, goal, backstory)` | `Process.Sequential / Hierarchical` | Task 输出链式连接 | 管理器 LLM 或静态顺序 |
| LangGraph | 节点函数 | 图边 + 条件 | `StateGraph` 归约器 | 图，确定性的 |
| Microsoft Agent Framework | 智能体 + 编排模式 | 模式特定 | 线程 / 上下文 | 模式特定 |
| Google ADK | 智能体 + A2A card | A2A task | A2A artifacts | 主机决定 |

表面差异看起来巨大。本质上：相同的四个旋钮。

### 为什么这很重要

一旦你看到了基元，框架对比就变成了一份简短清单：

- 编排器是信任 LLM 进行路由（Swarm），还是在代码中固定路由（LangGraph）？
- 共享状态是完整历史（GroupChat）还是投影的（StateGraph 归约器）？
- 智能体可以修改彼此的提示（CrewAI manager）还是只能交接（Swarm）？

这三个问题回答了 80% 的框架选择问题。你不再"选购最好的多智能体框架"，而是开始为你真正关心的轴进行设计。

### 无状态洞察

除了共享状态外，每个基元都是无状态的。智能体是（prompt, tools）的函数。交接是一个函数调用。编排器是一个调度器。**系统中唯一有状态的东西就是共享状态。** 这就是所有有趣的 bug 所在：内存中毒（Lesson 15）、消息排序、版本控制、写入争用。

隐藏共享状态的框架（Swarm）把问题推给调用者。集中化共享状态的框架（LangGraph checkpoint、AutoGen pool）使其可检查，但将协调成本转移到共享状态实现上。

### 单个基元的剖析

#### 智能体

```
Agent = (system_prompt, tools, model, optional_name)
```

没有记忆。没有状态。具有相同系统提示和工具的两个智能体是可互换的。所有看起来像每个智能体状态的东西实际上都在共享状态或交接协议中。

#### 交接

```
Handoff = (from_agent, to_agent, reason, payload)
```

三种实现占主导地位：

- **函数返回**——工具返回下一个智能体。这是 OpenAI Swarm 模式。智能体在其工具模式中携带路由。
- **图边**——LangGraph。边是声明式的。LLM 产生一个值；条件选择下一个节点。
- **发言者选择**——AutoGen GroupChat。一个选择器函数（有时本身是一个 LLM 调用）读取池子并选择下一个发言者。

#### 共享状态

```
SharedState = { messages: [], artifacts: {}, context: {} }
```

至少是一个消息列表。通常更多：结构化制品（CrewAI Task 输出）、类型化上下文（LangGraph 归约器）、外部内存（MCP、向量数据库）。

两种拓扑结构：**完整池**（每个智能体看到每条消息）和**投影**（智能体看到按角色作用域划分的视图）。完整池简单但扩展性差。投影池扩展性好但需要预先的模式设计。

#### 编排器

```
Orchestrator = ({state, last_speaker}) -> next_agent
```

四种风格：

- **静态**——图在构建时固定（LangGraph 确定性、CrewAI Sequential）。
- **LLM 选择**——LLM 读取池子并选择下一个发言者（AutoGen、CrewAI Hierarchical）。
- **交接驱动**——当前智能体通过调用交接工具决定（Swarm）。
- **队列驱动**——工作者从共享队列中拉取；没有显式的下一个发言者（群体架构、Matrix）。

### 框架之间的变化

一旦基元固定，剩下的设计决策是：

- **内存策略**——临时与持久检查点（LangGraph checkpointer）。
- **安全边界**——谁可以批准交接（人在环中）。
- **成本核算**——每个智能体的 token 预算。
- **可观测性**——跟踪交接、持久化状态以供重放。

所有这些都可以在基元之上实现。它们都不是新的基元。

## 构建

`code/main.py` 用约 150 行标准库 Python 实现了四个基元。没有真实的 LLM——每个智能体是一个脚本策略，这样焦点保持在协调结构上。

文件导出：

- `Agent`——一个包含 name、system prompt、tools、policy function 的数据类。
- `Handoff`——返回新智能体的函数。
- `SharedState`——线程安全的消息池。
- `Orchestrator`——三种变体：`StaticOrchestrator`、`HandoffOrchestrator`、`LLMSelectorOrchestrator`（模拟）。

演示通过所有三种编排器类型运行相同的三个智能体管道（research → write → review），并在最后打印消息池。你可以看到输出仅在*谁选下一个*上有所不同；智能体和共享状态在多次运行中完全相同。

运行：

```
python3 code/main.py
```

预期输出：三种编排器运行，每种模式一个。每个打印最终的消息池。如果研究者决定提早完成，交接驱动运行到达的智能体更少——这就是 LLM 路由权衡的缩影。

## 使用

`outputs/skill-primitive-mapper.md` 是一个技能，读取任何多智能体代码库或框架文档并返回四基元映射。在发布新框架时运行它以获得一段话的理解。

## 交付

在采用新框架之前，先为其编写基元映射。如果你做不到，说明文档不完整，或者框架在发明第五个基元（很少见——检查是否有你没见过的共享状态变体）。

将映射钉在你的架构文档中。当新团队成员加入时，先发给他们映射，再发 API 文档。当框架版本变更时，比较映射的差异，而不是变更日志。

## 练习

1. 使用不同的智能体策略运行 `code/main.py` 三次。观察编排器的选择如何改变哪些智能体运行。
2. 实现第四种编排器类型：队列驱动型，智能体轮询共享状态获取工作。可能发生什么死锁，如何检测？
3. 获取 LangGraph quickstart 并将其重写为四个基元。LangGraph 的哪些抽象是一对一映射的，哪些是便利包装？
4. 阅读 OpenAI Swarm cookbook。识别 Swarm 使四个基元中的哪个最符合人体工程学，以及哪个推给了调用者。
5. 在此表中找到一个完全隐藏共享状态的框架。解释当智能体需要在交接间协调而不重新读取历史时，什么会出问题。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|---|---|---|
| Agent | "带工具的 LLM" | 一个 `(system_prompt, tools, model)` 三元组。无状态。 |
| Handoff | "控制转移" | 命名下一个智能体和可选负载的结构化调用。三种实现：函数返回、图边、发言者选择。 |
| Shared state | "记忆/上下文" | 多智能体系统中唯一有状态的部分。消息池或黑板。 |
| Orchestrator | "协调器" | 决定谁下一个运行的组件。静态图、LLM 选择器、交接驱动或队列驱动。 |
| Primitive | "抽象" | 每个框架参数化的四个轴之一。不是框架特性。 |
| Message pool | "共享聊天历史" | 完整历史的共享状态。易于推理，扩展性差。 |
| Projected state | "作用域视图" | 按角色划分的共享状态视图。可扩展，需要模式设计。 |
| Speaker selection | "谁下一个说话" | 编排器模式，一个函数（通常是 LLM）从组中选择下一个智能体。 |

## 延伸阅读

- [OpenAI cookbook: Orchestrating Agents — Routines and Handoffs](https://developers.openai.com/cookbook/examples/orchestrating_agents)——交接驱动编排的最清晰阐述
- [AutoGen stable docs](https://microsoft.github.io/autogen/stable/)——GroupChat + 发言者选择是 LLM 选择编排的参考
- [LangGraph workflows and agents](https://docs.langchain.com/oss/python/langgraph/workflows-agents)——图边编排和基于归约器的共享状态
- [CrewAI introduction](https://docs.crewai.com/en/introduction)——角色-目标-背景故事智能体，Sequential/Hierarchical 流程
- [AG2 (community AutoGen continuation)](https://github.com/ag2ai/ag2)——在 Microsoft 将 v0.4 移入维护后的活跃 AutoGen v0.2 分支
