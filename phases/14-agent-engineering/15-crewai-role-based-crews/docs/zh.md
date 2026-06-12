# CrewAI：基于角色的团队与流程

> CrewAI 是 2026 年基于角色的多智能体框架。四个原语：Agent、Task、Crew、Process。两种顶层形态：Crews（自主、基于角色的协作）和 Flows（事件驱动、确定性）。文档直截了当地说："对于任何生产就绪的应用，从 Flow 开始。"

**类型：** Learn + Build
**语言：** Python（标准库）
**前置知识：** Phase 14 · 12（工作流模式），Phase 14 · 14（参与者模型）
**时间：** ~75 分钟

## 学习目标

- 说出 CrewAI 的四个原语（Agent、Task、Crew、Process）以及各自的所有权。
- 区分 Sequential、Hierarchical 和计划中的 Consensus 流程；为每种工作负载选择一个。
- 区分 Crews（自主基于角色）和 Flows（事件驱动确定性），并解释文档的生产推荐。
- 使用 `@tool` 装饰器和 `BaseTool` 子类连接工具；推理结构化输出与自由文本的区别。
- 说出 CrewAI 的四种记忆类型以及每种类型何时发挥作用。
- 实现一个标准库三智能体团队（研究员、写手、编辑）来生成简报。
- 发现 CrewAI 的三种失败模式：提示膨胀、管理者 LLM 税、脆弱的交接。

## 问题

采用多智能体框架的团队都会遇到同样的难题。"自主协作"在演示中听起来很棒。然后客户提交了一个 bug，你需要确定性的回放。或者财务部门询问一个 LLM 路由的团队每次运行的成本。或者值班人员需要知道凌晨 3 点哪个智能体卡住了。

自由形式的 LLM 路由团队无法干净地回答这些问题。纯 DAG 可以回答所有这些问题，但失去了头脑风暴智能体所需的探索形态。

CrewAI 的拆分诚实地反映了这种权衡。Crews 用于协作式、基于角色、探索性的工作。Flows 用于事件驱动、代码拥有、可审计的生产。同一个框架，两种形态，按表面选择。

## 概念

### 四个原语

CrewAI 的表面很小。记住这些，其余的都是配置。

- **Agent。** `role + goal + backstory + tools + (optional) llm`。背景故事是承重的。它塑造语气、判断力、智能体何时停止。工具是智能体可以调用的函数（更多见下文）。
- **Task。** `description + expected_output + agent + (optional) context + (optional) output_pydantic`。一个可复用的工作单元。`expected_output` 是合同。`context` 列出上游任务，其输出被传入。`output_pydantic` 强制结构化形状。
- **Crew。** 容器。拥有 `agents` 列表、`tasks` 列表、`process` 以及可选的 `memory` + `verbose` + `manager_llm` 设置。
- **Process。** 执行策略。Sequential、Hierarchical、Consensus（计划中）。选择运行的形状。

智能体不直接看到彼此。任务引用智能体。Crew 对任务进行排序。Process 决定谁选择下一个任务。这就是整个心智模型。

> **针对 CrewAI 0.86（2026 年 5 月）验证。** 更新版本可能重命名或合并流程类型；在依赖特定形状前，请查看 [CrewAI Processes 文档](https://docs.crewai.com/concepts/processes)。

### Sequential vs Hierarchical vs Consensus

- **Sequential。** 任务按声明顺序运行。任务 N 的输出作为 `context` 提供给任务 N+1。成本最低。最可预测。当顺序固定时使用。
- **Hierarchical。** 一个管理者 Agent（单独的 LLM 调用）在专业智能体之间路由。CrewAI 从你的 `manager_llm` 配置或默认配置中生成管理者。管理者每轮选择下一个任务，可以拒绝或重新路由。当有四个或更多专业智能体且顺序确实取决于先前输出时使用。
- **Consensus。** 计划中，当前在公共 API 中未实现。文档保留该名称用于未来的基于投票的流程。今天不要依赖它。

Hierarchical 在每个专业调用之上增加了一个每轮 LLM 调用（管理者）。在五步运行中，token 成本可能翻三倍。仅在需要路由时为其付费。

### Crews vs Flows

这是文档在 2026 年首先介绍的框架。

- **Crew。** LLM 驱动的自主性。框架在运行时选择形状。适用于：研究、头脑风暴、初稿，以及任何路径本身就是答案一部分的场景。难以回放。难以测试。原型制作成本低廉。
- **Flow。** 你拥有的事件驱动图。`@start` 标记入口。`@listen(topic)` 标记一个步骤，当另一个步骤发出该主题时触发。每个步骤是纯 Python（可以在内部调用 Crew）。适用于：生产。可观测。可测试。确定性。

文档的 2026 年生产推荐：从 Flow 开始。当自主性值得其成本时，将 Crews 作为 `Crew.kickoff()` 调用从 Flow 步骤内部引入。Flow 提供审计轨迹，Crew 提供探索。组合，而非选择。

### 工具集成

三种方式为 Agent 提供工具。选择最简单适合的。

1. **`@tool` 装饰器。** 纯函数变成工具。签名是模式；文档字符串是 LLM 看到的描述。最适合一次性辅助工具。
2. **`BaseTool` 子类。** 基于类的工具，具有显式参数模式、异步支持、重试。当工具有状态（客户端、缓存）或需要结构化参数时使用。
3. **内置工具包。** CrewAI 提供第一方适配器：`SerperDevTool`、`FileReadTool`、`DirectoryReadTool`、`CodeInterpreterTool`、`RagTool`、`WebsiteSearchTool`。一次导入即可连接。

结构化输出使用 Pydantic。在 Task 上传递 `output_pydantic=MyModel`。CrewAI 根据模型验证 LLM 响应，要么强制转换要么重试。将此与紧凑的 `expected_output` 字符串配对。自由文本输出适用于草稿；结构化输出是下游 Flows 可以消费的。

### 记忆钩子

CrewAI 开箱即用提供四种记忆类型。它们可以组合：一个 Crew 可以同时启用所有四种。

> **针对 CrewAI 0.86（2026 年 5 月）验证。** 最近的版本将所有内容路由到统一的 `Memory` 系统，该系统包装了这四个存储。下面的概念模型仍然成立，但公共类表面在更新版本中可能会合并为单个 `Memory` 入口点；查看 [CrewAI 记忆文档](https://docs.crewai.com/concepts/memory) 了解当前 API。

- **短期。** 单次运行内的对话缓冲区。结束时清除。
- **长期。** 跨运行持久化。存储在向量数据库（默认 Chroma，可更换）中。通过与当前任务的相似性检索。
- **实体。** 每个实体的事实。"客户 X 在企业计划中。"按键而非相似性检索。跨运行持久化。
- **上下文。** 组装时检索。在智能体需要时拉取相关记忆，而非预加载。

在 Crew 上使用 `memory=True` 或按类型配置启用。由你配置的嵌入提供商支持（默认为 OpenAI，可切换为本地）。记忆是 CrewAI 相对于较薄框架的价值所在；纯 LangGraph 需要你自己连接这些。

### CrewAI 何时适合

- 三到六个具有命名角色和协作工作流的智能体。起草、审查、规划、头脑风暴。
- 路由中 LLM 关于下一步的判断是价值的一部分时（Hierarchical）。
- 当团队更愿意阅读 `role + goal + backstory` 而不是阅读图定义时。

### CrewAI 何时不适合

- 具有严格顺序的确定性 DAG。使用 LangGraph（第 13 课）。图形状是正确的抽象；CrewAI 的角色框架是摩擦。
- 亚秒级延迟预算。Hierarchical 增加了往返次数。即使是 Sequential 也会序列化包含背景故事和先前输出的提示。
- 单智能体循环。跳过框架；智能体循环（第 1 课）加工具注册表更短。

第 17 课（智能体框架权衡）以矩阵形式列出了这一点。简短版本：CrewAI 位于"基于角色的协作"角落。

### 依赖关系形态

独立于 LangChain。Python 3.10 到 3.13。使用 `uv`。星标数：请参见 [crewAIInc/crewAI](https://github.com/crewAIInc/crewAI)（截至 2026 年 5 月的快照）。AWS Bedrock 集成有文档记录；供应商基准测试报告与 LangGraph 相比在 QA 工作负载上有显著加速，但方法（数据集、硬件、评估指标）未公开，因此将供应商数字仅视为方向性参考。

### 这种模式出错的地方

- **背景故事导致的提示膨胀。** 每个智能体 2000 字的背景故事和一个五智能体团队在第一次工具调用前就燃烧了上下文预算。保持背景故事在 200 字以内。在智能体之间复用短语；不要重复五次相同的风格。
- **管理者 LLM token 税。** Hierarchical 流程在每个专业调用之前增加一次管理者 LLM 调用。在一个五任务团队中，这是六次 LLM 调用而不是五次，而且管理者调用携带完整的任务列表加先前输出。除非路由依赖于输出，否则切换到 Sequential。
- **脆弱的交接。** 任务 N 的 `expected_output` 是"一个大纲"。任务 N+1 将其作为 `context` 读取，并尝试解析三个部分。LLM 产生了四个。下游智能体即兴发挥。使用任务 N 上的 `output_pydantic` 修复，使任务 N+1 读取类型化对象而非自由文本。
- **Crew 作为生产。** 自由形式的 Crew 在没有 Flow 包装器的情况下交付到生产。输出可变性高；回放不可能；值班人员无法将差运行与好运行进行对比。用 Flow 包装。

## 构建

`code/main.py` 实现了两种形状的标准库版本和一个三智能体团队。

形状：

- 与 CrewAI 表面匹配的 `Agent`、`Task` 数据类。
- `SequentialCrew.kickoff(inputs)` 按声明顺序运行任务，将输出作为 `context` 传递。
- `HierarchicalCrew.kickoff(topic)` 添加一个管理者 Agent 每轮选择下一个专业智能体，遇到"done"时停止。
- 使用 `@start` 和 `@listen(topic)` 装饰器、一个小型事件循环和一个轨迹的 `Flow`。
- 镜像 CrewAI `@tool` 形状的 `tool(name)` 装饰器。
- 具有 `short_term`、`long_term`、`entity` 存储的 `Memory`；模拟相似度使用 numpy。
- 模拟的 LLM 响应是基于角色加输入前缀的硬编码字符串。无网络。确定性。

具体演示：研究员、写手、编辑团队制作关于"2026 年智能体工程"的简报。研究员拉取（模拟的）来源。写手起草。编辑收紧。相同团队通过 Flow 运行以显示确定性形状。

运行：

```bash
python3 code/main.py
```

轨迹涵盖：Sequential 团队通过 `context` 传递输出、Hierarchical 团队由管理者选择（研究员、写手、编辑，然后"done"）、使用显式主题（`researched`、`drafted`、`edited`）运行相同三个步骤的 Flow、通过 `@tool` 路由的工具调用、以及跨两次启动持久化的长期记忆。

Crew 轨迹是流动的；管理者原则上可以重新排序。Flow 轨迹是固定的。这种选择就是本课的要点。

## 使用

- **CrewAI Flow** 用于生产。即使 Flow 只是一个调用 `Crew.kickoff()` 的步骤。Flow 提供了审计边界。
- **CrewAI Crew（Sequential）** 用于明确顺序的协作工作，特别是初稿和审查循环。
- **CrewAI Crew（Hierarchical）** 当路由依赖于输出且有四个或更多专业智能体时。
- **LangGraph**（第 13 课）用于显式状态机、持久恢复、严格排序。
- **AutoGen v0.4**（第 14 课）用于参与者模型并发和故障隔离。
- **OpenAI Agents SDK**（第 16 课）用于 OpenAI 优先的产品，带交接和护栏。
- **Claude Agent SDK**（第 17 课）用于 Claude 优先的产品，带子智能体和会话存储。

## 交付

`outputs/skill-crew-or-flow.md` 为任务选择 Crew 或 Flow，并搭建最小实现。硬性拒绝没有背景故事的 Crew、没有明确主题的 Flow、以及少于三个专业智能体的 Hierarchical。

## 陷阱

- **背景故事作为调味品。** 它塑造输出。为每个智能体测试三个变体；差异是真实存在的。选择一个，固定它。
- **跳过 `expected_output`。** 没有每任务的合同，下游任务会接收 LLM 产生的任何内容。Crew 运行；审计失败。
- **记忆始终开启。** 长期记忆每次运行都写入。向量数据库增长。检索变嘈杂。将写入范围限定在事实是持久化的任务上。
- **管理者提示漂移。** Hierarchical 的管理者提示是隐式的。如果路由变得奇怪，在详细模式下转储并阅读。
- **Crew 中的工具副作用。** Crew 可以比预期更频繁地调用工具。POST、DELETE、支付属于 Flow 步骤，永远不是 Crew 工具。

## 练习

1. 将 Sequential crew 转换为 Flow。计数可变性下降的接触点。注意可读性下降的地方。
2. 为团队添加实体记忆：关于客户的事实跨启动持久化。验证检索拉取正确的实体。
3. 实现 Hierarchical 流程，其中管理者拒绝路由到编辑，直到写手的输出至少有三个段落。追踪重试。
4. 为（模拟的）网络搜索连接 `BaseTool` 子类。比较轨迹形状与 `@tool` 装饰器版本。
5. 为编辑任务添加 `output_pydantic=Brief`，其中 `Brief` 有 `title`、`summary`、`sections`。让写手任务一次输出格式错误的 JSON；验证轨迹中 CrewAI 的重试行为。
6. 阅读 CrewAI 的文档介绍。将玩具移植到真实的 `crewai` API。标准库版本跳过了哪些保证？
7. 将 AgentOps 或 Langfuse（第 24 课）连接到真实运行。标准库版本中你错过了哪些轨迹？

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------------|------------------------|
| Agent | "人物角色" | 角色 + 目标 + 背景故事 + 工具 |
| Task | "工作单元" | 描述 + 预期输出 + 负责人 + 可选结构化输出 |
| Crew | "智能体团队" | Agent + Task + Process 的容器 |
| Process | "执行策略" | Sequential / Hierarchical / Consensus（计划中） |
| Flow | "确定性工作流" | 事件驱动、代码拥有、可测试 |
| Backstory | "人物角色提示" | Agent 的语气和判断力塑造器 |
| `@tool` | "函数工具" | 将函数变成 Agent 可调用工具的装饰器 |
| `BaseTool` | "类工具" | 基于类的工具，带参数模式、重试、异步支持 |
| 实体记忆 | "每个实体的事实" | 限定在客户/账户/问题范围内的记忆 |
| 长期记忆 | "跨运行记忆" | 向量支持的记忆，在启动之间持久化 |
| 上下文记忆 | "即时检索" | 在 Agent 需要时拉取的记忆 |
| 管理者 LLM | "路由器智能体" | Hierarchical 流程中额外选择下一个任务的 LLM |
| `expected_output` | "任务合同" | 告诉 Agent（和审计）返回什么形状的字符串 |

## 延伸阅读

- [CrewAI 文档介绍](https://docs.crewai.com/en/introduction)：概念和推荐的生产路径
- [CrewAI Flows 指南](https://docs.crewai.com/en/concepts/flows)：事件驱动形状、`@start`、`@listen`
- [CrewAI 工具参考](https://docs.crewai.com/en/concepts/tools)：`@tool`、`BaseTool`、内置工具包
- [CrewAI 记忆](https://docs.crewai.com/en/concepts/memory)：短期、长期、实体、上下文
- [Anthropic，构建有效的智能体](https://www.anthropic.com/research/building-effective-agents)：多智能体何时有帮助，何时没有
- [LangGraph 概述](https://docs.langchain.com/oss/python/langgraph/overview)：状态机替代方案
