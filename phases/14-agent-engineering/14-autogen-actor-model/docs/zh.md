# AutoGen v0.4：参与者模型与智能体框架

> AutoGen v0.4（微软研究院，2025 年 1 月）围绕参与者模型重新设计了智能体编排。异步消息交换、事件驱动智能体、故障隔离、天然并发。该框架现已进入维护模式，而 Microsoft Agent Framework（2025 年 10 月公开预览）将成为继任者。

**类型：** Learn + Build
**语言：** Python（标准库）
**前置知识：** Phase 14 · 01（智能体循环），Phase 14 · 12（工作流模式）
**时间：** ~75 分钟

## 学习目标

- 描述参与者模型：智能体作为参与者，消息作为唯一的 IPC，每个参与者故障隔离。
- 说出 AutoGen v0.4 的三个 API 层——Core、AgentChat、Extensions——以及各层的用途。
- 解释为什么将消息传递与处理解耦能带来故障隔离和天然并发。
- 在标准库中实现一个参与者运行时，并将一个双智能体代码审查流程移植到其上。

## 问题

大多数智能体框架是同步的：一个智能体生产，一个智能体消费，在调用栈中。故障会使栈崩溃。并发是后加的。分布需要重写。

AutoGen v0.4 的答案：参与者模型。每个智能体是一个带有私有收件箱的参与者。消息是唯一的交互方式。运行时将传递与处理解耦。故障隔离到一个参与者。并发是天生的。分布只是不同的传输层。

## 概念

### 参与者

参与者拥有：

- 私有状态（从外部永远不能直接触及）。
- 收件箱（消息队列）。
- 处理程序：`receive(message) -> effects`，其中 effects 可以是"回复"、"发送给其他参与者"、"生成新参与者"、"更新状态"、"停止自身"。

两个参与者不能共享内存。它们只能发送消息。

### AutoGen v0.4 的三个 API 层

1. **Core。** 低级参与者框架。`AgentRuntime`、`Agent`、`Message`、`Topic`。异步消息交换、事件驱动。
2. **AgentChat。** 任务驱动的高级 API（替代 v0.2 的 ConversableAgent）。`AssistantAgent`、`UserProxyAgent`、`RoundRobinGroupChat`、`SelectorGroupChat`。
3. **Extensions。** 集成——OpenAI、Anthropic、Azure、工具、记忆。

### 为什么解耦很重要

在 v0.2 模型中，调用 `agent_a.chat(agent_b)` 会同步阻塞 agent_a 直到 agent_b 返回。在 v0.4 中，`send(agent_b, msg)` 将消息放入 agent_b 的收件箱并返回。运行时会后传递。三个结果：

- **故障隔离。** 智能体 B 崩溃不会导致智能体 A 崩溃——运行时捕获 B 的处理程序中的故障并决定做什么（记录、重试、死信）。
- **天然并发。** 同时有多个消息在传输中；参与者并发处理其收件箱。
- **分发就绪。** 无论参与者是在进程内还是在另一台主机上，收件箱 + 传输是相同的抽象。

### 拓扑

- **RoundRobinGroupChat。** 智能体按固定轮换顺序轮流。
- **SelectorGroupChat。** 选择器智能体根据对话上下文选择下一个发言者。
- **Magentic-One。** 用于网页浏览、代码执行、文件处理的参考多智能体团队。基于 AgentChat 构建。

### 可观测性

内置 OpenTelemetry 支持。每条消息发出一个跨度；工具调用按照 2026 年 OTel GenAI 语义约定（第 23 课）携带 `gen_ai.*` 属性。

### 状态：维护模式

2026 年初：AutoGen v0.7.x 稳定，适用于研究和原型设计。微软已将活跃开发转移到 Microsoft Agent Framework（2025 年 10 月 1 日公开预览；1.0 GA 目标为 2026 年第一季度末）。AutoGen 模式可干净地向前移植——参与者模型是持久化的思想。

## 构建

`code/main.py` 实现一个标准库参与者运行时：

- `Message`——带有 `sender`、`recipient`、`topic`、`body` 的类型化负载。
- `Actor`——带有 `receive(message, runtime)` 的抽象类。
- `Runtime`——带有共享队列、传递、故障隔离的事件循环。
- 一个双参与者演示：`ReviewerAgent` 审查代码，`ChecklistAgent` 运行检查清单；它们交换消息直到达成共识。

运行：

```
python3 code/main.py
```

轨迹显示消息传递、一个参与者中的模拟故障不会导致另一个参与者崩溃、以及收敛到共享判定。

## 使用

- **AutoGen v0.4/v0.7**（维护）——稳定，适用于研究、原型设计、多智能体模式。
- **Microsoft Agent Framework**（公开预览）——前进路径；相同的参与者模型思想在更新的 API 中。
- **LangGraph 群体拓扑**（第 13 课）——通过共享工具交接的类似模式。
- **自定义参与者运行时**——当你需要特定的传输层（NATS、RabbitMQ、gRPC）时。

## 交付

`outputs/skill-actor-runtime.md` 生成一个最小参与者运行时以及一个针对给定多智能体任务的团队模板（RoundRobin 或 Selector）。

## 练习

1. 添加死信队列：当处理程序抛出异常时，将失败的消息搁置供人工检查。在你的玩具中，DLQ 被击中的频率如何？
2. 实现 `SelectorGroupChat`：选择器参与者根据对话状态选择谁处理下一条消息。
3. 添加分布式传输：将进程内队列替换为 JSON-over-HTTP 服务器，使参与者可以在单独的进程中运行。
4. 每条消息连接一个 OTel 跨度（或空操作替代）。按照第 23 课发出 `gen_ai.agent.name`、`gen_ai.operation.name`。
5. 阅读 AutoGen v0.4 的架构文章。将你的玩具移植到真实的 `autogen_core` API。你跳过了哪些在生产中重要的内容？

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------------|------------------------|
| 参与者 | "智能体" | 私有状态 + 收件箱 + 处理程序；无共享内存 |
| 消息 | "事件" | 类型化负载；参与者交互的唯一方式 |
| 收件箱 | "邮箱" | 每个参与者待处理消息的队列 |
| 运行时 | "智能体主机" | 路由消息和隔离故障的事件循环 |
| 主题 | "信道" | 参与者之间的命名发布-订阅路由 |
| 故障隔离 | "让它崩溃" | 一个参与者失败不会导致其他参与者崩溃 |
| RoundRobinGroupChat | "固定轮换团队" | 智能体按顺序轮流 |
| SelectorGroupChat | "上下文路由团队" | 选择器选择下一个发言者 |
| Magentic-One | "参考团队" | 用于网页 + 代码 + 文件的多智能体团队 |

## 延伸阅读

- [AutoGen v0.4，微软研究院](https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/) — 重新设计文章
- [LangGraph 概述](https://docs.langchain.com/oss/python/langgraph/overview) — 图形状的替代方案
- [OpenTelemetry GenAI 语义约定](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — AutoGen 默认发出的跨度
