# 交接与例程——无状态编排 (Handoffs and Routines — Stateless Orchestration)

> OpenAI 的 Swarm（2024 年 10 月）将多代理编排提炼为两个原语：**例程**(routines)（作为系统提示的指令 + 工具）和**交接**(handoffs)（一个返回另一个 Agent 的工具）。没有状态机，没有分支 DSL——LLM 通过调用正确的交接工具来路由。OpenAI Agents SDK（2025 年 3 月）是其生产级继任者。Swarm 本身仍然是最清晰的概念参考——其全部源代码只有几百行。这种模式之所以流行，是因为 API 表面大致就是"agent = prompt + tools; handoff = function returning agent"。局限性：无状态，所以内存是调用者的问题。

**类型:** Learn + Build
**语言:** Python (stdlib)
**前置知识:** Phase 16 · 04 (原始模型)
**时间:** ~60 分钟

## 问题 (Problem)

每个多代理框架都希望你学习它的 DSL：LangGraph 的节点和边，CrewAI 的 crew 和 task，AutoGen 的 GroupChat 和 manager。这些 DSL 是真正的抽象，但它们让事情看起来比实际需要的更重。

Swarm 朝相反的方向推进：使用模型已经拥有的工具调用能力。交接变成了工具调用。编排者是当前持有对话的任何一个代理。状态机隐含在代理的系统提示中。

## 概念 (Concept)

### 两个原语 (Two primitives)

**例程(Routine)。** 一个定义代理角色和可用工具的系统提示。把它想象成一个限定范围的指令集："你是一个分诊代理；如果用户问退款，交接给退款代理。"

**交接(Handoff)。** 代理可以调用的一个工具，它返回一个新的 Agent 对象。Swarm 运行时检测到 Agent 返回值，并在下一个轮次切换活跃代理。

这就是整个抽象。

```
def transfer_to_refunds():
    return refund_agent  # Swarm sees Agent return → switch active agent

triage_agent = Agent(
    name="triage",
    instructions="Route the user to the right specialist.",
    functions=[transfer_to_refunds, transfer_to_sales, transfer_to_support],
)
```

分诊代理的系统提示使其根据用户消息选择正确的交接。LLM 的工具调用完成路由。

### 为什么流行 (Why it is viral)

- **小 API(Small API)。** 只需学习两个概念。
- **使用模型已有的能力(Uses what the model already does)。** 工具调用在各提供商中已经是生产级。
- **无需状态机负担(No state-machine burden)。** 你不需要描述图；代理的提示描述了它们交接给谁。

### 无状态的权衡 (The stateless trade)

Swarm 在运行之间是显式无状态的。框架在一次运行期间保留消息历史，但不持久化任何东西。内存、连续性、长时间运行的任务——都是调用者的问题。

在生产环境中（OpenAI Agents SDK，2025 年 3 月），这是主要变化之一：SDK 在保留交接原语的同时，增加了内置的会话管理、护栏(guardrails)和追踪(tracing)。

### Swarm/交接何时适用 (When Swarm/handoffs fit)

- **分诊模式(Triage patterns)。** 一线代理将用户路由给专家。
- **基于技能的交接(Skill-based handoffs)。** "如果任务需要代码，调用编码者；如果需要研究，调用研究者。"
- **简短、有边界的对话(Short, bounded conversations)。** 客户支持、FAQ 转工单、简单工作流。

### Swarm 何时困难 (When Swarm struggles)

- **需要共享内存的长会话(Long sessions with shared memory)。** 交接将对话状态重置为新代理的提示加上历史。没有调用者管理的内存，就没有跨代理的持久状态。
- **并行执行(Parallel execution)。** 交接是一次一个——活跃代理切换。并行需要调用者编排多个 Swarm 运行。
- **审计和重放(Audit and replay)。** 无状态运行难以精确重放；LLM 的交接选择不是确定性的。

### OpenAI Agents SDK (2025 年 3 月)

生产级继任者增加了：

- **会话状态(Session state)。** 跨运行的持久线程。
- **护栏(Guardrails)。** 输入/输出验证钩子。
- **追踪(Tracing)。** 每次工具调用和交接都被记录。
- **交接过滤器(Handoff filters)。** 控制在交接时传递哪些上下文。

交接原语得以保留；生产级人体工程学被添加在周围。

### Swarm vs GroupChat

两者都使用 LLM 驱动的路由，但它们在**谁选择下一个**上有所不同：

- GroupChat：一个选择器（函数或 LLM）从外部选择下一个发言者。
- Swarm：当前代理通过调用交接工具来选择其继任者。

Swarm 是"代理决定下一步"；GroupChat 是"管理器决定下一步"。Swarm 的决策存在于活跃代理的工具调用中；GroupChat 的决策存在于 `GroupChatManager` 中。

## 构建 (Build It)

`code/main.py` 从头实现 Swarm：一个 Agent 数据类、一个交接机制（工具返回 Agent）和一个检测代理切换的运行循环。

演示：一个分诊代理路由到退款、销售或支持专家。每个专家有自己的工具。运行循环打印每次交接。

运行：

```
python3 code/main.py
```

## 使用 (Use It)

`outputs/skill-handoff-designer.md` 为给定任务设计一个交接拓扑：存在哪些代理，它们可以调用哪些交接，传递哪些上下文。

## 交付 (Ship It)

检查清单：

- **交接日志(Handoff logging)。** 每次交接写入一个追踪事件，包含来源代理、目标代理、上下文快照。
- **上下文传递规则(Context transfer rules)。** 决定交接时传递什么：完整历史（昂贵）、最近 N 条消息或摘要。
- **交接护栏(Guardrail on handoff)。** 交接给具有不同工具权限的专家时必须经过认证——否则提示注入可能强制进行非预期的交接。
- **循环检测(Loop detection)。** 两个代理来回交接是常见故障；使用简单的最近 K 环检查来检测。
- **回退代理(Fallback agent)。** 如果交接目标不存在，回退到安全的默认值。

## 练习 (Exercises)

1. 运行 `code/main.py`，分诊到退款代理。确认第二轮活跃代理是 refund。
2. 添加一个循环检测规则：如果相同的两个代理连续交接 3 次，强制退出。设计回退方案。
3. 阅读 OpenAI Agents SDK 文档中关于交接过滤器的部分。实现一个"交接时摘要"版本：传出代理在传入代理接管之前将上下文压缩为要点摘要。
4. 比较 Swarm 交接与 GroupChatManager 选择器。哪种模式使提示注入更严重，为什么？
5. 阅读 Swarm cookbook (https://developers.openai.com/cookbook/examples/orchestrating_agents)。找出 Swarm 做出的一个显式设计决策，以及 OpenAI Agents SDK 是改变了还是保留了它。

## 关键术语 (Key Terms)

| 术语 | 人们说的 | 实际含义 |
|------|----------|----------|
| Routine (例程) | "代理提示" | 系统提示 + 工具列表。定义角色和可用的交接。 |
| Handoff (交接) | "转移到另一个代理" | 活跃代理可以调用的一个工具，返回一个新的 Agent。运行时切换活跃代理。 |
| Stateless (无状态) | "运行之间没有内存" | Swarm 不持久化任何东西；内存是调用者的责任。 |
| Active agent (活跃代理) | "谁在说话" | 当前持有对话的代理。交接改变这个值。 |
| Context transfer (上下文传递) | "交接时传递什么" | 传入代理看到什么历史的策略：全部、最近 N 条或摘要。 |
| Handoff loop (交接循环) | "代理乒乓" | 两个代理不断互相交接的失败模式。 |
| OpenAI Agents SDK | "生产级 Swarm" | 2025 年 3 月继任者；在交接原语之上增加了会话、护栏、追踪。 |
| Handoff filter (交接过滤器) | "交接时的门控" | SDK 特性，用于在交接边界检查和修改上下文。 |

## 延伸阅读 (Further Reading)

- [OpenAI cookbook — Orchestrating Agents: Routines and Handoffs](https://developers.openai.com/cookbook/examples/orchestrating_agents) — 参考阐述
- [OpenAI Swarm repo](https://github.com/openai/swarm) — 原始实现，保留为概念参考
- [OpenAI Agents SDK docs](https://openai.github.io/openai-agents-python/) — 生产级继任者，包含会话和追踪
- [Anthropic handoff-in-Claude notes](https://docs.anthropic.com/en/docs/claude-code) — Claude Code 子代理如何通过 `Task` 使用类似交接的模式