# 长时间运行的后台代理：持久化执行

> 生产环境中的长周期代理不会在 `while True` 中运行。每个 LLM 调用都成为一个带有检查点、重试和重放的活动。Temporal 的 OpenAI Agents SDK 集成于 2026 年 3 月正式发布（GA）。Claude Code Routines（Anthropic）运行定时调度的 Claude Code 调用，无需持久的本地进程。会话在等待人工输入时暂停、在部署后存活，并从由 `thread_id` 键控的最新检查点恢复。在新的易用性背后是一个古老的模式 —— 工作流编排 —— 加上一个新的输入：LLM 调用作为非确定性活动，在恢复时必须被确定性重放。

**类型：** 学习
**语言：** Python（stdlib，最小持久化执行状态机）
**前置要求：** 第 15 阶段 · 10（权限模式），第 15 阶段 · 01（长周期代理）
**时间：** ~60 分钟

## 问题

考虑一个运行四小时的代理。它调用了三个工具，两次提示用户，并进行了四十次 LLM 调用。运行到一半时，它所在的宿主机重启了。会发生什么？

- 在幼稚的 `while True` 循环中：一切丢失。运行从头开始重启。三个工具调用（带有真实的副作用）再次执行。用户再次被提示他们已经批准过的事情。四十次 LLM 调用被重新计费。
- 使用持久化执行：运行从最近的检查点恢复。已经完成的活动不会重新执行；它们的结果从持久化日志中重放。用户不会重新批准他们已经批准过的事情。已经完成的 LLM 调用不会被重新计费。

这是工作流引擎已经提供十年的相同模式（Temporal、Cadence、Uber 的 Cherami）。新的是 LLM 调用现在成为一种活动 —— 非确定性、昂贵、带有副作用 —— 并且它们干净地适配这个模式。

本课的主线：长周期可靠性会衰减（METR 观察到"35 分钟退化" —— 成功率大致随周期长度呈二次方下降）。持久化执行使得运行时间可以超过可靠性曲线所支持的范围，这是一种新的方式 —— 如果设计正确则安全地失败，如果设计错误则不安全地失败。

## 概念

### 活动、工作流和重放

- **工作流（Workflow）**：确定性编排代码。定义活动的序列、分支和等待。必须是确定性的，以便可以从事件日志中重放而不会出现令人惊讶的分歧。
- **活动（Activity）**：一个非确定性、可能失败的单元工作。LLM 调用、工具调用、文件写入、HTTP 请求。每个活动都记录其输入和（完成后）输出。
- **事件日志（Event log）**：持久化后端存储。每个活动的开始、完成、失败、重试以及每个工作流决策都被记录。
- **重放（Replay）**：恢复时，工作流代码从头开始重新运行；每个已经完成的活动返回其记录的结果而不重新执行。只有尚未完成的活动才会实际运行。

这与 React 针对虚拟 DOM 重新渲染、或 Git 从提交重建工作树是相同的形状。编排器中的确定性是使持久化变得廉价的原因。

### 为什么 LLM 调用适配这个模式

LLM 调用是：
- 非确定性的（temperature > 0；即使 temperature 为 0，也会随模型版本漂移）。
- 昂贵的（金钱和延迟）。
- 可能失败的（速率限制、超时）。
- 有副作用的（如果它们调用工具）。

这正是活动的画像。将每个 LLM 调用包装为一个活动，就获得了带指数退避的重试、跨重启的检查点，以及用于调试的可重放轨迹。

### 由 `thread_id` 键控的检查点

LangGraph、Microsoft Agent Framework、Cloudflare Durable Objects 和 Claude Code Routines 都收敛到相同的 API 形状：一个 `thread_id`（或等效物）标识会话；每个状态转换持久化到后端（PostgreSQL 为默认，SQLite 用于开发，Redis 用于缓存）；恢复读取最新的检查点。

后端选择很重要：

- **PostgreSQL**：持久化、可查询、在部署后存活。LangGraph 的默认选择。
- **SQLite**：仅限本地开发；跨主机丢失数据。
- **Redis**：快速但为临时性，除非配置了 AOF/快照。
- **Cloudflare Durable Objects**：透明分布式；由唯一键限定范围；存活数小时到数周。

### 人工输入作为一等状态

提议-然后-提交（第 15 课）需要一个持久的"等待人工"状态。工作流暂停，外部队列持有待处理的请求，审批从该确切点恢复。没有持久化，这是尽力而为；有了它，隔夜的审批到达后，工作流在早上继续执行。

### 35 分钟退化

METR 观察到，每个被测量的代理类别在连续运行约 35 分钟后都显示出可靠性衰减。任务持续时间翻倍，失败率大约翻两番。持久化执行并不能解决这个问题；它让你运行的时间超过可靠性曲线所支持的范围。安全的模式是将持久化与检查点结合使用，这些检查点在重新进入时需要新的人工介入，并与预算终止开关（第 13 课）结合，以限制总计算量，无论挂钟时间如何。

### 何时持久化执行是错误的答案

- 运行时间短于几分钟且无需人工输入。开销大于收益。
- 严格的只读信息检索。
- 正确性要求在一个上下文窗口内端到端完成的任务（某些推理任务；某些一次性生成）。

```figure
memory-consolidation
```

## 使用它

`code/main.py` 使用 stdlib Python 实现了一个最小的持久化执行引擎。它支持：

- `@activity` 装饰器，将输入和输出记录到 JSON 事件日志中。
- 一个编排活动序列的工作流函数。
- 一个 `run_or_replay(workflow, event_log)` 函数，重放已完成的活动而不重新执行它们。

驱动程序模拟一个三活动工作流，中途崩溃，并展示 (a) 幼稚重试重新执行一切与 (b) 重放仅运行缺失活动之间的区别。

## 交付它

`outputs/skill-durable-execution-review.md` 审查一个提议的长时间运行代理部署的正确持久化执行形态：活动、确定性、检查点后端、人工输入状态，以及恢复时人工介入策略。

## 练习

1. 运行 `code/main.py`。观察幼稚重试和重放之间活动执行次数的差异。更改崩溃点，并展示重放计数相应变化。

2. 将玩具引擎转换为显式使用 `thread_id`。模拟两个并发会话共享同一个引擎，并确认它们的事件日志不会冲突。

3. 在玩具引擎中选取一个活动。引入一个非确定性（工作流决策中的挂钟时间戳）。演示重放时的分歧。解释真实引擎如何处理这个问题（副作用注册、`Workflow.now()` API）。

4. 阅读 LangChain 的"Runtime behind production deep agents"文章。列出运行时持久化的每个状态，并说明每个状态覆盖哪种失败模式。

5. 为一个 6 小时的自主编码任务设计一个检查点策略。你在哪里设置检查点？崩溃后恢复看起来是什么样的？哪些需要新的人工介入？

## 关键术语

| 术语 | 人们说的 | 实际含义 |
|---|---|---|
| 工作流 (Workflow) | "代理的脚本" | 确定性编排代码；可从事件日志重放 |
| 活动 (Activity) | "一个步骤" | 非确定性单元（LLM 调用、工具调用）；前后均记录 |
| 事件日志 (Event log) | "后端存储" | 每个状态转换的持久化记录 |
| 重放 (Replay) | "恢复" | 重新运行工作流；已完成的活动返回记录的结果而不重新执行 |
| 检查点 (Checkpoint) | "保存点" | 由 thread_id 键控的持久化状态；恢复时取最新 |
| thread_id | "会话键" | 限定持久化状态范围的标识符 |
| 35 分钟退化 (35-minute degradation) | "可靠性衰减" | METR：成功率随周期长度大致呈二次方下降 |
| 非确定性 (Non-determinism) | "重放时的漂移" | 挂钟时间、随机数、LLM 输出；必须注册为副作用 |

## 延伸阅读

- [Anthropic — Claude Code Agent SDK: agent loop](https://code.claude.com/docs/en/agent-sdk/agent-loop) —— 预算、回合数和恢复语义。
- [Microsoft — Agent Framework: human-in-the-loop and checkpointing](https://learn.microsoft.com/en-us/agent-framework/workflows/human-in-the-loop) —— RequestInfoEvent 形状。
- [LangChain — The Runtime Behind Production Deep Agents](https://www.langchain.com/conceptual-guides/runtime-behind-production-deep-agents) —— 具体的运行时需求。
- [OpenAI Agents SDK + Temporal integration (Trigger.dev announcement)](https://trigger.dev) —— LLM 调用的活动形状。
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) —— 35 分钟退化参考。