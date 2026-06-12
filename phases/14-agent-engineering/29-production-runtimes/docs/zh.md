# 生产运行时：队列、事件、定时任务 (Production Runtimes: Queue, Event, Cron)

> 生产环境中的 Agent 运行在六种运行时形态之上：请求-响应、流式、持久化执行、基于队列的后台任务、事件驱动和定时调度。在选择框架之前，先选对形态。可观测性 (Observability) 在每种形态下都是承重结构。

**类型：** Learn
**语言：** Python (stdlib)
**前置知识：** Phase 14 · 13 (LangGraph), Phase 14 · 22 (Voice)
**时间：** ~60 分钟

## 学习目标

- 列举六种生产运行时形态，并将每种形态对应到框架/产品模式。
- 解释为什么持久化执行 (Durable Execution) 对长周期任务至关重要。
- 描述事件驱动运行时以及 Claude Managed Agents 的适用场景。
- 解释"可观测性即承重结构"这一论断对多步骤 Agent 的意义。

## 问题

生产环境中的 Agent 会以 Jupyter Notebook 无法暴露的方式失败：第 37 步网络超时、用户中途挂断语音通话、定时任务因机器重启而终止、后台工作进程内存耗尽。运行时形态决定了哪些失败是可以恢复的。

## 概念

### 请求-响应 (Request-response)

- 同步 HTTP。用户等待完成。
- 仅适用于短任务（<30 秒）。
- 技术栈：Agno (Python + FastAPI)、Mastra (TypeScript + Express/Hono/Fastify/Koa)。
- 可观测性：标准 HTTP 访问日志 + OTel spans。

### 流式 (Streaming)

- 使用 SSE 或 WebSocket 实现渐进式输出。
- LiveKit 将其扩展到 WebRTC 用于语音/视频（第 22 课）。
- 技术栈：任何支持流式的框架 + 处理 SSE/WS 的前端。
- 可观测性：每块数据的耗时、首 token 延迟、尾延迟。

### 持久化执行 (Durable execution)

- 每一步后检查点保存状态；失败时自动恢复。
- AutoGen v0.4 的 actor 模型将故障隔离到单个 Agent（第 14 课）。
- LangGraph 的核心差异化能力（第 13 课）。
- 当步骤数未知且恢复成本高时至关重要。

### 基于队列 / 后台 (Queue-based / background)

- 任务进入队列，工作进程取走执行，结果通过 webhook 或 pub/sub 回流。
- 对长周期 Agent 至关重要（每个任务数十到数百步，参见 Anthropic 的 computer use 公告）。
- 技术栈：Celery (Python)、BullMQ (Node)、SQS + Lambda (AWS)、自定义。
- 可观测性：队列深度、每任务延迟分布、DLQ 大小。

### 事件驱动 (Event-driven)

- Agent 订阅触发器：新邮件、PR 打开、定时触发。
- Claude Managed Agents 开箱即用（第 17 课）。
- CrewAI Flows（第 15 课）构建事件驱动的确定性工作流。
- 可观测性：触发源、事件到启动的延迟、Agent 延迟。

### 定时调度 (Scheduled)

- 以 Cron 形态定期运行的 Agent。
- 结合持久化执行，使失败的夜间运行能在下一个 tick 恢复。
- 技术栈：Kubernetes CronJob + 持久化框架；托管服务（Render cron、Vercel cron）。

### 2026 年部署模式

- **CrewAI Flows** 用于事件驱动的生产环境。
- **Agno** 无状态 FastAPI 用于 Python 微服务。
- **Mastra** 服务器适配器（Express、Hono、Fastify、Koa）用于嵌入。
- **Pipecat Cloud / LiveKit Cloud** 用于托管语音（第 22 课）。
- **Claude Managed Agents** 用于托管的长时间运行异步任务。

### 可观测性是承重结构

没有 OpenTelemetry GenAI spans（第 23 课）加上 Langfuse/Phoenix/Opik 后端（第 24 课），你无法调试一个在第 40 步失败的多步骤 Agent。这对生产环境来说不是可选项。这是"快速调试"和"从头重放并加更多日志"之间的区别。

### 生产运行时常见的失败点

- **形态选择错误。** 为一个 5 分钟的任务选择请求-响应模式。用户挂断；工作进程堆积；重试叠加。
- **没有 DLQ。** 队列工作进程没有死信队列。失败的任务消失无踪。
- **不透明的后台任务。** 后台 Agent 运行没有 trace 导出。失败在用户报告之前完全不可见。
- **跳过持久化状态。** 任何超过 30 秒且无法承受重启代价的运行都需要持久化执行。

## 动手构建

`code/main.py` 是一个基于 stdlib 的多形态演示：

- 请求-响应端点（普通函数）。
- 流式处理器（生成器）。
- 带 DLQ 的基于队列的工作进程。
- 事件触发器注册表。
- 定时任务调度器。

运行方式：

```bash
python3 code/main.py
```

输出：五个 trace，展示同一任务在每种形态下的行为。相同的 Agent 逻辑，不同的外部外壳。持久化执行（第六种形态）有意留到第 13 课结合 LangGraph 检查点机制讲解。

## 使用场景

- **请求-响应** 用于聊天式 UX。
- **流式** 用于渐进式响应。
- **持久化** 用于长周期任务。
- **队列** 用于批处理 / 异步 / 长时间运行。
- **事件** 用于 Agent 的响应式行为。
- **定时任务** 用于运维（记忆整合、评估、成本报告）。

## 交付物

`outputs/skill-runtime-shape.md` 为任务选择合适的运行时形态并配置可观测性需求。

## 练习

1. 将你在第 01 课实现的 ReAct 循环移植到所有六种形态中。哪种形态适合哪种产品场景？
2. 为基于队列的演示添加 DLQ。模拟 10% 的任务失败率；展示 DLQ 大小。
3. 编写一个定时触发的评估 Agent，每晚对你当天排名前 20 的 trace 运行评估。
4. 实现带背压的流式处理：如果客户端速度慢，暂停 Agent。这与轮次预算 (turn budget) 如何交互？
5. 阅读 Claude Managed Agents 文档。什么情况下你会将自托管的长周期 Agent 迁移到托管服务？

## 关键术语

| 术语 | 人们说的 | 实际含义 |
|------|----------|----------|
| Request-response | "同步" | 用户等待；仅限短任务 |
| Streaming | "SSE / WS" | 渐进式输出；更好的 UX；每块数据可观测延迟 |
| Durable execution | "从失败中恢复" | 检查点保存状态；从最后一步重启 |
| Queue-based | "后台任务" | 生产者 / 工作进程池 / DLQ |
| Event-driven | "基于触发器" | Agent 响应外部事件 |
| DLQ | "死信队列" | 失败任务的停车场 |
| Claude Managed Agents | "托管运行环境" | Anthropic 托管的长时间运行异步任务，带缓存和压缩 |

## 延伸阅读

- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) — 持久化执行详情
- [Claude Managed Agents overview](https://platform.claude.com/docs/en/managed-agents/overview) — 托管的长时间运行异步任务
- [Anthropic, Introducing computer use](https://www.anthropic.com/news/3-5-models-and-computer-use) — "每个任务数十到数百步"
- [AutoGen v0.4 (Microsoft Research)](https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/) — actor 模型故障隔离