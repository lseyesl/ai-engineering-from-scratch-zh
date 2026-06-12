# OpenAI Agents SDK：交接、护栏、追踪

> OpenAI Agents SDK 是构建在 Responses API 上的轻量级多智能体框架。五个原语：Agent、Handoff、Guardrail、Session、Tracing。交接是名为 `transfer_to_<agent>` 的工具。护栏在输入或输出上触发。追踪默认开启。

**类型：** Learn + Build
**语言：** Python（标准库）
**前置知识：** Phase 14 · 01（智能体循环），Phase 14 · 06（工具使用）
**时间：** ~75 分钟

## 学习目标

- 说出 OpenAI Agents SDK 的五个原语。
- 解释交接：为什么它们被建模为工具，模型看到什么名称形状，以及上下文如何转移。
- 区分输入护栏、输出护栏和工具护栏；解释 `run_in_parallel` 与阻塞模式。
- 实现一个带有交接 + 护栏 + 跨度风格追踪的标准库运行时。

## 问题

不能干净委派的智能体最终把所有内容塞进一个提示中。没有护栏的智能体会泄露 PII、输出违反策略的内容，或永远循环。OpenAI 的 SDK 将三个使多智能体工作可行的原语编码化。

## 概念

### 五个原语

1. **Agent。** LLM + 指令 + 工具 + 交接。
2. **Handoff。** 委派给另一个智能体。对模型来说，它呈现为名为 `transfer_to_<agent_name>` 的工具。
3. **Guardrail。** 对输入（仅第一个智能体）、输出（仅最后一个智能体）或工具调用（每个函数工具）的验证。
4. **Session。** 跨轮次的自动对话历史。
5. **Tracing。** LLM 生成、工具调用、交接、护栏的内置跨度。

### 交接作为工具

模型在其工具列表中看到 `transfer_to_billing_agent`。调用它向运行时发出信号：

1. 复制对话上下文（或通过 `nest_handoff_history` beta 折叠它）。
2. 用其指令初始化目标智能体。
3. 用目标智能体继续运行。

这是产品化的监督者模式（第 13 课 / 第 28 课）。

### 护栏

三种风格：

- **输入护栏。** 在第一个智能体的输入上运行。在任何 LLM 调用之前拒绝不安全或超出范围的请求。
- **输出护栏。** 在最后一个智能体的输出上运行。捕获 PII 泄露、策略违规、格式错误的响应。
- **工具护栏。** 按函数工具运行。验证参数、检查权限、审计执行。

模式：

- **并行**（默认）。护栏 LLM 与主 LLM 同时运行。更低的尾部延迟。如果触发，主 LLM 的工作被丢弃（token 浪费）。
- **阻塞**（`run_in_parallel=False`）。护栏 LLM 先运行。如果触发，主调用上不浪费 token。

触发线引发 `InputGuardrailTripwireTriggered` / `OutputGuardrailTripwireTriggered`。

### 追踪

默认开启。每个 LLM 生成、工具调用、交接和护栏发出一个跨度。`OPENAI_AGENTS_DISABLE_TRACING=1` 选择退出。`add_trace_processor(processor)` 将跨度分发到你自己的后端以及 OpenAI 的后端。

### 会话

`Session` 在后端（SQLite、Redis、自定义）中存储对话历史。`Runner.run(agent, input, session=session)` 自动加载和追加。

### 这种模式出错的地方

- **交接漂移。** 智能体 A 交接给智能体 B，B 又交接回 A。添加跳跃计数器。
- **护栏绕过。** 工具护栏仅在函数工具上触发；内置工具（文件读取器、网页获取器）需要单独的策略。
- **过度追踪。** 跨度中的敏感内容。与 OTel GenAI 内容捕获规则（第 23 课）配对——外部存储，按 ID 引用。

## 构建

`code/main.py` 在标准库中实现 SDK 形状：

- `Agent`、`FunctionTool`、`Handoff`（作为具有传输语义的函数工具）。
- `Runner` 带有输入/输出/工具护栏、交接分派和跳跃计数器。
- 一个简单的跨度发射器，显示轨迹形状。
- 一个分诊智能体，根据用户查询交接给账单或支持；护栏在一个输入上触发。

运行：

```
python3 code/main.py
```

轨迹显示两次成功的交接、一次输入护栏触发和一个镜像真实 SDK 发出的跨度树。

## 使用

- **OpenAI Agents SDK** 用于 OpenAI 优先的产品。
- **Claude Agent SDK**（第 17 课）用于 Claude 优先的产品。
- **LangGraph**（第 13 课）当你需要显式状态和持久恢复时。
- **自定义**当你需要精确控制（语音、多提供商、联邦部署）时。

## 交付

`outputs/skill-agents-sdk-scaffold.md` 搭建一个 Agents SDK 应用，包含分诊智能体、交接、输入/输出/工具护栏、会话存储和追踪处理器。

## 练习

1. 添加交接跳跃计数器：N 次传输后拒绝。追踪行为。
2. 实现 `nest_handoff_history` 作为选项——在传输前将先前的消息折叠为一条摘要。
3. 编写阻塞输出护栏。比较会触发它的提示与通过的提示的延迟。
4. 将 `add_trace_processor` 连接到 JSON 日志器。每次跨度发出什么形状？
5. 阅读 SDK 文档。将你的标准库玩具移植到 `openai-agents-python`。你哪里建模错了？

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------------|------------------------|
| Agent | "LLM + 指令" | SDK 中的 Agent 类型；拥有工具和交接 |
| Handoff | "转移" | 模型调用以委派给另一个智能体的工具 |
| Guardrail | "策略检查" | 对输入/输出/工具调用的验证 |
| Tripwire | "护栏触发" | 护栏拒绝时引发的异常 |
| Session | "历史存储" | 运行之间持久化的对话记忆 |
| Tracing | "跨度" | LLM + 工具 + 交接 + 护栏的内置可观测性 |
| 阻塞护栏 | "顺序检查" | 护栏先运行；触发时不浪费 token |
| 并行护栏 | "并发检查" | 护栏同时运行；延迟更低，触发时浪费 token |

## 延伸阅读

- [OpenAI Agents SDK 文档](https://openai.github.io/openai-agents-python/) — 原语、交接、护栏、追踪
- [Claude Agent SDK 概述](https://platform.claude.com/docs/en/agent-sdk/overview) — Claude 风格的对应产品
- [Anthropic，构建有效的智能体](https://www.anthropic.com/research/building-effective-agents) — 何时使用交接
- [OpenTelemetry GenAI 语义约定](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — Agents SDK 跨度映射的标准
