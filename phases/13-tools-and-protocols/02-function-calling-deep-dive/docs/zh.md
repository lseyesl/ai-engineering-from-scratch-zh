# 函数调用深入——OpenAI、Anthropic、Gemini

> 三大前沿提供商在 2024 年趋同于相同的工具调用循环，然后在其他所有方面分道扬镳。OpenAI 使用 `tools` 和 `tool_calls`。Anthropic 使用 `tool_use` 和 `tool_result` 块。Gemini 使用 `functionDeclarations` 和唯一 ID 关联。本课程并排比较这三者，使在一个提供商上运行的代码在移植时不会出错。

**类型：** 构建
**语言：** Python（标准库，schema 转换器）
**前置要求：** Phase 13 · 01（工具接口）
**时间：** ~75 分钟

## 学习目标

- 陈述 OpenAI、Anthropic 和 Gemini 函数调用载荷之间的三个形状差异（声明、调用、结果）。
- 将一个工具声明翻译成所有三种提供商格式，并预测严格模式的约束差异。
- 在每个提供商中使用 `tool_choice` 来强制、禁止或自动选择工具调用。
- 了解每个提供商的硬限制（工具数量、schema 深度、参数长度）以及违反限制时各自发出的错误信号。

## 问题

函数调用请求的形状因提供商而异。来自 2026 年生产技术栈的三个具体例子：

**OpenAI Chat Completions / Responses API。** 你传递 `tools: [{type: "function", function: {name, description, parameters, strict}}]`。模型的响应包含 `choices[0].message.tool_calls: [{id, type: "function", function: {name, arguments}}]`，其中 `arguments` 是你必须解析的 JSON 字符串。严格模式（`strict: true`）通过受限解码强制 schema 合规。

**Anthropic Messages API。** 你传递 `tools: [{name, description, input_schema}]`。响应以 `content: [{type: "text"}, {type: "tool_use", id, name, input}]` 的形式返回。`input` 已经解析（是一个对象，而不是字符串）。你用一个包含 `{type: "tool_result", tool_use_id, content}` 块的新的 `user` 消息回复。

**Google Gemini API。** 你传递 `tools: [{functionDeclarations: [{name, description, parameters}]}]`（嵌套在 `functionDeclarations` 下）。响应以 `candidates[0].content.parts: [{functionCall: {name, args, id}}]` 的形式到达，其中 `id` 在 Gemini 3 及以上版本中是唯一的，用于并行调用关联。你用 `{functionResponse: {name, id, response}}` 回复。

相同的循环。不同的字段名、不同的嵌套、不同的字符串与对象约定、不同的关联机制。一个在 OpenAI 上编写天气智能体的团队移植到 Anthropic 需要两天，移植到 Gemini 还需要一天，仅仅是为了管道代码。

本课程构建了一个转换器，将三种格式统一为一个规范的工具声明，并在边缘进行路由。Phase 13 · 17 将相同的模式推广为 LLM 网关。

## 概念

### 公共结构

每个提供商都需要五样东西：

1. **工具列表。** 每个工具的名称、描述和输入 schema。
2. **工具选择。** 强制特定工具、禁止工具或让模型决定。
3. **调用发出。** 命名工具和参数的结构化输出。
4. **调用 ID。** 将响应关联到正确的调用（对并行调用很重要）。
5. **结果注入。** 将结果与调用关联的消息或块。

### 形状差异，逐个字段

| 方面 | OpenAI | Anthropic | Gemini |
|--------|--------|-----------|--------|
| 声明外壳 | `{type: "function", function: {...}}` | `{name, description, input_schema}` | `{functionDeclarations: [{...}]}` |
| Schema 字段 | `parameters` | `input_schema` | `parameters` |
| 响应容器 | 助手消息上的 `tool_calls[]` | 类型为 `tool_use` 的 `content[]` | 类型为 `functionCall` 的 `parts[]` |
| 参数类型 | 字符串化 JSON | 解析后的对象 | 解析后的对象 |
| ID 格式 | `call_...`（OpenAI 生成） | `toolu_...`（Anthropic） | UUID（Gemini 3+） |
| 结果块 | 角色 `tool`，`tool_call_id` | 带有 `tool_result`、`tool_use_id` 的 `user` | 带有匹配 `id` 的 `functionResponse` |
| 强制工具 | `tool_choice: {type: "function", function: {name}}` | `tool_choice: {type: "tool", name}` | `tool_config: {function_calling_config: {mode: "ANY"}}` |
| 禁止工具 | `tool_choice: "none"` | `tool_choice: {type: "none"}` | `mode: "NONE"` |
| 严格 schema | `strict: true` | schema 即 schema（始终强制） | 请求级别的 `responseSchema` |

### 你实际会遇到的限制

- **OpenAI。** 每个请求 128 个工具。Schema 深度 5。参数字符串 <= 8192 字节。严格模式不允许 `$ref`、`oneOf`/`anyOf`/`allOf` 的重叠，每个属性必须在 `required` 中列出。
- **Anthropic。** 每个请求 64 个工具。Schema 深度实际上无限制，但实际限制为 10。没有严格模式标志；schema 是一个契约，模型倾向于遵守。
- **Gemini。** 每个请求 64 个函数。Schema 类型是 OpenAPI 3.0 子集（与 JSON Schema 2020-12 略有不同）。自 Gemini 3 起并行调用具有唯一 ID。

### `tool_choice` 行为

每个人都支持的三种模式，名称不同：

- **Auto。** 模型选择工具或文本。默认。
- **Required / Any。** 模型必须调用至少一个工具。
- **None。** 模型不得调用工具。

再加上每个提供商独特的模式：

- **OpenAI。** 按名称强制特定工具。
- **Anthropic。** 按名称强制特定工具；`disable_parallel_tool_use` 标志分离单次与多次调用。
- **Gemini。** `mode: "VALIDATED"` 无论模型意图如何，都通过 schema 验证器路由每个响应。

### 并行调用

OpenAI 的 `parallel_tool_calls: true`（默认）在一个助手消息中发出多个调用。你全部运行它们，并用一个包含每个 `tool_call_id` 一个条目的批处理工具角色消息回复。Anthropic 历史上是单次调用；`disable_parallel_tool_use: false`（自 Claude 3.5 起默认）启用多调用。Gemini 2 允许并行调用但不提供稳定的 ID；Gemini 3 添加了 UUID，因此乱序响应可以干净地关联。

### 流式传输

三个都支持流式工具调用。线路格式不同：

- **OpenAI。** `tool_calls[i].function.arguments` 的增量块逐步到达。你累积直到 `finish_reason: "tool_calls"`。
- **Anthropic。** Block-start / block-delta / block-stop 事件。`input_json_delta` 块携带部分参数。
- **Gemini。** `streamFunctionCallArguments`（Gemini 3 新增）发出带有 `functionCallId` 的块，因此多个并行调用可以交错。

Phase 13 · 03 深入探讨并行 + 流式重组。本课程侧重于声明和单次调用的形状。

### 错误与修复

无效参数错误看起来也不一样：

- **OpenAI（非严格）。** 模型返回 `arguments: "{bad json}"`，你的 JSON 解析失败，你注入一条错误消息并重新调用。
- **OpenAI（严格）。** 验证在解码期间发生；无效 JSON 不可能，但可能出现 `refusal`。
- **Anthropic。** `input` 可能包含意外字段；schema 是建议性的。在服务端验证。
- **Gemini。** OpenAPI 3.0 的怪癖：对象字段上的 `enum` 被静默忽略；你自己验证。

### 转换器模式

代码中的规范工具声明看起来像这样：

```python
Tool(
    name="get_weather",
    description="Use when ...",
    input_schema={"type": "object", "properties": {...}, "required": [...]},
    strict=True,
)
```

三个小函数将其翻译成三种提供商的形状。`code/main.py` 中的测试工具正是这样做的，然后通过每个提供商的响应形状往返一个假的工具调用。无需网络——本课程教授的是形状，而不是 HTTP。

生产团队将此转换器包装在 `AbstractToolset`（Pydantic AI）、`UniversalToolNode`（LangGraph）或 `BaseTool`（LlamaIndex）中。Phase 13 · 17 提供了一个在三个提供商任何一个前面暴露 OpenAI 形状 API 的网关。

## 使用它

`code/main.py` 定义了一个规范的 `Tool` 数据类和三个发射 OpenAI、Anthropic 和 Gemini 声明 JSON 的转换器。然后它解析每个提供商形状的手工响应为相同的规范调用对象，演示了语义在表面之下是相同的。运行它并排比较三个声明。

关注点：

- 三个声明块仅在外壳和字段名上不同。
- 三个响应块在调用所在位置（顶层 `tool_calls`、`content[]` 块、`parts[]` 条目）不同。
- 一个 `canonical_call()` 函数从所有三种响应形状中提取 `{id, name, args}`。

## 交付物

本课程产生 `outputs/skill-provider-portability-audit.md`。给定一个针对一个提供商的函数调用集成，该技能产生一个可移植性审计：它依赖哪些提供商限制、哪些字段需要重命名、以及移植到其他提供商时会出现什么问题。

## 练习

1. 运行 `code/main.py` 并验证三个提供商的声明 JSON 都序列化了相同的底层 `Tool` 对象。修改规范工具以添加一个枚举参数，并确认只有 Gemini 转换器需要处理 OpenAPI 怪癖。

2. 为每个提供商添加一个 `ListToolsResponse` 解析器，用于提取模型在 `list_tools` 或发现调用后返回的工具列表。OpenAI 本身没有这个功能；注意这种不对称。

3. 实现 `tool_choice` 转换：将规范的 `ToolChoice(mode="force", tool_name="x")` 映射到所有三种提供商的形状。然后映射 `mode="any"` 和 `mode="none"`。查看课程的差异表。

4. 选择一个提供商并完整阅读其函数调用指南。找到其 schema 规范中其他两个提供商不支持的一个字段。候选：OpenAI `strict`、Anthropic `disable_parallel_tool_use`、Gemini `function_calling_config.allowed_function_names`。

5. 编写一个测试向量：一个参数违反声明 schema 的工具调用。通过每个提供商的验证器运行它（课程 01 中的标准库验证器可以作为代理），并记录触发哪些错误。记录你会在生产中使用哪个提供商的严格性。

## 关键术语

| 术语 | 人们说的 | 实际含义 |
|------|----------|----------|
| 函数调用 | "工具使用" | 提供商级别 API，用于结构化工具调用发出 |
| 工具声明 | "工具规范" | 名称 + 描述 + JSON Schema 输入载荷 |
| `tool_choice` | "强制 / 禁止" | 自动 / 必需 / 无 / 特定名称模式 |
| 严格模式 | "Schema 强制" | OpenAI 标志，约束解码以匹配 schema |
| `tool_use` 块 | "Anthropic 的调用形状" | 带有 id、name、input 的内联内容块 |
| `functionCall` 部分 | "Gemini 的调用形状" | 包含 name、args 和 id 的 `parts[]` 条目 |
| 参数即字符串 | "字符串化 JSON" | OpenAI 将 args 返回为 JSON 字符串，而不是对象 |
| 并行工具调用 | "一次扇出" | 一个助手消息中的多个工具调用 |
| 拒绝 | "模型拒绝" | 严格模式下代替调用的拒绝块 |
| OpenAPI 3.0 子集 | "Gemini schema 怪癖" | Gemini 使用类似 JSON Schema 的方言，但有细微差别 |

## 延伸阅读

- [OpenAI — 函数调用指南](https://platform.openai.com/docs/guides/function-calling) — 包括严格模式和并行调用的规范参考
- [Anthropic — 工具使用概述](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/overview) — `tool_use` 和 `tool_result` 块语义
- [Google — Gemini 函数调用](https://ai.google.dev/gemini-api/docs/function-calling) — 并行调用、唯一 ID 和 OpenAPI 子集
- [Vertex AI — 函数调用参考](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/multimodal/function-calling) — Gemini 的企业级界面
- [OpenAI — 结构化输出](https://platform.openai.com/docs/guides/structured-outputs) — 严格模式 schema 强制细节
