# MCP 基础——原语、生命周期、JSON-RPC 基础

> MCP 出现之前，每次集成都是定制的。模型上下文协议由 Anthropic 于 2024 年 11 月首次发布，现在由 Linux 基金会的 Agentic AI Foundation 管理，它标准化了发现和调用，使任何客户端都可与任何服务器通信。2025-11-25 规范命名了六个原语（三个服务器端，三个客户端）、一个三阶段生命周期和一个 JSON-RPC 2.0 线路格式。学会这些，本阶段 MCP 章节的其余部分就只是阅读。

**类型：** 学习
**语言：** Python（标准库，JSON-RPC 解析器）
**前置要求：** Phase 13 · 01 到 05（工具接口和函数调用）
**时间：** ~45 分钟

## 学习目标

- 命名所有六个 MCP 原语（工具、资源、提示在服务器端；根目录、采样、启发在客户端）并各给出一个用例。
- 走查三阶段生命周期（初始化、操作、关闭）并说明每个阶段谁发送哪条消息。
- 解析和发出 JSON-RPC 2.0 请求、响应和通知信封。
- 解释 `initialize` 时能力协商是什么以及没有它会出现什么问题。

## 问题

在 MCP 之前，每个使用工具的智能体都有自己的协议。Cursor 有一个 MCP 形状但不兼容的工具系统。Claude Desktop 附带了不同的一个。VS Code 的 Copilot 扩展有第三个。一个构建"Postgres 查询"工具的团队为每个不同的主机 API 编写了三次相同的工具。重用需要复制代码。

结果是单次定制的寒武纪爆发，制约了生态系统的增长速度。

MCP 通过标准化线路格式解决了这个问题。一个 MCP 服务器可以在每个 MCP 客户端中工作：Claude Desktop、ChatGPT、Cursor、VS Code、Gemini、Goose、Zed、Windsurf，到 2026 年 4 月超过 300 个客户端。1.1 亿月 SDK 下载量。超过 10,000 个公共服务器。Linux 基金会在 2025 年 12 月在新的 Agentic AI Foundation 下接手了管理。

本阶段使用的规范修订版是 **2025-11-25**。它添加了异步任务（SEP-1686）、URL 模式启发（SEP-1036）、带工具的采样（SEP-1577）、增量范围同意（SEP-835）和 OAuth 2.1 资源指示器语义。Phase 13 · 09 到 16 涵盖这些扩展。本课程止于基础。

## 概念

### 三个服务器端原语

1. **工具。** 可调用的动作。与 Phase 13 · 01 相同的四步循环。
2. **资源。** 暴露的数据。只读内容，通过 URI 寻址：`file:///path`、`db://query/...`、自定义方案。
3. **提示。** 可重用的模板。主机 UI 中的斜杠命令；服务器提供模板，客户端填写参数。

### 三个客户端原语

4. **根目录。** 服务器被允许触及的 URI 集合。客户端声明它们；服务器尊重它们。
5. **采样。** 服务器请求客户端的模型执行补全。启用服务器托管的智能体循环，而无需服务端 API 密钥。
6. **启发。** 服务器在运行中请求客户端的用户提供结构化输入。表单或 URL（SEP-1036）。

MCP 中的每个能力正好属于这六个之一。Phase 13 · 10 到 14 逐一深入介绍。

### 线路格式：JSON-RPC 2.0

每条消息都是一个 JSON 对象，包含以下字段：

- 请求：`{jsonrpc: "2.0", id, method, params}`。
- 响应：`{jsonrpc: "2.0", id, result | error}`。
- 通知：`{jsonrpc: "2.0", method, params}`——没有 `id`，不需要响应。

基础规范有大约 15 个方法，按原语分组。重要的有：

- `initialize` / `initialized`（握手）
- `tools/list`、`tools/call`
- `resources/list`、`resources/read`、`resources/subscribe`
- `prompts/list`、`prompts/get`
- `sampling/createMessage`（服务器到客户端）
- `notifications/tools/list_changed`、`notifications/resources/updated`、`notifications/progress`

### 三阶段生命周期

**阶段 1：初始化。**

客户端发送带有其 `capabilities` 和 `clientInfo` 的 `initialize`。服务器用其自己的 `capabilities`、`serverInfo` 和它说的规范版本回复。客户端在消化响应后发送 `notifications/initialized`。从这时起，任一方都可根据协商的能力发送请求。

**阶段 2：操作。**

双向。客户端调用 `tools/list` 进行发现，然后调用 `tools/call` 进行调用。如果服务器声明了采样能力，它可以发送 `sampling/createMessage`。当工具集变化时，服务器可以发送 `notifications/tools/list_changed`。当用户更改根范围时，客户端可以发送 `notifications/roots/list_changed`。

**阶段 3：关闭。**

任一方关闭传输。MCP 中没有结构化关闭方法；传输（stdio 或 Streamable HTTP，Phase 13 · 09）携带连接结束信号。

### 能力协商

`initialize` 握手时的 `capabilities` 就是契约。服务器示例：

```json
{
  "tools": {"listChanged": true},
  "resources": {"subscribe": true, "listChanged": true},
  "prompts": {"listChanged": true}
}
```

服务器声明它可以发出 `tools/list_changed` 通知并支持 `resources/subscribe`。客户端通过声明自己的来确认：

```json
{
  "roots": {"listChanged": true},
  "sampling": {},
  "elicitation": {}
}
```

如果客户端没有声明 `sampling`，服务器不得调用 `sampling/createMessage`。对称地：如果服务器没有声明 `resources.subscribe`，客户端不得尝试订阅。

这就是防止生态系统漂移的原因。不支持采样的客户端仍然是有效的 MCP 客户端；不调用 `sampling` 的服务器仍然是有效的 MCP 服务器。它们只是不一起使用该功能。

### 结构化内容和错误形状

`tools/call` 返回一个类型化块的 `content` 数组：`text`、`image`、`resource`。Phase 13 · 14 将 MCP Apps（`ui://` 交互式 UI）添加到该列表中。

错误使用 JSON-RPC 错误码。规范定义的补充：`-32002`"找不到资源"、`-32603`"内部错误"，加上 MCP 特定的错误数据作为 `error.data`。

### 客户端能力与工具调用细节

一个常见的混淆：`capabilities.tools` 是客户端是否支持工具列表更改通知。客户端是否将调用特定工具是由其模型驱动的运行时选择，而不是能力标志。能力标志是规范级别的契约。模型的选择是正交的。

### 为什么是 JSON-RPC 而不是 REST？

JSON-RPC 2.0（2010 年）是一个轻量级双向协议。REST 是客户端发起的。MCP 需要服务器发起的消息（采样、通知），因此 JSON-RPC 及其对称的请求/响应形状是自然选择。JSON-RPC 也能干净地组合在 stdio 和 WebSocket/Streamable HTTP 上，而无需重新发明 HTTP 的请求形状。

```figure
mcp-tool-call
```

## 使用它

`code/main.py` 提供了一个最小的 JSON-RPC 2.0 解析器和发射器，然后手动走完 `initialize` → `tools/list` → `tools/call` → `shutdown` 序列，打印每条消息。没有真正的传输；只有消息形状。与"延伸阅读"中链接的规范进行比较，验证每个信封。

关注点：

- `initialize` 双向声明能力；响应包含 `serverInfo` 和 `protocolVersion: "2025-11-25"`。
- `tools/list` 返回一个 `tools` 数组；每个条目有 `name`、`description`、`inputSchema`。
- `tools/call` 使用 `params.name` 和 `params.arguments`。
- 响应 `content` 是 `{type, text}` 块的数组。

## 交付物

本课程产生 `outputs/skill-mcp-handshake-tracer.md`。给定一个 MCP 客户端-服务器交互的 pcap 风格记录，该技能对每条消息进行注释，注明属于哪个原语、哪个生命周期阶段以及依赖哪个能力。

## 练习

1. 运行 `code/main.py`。确定能力协商发生的行，并描述如果服务器没有声明 `tools.listChanged` 会有什么变化。

2. 扩展解析器以处理 `notifications/progress`。消息形状：`{method: "notifications/progress", params: {progressToken, progress, total}}`。在长时间运行的 `tools/call` 进行时发射它，并确认客户端处理程序会显示进度条。

3. 从头到尾阅读 MCP 2025-11-25 规范——整个文档大约 80 页。确定大多数服务器不需要的一个能力标志。提示：它与资源订阅有关。

4. 在纸上勾画一个假设的"cron 作业"功能所属的原语。（提示：服务器希望客户端在预定时间调用它。目前六个原语都不适合。）MCP 的 2026 路线图有一个相关的草案 SEP。

5. 从 GitHub 上一个开源 MCP 服务器解析一个会话日志。计数请求与响应与通知消息。计算生命周期与操作流量之间的比例。

## 关键术语

| 术语 | 人们说的 | 实际含义 |
|------|----------|----------|
| MCP | "模型上下文协议" | 模型到工具发现和调用的开放协议 |
| 服务器原语 | "服务器暴露的内容" | 工具（动作）、资源（数据）、提示（模板） |
| 客户端原语 | "客户端让服务器使用的内容" | 根目录（范围）、采样（LLM 回调）、启发（用户输入） |
| JSON-RPC 2.0 | "线路格式" | 对称的请求/响应/通知信封 |
| `initialize` 握手 | "能力协商" | 第一对消息；服务器和客户端声明它们支持的功能 |
| `tools/list` | "发现" | 客户端询问服务器当前工具集 |
| `tools/call` | "调用" | 客户端要求服务器使用参数执行工具 |
| `notifications/*_changed` | "变更事件" | 服务器告诉客户端其原语列表已更改 |
| 内容块 | "类型化结果" | 工具结果中的 `{type: "text" \| "image" \| "resource" \| "ui_resource"}` |
| SEP | "规范演进提案" | 命名的草案提案（例如用于异步任务的 SEP-1686） |

## 延伸阅读

- [模型上下文协议 — 2025-11-25 规范](https://modelcontextprotocol.io/specification/2025-11-25) — 规范参考文档
- [模型上下文协议 — 架构概念](https://modelcontextprotocol.io/docs/concepts/architecture) — 六个原语的心智模型
- [Anthropic — 介绍模型上下文协议](https://www.anthropic.com/news/model-context-protocol) — 2024 年 11 月发布文章
- [MCP 博客 — MCP 一周年](https://blog.modelcontextprotocol.io/posts/2025-11-25-first-mcp-anniversary/) — 一年回顾和 2025-11-25 规范变更
- [WorkOS — MCP 2025-11-25 规范更新](https://workos.com/blog/mcp-2025-11-25-spec-update) — SEP-1686、1036、1577、835 和 1724 总结
