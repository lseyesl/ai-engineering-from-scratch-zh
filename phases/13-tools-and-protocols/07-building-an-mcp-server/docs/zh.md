# 构建 MCP 服务器——Python + TypeScript SDK

> 大多数 MCP 教程只展示 stdio hello-world。一个真正的服务器暴露工具加资源加提示，处理能力协商，发出结构化错误，并且在各个 SDK 中工作方式相同。本课程端到端构建一个笔记服务器：标准库 stdio 传输、JSON-RPC 分发、三个服务器端原语，以及一个纯函数风格，在毕业后可以无缝移植到 Python SDK 的 FastMCP 或 TypeScript SDK。

**类型：** 构建
**语言：** Python（标准库，stdio MCP 服务器）
**前置要求：** Phase 13 · 06（MCP 基础）
**时间：** ~75 分钟

## 学习目标

- 实现 `initialize`、`tools/list`、`tools/call`、`resources/list`、`resources/read`、`prompts/list` 和 `prompts/get` 方法。
- 编写一个分派循环，从 stdin 读取 JSON-RPC 消息并将响应写入 stdout。
- 根据 JSON-RPC 2.0 规范和 MCP 附加代码发出结构化错误响应。
- 将标准库实现升级到 FastMCP（Python SDK）或 TypeScript SDK，而无需重写工具逻辑。

## 问题

在你使用远程传输（Phase 13 · 09）或认证层（Phase 13 · 16）之前，你需要一个干净的本地服务器。本地意味着 stdio：服务器作为子进程由客户端生成，消息通过换行分隔的 stdin/stdout 流动。

2025-11-25 规范规定，stdio 消息编码为带有显式 `\n` 分隔符的 JSON 对象。这里没有 SSE；SSE 是旧的远程模式，将在 2026 年中移除（Atlassian 的 Rovo MCP 服务器于 2026 年 6 月 30 日弃用它；Keboola 于 2026 年 4 月 1 日）。对于 stdio，每行一个 JSON 对象就是整个线路格式。

笔记服务器是一个很好的形状，因为它能练习所有三个服务器端原语。工具执行变更（`notes_create`）。资源暴露数据（`notes://{id}`）。提示提供模板（`review_note`）。这个形状推广到任何领域。

## 概念

### 分派循环

```
循环:
  line = stdin.readline()
  msg = json.loads(line)
  如果有 id:
    处理请求 -> 写入响应
  否则:
    处理通知 -> 不响应
```

三条规则：

- 不要向 stdout 打印任何不是 JSON-RPC 信封的内容。调试日志写入 stderr。
- 每个请求必须与携带相同 `id` 的响应匹配。
- 通知不得响应。

### 实现 `initialize`

```python
def initialize(params):
    return {
        "protocolVersion": "2025-11-25",
        "capabilities": {
            "tools": {"listChanged": True},
            "resources": {"listChanged": True, "subscribe": False},
            "prompts": {"listChanged": False},
        },
        "serverInfo": {"name": "notes", "version": "1.0.0"},
    }
```

只声明你支持的内容。客户端依赖能力集来门控功能。

### 实现 `tools/list` 和 `tools/call`

`tools/list` 返回 `{tools: [...]}`，每个条目包含 `name`、`description`、`inputSchema`。`tools/call` 接受 `{name, arguments}` 并返回 `{content: [blocks], isError: bool}`。

内容块是类型化的。最常见的：

```json
{"type": "text", "text": "找到 2 条笔记"}
{"type": "resource", "resource": {"uri": "notes://14", "text": "..."}}
{"type": "image", "data": "<base64>", "mimeType": "image/png"}
```

工具错误有两种形状。协议级别的错误（未知方法、错误参数）是 JSON-RPC 错误。工具级别的错误（调用有效但工具失败）作为 `{content: [...], isError: true}` 返回。这让模型在其上下文中看到失败。

### 实现资源

资源默认是只读的。`resources/list` 返回清单；`resources/read` 返回内容。URI 可以是 `file://...`、`http://...` 或自定义方案如 `notes://`。

当你将数据暴露为资源而不是工具时：

- 模型不会"调用"它；客户端可以在用户请求时将其注入上下文。
- 订阅让服务器在资源更改时推送更新（Phase 13 · 10）。
- Phase 13 · 14 通过 `ui://` 将其扩展到交互式资源。

### 实现提示

提示是带命名参数的模板。主机将它们作为斜杠命令暴露。一个 `review_note` 提示可以接受 `note_id` 参数，并生成一个多消息提示模板，客户端将其馈送给其模型。

### Stdio 传输细节

- 换行分隔的 JSON。没有长度前缀的帧。
- 不要缓冲。每次写入后 `sys.stdout.flush()`。
- 客户端控制生命周期。当 stdin 关闭（EOF）时，干净退出。
- 不要静默处理 SIGPIPE；记录并退出。

### 注解

每个工具可以携带描述安全属性的 `annotations`：

- `readOnlyHint: true` — 纯读取，安全重试。
- `destructiveHint: true` — 不可逆的副作用；客户端应确认。
- `idempotentHint: true` — 相同输入产生相同输出。
- `openWorldHint: true` — 与外部系统交互。

客户端使用这些来决定 UX（确认对话框、状态指示器）和路由（Phase 13 · 17）。

### 升级路径

`code/main.py` 中的标准库服务器大约 180 行。FastMCP（Python）将相同的逻辑简化为装饰器风格：

```python
from fastmcp import FastMCP
app = FastMCP("notes")

@app.tool()
def notes_search(query: str, limit: int = 10) -> list[dict]:
    ...
```

TypeScript SDK 有等效的形状。升级路径在你准备好时是即插即用的；概念（能力、分派、内容块）是相同的。

## 使用它

`code/main.py` 是一个完整的基于 stdio 的笔记 MCP 服务器，仅使用标准库。它处理 `initialize`、三个工具的 `tools/list` 和 `tools/call`（`notes_list`、`notes_search`、`notes_create`）、每个笔记的 `resources/list` 和 `resources/read`，以及一个 `review_note` 提示。你可以通过管道传输 JSON-RPC 消息来驱动它：

```
echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' | python main.py
```

关注点：

- 分派器是一个以方法名为键的 `dict[str, Callable]`。
- 每个工具执行器返回内容块列表，而不是裸字符串。
- 执行器抛出时设置 `isError: true`。

## 交付物

本课程产生 `outputs/skill-mcp-server-scaffolder.md`。给定一个领域（笔记、工单、文件、数据库），该技能以正确的工具/资源/提示拆分和 SDK 升级路径脚手架一个 MCP 服务器。

## 练习

1. 运行 `code/main.py` 并用手工构建的 JSON-RPC 消息驱动它。练习 `notes_create`，然后 `resources/read` 来检索新笔记。

2. 添加一个带有 `annotations: {destructiveHint: true}` 的 `notes_delete` 工具。验证客户端会显示一个确认对话框（这需要一个真实主机，如 Claude Desktop）。

3. 实现 `resources/subscribe`，使服务器在笔记被修改时推送 `notifications/resources/updated`。添加一个保活任务。

4. 将服务器移植到 FastMCP。Python 文件应该缩小到 80 行以下。线路行为必须相同；使用相同的 JSON-RPC 测试工具验证。

5. 阅读规范的 `server/tools` 部分，并确定本课程服务器未实现的一个工具定义字段。（提示：有多个；选择一个并添加它。）

## 关键术语

| 术语 | 人们说的 | 实际含义 |
|------|----------|----------|
| MCP 服务器 | "暴露工具的东西" | 在 stdio 或 HTTP 上说 MCP JSON-RPC 的进程 |
| stdio 传输 | "子进程模型" | 服务器由客户端生成；通过 stdin/stdout 通信 |
| 分派器 | "方法路由器" | JSON-RPC 方法名到处理函数的映射 |
| 内容块 | "工具结果块" | 工具响应 `content` 数组中的类型化元素 |
| `isError` | "工具级别失败" | 标识工具失败；与 JSON-RPC 错误区分 |
| 注解 | "安全提示" | readOnly / destructive / idempotent / openWorld 标志 |
| FastMCP | "Python SDK" | 装饰器风格的 MCP 协议高级框架 |
| 资源 URI | "可寻址数据" | 标识资源的 `file://`、`db://` 或自定义方案 |
| 提示模板 | "斜杠命令简要说明" | 服务器为宿主机提供的带参数槽的模板 |
| 能力声明 | "功能切换" | 在 `initialize` 中声明的每个原语的标志 |

## 延伸阅读

- [模型上下文协议 — Python SDK](https://github.com/modelcontextprotocol/python-sdk) — 参考 Python 实现
- [模型上下文协议 — TypeScript SDK](https://github.com/modelcontextprotocol/typescript-sdk) — 并行 TS 实现
- [FastMCP — 服务器框架](https://gofastmcp.com/) — MCP 服务器的装饰器风格 Python API
- [MCP — 快速入门服务器指南](https://modelcontextprotocol.io/quickstart/server) — 使用任一 SDK 的端到端教程
- [MCP — 服务器工具规范](https://modelcontextprotocol.io/specification/2025-11-25/server/tools) — tools/* 消息的完整参考
