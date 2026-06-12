# Claude Agent SDK：子智能体与会话存储

> Claude Agent SDK 是 Claude Code 框架的库形式。内置工具、用于上下文隔离的子智能体、钩子、W3C 追踪传播、会话存储。Claude Managed Agents 是用于长时间运行异步工作的托管替代方案。

**类型：** Learn + Build
**语言：** Python（标准库）
**前置知识：** Phase 14 · 01（智能体循环），Phase 14 · 10（技能库）
**时间：** ~75 分钟

## 学习目标

- 解释 Anthropic Client SDK（原始 API）和 Claude Agent SDK（框架形状）之间的区别。
- 描述子智能体——并行化和上下文隔离——以及何时使用它们。
- 说出 Python SDK 的会话存储面（`append`、`load`、`list_sessions`、`delete`、`list_subkeys`）和 `--session-mirror` 的作用。
- 在标准库中实现一个带有内置工具、具有隔离上下文的子智能体生成、生命周期钩子和会话存储的框架。

## 问题

原始 LLM API 只能进行一次往返。生产智能体需要工具执行、MCP 服务器、生命周期钩子、子智能体生成、会话持久化、追踪传播。Claude Agent SDK 将这种形状作为库打包——与 Claude Code 使用的框架相同，为自定义智能体开放。

## 概念

### Client SDK vs Agent SDK

- **Client SDK（`anthropic`）。** 原始 Messages API。你拥有循环、工具、状态。
- **Agent SDK（`claude-agent-sdk`）。** 内置工具执行、MCP 连接、钩子、子智能体生成、会话存储。Claude Code 循环作为库。

### 内置工具

SDK 开箱即用提供 10+ 工具：文件读/写、shell、grep、glob、网页获取等。自定义工具通过标准工具模式接口注册。

### 子智能体

Anthropic 记录的两个用途：

1. **并行化。** 并发运行独立工作。"找到这 20 个模块每个的测试文件"是 20 个并行子智能体任务。
2. **上下文隔离。** 子智能体使用自己的上下文窗口；仅结果返回给编排器。编排器的预算得以保留。

Python SDK 近期新增：`list_subagents()`、`get_subagent_messages()` 用于读取子智能体记录。

### 会话存储

与 TypeScript 协议一致：

- `append(session_id, message)`——添加一轮。
- `load(session_id)`——恢复对话。
- `list_sessions()`——枚举。
- `delete(session_id)`——级联删除子智能体会话。
- `list_subkeys(session_id)`——列出子智能体键。

`--session-mirror`（CLI 标志）在流式传输时将记录镜像到外部文件，用于调试。

### 钩子

你可以注册的生命周期钩子：

- `PreToolUse`、`PostToolUse`——门控或审计工具调用。
- `SessionStart`、`SessionEnd`——设置和拆卸。
- `UserPromptSubmit`——在模型看到用户输入之前行动。
- `PreCompact`——在上下文压缩前运行。
- `Stop`——智能体退出时清理。
- `Notification`——侧信道警报。

钩子是 pro-workflow（Phase 14 课程参考）和类似系统添加横切行为的方式。

### W3C 追踪上下文

调用者上的 OTel 跨度通过 W3C 追踪上下文头传播到 CLI 子进程中。整个多进程轨迹在你的后端显示为一条轨迹。

### Claude Managed Agents

托管替代方案（beta 头 `managed-agents-2026-04-01`）。长时间运行异步工作、内置提示缓存、内置压缩。以控制权换取托管基础设施。

### 这种模式出错的地方

- **子智能体过度生成。** 为 100 个小任务生成 100 个子智能体。开销占主导。改为批量处理。
- **钩子蔓延。** 每个团队添加钩子；启动时间膨胀。每季度审查钩子。
- **会话膨胀。** 会话累积；大小增长。使用 `list_sessions` + 过期策略。

## 构建

`code/main.py` 在标准库中实现 SDK 形状：

- `Tool`、`ToolRegistry` 带有内置 `read_file`、`write_file`、`list_dir`。
- `Subagent`——私有上下文、隔离运行、返回结果。
- `SessionStore`——append、load、list、delete、list_subkeys。
- `Hooks`——`pre_tool_use`、`post_tool_use`、`session_start`、`session_end`。
- 一个演示：主智能体并行生成 3 个子智能体（每个隔离），聚合结果，持久化会话。

运行：

```
python3 code/main.py
```

轨迹显示子智能体上下文隔离（编排器上下文大小保持有界）、钩子执行和会话持久化。

## 使用

- **Claude Agent SDK** 用于想要 Claude Code 框架形状的 Claude 优先产品。
- **Claude Managed Agents** 用于托管的长时间运行异步工作。
- **OpenAI Agents SDK**（第 16 课）用于 OpenAI 优先的对应产品。
- **LangGraph + 自定义工具** 如果你想要图形状的状态机。

## 交付

`outputs/skill-claude-agent-scaffold.md` 搭建一个 Claude Agent SDK 应用，包含子智能体、钩子、会话存储、MCP 服务器连接和 W3C 追踪传播。

## 练习

1. 添加一个子智能体生成器，将 20 个任务分批成每组 5 个并行子智能体。测量编排器上下文大小与每个任务一个子智能体的对比。
2. 实现一个 `PreToolUse` 钩子，对 `write_file` 调用进行速率限制（每会话每分钟 5 次）。追踪行为。
3. 连接 `list_subkeys` 以渲染子智能体树。深度嵌套看起来是什么样？
4. 将玩具移植到真正的 `claude-agent-sdk` Python 包。工具注册有什么变化？
5. 阅读 Claude Managed Agents 文档。你何时会从自托管切换到托管？

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------------|------------------------|
| Agent SDK | "Claude Code 作为库" | 框架形状：工具、MCP、钩子、子智能体、会话存储 |
| 子智能体 | "子智能体" | 单独的上下文、自己的预算；结果向上冒泡 |
| 会话存储 | "对话数据库" | 持久化、加载、列出、删除轮次，级联子智能体 |
| 钩子 | "生命周期回调" | 前/后工具、会话、提示提交、压缩、停止 |
| W3C 追踪上下文 | "跨进程追踪" | 父跨度传播到 CLI 子进程 |
| Managed Agents | "托管框架" | Anthropic 托管的长时间运行异步工作 |
| `--session-mirror` | "记录镜像" | 流式传输时将会话轮次写入外部文件 |
| MCP 服务器 | "工具面" | 连接到智能体的外部工具/资源来源 |

## 延伸阅读

- [Claude Agent SDK 概述](https://platform.claude.com/docs/en/agent-sdk/overview) — Claude Code 的库形式
- [Anthropic，使用 Claude Agent SDK 构建智能体](https://www.anthropic.com/engineering/building-agents-with-the-claude-agent-sdk) — 生产模式
- [Claude Managed Agents 概述](https://platform.claude.com/docs/en/managed-agents/overview) — 托管替代方案
- [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) — 对应产品
