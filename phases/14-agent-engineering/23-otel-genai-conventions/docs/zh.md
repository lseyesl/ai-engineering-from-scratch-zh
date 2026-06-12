# OpenTelemetry GenAI 语义约定

> OpenTelemetry 的 GenAI SIG（2024 年 4 月启动）定义了智能体遥测的标准模式。跨度名称、属性和内容捕获规则跨供应商趋同，使智能体轨迹在 Datadog、Grafana、Jaeger 和 Honeycomb 中含义相同。

**类型：** Learn + Build
**语言：** Python（标准库）
**前置知识：** Phase 14 · 13（LangGraph），Phase 14 · 24（可观测性平台）
**时间：** ~60 分钟

## 学习目标

- 说出 GenAI 跨度类别：模型/客户端、智能体、工具。
- 区分 `invoke_agent` 的 CLIENT 与 INTERNAL 跨度以及各自何时适用。
- 列出顶级 GenAI 属性：提供商名称、请求模型、数据源 ID。
- 解释内容捕获合同：选择加入、`OTEL_SEMCONV_STABILITY_OPT_IN`、外部引用推荐。

## 问题

每个供应商发明自己的跨度名称。运维团队最终构建每个框架的仪表板。OpenTelemetry 的 GenAI SIG 通过定义一个整个生态系统瞄准的单一标准来解决这个问题。

## 概念

### 跨度类别

1. **模型/客户端跨度。** 涵盖原始 LLM 调用。由提供商 SDK（Anthropic、OpenAI、Bedrock）和框架模型适配器发出。
2. **智能体跨度。** `create_agent`（构造智能体时）和 `invoke_agent`（运行智能体时）。
3. **工具跨度。** 每个工具调用一个；通过父子关系连接到智能体跨度。

### 智能体跨度命名

- 跨度名称：如果命名了则为 `invoke_agent {gen_ai.agent.name}`；回退为 `invoke_agent`。
- 跨度种类：
  - **CLIENT** — 用于远程智能体服务（OpenAI Assistants API、Bedrock Agents）。
  - **INTERNAL** — 用于进程内智能体框架（LangChain、CrewAI、本地 ReAct）。

### 关键属性

- `gen_ai.provider.name` — `anthropic`、`openai`、`aws.bedrock`、`google.vertex`。
- `gen_ai.request.model` — 模型 ID。
- `gen_ai.response.model` — 解析后的模型（由于路由可能不同于请求）。
- `gen_ai.agent.name` — 智能体标识符。
- `gen_ai.operation.name` — `chat`、`completion`、`invoke_agent`、`tool_call`。
- `gen_ai.data_source.id` — 用于 RAG：查询了哪个语料库或存储。

存在针对 Anthropic、Azure AI Inference、AWS Bedrock、OpenAI 的技术特定约定。

### 内容捕获

默认规则：仪表化默认不捕获输入/输出。通过以下方式选择加入：

- `gen_ai.system_instructions`
- `gen_ai.input.messages`
- `gen_ai.output.messages`

推荐的生产模式：外部存储内容（S3、你的日志存储），在跨度上记录引用（指针 ID，不是原文）。这是第 27 课内容中毒防御在可观测性中的连线。

### 稳定性

截至 2026 年 3 月，大多数约定是实验性的。选择加入稳定预览：

```
OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental
```

Datadog v1.37+ 原生将 GenAI 属性映射到其 LLM 可观测性模式中。其他后端（Grafana、Honeycomb、Jaeger）支持原始属性。

### 这种模式出错的地方

- **在跨度中捕获完整提示。** PII、秘密、客户数据在运维可读取的轨迹中。外部存储。
- **缺少 `gen_ai.provider.name`。** 当属性缺失时，多提供商仪表板会崩溃。
- **没有父链接的跨度。** 孤立的工具跨度。始终传播上下文。
- **未设置稳定性选择加入。** 你的属性可能在后端升级时被重命名。

## 构建

`code/main.py` 实现一个匹配 GenAI 约定的标准库跨度发射器：

- 带有 GenAI 属性模式的 `Span`。
- 带有 `start_span`、嵌套上下文的 `Tracer`。
- 一个发出以下内容的脚本化智能体运行：`create_agent`、`invoke_agent`（INTERNAL）、每工具跨度、LLM 调用的 `chat` 跨度。
- 一个外部存储提示并在跨度上记录 ID 的内容捕获模式。

运行：

```
python3 code/main.py
```

输出：一个包含所有必需 GenAI 属性的跨度树，以及一个显示选择加入内容引用的"外部存储"。

## 使用

- **Datadog LLM 可观测性**（v1.37+）原生映射属性。
- **Langfuse / Phoenix / Opik**（第 24 课）——自动仪表化生态系统。
- **Jaeger / Honeycomb / Grafana Tempo**——原始 OTel 轨迹；从 GenAI 属性构建仪表板。
- **自托管**——运行带有 GenAI 处理器的 OTel Collector。

## 交付

`outputs/skill-otel-genai.md` 将 OTel GenAI 跨度接入现有智能体，包含内容捕获默认值和外部引用存储。

## 练习

1. 用 `invoke_agent`（INTERNAL）+ 每工具跨度仪表化你的第 01 课 ReAct 循环。发送到 Jaeger 实例。
2. 添加"仅引用"模式的内容捕获：提示到 SQLite，跨度属性只携带行 ID。
3. 阅读 `gen_ai.data_source.id` 的规范。将其连接到你的第 09 课 Mem0 搜索。
4. 设置 `OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental` 并验证你的属性不会被 collector 重命名。
5. 构建仪表板："哪些工具错误与哪些模型相关"仅从 GenAI 属性。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------------|------------------------|
| GenAI SIG | "OpenTelemetry GenAI 组" | 定义模式的 OTel 工作组 |
| invoke_agent | "智能体跨度" | 表示智能体运行的跨度名称 |
| CLIENT 跨度 | "远程调用" | 对远程智能体服务的调用跨度 |
| INTERNAL 跨度 | "进程内" | 进程内智能体运行的跨度 |
| gen_ai.provider.name | "提供商" | anthropic / openai / aws.bedrock / google.vertex |
| gen_ai.data_source.id | "RAG 源" | 检索命中的哪个语料库/存储 |
| 内容捕获 | "提示日志记录" | 消息的选择加入捕获；生产中外部存储 |
| 稳定性选择加入 | "预览模式" | 固定实验性约定的环境变量 |

## 延伸阅读

- [OpenTelemetry GenAI 语义约定](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — 规范
- [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) — 默认的 GenAI 跨度
- [AutoGen v0.4（微软研究院）](https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/) — 内置 OTel 跨度
- [Claude Agent SDK](https://platform.claude.com/docs/en/agent-sdk/overview) — W3C 追踪上下文传播
