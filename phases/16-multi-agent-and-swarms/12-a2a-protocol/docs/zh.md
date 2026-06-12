# A2A——代理间协议 (The Agent-to-Agent Protocol)

> Google 于 2025 年 4 月宣布 A2A；到 2026 年 4 月，规范位于 https://a2a-protocol.org/latest/specification/，超过 150 个组织支持它。A2A 是 MCP（第 13 课）的水平补充：MCP 是垂直的（代理 ↔ 工具），A2A 是对等的（代理 ↔ 代理）。它定义了 Agent Card（发现）、带有工件(artifacts)的任务（文本、结构化数据、视频）、不透明的任务生命周期(opaque task lifecycles)和认证(auth)。生产系统越来越多地将 MCP 与 A2A 配对。Google Cloud 在 2025-2026 年期间将 A2A 支持集成到 Vertex AI Agent Builder 中。

**类型:** Learn + Build
**语言:** Python (stdlib, `http.server`, `json`)
**前置知识:** Phase 16 · 04 (原始模型)
**时间:** ~75 分钟

## 问题 (Problem)

你的代理需要调用另一个系统上的另一个代理。怎么做？你可以暴露一个 HTTP 端点，定义一个定制的 JSON schema，然后希望对方能理解。每对代理都变成了一次定制集成。

A2A 就是那次调用的通用线路协议(wire protocol)。标准发现、标准任务模型、标准传输、标准工件。就像 HTTP+REST 但将代理视为一等公民。

## 概念 (Concept)

### 四个要素 (The four elements)

**Agent Card (代理卡片)。** 位于 `/.well-known/agent.json` 的一个 JSON 文档，描述代理：名称、技能、端点、支持的模态(modalities)、认证要求。发现通过读取卡片完成。

```
GET https://agent.example.com/.well-known/agent.json
→ {
    "name": "code-review-agent",
    "skills": ["review-python", "review-typescript"],
    "endpoints": {
      "tasks": "https://agent.example.com/tasks"
    },
    "auth": {"type": "bearer"},
    "modalities": ["text", "structured"]
  }
```

**Task (任务)。** 工作单元。一个异步、有状态的对象，具有生命周期：`submitted → working → completed / failed / canceled`。客户端发送任务，轮询或订阅更新。

**Artifact (工件)。** 任务产生的结果类型。文本、结构化 JSON、图像、视频、音频。工件是有类型的，因此不同模态是一等公民。

**Opaque lifecycle (不透明生命周期)。** A2A 不规定远程代理*如何*解决任务。客户端看到状态转换和工件；实现可以使用任何框架。

### MCP/A2A 的分工 (The MCP/A2A split)

- **MCP**（第 13 课）：代理 ↔ 工具。代理通过 JSON-RPC 对工具服务器进行读写。默认无状态。
- **A2A**：代理 ↔ 代理。对等协议；双方都是拥有自己推理能力的代理。

生产级多代理系统两者都使用。一个 A2A 对等体在其一侧调用 MCP 工具。这种分工使两个关注点保持清晰。

### 发现流程 (Discovery flow)

```
Client                     Agent server
  ├──GET /.well-known/agent.json──>
  <──Agent Card JSON─────────────
  ├──POST /tasks {skill, input}──>
  <──201 task_id, state=submitted
  ├──GET /tasks/{id}──────────────>
  <──state=working, 42% done──────
  ├──GET /tasks/{id}──────────────>
  <──state=completed, artifacts──
```

或者使用流式：SSE 订阅 `/tasks/{id}/events` 以获取推送更新。

### 认证 (Auth)

A2A 支持三种常见模式：

- **Bearer token (Bearer 令牌)** — OAuth2 或不透明令牌。
- **mTLS** — 双向 TLS；组织之间相互证明身份。
- **Signed requests (签名请求)** — 对负载的 HMAC。

认证在 Agent Card 中声明；客户端发现并遵守。

### 截至 2026 年 4 月的 150+ 个组织 (150+ organizations by April 2026)

企业采用推动了 A2A 的规模。头条新闻：A2A 成为企业代理系统跨越信任边界的方式。Google Cloud 发布了 Vertex AI Agent Builder 的 A2A 支持；Microsoft Agent Framework 支持它；大多数主流框架（LangGraph、CrewAI、AutoGen）都提供了 A2A 适配器。

### A2A 的优势 (Where A2A wins)

- **跨组织调用(Cross-organization calls)。** 公司 A 的代理调用公司 B 的代理。没有 A2A，每对都是一个定制合同。
- **异构框架(Heterogeneous frameworks)。** LangGraph 代理调用 CrewAI 代理调用自定义 Python 代理。A2A 将其标准化。
- **类型化工件(Typed artifacts)。** 视频结果、结构化 JSON、音频——都是一等公民。
- **长时间运行的任务(Long-running tasks)。** 不透明生命周期 + 轮询使耗时数小时的任务变得简单。

### A2A 的困难 (Where A2A struggles)

- **延迟敏感的微调用(Latency-sensitive micro-calls)。** A2A 的生命周期是异步的。亚毫秒级的代理间通信不适合；使用直接 RPC。
- **紧耦合的进程内代理(Tight-coupled in-process agents)。** 如果两个代理在同一个 Python 进程中运行，A2A 的 HTTP 往返是过度设计。
- **小团队(Small teams)。** 规范开销是真实存在的；仅限内部的代理可能不需要这种形式化。

### A2A vs ACP, ANP, NLIP

2024-2026 年出现了几个相关规范：

- **ACP** (IBM/Linux Foundation) — A2A 的前身，范围更窄。
- **ANP** (Agent Network Protocol) — 对等发现为主，去中心化优先。
- **NLIP** (Ecma Natural Language Interaction Protocol，2025 年 12 月标准化) — 自然语言内容类型。

截至 2026 年 4 月，A2A 是采用最广泛的对等协议。参见 arXiv:2505.02279 (Liu et al., "A Survey of Agent Interoperability Protocols") 以进行比较。

## 构建 (Build It)

`code/main.py` 使用 `http.server` 和 JSON 实现了一个最小化的 A2A 服务器和客户端。服务器：

- 暴露 `/.well-known/agent.json`，
- 接受 `POST /tasks`，
- 管理任务状态，
- 在 `GET /tasks/{id}` 上返回工件。

客户端：

- 获取 Agent Card，
- 提交任务，
- 轮询直到完成，
- 读取工件。

运行：

```
python3 code/main.py
```

脚本在后台线程中启动服务器，然后针对它运行客户端。你将看到完整的流程：发现、提交、轮询、工件。

## 使用 (Use It)

`outputs/skill-a2a-integrator.md` 设计一个 A2A 集成：Agent Card 内容、任务 schema、认证选择、流式 vs 轮询。

## 交付 (Ship It)

检查清单：

- **固定规范版本(Pin the spec version)。** A2A 仍在演进；Agent Card 应声明协议版本。
- **幂等的任务创建(Idempotent task creation)。** 重复提交（网络重试）应产生一个任务。
- **工件 schema (Artifact schemas)。** 声明代理返回的形状；消费者应进行验证。
- **速率限制 + 认证(Rate limits + auth)。** A2A 是对外公开的；应用标准的 Web 安全措施。
- **失败任务的死信队列(Dead-letter for failed tasks)。** 随时间检查模式以发现重复出现的失败类型。

## 练习 (Exercises)

1. 运行 `code/main.py`。确认客户端发现服务器并接收到正确的工件。
2. 在服务器上添加第二个技能（例如 "summarize"）。更新 Agent Card。编写一个根据任务类型选择技能的客户端。
3. 实现一个 SSE 流式端点：`/tasks/{id}/events` 发出状态变更。客户端需要做什么不同的事情？
4. 阅读 A2A 规范 (https://a2a-protocol.org/latest/specification/)。找出规范强制要求但本演示未实现的三个事项。
5. 比较 A2A (Agent Card 发现) 与 MCP (通过 `listTools` 的服务端能力列举)。自描述代理与能力探测之间的权衡是什么？

## 关键术语 (Key Terms)

| 术语 | 人们说的 | 实际含义 |
|------|----------|----------|
| A2A (代理间协议) | "代理到代理" | 代理跨系统调用其他代理的对等协议。Google 2025。 |
| Agent Card (代理卡片) | "代理的名片" | `/.well-known/agent.json` 处的 JSON，描述技能、端点、认证。 |
| Task (任务) | "工作单元" | 具有生命周期的异步有状态对象；完成时产生工件。 |
| Artifact (工件) | "结果" | 类型化输出：文本、结构化 JSON、图像、视频、音频。一等媒体。 |
| Opaque lifecycle (不透明生命周期) | "如何解决是代理的事" | 客户端看到状态转换；服务端可以自由选择框架/工具。 |
| Discovery (发现) | "找到代理" | `GET /.well-known/agent.json` 返回卡片。 |
| MCP vs A2A | "工具 vs 对等" | MCP：垂直 代理 ↔ 工具。A2A：水平 代理 ↔ 代理。 |
| ACP / ANP / NLIP | "兄弟协议" | 相邻规范；A2A 是 2026 年采用最广泛的。 |

## 延伸阅读 (Further Reading)

- [A2A specification](https://a2a-protocol.org/latest/specification/) — 权威规范
- [Google Developers Blog — A2A announcement](https://developers.googleblog.com/en/a2a-a-new-era-of-agent-interoperability/) — 2025 年 4 月发布文章
- [A2A GitHub repo](https://github.com/a2aproject/A2A) — 参考实现和 SDK
- [Liu et al. — A Survey of Agent Interoperability Protocols](https://arxiv.org/html/2505.02279v1) — MCP, ACP, A2A, ANP 比较