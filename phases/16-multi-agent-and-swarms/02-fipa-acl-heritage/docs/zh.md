# FIPA-ACL 与言语行为理论的传统

> 在 MCP 和 A2A 之前，有 FIPA-ACL。2000 年，IEEE 智能物理代理基金会（Foundation for Intelligent Physical Agents）批准了一种智能体通信语言，包含二十种施为语（performatives）、两种内容语言和一组交互协议——合同网、订阅/通知、条件请求。它从工业界淡出，因为本体论（ontology）开销对网络来说过于沉重，但 LLM 对多智能体系统的复兴正在悄悄地重新实现相同的想法，只是没有形式语义：JSON 契约代替了施为语，自然语言代替了本体论。本课程认真研读 FIPA-ACL，这样你就能看出哪些 2026 年的协议决策是重新发明，哪些是创新，以及当前浪潮将在哪里重新发现 2000 年代已经解决的问题。

**类型：** 学习
**语言：** Python（标准库）
**前置知识：** 阶段 16 · 01（为什么需要多智能体）
**时间：** ~60 分钟

## 问题

2026 年的智能体协议格局很热闹：MCP 用于工具，A2A 用于智能体，ACP 用于企业审计，ANP 用于去中心化信任，NLIP 用于自然语言内容，还有 CA-MCP 和二十多个研究提案。每个规范都宣称自己是基础性的。

诚实的解读是，它们大多在重新发现一个非常具体的、二十年前的决策树。言语行为理论（Speech-act theory）来自 Austin（1962）和 Searle（1969），告诉我们"话语就是行动"。KQML（1993）将其转化为线路协议。FIPA-ACL（2000 年批准）产生了参考标准化：二十种施为语、SL0/SL1 内容语言、用于合同网和订阅-通知的交互协议。JADE 和 JACK 是 Java 参考平台。这项工作在 2010 年左右逐渐消失，因为本体论开销太大，而网络正在胜出。

当你看到 MCP 的 `tools/call`、A2A 的任务生命周期或 CA-MCP 的共享上下文存储时，你看到的是更柔和、JSON 原生的 FIPA 决策重述。了解这段传统告诉你两件事：哪些新的"创新"实际上是重新发明，以及新规范将重新发现哪些旧的失败模式。

## 概念

### 言语行为，一段话概括

Austin 注意到有些句子不是在描述世界——它们是在改变世界。"我承诺。""我请求。""我宣布。"他称这些为施为性话语（performative utterances）。Searle 将其形式化为五个类别：断言（assertive）、指令（directive）、承诺（commissive）、表达（expressive）、宣告（declarative）。KQML（Finin 等人，1993）将其操作化用于软件智能体：一条消息是一个施为语（动作）加上内容（动作的对象）。FIPA-ACL 清理了 KQML 的空白，并围绕二十种施为语进行了标准化。

### 二十种 FIPA 施为语（部分列表）

| 施为语 | 意图 |
|--------|------|
| `inform` | "我告诉你 P 为真" |
| `request` | "我请求你做 X" |
| `query-if` | "P 为真吗？" |
| `query-ref` | "X 的值是多少？" |
| `propose` | "我提议我们做 X" |
| `accept-proposal` | "我接受这个提议" |
| `reject-proposal` | "我拒绝这个提议" |
| `agree` | "我同意做 X" |
| `refuse` | "我拒绝做 X" |
| `confirm` | "我确认 P 为真" |
| `disconfirm` | "我否认 P" |
| `not-understood` | "你的消息无法解析" |
| `cfp` | "就 X 征集提案" |
| `subscribe` | "当 X 变化时通知我" |
| `cancel` | "取消正在进行的 X" |
| `failure` | "我尝试了 X 但失败了" |

完整列表在 `fipa00037.pdf`（FIPA ACL 消息结构）中。重点不是记住它——重点是这些施为语中的每一个都对应一个 LLM 协议最终会重新添加的原语。

### 规范的 FIPA-ACL 消息

```
(inform
  :sender       agent1@platform
  :receiver     agent2@platform
  :content      "((price IBM 83))"
  :language     SL0
  :ontology     finance
  :protocol     fipa-request
  :conversation-id   conv-42
  :reply-with   msg-17
)
```

七个字段携带协议信封；一个字段（`content`）携带有效载荷。其余字段正是你每次将重试、线程和本体论塞进 JSON 协议时重新发明的东西。

### 两个传统平台

**JADE**（Java Agent DEvelopment framework，1999–2020 年代）是使用最广泛的 FIPA 兼容运行时。智能体扩展一个基类，交换 ACL 消息，在容器内运行，并使用"行为"进行协调。交互协议库内置了合同网、订阅-通知、条件请求和提议-接受。

**JACK**（Agent Oriented Software，商业）强调在 FIPA 消息之上的 BDI（信念-欲望-意图，Belief-Desire-Intention）推理。更形式化，采用更少。

两者都在网络栈吞噬了多智能体用例后衰落。MCP 和 A2A 是 2026 年的运行时"容器"。

### FIPA 为何衰落

- **本体论开销。** FIPA 需要一个共享的本体论来解析 `content`。就本体论达成一致是一个长达数年的标准化过程。网络只用了 HTTP + JSON。
- **没人用的形式语义。** SL（语义语言，Semantic Language）提供了严格的真值条件，但大多数生产系统使用自由形式的内容，忽略了形式主义。
- **工具锁定。** JADE 只支持 Java；JACK 是商业产品。多语言团队绕过了两者。
- **网络赢得了栈。** REST，然后是 JSON-RPC，然后是 gRPC 取代了 ACL 的传输层。

### LLM 复兴是 FIPA-lite

比较 FIPA 的 `request` 和 MCP 的 `tools/call`：

```
(request                                {
  :sender  agent1                         "jsonrpc": "2.0",
  :receiver tool-server                   "method":  "tools/call",
  :content "(lookup stock IBM)"           "params":  {"name":"lookup_stock",
  :ontology finance                                   "arguments":{"symbol":"IBM"}},
  :conversation-id c42                    "id": 42
)                                        }
```

相同的信封，不同的语法。两者都携带：谁、给谁、意图、有效载荷、关联 ID。两者之间没有革命——它们是同一设计上的不同权衡。

Liu 等人 2025 年的综述（"A Survey of Agent Interoperability Protocols: MCP, ACP, A2A, ANP"，arXiv:2505.02279）使这一谱系明确：MCP 对应工具使用施为语，A2A 对应智能体对等体施为语，ACP 对应审计追踪施为语，ANP 对应去中心化身份扩展。新规范是具有 JSON 语法和更宽松语义的 ACL 后代。

### 权衡，直说

**FIPA 给了你而现代规范放弃的：**

- 形式语义——你可以证明 `inform` 意味着发送者相信该内容。
- 施为语的规范目录——你不必重新争论"我们应该有 `cancel` 吗？"。
- 数十年的交互协议模式——合同网、订阅-通知、提议-接受——具有已知的正确性属性。

**现代规范给了你而 FIPA 没有的：**

- JSON 原生有效载荷，与所有现代工具兼容。
- LLM 无需手工编码本体论即可解释的自然语言内容。
- 网络栈传输（HTTP、SSE、WebSocket）。
- 通过自描述文档（MCP `listTools`、A2A Agent Card）进行能力发现。

更宽松的意图语义，换取更简单的实现。这就是确切的权衡。

### 值得移植的交互协议

FIPA 提供了约 15 个交互协议。三个值得带入 LLM 多智能体系统：

1. **合同网协议（CNP）。** 管理者发出 `cfp`（征集提案）；投标者以 `propose` 回应；管理者接受/拒绝。这是规范的任务市场模式（阶段 16 · 16 协商）。
2. **订阅/通知。** 订阅者发送 `subscribe`；每当主题变化时，发布者发送 `inform`。这就是 2026 年的每个事件总线。
3. **条件请求。** "当条件 Y 成立时做 X。"带前置条件的延迟动作。2026 年的类比是持久化工作流引擎中的延迟任务（阶段 16 · 22 生产扩展）。

每个都干净地映射到现代消息队列、HTTP + 轮询或 SSE 流。

### 放弃本体论会出什么问题

没有共享的本体论，智能体从自然语言内容中推断含义。2026 年有记录的失败模式是**语义漂移（semantic drift）**：两个智能体使用同一个词（"customer"）表示微妙不同的概念，接收智能体基于错误的解释行动，没有模式验证器能捕获它。FIPA 的本体论要求会在解析时拒绝该消息。

不完全采用本体论的缓解措施：

- 对 `content` 使用 JSON Schema——在线路层面拒绝结构错误。
- 类型化工件（A2A）——拒绝错误的模态。
- 在信封中显式声明施为语——即使内容是自然语言，也能使意图明确。

### 2026 年规范，映射到言语行为传统

| 现代规范 | FIPA 类比 | 保留了什么 | 放弃了什么 |
|---------|-----------|-----------|-----------|
| MCP `tools/call` | `request` | 显式意图、关联 ID | 形式语义、本体论 |
| MCP `resources/read` | `query-ref` | 显式意图、关联 ID | 形式语义 |
| A2A 任务生命周期 | 合同网 + 条件请求 | 异步生命周期、状态转换 | 形式完备性保证 |
| A2A 流式事件 | 订阅/通知 | 异步推送 | 类型化谓词订阅 |
| CA-MCP 共享上下文 | 黑板（Hayes-Roth 1985） | 多写者共享内存 | 逻辑一致性模型 |
| NLIP | 自然语言内容 | LLM 原生 | 模式 |

从上到下阅读表格，模式是：保留结构原语，放弃形式主义，让 LLM 用自然语言弥补歧义。

## 构建

`code/main.py` 实现了一个纯标准库的 FIPA-ACL 翻译器。它编码和解码规范的 ACL 信封，并展示每个 MCP / A2A 消息形状如何归结为相同的七个字段。演示：

- 将五条 MCP 风格和 A2A 风格的消息编码为 FIPA-ACL。
- 将 FIPA-ACL 解码回现代等价物。
- 使用 `cfp`、`propose`、`accept-proposal`、`reject-proposal` 运行一个管理者与三个投标者之间的玩具合同网协商。

运行：

```
python3 code/main.py
```

输出是一个并排追踪，显示每条现代消息的 2026 JSON 形式和 FIPA-ACL 形式，然后是一个合同网投标的往返。相同的协议原语在往返中存活；只有语法不同。

## 使用

`outputs/skill-fipa-mapper.md` 是一个技能，它读取任何智能体协议规范并生成 FIPA-ACL 映射。在采用新协议之前使用它来回答："这真的是新的，还是带有 JSON 语法的 `inform`？"

## 交付

不要带回 FIPA-ACL。带回它的检查清单：

- 每条消息的意图原语（施为语）是什么？
- 是否有用于请求-响应和取消的关联 ID？
- 是否有显式的内容语言（JSON-RPC、纯文本、结构化类型化工件）？
- 交互协议是一等公民，还是你在从头重新实现合同网？
- 当两个智能体对内容含义有分歧时会发生什么（语义漂移）？

在将任何新协议投入生产之前，记录这五个问题。

## 练习

1. 运行 `code/main.py`。观察往返编码。识别哪个 FIPA 施为语对应 `tools/call`、`resources/read` 和 A2A 任务创建。
2. 扩展合同网演示，添加一个 `cancel` 施为语，让管理者可以在投标进行中撤回任务。`cancel` 解决了哪些仅靠重试无法解决的失败情况？
3. 阅读 FIPA ACL 消息结构（http://www.fipa.org/specs/fipa00037/）第 4.1–4.3 节。选择一个本课程未涵盖的施为语，并描述其现代 JSON-RPC 类比。
4. 阅读 Liu 等人，arXiv:2505.02279。对于 MCP、A2A、ACP、ANP 中的每一个，列出它们保留和放弃的 FIPA 施为语家族。
5. 为你自己系统中 `request` 施为语的 `content` 字段设计一个最小的 JSON Schema。这个模式给了你纯自然语言没有的什么，以及它付出了什么代价？

## 关键术语

| 术语 | 人们说的 | 实际含义 |
|------|---------|---------|
| Speech act（言语行为） | "一种做事情的话语" | Austin/Searle：作为行动的话语。ACL 的理论父体。 |
| FIPA | "那个旧的 XML 东西" | IEEE 智能物理代理基金会。2000 年标准化了 ACL。 |
| ACL | "智能体通信语言" | FIPA 的信封格式：施为语 + 内容 + 元数据。 |
| Performative（施为语） | "动词" | 消息的意图类别：`inform`、`request`、`propose`、`cfp` 等。 |
| KQML | "FIPA 的前身" | 知识查询与操作语言（1993）。更简单，范围更窄。 |
| Ontology（本体论） | "共享词汇表" | 内容语言所讨论概念的形式化定义。 |
| SL0 / SL1 | "FIPA 内容语言" | 语义语言级别 0 和 1——形式化内容语言家族。 |
| Contract Net（合同网） | "任务市场" | 管理者发出 cfp；投标者提议；管理者接受。规范的交互协议。 |
| Interaction protocol（交互协议） | "消息模式" | 具有已知正确性的施为语序列：条件请求、订阅-通知等。 |

## 延伸阅读

- [Liu 等人 — A Survey of Agent Interoperability Protocols: MCP, ACP, A2A, ANP](https://arxiv.org/html/2505.02279v1) — 连接现代规范与 FIPA 传统的 2025 年规范综述
- [FIPA ACL Message Structure Specification (fipa00037)](http://www.fipa.org/specs/fipa00037/) — 2000 年批准的信封格式
- [FIPA Communicative Act Library Specification (fipa00037)](http://www.fipa.org/specs/fipa00037/) — 完整的施为语目录
- [MCP specification 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25) — 现代工具使用等价物，对应 `request`/`query-ref`
- [A2A specification](https://a2a-protocol.org/latest/specification/) — 现代智能体对等体等价物，对应合同网和订阅-通知