# 共享内存与黑板模式 (Shared Memory and Blackboard Patterns)

> 2026 年的多智能体系统中，两种方法并存：**消息池 (message pool)**（每个人都能看到所有人的消息，如 AutoGen GroupChat 或 MetaGPT）和**带订阅的黑板 (blackboard with subscription)**（智能体订阅相关事件，如 Context-Aware MCP 或 Matrix 框架）。两者都是多智能体系统中唯一的有状态部分——也就是说，两者都是有趣 bug 的藏身之处。参考故障模式是**内存投毒 (memory poisoning)**：一个智能体幻觉出一个"事实"，其他智能体将其视为已验证，准确率逐渐衰减，这种衰减比即时崩溃更难调试。本课程用 stdlib 构建这两种结构，注入投毒攻击，并展示三种在生产中实际有效的缓解措施。

**Type:** Learn + Build
**Languages:** Python (stdlib, `threading`)
**Prerequisites:** Phase 16 · 04 (Primitive Model), Phase 16 · 09 (Parallel Swarm Networks)
**Time:** ~75 minutes

## 问题 (Problem)

多智能体系统需要一个地方让智能体共享事实。一个直观的选择是"在消息中传递一切"——但这用额外的复制重新发明了共享状态。另一种是"给每个人一个全局日志"——但全局日志会无界增长且容易投毒。第三种是"为每个智能体投影一个视图"——可扩展但模式负担重。

当一个智能体产生幻觉并将幻觉写入共享状态时，每个读取该状态的下游智能体都会将幻觉当作事实接受。等到人类注意到时，推理链已经深入五步，根本原因是最初写入的第三条消息。调试多智能体的准确率衰减比调试崩溃更难。

这就是内存投毒。它是 MAST 分类法（Cemri 等人，arXiv:2503.13657）中记录第二多的故障家族，并且是结构性的：任何没有溯源 (provenance) 和不可写入验证器 (unwritable verifier) 的共享内存设计最终都会出现它。

## 概念 (Concept)

### 两种主要拓扑 (The two main topologies)

**全量消息池 (Full message pool)。** 每个智能体读取每条消息。AutoGen GroupChat 和 MetaGPT 使用这种方式。简单、透明、可检查，但无法扩展到约 10 个智能体以上，因为每个智能体的上下文会被其他智能体的工作填满。

```
agent-A ──write──▶ ┌────────────────┐ ◀──read── agent-D
                    │ message pool   │
agent-B ──write──▶ │                │ ◀──read── agent-E
                    │ (global log)   │
agent-C ──write──▶ └────────────────┘ ◀──read── agent-F
```

**带订阅的黑板 (Blackboard with subscription)。** 智能体声明对主题的兴趣；底层路由只传递相关消息。CA-MCP（arXiv:2601.11595）和 Matrix 去中心化框架（arXiv:2511.21686）使用这种方式。扩展性更好，但需要预先设计模式以使订阅有意义。

```
                   ┌─ topic: prices ──┐
agent-A ──pub────▶ │                  │ ──▶ agent-D (subscribed)
                   ├─ topic: orders ──┤
agent-B ──pub────▶ │                  │ ──▶ agent-E (subscribed)
                   ├─ topic: alerts ──┤
agent-C ──pub────▶ │                  │ ──▶ agent-F (subscribed)
                   └──────────────────┘
```

### 各自何时胜出 (When each wins)

- **全量池**在智能体数量少（< 10）、异构、对话短周期时胜出。当每个人都能看到一切时，推理谁说了什么是微不足道的。
- **黑板**在智能体数量多、角色同质但实例众多（群体）、对话长期运行时胜出。路由节省了 token 成本和上下文污染。

生产系统通常混合使用：顶层使用小型全量池（规划层），下层使用黑板（工作层）。

### 内存投毒，一个场景 (Memory poisoning, in one scenario)

三个智能体处理一个研究任务。智能体 A 是检索智能体。智能体 B 是摘要生成器。智能体 C 是分析师。

1. A 获取一个页面并写入共享状态："该研究报告了 42% 的准确率提升。"
2. 获取的页面实际上说的是"4.2% 的提升。"A 幻觉了一个小数点。
3. B 读取共享状态，写入："报告了 42% 的大幅准确率提升（来源：A）。"
4. C 读取共享状态，写入："建议采纳——42% 的提升是变革性的。"
5. 最终报告引用了一个从未存在过的 42% 数字。

没有智能体崩溃。没有测试失败。系统"正常工作"。幻觉通过共享状态从一个智能体的上下文传播到了每个下游智能体的推理中。

### 为什么这是结构性的 (Why this is structural)

没有共享状态，智能体 A 的幻觉就停留在 A 的上下文中。下游智能体会重新获取或重新推导，可能会发现错误。有了天真的共享状态，A 的上下文就成了每个人的上下文，幻觉被洗白成了事实。

问题不在于共享状态本身——而在于**没有溯源和没有独立验证器**的共享状态。三种缓解措施解决了这个问题：

1. **为每次写入附加溯源 (Attribute provenance on every write)。** 共享状态中的每条记录都记录了谁写的、何时写的、在什么提示下写的，以及（如果适用）智能体引用了什么来源。下游智能体以基于溯源的怀疑态度读取。
2. **版本化写入；将其视为仅追加 (Version writes; treat them as append-only)。** 更正是一条取代旧条目的新条目，而不是原地更新。审计轨迹得以保留。
3. **至少保留一个不能写入共享状态的智能体 (Keep at least one agent that cannot write to shared state)。** 一个只读验证器智能体采样条目、重新获取来源并标记不一致。因为它不能写入池，所以不会被池投毒。

### 黑板先例 (Blackboard precedent, Hayes-Roth, 1985)

黑板模式比 LLM 智能体早了四十年。Hayes-Roth（1985，"A Blackboard Architecture for Control"）描述了观察全局黑板、贡献部分解决方案并触发其他来源的专业知识源 (Knowledge Sources)。2026 年的黑板（CA-MCP、Matrix）是相同的模式，只是将 LLM 智能体作为知识源，将 JSON 数据块作为部分解决方案。旧文献已经记录了写入争用、机会性控制和一致性的解决方案，现代系统正在重新发现这些方案。

### 投影与全视图 (Projection vs full view)

纯黑板给每个订阅者相同的投影（按主题范围限定）。一种更激进的设计是**按智能体投影 (per-agent projection)**：每个智能体获得一个根据其角色定制的视图。LangGraph 的状态归约器 (state reducers) 是 2026 年的典型实现——归约函数将全局状态折叠成角色特定的切片。

按智能体投影扩展性更好，但需要模式。没有模式，你就要在每个智能体的提示中临时构建投影。

### 写入争用模式 (Write-contention patterns)

多个智能体同时写入是一个并发问题，而不仅仅是 LLM 问题。三种模式有效：

- **顺序写入器（单一生产者）(Sequential writer)。** 所有写入通过一个协调智能体串行化。简单，但会成为瓶颈。
- **带版本化的乐观并发 (Optimistic concurrency with versioning)。** 每条记录都有一个版本；写入者在版本不匹配时失败并重试。经典的数据库技术。
- **主题分区 (Topic partitioning)。** 不同的智能体拥有不同的主题。没有跨主题争用。需要设计分区边界。

大多数 2026 框架默认使用顺序写入器，因为 LLM 调用足够慢，争用很少发生，瓶颈不会造成伤害。

### 不可写入验证器 (The unwritable verifier)

最关键的缓解措施是只读验证器。实现规则：

- 验证器与团队共享状态（读取黑板或池）。
- 验证器没有共享状态的写入句柄——只有到一个单独验证通道的句柄。
- 验证器独立获取写入中引用的来源。标记不一致。
- 验证器自身的输出路由到人类或一个单独的决策智能体，永远不会反馈到池中。

没有这种分离，验证器的输出会成为池中的新条目，这意味着被投毒的池会投毒验证器，进而投毒其验证结果。

## 构建 (Build It)

`code/main.py` 使用 stdlib Python 实现了两种拓扑，外加一个玩具投毒攻击和三种缓解措施。

- `MessagePool` — 线程安全的仅追加日志，支持完整读取。
- `Blackboard` — 按主题键的发布/订阅，支持按智能体订阅。
- `ProvenanceEntry` — 每次写入记录 (writer, timestamp, prompt_hash, source_uri)。
- `PoisoningScenario` — 运行一个三智能体研究任务，其中智能体 A 幻觉了一个小数点。打印最终报告。
- `Verifier` — 一个只读智能体，重新获取来源并标记不一致。在验证器存在的情况下运行相同的场景。

运行：

```
python3 code/main.py
```

预期输出：
- 运行 1（无验证器）：幻觉的 42% 传播到最终报告。
- 运行 2（有验证器）：验证器标记不一致，池被标记为"flagged"，最终报告包含撤回。

## 使用 (Use It)

`outputs/skill-memory-auditor.md` 是一个技能，用于审计任何多智能体系统的共享内存设计，检查溯源、版本化和验证器分离。在生产前对新多智能体架构运行它。

## 交付 (Ship It)

对于任何共享内存设计：

- 在每次写入时记录溯源：`(writer, timestamp, prompt_hash, tool_calls_cited, source_uri)`。
- 使日志仅追加。更正是引用被取代条目的新条目。
- 至少部署一个具有独立来源访问权限的只读验证器智能体。
- 将验证器输出路由到单独的通道，而不是返回共享池。
- 记录被取代写入的比例——上升的比例是幻觉模式的早期证据。

## 练习 (Exercises)

1. 运行 `code/main.py`。确认运行 1 传播了幻觉，运行 2 捕获了它。
2. 添加第二个幻觉：智能体 B 虚构了一个数据集大小。验证器应该在不针对任何一个进行手工调整的情况下捕获两者。
3. 将全量池切换为带主题分区 (`prices`, `summaries`, `analyses`) 的黑板。主题分区使哪些投毒场景更难实施，对哪些场景没有帮助？
4. 阅读 Hayes-Roth（1985，"A Blackboard Architecture for Control"）。找出论文中本课程未讨论的两个控制模式，2026 系统会从中受益。
5. 阅读 CA-MCP（arXiv:2601.11595）。将其 Shared Context Store 映射到 `code/main.py` 中的 MessagePool 或 Blackboard 类。CA-MCP 在此基础上添加了哪些原语？

## 关键术语 (Key Terms)

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Message pool | "共享聊天历史" | 仅追加日志，每个智能体都读取。完全透明，扩展性差。 |
| Blackboard | "共享工作空间" | 按主题键的发布/订阅。智能体订阅相关主题。扩展性更好。 |
| Provenance | "谁写了什么" | 每次写入的元数据：写入者、时间戳、提示、来源。 |
| Memory poisoning | "幻觉传播" | 一个智能体的错误进入共享状态，下游智能体将其当作事实接受。 |
| Append-only | "无原地更新" | 更正是取代旧条目的新条目。保留审计轨迹。 |
| Unwritable verifier | "独立审计员" | 只读智能体，重新获取来源并标记不一致。 |
| Projection | "限定范围的视图" | 从全局状态计算出的按智能体视图。LangGraph 归约器是典型例子。 |
| Knowledge Source | "专家智能体" | Hayes-Roth 1985 年对黑板参与者的术语。 |

## 延伸阅读 (Further Reading)

- [Cemri et al. — Why Do Multi-Agent LLM Systems Fail?](https://arxiv.org/abs/2503.13657) — MAST 分类法；内存投毒是协调失败的一个子家族
- [CA-MCP — Context-Aware Multi-Server MCP](https://arxiv.org/abs/2601.11595) — 用于协调 MCP 服务器的共享上下文存储
- [Matrix — decentralized multi-agent framework](https://arxiv.org/abs/2511.21686) — 基于消息队列的黑板，无中央编排器
- [LangGraph state and reducers](https://docs.langchain.com/oss/python/langgraph/workflows-agents) — 生产中的按智能体投影模式
- [Anthropic — How we built our multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system) — 来自生产部署的溯源和验证笔记