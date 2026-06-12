# 群聊与发言者选择 (Group Chat and Speaker Selection)

> AutoGen GroupChat 和 AG2 GroupChat 在 N 个代理之间共享一个对话；一个选择器函数（LLM、轮询或自定义）决定谁下一个发言。这是涌现式多代理对话的原型——代理不知道自己在静态图中的角色，它们只是对共享池做出反应。AutoGen v0.2 的 GroupChat 语义在 AG2 分支中得到保留；AutoGen v0.4 将其重写为事件驱动的 actor 模型。Microsoft 于 2026 年 2 月将 AutoGen 置于维护模式，并将其与 Semantic Kernel 合并为 Microsoft Agent Framework（2026 年 2 月 RC）。GroupChat 原语在 AG2 和 Microsoft Agent Framework 中都得以保留——一次学习，随处使用。

**类型:** Learn + Build
**语言:** Python (stdlib)
**前置知识:** Phase 16 · 04 (原始模型)
**时间:** ~60 分钟

## 问题 (Problem)

静态图 (LangGraph) 在工作流已知时效果很好。但真实对话不是静态的：有时编码者问审查者，有时问研究者，有时问写作者。硬编码每一种可能的交接会产生边爆炸。你需要的是*代理对共享池做出反应*，由某个函数决定谁下一个说话。

这正是 AutoGen GroupChat 所做的。

## 概念 (Concept)

### 结构 (The shape)

```
              ┌─── shared pool ────┐
              │   m1  m2  m3  ...  │
              └─────────┬──────────┘
                        │ (everyone reads all)
      ┌───────┬─────────┼─────────┬───────┐
      ▼       ▼         ▼         ▼       ▼
    Agent A  Agent B  Agent C  Agent D  Selector
                                           │
                                           ▼
                                  "next speaker = C"
```

每个代理看到每条消息。在每个轮次调用一个选择器函数来决定谁下一个发言。

### 三种选择器风格 (The three selector flavors)

**轮询(Round-robin)。** 固定循环。确定性。随 N 线性扩展，但忽略上下文——即使话题是法律审查，编码者也会获得发言权。

**LLM 选择(LLM-selected)。** 调用 LLM 读取最近的对话池并返回最佳的下一个发言者。上下文感知但速度慢：每个轮次增加一次 LLM 调用。AutoGen 的默认方式。

**自定义(Custom)。** 一个 Python 函数，包含你想要的任何逻辑。典型做法：LLM 选择加回退规则（例如，"总是在编码者之后给验证者发言权"）。

### ConversableAgent API

```
agent = ConversableAgent(
    name="coder",
    system_message="You write Python.",
    llm_config={...},
)
chat = GroupChat(agents=[coder, reviewer, tester], messages=[])
manager = GroupChatManager(groupchat=chat, llm_config={...})
```

`GroupChatManager` 持有选择器。当一个代理完成一轮时，管理器调用选择器，选择器返回下一个代理。循环持续直到终止条件。

### 终止 (Termination)

三种常见模式：

- **最大轮次(Max rounds)。** 总轮数的硬上限。
- **"TERMINATE" 令牌(Token)。** 代理可以发出哨兵消息；管理器在出现该消息时停止。
- **目标达成检查(Goal-reached check)。** 一个轻量级验证器在每个轮次运行，并在完成时停止对话。

### AutoGen → AG2 分裂与 Microsoft Agent Framework 合并

2025 年初，Microsoft 开始围绕事件驱动的 actor 模型对 AutoGen (v0.4) 进行重大重写。社区将 AutoGen v0.2 的 GroupChat 语义分支为 AG2，保留了早期采用者已集成的 API。

2026 年 2 月，Microsoft 宣布 AutoGen 将进入维护模式，事件驱动的 actor 模型将合并到 **Microsoft Agent Framework**（2026 年 2 月 RC，现已与 Semantic Kernel 合并）。GroupChat 概念在两个轨道中都得以保留；实现细节有所不同。AG2 是 v0.2 兼容代码的首选上游。

### GroupChat 何时适用 (When GroupChat fits)

- **涌现式对话(Emergent conversations)。** 你不想预先连接每一个可能的下一发言者。
- **角色混合任务(Role-mixing tasks)。** 编码者问研究者，研究者问档案员，档案员又问回编码者。流程不是 DAG。
- **探索性问题求解(Exploratory problem-solving)。** 想象"头脑风暴会议"，而不是"流水线"。

### 何时失败 (When it fails)

- **严格确定性(Strict determinism)。** LLM 选择器可能不一致。相同提示，不同运行，不同下一发言者。
- **谄媚级联(Sycophancy cascades)。** 代理顺从发言最自信的人。显式地加入反制提示。
- **上下文膨胀(Context bloat)。** 每个代理读取每条消息；10 轮后上下文变得巨大。使用投影(Projection)（第 15 课）来限定视图范围。
- **热门发言者(Hot speakers)。** 一个代理主导对话，因为选择器偏向其专长。引入发言者平衡作为选择器的一个特性。

### 群聊 vs 监督者 (Group chat vs supervisor)

相同的原语，不同的默认值：

- 监督者：一个代理规划，其他代理执行。选择器是"问规划者做什么"。
- 群聊：所有代理是对等的；选择器是对共享池的一个函数。

两者都使用第 04 课的四个原语。群聊默认使用 LLM 选择的编排和全池共享状态。

## 构建 (Build It)

`code/main.py` 使用 stdlib 从头实现了一个 GroupChat。三个代理（编码者、审查者、管理者），轮询和 LLM 选择两种变体，以及在 `TERMINATE` 令牌上的终止。

该演示打印对话记录以及两种变体的选择器决策轨迹。

运行：

```
python3 code/main.py
```

## 使用 (Use It)

`outputs/skill-groupchat-selector.md` 为给定任务配置一个 GroupChat 选择器——轮询 vs LLM 选择 vs 自定义，以及使用哪些选择器输入（最近消息、代理专长、轮次计数）。

## 交付 (Ship It)

检查清单：

- **最大轮次上限(Max rounds cap)。** 始终设置。典型任务 10-20 轮。
- **发言者平衡指标(Speaker-balance metric)。** 追踪每个代理的轮次；当不均衡超过阈值时发出警报。
- **终止令牌(Termination token)。** `TERMINATE` 或一个专门的验证者代理。
- **投影或限定内存(Projection or scoped memory)。** 大约 10 条消息后，考虑给每个代理只提供一个限定视图以防止上下文膨胀。
- **选择器日志(Selector logging)。** 对于 LLM 选择的变体，记录选择器的输入和选择。否则调试是不可能的。

## 练习 (Exercises)

1. 运行 `code/main.py`。比较轮询 vs LLM 选择下的对话。哪种模式下哪个代理占主导？
2. 在选择器中添加一个"每个代理最大发言次数"规则。它如何影响对话记录？
3. 实现一个目标达成终止：当审查者返回"approved"时停止。在达到轮次上限之前它多久触发一次？
4. 阅读 AutoGen stable 文档中关于 GroupChat 的部分 (https://microsoft.github.io/autogen/stable/user-guide/core-user-guide/design-patterns/group-chat.html)。找出 `GroupChatManager` 使用的默认选择器。
5. 阅读 AG2 仓库 (https://github.com/ag2ai/ag2) 并比较其 v0.2 GroupChat 与 v0.4 事件驱动版本。v0.4 增加了什么具体属性（吞吐量、容错性、可组合性）？

## 关键术语 (Key Terms)

| 术语 | 人们说的 | 实际含义 |
|------|----------|----------|
| GroupChat (群聊) | "代理在一个聊天室" | 共享消息池 + 选择器函数。AutoGen / AG2 原语。 |
| Speaker selection (发言者选择) | "谁下一个说话" | 选择下一个代理的函数。轮询、LLM 选择或自定义。 |
| GroupChatManager (群聊管理器) | "会议主持人" | AutoGen 组件，拥有选择器并在轮次上循环。 |
| ConversableAgent (可对话代理) | "基础代理" | AutoGen 基类；可以发送和接收消息的代理。 |
| Termination token (终止令牌) | "'停止'词" | 哨兵字符串（通常是 `TERMINATE`），结束对话。 |
| Hot speaker (热门发言者) | "一个代理主导" | 选择器持续选择同一个代理的失败模式。 |
| Context bloat (上下文膨胀) | "池无限增长" | 每个代理读取所有先前消息；上下文随轮次增长。 |
| Projection (投影) | "限定视图" | 按角色限定共享池视图以防止上下文膨胀。 |

## 延伸阅读 (Further Reading)

- [AutoGen group chat docs](https://microsoft.github.io/autogen/stable/user-guide/core-user-guide/design-patterns/group-chat.html) — 参考实现
- [AG2 repo](https://github.com/ag2ai/ag2) — 社区 AutoGen v0.2 延续
- [Microsoft Agent Framework docs](https://microsoft.github.io/agent-framework/) — 合并后的继任者，2026 年 2 月 RC
- [AutoGen v0.4 release notes](https://microsoft.github.io/autogen/stable/) — 事件驱动 actor 模型重写详情