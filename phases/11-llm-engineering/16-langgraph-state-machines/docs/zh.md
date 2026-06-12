# LangGraph——代理的状态机

> 手写的 ReAct 循环是一个 `while True`。用 LangGraph 写的 ReAct 循环是一个你可以检查点、中断、分支和时间旅行的图。代理没有变。它周围的框架变了。

**类型：** Build
**语言：** Python
**前置要求：** 阶段 11 · 09（函数调用），阶段 11 · 14（模型上下文协议）
**预计时间：** ~75 分钟

## 问题

你发布了一个函数调用代理。它工作了三个轮次，然后出了问题：模型尝试返回 500 的工具，用户中途改变主意，或者代理决定在没有人工签署的情况下退款订单。`while True:` 循环没有任何钩子。你不能暂停它，不能回退它，也不能分支进入"如果模型选了另一个工具会怎样"。一旦你把它发布出演示阶段，代理就变成一个要么成功要么失败的黑箱。

下一步一旦你看清就显而易见。代理已经是一个状态机——系统提示加上消息历史加上待处理的工具调用加上下一个动作。使状态机显式化：节点包括"模型思考"、"工具运行"、"人工批准"，边是它们之间的条件转换。一旦图是显式的，框架免费获得四样东西：检查点（在步骤之间保存状态）、中断（暂停等待人工）、流式（流式传输 token 和中间事件）和时间旅行（回退到先前状态并尝试不同的分支）。

LangGraph 是实现这种抽象的库。它不是 LangChain 意义上的代理框架（"这里有个 AgentExecutor，祝你好运"）。它是一个图运行时，具有一等公民状态、一等公民持久化和一等公民中断。代理循环是你画出来的，而不是手写的。

## 概念

![LangGraph StateGraph：节点、边和检查点器](../assets/langgraph-stategraph.svg)

一个 `StateGraph` 有三样东西。

1. **状态。** 一个贯穿图的类型化字典（TypedDict 或 Pydantic 模型）。每个节点接收完整状态并返回部分更新，LangGraph 使用每个字段的*reducer*进行合并——对于应该累积的列表使用 `operator.add`，默认是覆盖。
2. **节点。** Python 函数 `state -> partial_state`。每个都是一个离散步骤："调用模型"、"运行工具"、"总结"。
3. **边。** 节点之间的转换。静态边去一个地方。条件边接受路由函数 `state -> next_node_name`，使图可以根据模型输出分支。

你编译图。编译绑定拓扑，附加检查点器（可选但对生产至关重要），并返回一个可运行对象。你用初始状态和 `thread_id` 调用它。每一步执行持久化一个以 `(thread_id, checkpoint_id)` 为键的检查点。

### 四大超能力

**检查点。** 每个节点转换将新状态写入存储（测试用内存，生产用 Postgres/Redis/SQLite）。通过相同的 `thread_id` 再次调用图来恢复。图从暂停处继续。

**中断。** 用 `interrupt_before=["human_review"]` 标记一个节点，执行在运行该节点之前停止。状态持久化。你的 API 以"等待批准"响应给用户。稍后对相同 `thread_id` 的请求，带上 `Command(resume=...)` 恢复执行。

**流式。** `graph.stream(state, mode="updates")` 在发生时产生状态增量。`mode="messages"` 流式传输模型节点内部的 LLM token。`mode="values"` 产生完整快照。你选择在你的 UI 中展示什么。

**时间旅行。** `graph.get_state_history(thread_id)` 返回完整的检查点日志。传递任何先前的 `checkpoint_id` 给 `graph.invoke`，你就从那个点分叉。非常适合调试（"如果模型选了工具 B 会怎样？"）和回归测试（重放生产轨迹）。

### Reducer 是关键

每个状态字段都有一个 reducer。大多数默认值就很好——新值覆盖旧值。但消息列表需要 `operator.add`，以便新消息追加而不是替换。并行边通过 reducer 合并它们的更新。如果两个节点都更新 `messages` 而你还记用了 `Annotated[list, add_messages]`，第二个静默获胜，你丢失了一半的轮次。Reducer 是库中唯一微妙的地方；把它弄对，其余的就组合起来了。

### 四个节点的 ReAct 图

生产 ReAct 代理有四个节点和两条边：

1. `agent`——用当前消息历史调用 LLM。返回助手消息（可能包含 tool_calls）。
2. `tools`——执行最后一条助手消息中的任何 tool_calls，将工具结果作为工具消息追加。
3. 从 `agent` 出发的条件边——如果最后一条消息有 tool_calls 则路由到 `tools`，否则到 `END`。
4. 从 `tools` 回到 `agent` 的静态边。

就这样。你得到了完整的 ReAct 循环（思考 → 行动 → 观察 → 思考 → ...）以及检查点、中断和流式，大约 40 行代码。

### StateGraph vs Send（扇出）

`Send(node_name, state)` 让节点分派并行子图。例如：代理决定同时查询三个检索器。每个 `Send` 生成目标节点的并行执行；它们的输出通过状态 reducer 合并。这是 LangGraph 表达编排器-工作者模式的方式，无需线程原语。

### 子图

编译后的图可以是另一个图中的一个节点。外部图看到一个节点；内部图有自己的状态和检查点。这就是团队构建监督者-工作者代理的方式：监督者图将用户意图路由到每个领域的工人子图。

## 构建它

### 步骤 1：状态和节点

```python
from typing import Annotated, TypedDict
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

def agent_node(state: State) -> dict:
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

def should_continue(state: State) -> str:
    last = state["messages"][-1]
    return "tools" if getattr(last, "tool_calls", None) else END

tool_node = ToolNode(tools=[search_web, read_file])

graph = StateGraph(State)
graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)
graph.set_entry_point("agent")
graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
graph.add_edge("tools", "agent")

app = graph.compile(checkpointer=MemorySaver())
```


`add_messages` 是使消息列表累积而不是覆盖的 reducer。忘记它是最常见的 LangGraph 错误。

### 步骤 2：使用线程运行
```python
config = {"configurable": {"thread_id": "user-42"}}
for event in app.stream(
    {"messages": [HumanMessage("find the Anthropic headquarters address")]},
    config,
    stream_mode="updates",
):
    print(event)
```


每次更新是一个字典 `{node_name: state_delta}`。你的前端可以将这些流式传输到 UI，让用户看到"代理正在思考…调用 search_web…获取结果…正在回答。"

### 步骤 3：添加人机循环中断

标记一个节点，使执行在运行前暂停。
```python
app = graph.compile(
    checkpointer=MemorySaver(),
    interrupt_before=["tools"],  # pause before every tool call
)

state = app.invoke({"messages": [HumanMessage("delete the production database")]}, config)
# state["__interrupt__"] is set. Inspect proposed tool calls.
# If approved:
from langgraph.types import Command
app.invoke(Command(resume=True), config)
# If denied: write a rejection message and resume
app.update_state(config, {"messages": [AIMessage("Blocked by human reviewer.")]})
```


状态、检查点和线程都在中断期间持久化。除执行期间外，没有任何内容在内存中。

### 步骤 4：用于调试的时间旅行
```python
history = list(app.get_state_history(config))
for snapshot in history:
    print(snapshot.values["messages"][-1].content[:80], snapshot.config)

# Fork from a prior checkpoint
target = history[3].config  # three steps back
for event in app.stream(None, target, stream_mode="values"):
    pass  # replay from that point forward
```

Passing `None` as the input replays from the given checkpoint; passing a value appends it as an update to that checkpoint's state before resuming. This is how you reproduce a bad agent run without re-running the whole conversation.


传递 `None` 作为输入从给定的检查点重放；传递一个值在恢复前将其作为更新追加到该检查点的状态。这就是你复现代理错误运行的方式，无需重新运行整个对话。
```python
from langgraph.checkpoint.postgres import PostgresSaver

with PostgresSaver.from_conn_string("postgresql://...") as checkpointer:
    checkpointer.setup()
    app = graph.compile(checkpointer=checkpointer)
```


SQLite、Redis 和 Postgres 都已发布。`MemorySaver` 用于测试。任何需要跨重启持久化的东西都想要真实的存储。

## 技能

> 你将代理构建为图，而不是 `while True` 循环。

在你使用 LangGraph 之前，做一个 60 秒的设计：

1. **命名节点。** 每个离散决策或副作用动作是一个节点。"代理思考"、"工具运行"、"评审者批准"、"响应流式传输"。如果你列不出来，任务还没有代理的形状。
2. **声明状态。** 最小的 TypedDict，每个列表字段有一个 reducer。不要把所有东西都塞进 `messages`；将任务特定字段（工作计划 `plan`、预算计数器 `budget`、检索文档列表 `retrieved_docs`）提升到顶层。
3. **绘制边。** 除非下一步取决于模型输出，否则使用静态边。每个条件边需要一个带有命名分支的路由函数。
4. **提前选择检查点器。** `MemorySaver` 用于测试，Postgres/Redis/SQLite 用于其他任何场景。没有检查点器就不要发布——没有检查点器意味着没有恢复、没有中断、没有时间旅行。
5. **在工具运行前决定中断，而不是之后。** 批准放在进入副作用节点的边上，这样你可以在造成伤害前取消；验证放在模型输出的边上，这样你可以廉价拒绝错误调用。
6. **默认流式传输。** `mode="updates"` 用于 UI，`mode="messages"` 用于模型节点内的 token 级流式传输，`mode="values"` 用于评估期间的完整快照。

拒绝发布没有检查点器的 LangGraph 代理。拒绝发布在副作用*之后*中断的代理。拒绝发布没有 `add_messages` 作为 reducer 的 `messages` 字段。

## 练习

1. **简单。** 实现上述四节点 ReAct 图，带一个计算器工具和一个网络搜索工具。验证 `list(app.get_state_history(config))` 对于两轮对话返回至少四个检查点。
2. **中等。** 添加一个在 `agent` 之前运行的 `planner` 节点，将结构化的 `plan: list[str]` 写入状态。让 `agent` 标记计划步骤为完成。如果 `plan` 在检查点恢复时丢失（错误的 reducer），测试失败。
3. **困难。** 构建一个监督者图，使用 `Send` 在三个子图（`researcher`、`writer`、`reviewer`）之间路由。每个子图有自己的状态和检查点器。在外部图上添加 `interrupt_before=["writer"]`，使人可以批准研究简报。确认从先前检查点的时间旅行只重新运行分叉的分支。

## 关键术语

| 术语 | 人们说的 | 实际含义 |
|------|-----------------|-----------------------|
| StateGraph | "LangGraph 的图" | 编译前添加节点和边的构建器对象。 |
| Reducer | "字段如何合并" | 当节点返回该字段的更新时应用的函数 `(old, new) -> merged`；默认是覆盖，`add_messages` 追加。 |
| 线程 | "对话 ID" | 一个 `thread_id` 字符串，作用域为一次会话的所有检查点。 |
| 检查点 | "暂停的状态" | 节点转换后完整图状态的持久化快照，以 `(thread_id, checkpoint_id)` 为键。 |
| 中断 | "等待人工" | `interrupt_before` / `interrupt_after` 在节点边界停止执行；用 `Command(resume=...)` 恢复。 |
| 时间旅行 | "从先前步骤分叉" | `graph.invoke(None, config_with_old_checkpoint_id)` 从该检查点向前重放。 |
| Send | "并行子图分发" | 节点可以返回的构造函数，以生成目标节点的 N 个并行执行。 |
| 子图 | "作为节点的编译图" | 用作另一个图中节点的编译 StateGraph；保持自己的状态范围。 |

## 延伸阅读

- [LangGraph 文档](https://langchain-ai.github.io/langgraph/)——StateGraph、reducer、检查点器和中断的权威参考。
- [LangGraph 概念：状态、reducer、检查点器](https://langchain-ai.github.io/langgraph/concepts/low_level/)——本课程使用的思维模型，直接来自官方来源。
- [LangGraph 持久化和检查点](https://langchain-ai.github.io/langgraph/concepts/persistence/)——关于 Postgres/SQLite/Redis 存储、检查点命名空间和线程 ID 的详细信息。
- [LangGraph 人机循环](https://langchain-ai.github.io/langgraph/concepts/human_in_the_loop/)——`interrupt_before`、`interrupt_after`、`Command(resume=...)` 和编辑状态模式。
- [Yao 等人, "ReAct: Synergizing Reasoning and Acting in Language Models" (ICLR 2023)](https://arxiv.org/abs/2210.03629)——每个 LangGraph 代理实现的模式；阅读它以了解推理轨迹原理。
- [Anthropic——构建有效代理（2024 年 12 月）](https://www.anthropic.com/research/building-effective-agents)——哪种图形形状（链、路由器、编排器-工作者、评估器-优化器）更适合什么场景。
- 阶段 11 · 09（函数调用）——每个 LangGraph 代理节点重用的工具调用原语。
- Phase 11 · 17 (Agent framework tradeoffs) — when to pick LangGraph over CrewAI, AutoGen, or Agno.
