# 并行工具调用与工具的流式传输

> 三次独立的天气查询串行执行需要三次往返。并行运行它们，总时间就缩减到最慢的单次调用的时间。每个前沿提供商现在都在一次回复中发出多个工具调用。回报是真实的；管道是微妙的。本课程讲解两个部分：并行扇出和流式参数重组，重点强调 ID 关联陷阱。

**类型：** 构建
**语言：** Python（标准库，线程池 + 流式测试工具）
**前置要求：** Phase 13 · 02（函数调用深入）
**时间：** ~75 分钟

## 学习目标

- 解释为什么 `parallel_tool_calls: true` 存在以及何时禁用它。
- 在并行扇出期间将流式参数块关联到正确的工具调用 ID。
- 将部分 `arguments` 字符串重组为完整的 JSON，而不提前解析。
- 运行一个三城市天气基准测试，展示串行与并行的延迟差异。

## 问题

没有并行调用，回答"班加罗尔、东京和苏黎世的天气如何"的智能体会这样做：

```
用户 -> LLM
LLM -> 调用 get_weather(班加罗尔)
主机 -> 运行执行器，用结果回复
LLM -> 调用 get_weather(东京)
主机 -> 运行执行器，用结果回复
LLM -> 调用 get_weather(苏黎世)
主机 -> 运行执行器，用结果回复
LLM -> 最终文本答案
```

三次 LLM 往返，每次还需要支付执行器延迟。大约是理想墙钟时间的 4 倍。

使用并行调用：

```
用户 -> LLM
LLM -> 调用 get_weather(班加罗尔); 调用 get_weather(东京); 调用 get_weather(苏黎世)
主机 -> 并发运行所有三个执行器，用三个结果回复
LLM -> 最终文本答案
```

一次 LLM 往返。执行器时间是最长的一个，不是总和。生产基准测试显示，在 OpenAI、Anthropic 和 Gemini 上，扇出工作负载的墙钟时间减少了 60% 到 70%。

代价是关联复杂性。当三个调用乱序完成时，你的结果必须携带匹配的 `tool_call_id`，以便模型能够对齐。当结果流式传输时，你必须在执行之前将部分参数片段组装成完整的 JSON。Gemini 3 添加了唯一 ID，部分原因是为了解决两个对同一工具的并行调用无法区分这一现实世界问题。

## 概念

### 启用并行

- **OpenAI。** `parallel_tool_calls: true` 默认开启。设置为 `false` 强制串行。
- **Anthropic。** 通过 `disable_parallel_tool_use: false` 实现并行（Claude 3.5 及以上版本默认）。设置为 `true` 为串行。
- **Gemini。** 始终支持并行；`tool_config.function_calling_config.mode = "AUTO"` 让模型决定。

当工具具有顺序依赖关系（`create_file` 然后 `write_file`）时、当一个调用的输出是另一个调用的输入时、或当速率限制器无法处理扇出时，禁用并行。

### ID 关联

模型发出的每个调用都有一个 `id`。主机返回的每个结果必须包含相同的 `id`。没有这个，结果就模棱两可。

- **OpenAI。** 每个工具角色消息上的 `tool_call_id`。
- **Anthropic。** 每个 `tool_result` 块上的 `tool_use_id`。
- **Gemini。** 每个 `functionResponse` 上的 `id`（Gemini 3 及以上版本；Gemini 2 按名称匹配，这在同名并行调用时出问题）。

### 并发运行调用

主机在每个调用的执行器上运行自己的线程、协程或远程工作器。最简单的测试工具使用线程池；生产环境使用带 `asyncio.gather` 的 asyncio 或结构化并发。完成顺序不可预测——id 就是标识符。

一个常见 bug：按调用列表顺序而不是完成顺序回复结果。这通常可以工作，因为模型只关心 `tool_call_id`，但如果结果被丢弃或重复，乱序提交会使调试更困难。倾向于按完成顺序使用显式 ID 回复。

### 流式工具调用

当模型流式传输时，`arguments` 分块到达。三个并行调用的三个独立流块在线上交错。你需要每个 ID 一个累加器。

各提供商的形状：

- **OpenAI。** 每个块是 `choices[0].delta.tool_calls[i].function.arguments`（部分字符串）。块携带 `index`（在调用列表中的位置）。你按索引累积，在 `id` 首次出现时读取它，并在 `finish_reason = "tool_calls"` 时解析 JSON。
- **Anthropic。** 流事件是 `message_start`，然后是每个块的 `content_block_start`（类型为 `tool_use`，包含 id、name、空 input）。`content_block_delta` 事件携带 `input_json_delta` 块。`content_block_stop` 关闭每个块。
- **Gemini。** `streamFunctionCallArguments`（Gemini 3 及以上版本）发出带有 `functionCallId` 的块，因此调用可以干净地交错。在 Gemini 3 之前，流式传输一次返回一个完整的调用。

### 部分 JSON 与提前解析陷阱

在 `arguments` 完成之前你不能解析它。部分 JSON 如 `{"city": "Beng` 无效且会抛出异常。正确的门控是提供商的调用结束信号：OpenAI 的 `finish_reason = "tool_calls"`、Anthropic 的 `content_block_stop` 或 Gemini 的流结束事件。只有那时才尝试 `json.loads`。更稳健的方法使用增量 JSON 解析器，在结构完成时产生事件；OpenAI 的流式指南推荐此方法用于显示实时"思考"指示器的 UX。括号计数作为完整性测试不可靠（引号字符串内或转义内容中的括号会导致误报），只应作为非正式调试启发式使用。

### 乱序完成

```
call_A: 快速 API，先返回
call_B: 慢速 API，其次返回
call_C: 中速 API，最后返回
```

主机回复仍须引用 ID：

```
[{role: "tool", tool_call_id: "call_A", content: ...},
 {role: "tool", tool_call_id: "call_B", content: ...},
 {role: "tool", tool_call_id: "call_C", content: ...}]
```

回复中的顺序对于 OpenAI 或 Anthropic 的正确性不重要。只要 ID 匹配，Gemini 接受任何顺序。

### 基准测试：串行 vs 并行

`code/main.py` 中的测试工具模拟了三个执行器，延迟分别为 400、600 和 800 毫秒。串行总时间为 1800 毫秒。并行运行时间为 max(400, 600, 800) = 800 毫秒。差异是恒定的而不是比例的，因此节省随工具数量增加而增长。

现实世界的注意事项：并行调用对下游 API 造成压力。对速率限制服务进行 10 路扇出会失败。Phase 13 · 17 涵盖网关级别的背压；重试语义在未来的阶段计划中。

### 流式扇出墙钟时间

如果模型本身是流式的，你可以在一个调用的参数完成后立即开始执行，而不是等待所有调用完成。这是 OpenAI 文档中介绍的一种优化，但并非所有 SDK 都暴露此功能。本课程中的测试工具就是这样做的：一旦模拟流产生了一个完整的参数对象，主机就启动该调用。

## 使用它

`code/main.py` 有两个部分。第一部分使用 `concurrent.futures.ThreadPoolExecutor` 串行和并行运行三个模拟天气调用，并打印墙钟时间。第二部分重放一个假的流式响应——三个并行调用的 `arguments` 块交错在一个流上——并使用 `StreamAccumulator` 按 ID 重组它们。没有 LLM，没有网络，只有重组逻辑。

关注点：

- 串行定时器达到 1.8 秒。并行定时器在相同的假延迟上达到 0.8 秒。
- 累加器通过按 ID 缓冲并在每个调用的 JSON 完成时才解析来处理乱序到达的块。
- 执行器在某个 ID 的参数完成后立即启动，而不是在所有流结束后。

## 交付物

本课程产生 `outputs/skill-parallel-call-safety-check.md`。给定一个工具注册中心，该技能审计哪些工具可以安全地并行化、哪些具有顺序依赖关系、哪些会压垮下游速率限制——返回一个带有每工具 `parallel_safe` 标志的修改后的注册中心。

## 练习

1. 运行 `code/main.py` 并改变模拟延迟。确认并行与串行的比率大约为 `max/sum`（实际运行由于线程调度、序列化和测试工具开销而与理想值略有偏差）。在什么延迟分布下并行不再重要？

2. 扩展累加器以处理"调用在流中间被取消"的情况，丢弃其缓冲区并发出一个 `cancelled` 事件。哪个提供商明确记录了这种情况？检查 Anthropic 的 `content_block_stop` 语义和 OpenAI 的 `finish_reason: "length"` 行为。

3. 用 `asyncio.gather` 替换线程池。对两者进行基准测试。由于较低的上下文切换成本，你应该在异步上看到小幅优势，但前提是执行器进行实际的 I/O 操作。

4. 选择两个不应该并行化的工具（例如 `create_file` 然后 `write_file`）。向注册中心添加一个 `ordering_dependency` 图，并在该图上门控并行扇出。这是依赖感知调度的最小机制，未来的智能体工程阶段将对其形式化。

5. 阅读 OpenAI 的并行函数调用部分和 Anthropic 的 `disable_parallel_tool_use` 文档。确定 Anthropic 建议禁用并行性的一个现实世界工具类型。（提示：对同一资源的后果性变更。）

## 关键术语

| 术语 | 人们说的 | 实际含义 |
|------|----------|----------|
| 并行工具调用 | "一次扇出" | 模型在一次助手消息中发出多个工具调用 |
| `parallel_tool_calls` | "OpenAI 的标志" | 启用或禁用多调用发出 |
| `disable_parallel_tool_use` | "Anthropic 的反向标志" | 选择退出标志；默认启用并行 |
| 工具调用 ID | "关联句柄" | 每个调用的标识符，结果消息必须回显 |
| 累加器 | "流缓冲区" | 每个 ID 的部分 `arguments` 块的字符串缓冲区 |
| 乱序完成 | "最快先完成" | 并行调用以不可预测的顺序完成；ID 是粘合剂 |
| 依赖图 | "排序约束" | 输出作为其他工具输入的工具；不能并行化 |
| 提前解析陷阱 | "JSON.parse 爆炸了" | 试图解析不完整的 `arguments` 字符串 |
| `streamFunctionCallArguments` | "Gemini 3 功能" | 每个调用具有唯一 ID 的流式参数块 |
| 完成顺序回复 | "不等全部" | 结果到达时按 ID 回复 |

## 延伸阅读

- [OpenAI — 并行函数调用](https://platform.openai.com/docs/guides/function-calling#parallel-function-calling) — 默认行为和退出标志
- [Anthropic — 工具使用：实现工具使用](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/implementing-tool-use) — `disable_parallel_tool_use` 和结果批处理
- [Google — Gemini 函数调用并行部分](https://ai.google.dev/gemini-api/docs/function-calling) — Gemini 3 中 ID 关联的并行调用
- [OpenAI — 带工具的流式响应](https://platform.openai.com/docs/api-reference/responses-streaming) — OpenAI 流的分块参数重组
- [Anthropic — 流式消息](https://docs.anthropic.com/en/api/messages-streaming) — 带有 `input_json_delta` 的 `content_block_delta`
