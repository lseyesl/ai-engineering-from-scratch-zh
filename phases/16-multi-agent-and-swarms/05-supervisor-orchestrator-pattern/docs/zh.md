# 监督者 / 编排者-工作者模式 (Supervisor / Orchestrator-Worker Pattern)

> 一个主导智能体负责规划和委派；专业工作者在并行上下文中执行并汇报。这是 Anthropic 研究系统（Claude Opus 4 为主导，Sonnet 4 为子智能体）背后的模式，在内部研究评测中比单智能体 Opus 4 高出 +90.2%。Anthropic 的工程博文指出，BrowseComp 上 80% 的方差仅由 token 用量解释——多智能体胜出的主要原因是每个子智能体都获得全新的上下文窗口。本课程从原语构建监督者模式，并涵盖 2026 年生产部署中的工程经验。

**类型：** 学习 + 构建 (Learn + Build)
**语言：** Python（标准库，`threading`）
**前置知识：** 阶段 16 · 04（原语模型）
**时间：** ~75 分钟

## 问题 (Problem)

研究是单智能体系统失败的典型任务。你问"2023 年到 2026 年间多智能体系统发生了什么变化？"一个智能体顺序阅读五篇论文，用它们的文本填满一半上下文，然后必须一起推理所有内容。读到第五篇时已经忘了第一篇。它无法并行化。

监督者模式解决了这个问题：一个主导智能体规划搜索，将每个子问题委派给一个工作者，然后综合结果。每个工作者针对一个狭窄问题拥有自己的 200k token 窗口。主导智能体从未看到原始论文——只看到工作者的摘要。

Anthropic 的生产研究系统报告称，在内部研究评测中比单个 Opus 4 高出 +90.2%。同一篇博文指出，BrowseComp 上 80% 的方差由 *token 用量* 解释。每个子智能体拥有全新上下文是主要机制。

## 概念 (Concept)

### 模式 (The pattern)

```
                 ┌──────────────┐
                 │   主导       │  规划、分解、
                 │  (Opus 4)    │  综合
                 └──┬────┬───┬──┘
                    │    │   │
            ┌───────┘    │   └───────┐
            ▼            ▼           ▼
      ┌─────────┐  ┌─────────┐  ┌─────────┐
      │ 工作者1  │  │ 工作者2  │  │ 工作者3  │
      │(Sonnet)  │  │(Sonnet)  │  │(Sonnet)  │
      └─────────┘  └─────────┘  └─────────┘
        全新         全新         全新
        上下文       上下文       上下文
```

主导智能体从不阅读原始材料。工作者在主导综合之前互不看到彼此的工作。每条箭头都是一次带有狭窄产物的交接。

### 为什么胜出 (Why it wins)

三种机制：

1. **每个子智能体拥有全新上下文 (Fresh context per subagent)。** 一个探索"FIPA-ACL 遗产"的工作者不会携带主导规划时花费的 40k token。它为一个问题获得 200k 窗口。
2. **通过提示实现专业化 (Specialization via prompt)。** 主导的提示是"分解并综合"，而不是"研究"。每个工作者的提示是狭窄的："找出 X 发生了什么变化。"聚焦的提示产生聚焦的输出。
3. **并行性 (Parallelism)。** 工作者并发运行。墙上时钟时间大致是 `max(worker_times) + plan + synthesis`，而不是 `sum(worker_times)`。

### 工程经验——Anthropic 2025 (Engineering lessons)

Anthropic 的博文列出了几条在 2026 年仍然相关的生产经验：

- **按查询复杂度调整投入 (Scale effort to query complexity)。** 简单查询：一个智能体，3-10 次工具调用。复杂查询：10+ 个智能体。主导必须自行评估，而不是由调用方决定。
- **先广后深 (Broad then narrow)。** 先分解为宽泛的子问题，如果答案需要深度，再为每个子问题生成更多工作者。
- **彩虹部署 (Rainbow deployments)。** 智能体是长期运行且有状态的。传统的蓝绿部署不适用。Anthropic 使用彩虹部署：新版本逐步推出，旧版本逐步排空。
- **Token 用量占主导 (Token usage dominates)。** 多智能体大约是单智能体的 15 倍 token。只在任务价值证明成本合理时才运行。

### LangGraph 的转向 (The LangGraph turn)

LangGraph 最初发布了一个 `langgraph-supervisor` 库，带有高级 `create_supervisor` 辅助函数。2025 年，LangChain 将推荐转向通过直接工具调用来实现监督者模式，因为工具调用能更好地控制 *监督者看到什么*（上下文工程）。该库仍然可用；文档现在推荐工具调用形式。

### 失败模式 (The failure modes)

- **主导幻觉规划 (Lead hallucinates the plan)。** 如果主导生成的子问题没有真正分解原始问题，工作者会在错误的目标上做精确的研究。
- **工作者过度探索 (Workers over-explore)。** 没有明确的边界约束，工作者会偏离分配的子问题，污染综合步骤。
- **综合冲突 (Synthesis conflicts)。** 两个工作者返回矛盾的事实。主导必须要么重新询问（增加一轮），要么明确记录分歧。默默选择一方是最糟糕的失败：用户永远不知道发生了分歧。

### 何时监督者不适用 (When supervisor is wrong)

- **顺序任务 (Sequential tasks)。** 如果步骤 2 确实需要步骤 1 的输出，并行性毫无意义。使用流水线（CrewAI Sequential，LangGraph 线性图）。
- **简单查询 (Simple queries)。** 单智能体处理更快更便宜。在生成工作者之前使用主导的"按复杂度调整投入"检查。
- **严格确定性 (Strict determinism)。** 监督者使用 LLM 选择的委派。当审计/回放比适应性更重要时，静态图更优。

```figure
supervisor-hierarchy
```

## 构建 (Build It)

`code/main.py` 使用 `threading` 实现了一个监督者和三个并行工作者。主导将查询分解为子问题，工作者在每个子问题上并发运行，主导综合结果。没有真实的 LLM——工作者是脚本化的，模拟获取和总结。

关键结构：

- `Lead.plan(query)` 将查询拆分为 3 个子问题。
- `Worker.run(sub_q)` 返回一个模拟摘要（生产环境中可以是任何使用工具的智能体）。
- `Lead.run(query)` 在线程中启动工作者，等待完成，然后综合。

运行：

```
python3 code/main.py
```

输出显示规划、并行工作者轨迹（带开始/结束时间戳）以及最终的综合结果。你可以看到墙上时钟的优势：三个 0.3 秒的工作者在约 0.35 秒内完成，而不是 0.9 秒。

## 使用 (Use It)

`outputs/skill-supervisor-designer.md` 接收用户查询并生成监督者模式设计：主导系统提示、工作者角色、子问题分解规则以及综合模板。在构建新的研究型智能体系统之前使用此工具。

## 交付 (Ship It)

部署监督者模式前的检查清单：

- **模型配对 (Model pairing)。** 主导使用推理级模型（Opus 类，`o3` 类）。工作者使用更快更便宜的模型（Sonnet，`o4-mini`）。
- **工作者超时 (Worker timeout)。** 任何运行时间超过 2 倍中位数的工作者被终止；主导要么重新生成（范围更窄），要么跳过它继续。
- **每个工作者的 token 上限 (Token cap per worker)。** 硬限制（比如预期综合输入的 10 倍）防止失控的工作者超出预算。
- **可观测性 (Observability)。** 追踪主导的规划、每个工作者的工具调用以及综合结果。这是任何事后调试的基础。
- **彩虹部署 (Rainbow rollout)。** 有状态的长期运行智能体需要渐进版本过渡，而不是热切换。

## 练习 (Exercises)

1. 运行 `code/main.py`，然后将主导修改为生成 5 个工作者而不是 3 个。观察墙上时钟效果。在这个演示中，工作者数量达到多少时生成开销超过并行节省？
2. 实现工作者超时：终止任何运行超过 0.5 秒的工作者，让主导综合剩余结果。你需要什么可观测性来知道工作者被截断了？
3. 在主导的综合步骤中添加冲突检测：如果两个工作者返回矛盾的答案，主导记录分歧而不是选择其中一个。如何在不调用 LLM 的情况下检测矛盾？
4. 阅读 Anthropic 的研究系统工程博文。列出这个玩具演示需要采用才能在生产环境中运行的三项实践。
5. 比较 LangGraph 的 `create_supervisor`（旧版）与新的工具调用推荐。哪个能更好地控制监督者看到什么？为什么 Anthropic 明确只传递子答案而不是原始工作者上下文到综合步骤？

## 关键术语 (Key Terms)

| 术语 | 人们说的 | 实际含义 |
|------|----------|----------|
| 监督者 (Supervisor) | "主导智能体" | 一个编排智能体，负责规划、委派和综合。自己不执行实际工作。 |
| 工作者 (Worker) | "子智能体" | 由监督者调用的聚焦智能体，具有狭窄范围和自己的上下文窗口。 |
| 编排者-工作者 (Orchestrator-worker) | "监督者模式" | 同一事物，不同名称。2026 年的文献两者都用。 |
| 全新上下文 (Fresh context) | "干净窗口" | 工作者的上下文从其系统提示和分配的问题开始，而不是主导的历史记录。 |
| 彩虹部署 (Rainbow deployment) | "渐进式推出" | 长期运行的有状态智能体需要版本化的排空和替换，而不是蓝绿部署。 |
| Token 主导 (Token dominance) | "上下文就是变量" | 根据 Anthropic，研究评测中 80% 的方差来自总 token 用量，而非模型选择。 |
| 按复杂度调整投入 (Scale effort) | "匹配智能体数量与复杂度" | 主导评估查询难度，相应地生成 1 个或 10+ 个工作者。 |
| 综合冲突 (Synthesis conflict) | "工作者意见不一" | 两个工作者返回矛盾的事实；主导必须呈现分歧，而不是默默选择其中一个。 |

## 延伸阅读 (Further Reading)

- [Anthropic engineering — How we built our multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system) —— 监督者模式的生产参考
- [LangGraph workflows and agents](https://docs.langchain.com/oss/python/langgraph/workflows-agents) —— 工具调用监督者现在是推荐形式
- [LangGraph supervisor reference](https://reference.langchain.com/python/langgraph-supervisor) —— 旧版辅助函数，2026 年仍在生产中使用
- [OpenAI cookbook — Orchestrating Agents: Routines and Handoffs](https://developers.openai.com/cookbook/examples/orchestrating_agents) —— 基于交接的监督者变体