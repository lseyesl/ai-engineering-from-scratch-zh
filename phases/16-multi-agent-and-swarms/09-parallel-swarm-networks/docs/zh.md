# 并行 / 群集 / 网络化架构 (Parallel / Swarm / Networked Architectures)

> 与监督者模式对比：没有中央决策者。代理读取共享事件总线，异步领取任务，将结果写回。LangGraph 明确支持"群集架构"(Swarm Architecture)，用于去中心化、动态环境。Matrix (arXiv:2511.21686) 将控制流和数据流都表示为通过分布式队列传递的序列化消息，以消除编排者瓶颈。其权衡是明确的：用确定性和可追溯性换取可扩展性。群集适合包含许多独立子问题的任务；不适合需要单一连贯计划的任务。

**类型:** Learn + Build
**语言:** Python (stdlib, `threading`, `queue`)
**前置知识:** Phase 16 · 05 (监督者模式), Phase 16 · 04 (原始模型)
**时间:** ~75 分钟

## 问题 (Problem)

监督者模式可以扩展到几个工作节点。但几百个呢？监督者本身就成了瓶颈：关于谁做什么的每个决策都通过一个代理漏斗。一个缓慢的计划步骤就会拖垮整个系统。

群集架构(Swarm architectures)翻转了设计。不是由中央规划者分配工作，而是工作节点从共享队列中领取工作。"协调"内置于事件总线的语义中。没有编排者；系统一直扩展到队列成为瓶颈为止。

## 概念 (Concept)

### 结构 (The shape)

```
                ┌──── shared queue ────┐
                │                      │
       ┌────────┼────────┐  ◄──────┬───┘
       ▼        ▼        ▼         │
     Worker  Worker  Worker   Worker
      A       B       C        D
       │        │        │         │
       └────────┴────────┴─────────┘
                 │
                 ▼
            results pool
```

没有编排者。每个工作节点重复：拉取任务、处理、写入结果（并可选择性地将后续任务入队）。

### 群集何时适用 (When swarm fits)

- **大量独立任务。** 爬取、转换、分类。任务之间互不依赖。
- **可变时长的工作。** 如果某些任务耗时 100ms 而其他任务耗时 10s，群集会自动平衡负载——快速的工作节点领取下一个任务。监督者必须预先估算时长。
- **吞吐量优先于确定性。** 你关心总完成时间，而不是严格排序。

### 群集何时失败 (When swarm fails)

- **有序工作流。** 如果步骤 3 需要步骤 2 的输出，群集可能让步骤 3 在步骤 2 完成之前就触发。
- **全局计划任务。** 复杂的研究问题受益于规划者。一群研究者产生的是独立的事实，而不是一份连贯的报告。
- **调试。** 没有中央日志且工作是异步的，重现一个 bug 代价高昂。

### Matrix (arXiv:2511.21686)

Matrix 是 2025 年的论文，它将群集推向了自然结论：控制流和数据流都是分布式队列上的序列化消息。没有中央协调器。容错来自消息持久性。可扩展性是消息代理的问题，而不是系统的问题。

贡献：一种编程模型，其中多代理协调变成了"这个代理订阅哪个消息主题？"而不是"监督者下一步选哪个代理？"这使得系统看起来像一个发布/订阅事件网格(pub/sub event mesh)。

### LangGraph 的群集架构 (LangGraph's Swarm Architecture)

LangGraph 2025 文档明确将"群集架构"(Swarm Architecture)描述为多代理模式之一：代理是节点，但边形成有环的有向图，任何节点都可以从池中被激活。工作节点根据条件从可用工作中选择，而不是由监督者分配。

### 失败模式：饥饿和热点 (Failure mode: starvation and hot-spotting)

如果所有工作节点都拉取最快可用的任务，长时间运行的任务永远不会被选中，直到它们是唯一剩下的任务。经典的队列饥饿(Queue starvation)。

缓解措施：
- 带显式老化(aging)的优先级队列（等待时间越长优先级越高）。
- 工作节点专业化：某些工作节点只处理"长"任务。
- 背压(Back-pressure)：限制进入队列的快速任务数量。

### 基于内容的路由关联 (The content-based routing link)

群集与基于内容的路由（第 22 课）自然配对。不是使用通用队列，而是每种消息类型一个队列。专业工作节点只订阅其类型。这是可扩展到数千个代理的消息总线架构的基础。

## 构建 (Build It)

`code/main.py` 实现了一个由 4 个工作线程组成的群集，从共享的 `queue.Queue` 中拉取任务。任务具有可变时长（有些快，有些慢）。该演示对比了：

- **顺序基线(Sequential baseline)：** 一个工作节点串行处理所有任务。
- **固定分配(Fixed assignment)：** 每个任务预先分配给特定工作节点（监督者风格）。
- **群集(Swarm)：** 工作节点从共享队列中拉取任务。

群集自动平衡负载；固定分配在分配的任务较慢时会让快速工作节点闲置。

运行：

```
python3 code/main.py
```

输出显示每个工作节点的任务计数（群集分配不均匀但最优）以及挂钟时间。

## 使用 (Use It)

`outputs/skill-swarm-fit.md` 评估一个任务应该使用群集还是监督者。输入：任务独立性、时长方差、排序要求、可调试性需求。

## 交付 (Ship It)

检查清单：

- **带老化的优先级队列(Priority queue with aging)。** 防止长任务饥饿。
- **工作节点幂等性(Worker idempotency)。** 如果工作节点在运行中崩溃，一个任务可能被拉取多次。工作节点必须是幂等的。
- **持久队列(Durable queue)。** 生产环境使用 Kafka、Redis Streams 或数据库支持的队列。`queue.Queue` 仅在内存中。
- **每个任务的可观测性(Observability per task)。** 每个任务有一个追踪 ID；每个工作节点记录开始/结束时间。
- **背压(Back-pressure)。** 如果队列增长速度超过工作节点消耗速度，减慢生产者。

## 练习 (Exercises)

1. 运行 `code/main.py`。在可变时长工作负载上，群集比顺序快多少？比固定分配快多少？
2. 添加一个优先级队列变体（使用 `queue.PriorityQueue`）。按任务的"重要性"字段分配优先级。观察低优先级任务在持续负载下是否会饿死。
3. 实现一个热点检测器：当某个工作节点处理的任务数是最慢工作节点的 3 倍时记录日志。这表明任务时长分布有什么问题？
4. 阅读 Matrix 论文 (arXiv:2511.21686) 的摘要和第 3 节。找出 Matrix 接受的一个具体权衡（可扩展性收益）和它放弃的一个权衡（可追溯性、确定性）。
5. 将群集演示改为使用 `(task_type, payload)` 元组的 `queue.Queue`，工作节点只订阅特定类型。当任务异构时，什么样的路由规则有意义？

## 关键术语 (Key Terms)

| 术语 | 人们说的 | 实际含义 |
|------|----------|----------|
| Swarm architecture (群集架构) | "去中心化代理" | 工作节点从共享队列拉取任务；没有中央编排者。 |
| Event bus (事件总线) | "代理订阅主题" | 按类型或内容将任务路由到工作节点的消息代理。 |
| Starvation (饥饿) | "任务从不运行" | 低优先级任务因持续到达的高优先级工作而从未被选中。 |
| Hot-spotting (热点) | "一个工作节点不堪重负" | 负载不均衡，一个工作节点获得大部分任务。 |
| Back-pressure (背压) | "减慢生产者" | 当队列满时向上游发出停止生产的信号机制。 |
| Idempotent worker (幂等工作节点) | "安全地重新运行" | 一个任务处理两次产生相同结果。因工作节点可能在运行中崩溃而必需。 |
| Durable queue (持久队列) | "崩溃后幸存" | 由磁盘或复制存储支持的队列；工作节点崩溃时任务不会丢失。 |
| Matrix framework (Matrix 框架) | "完全消息传递群集" | 数据和控制流都是分布式队列上的序列化消息。 |

## 延伸阅读 (Further Reading)

- [LangGraph workflows and agents — Swarm Architecture](https://docs.langchain.com/oss/python/langgraph/workflows-agents) — 显式群集支持
- [Matrix — A Decentralized Framework for Multi-Agent Systems](https://arxiv.org/abs/2511.21686) — 完全消息传递群集
- [Anthropic engineering — why supervisor not swarm in Research](https://www.anthropic.com/engineering/multi-agent-research-system) — 为什么一个特定生产系统明确选择监督者而非群集
- [AutoGen v0.4 actor-model docs](https://microsoft.github.io/autogen/stable/) — 事件驱动 actor 重写，比 v0.2 的 GroupChat 更接近群集