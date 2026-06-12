# 分层架构及其失败模式 (Hierarchical Architecture and Its Failure Mode)

> 分层就是监督者嵌套。管理者智能体管理子管理者，子管理者管理工作者。CrewAI 的 `Process.hierarchical` 是教科书式的实现：一个 `manager_llm` 动态委派任务并验证输出。LangGraph 的等价物是 `create_supervisor(create_supervisor(...))`。当任务本身就是真实的组织架构图时，这是自然的模式。但它也是最容易崩溃为管理循环的模式——管理者智能体分配工作不当、误解子输出或无法达成共识。顺序模式往往比它更好。

**类型：** 学习 + 构建 (Learn + Build)
**语言：** Python（标准库）
**前置知识：** 阶段 16 · 05（监督者模式）
**时间：** ~60 分钟

## 问题 (Problem)

一旦理解了监督者模式，自然的下一步就是"如果工作者本身就是监督者呢？"团队有子团队；公司有部门的部门。分层架构反映了这一点。

问题在于：LLM 管理者与人类管理者不同。人类管理者对其下属所知内容有稳定的先验知识。LLM 管理者每轮都根据其上下文中的内容重新推理整个组织。该上下文中微小的漂移就会导致整个树错误分配工作。

## 概念 (Concept)

### 结构 (The shape)

```
                  Manager
                  ┌─────┐
                  └──┬──┘
            ┌────────┴────────┐
            ▼                 ▼
        Sub-Mgr A         Sub-Mgr B
        ┌─────┐           ┌─────┐
        └──┬──┘           └──┬──┘
          ┌┴──┬──┐          ┌┴──┐
          ▼   ▼  ▼          ▼   ▼
        W1  W2  W3         W4  W5
```

每个内部节点负责规划、委派和综合。只有叶子节点执行实际工作。

### 优势所在 (Where it shines)

- **清晰的组织映射 (Clear org mapping)。** 如果实际任务是部门级的（"法务审阅文档，财务审阅文档，工程审阅文档，然后为高管总结"），层级结构是明确的。
- **本地总结 (Local summarization)。** 每个子管理者在顶层管理者看到之前综合其团队的输出。顶层管理者看到三个子管理者的摘要，而不是十五个工作者的输出。

### 问题所在 (Where it breaks)

2026 年的事后分析不断发现的三种失败模式：

1. **任务分配错误 (Task assignment error)。** 管理者阅读目标，幻觉出一个分解方案，并委派给错误的子管理者。因为子管理者忠实地执行分配的任务，错误只在顶层综合时才暴露——距离人类本可以捕捉到的地方隔了一层。
2. **输出误解 (Output misinterpretation)。** 子管理者返回"无法验证主张 X"。顶层管理者总结为"主张 X 未确认"。含义在每一层漂移。
3. **共识循环 (Consensus loops)。** 两个子管理者意见不一；顶层管理者要求他们协调；他们重新向下委派；工作者重新运行；子管理者返回略有不同的答案；循环。CrewAI 的 `Process.hierarchical` 通过步骤限制来防止这种情况，但限制本身现在成了一个超参数。

### 决定性问题 (The deciding question)

顺序（线性流水线）vs 分层：你的任务真的有独立的子团队，还是它只是一个假装成树结构的线性流程？如果是后者，使用顺序模式。如果是前者，使用分层模式但要预算明确的协调规则。

### CrewAI 的实现 (CrewAI's implementation)

`Process.hierarchical` 将一个管理者 LLM 置于专业团队之上。管理者：

- 接收顶层任务，
- 将子任务分配给团队，
- 评估团队输出，
- 决定是接受、重新委派还是迭代。

文档：https://docs.crewai.com/en/introduction（在"核心概念"下查找"分层流程"）。

### LangGraph 的实现 (LangGraph's implementation)

LangGraph 使用嵌套的 `create_supervisor` 调用。内部监督者有自己的图；外部监督者将内部图视为一个不透明节点。这在调试方面比 CrewAI 更清晰（你可以分别逐步调试每个图），但更难表达树的动态重塑。

参考：https://reference.langchain.com/python/langgraph-supervisor。

## 构建 (Build It)

`code/main.py` 运行一个 3 层层级结构：

- 顶层管理者：将任务拆分为"工程"和"法务"分支，
- 工程子管理者：拆分为"前端"和"后端"工作者，
- 法务子管理者：一个工作者。

演示对比了顺利路径（所有人意见一致）与一个**扰动路径**——顶层管理者的分解将"法务"错误标记为"财务"，并观察错误级联——子管理者忠实地执行财务工作，顶层综合器报告财务发现，原始法务问题未被回答。

运行：

```
python3 code/main.py
```

输出显示两条路径，并清晰并排展示"被问的问题"与"实际交付的内容"。

## 使用 (Use It)

`outputs/skill-hierarchy-fitness.md` 评估给定任务应该使用分层、顺序还是扁平监督者。输入：任务描述、组织架构、协调预算。输出：模式推荐及需要防范的具体失败模式。

## 交付 (Ship It)

如果你交付分层架构：

- **树深度限制为 2 (Cap tree depth at 2)。** 三层已经将大多数错误隐藏在可观测性之外。
- **明确的协调预算 (Explicit reconciliation budget)。** 设置顶层管理者必须提交前的最大轮数。通常为 2。
- **每个综合结果的可追溯性 (Provenance on every synthesis)。** 每个节点的摘要必须引用产生它的叶子输出。
- **分解漂移告警 (Alert on decomposition drift)。** 记录管理者每步的分解；与用户查询进行对比。如果分解不再覆盖查询，触发告警。

## 练习 (Exercises)

1. 运行 `code/main.py` 并比较顺利路径与扰动路径。需要多少层管理者交接，顶层输出才会完全偏离用户的问题？
2. 添加第三层（顶层 → 子层 → 子子层 → 工作者）。测量随着深度增加，扰动路径自我纠正与完全偏离的频率。
3. 在每个子管理者处实现一个"金丝雀"工作者，始终被问相同的原始用户问题（不变）。使用金丝雀的答案检测分解漂移。当金丝雀与综合答案不一致时，管理者应如何反应？
4. 阅读 CrewAI 的 `Process.hierarchical` 文档。找出 CrewAI 应用的一个具体防护措施（步骤限制、manager_llm 约束），并描述它针对哪种失败模式。
5. 比较嵌套的 LangGraph 监督者与 CrewAI 分层架构。哪个更容易检测协调循环？

## 关键术语 (Key Terms)

| 术语 | 人们说的 | 实际含义 |
|------|----------|----------|
| 分层 (Hierarchical) | "组织架构图模式" | 监督者之上还有监督者；只有叶子节点执行工作。 |
| 管理者 LLM (Manager LLM) | "老板" | 在内部节点负责分解、分配和验证的 LLM。 |
| 分解漂移 (Decomposition drift) | "老板跑偏了" | 顶层管理者的拆分不再覆盖原始问题。 |
| 协调循环 (Reconciliation loop) | "无尽的会议" | 子管理者意见不一；顶层重新委派；工作者重新运行；循环直到预算耗尽。 |
| 深度-2 上限 (Depth-2 ceiling) | "不要超过 2 层" | 经验防护措施：3 层以上会破坏可观测性。 |
| 金丝雀问题 (Canary question) | "每层的真实基准" | 一个始终被问相同原始查询（不变）的工作者，用于检测漂移。 |
| 可追溯链 (Provenance chain) | "谁说了什么" | 从每个综合结果追溯到产生它的叶子输出。 |

## 延伸阅读 (Further Reading)

- [CrewAI introduction — Process.hierarchical](https://docs.crewai.com/en/introduction) —— 教科书式的分层实现，使用管理者 LLM
- [LangGraph supervisor reference](https://reference.langchain.com/python/langgraph-supervisor) —— 通过 `create_supervisor` 实现嵌套监督者
- [Anthropic engineering — Research system](https://www.anthropic.com/engineering/multi-agent-research-system) —— Anthropic 为何刻意选择扁平监督者而非分层架构
- [Cemri et al. — Why Do Multi-Agent LLM Systems Fail?](https://arxiv.org/abs/2503.13657) —— MAST 分类法；协调失败部分记录了分解漂移