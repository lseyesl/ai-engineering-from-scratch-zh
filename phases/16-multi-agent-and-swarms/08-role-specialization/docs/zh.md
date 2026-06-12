# 角色专业化——规划者、批评者、执行者、验证者 (Role Specialization — Planner, Critic, Executor, Verifier)

> 2026 年最常见的多智能体分解：一个智能体规划，一个执行，一个批评或验证。MetaGPT（arXiv:2308.00352）将其形式化为编码到角色提示中的标准操作程序——产品经理、架构师、项目经理、工程师、QA 工程师——遵循 `Code = SOP(Team)`。ChatDev（arXiv:2307.07924）通过"聊天链"和"沟通去幻觉"（智能体明确请求缺失的细节）将设计师、程序员、评审者、测试者串联起来。验证者是承重角色：Cemri 等人（MAST，arXiv:2503.13657）表明每个多智能体失败都可以追溯到缺失或损坏的验证。普华永道报告称，在 CrewAI 中通过结构化验证循环获得了 7 倍的准确率提升（10% → 70%）。

**类型：** 学习 + 构建 (Learn + Build)
**语言：** Python（标准库）
**前置知识：** 阶段 16 · 04（原语模型），阶段 16 · 05（监督者）
**时间：** ~60 分钟

## 问题 (Problem)

通用的多智能体系统产生通用的输出。三个程序员在群聊中写出三种风味相同的平庸代码。你可以添加更多智能体，添加更多轮次，但仍然无法跨越质量门槛。

解决方案不是更多智能体——而是*不同*的智能体。分配不同的角色。给批评者规划者没有的工具。给验证者一个客观的测试套件。现在系统有了带有基于事实纠正的内部分歧，而不仅仅是并行猜测。

## 概念 (Concept)

### 四个经典角色 (The four canonical roles)

**规划者 (Planner)。** 阅读目标，生成步骤列表或规格。工具：知识检索、文档。输出：结构化计划。

**执行者 (Executor)。** 一次读取一个计划步骤，生成产物。工具：实际工作工具（代码编译器、shell、API 客户端）。输出：产物。

**批评者 (Critic)。** 对照规划者的意图阅读执行者的输出。工具：对产物的只读访问、静态分析。输出：接受/拒绝并附理由。

**验证者 (Verifier)。** 读取产物并运行确定性检查。工具：测试运行器、类型检查器、模式验证器。输出：通过/失败并附证据。

批评者是主观的、有观点的，通常是基于 LLM 的。验证者是客观的、确定性的，通常是基于代码的。它们不是同一个角色。

### MetaGPT 的 SOP 模式 (MetaGPT's SOP pattern)

MetaGPT（arXiv:2308.00352）将软件工程 SOP 编码为角色提示：

- **产品经理 (Product Manager)** 编写 PRD。
- **架构师 (Architect)** 生成系统设计。
- **项目经理 (Project Manager)** 拆分任务。
- **工程师 (Engineer)** 实现。
- **QA 工程师 (QA Engineer)** 运行测试。

每个角色都有严格的输入/输出模式。角色提示说明该角色*是什么*以及它*必须产生什么*。`Code = SOP(Team)` 的表述——确定性的 SOP 将 LLM 团队转变为可预测的流水线。

### ChatDev 的沟通去幻觉 (ChatDev's communicative dehallucination)

ChatDev 增加了一个关键动作：当执行者需要计划中没有的特定细节时，它在继续之前明确询问设计师。这防止了经典的 LLM 失败——看似合理地编造细节。

实现：角色提示包含"当你需要未被提供的特定信息时，在产生输出之前按名称询问相关角色。"

### 为什么验证者最重要 (Why verifier matters most)

Cemri 等人（MAST）追踪了 1642 次多智能体执行失败。21.3% 是验证缺口——系统交付了一个无人检查的答案。剩下的 79% 通常可以追溯到"有一个检查但静默失败或从未运行。"验证者是承重角色。

普华永道报告（CrewAI 部署，2025）称，添加结构化验证循环将准确率从 10% 提升到 70%。一个角色带来 7 倍增益。

### 批评者 vs 验证者 (Critic vs verifier)

- 批评者是一个 LLM，审查产物的质量。主观的。可能被看似合理的文字所欺骗。
- 验证者是一个确定性程序，在产物上运行。客观的。给出通过/失败并附证据。

两者都用。批评者捕捉验证者无法表达的风格问题。验证者捕捉批评者看不到的错误，因为它们只在运行时才显现。

### 反模式 (The anti-pattern)

你系统中的每个角色都是 LLM，每个角色的输出都是"看起来不错。"经典的 MAST 失败模式。至少添加一个验证者，其通过/失败由代码决定，而不是由 LLM 决定。

### 框架映射 (Framework mappings)

- **CrewAI** —— `Agent(role, goal, backstory)` 是教科书式的专业化接口。
- **LangGraph** —— 节点可以有专门的提示；边强制执行流水线。
- **AutoGen** —— 在 GroupChat 中使用单字名称的角色特定 ConversableAgent。
- **OpenAI Agents SDK** —— 角色专业化 Agent 之间的交接工具。

## 构建 (Build It)

`code/main.py` 实现了一个 4 角色流水线，构建一个简单的 Python 函数：

- **规划者 (Planner)** 生成规格。
- **执行者 (Executor)** 生成代码字符串。
- **批评者 (Critic)**（模拟 LLM）标记明显的问题。
- **验证者 (Verifier)** 在沙箱（`exec`）中针对测试用例运行生成的代码。

演示运行两次：一次执行者生成正确的代码（批评者 + 验证者都通过），一次执行者生成不符合规格的代码（批评者因为看起来合理而错过了错误，验证者因为测试失败而捕获了它）。

运行：

```
python3 code/main.py
```

## 使用 (Use It)

`outputs/skill-role-designer.md` 接收一个任务并生成角色名单（3-5 个角色）、每个角色的输入/输出模式以及验证者检查。在将智能体接入框架之前使用此工具。

## 交付 (Ship It)

检查清单：

- **至少一个确定性验证者 (At least one deterministic verifier)。** 永远不要全 LLM。
- **每个角色有明确的 I/O 模式 (Explicit I/O schema per role)。** 规划者返回规格，而不是散文；执行者读取该模式。
- **沟通去幻觉 (Communicative dehallucination)。** 执行者必须在信息缺失时询问规划者；永远不要编造。
- **批评者/验证者顺序 (Critic/verifier ordering)。** 先运行批评者（便宜，捕捉设计问题），再运行验证者（慢，捕捉错误）。
- **循环预算 (Loop budget)。** 最多 2 轮批评者-执行者修订，之后升级到人工。

## 练习 (Exercises)

1. 运行 `code/main.py` 并观察验证者如何捕获批评者遗漏的错误。添加一个静态分析检查（统计 `return` 出现次数）作为额外的验证者。它捕获了运行时测试遗漏的什么？
2. 添加第 5 个角色："需求分析师"，将用户愿望转化为规划者就绪的规格。哪些沟通去幻觉请求应该向上流向它？
3. 阅读 MetaGPT 第 3 节（"智能体"）。列出 MetaGPT 5 个角色中每个角色的输入/输出模式。
4. 阅读 ChatDev 的聊天链图（arXiv:2307.07924 图 3）。指出沟通去幻觉在何处打破了一个否则会无限循环的循环。
5. 普华永道的 7 倍准确率提升来自验证循环。假设三个任务，添加验证者不会有所帮助——即确定性检查正确性不可能或成本过高的情况。

## 关键术语 (Key Terms)

| 术语 | 人们说的 | 实际含义 |
|------|----------|----------|
| 角色专业化 (Role specialization) | "不同的智能体，不同的工作" | 为规划者/执行者/批评者/验证者角色调整的不同系统提示。 |
| SOP 模式 (SOP pattern) | "编码的标准操作程序" | MetaGPT 的框架：每个角色严格的 I/O 模式将团队转变为流水线。 |
| 沟通去幻觉 (Communicative dehallucination) | "编造之前先问" | ChatDev 模式：当细节缺失时，执行者询问规划者而不是编造一个。 |
| 批评者 (Critic) | "LLM 评审者" | 主观的、有观点的评审者。捕捉风格问题。可能被看似合理的文字欺骗。 |
| 验证者 (Verifier) | "确定性检查" | 基于代码的通过/失败。测试运行器、类型检查器、模式验证器。无法被欺骗。 |
| 验证缺口 (Verification gap) | "没人检查" | MAST 失败的 21.3%。答案交付时没有经过本可以捕获错误的检查。 |
| 修订循环 (Revision loop) | "批评者打回重做" | 批评者拒绝触发执行者重新运行并附带反馈。需要预算。 |
| 全 LLM 反模式 (All-LLM anti-pattern) | "看起来不错" | 每个角色都是 LLM，没有确定性检查。经典的 MAST 失败。 |

## 延伸阅读 (Further Reading)

- [Hong et al. — MetaGPT: Meta Programming for Multi-Agent Collaboration](https://arxiv.org/abs/2308.00352) —— SOP 作为角色提示的参考论文
- [Qian et al. — Communicative Agents for Software Development (ChatDev)](https://arxiv.org/abs/2307.07924) —— 聊天链 + 沟通去幻觉
- [Cemri et al. — Why Do Multi-Agent LLM Systems Fail?](https://arxiv.org/abs/2503.13657) —— MAST 分类法；验证缺口占失败的 21.3%
- [CrewAI docs — Agent roles](https://docs.crewai.com/en/introduction) —— 生产环境角色规格接口