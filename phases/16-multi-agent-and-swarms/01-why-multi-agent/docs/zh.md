# 为什么需要多智能体？

> 一个智能体撞墙了。明智的做法不是造一个更大的智能体，而是用更多的智能体。

**类型：** 学习
**语言：** TypeScript
**前置知识：** 阶段 14（智能体工程）
**时间：** ~60 分钟

## 学习目标

- 识别单智能体的天花板（上下文溢出、专业混杂、顺序瓶颈），并解释何时拆分多个智能体是正确的选择
- 比较编排模式（流水线、并行扇出、监督者、分层），并为给定的任务结构选择合适的一种
- 设计一个具有清晰角色边界、共享状态和通信契约的多智能体系统
- 分析多智能体复杂性（延迟、成本、调试难度）与单智能体简单性之间的权衡

## 问题

你在阶段 14 中构建了一个单智能体。它能工作。它可以读取文件、运行命令、调用 API 并对结果进行推理。然后你把它指向一个真实的代码库：200 个文件、三种语言、依赖基础设施的测试，以及一个在写代码之前需要研究外部 API 的需求。

智能体卡住了。不是因为 LLM 笨，而是因为任务超出了单个智能体循环能处理的范围。上下文窗口被文件内容填满。智能体忘记了 40 次工具调用前读过的内容。它试图同时成为研究员、程序员和审查员，结果三件事都做得很差。

这就是单智能体天花板。每当任务需要以下条件时，你都会碰到它：

- **超出单个窗口容量的上下文** — 读取 50 个文件会超过 20 万 token
- **不同阶段需要不同的专业知识** — 研究需要与代码生成不同的提示词
- **可以并行完成的工作** — 为什么顺序读取三个文件，当你可以同时读取它们？

## 概念

### 单智能体天花板

一个单智能体就是一个循环、一个上下文窗口、一个系统提示词。想象一下：

```
┌─────────────────────────────────────────┐
│              单智能体                    │
│                                         │
│  ┌───────────────────────────────────┐  │
│  │          上下文窗口                │  │
│  │                                   │  │
│  │  研究笔记                         │  │
│  │  + 代码文件                       │  │
│  │  + 测试输出                       │  │
│  │  + 审查反馈                       │  │
│  │  + API 文档                       │  │
│  │  + ...                            │  │
│  │                                   │  │
│  │  ██████████████████████ 已满 ███  │  │
│  └───────────────────────────────────┘  │
│                                         │
│  一个系统提示词试图覆盖                   │
│  研究 + 编码 + 审查 + 测试               │
│                                         │
│  结果：每件事都平庸                      │
└─────────────────────────────────────────┘
```

三件事会出问题：

1. **上下文饱和** — 工具结果不断堆积。到第 30 轮时，智能体已经消耗了 15 万 token 的文件内容、命令输出和先前的推理。第 5 轮的关键细节丢失了。

2. **角色混淆** — 一个说"你是研究员、程序员、审查员和测试员"的系统提示词，会产生一个半研究、半编码、永远完不成审查的智能体。

3. **顺序瓶颈** — 智能体读取文件 A，然后文件 B，然后文件 C。三次串行 LLM 调用。三次串行工具执行。没有并行。

### 多智能体解决方案

拆分工作。给每个智能体一个任务、一个上下文窗口和一个为该任务调优的系统提示词：

```
┌──────────────────────────────────────────────────────────┐
│                      编排器                              │
│                                                          │
│  "构建一个用户管理的 REST API"                            │
│                                                          │
│         ┌──────────┬──────────┬──────────┐               │
│         │          │          │          │               │
│         ▼          ▼          ▼          ▼               │
│   ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │
│   │  研究员  │ │  程序员  │ │  审查员  │ │  测试员  │  │
│   │          │ │          │ │          │ │          │  │
│   │ 读取文档 │ │ 根据研究 │ │ 检查代码 │ │ 运行测试 │  │
│   │ 查找模式 │ │ 和规范   │ │ 质量     │ │ 报告结果 │  │
│   │          │ │ 编写代码 │ │ 发现缺陷 │ │          │  │
│   └─────┬────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘  │
│         │           │            │             │         │
│         └───────────┴────────────┴─────────────┘         │
│                          │                               │
│                     合并结果                              │
└──────────────────────────────────────────────────────────┘
```

每个智能体拥有：
- 一个专注的系统提示词（"你是一个代码审查员。你唯一的工作是发现缺陷。"）
- 自己的上下文窗口（不被其他智能体的工作污染）
- 清晰的输入/输出契约（接收研究笔记，输出代码）

### 实际系统案例

**Claude Code 子智能体** — 当 Claude Code 使用 `Task` 生成子智能体时，它会创建一个具有限定范围任务的子智能体。父智能体保持上下文清洁。子智能体执行专注的工作并返回摘要。

**Devin** — 运行一个规划智能体、一个编码智能体和一个浏览器智能体。规划智能体将工作分解为步骤。编码智能体编写代码。浏览器智能体研究文档。每个都有独立的上下文。

**多智能体编码团队（SWE-bench）** — SWE-bench 上表现最好的系统使用一个读取代码库的研究员、一个设计修复方案的规划员和一个实现它的程序员。单智能体系统得分较低。

**ChatGPT Deep Research** — 并行生成多个搜索智能体，每个探索不同的角度，然后综合结果。

### 频谱

多智能体不是二元的。它是一个频谱：

```
简单 ──────────────────────────────────────────── 复杂

 单智能体      子智能体        流水线         团队          群体

  ┌───┐       ┌───┐        ┌───┐───┐    ┌───┐───┐    ┌─┐┌─┐┌─┐
  │ A │       │ A │        │ A │ B │    │ A │ B │    │ ││ ││ │
  └───┘       └─┬─┘        └───┘─┬─┘    └─┬─┘─┬─┘    └┬┘└┬┘└┬┘
                │                │        │   │       ┌┴──┴──┴┐
              ┌─┴─┐          ┌───┘───┐    │   │       │共享   │
              │ a │          │ C │ D │  ┌─┴───┴─┐    │状态   │
              └───┘          └───┘───┘  │ 消息   │    └───────┘
                                        │ 总线   │
  1 个循环     父 +          按阶段      │       │    N 个对等体
  1 个上下文   子任务         分阶段      └───────┘    涌现行为
                                        显式角色
```

**单智能体** — 一个循环，一个提示词。适用于简单任务。

**子智能体** — 父智能体为专注的子任务生成子智能体。父智能体维护计划。子智能体报告结果。这就是 Claude Code 的做法。

**流水线** — 智能体按顺序运行。智能体 A 的输出成为智能体 B 的输入。适用于分阶段工作流：研究 -> 编码 -> 审查 -> 测试。

**团队** — 智能体通过共享消息总线并行运行。每个都有角色。编排器进行协调。适用于需要同时使用不同技能的场景。

**群体** — 许多相同或几乎相同的智能体，共享状态。没有固定的编排器。智能体从队列中获取工作。适用于高吞吐量的并行任务。

### 四种多智能体模式

#### 模式 1：流水线

```
输入 ──▶ 智能体 A ──▶ 智能体 B ──▶ 智能体 C ──▶ 输出
          （研究）      （编码）      （审查）
```

每个智能体转换数据并向前传递。易于推理。一个阶段的失败会阻塞其余部分。

#### 模式 2：扇出 / 扇入

```
                 ┌──▶ 智能体 A ──┐
                 │              │
输入 ──▶ 拆分 ──├──▶ 智能体 B ──├──▶ 合并 ──▶ 输出
                 │              │
                 └──▶ 智能体 C ──┘
```

将工作拆分到并行智能体，然后合并结果。适用于可分解为独立子任务的任务。

#### 模式 3：编排器-工作者

```
                     ┌──────────┐
                     │  编排器  │
                     └──┬───┬───┘
                   任务  │   │  任务
                  ┌─────┘   └─────┐
                  ▼               ▼
            ┌──────────┐   ┌──────────┐
            │ 工作者 A │   │ 工作者 B │
            └──────────┘   └──────────┘
```

一个智能编排器决定做什么，委派给工作者，并综合结果。编排器本身就是一个拥有生成工作者工具的智能体。

#### 模式 4：对等群体

```
         ┌───┐ ◄──── 消息 ────▶ ┌───┐
         │ A │                  │ B │
         └─┬─┘                  └─┬─┘
           │                      │
      消息  │    ┌───────────┐     │ 消息
           └───▶│  共享     │◄────┘
                │  状态     │
           ┌───▶│  / 队列   │◄────┐
           │    └───────────┘     │
      消息  │                      │ 消息
         ┌─┴─┐                  ┌─┴─┐
         │ C │ ◄──── 消息 ────▶ │ D │
         └───┘                  └───┘
```

没有中央编排器。智能体点对点通信。决策从交互中涌现。更难调试，但可以扩展到大量智能体。

### 何时不使用多智能体

多智能体增加了复杂性。智能体之间的每条消息都是一个潜在的故障点。调试从"阅读一个对话"变成了"追踪跨五个智能体的消息"。

**保持单智能体当：**
- 任务适合一个上下文窗口（工作数据低于约 10 万 token）
- 不同阶段不需要不同的系统提示词
- 顺序执行足够快
- 任务足够简单，拆分带来的开销大于价值

**复杂性成本：**
- 每个智能体边界都是一个有损压缩步骤：智能体 A 的完整上下文被总结为给智能体 B 的一条消息
- 协调逻辑（谁做什么、何时做、按什么顺序）本身就是缺陷的来源
- 延迟增加：N 个智能体意味着至少 N 次串行 LLM 调用，如果需要来回沟通则更多
- 成本倍增：每个智能体独立消耗 token

经验法则：如果一个任务需要少于 20 次工具调用且适合 10 万 token，保持单智能体。

```figure
swarm-messages
```

## 构建

### 步骤 1：过载的单智能体

这是一个试图做所有事情的单智能体。它有一个庞大的系统提示词和一个同时容纳研究、代码和审查的上下文窗口：

```typescript
type AgentResult = {
  content: string;
  tokensUsed: number;
  toolCalls: number;
};

async function singleAgentApproach(task: string): Promise<AgentResult> {
  const systemPrompt = `你是一名全栈开发者。你必须：
1. 研究需求
2. 编写代码
3. 审查代码中的缺陷
4. 编写测试
在单个对话中完成所有这些。`;

  const contextWindow: string[] = [];
  let totalTokens = 0;
  let totalToolCalls = 0;

  const research = await fakeLLMCall(systemPrompt, `研究：${task}`);
  contextWindow.push(research.output);
  totalTokens += research.tokens;
  totalToolCalls += research.calls;

  const code = await fakeLLMCall(
    systemPrompt,
    `基于此研究：\n${contextWindow.join("\n")}\n\n现在为以下任务编写代码：${task}`
  );
  contextWindow.push(code.output);
  totalTokens += code.tokens;
  totalToolCalls += code.calls;

  const review = await fakeLLMCall(
    systemPrompt,
    `基于所有先前的上下文：\n${contextWindow.join("\n")}\n\n审查代码。`
  );
  contextWindow.push(review.output);
  totalTokens += review.tokens;
  totalToolCalls += review.calls;

  return {
    content: contextWindow.join("\n---\n"),
    tokensUsed: totalTokens,
    toolCalls: totalToolCalls,
  };
}
```

这种方法的问题：
- 上下文窗口随着每个阶段增长。到审查步骤时，它包含了研究笔记、代码和先前的推理。
- 系统提示词是通用的。不能为每个阶段调优。
- 没有任何东西并行运行。

### 步骤 2：专业智能体

现在拆分它。每个智能体得到一个任务：

```typescript
type SpecialistAgent = {
  name: string;
  systemPrompt: string;
  run: (input: string) => Promise<AgentResult>;
};

function createSpecialist(name: string, systemPrompt: string): SpecialistAgent {
  return {
    name,
    systemPrompt,
    run: async (input: string) => {
      const result = await fakeLLMCall(systemPrompt, input);
      return {
        content: result.output,
        tokensUsed: result.tokens,
        toolCalls: result.calls,
      };
    },
  };
}

const researcher = createSpecialist(
  "researcher",
  "你是一名技术研究员。阅读文档，查找模式，总结发现。只输出实现所需的事实。"
);

const coder = createSpecialist(
  "coder",
  "你是一名高级 TypeScript 开发者。根据需求和研究笔记，编写干净、经过测试的代码。仅此而已。"
);

const reviewer = createSpecialist(
  "reviewer",
  "你是一名代码审查员。发现缺陷、安全问题和逻辑错误。要具体。引用行号。"
);
```

每个专业智能体都有一个专注的提示词。每个都获得一个干净的上下文窗口，只包含它需要的输入。

### 步骤 3：通过消息协调

使用显式消息传递将专业智能体连接起来：

```typescript
type AgentMessage = {
  from: string;
  to: string;
  content: string;
  timestamp: number;
};

async function multiAgentApproach(task: string): Promise<AgentResult> {
  const messages: AgentMessage[] = [];
  let totalTokens = 0;
  let totalToolCalls = 0;

  const researchResult = await researcher.run(task);
  messages.push({
    from: "researcher",
    to: "coder",
    content: researchResult.content,
    timestamp: Date.now(),
  });
  totalTokens += researchResult.tokensUsed;
  totalToolCalls += researchResult.toolCalls;

  const coderInput = messages
    .filter((m) => m.to === "coder")
    .map((m) => `[来自 ${m.from}]：${m.content}`)
    .join("\n");

  const codeResult = await coder.run(coderInput);
  messages.push({
    from: "coder",
    to: "reviewer",
    content: codeResult.content,
    timestamp: Date.now(),
  });
  totalTokens += codeResult.tokensUsed;
  totalToolCalls += codeResult.toolCalls;

  const reviewerInput = messages
    .filter((m) => m.to === "reviewer")
    .map((m) => `[来自 ${m.from}]：${m.content}`)
    .join("\n");

  const reviewResult = await reviewer.run(reviewerInput);
  messages.push({
    from: "reviewer",
    to: "orchestrator",
    content: reviewResult.content,
    timestamp: Date.now(),
  });
  totalTokens += reviewResult.tokensUsed;
  totalToolCalls += reviewResult.toolCalls;

  return {
    content: messages.map((m) => `[${m.from} -> ${m.to}]：${m.content}`).join("\n\n"),
    tokensUsed: totalTokens,
    toolCalls: totalToolCalls,
  };
}
```

每个智能体只接收发给它的消息。没有上下文污染。研究员 5 万 token 的文档阅读永远不会进入审查员的上下文。

### 步骤 4：比较

```typescript
async function compare() {
  const task = "为 Express.js API 构建一个限速中间件";

  console.log("=== 单智能体 ===");
  const single = await singleAgentApproach(task);
  console.log(`Token：${single.tokensUsed}`);
  console.log(`工具调用：${single.toolCalls}`);

  console.log("\n=== 多智能体 ===");
  const multi = await multiAgentApproach(task);
  console.log(`Token：${multi.tokensUsed}`);
  console.log(`工具调用：${multi.toolCalls}`);
}
```

多智能体版本使用更多的总 token（三个智能体，三次独立的 LLM 调用），但每个智能体的上下文保持干净。每个阶段的质量提高了，因为系统提示词是专门化的。

## 使用

本课程产生一个可复用的提示词，用于决定何时使用多智能体。见 `outputs/prompt-multi-agent-decision.md`。

## 练习

1. 添加第四个专业智能体：一个"测试员"智能体，从程序员接收代码，从审查员接收审查反馈，然后编写测试
2. 修改流水线，使审查员可以将反馈发回给程序员进行修订循环（最多 2 轮）
3. 将顺序流水线转换为扇出：并行运行研究员和"需求分析员"智能体，然后在传递给程序员之前合并它们的输出

## 关键术语

| 术语 | 人们说的 | 实际含义 |
|------|---------|---------|
| Swarm（群体） | "AI 智能体的蜂群思维" | 一组对等智能体，共享状态，没有固定领导者。行为从局部交互中涌现。 |
| Orchestrator（编排器） | "老板智能体" | 一个智能体，其工具包括生成和管理其他智能体。它规划和委派，但可能不执行实际工作。 |
| Coordinator（协调器） | "交通警察" | 一个非智能体组件（通常只是代码，不是 LLM），根据规则在智能体之间路由消息。 |
| Consensus（共识） | "智能体达成一致" | 一种协议，多个智能体必须达成一致才能继续。用于需要解决冲突输出的场景。 |
| Emergent behavior（涌现行为） | "智能体自己搞定了" | 从智能体交互中产生的系统级模式，并非显式编程。可能有用也可能有害。 |
| Fan-out / fan-in（扇出/扇入） | "智能体的 Map-Reduce" | 将任务拆分到并行智能体（扇出），然后合并它们的结果（扇入）。 |
| Message passing（消息传递） | "智能体互相交谈" | 智能体之间的通信机制：从一个智能体发送到另一个智能体的结构化数据，取代共享上下文窗口。 |

## 延伸阅读

- [The Landscape of Emerging AI Agent Architectures](https://arxiv.org/abs/2409.02977) — 多智能体模式综述
- [AutoGen: Enabling Next-Gen LLM Applications](https://arxiv.org/abs/2308.08155) — 微软的多智能体对话框架
- [Claude Code subagents documentation](https://docs.anthropic.com/en/docs/claude-code) — Claude Code 如何使用 Task 委派
- [CrewAI documentation](https://docs.crewai.com/) — 基于角色的多智能体框架