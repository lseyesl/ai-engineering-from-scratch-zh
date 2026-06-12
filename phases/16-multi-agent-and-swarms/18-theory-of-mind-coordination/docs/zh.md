# 心智理论与涌现协调 (Theory of Mind and Emergent Coordination)

> Li 等人（arXiv:2310.10701）表明，LLM 智能体在合作文本游戏中展现出**涌现的高阶心智理论**（Theory of Mind, ToM）——推理另一个智能体对第三个智能体信念的信念——但由于上下文管理和幻觉，在长周期规划上失败。Riedl（arXiv:2510.05174）测量了群体中的高阶协同，发现**只有**在 ToM 提示条件下才会产生身份关联分化和目标导向互补性；低容量 LLM 只表现出虚假涌现。也就是说，协调涌现是提示条件依赖和模型依赖的，并非免费。本课实现一个最小 ToM 感知智能体，在有和无 ToM 提示的条件下运行合作任务，并对照 Riedl 2025 协议测量协调增量。

**类型：** Learn + Build
**语言：** Python (stdlib)
**前置知识：** Phase 16 · 07（心智社会与辩论），Phase 16 · 17（生成式智能体）
**时间：** ~75 分钟

## 问题 (Problem)

多智能体协调常常看起来像魔法：智能体分工协作、相互预判、避免冗余。通常这种"涌现"是提示工程的产物——有人告诉智能体要"协调"。去掉提示，协调就消失了。

Riedl 2025 年的发现更为严格：在受控条件下，协调只有在智能体被提示推理**其他智能体的心智**（ToM）时才会涌现。没有 ToM 提示，即使是强大的模型也表现出无法通过统计控制的协调模式。这对生产很重要：团队发布"多智能体协调"功能，但这些功能是提示依赖且脆弱的。

本课将 ToM 视为一种特定能力（推理关于信念的信念），构建一个最小 ToM 感知智能体，并测量真正的协调与提示装饰之间的区别。

## 概念 (Concept)

### ToM 的含义 (What ToM means)

发展心理学：3 岁儿童认为任何人的内心世界都与自己相同。5 岁儿童理解他人有不同的信念。7 岁儿童推理关于信念的信念（"她认为我认为球在杯子下面"）。这些分别是零阶、一阶和二阶 ToM。

对于 LLM 智能体，ToM 阶次映射为：

- **零阶 (Zeroth-order)：** 没有他人的模型。智能体仅基于自身观察行动。
- **一阶 (First-order)：** 智能体拥有每个其他智能体信念的模型。"Alice 相信 X。"
- **二阶 (Second-order)：** 智能体建模递归信念。"Alice 相信 Bob 相信 X。"

Li 等人 2023 年发现，一阶和二阶 ToM 在合作游戏中的 LLM 智能体中涌现，但在长周期和不可靠通信下会退化。

### Sally-Anne 测试简述

一个 1985 年的错误信念测试：Sally 把弹珠放在篮子 A 中，然后离开。Anne 把它移到篮子 B。Sally 回来时会去哪里找？有一阶 ToM 的孩子说篮子 A（Sally 的信念与现实不同）。没有的孩子说篮子 B。

GPT-4 时代的 LLM 在直接提问时能通过 Sally-Anne 式测试。但当叙述很长、场景多次变化或问题间接表达时，它们就会失败。这就是 2026 年生产 LLM 中 ToM 的实际状态。

### Riedl 的协调测量

Riedl（arXiv:2510.05174）构建了一个群体规模测试：N 个智能体，一个合作目标，可变的提示条件。测量：

1. **身份关联分化 (Identity-linked differentiation)。** 智能体是否随时间发展出稳定的角色区分？
2. **目标导向互补性 (Goal-directed complementarity)。** 智能体的行动是否互补（不同子任务）而非重复？
3. **高阶协同 (Higher-order synergy)。** 一个统计度量，衡量群体是否实现了任何子集都无法实现的结果。

结果：只有在 ToM 提示条件下，所有三个指标才产生高于基线的信号。没有 ToM 提示时，中等容量模型的指标接近随机水平。大型模型在没有显式 ToM 提示的情况下表现出一些协调，但效果小于显式提示。

### 协调幻觉 (The coordination illusion)

没有统计控制时，演示中的"涌现协调"通常反映：

- 内嵌协调的提示工程（系统提示说"一起工作"）。
- 观察者偏差（我们看到预期的模式）。
- 事后选择成功的运行。

那些宣传"涌现协调"但没有可测量信号的生产系统应被视为营销。在声称之前先测量。

### 最小 ToM 感知智能体 (A minimal ToM-aware agent)

结构：

```
agent state:
  own_beliefs:    {facts the agent believes}
  other_models:   {other_agent_id -> {beliefs_the_agent_attributes_to_them}}
  actions_last_N: [history of others' actions]

observation update:
  - update own_beliefs from direct observation
  - update other_models[agent_id] from their action + prior beliefs

action selection:
  - enumerate candidate actions
  - for each, predict what each other agent will do next given their modeled beliefs
  - pick action that maximizes joint outcome under those predictions
```

`other_models` 属性就是 ToM 状态。一阶 ToM 只保留一层。二阶 ToM 添加 `other_models[i][other_models_of_j]`——我认为智能体 i 认为智能体 j 相信什么。

### 为什么长周期有害 (Why long-horizon hurts)

Li 等人记录：上下文限制导致智能体忘记哪个信念属于谁。幻觉向其他智能体模型添加虚假信念。两者都产生"我以为他认为 X"的错误，并随时间累积。

论文及 2024-2026 后续工作中记录的缓解措施：

- **提示中的显式 ToM 状态。** 结构化格式：`{agent_id: belief_list}`。强制检索保持身份-信念绑定。
- **更短的推理链。** 每轮更少的 ToM 更新减少累积幻觉。
- **外部 ToM 存储。** 将模型维护在 LLM 上下文之外；每轮只注入相关部分。

### ToM 在生产中失败的地方 (Where ToM fails in production)

- **对抗性场景。** 具有良好 ToM 的智能体更容易被操纵（你可以建模它们对你的建模，然后利用）。
- **异构团队。** 当模型不同时，对一个对手有效的 ToM 模型无法泛化。
- **依赖真实情况的任务。** ToM 是关于信念的；如果正确性取决于事实，ToM 可能成为干扰。

### 你能实际测量的协调 (The coordination you can actually measure)

三个实际信号表明团队的协调是真实的而非提示装饰：

1. **随时间互补 (Complementarity over time)。** 在多轮任务中，智能体的行动是否覆盖不相交的子任务？
2. **预判 (Anticipation)。** 智能体 A 在 T+1 轮的行动是否依赖于对 B 在 T+2 轮行动的预测，且该预测被证明正确？
3. **修正 (Correction)。** 当 A 在 T 轮误读 B 的信念时，A 是否在 T+2 轮之前修正？

这些可以在记录的多智能体系统中测量。它们是"协调"叙事的实质性版本。

## 构建 (Build It)

`code/main.py` 实现：

- `ToMAgent` — 跟踪自身信念和每个其他智能体的信念模型。
- 一个合作任务：三个智能体必须从三个盒子中收集三个令牌；每个盒子只能放一个令牌。智能体不能通信；它们从彼此的行动推断意图。
- 两种配置：`zeroth_order`（无 ToM）和 `first_order`（具有一阶信念模型的 ToM）。
- 在 200 次随机试验中测量：完成率、重复率（两个智能体 targeting 同一个盒子）、平均完成轮数。

运行：

```
python3 code/main.py
```

预期输出：零阶智能体重复率约 35%，10 轮内完成约 60% 的试验。一阶 ToM 智能体重复率约 5%，完成约 95%。这个增量就是可测量的协调效果。

## 使用 (Use It)

`outputs/skill-tom-auditor.md` 是一个技能，用于审计多智能体系统的"涌现协调"声明。检查提示装饰、相对于对照组的统计显著性以及测量的互补性。

## 交付 (Ship It)

协调声明检查清单：

- **对照组 (Control condition)。** 你的系统在没有协调提示时的版本。两者都测量。
- **统计检验 (Statistical test)。** 系统和对照组之间的差异在你的指标上是否在 `p < 0.05` 水平显著？
- **互补性度量 (Complementarity measure)。** 随时间的行为不相交性，而不仅仅是最终成功。
- **失败案例日志 (Failure-case log)。** 当智能体协调失败时，ToM 状态看起来如何？
- **模型容量披露 (Model-capacity disclosure)。** 如果效果在较小模型上消失，请说明。

## 练习 (Exercises)

1. 运行 `code/main.py`。确认一阶 ToM 将重复率降低约 7 倍。当扩展到 5 个智能体和 5 个盒子时，差距是否持续？
2. 实现二阶 ToM（智能体 A 建模 B 对 C 的看法）。它是否优于一阶 ToM？在什么任务上？
3. 向 ToM 状态注入**幻觉**：每轮随机翻转一个信念。这对一阶性能有多大影响？
4. 阅读 Li 等人（arXiv:2310.10701）。复现"长周期退化"发现：当轮数从 10 增加到 30 时，你的一阶 ToM 性能如何变化？
5. 阅读 Riedl 2025（arXiv:2510.05174）。在你的模拟日志上实现高阶协同统计量。在没有 ToM 提示条件的情况下，效果是否存在？

## 关键术语 (Key Terms)

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Theory of Mind (心智理论) | "理解他人的心智" | 建模另一个智能体信念的能力。按阶次分级（0, 1, 2+）。 |
| Sally-Anne test (Sally-Anne 测试) | "错误信念测试" | 1985 年发展心理学；LLM 通过简单版本，在复杂版本上失败。 |
| First-order ToM (一阶 ToM) | "A 相信 X" | 建模一个他人关于事实的信念。 |
| Second-order ToM (二阶 ToM) | "A 相信 B 相信 X" | 递归建模深一层。 |
| Identity-linked differentiation (身份关联分化) | "随时间稳定的角色" | Riedl 的指标：角色持续存在，而非随机。 |
| Goal-directed complementarity (目标导向互补性) | "不相交的行动" | 智能体 targeting 不同子任务，而非相同任务。 |
| Higher-order synergy (高阶协同) | "群体超越任何子集" | Riedl 用于真实协调的统计度量。 |
| Coordination illusion (协调幻觉) | "看起来协调" | 没有可测量信号的提示装饰式协调外观。 |

## 延伸阅读 (Further Reading)

- [Li et al. — Theory of Mind for Multi-Agent Collaboration via Large Language Models](https://arxiv.org/abs/2310.10701) — 合作游戏中的涌现 ToM；长周期失败模式
- [Riedl — Emergent Coordination in Multi-Agent Language Models](https://arxiv.org/abs/2510.05174) — 群体规模测量；ToM 提示是承重条件
- [Premack & Woodruff — Does the chimpanzee have a theory of mind?](https://www.cambridge.org/core/journals/behavioral-and-brain-sciences/article/does-the-chimpanzee-have-a-theory-of-mind/1E96B02CD9850E69AF20F81FA7EB3595) — 1978 年 ToM 概念的起源
- [Baron-Cohen, Leslie, Frith — Does the autistic child have a theory of mind?](https://www.cambridge.org/core/journals/behavioral-and-brain-sciences/article/does-the-autistic-child-have-a-theory-of-mind/) — Sally-Anne 论文（1985）