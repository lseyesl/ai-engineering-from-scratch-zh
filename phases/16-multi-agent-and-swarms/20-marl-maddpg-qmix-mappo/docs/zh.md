# MARL — MADDPG、QMIX、MAPPO

> 多智能体协调的强化学习遗产，在 2026 年仍然影响着 LLM 智能体系统。**MADDPG**（Lowe 等人，NeurIPS 2017，arXiv:1706.02275）引入了集中式训练分散式执行（CTDE）：每个评论家在训练期间看到所有智能体的状态和动作；测试时只运行本地行动者。适用于合作、竞争和混合场景。**QMIX**（Rashid 等人，ICML 2018，arXiv:1803.11485）是带有单调混合网络的价值分解；每个智能体的 Q 组合成联合 Q，使得 `argmax` 干净地分布——在 StarCraft Multi-Agent Challenge（SMAC）上占主导。**MAPPO**（Yu 等人，NeurIPS 2022，arXiv:2103.01955）是带有集中式价值函数的 PPO；在粒子世界、SMAC、Google Research Football、Hanabi 上"出奇地有效"，几乎无需调参。这些为必须分散执行的智能体团队训练策略提供了基础。MAPPO 是 **2026 年合作 MARL 的默认基线**。本课程从小型网格世界玩具构建每一个，在接触 LLM 智能体训练之前将这三个想法刻入肌肉记忆。

**类型：** Learn
**语言：** Python（标准库，小型无 NumPy 实现）
**前置知识：** Phase 09（强化学习），Phase 16 · 09（并行群体网络）
**时间：** ~90 分钟

## 问题

LLM 智能体系统越来越多地训练智能体间协调的策略：何时让出、何时行动、调用哪个同伴。告诉你如何训练此类策略的文献是多智能体强化学习（MARL），它先于 LLM 浪潮，并有一小组主导算法。

没有模式词汇的 MARL 论文读起来很痛苦。集中式训练分散式执行（CTDE）、价值分解和集中式评论家不是流行语——它们是对特定问题的特定答案：

- 独立 RL（每个智能体单独学习）从每个智能体的角度来看是非平稳的。不好。
- 集中式 RL（一个智能体控制所有）不能扩展且违反执行约束。
- CTDE 两全其美：用全局信息训练，用本地策略部署。

## 概念

### 论文使用的三种环境

- **粒子世界（multi-agent particle env）。** 简单的 2D 物理，带有合作/竞争任务。MADDPG 的原始测试平台。
- **StarCraft Multi-Agent Challenge（SMAC）。** 合作微观管理，部分观察。QMIX 的测试平台。离散动作，连续状态。
- **Google Research Football、Hanabi、MPE。** MAPPO 的基线。

不同的环境有不同的动作/观察类型。算法据此选择。

### MADDPG（2017）——CTDE 模式

每个智能体 `i` 有一个行动者 `mu_i(o_i)`，将其自身观察映射到动作。每个智能体还有一个评论家 `Q_i(x, a_1, ..., a_n)`，在训练期间看到所有观察和所有动作。行动者根据评论家的评估通过策略梯度更新。

```
actor update:    grad_theta_i J = E[grad_theta mu_i(o_i) * grad_a_i Q_i(x, a_1..n) at a_i=mu_i(o_i)]
critic update:   TD on Q_i(x, a_1..n) given next-state joint estimate
```

为什么 CTDE：在训练时，我们知道每个人的动作；我们用此来减少每个评论家的方差。在部署时，每个智能体只看到 `o_i` 并调用 `mu_i(o_i)`。

失败模式：评论家随 N 个智能体增长（输入包含所有动作）。在没有近似的情况下无法扩展到约 10 个以上智能体。

### QMIX（2018）——价值分解

仅限合作。全局奖励是每个智能体 Q 值的单调函数之和：

```
Q_tot(tau, a) = f(Q_1(tau_1, a_1), ..., Q_n(tau_n, a_n)),   df/dQ_i >= 0
```

单调性保证了 `argmax_a Q_tot` 可以通过每个智能体独立选择 `argmax_{a_i} Q_i` 来计算。这正是你需要的**分散式执行属性**。在训练时，一个混合网络从每个智能体的 Q 产生 `Q_tot`。

为什么 QMIX 在 SMAC 上获胜：合作性 StarCraft 微观管理具有同质智能体、本地观察、全局奖励——完美适合价值分解。

失败模式：单调性约束是限制性的；有些任务的奖励结构不是单调可分解的（一个智能体为团队牺牲）。扩展（QTRAN、QPLEX）放宽了这一点。

### MAPPO（2022）——被忽视的默认值

多智能体 PPO：带有集中式价值函数的 PPO。每个智能体有自己的策略；所有智能体共享（或各自拥有）看到完整状态的价值函数。Yu 等人 2022 年在五个基准测试上将 MAPPO 与 MADDPG、QMIX 及其扩展进行了比较，发现：

- MAPPO 在粒子世界、SMAC、Google Research Football、Hanabi、MPE 上匹配或击败 off-policy MARL 方法。
- 几乎不需要超参数调优。
- 训练稳定；跨种子可复现。

社区在这篇论文之前低估了 on-policy MARL。在 2026 年，MAPPO 是合作 MARL 的默认基线；任何新方法必须击败它。

### 为什么 LLM 智能体工程师应该关心

三个直接用途：

1. **路由器训练。** 一个元智能体选择哪个子智能体处理任务。这是一个具有 N 个分散式子智能体和一个集中式路由器的 MARL 问题。MAPPO 适合。
2. **角色涌现。** 在生成式智能体模拟中，训练智能体随时间采用互补角色是一个伪装下的 MARL 问题。QMIX 风格的价值分解通过构造强制互补性。
3. **多智能体工具使用。** 当智能体共享工具并竞争预算时，通过 CTDE 训练它们产生可部署的、尊重资源约束的本地策略。

实际警告：在 2026 年，大多数生产 LLM 智能体系统通过提示引导策略而非训练。MARL 在你拥有（a）大量交互数据、（b）清晰的奖励信号和（c）投资训练基础设施的意愿时发挥作用。

### CTDE 作为超越 RL 的设计模式

即使没有训练，CTDE 也是一个有用的架构模式：

- 在*设计*期间，假设完整的团队可见性。
- 在*运行时*，强制分散式执行：每个智能体只看到 `o_i`。

该模式强制你明确每个智能体的状态并提前考虑部分可观察性。许多生产多智能体系统静默地假设到处都有共享状态——CTDE 纪律防止了这一点。

### 非平稳性问题

当多个智能体同时学习时，每个智能体的环境（包括其他人的策略）是非平稳的。经典的单智能体 RL 证明失效。本课程中的 MARL 算法都解决了这个问题：

- MADDPG：全局评论家看到所有动作，因此其价值估计是平稳的。
- QMIX：价值分解将学习转移到联合 Q 空间，其中最优性被良好定义。
- MAPPO：集中式价值函数抑制了来自他人策略变化的方差。

在 LLM 智能体系统中，非平稳性表现为"我的智能体上个月还能工作，现在上游的那个其他智能体变了，我的出问题了。"用 CTDE 训练 MARL 是原则性修复；提示级别的修复更快但持久性差。

### 本课程不涵盖的内容

训练实际的网络是 Phase 09 的话题。本课程构建脚本化策略版本，展示 CTDE、价值分解和集中式价值模式，无需梯度更新。目标是在你拿起完整的 MARL 库（PyMARL、MARLlib、RLlib multi-agent）之前内化这些模式。

## 构建

`code/main.py` 在一个微小的 2 智能体合作网格世界上实现了三种模式演示：

- 环境：4x4 网格上的 2 个智能体，一个奖励球。奖励 = 1 如果任何智能体到达球；任务完成。
- `IndependentAgents`——每个智能体将其他视为环境。基线。
- `MADDPGStyle`——集中式评论家计算联合价值；行动者策略据此更新。脚本化策略改进。
- `QMIXStyle`——使用单调混合器的价值分解。
- `MAPPOStyle`——集中式价值函数；策略针对共享基线更新。

所有四个运行相同的回合并报告平均步数到目标。CTDE 变体收敛到比独立基线更短的路径。

运行：

```
python3 code/main.py
```

预期输出：独立智能体平均约 6 步；CTDE 变体收敛到约 3.5 步（4x4 网格的最优值为 3）。尽管是脚本化策略，模式差异仍然显现。

## 使用

`outputs/skill-marl-picker.md` 是一个技能，为给定的多智能体任务选择 MARL 算法：合作 vs 竞争、同质 vs 异质、动作空间类型、规模、奖励信号。

## 交付

MARL 在生产中很少见。当你确实使用时：

- **从 MAPPO 开始。** 2022 年的论文将其确立为基线；先复现它节省了追逐更花哨方法的时间。
- **记录每个智能体的观察和动作流。** 没有每个智能体的轨迹，调试 MARL 无望。
- **将训练代码与执行代码分离。** CTDE 是一种纪律；让执行路径真的只看到 `o_i`。
- **奖励塑形警告。** MARL 对奖励设计极其敏感。塑形中的一个协调错误，智能体就会学会利用它。运行对抗性测试。
- **对于 LLM 智能体**，首先考虑提示级别策略。只有在交互数据 + 奖励信号 + 基础设施三者齐备时才投入 MARL 训练。

## 练习

1. 运行 `code/main.py`。测量独立智能体和 MAPPO 风格智能体之间的步数到目标差距。在 6x6 网格上差距会扩大还是缩小？
2. 实现竞争变体：两个智能体，一个球，只有先到达的获得奖励。哪种模式能干净地处理竞争？历史上是 MADDPG。
3. 阅读 MADDPG（arXiv:1706.02275）第 3 节。用你自己的话以伪代码符号化地实现确切的评论家更新规则。
4. 阅读 MAPPO（arXiv:2103.01955）。为什么作者认为集中式价值 + PPO 在其基准测试上击败 off-policy MARL？列出三个最有力的主张。
5. 将 CTDE 作为设计模式应用于一个假设的 LLM 智能体系统（例如，研究者智能体 + 总结者 + 编码者）。在运行时不可用的联合信息在设计时有哪些？

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|---|---|---|
| MARL | "多智能体 RL" | 多智能体系统的强化学习。 |
| CTDE | "集中式训练，分散式执行" | 用全局信息训练；用本地策略部署。 |
| MADDPG | "多智能体 DDPG" | CTDE，每个评论家看到所有观察和动作。 |
| QMIX | "价值分解" | 每个智能体 Q 的单调混合。仅合作。 |
| MAPPO | "多智能体 PPO" | 带有集中式价值函数的 PPO。2026 默认基线。 |
| Value decomposition | "单个 Q 的和" | 联合 Q 表示为每个智能体 Q 的单调函数。 |
| Non-stationarity | "移动目标" | 每个智能体的环境随其他智能体学习而变化。核心 MARL 问题。 |
| On-policy / off-policy | "从当前/重放中学习" | PPO 是 on-policy（MAPPO）；DDPG 和 Q-learning 是 off-policy。 |
| SMAC | "StarCraft 多智能体挑战" | 合作微观管理基准测试；QMIX 的主场。 |

## 延伸阅读

- [Lowe et al. — Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments](https://arxiv.org/abs/1706.02275)——MADDPG；NeurIPS 2017
- [Rashid et al. — QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning](https://arxiv.org/abs/1803.11485)——QMIX；ICML 2018
- [Yu et al. — The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games](https://arxiv.org/abs/2103.01955)——MAPPO；NeurIPS 2022
- [BAIR blog post on MAPPO](https://bair.berkeley.edu/blog/2021/07/14/mappo/)——MAPPO 结果的可读阐述
- [SMAC repository](https://github.com/oxwhirl/smac)——StarCraft 多智能体挑战
