# 时序差分——Q-Learning 与 SARSA

> 蒙特卡洛等待片段结束。TD 在每一步之后通过引导下一价值估计来更新。Q-learning 是离策略且乐观的；SARSA 是在策略且谨慎的。两者都是一行代码。两者都支撑着本阶段中的每个深度 RL 方法。

**类型：** 构建
**语言：** Python
**前置知识：** 阶段 9 · 01（MDP），阶段 9 · 02（动态规划），阶段 9 · 03（蒙特卡洛）
**时间：** ~75 分钟

## 问题

蒙特卡洛有效，但它有两个昂贵的需求。它需要终止的片段，并且它只在最终回报进来之后才更新。如果你的片段是 1,000 步，MC 等待 1,000 步才更新任何东西。它是高方差、低偏差的，在实践中很慢。

动态规划有相反的特点——零方差引导回溯——但需要已知模型。

时序差分（TD）学习在两者之间取中。从单个转移 `(s, a, r, s')`，形成一个一步目标 `r + γ V(s')` 并将 `V(s)` 推向它。没有模型。没有完整片段。使用右侧的近似 `V` 会产生偏差，但方差远低于 MC，并且从第一步开始就在线更新。

这是所有现代 RL——DQN、A2C、PPO、SAC——所围绕的支点。阶段 9 的其余部分是函数逼近的技巧层，构建在本课中你将编写的一步 TD 更新之上。

## 概念

![Q-learning vs SARSA：离策略 max vs 在策略 Q(s', a')](../assets/td.svg)

**V 的 TD(0) 更新：**

`V(s) ← V(s) + α [r + γ V(s') - V(s)]`

括号中的量是 TD 误差 `δ = r + γ V(s') - V(s)`。它是 MC 中 `G_t - V(s_t)` 的在线类比。收敛需要 `α` 满足 Robbins-Monro（`Σ α = ∞`，`Σ α² < ∞`）且所有状态被无限频繁访问。

**Q-learning。** 一种用于控制的离策略 TD 方法：

`Q(s, a) ← Q(s, a) + α [r + γ max_{a'} Q(s', a') - Q(s, a)]`

`max` 假设从 `s'` 开始将遵循*贪心*策略，无论智能体实际采取什么动作。这种解耦使得 Q-learning 能够在智能体通过 ε-贪心探索时学习 `Q*`。Mnih 等人（2015）将其转化为 Atari 上的深度 Q-learning（第 5 课）。

**SARSA。** 一种在策略 TD 方法：

`Q(s, a) ← Q(s, a) + α [r + γ Q(s', a') - Q(s, a)]`

名称来自元组 `(s, a, r, s', a')`。SARSA 使用智能体*实际*下一步将采取的动作 `a'`，而不是贪心的 `argmax`。收敛到运行中的任何 ε-贪心 `π` 的 `Q^π`，在极限 `ε → 0` 时变为 `Q*`。

**悬崖行走的区别。** 在经典的悬崖行走任务（掉下悬崖 = 奖励 -100）中，Q-learning 学习沿悬崖边缘的最优路径，但在探索期间偶尔受到惩罚。SARSA 学习离悬崖一步的更安全路径，因为它将探索噪声纳入其 Q-值。经过训练，两者在 `ε → 0` 时都达到最优。在实践中这很重要：当在部署期间实际发生探索时，SARSA 的行为更保守。

**期望 SARSA。** 用 `Q(s', a')` 在 `π` 下的期望值替换它：

`Q(s, a) ← Q(s, a) + α [r + γ Σ_{a'} π(a'|s') Q(s', a') - Q(s, a)]`

方差低于 SARSA（无需对 `a'` 采样），相同的在策略目标。在现代教科书中通常是默认选择。

**n-步 TD 和 TD(λ)。** 通过等待 `n` 步再引导来在 TD(0) 和 MC 之间插值。`n=1` 是 TD，`n=∞` 是 MC。TD(λ) 使用几何权重 `(1-λ)λ^{n-1}` 对所有 `n` 取平均。大多数深度 RL 使用 `n` 在 3 到 20 之间。

```figure
qlearning-gridworld
```

## 动手实现

### 步骤 1：在 ε-贪心策略上的 SARSA

```python
def sarsa(env, episodes, alpha=0.1, gamma=0.99, epsilon=0.1):
    Q = defaultdict(lambda: {a: 0.0 for a in ACTIONS})

    def choose(s):
        if random() < epsilon:
            return choice(ACTIONS)
        return max(Q[s], key=Q[s].get)

    for _ in range(episodes):
        s = env.reset()
        a = choose(s)
        while True:
            s_next, r, done = env.step(s, a)
            a_next = choose(s_next) if not done else None
            target = r + (gamma * Q[s_next][a_next] if not done else 0.0)
            Q[s][a] += alpha * (target - Q[s][a])
            if done:
                break
            s, a = s_next, a_next
    return Q
```

八行。与 Q-learning 的*唯一*区别是目标行。

### 步骤 2：Q-learning

```python
def q_learning(env, episodes, alpha=0.1, gamma=0.99, epsilon=0.1):
    Q = defaultdict(lambda: {a: 0.0 for a in ACTIONS})
    for _ in range(episodes):
        s = env.reset()
        while True:
            a = choose(s, Q, epsilon)
            s_next, r, done = env.step(s, a)
            target = r + (gamma * max(Q[s_next].values()) if not done else 0.0)
            Q[s][a] += alpha * (target - Q[s][a])
            if done:
                break
            s = s_next
    return Q
```

`max` 将目标与行为解耦。这一个符号是在策略和离策略之间的区别。

### 步骤 3：学习曲线

跟踪每 100 个片段的平均回报。Q-learning 在简单的确定性 GridWorld 上收敛更快；SARSA 在悬崖行走上更保守。在 `code/main.py` 中的 4×4 GridWorld 上，两者在约 2,000 个片段后都接近最优，使用 `α=0.1, ε=0.1`。

### 步骤 4：与 DP 真相比较

运行价值迭代（第 2 课）以获得 `Q*`。检查 `max_{s,a} |Q_learned(s,a) - Q*(s,a)|`。一个健康的表格 TD 智能体在 10,000 个片段后在 4×4 GridWorld 上应落在 `~0.5` 以内。

## 陷阱

- **初始 Q 值很重要。** 乐观初始化（对于负奖励任务 `Q = 0`）鼓励探索。悲观初始化可能永远困住贪心策略。
- **α 调度。** 恒定 `α` 对非平稳问题有效。衰减 `α_n = 1/n` 在理论上给出收敛，但在实践中太慢——将 `α` 固定在 `[0.05, 0.3]` 并监控学习曲线。
- **ε 调度。** 从高开始（`ε=1.0`），衰减到 `ε=0.05`。"GLIE"（极限贪心与无限探索）是收敛条件。
- **Q-learning 中的最大偏差。** 当 `Q` 有噪声时，`max` 算子偏向上偏。导致过高估计——Hasselt 的双 Q-learning（第 5 课中 DDQN 使用）通过两个 Q 表修复此问题。
- **非终止片段。** TD 可以在没有终止的情况下学习，但你需要要么限制步数，要么在限制处正确处理引导。标准：将限制视为非终止，继续引导。
- **状态哈希。** 如果状态是元组/张量，使用可哈希的键（元组，不是列表；四舍五入的浮点数元组，不是原始值）。

## 使用

2026 年 TD 格局：

| 任务 | 方法 | 原因 |
|------|--------|--------|
| 小型表格环境 | Q-learning | 直接学习最优策略。 |
| 在策略安全关键 | SARSA / 期望 SARSA | 在探索期间保守。 |
| 高维状态 | DQN（阶段 9 · 05） | 带经验回放和目标网络的神经网络 Q-函数。 |
| 连续动作 | SAC / TD3（阶段 9 · 07） | Q-网络上的 TD 更新；策略网络输出动作。 |
| LLM RL（基于奖励模型） | PPO / GRPO（阶段 9 · 08, 12） | 通过 GAE 使用 TD 风格优势的演员-评论家。 |
| 离线 RL | CQL / IQL（阶段 9 · 08） | 带保守正则化的 Q-learning。 |

你在 2026 年论文中读到的 90% 的"RL"都是 Q-learning 或 SARSA 的某种扩展。在深入之前，把表格更新刻在你的手指上。

## 产出

保存为 `outputs/skill-td-agent.md`：

```markdown
---
name: td-agent
description: 为表格或小特征 RL 任务在 Q-learning、SARSA、期望 SARSA 之间选择。
version: 1.0.0
phase: 9
lesson: 4
tags: [rl, td-learning, q-learning, sarsa]
---

给定一个表格或小特征环境，输出：

1. 算法。Q-learning / SARSA / 期望 SARSA / n-步变体。一句话理由，与在策略 vs 离策略和方差相关。
2. 超参数。α、γ、ε、衰减调度。
3. 初始化。Q_0 值（乐观 vs 零）和理由。
4. 收敛诊断。目标学习曲线，如果可能进行 `|Q - Q*|` 检查。
5. 部署注意事项。推理时探索将如何表现？是否需要 SARSA 的保守性？

拒绝将表格 TD 应用于状态空间 > 10⁶。拒绝在没有最大偏差警告的情况下部署 Q-learning 智能体。标记任何全程将 ε 保持在 1.0 训练的智能体（没有利用阶段）。
```

## 练习

1. **简单。** 在 4×4 GridWorld 上实现 Q-learning 和 SARSA。在 2,000 个片段上绘制学习曲线（每 100 个片段的平均回报）。哪个收敛更快？
2. **中等。** 构建悬崖行走环境（4×12，最后一行是悬崖，奖励 -100 并重置到起点）。比较 Q-learning 和 SARSA 的最终策略。截屏每个策略所走的路径。哪个更接近悬崖？
3. **困难。** 实现双 Q-learning。在噪声奖励 GridWorld（每步奖励添加高斯噪声 σ=5）上，展示 Q-learning 显著高估 `V*(0,0)` 而双 Q-learning 不会。

## 关键术语

| 术语 | 人们说它是什么 | 它实际意味着什么 |
|------|-----------------|-----------------------|
| TD 误差 | "更新信号" | `δ = r + γ V(s') - V(s)`，引导的残差。 |
| TD(0) | "一步 TD" | 每次转移后仅使用下一状态的估计进行更新。 |
| Q-learning | "离策略 RL 101" | 对下一状态动作使用 `max` 的 TD 更新；无论行为策略如何都学习 `Q*`。 |
| SARSA | "在策略 Q-learning" | 使用实际下一动作的 TD 更新；学习当前 ε-贪心 π 的 `Q^π`。 |
| 期望 SARSA | "低方差 SARSA" | 用 `π` 下的期望替换采样的 `a'`。 |
| GLIE | "正确的探索调度" | 极限贪心与无限探索；Q-learning 收敛所需。 |
| 引导 | "在目标中使用当前估计" | 区分 TD 和 MC 的原因。偏差的来源但方差大大降低。 |
| 最大化偏差 | "Q-learning 高估" | 在噪声估计上的 `max` 偏向上；由双 Q-learning 修复。 |

## 延伸阅读

- [Watkins & Dayan (1992). Q-learning](https://link.springer.com/article/10.1007/BF00992698) — 原始论文和收敛证明。
- [Sutton & Barto (2018). Ch. 6 — Temporal-Difference Learning](http://incompleteideas.net/book/RLbook2020.pdf) — TD(0)、SARSA、Q-learning、期望 SARSA。
- [Hasselt (2010). Double Q-learning](https://papers.nips.cc/paper_files/paper/2010/hash/091d584fced301b442654dd8c23b3fc9-Abstract.html) — 修复最大化偏差。
- [Seijen, Hasselt, Whiteson, Wiering (2009). A Theoretical and Empirical Analysis of Expected SARSA](https://ieeexplore.ieee.org/document/4927542) — 期望 SARSA 的动机。
- [Rummery & Niranjan (1994). On-line Q-learning using connectionist systems](https://www.researchgate.net/publication/2500611_On-Line_Q-Learning_Using_Connectionist_Systems) — 创造 SARSA 一词的论文（当时称为"修改的连接主义 Q-learning"）。
- [Sutton & Barto (2018). Ch. 7 — n-step Bootstrapping](http://incompleteideas.net/book/RLbook2020.pdf) — 将 TD(0) 推广到 TD(n)，从 Q-learning 到资格迹再到 PPO 中 GAE 的路径。
