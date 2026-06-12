# 动态规划——策略迭代与价值迭代

> 动态规划是作弊的强化学习。你已经知道转移和奖励函数；你只需迭代贝尔曼方程直到 `V` 或 `π` 停止变化。它是每个基于采样的方法都试图接近的基准。

**类型：** 构建
**语言：** Python
**前置知识：** 阶段 9 · 01（MDP）
**时间：** ~75 分钟

## 问题

你有一个已知模型的 MDP：对于任何状态-动作对，你可以查询 `P(s' | s, a)` 和 `R(s, a, s')`。库存管理者知道需求分布。棋盘游戏有确定性转移。一个 GridWorld 是四行 Python 代码。你有一个*模型*。

无模型 RL（Q-learning、PPO、REINFORCE）是为你没有模型的情况发明的——你只能从环境中采样。但当你确实有模型时，存在更快、更好的方法：动态规划。贝尔曼在 1957 年设计了它们。它们仍然定义了正确性：当人们说"这个 MDP 的最优策略"时，他们指的是 DP 会返回的策略。

在 2026 年你需要它们有三个原因。首先，RL 研究中的每个表格环境（GridWorld、FrozenLake、CliffWalking）都用 DP 求解以产生黄金标准策略。其次，精确值让你可以*调试*采样方法：如果 Q-learning 对 `V*(s_0)` 的估计与 DP 答案相差 30%，你的 Q-learning 有一个 bug。第三，现代离线 RL 和规划方法（MCTS、AlphaZero 的搜索、阶段 9 · 10 中的基于模型的 RL）都在学习到的或给定的模型上迭代贝尔曼回溯。

## 概念

![策略迭代和价值迭代，并排比较](../assets/dp.svg)

**两种算法，都是贝尔曼上的不动点迭代。**

**策略迭代。** 交替两个步骤直到策略停止变化。

1. *评估：* 给定策略 `π`，通过反复应用 `V(s) ← Σ_a π(a|s) Σ_{s',r} P(s',r|s,a) [r + γ V(s')]` 直到收敛来计算 `V^π`。
2. *改进：* 给定 `V^π`，使 `π` 关于 `V^π` 贪心：`π(s) ← argmax_a Σ_{s',r} P(s',r|s,a) [r + γ V(s')]`。

收敛是有保证的，因为 (a) 每个改进步骤要么保持 `π` 不变，要么严格增加某个状态的 `V^π`，(b) 确定性策略的空间是有限的。即使对于大状态空间，通常也在约 5-20 次外部迭代中收敛。

**价值迭代。** 将评估和改进合并为一次扫描。应用贝尔曼*最优性*方程：

`V(s) ← max_a Σ_{s',r} P(s',r|s,a) [r + γ V(s')]`

重复直到 `max_s |V_{new}(s) - V(s)| < ε`。最后通过取贪心动作提取策略。每次迭代严格更快——没有内部评估循环——但通常需要更多迭代才能收敛。

**广义策略迭代（GPI）。** 统一框架。价值函数和策略锁定在一个双向改进循环中；任何驱动两者走向相互一致的方法（异步价值迭代、修改的策略迭代、Q-learning、演员-评论家、PPO）都是 GPI 的一个实例。

**为什么 `γ < 1` 很重要。** 贝尔曼算子是上确界范数上的 `γ`-压缩映射：`||T V - T V'||_∞ ≤ γ ||V - V'||_∞`。压缩意味着唯一不动点和几何收敛。放弃 `γ < 1` 你就失去了保证——你需要一个有限视野或吸收终止状态。

```figure
value-iteration-gamma
```

## 动手实现

### 步骤 1：构建 GridWorld MDP 模型

使用第 1 课中相同的 4×4 GridWorld。我们添加一个随机变体：智能体以概率 `0.1` 滑向随机的垂直方向。

```python
SLIP = 0.1

def transitions(state, action):
    if state == TERMINAL:
        return [(state, 0.0, 1.0)]
    outcomes = []
    for direction, prob in action_probs(action):
        outcomes.append((apply_move(state, direction), -1.0, prob))
    return outcomes
```

`transitions(s, a)` 返回一个 `(s', r, p)` 列表。这就是整个模型。

### 步骤 2：策略评估

给定策略 `π(s) = {action: prob}`，迭代贝尔曼方程直到 `V` 停止变化：

```python
def policy_evaluation(policy, gamma=0.99, tol=1e-6):
    V = {s: 0.0 for s in states()}
    while True:
        delta = 0.0
        for s in states():
            v = sum(pi_a * sum(p * (r + gamma * V[s_prime])
                              for s_prime, r, p in transitions(s, a))
                   for a, pi_a in policy(s).items())
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        if delta < tol:
            return V
```

### 步骤 3：策略改进

用关于 `V` 的贪心策略替换 `π`。如果 `π` 没有变化，返回——我们处于最优。

```python
def policy_improvement(V, gamma=0.99):
    new_policy = {}
    for s in states():
        best_a = max(
            ACTIONS,
            key=lambda a: sum(p * (r + gamma * V[s_prime])
                              for s_prime, r, p in transitions(s, a)),
        )
        new_policy[s] = best_a
    return new_policy
```

### 步骤 4：将它们缝合在一起

```python
def policy_iteration(gamma=0.99):
    policy = {s: "up" for s in states()}   # 任意起点
    for _ in range(100):
        V = policy_evaluation(lambda s: {policy[s]: 1.0}, gamma)
        new_policy = policy_improvement(V, gamma)
        if new_policy == policy:
            return V, policy
        policy = new_policy
```

4×4 的典型收敛：4-6 次外部迭代。输出 `V*(0,0) ≈ -6` 和一个严格减少步数的策略。

### 步骤 5：价值迭代（单循环版本）

```python
def value_iteration(gamma=0.99, tol=1e-6):
    V = {s: 0.0 for s in states()}
    while True:
        delta = 0.0
        for s in states():
            v = max(sum(p * (r + gamma * V[s_prime])
                       for s_prime, r, p in transitions(s, a))
                   for a in ACTIONS)
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        if delta < tol:
            break
    policy = policy_improvement(V, gamma)
    return V, policy
```

相同的不动点，更少的代码行。

## 陷阱

- **忘记处理终止状态。** 如果你对吸收状态应用贝尔曼方程，它仍然会选择一个什么都不改变的"最佳动作"。用 `if s == terminal: V[s] = 0` 守卫。
- **上确界范数 vs L2 收敛。** 使用 `max |V_new - V|`，而非平均值。理论保证是基于上确界范数的。
- **原地 vs 同步更新。** 原地更新 `V[s]`（Gauss-Seidel）比使用单独的 `V_new` 字典（Jacobi）收敛更快。生产代码使用原地更新。
- **策略平局。** 如果两个动作具有相同的 Q-值，`argmax` 可能在每次迭代中以不同方式打破平局，导致"策略稳定"检查振荡。使用稳定的平局打破机制（固定顺序中的第一个动作）。
- **状态空间爆炸。** DP 是 `O(|S| · |A|)` 每次扫描。最多适用于约 10⁷ 个状态。超过此限度，你需要函数逼近（阶段 9 · 05 起）。

## 使用

在 2026 年，DP 是正确性基线和规划器的内部循环：

| 用例 | 方法 |
|----------|--------|
| 精确求解小型表格 MDP | 价值迭代（更简单）或策略迭代（更少外部步骤） |
| 验证 Q-learning / PPO 实现 | 在玩具环境上与 DP-最优 V* 比较 |
| 基于模型的 RL（阶段 9 · 10） | 在学习到的转移模型上的贝尔曼回溯 |
| AlphaZero / MuZero 中的规划 | 蒙特卡洛树搜索 = 异步贝尔曼回溯 |
| 离线 RL（CQL、IQL） | 保守 Q-迭代——对 OOD 动作施加惩罚的 DP |

每次有人提到"最优价值函数"，他们的意思是"DP 不动点"。当你在论文中看到 `V*` 或 `Q*` 时，想象这个循环。

## 产出

保存为 `outputs/skill-dp-solver.md`：

```markdown
---
name: dp-solver
description: 通过策略迭代或价值迭代精确求解小型表格 MDP。报告收敛行为。
version: 1.0.0
phase: 9
lesson: 2
tags: [rl, dynamic-programming, bellman]
---

给定一个具有已知模型的 MDP，输出：

1. 选择。策略迭代 vs 价值迭代。理由与 |S|、|A|、γ 相关。
2. 初始化。V_0，起始策略。收敛敏感性。
3. 停止。上确界范数容差 ε。预期扫描次数。
4. 验证。V*(s_0) 精确计算。提取贪心策略。
5. 使用。这个基线将如何用于调试/评估基于采样的方法。

拒绝在状态空间 > 10⁷ 上运行 DP。拒绝在没有上确界范数检查的情况下声称收敛。标记任何无限视野任务上的 γ ≥ 1 作为违反保证。
```

## 练习

1. **简单。** 以 `γ ∈ {0.9, 0.99}` 在 4×4 GridWorld 上运行价值迭代。多少次扫描直到 `max |ΔV| < 1e-6`？将 `V*` 打印为 4×4 网格。
2. **中等。** 在*随机* GridWorld（滑动概率 `0.1`）上比较策略迭代 vs 价值迭代。统计：扫描次数、挂钟时间、最终 `V*(0,0)`。哪个在迭代次数上收敛更快？在挂钟时间上呢？
3. **困难。** 构建修改的策略迭代：在评估步骤中，只运行 `k` 次扫描而不是到收敛。绘制 `V*(0,0)` 误差 vs `k`，`k ∈ {1, 2, 5, 10, 50}`。曲线告诉了你关于评估/改进权衡的什么信息？

## 关键术语

| 术语 | 人们说它是什么 | 它实际意味着什么 |
|------|-----------------|-----------------------|
| 策略迭代 | "DP 算法" | 交替评估（`V^π`）和改进（关于 `V^π` 的贪心 `π`）直到策略停止变化。 |
| 价值迭代 | "更快的 DP" | 一次扫描中应用的贝尔曼最优性回溯；几何收敛到 `V*`。 |
| 贝尔曼算子 | "递归" | `(T V)(s) = max_a Σ P (r + γ V(s'))`；上确界范数上的 `γ`-压缩映射。 |
| 压缩映射 | "为什么 DP 收敛" | 任何具有 `||T x - T y|| ≤ γ ||x - y||` 的算子 `T` 都有唯一不动点。 |
| GPI | "一切都是 DP" | 广义策略迭代：任何驱动 `V` 和 `π` 走向相互一致的方法。 |
| 同步更新 | "Jacobi 风格" | 在一次扫描中使用旧的 `V`；可清晰地分析但较慢。 |
| 原地更新 | "Gauss-Seidel 风格" | 在更新过程中使用 `V`；实践中收敛更快。 |

## 延伸阅读

- [Sutton & Barto (2018). Ch. 4 — Dynamic Programming](http://incompleteideas.net/book/RLbook2020.pdf) — 策略迭代和价值迭代的权威阐述。
- [Bertsekas (2019). Reinforcement Learning and Optimal Control](http://www.athenasc.com/rlbook.html) — 压缩映射论证的严格处理。
- [Puterman (2005). Markov Decision Processes](https://onlinelibrary.wiley.com/doi/book/10.1002/9780470316887) — 修改的策略迭代及其收敛分析。
- [Howard (1960). Dynamic Programming and Markov Processes](https://mitpress.mit.edu/9780262582300/dynamic-programming-and-markov-processes/) — 原始策略迭代论文。
- [Bertsekas & Tsitsiklis (1996). Neuro-Dynamic Programming](http://www.athenasc.com/ndpbook.html) — 从 DP 到近似 DP / 深度 RL 的桥梁，后续每课都使用。
