# 深度 Q 网络（DQN）

> 2013 年：Mnih 用单个 Q 学习网络处理原始像素，在七个 Atari 游戏上击败了所有经典 RL 智能体。2015 年：扩展到 49 个游戏，发表在《自然》杂志上，开启了深度 RL 时代。DQN 是 Q 学习加上三个使函数逼近稳定的技巧。

**类型：** 构建
**语言：** Python
**前置知识：** 阶段 3 · 03（反向传播），阶段 9 · 04（Q-learning、SARSA）
**时间：** ~75 分钟

## 问题

表格 Q 学习需要为每个（状态、动作）对单独存储一个 Q 值。一个棋盘有约 10⁴³ 个状态。一帧 Atari 图像是 210×160×3 = 100,800 个特征。表格 RL 在数千个状态时就已失效，更不用说数十亿了。

修复方案事后看来显而易见：用神经网络替换 Q 表，`Q(s, a; θ)`。但从显而易见到真正实现，花了几十年时间。使用朴素函数逼近的 Q 学习在"致命三要素"——函数逼近 + 引导 + 离策略学习——下会发散。Mnih 等人（2013 年、2015 年）发现了三个稳定学习的工程技巧：

1. **经验回放** 使转移去相关。
2. **目标网络** 冻结引导目标。
3. **奖励裁剪** 归一化梯度量级。

Atari 上的 DQN 是第一个用单一架构、单一超参数集从原始像素解决数十个控制问题的方法。此后所有的"深度 RL"——DDQN、Rainbow、Dueling、Distributional、R2D2、Agent57——都建立在这个三技巧基础之上。

## 概念

**目标。** DQN 在神经 Q 函数上最小化一步 TD 损失：

`L(θ) = E_{(s,a,r,s')~D} [ (r + γ max_{a'} Q(s', a'; θ^-) - Q(s, a; θ))² ]`

`θ` = 在线网络，每一步通过梯度下降更新。`θ^-` = 目标网络，定期从 `θ` 复制（每约 10,000 步一次）。`D` = 过去转移的回放缓冲区。

**三个技巧，按重要性排序：**

**经验回放。** 一个容量约为 `10⁶` 条转移的环形缓冲区。每个训练步骤均匀随机采样一个小批量。这打破了时间相关性（连续帧几乎相同），让网络能够多次从罕见的有奖励转移中学习，并使连续的梯度更新去相关。没有它，在 Atari 上使用神经网络进行在策略 TD 会发散。

**目标网络。** 在贝尔曼方程两侧使用同一个网络 `Q(·; θ)` 会使目标每次更新都在移动——"追逐自己的尾巴"。修复方案：保留第二个网络 `Q(·; θ^-)`，其权重冻结。每 `C` 步，复制 `θ → θ^-`。这使回归目标在数千个梯度步内保持稳定。软更新 `θ^- ← τ θ + (1-τ) θ^-`（用于 DDPG、SAC）是一个更平滑的变体。

**奖励裁剪。** Atari 的奖励量级从 1 到 1000+ 不等。裁剪到 `{-1, 0, +1}` 可以防止任何一个游戏主导梯度。当奖励量级有意义时这种方法不正确；但对于仅符号重要的 Atari 来说可以接受。

**双 DQN。** Hasselt（2016 年）修复了最大化偏差：使用在线网络*选择*动作，使用目标网络*评估*它。

`target = r + γ Q(s', argmax_{a'} Q(s', a'; θ); θ^-)`

即插即用，效果始终更好。默认使用它。

**其他改进（Rainbow，2017 年）：** 优先回放（更多采样高 TD 误差的转移）、决斗架构（分离 `V(s)` 和优势头）、噪声网络（学习的探索）、n-步回报、分布 Q（C51/QR-DQN）、多步引导。每个增加几个百分点；收益大致可叠加。

## 动手实现

这里的代码是纯标准库实现——我们使用手工构建的单隐藏层 MLP 在小型连续 GridWorld 上运行，因此每个训练步骤只需微秒级别。算法与大规模 Atari DQN 完全相同。

### 步骤 1：回放缓冲区

```python
class ReplayBuffer:
    def __init__(self, capacity):
        self.buf = []
        self.capacity = capacity
    def push(self, s, a, r, s_next, done):
        if len(self.buf) == self.capacity:
            self.buf.pop(0)
        self.buf.append((s, a, r, s_next, done))
    def sample(self, batch, rng):
        return rng.sample(self.buf, batch)
```

Atari 大约需要 50,000 容量；我们的玩具环境 5,000 就足够了。

### 步骤 2：一个小型 Q 网络（手动 MLP）

```python
class QNet:
    def __init__(self, n_in, n_hidden, n_actions, rng):
        self.W1 = [[rng.gauss(0, 0.3) for _ in range(n_in)] for _ in range(n_hidden)]
        self.b1 = [0.0] * n_hidden
        self.W2 = [[rng.gauss(0, 0.3) for _ in range(n_hidden)] for _ in range(n_actions)]
        self.b2 = [0.0] * n_actions
    def forward(self, x):
        h = [max(0.0, sum(w * xi for w, xi in zip(row, x)) + b) for row, b in zip(self.W1, self.b1)]
        q = [sum(w * hi for w, hi in zip(row, h)) + b for row, b in zip(self.W2, self.b2)]
        return q, h
```

前向传播：线性 → ReLU → 线性。这就是整个网络。

### 步骤 3：DQN 更新

```python
def train_step(online, target, batch, gamma, lr):
    grads = zeros_like(online)
    for s, a, r, s_next, done in batch:
        q, h = online.forward(s)
        if done:
            y = r
        else:
            q_next, _ = target.forward(s_next)
            y = r + gamma * max(q_next)
        td_error = q[a] - y
        accumulate_grads(grads, online, s, h, a, td_error)
    apply_sgd(online, grads, lr / len(batch))
```

其形状与第 4 课的 Q 学习相同，有两个区别：（a）我们通过可微的 `Q(·; θ)` 进行反向传播，而不是索引表；（b）目标使用 `Q(·; θ^-)`。

### 步骤 4：主循环

对于每个片段，在 `Q(·; θ)` 上 ε-贪心地行动，将转移推入缓冲区，采样一个小批量，执行一次梯度步，定期同步 `θ^- ← θ`。模式如下：

```python
for episode in range(N):
    s = env.reset()
    while not done:
        a = epsilon_greedy(online, s, epsilon)
        s_next, r, done = env.step(s, a)
        buffer.push(s, a, r, s_next, done)
        if len(buffer) >= batch:
            train_step(online, target, buffer.sample(batch), gamma, lr)
        if steps % sync_every == 0:
            target = copy(online)
        s = s_next
```

在我们的微型 GridWorld 上，使用 16 维的 one-hot 状态，智能体在大约 500 个片段内学到接近最优的策略。在 Atari 上，将此扩展到 2 亿帧并添加 CNN 特征提取器。

## 陷阱

- **致命三要素。** 函数逼近 + 离策略 + 引导可能导致发散。DQN 通过目标网络 + 回放来缓解；不要移除任何一个。
- **探索。** ε 必须衰减，通常在前 ~10% 的训练中从 1.0 衰减到 0.01。如果没有足够的早期探索，Q 网络会收敛到一个局部盆地。
- **高估。** 在噪声 Q 上取 `max` 会偏向上偏。生产环境中始终使用双 DQN。
- **奖励尺度。** 裁剪或归一化奖励；梯度量级与奖励量级成正比。
- **回放缓冲区冷启动。** 在缓冲区中积累到几千条转移之前不要训练。在约 20 个样本上的早期梯度会过拟合。
- **目标同步频率。** 太频繁 ≈ 没有目标网络；太不频繁 ≈ 目标过时。Atari DQN 使用 10,000 个环境步。经验法则：每约训练水平的 1/100 同步一次。
- **观察预处理。** Atari DQN 堆叠 4 帧以使状态满足马尔可夫性。任何包含速度信息的环境都需要帧堆叠或循环状态。

## 使用

到 2026 年，DQN 很少是最先进的，但它仍然是参考性的离策略算法：

| 任务 | 首选方法 | 为什么不是 DQN？ |
|------|----------|-----------------|
| 离散动作 Atari 类任务 | Rainbow DQN 或 Muesli | 相同框架，更多技巧。 |
| 连续控制 | SAC / TD3（阶段 9 · 07） | DQN 没有策略网络。 |
| 在策略 / 高吞吐量 | PPO（阶段 9 · 08） | 没有回放缓冲区；更容易扩展。 |
| 离线 RL | CQL / IQL / Decision Transformer | 保守 Q 目标，没有引导爆炸。 |
| 大型离散动作空间（推荐系统） | 带动作嵌入的 DQN 或 IMPALA | 可以；装饰很重要。 |
| LLM RL | PPO / GRPO | 序列级别，非步骤级别；不同的损失函数。 |

其中的经验教训仍然适用。回放和目标网络出现在 SAC、TD3、DDPG、SAC-X、AlphaZero 的自对弈缓冲区和每种离线 RL 方法中。奖励裁剪作为 PPO 中的优势归一化继续存在。这个架构就是蓝图。

## 产出

保存为 `outputs/skill-dqn-trainer.md`：

```markdown
---
name: dqn-trainer
description: 为离散动作 RL 任务生成 DQN 训练配置（缓冲区、目标同步、ε 调度、奖励裁剪）。
version: 1.0.0
phase: 9
lesson: 5
tags: [rl, dqn, deep-rl]
---

给定一个离散动作环境（观察形状、动作数量、视野、奖励尺度），输出：

1. 网络。架构（MLP / CNN / Transformer）、特征维度、深度。
2. 回放缓冲区。容量、小批量大小、预热大小。
3. 目标网络。同步策略（每 C 步硬拷贝或软 τ）。
4. 探索。ε 起始 / 结束 / 调度长度。
5. 损失。Huber vs MSE、梯度裁剪值、奖励裁剪规则。
6. 双 DQN。默认开启，除非有显式理由禁用。

拒绝交付没有目标网络、没有回放缓冲区或 ε 保持在 1 的 DQN。拒绝连续动作任务（引导至 SAC / TD3）。标记任何奖励范围 > 每步均值 10 倍的需要裁剪或缩放归一化。
```

## 练习

1. **简单。** 运行 `code/main.py`。绘制每个片段的回报曲线。运行均值超过 -10 需要多少个片段？
2. **中等。** 禁用目标网络（在贝尔曼目标两侧使用在线网络）。测量训练不稳定性——回报是否振荡或发散？
3. **困难。** 添加双 DQN：使用在线网络选择 `argmax a'`，目标网络进行评估。在噪声奖励 GridWorld 上，比较使用和不使用双 DQN 时 1,000 个片段后的 `Q(s_0, best_a)` 偏差与真实 `V*(s_0)`。

## 关键术语

| 术语 | 人们说它是什么 | 它实际意味着什么 |
|------|-----------------|-----------------------|
| DQN | "深度 Q 学习" | 带神经 Q 函数、回放缓冲区和目标网络的 Q 学习。 |
| 经验回放 | "打乱的转移" | 每个梯度步均匀采样的环形缓冲区；使数据去相关。 |
| 目标网络 | "冻结的引导" | 在贝尔曼目标中使用的 Q 的定期副本；稳定训练。 |
| 致命三要素 | "为什么 RL 发散" | 函数逼近 + 引导 + 离策略 = 没有收敛保证。 |
| 双 DQN | "修复最大化偏差" | 在线网络选择动作，目标网络评估它。 |
| 决斗 DQN | "V 和 A 头" | 将 Q = V + A - mean(A) 分解；相同输入，更好的梯度流。 |
| Rainbow | "所有技巧" | 在一个算法中组合了 DDQN + PER + 决斗 + n-步 + 噪声 + 分布。 |
| PER | "优先回放" | 按 TD 误差量级成比例地采样转移。 |

## 延伸阅读

- [Mnih et al. (2013). Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602) — 2013 年 NeurIPS 研讨会论文，开启了深度 RL。
- [Mnih et al. (2015). Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236) — 《自然》论文，49 游戏 DQN。
- [Hasselt, Guez, Silver (2016). Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461) — DDQN。
- [Wang et al. (2016). Dueling Network Architectures](https://arxiv.org/abs/1511.06581) — 决斗 DQN。
- [Hessel et al. (2018). Rainbow: Combining Improvements in Deep RL](https://arxiv.org/abs/1710.02298) — 堆叠技巧的论文。
- [OpenAI Spinning Up — DQN](https://spinningup.openai.com/en/latest/algorithms/dqn.html) — 清晰的现代阐述。
- [Sutton & Barto (2018). Ch. 9 — On-policy Prediction with Approximation](http://incompleteideas.net/book/RLbook2020.pdf) — 关于"致命三要素"（函数逼近 + 引导 + 离策略）的教科书式处理，DQN 的目标网络和回放缓冲区正是为此设计的。
- [CleanRL DQN implementation](https://docs.cleanrl.dev/rl-algorithms/dqn/) — 在消融研究中用作参考的单文件 DQN；值得与本课的手写版本一起阅读。
